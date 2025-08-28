#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define SHARED_SIZE (BLOCK_SIZE * 2)

// Compare and exchange
__device__ __forceinline__ void compareExchange(uint32_t* a, uint32_t* b, bool dir) {
    uint32_t x = *a;
    uint32_t y = *b;
    if ((x > y) == dir) {
        *a = y;
        *b = x;
    }
}

// Local bitonic sort kernel (for smaller chunks)
__global__ void bitonicSortLocal(uint32_t* data, uint32_t n, bool ascending) {
    __shared__ uint32_t shared_data[SHARED_SIZE];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t global_tid = bid * SHARED_SIZE + tid;
    
    // Load data into shared memory
    shared_data[tid] = (global_tid < n) ? data[global_tid] : (ascending ? UINT32_MAX : 0);
    shared_data[tid + BLOCK_SIZE] = (global_tid + BLOCK_SIZE < n) ? 
        data[global_tid + BLOCK_SIZE] : (ascending ? UINT32_MAX : 0);
    
    __syncthreads();
    
    // Bitonic sort in shared memory
    for (uint32_t stage = 2; stage <= SHARED_SIZE; stage <<= 1) {
        for (uint32_t substage = stage >> 1; substage > 0; substage >>= 1) {
            uint32_t thread_group = tid / substage;
            uint32_t thread_local = tid % substage;
            
            bool dir = ((thread_group / (stage / substage)) & 1) == 0;
            if (!ascending) dir = !dir;
            
            uint32_t i = thread_group * substage + thread_local;
            uint32_t j = i + substage;
            
            if (thread_local < (substage >> 1) && j < SHARED_SIZE) {
                compareExchange(&shared_data[i], &shared_data[j], dir);
            }
            
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_tid < n) data[global_tid] = shared_data[tid];
    if (global_tid + BLOCK_SIZE < n) data[global_tid + BLOCK_SIZE] = shared_data[tid + BLOCK_SIZE];
}

// Global bitonic merge kernel (for larger stages)
__global__ void bitonicMergeGlobal(uint32_t* data, uint32_t n, uint32_t stage, uint32_t substage, bool ascending) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint32_t group_size = 1U << substage;
    uint32_t group_id = tid / group_size;
    uint32_t local_id = tid % group_size;
    
    if (local_id >= (group_size >> 1)) return;
    
    uint32_t stage_size = 1U << stage;
    bool sort_dir = ((group_id / (stage_size / group_size)) & 1) == 0;
    if (!ascending) sort_dir = !sort_dir;
    
    uint32_t distance = group_size >> 1;
    uint32_t i = group_id * group_size + local_id;
    uint32_t j = i + distance;
    
    if (i < n && j < n) {
        compareExchange(&data[i], &data[j], sort_dir);
    }
}

// Single entry point for CUDA bitonic sort
// data: array to sort (must be on GPU)
// n: array size (must be power of 2)
// ascending: sort direction
extern "C" void cuda_bitonic_sort(uint32_t* data, uint32_t n, bool ascending) {
    if (n <= 1) return;
    
    // Ensure n is power of 2
    uint32_t temp = n;
    bool isPowerOf2 = (temp & (temp - 1)) == 0;
    if (!isPowerOf2) {
        printf("Error: Array size must be power of 2\n");
        return;
    }
    
    if (n <= SHARED_SIZE) {
        // Use single block for small arrays
        uint32_t block_size = (n <= BLOCK_SIZE) ? n : BLOCK_SIZE;
        bitonicSortLocal<<<1, block_size>>>(data, n, ascending);
    } else {
        // Multi-stage approach for larger arrays
        
        // Stage 1: Local sort within blocks
        uint32_t num_blocks = (n + SHARED_SIZE - 1) / SHARED_SIZE;
        bitonicSortLocal<<<num_blocks, BLOCK_SIZE>>>(data, n, ascending);
        cudaDeviceSynchronize();
        
        // Stage 2: Global merging stages
        uint32_t log_shared = 0;
        uint32_t temp_shared = SHARED_SIZE;
        while (temp_shared > 1) { log_shared++; temp_shared >>= 1; }
        
        uint32_t log_n = 0;
        temp = n;
        while (temp > 1) { log_n++; temp >>= 1; }
        
        for (uint32_t stage = log_shared + 1; stage <= log_n; stage++) {
            for (uint32_t substage = stage - 1; substage >= log_shared + 1; substage--) {
                uint32_t threads_needed = n >> 1;
                uint32_t blocks_needed = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                bitonicMergeGlobal<<<blocks_needed, BLOCK_SIZE>>>(data, n, stage, substage, ascending);
                cudaDeviceSynchronize();
            }
        }
    }
}
