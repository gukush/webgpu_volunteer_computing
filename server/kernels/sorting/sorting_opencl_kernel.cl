#define LOCAL_SIZE 256
#define SHARED_SIZE (LOCAL_SIZE * 2)

inline void compareExchange(__global uint* a, __global uint* b, bool dir) {
    uint x = *a;
    uint y = *b;
    if ((x > y) == dir) {
        *a = y;
        *b = x;
    }
}

inline void compareExchangeLocal(__local uint* a, __local uint* b, bool dir) {
    uint x = *a;
    uint y = *b;
    if ((x > y) == dir) {
        *a = y;
        *b = x;
    }
}

__kernel void bitonicSortLocal(__global uint* data, uint n, uint ascending) {
    __local uint local_data[SHARED_SIZE];
    
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    uint global_tid = gid * SHARED_SIZE + lid;
    
    if (global_tid < n) {
        local_data[lid] = data[global_tid];
    } else {
        local_data[lid] = ascending ? UINT_MAX : 0;
    }
    
    if (global_tid + LOCAL_SIZE < n) {
        local_data[lid + LOCAL_SIZE] = data[global_tid + LOCAL_SIZE];
    } else {
        local_data[lid + LOCAL_SIZE] = ascending ? UINT_MAX : 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint stage = 2; stage <= SHARED_SIZE; stage <<= 1) {
        for (uint substage = stage >> 1; substage > 0; substage >>= 1) {
            uint thread_group = lid / substage;
            uint thread_local = lid % substage;
            
            bool dir = ((thread_group / (stage / substage)) & 1) == 0;
            if (!ascending) dir = !dir;
            
            uint i = thread_group * substage + thread_local;
            uint j = i + substage;
            
            if (thread_local < (substage >> 1) && j < SHARED_SIZE) {
                compareExchangeLocal(&local_data[i], &local_data[j], dir);
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    if (global_tid < n) {
        data[global_tid] = local_data[lid];
    }
    if (global_tid + LOCAL_SIZE < n) {
        data[global_tid + LOCAL_SIZE] = local_data[lid + LOCAL_SIZE];
    }
}

__kernel void bitonicMergeGlobal(__global uint* data, uint n, uint stage, uint substage, uint ascending) {
    uint gid = get_global_id(0);
    
    uint group_size = 1U << substage;
    uint group_id = gid / group_size;
    uint local_id = gid % group_size;
    
    if (local_id >= (group_size >> 1)) return;
    
    uint stage_size = 1U << stage;
    bool sort_dir = ((group_id / (stage_size / group_size)) & 1) == 0;
    if (!ascending) sort_dir = !sort_dir;
    
    uint distance = group_size >> 1;
    uint i = group_id * group_size + local_id;
    uint j = i + distance;
    
    if (i < n && j < n) {
        compareExchange(&data[i], &data[j], sort_dir);
    }
}
