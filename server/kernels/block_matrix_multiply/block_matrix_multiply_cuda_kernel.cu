extern "C" __global__ void block_matrix_multiply(
          int block_size,            // Uniform 0: Block size
          int matrix_size,           // Uniform 1: Matrix size
          const float* block_a,      // Input 0: A block data
          const float* block_b,      // Input 1: B block data
          float* partial_result      // Output 0: Result block
      ) {
          int row = blockIdx.x * blockDim.x + threadIdx.x;
          int col = blockIdx.y * blockDim.y + threadIdx.y;

          // Bounds checking
          if (row >= block_size || col >= block_size) return;

          // Debug: Print thread info for first few threads
          if (row == 0 && col == 0) {
              printf("CUDA kernel: block_size=%d, matrix_size=%d\\n", block_size, matrix_size);
              printf("CUDA kernel: gridDim=(%d,%d,%d), blockDim=(%d,%d,%d)\\n",
                    gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
          }

          float sum = 0.0f;
          for (int k = 0; k < block_size; k++) {
              float a_val = block_a[row * block_size + k];
              float b_val = block_b[k * block_size + col];
              sum += a_val * b_val;
          }

          int output_idx = row * block_size + col;
          partial_result[output_idx] = sum;

          // Debug: Print first result
          if (row == 0 && col == 0) {
              printf("CUDA kernel: first result = %f (a[0]=%f, b[0]=%f)\\n",
                    sum, block_a[0], block_b[0]);
          }
      }
