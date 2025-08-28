__kernel void block_matrix_multiply(
          const uint block_size,              // position 0 - uniforms first (matches executor)
          const uint matrix_size,             // position 1
          __global const float* block_a,      // position 2 - inputs second
          __global const float* block_b,      // position 3
          __global float* partial_result      // position 4 - outputs last
      ) {
          int row = get_global_id(0);
          int col = get_global_id(1);

          if (row >= block_size || col >= block_size) return;

          float sum = 0.0f;
          for (int k = 0; k < block_size; k++) {
              sum += block_a[row * block_size + k] * block_b[k * block_size + col];
          }

          partial_result[row * block_size + col] = sum;
      }
