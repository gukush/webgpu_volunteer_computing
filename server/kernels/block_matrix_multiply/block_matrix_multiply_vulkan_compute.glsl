#version 450
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

    layout(set = 0, binding = 0) uniform BlockParams {
      uint block_size;
      uint matrix_size;
    } params;

    layout(set = 0, binding = 1) readonly buffer BlockA {
      float block_a[];
    };

    layout(set = 0, binding = 2) readonly buffer BlockB {
      float block_b[];
    };

    layout(set = 0, binding = 3) writeonly buffer PartialResult {
      float partial_result[];
    };

    void main() {
      uint row = gl_GlobalInvocationID.x;
      uint col = gl_GlobalInvocationID.y;

      if (row >= params.block_size || col >= params.block_size) {
        return;
      }

      float sum = 0.0;
      for (uint k = 0; k < params.block_size; k++) {
        float a_val = block_a[row * params.block_size + k];
        float b_val = block_b[k * params.block_size + col];
        sum = sum + a_val * b_val;
      }

      partial_result[row * params.block_size + col] = sum;
    }
