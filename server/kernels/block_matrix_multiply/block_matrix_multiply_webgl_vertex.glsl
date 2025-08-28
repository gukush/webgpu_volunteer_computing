#version 300 es
      precision highp float;
      precision highp sampler2D;

      in float a_index;
      uniform int u_block_size;
      uniform sampler2D u_input_0; // A block, size = block_size x block_size
      uniform sampler2D u_input_1; // B block, size = block_size x block_size

      out float v_result;

      void main() {
        int idx = int(a_index);
        int n = u_block_size;
        int r = idx / n;
        int c = idx % n;

        float sum = 0.0;
        for (int k = 0; k < n; ++k) {
          float a_val = texelFetch(u_input_0, ivec2(k, r), 0).r; // A[r,k]
          float b_val = texelFetch(u_input_1, ivec2(c, k), 0).r; // B[k,c]
          sum += a_val * b_val;
        }
        v_result = sum;
        gl_Position = vec4(0.0);   // rasterizer discard will be enabled
        gl_PointSize = 1.0;
      }
