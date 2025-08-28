struct BlockParams {
        block_size: u32,
        matrix_size: u32,
      }

      @group(0) @binding(0) var<uniform> params: BlockParams;
      @group(0) @binding(1) var<storage, read> block_a: array<f32>;
      @group(0) @binding(2) var<storage, read> block_b: array<f32>;
      @group(0) @binding(3) var<storage, read_write> partial_result: array<f32>;

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let row = gid.x;
        let col = gid.y;
        if (row >= params.block_size || col >= params.block_size) {
          return;
        }
        var sum = 0.0;
        for (var k = 0u; k < params.block_size; k++) {
          let a_val = block_a[row * params.block_size + k];
          let b_val = block_b[k * params.block_size + col];
          sum = sum + a_val * b_val;
        }
        partial_result[row * params.block_size + col] = sum;
      }
