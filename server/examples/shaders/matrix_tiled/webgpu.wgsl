struct TileParams {
    matrix_n: u32,
    tile_start_row: u32,
    tile_start_col: u32,
    tile_rows: u32,
    tile_cols: u32,
    tile_size: u32,
}

@group(0) @binding(0) var<uniform> params: TileParams;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> tile_output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.matrix_n;
    let local_row = global_id.x;
    let local_col = global_id.y;

    // Bounds checking
    if (local_row >= params.tile_rows || local_col >= params.tile_cols) {
        return;
    }

    let global_row = params.tile_start_row + local_row;
    let global_col = params.tile_start_col + local_col;

    if (global_row >= n || global_col >= n) {
        return;
    }

    // Input layout: [matrix_size_header, A_data..., B_data...]
    let header_size = 1u;
    let a_offset = header_size;
    let b_offset = header_size + n * n;

    // Matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
    var sum = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
        let a_val = input_data[a_offset + global_row * n + k];
        let b_val = input_data[b_offset + k * n + global_col];
        sum = sum + a_val * b_val;
    }

    // Store result in tile-local coordinates
    tile_output[local_row * params.tile_cols + local_col] = sum;
}
