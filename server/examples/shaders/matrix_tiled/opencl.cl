
__kernel void main(
    __global const float* input_data,
    __global float* tile_output,
    const uint matrix_n,
    const uint tile_start_row,
    const uint tile_start_col,
    const uint tile_rows,
    const uint tile_cols,
    const uint tile_size
) {
    uint local_row = get_global_id(0);
    uint local_col = get_global_id(1);
    
    // Bounds checking
    if (local_row >= tile_rows || local_col >= tile_cols) {
        return;
    }
    
    uint global_row = tile_start_row + local_row;
    uint global_col = tile_start_col + local_col;
    
    if (global_row >= matrix_n || global_col >= matrix_n) {
        return;
    }
    
    // Input layout: [matrix_size_header, A_data..., B_data...]
    uint header_size = 1;
    uint a_offset = header_size;
    uint b_offset = header_size + matrix_n * matrix_n;
    
    // Matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
    float sum = 0.0f;
    for (uint k = 0; k < matrix_n; k++) {
        float a_val = input_data[a_offset + global_row * matrix_n + k];
        float b_val = input_data[b_offset + k * matrix_n + global_col];
        sum += a_val * b_val;
    }
    
    // Store result in tile-local coordinates
    tile_output[local_row * tile_cols + local_col] = sum;
}
