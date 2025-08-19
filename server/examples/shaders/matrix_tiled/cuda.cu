
extern "C" __global__ void main(
    float* input_data,
    float* tile_output,
    uint32_t matrix_n,
    uint32_t tile_start_row,
    uint32_t tile_start_col,
    uint32_t tile_rows,
    uint32_t tile_cols,
    uint32_t tile_size
) {
    uint32_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds checking
    if (local_row >= tile_rows || local_col >= tile_cols) {
        return;
    }
    
    uint32_t global_row = tile_start_row + local_row;
    uint32_t global_col = tile_start_col + local_col;
    
    if (global_row >= matrix_n || global_col >= matrix_n) {
        return;
    }
    
    // Input layout: [matrix_size_header, A_data..., B_data...]
    uint32_t header_size = 1;
    uint32_t a_offset = header_size;
    uint32_t b_offset = header_size + matrix_n * matrix_n;
    
    // Matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
    float sum = 0.0f;
    for (uint32_t k = 0; k < matrix_n; k++) {
        float a_val = input_data[a_offset + global_row * matrix_n + k];
        float b_val = input_data[b_offset + k * matrix_n + global_col];
        sum += a_val * b_val;
    }
    
    // Store result in tile-local coordinates
    tile_output[local_row * tile_cols + local_col] = sum;
}
