
#version 310 es
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std140, binding = 0) uniform TileParams {
    uint matrix_n;
    uint tile_start_row;
    uint tile_start_col;
    uint tile_rows;
    uint tile_cols;
    uint tile_size;
};

layout(std430, binding = 1) readonly buffer InputData {
    float input_data[];
};

layout(std430, binding = 2) writeonly buffer TileOutput {
    float tile_output[];
};

void main() {
    uint local_row = gl_GlobalInvocationID.x;
    uint local_col = gl_GlobalInvocationID.y;
    
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
    uint header_size = 1u;
    uint a_offset = header_size;
    uint b_offset = header_size + matrix_n * matrix_n;
    
    // Matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
    float sum = 0.0;
    for (uint k = 0u; k < matrix_n; k++) {
        float a_val = input_data[a_offset + global_row * matrix_n + k];
        float b_val = input_data[b_offset + k * matrix_n + global_col];
        sum += a_val * b_val;
    }
    
    // Store result in tile-local coordinates
    tile_output[local_row * tile_cols + local_col] = sum;
}
