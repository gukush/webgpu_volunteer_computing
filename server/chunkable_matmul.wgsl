// Chunkable WGSL Shader for Matrix Multiplication

struct ChunkParams {
    matrix_n: u32,                 // The N dimension of the N x N matrices (original 'size')
    output_start_row: u32,         // The starting row index in the *original C matrix* this chunk computes
    rows_to_compute_this_chunk: u32, // Number of rows of C this chunk is responsible for
    // total_original_input_size_bytes: u32, // (Optional, server sends this, client might not need it in shader if input is always full)
                                          // This can be part of the generic chunkUniforms the server provides.
};

// @group(0) @binding(0) is conventionally for uniforms in the framework
@group(0) @binding(0) var<uniform> params: ChunkParams;

// @group(0) @binding(1) is conventionally for the primary input data
// For this matrix multiplication, the "chunk input" is the full original input data
// containing the original size (though we'll use params.matrix_n) and matrices A and B.
// Format: [original_size_param, A_elements..., B_elements...]
@group(0) @binding(1) var<storage, read> original_input_data: array<f32>;

// @group(0) @binding(2) is conventionally for the output data of this chunk
// This buffer will be sized by the client/server to hold:
// `rows_to_compute_this_chunk * params.matrix_n` f32 elements.
@group(0) @binding(2) var<storage, read_write> chunk_output_data: array<f32>;


@compute @workgroup_size(8, 8, 1) // Keep the shader's original workgroup size
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Use matrix_n from uniform parameters for clarity and consistency.
    let n = params.matrix_n;

    // global_id.x refers to the row index *within the set of rows this chunk is computing*.
    // global_id.y refers to the column index *within the full width of the output matrix*.
    let local_row_in_chunk = global_id.x;
    let col_in_matrix = global_id.y;

    // Boundary check for the work assigned to this specific chunk.
    // Ensure we don't try to compute more rows than assigned or more columns than exist.
    if (local_row_in_chunk >= params.rows_to_compute_this_chunk || col_in_matrix >= n) {
        return;
    }

    // Determine the actual row in the *original, full C matrix* this invocation corresponds to.
    let original_matrix_row = params.output_start_row + local_row_in_chunk;

    // This is an additional safety check; if server parameters are correct,
    // original_matrix_row should always be < n if local_row_in_chunk < params.rows_to_compute_this_chunk.
    if (original_matrix_row >= n) {
        return;
    }

    var sum = 0.0;

    // The original_input_data still has the layout: [size_val, A_data..., B_data...].
    // We use params.matrix_n (which is 'n') for dimensions.
    // The element at original_input_data[0] could be 'n', but relying on the uniform is cleaner.
    let a_data_start_offset = 1u; // Data for matrix A starts after the first element (original size parameter).
    let b_data_start_offset = 1u + n * n; // Data for matrix B starts after A's data.

    for (var k = 0u; k < n; k = k + 1u) {
        let a_val = original_input_data[a_data_start_offset + original_matrix_row * n + k];
        let b_val = original_input_data[b_data_start_offset + k * n + col_in_matrix];
        sum = sum + a_val * b_val;
    }

    // Write the result to the chunk_output_data.
    // The output is indexed by `local_row_in_chunk` because chunk_output_data
    // is sized only for the rows this chunk is computing.
    chunk_output_data[local_row_in_chunk * n + col_in_matrix] = sum;
}