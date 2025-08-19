@group(0) @binding(0) var<storage, read> input_data: array<f32>; // [size, A_elements..., B_elements...]
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>; // [C_elements...]

@compute @workgroup_size(8, 8, 1) // Shader's internal workgroup size
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = u32(input_data[0]);

    let r = global_id.x; // row
    let c = global_id.y; // col

    if (r >= size || c >= size) {
        return;
    }

    var sum = 0.0;
    // Offset for matrix A is 1 (after size)
    // Offset for matrix B is 1 + size*size
    let a_offset = 1u;
    let b_offset = 1u + size * size;

    for (var k = 0u; k < size; k = k + 1u) {
        sum = sum + input_data[a_offset + r * size + k] * input_data[b_offset + k * size + c];
    }
    output_data[r * size + c] = sum;
}
