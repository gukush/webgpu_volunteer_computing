// WebGPU shader for simple 2-layer neural network with ReLU activation
// This shader processes neural network inference in parallel across batch samples

struct NeuralParams {
  input_size: u32,
  hidden_size: u32,
  output_size: u32,
  batch_size: u32,
}

@group(0) @binding(0) var<uniform> params: NeuralParams;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read> weights_layer1: array<f32>;
@group(0) @binding(3) var<storage, read> weights_layer2: array<f32>;
@group(0) @binding(4) var<storage, read> biases_layer1: array<f32>;
@group(0) @binding(5) var<storage, read> biases_layer2: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_data: array<f32>;

// ReLU activation function
fn relu(x: f32) -> f32 {
  return select(0.0, x, x > 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch_idx = gid.x;

  // Bounds checking
  if (batch_idx >= params.batch_size) {
    return;
  }

  // Hidden layer computation
  var hidden_layer: array<f32, 1024>; // Assuming max hidden size of 1024

  for (var i = 0u; i < params.hidden_size; i++) {
    var sum = 0.0;

    // Matrix multiplication: input * weights_layer1
    for (var j = 0u; j < params.input_size; j++) {
      let input_idx = batch_idx * params.input_size + j;
      let weight_idx = i * params.input_size + j;
      sum = sum + input_data[input_idx] * weights_layer1[weight_idx];
    }

    // Add bias and apply ReLU activation
    hidden_layer[i] = relu(sum + biases_layer1[i]);
  }

  // Output layer computation
  for (var i = 0u; i < params.output_size; i++) {
    var sum = 0.0;

    // Matrix multiplication: hidden * weights_layer2
    for (var j = 0u; j < params.hidden_size; j++) {
      let weight_idx = i * params.hidden_size + j;
      sum = sum + hidden_layer[j] * weights_layer2[weight_idx];
    }

    // Add bias (no activation for output layer in classification)
    let output_idx = batch_idx * params.output_size + i;
    output_data[output_idx] = sum + biases_layer2[i];
  }
}
