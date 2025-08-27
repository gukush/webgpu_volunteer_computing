// SimpleNeuralNetworkChunkingStrategy.js - Simple 2-layer neural network with ReLU activation
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';
import { info } from '../logger.js';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';

export default class SimpleNeuralNetworkChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('simple_neural_network');
  }

  defineInputSchema() {
    return {
      inputs: [
        { name: 'input_data', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'weights_layer1', type: 'storage_buffer', binding: 2, elementType: 'f32' },
        { name: 'weights_layer2', type: 'storage_buffer', binding: 3, elementType: 'f32' },
        { name: 'biases_layer1', type: 'storage_buffer', binding: 4, elementType: 'f32' },
        { name: 'biases_layer2', type: 'storage_buffer', binding: 5, elementType: 'f32' }
      ],
      outputs: [
        { name: 'output_data', type: 'storage_buffer', binding: 6, elementType: 'f32' }
      ],
      uniforms: [
        {
          name: 'params',
          type: 'uniform_buffer',
          binding: 0,
          fields: [
            { name: 'input_size', type: 'u32' },
            { name: 'hidden_size', type: 'u32' },
            { name: 'output_size', type: 'u32' },
            { name: 'batch_size', type: 'u32' }
          ]
        }
      ]
    };
  }

  planExecution(plan) {
    const { inputSize, hiddenSize, outputSize, batchSize } = plan.metadata;

    if (!inputSize || !hiddenSize || !outputSize || !batchSize) {
      throw new Error("plan.metadata must include 'inputSize', 'hiddenSize', 'outputSize', and 'batchSize'.");
    }

    // For neural networks, we can process multiple batches in parallel
    // Each chunk processes a subset of the batch
    const totalChunks = Math.ceil(batchSize / plan.metadata.chunkSize || 1);
    const chunkSize = Math.min(plan.metadata.chunkSize || batchSize, batchSize);

    return {
      strategy: this.name,
      totalChunks,
      assemblyStrategy: 'simple_neural_network_assembly',
      metadata: {
        ...plan.metadata,
        totalChunks,
        chunkSize,
        inputSize,
        hiddenSize,
        outputSize,
        batchSize
      }
    };
  }

  async createChunkDescriptors(plan) {
    const { inputSize, hiddenSize, outputSize, batchSize, chunkSize, totalChunks } = plan.metadata;
    const framework = plan.framework || 'webgpu';

    if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] createChunkDescriptors called with framework: ${framework}`);

    // Get input data
    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'neural_network_data');
    let inlineData = null;

    if (!inputFileRef && plan.metadata?.inputData) {
      inlineData = Buffer.from(plan.metadata.inputData, 'base64');
    }

    if (!inputFileRef && !inlineData) {
      throw new Error('SimpleNeuralNetwork requires input file or inline inputData. Upload files first via POST /api/workloads/:id/inputs');
    }

    const descriptors = [];
    let fileHandle = null;

    try {
      if (inputFileRef?.path) {
        fileHandle = await fs.open(inputFileRef.path, 'r');
      }

      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        const startBatch = chunkIndex * chunkSize;
        const endBatch = Math.min(startBatch + chunkSize, batchSize);
        const actualChunkSize = endBatch - startBatch;

        let inputChunk, weights1Chunk, weights2Chunk, biases1Chunk, biases2Chunk;

        if (fileHandle) {
          inputChunk = await this.readChunkFromFile(fileHandle, 'input', startBatch, actualChunkSize, inputSize);
          weights1Chunk = await this.readChunkFromFile(fileHandle, 'weights1', 0, hiddenSize * inputSize, 1);
          weights2Chunk = await this.readChunkFromFile(fileHandle, 'weights2', 0, outputSize * hiddenSize, 1);
          biases1Chunk = await this.readChunkFromFile(fileHandle, 'biases1', 0, hiddenSize, 1);
          biases2Chunk = await this.readChunkFromFile(fileHandle, 'biases2', 0, outputSize, 1);
        } else if (inlineData) {
          inputChunk = this.extractChunkFromBuffer(inlineData, 'input', startBatch, actualChunkSize, inputSize);
          weights1Chunk = this.extractChunkFromBuffer(inlineData, 'weights1', 0, hiddenSize * inputSize, 1);
          weights2Chunk = this.extractChunkFromBuffer(inlineData, 'weights2', 0, outputSize * hiddenSize, 1);
          biases1Chunk = this.extractChunkFromBuffer(inlineData, 'biases1', 0, hiddenSize, 1);
          biases2Chunk = this.extractChunkFromBuffer(inlineData, 'biases2', 0, outputSize, 1);
        } else {
          throw new Error('No input data available');
        }

        const descriptor = this.createFrameworkSpecificDescriptor(
          framework, chunkIndex, startBatch, actualChunkSize,
          inputChunk, weights1Chunk, weights2Chunk, biases1Chunk, biases2Chunk,
          inputSize, hiddenSize, outputSize, plan.parentId
        );

        descriptors.push(descriptor);
      }

      return descriptors;

    } finally {
      if (fileHandle) {
        await fileHandle.close();
      }
    }
  }

  /**
   * STREAMING MODE: Create and dispatch chunk descriptors on-demand
   * This method creates chunks one by one and immediately dispatches them via callback
   */
  async createChunkDescriptorsStreaming(plan, dispatchCallback) {
    const { inputSize, hiddenSize, outputSize, batchSize, chunkSize, totalChunks } = plan.metadata;
    const framework = plan.framework || 'webgpu';

    if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Starting streaming chunk creation`);
    if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Framework: ${framework}`);
    if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Network: ${inputSize} → ${hiddenSize} → ${outputSize}`);
    if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Batch: ${batchSize}, Chunk: ${chunkSize}, Total: ${totalChunks}`);

    // Get input data
    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'neural_network_data');
    let inlineData = null;

    if (!inputFileRef && plan.metadata?.inputData) {
      inlineData = Buffer.from(plan.metadata.inputData, 'base64');
    }

    if (!inputFileRef && !inlineData) {
      throw new Error('SimpleNeuralNetwork streaming requires input file or inline inputData');
    }

    let fileHandle = null;
    let chunkIndex = 0;
    let dispatchedCount = 0;
    let errors = [];

    try {
      // Open file if using file input
      if (inputFileRef?.path) {
        fileHandle = await fs.open(inputFileRef.path, 'r');
        console.log(` Opened input file: ${inputFileRef.path}`);
      }

      // Create and dispatch chunks one by one
      for (let i = 0; i < totalChunks; i++) {
        try {
          const startBatch = i * chunkSize;
          const endBatch = Math.min(startBatch + chunkSize, batchSize);
          const actualChunkSize = endBatch - startBatch;

          let inputChunk, weights1Chunk, weights2Chunk, biases1Chunk, biases2Chunk;

          if (fileHandle) {
            inputChunk = await this.readChunkFromFile(fileHandle, 'input', startBatch, actualChunkSize, inputSize);
            weights1Chunk = await this.readChunkFromFile(fileHandle, 'weights1', 0, hiddenSize * inputSize, 1);
            weights2Chunk = await this.readChunkFromFile(fileHandle, 'weights2', 0, outputSize * hiddenSize, 1);
            biases1Chunk = await this.readChunkFromFile(fileHandle, 'biases1', 0, hiddenSize, 1);
            biases2Chunk = await this.readChunkFromFile(fileHandle, 'biases2', 0, outputSize, 1);
          } else if (inlineData) {
            inputChunk = this.extractChunkFromBuffer(inlineData, 'input', startBatch, actualChunkSize, inputSize);
            weights1Chunk = this.extractChunkFromBuffer(inlineData, 'weights1', 0, hiddenSize * inputSize, 1);
            weights2Chunk = this.extractChunkFromBuffer(inlineData, 'weights2', 0, outputSize * hiddenSize, 1);
            biases1Chunk = this.extractChunkFromBuffer(inlineData, 'biases1', 0, hiddenSize, 1);
            biases2Chunk = this.extractChunkFromBuffer(inlineData, 'biases2', 0, outputSize, 1);
          } else {
            throw new Error('No input data available');
          }

          // Create framework-specific descriptor
          const descriptor = this.createFrameworkSpecificDescriptor(
            framework, chunkIndex, startBatch, actualChunkSize,
            inputChunk, weights1Chunk, weights2Chunk, biases1Chunk, biases2Chunk,
            inputSize, hiddenSize, outputSize, plan.parentId
          );

          // Add streaming-specific metadata
          descriptor.streamingMetadata = {
            createdAt: Date.now(),
            batchRange: { start: startBatch, end: endBatch, size: actualChunkSize },
            isStreaming: true,
            totalChunks: totalChunks
          };

          console.log(` Dispatching chunk ${descriptor.chunkId} (${chunkIndex + 1}/${totalChunks})`);

          // IMMEDIATELY dispatch this chunk
          await dispatchCallback(descriptor);

          dispatchedCount++;
          chunkIndex++;

          // Small delay to prevent overwhelming the system
          if (chunkIndex % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 10));
          }

        } catch (chunkError) {
          const errorMsg = `Failed to create/dispatch chunk ${i}: ${chunkError.message}`;
          console.error(` ${errorMsg}`);
          errors.push(errorMsg);

          // Continue with next chunk instead of failing completely
          chunkIndex++;
          continue;
        }
      }

      if (errors.length > 0) {
        console.warn(`️  Streaming completed with ${errors.length} errors out of ${totalChunks} chunks`);
      }

      if (__DEBUG_ON__) console.log(` Streaming chunk creation completed: ${dispatchedCount}/${totalChunks} chunks dispatched`);

      return {
        success: true,
        mode: 'streaming',
        totalChunks: totalChunks,
        dispatchedChunks: dispatchedCount,
        errors: errors.length > 0 ? errors : undefined,
        framework,
        strategy: this.name,
        metadata: {
          inputSize,
          hiddenSize,
          outputSize,
          batchSize,
          chunkSize,
          dispatchDuration: Date.now() - (plan.startTime || Date.now())
        }
      };

    } catch (error) {
      console.error(` Streaming chunk creation failed:`, error);

      return {
        success: false,
        error: `Streaming chunk creation failed: ${error.message}`,
        partialResults: {
          dispatchedChunks: dispatchedCount,
          totalExpected: totalChunks,
          errors
        }
      };

    } finally {
      // Always close file handle
      if (fileHandle) {
        try {
          await fileHandle.close();
          console.log(` Closed input file`);
        } catch (closeError) {
          console.warn(`️  Failed to close file handle:`, closeError.message);
        }
      }
    }
  }

  createFrameworkSpecificDescriptor(framework, chunkIndex, startBatch, chunkSize,
                                  inputChunk, weights1Chunk, weights2Chunk, biases1Chunk, biases2Chunk,
                                  inputSize, hiddenSize, outputSize, parentId) {
    const baseDescriptor = {
      chunkId: `neural-chunk-${chunkIndex}`,
      chunkIndex,
      parentId,
      framework,
      inputs: [
        { name: 'input_data', data: inputChunk.toString('base64') },
        { name: 'weights_layer1', data: weights1Chunk.toString('base64') },
        { name: 'weights_layer2', data: weights2Chunk.toString('base64') },
        { name: 'biases_layer1', data: biases1Chunk.toString('base64') },
        { name: 'biases_layer2', data: biases2Chunk.toString('base64') }
      ],
      outputs: [{ name: 'output_data', size: chunkSize * outputSize * 4 }],
      metadata: {
        input_size: inputSize,
        hidden_size: hiddenSize,
        output_size: outputSize,
        batch_size: chunkSize
      },
      assemblyMetadata: {
        chunkIndex,
        startBatch,
        chunkSize,
        outputSize
      }
    };

    switch (framework) {
      case 'webgpu':
        return {
          ...baseDescriptor,
          kernel: this.getWebGPUShader(),
          entry: 'main',
          workgroupCount: [Math.ceil(chunkSize / 64), 1, 1]
        };

      case 'webgl':
        return {
          ...baseDescriptor,
          webglShaderType: 'transform_feedback',
          webglVertexShader: this.getWebGLVertexShader(),
          webglFragmentShader: this.getWebGLFragmentShader(),
          webglVaryings: ['v_output'],
          webglNumElements: chunkSize * outputSize,
          webglInputSpec: {
            type: 'texture',
            format: 'float32',
            internalFormat: 'R32F'
          }
        };

      case 'javascript':
        return {
          ...baseDescriptor,
          kernel: this.getJavaScriptKernel(),
          entry: 'simpleNeuralNetwork',
          workgroupCount: [1, 1, 1],
          jsExecutionHints: {
            algorithm: 'simple_neural_network',
            parallelizable: false,
            memoryAccess: 'sequential'
          }
        };

      case 'vulkan':
        return {
          ...baseDescriptor,
          kernel: this.getVulkanShader(),
          entry: 'main',
          workgroupCount: [Math.ceil(chunkSize / 64), 1, 1],
          shaderType: 'glsl'
        };

      case 'cuda':
        return {
          ...baseDescriptor,
          kernel: this.getCUDAKernel(),
          entry: 'simple_neural_network',
          blockDim: [64, 1, 1],
          gridDim: [Math.ceil(chunkSize / 64), 1, 1]
        };

      case 'opencl':
        return {
          ...baseDescriptor,
          kernel: this.getOpenCLKernel(),
          entry: 'simple_neural_network',
          globalWorkSize: [chunkSize],
          localWorkSize: [64]
        };

      default:
        throw new Error(`Unsupported framework: ${framework}`);
    }
  }

  getWebGPUShader() {
    return `
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

      fn relu(x: f32) -> f32 {
        return select(0.0, x, x > 0.0);
      }

      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let batch_idx = gid.x;
        if (batch_idx >= params.batch_size) {
          return;
        }

        // Hidden layer computation
        var hidden_layer: array<f32, 1024>; // Assuming max hidden size of 1024
        for (var i = 0u; i < params.hidden_size; i++) {
          var sum = 0.0;
          for (var j = 0u; j < params.input_size; j++) {
            let input_idx = batch_idx * params.input_size + j;
            let weight_idx = i * params.input_size + j;
            sum = sum + input_data[input_idx] * weights_layer1[weight_idx];
          }
          hidden_layer[i] = relu(sum + biases_layer1[i]);
        }

        // Output layer computation
        for (var i = 0u; i < params.output_size; i++) {
          var sum = 0.0;
          for (var j = 0u; j < params.hidden_size; j++) {
            let weight_idx = i * params.hidden_size + j;
            sum = sum + hidden_layer[j] * weights_layer2[weight_idx];
          }
          let output_idx = batch_idx * params.output_size + i;
          output_data[output_idx] = sum + biases_layer2[i];
        }
      }
    `;
  }

  getWebGLVertexShader() {
    return `#version 300 es
      precision highp float;
      precision highp sampler2D;

      in float a_index;
      uniform int u_input_size;
      uniform int u_hidden_size;
      uniform int u_output_size;
      uniform int u_batch_size;
      uniform sampler2D u_input_0; // input_data
      uniform sampler2D u_input_1; // weights_layer1
      uniform sampler2D u_input_2; // weights_layer2
      uniform sampler2D u_input_3; // biases_layer1
      uniform sampler2D u_input_4; // biases_layer2

      out float v_output;

      float relu(float x) {
        return max(0.0, x);
      }

      void main() {
        int idx = int(a_index);
        int batch_idx = idx / u_output_size;
        int output_idx = idx % u_output_size;

        if (batch_idx >= u_batch_size) {
          v_output = 0.0;
          gl_Position = vec4(0.0);
          gl_PointSize = 1.0;
          return;
        }

        // Hidden layer computation
        float hidden_layer[1024]; // Max hidden size
        for (int i = 0; i < u_hidden_size; i++) {
          float sum = 0.0;
          for (int j = 0; j < u_input_size; j++) {
            int input_idx = batch_idx * u_input_size + j;
            int weight_idx = i * u_input_size + j;
            float input_val = texelFetch(u_input_0, ivec2(input_idx, 0), 0).r;
            float weight_val = texelFetch(u_input_1, ivec2(weight_idx, 0), 0).r;
            sum += input_val * weight_val;
          }
          float bias_val = texelFetch(u_input_3, ivec2(i, 0), 0).r;
          hidden_layer[i] = relu(sum + bias_val);
        }

        // Output layer computation
        float sum = 0.0;
        for (int j = 0; j < u_hidden_size; j++) {
          int weight_idx = output_idx * u_hidden_size + j;
          float weight_val = texelFetch(u_input_2, ivec2(weight_idx, 0), 0).r;
          sum += hidden_layer[j] * weight_val;
        }
        float bias_val = texelFetch(u_input_4, ivec2(output_idx, 0), 0).r;
        v_output = sum + bias_val;

        gl_Position = vec4(0.0);
        gl_PointSize = 1.0;
      }
    `;
  }

  getWebGLFragmentShader() {
    return `#version 300 es
      precision highp float;
      out vec4 fragColor;

      void main() {
        fragColor = vec4(1.0);
      }
    `;
  }

  getJavaScriptKernel() {
    return `
      function simpleNeuralNetwork(inputData, weights1, weights2, biases1, biases2, inputSize, hiddenSize, outputSize, batchSize) {
        const output = new Float32Array(batchSize * outputSize);

        function relu(x) {
          return Math.max(0, x);
        }

        for (let batch = 0; batch < batchSize; batch++) {
          // Hidden layer computation
          const hiddenLayer = new Float32Array(hiddenSize);
          for (let i = 0; i < hiddenSize; i++) {
            let sum = 0;
            for (let j = 0; j < inputSize; j++) {
              const inputIdx = batch * inputSize + j;
              const weightIdx = i * inputSize + j;
              sum += inputData[inputIdx] * weights1[weightIdx];
            }
            hiddenLayer[i] = relu(sum + biases1[i]);
          }

          // Output layer computation
          for (let i = 0; i < outputSize; i++) {
            let sum = 0;
            for (let j = 0; j < hiddenSize; j++) {
              const weightIdx = i * hiddenSize + j;
              sum += hiddenLayer[j] * weights2[weightIdx];
            }
            const outputIdx = batch * outputSize + i;
            output[outputIdx] = sum + biases2[i];
          }
        }

        return output;
      }
    `;
  }

  getCUDAKernel() {
    return `
      extern "C" __global__ void simple_neural_network(
          int input_size,           // Uniform 0
          int hidden_size,          // Uniform 1
          int output_size,          // Uniform 2
          int batch_size,           // Uniform 3
          const float* input_data,  // Input 0
          const float* weights1,    // Input 1
          const float* weights2,    // Input 2
          const float* biases1,     // Input 3
          const float* biases2,     // Input 4
          float* output_data        // Output 0
      ) {
          int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

          if (batch_idx >= batch_size) return;

          __shared__ float hidden_layer[1024]; // Max hidden size

          // Hidden layer computation
          for (int i = 0; i < hidden_size; i++) {
              float sum = 0.0f;
              for (int j = 0; j < input_size; j++) {
                  int input_idx = batch_idx * input_size + j;
                  int weight_idx = i * input_size + j;
                  sum += input_data[input_idx] * weights1[weight_idx];
              }
              hidden_layer[i] = fmaxf(0.0f, sum + biases1[i]); // ReLU
          }
          __syncthreads();

          // Output layer computation
          for (int i = 0; i < output_size; i++) {
              float sum = 0.0f;
              for (int j = 0; j < hidden_size; j++) {
                  int weight_idx = i * hidden_size + j;
                  sum += hidden_layer[j] * weights2[weight_idx];
              }
              int output_idx = batch_idx * output_size + i;
              output_data[output_idx] = sum + biases2[i];
          }
      }
    `;
  }

  getOpenCLKernel() {
    return `
      __kernel void simple_neural_network(
          const uint input_size,              // Uniform 0
          const uint hidden_size,             // Uniform 1
          const uint output_size,             // Uniform 2
          const uint batch_size,              // Uniform 3
          __global const float* input_data,   // Input 0
          __global const float* weights1,     // Input 1
          __global const float* weights2,     // Input 2
          __global const float* biases1,      // Input 3
          __global const float* biases2,      // Input 4
          __global float* output_data         // Output 0
      ) {
          int batch_idx = get_global_id(0);

          if (batch_idx >= batch_size) return;

          __local float hidden_layer[1024]; // Max hidden size

          // Hidden layer computation
          for (int i = 0; i < hidden_size; i++) {
              float sum = 0.0f;
              for (int j = 0; j < input_size; j++) {
                  int input_idx = batch_idx * input_size + j;
                  int weight_idx = i * input_size + j;
                  sum += input_data[input_idx] * weights1[weight_idx];
              }
              hidden_layer[i] = fmax(0.0f, sum + biases1[i]); // ReLU
          }
          barrier(CLK_LOCAL_MEM_FENCE);

          // Output layer computation
          for (int i = 0; i < output_size; i++) {
              float sum = 0.0f;
              for (int j = 0; j < hidden_size; j++) {
                  int weight_idx = i * hidden_size + j;
                  sum += hidden_layer[j] * weights2[weight_idx];
              }
              int output_idx = batch_idx * output_size + i;
              output_data[output_idx] = sum + biases2[i];
          }
      }
    `;
  }

  getVulkanShader() {
    return `#version 450
      layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

      layout(set = 0, binding = 0) uniform NeuralParams {
        uint input_size;
        uint hidden_size;
        uint output_size;
        uint batch_size;
      } params;

      layout(set = 0, binding = 1) readonly buffer InputData {
        float input_data[];
      };

      layout(set = 0, binding = 2) readonly buffer Weights1 {
        float weights1[];
      };

      layout(set = 0, binding = 3) readonly buffer Weights2 {
        float weights2[];
      };

      layout(set = 0, binding = 4) readonly buffer Biases1 {
        float biases1[];
      };

      layout(set = 0, binding = 5) readonly buffer Biases2 {
        float biases2[];
      };

      layout(set = 0, binding = 6) writeonly buffer OutputData {
        float output_data[];
      };

      shared float hidden_layer[1024]; // Max hidden size

      float relu(float x) {
        return max(0.0, x);
      }

      void main() {
        uint batch_idx = gl_GlobalInvocationID.x;

        if (batch_idx >= params.batch_size) {
          return;
        }

        // Hidden layer computation
        for (uint i = 0; i < params.hidden_size; i++) {
          float sum = 0.0;
          for (uint j = 0; j < params.input_size; j++) {
            uint input_idx = batch_idx * params.input_size + j;
            uint weight_idx = i * params.input_size + j;
            sum = sum + input_data[input_idx] * weights1[weight_idx];
          }
          hidden_layer[i] = relu(sum + biases1[i]);
        }
        memoryBarrierShared();

        // Output layer computation
        for (uint i = 0; i < params.output_size; i++) {
          float sum = 0.0;
          for (uint j = 0; j < params.hidden_size; j++) {
            uint weight_idx = i * params.hidden_size + j;
            sum = sum + hidden_layer[j] * weights2[weight_idx];
          }
          uint output_idx = batch_idx * params.output_size + i;
          output_data[output_idx] = sum + biases2[i];
        }
      }
    `;
  }

  async readChunkFromFile(fileHandle, dataType, startIdx, count, size) {
    const floatSize = 4;
    let offset = 0;
    let dataSize = 0;

    // Calculate offset based on data type
    switch (dataType) {
      case 'input':
        offset = 4; // Skip header
        dataSize = count * size;
        break;
      case 'weights1':
        offset = 4 + 1024 * 784 * 4; // Skip header + input data
        dataSize = count * size;
        break;
      case 'weights2':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4; // Skip header + input + weights1
        dataSize = count * size;
        break;
      case 'biases1':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4 + 1024 * 1024 * 4; // Skip header + input + weights1 + weights2
        dataSize = count;
        break;
      case 'biases2':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4 + 1024 * 1024 * 4 + 1024 * 4; // Skip header + input + weights1 + weights2 + biases1
        dataSize = count;
        break;
      default:
        throw new Error(`Unknown data type: ${dataType}`);
    }

    const buffer = Buffer.alloc(dataSize * floatSize);
    await fileHandle.read(buffer, 0, dataSize * floatSize, offset);
    return buffer;
  }

  extractChunkFromBuffer(combinedBuffer, dataType, startIdx, count, size) {
    const floatSize = 4;
    let offset = 0;
    let dataSize = 0;

    // Calculate offset based on data type (same logic as file reading)
    switch (dataType) {
      case 'input':
        offset = 4; // Skip header
        dataSize = count * size;
        break;
      case 'weights1':
        offset = 4 + 1024 * 784 * 4; // Skip header + input data
        dataSize = count * size;
        break;
      case 'weights2':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4; // Skip header + input + weights1
        dataSize = count * size;
        break;
      case 'biases1':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4 + 1024 * 1024 * 4; // Skip header + input + weights1 + weights2
        dataSize = count;
        break;
      case 'biases2':
        offset = 4 + 1024 * 784 * 4 + 1024 * 1024 * 4 + 1024 * 1024 * 4 + 1024 * 4; // Skip header + input + weights1 + weights2 + biases1
        dataSize = count;
        break;
      default:
        throw new Error(`Unknown data type: ${dataType}`);
    }

    const buffer = Buffer.alloc(dataSize * floatSize);
    combinedBuffer.copy(buffer, 0, offset, offset + dataSize * floatSize);
    return buffer;
  }

  async validateInputs(uploadedFiles, metadata) {
    const { inputSize, hiddenSize, outputSize, batchSize } = metadata;

    const dataFile = uploadedFiles.find(f =>
      f.name === 'neural_network_data' || f.name === 'input'
    );

    if (!dataFile) {
      return {
        valid: false,
        errors: ['No neural_network_data or input file uploaded']
      };
    }

    // Calculate expected file size
    const headerSize = 4;
    const inputDataSize = batchSize * inputSize * 4;
    const weights1Size = hiddenSize * inputSize * 4;
    const weights2Size = outputSize * hiddenSize * 4;
    const biases1Size = hiddenSize * 4;
    const biases2Size = outputSize * 4;
    const expectedSize = headerSize + inputDataSize + weights1Size + weights2Size + biases1Size + biases2Size;

    if (dataFile.size !== expectedSize) {
      return {
        valid: false,
        errors: [
          `Data file size mismatch: expected ${expectedSize} bytes, got ${dataFile.size} bytes`
        ]
      };
    }

    return { valid: true };
  }
}