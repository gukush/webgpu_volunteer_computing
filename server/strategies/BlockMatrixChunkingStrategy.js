// ENHANCED: BlockMatrixChunkingStrategy.js - Now provides framework-specific shaders
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';

export default class BlockMatrixChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('block_matrix');
  }

  defineInputSchema() {
    return {
      inputs: [
        { name: 'matrix_a_block', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'matrix_b_block', type: 'storage_buffer', binding: 2, elementType: 'f32' }
      ],
      outputs: [
        { name: 'partial_result', type: 'storage_buffer', binding: 3, elementType: 'f32' }
      ],
      uniforms: [
        {
          name: 'params',
          type: 'uniform_buffer',
          binding: 0,
          fields: [
            { name: 'block_size', type: 'u32' },
            { name: 'matrix_size', type: 'u32' }
          ]
        }
      ]
    };
  }

  planExecution(plan) {
    const matrixSize = plan.metadata?.matrixSize;
    const blockSize = plan.metadata?.blockSize;

    if (!matrixSize || !blockSize) {
      throw new Error("plan.metadata must include 'matrixSize' and 'blockSize'.");
    }

    if (matrixSize % blockSize !== 0) {
      throw new Error(`Matrix size ${matrixSize} must be divisible by block size ${blockSize}`);
    }

    const blocksPerDim = Math.floor(matrixSize / blockSize);
    const totalChunks = blocksPerDim * blocksPerDim * blocksPerDim;
    const blockElementCount = blockSize * blockSize;
    const blockByteSize = blockElementCount * 4;

    return {
      strategy: this.name,
      totalChunks,
      assemblyStrategy: 'block_matrix_assembly',
      metadata: {
        ...plan.metadata,
        blocksPerDim,
        blockElementCount,
        blockByteSize,
        totalChunks
      }
    };
  }

  async createChunkDescriptors(plan) {
    const { matrixSize, blockSize, blocksPerDim, blockByteSize } = plan.metadata;
    const framework = plan.framework || 'webgpu';
    console.log(`[STRATEGY DEBUG] createChunkDescriptors called with framework: ${framework}`);
    console.log(`[STRATEGY DEBUG] Plan framework: ${plan.framework}`);
    // Get input data
    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'combined_matrix')
                      || (plan.inputRefs || []).find(r => r.name === 'input');

    let inlineCombinedBuffer = null;
    if (!inputFileRef && plan.metadata?.inputData) {
      inlineCombinedBuffer = Buffer.from(plan.metadata.inputData, 'base64');
    }

    if (!inputFileRef && !inlineCombinedBuffer) {
      throw new Error('BlockMatrix requires input file or inline inputData. Upload files first via POST /api/workloads/:id/inputs');
    }

    if (inputFileRef) {
      const expectedSize = 4 + matrixSize * matrixSize * 2 * 4;
      if (inputFileRef.size !== expectedSize) {
        throw new Error(`Input file size mismatch: expected ${expectedSize} bytes, got ${inputFileRef.size} bytes`);
      }
    }

    const descriptors = [];
    let fileHandle = null;

    try {
      if (inputFileRef?.path) {
        fileHandle = await fs.open(inputFileRef.path, 'r');
      }

      let chunkIndex = 0;
      for (let i = 0; i < blocksPerDim; i++) {
        for (let j = 0; j < blocksPerDim; j++) {
          for (let k = 0; k < blocksPerDim; k++) {
            const aCoords = { row: i, col: k };
            const bCoords = { row: k, col: j };

            let blockA, blockB;
            if (fileHandle) {
              blockA = await this.readBlockFromHandle(fileHandle, 'A', aCoords, blockSize, matrixSize);
              blockB = await this.readBlockFromHandle(fileHandle, 'B', bCoords, blockSize, matrixSize);
            } else if (inlineCombinedBuffer) {
              blockA = this.extractBlockFromBuffer(inlineCombinedBuffer, 'A', aCoords, blockSize, matrixSize);
              blockB = this.extractBlockFromBuffer(inlineCombinedBuffer, 'B', bCoords, blockSize, matrixSize);
            } else {
              throw new Error('No input data available');
            }
            console.log(`[STRATEGY DEBUG] About to create descriptor for framework: ${framework}, chunk: ${i}-${j}-k${k}`);
            // NEW: Framework-specific descriptor creation
            const descriptor = this.createFrameworkSpecificDescriptor(
              framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, plan.parentId, blockByteSize
            );
            console.log(`[STRATEGY DEBUG] Created descriptor keys:`, Object.keys(descriptor));
            console.log(`[STRATEGY DEBUG] Descriptor has webglVertexShader:`, !!descriptor.webglVertexShader);
            console.log(`[STRATEGY DEBUG] Descriptor has kernel:`, !!descriptor.kernel);
            console.log(`[STRATEGY DEBUG] Descriptor framework:`, descriptor.framework);
            if (descriptor.webglVertexShader) {
              console.log(`[STRATEGY DEBUG] WebGL vertex shader length:`, descriptor.webglVertexShader.length);
              console.log(`[STRATEGY DEBUG] WebGL vertex shader preview:`, descriptor.webglVertexShader.substring(0, 100) + '...');
            }
            descriptors.push(descriptor);
            chunkIndex++;
          }
        }
      }

      return descriptors;

    } finally {
      if (fileHandle) {
        await fileHandle.close();
      }
    }
  }

  // NEW: Create framework-specific chunk descriptors
  createFrameworkSpecificDescriptor(framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, parentId, blockByteSize) {
    console.log(`[STRATEGY DEBUG] createFrameworkSpecificDescriptor called with framework: ${framework}`);
    const baseDescriptor = {
      chunkId: `block-${i}-${j}-k${k}`,
      chunkIndex,
      parentId,
      framework,
      inputs: [
        { name: 'matrix_a_block', data: blockA.toString('base64') },
        { name: 'matrix_b_block', data: blockB.toString('base64') }
      ],
      outputs: [{ name: 'partial_result', size: blockByteSize }],
      metadata: { block_size: blockSize, matrix_size: matrixSize },
      assemblyMetadata: { outputBlockRow: i, outputBlockCol: j, kIndex: k }
    };
    console.log(`[STRATEGY DEBUG] Base descriptor created, framework: ${framework}`);

    switch (framework) {
      case 'webgpu':
        return {
          ...baseDescriptor,
          kernel: this.getWebGPUShader(),
          entry: 'main',
          workgroupCount: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1]
        };

      case 'webgl':
        console.log(`[STRATEGY DEBUG] Creating WebGL descriptor`);
        const webglVertexShader = this.getWebGLVertexShader();
        const webglFragmentShader = this.getWebGLFragmentShader();
        console.log(`[STRATEGY DEBUG] Retrieved WebGL shaders:`);
        console.log(`[STRATEGY DEBUG] - Vertex shader length:`, webglVertexShader ? webglVertexShader.length : 'undefined');
        console.log(`[STRATEGY DEBUG] - Fragment shader length:`, webglFragmentShader ? webglFragmentShader.length : 'undefined');

        const descriptor = {
          ...baseDescriptor,
          // WebGL-specific configuration
          webglShaderType: 'transform_feedback',
          webglVertexShader: webglVertexShader,
          webglFragmentShader: webglFragmentShader,
          webglVaryings: ['v_result'],
          webglNumElements: blockSize * blockSize,
          webglInputSpec: {
            type: 'texture',
            format: 'float32',
            internalFormat: 'R32F'
          }
        };

        console.log(`[STRATEGY DEBUG] Final descriptor keys:`, Object.keys(descriptor));
        console.log(`[STRATEGY DEBUG] Has webglVertexShader:`, !!descriptor.webglVertexShader);
        return descriptor;

      case 'vulkan':  // NEW: Add Vulkan support
        console.log(`[STRATEGY DEBUG] Creating Vulkan descriptor`);
        return {
          ...baseDescriptor,
          kernel: this.getVulkanShader(),
          entry: 'main',
          workgroupCount: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1],
          shaderType: 'glsl'  // Indicate this is GLSL for Vulkan
        };

      case 'cuda':
        return {
          ...baseDescriptor,
          kernel: this.getCUDAKernel(),
          entry: 'block_matrix_multiply',
          blockDim: [16, 16, 1],
          gridDim: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1]
        };

      case 'opencl':
        return {
          ...baseDescriptor,
          kernel: this.getOpenCLKernel(),
          entry: 'block_matrix_multiply',
          globalWorkSize: [blockSize, blockSize],
          localWorkSize: [16, 16]
        };

      default:
        throw new Error(`Unsupported framework: ${framework}`);
    }
  }
  // WebGPU shader (existing)
  getWebGPUShader() {
    return `
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
    `;
  }

  // NEW: WebGL vertex shader for matrix block multiplication
  getWebGLVertexShader() {
    return `#version 300 es
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

    `;
  }

  // NEW: WebGL fragment shader (minimal, required but not used for transform feedback)
  getWebGLFragmentShader() {
    return `#version 300 es
      precision highp float;
      out vec4 fragColor;

      void main() {
        fragColor = vec4(1.0);
      }
    `;
  }

  // NEW: CUDA kernel
  getCUDAKernel() {
    return `
      extern "C" __global__ void block_matrix_multiply(
          const float* block_a,
          const float* block_b,
          float* partial_result,
          int block_size,
          int matrix_size
      ) {
          int row = blockIdx.x * blockDim.x + threadIdx.x;
          int col = blockIdx.y * blockDim.y + threadIdx.y;

          if (row >= block_size || col >= block_size) return;

          float sum = 0.0f;
          for (int k = 0; k < block_size; k++) {
              sum += block_a[row * block_size + k] * block_b[k * block_size + col];
          }

          partial_result[row * block_size + col] = sum;
      }
    `;
  }

  // NEW: OpenCL kernel
  getOpenCLKernel() {
    return `
      __kernel void block_matrix_multiply(
          const uint block_size,              // position 0 - uniforms first (matches executor)
          const uint matrix_size,             // position 1
          __global const float* block_a,      // position 2 - inputs second
          __global const float* block_b,      // position 3
          __global float* partial_result      // position 4 - outputs last
      ) {
          int row = get_global_id(0);
          int col = get_global_id(1);

          if (row >= block_size || col >= block_size) return;

          float sum = 0.0f;
          for (int k = 0; k < block_size; k++) {
              sum += block_a[row * block_size + k] * block_b[k * block_size + col];
          }

          partial_result[row * block_size + col] = sum;
      }
    `;
  }
      getVulkanShader() {
      return `#version 450
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

    layout(set = 0, binding = 0) uniform BlockParams {
      uint block_size;
      uint matrix_size;
    } params;

    layout(set = 0, binding = 1) readonly buffer BlockA {
      float block_a[];
    };

    layout(set = 0, binding = 2) readonly buffer BlockB {
      float block_b[];
    };

    layout(set = 0, binding = 3) writeonly buffer PartialResult {
      float partial_result[];
    };

    void main() {
      uint row = gl_GlobalInvocationID.x;
      uint col = gl_GlobalInvocationID.y;

      if (row >= params.block_size || col >= params.block_size) {
        return;
      }

      float sum = 0.0;
      for (uint k = 0; k < params.block_size; k++) {
        float a_val = block_a[row * params.block_size + k];
        float b_val = block_b[k * params.block_size + col];
        sum = sum + a_val * b_val;
      }

      partial_result[row * params.block_size + col] = sum;
    }`;
    }
  // Existing helper methods remain unchanged
  async readBlockFromHandle(fileHandle, matrixType, blockCoords, blockSize, matrixSize) {
    const floatSize = 4;
    const matrixAOffset = 4;
    const matrixBOffset = matrixAOffset + matrixSize * matrixSize * floatSize;
    const baseOffset = (matrixType === 'A') ? matrixAOffset : matrixBOffset;

    const startRow = blockCoords.row * blockSize;
    const startCol = blockCoords.col * blockSize;

    const blockBuffer = Buffer.alloc(blockSize * blockSize * floatSize);
    const rowBytes = blockSize * floatSize;

    for (let r = 0; r < blockSize; r++) {
      const filePos = baseOffset + ((startRow + r) * matrixSize * floatSize) + (startCol * floatSize);
      await fileHandle.read(blockBuffer, r * rowBytes, rowBytes, filePos);
    }

    return blockBuffer;
  }

  extractBlockFromBuffer(combinedBuffer, matrixType, blockCoords, blockSize, matrixSize) {
    const floatSize = 4;
    const matrixAOffset = 4;
    const matrixBOffset = matrixAOffset + matrixSize * matrixSize * floatSize;
    const baseOffset = (matrixType === 'A') ? matrixAOffset : matrixBOffset;

    const startRow = blockCoords.row * blockSize;
    const startCol = blockCoords.col * blockSize;

    const blockBuffer = Buffer.alloc(blockSize * blockSize * floatSize);
    const rowBytes = blockSize * floatSize;

    for (let r = 0; r < blockSize; r++) {
      const sourcePos = baseOffset + ((startRow + r) * matrixSize * floatSize) + (startCol * floatSize);
      const destPos = r * rowBytes;
      combinedBuffer.copy(blockBuffer, destPos, sourcePos, sourcePos + rowBytes);
    }

    return blockBuffer;
  }

  async validateInputs(uploadedFiles, metadata) {
    const { matrixSize } = metadata;

    const matrixFile = uploadedFiles.find(f =>
      f.name === 'combined_matrix' || f.name === 'input'
    );

    if (!matrixFile) {
      return {
        valid: false,
        errors: ['No combined_matrix or input file uploaded']
      };
    }

    const expectedSize = 4 + matrixSize * matrixSize * 2 * 4;
    if (matrixFile.size !== expectedSize) {
      return {
        valid: false,
        errors: [
          `Matrix file size mismatch: expected ${expectedSize} bytes, got ${matrixFile.size} bytes`
        ]
      };
    }

    return { valid: true };
  }
}

