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



/**
 * STREAMING MODE: Create and dispatch chunk descriptors on-demand
 * This method creates chunks one by one and immediately dispatches them via callback
 */
async createChunkDescriptorsStreaming(plan, dispatchCallback) {
  const { matrixSize, blockSize, blocksPerDim, blockByteSize } = plan.metadata;
  const framework = plan.framework || 'webgpu';

  console.log(`🌊 [CHUNKING STRATEGY] Starting streaming chunk creation`);
  console.log(`🌊 [CHUNKING STRATEGY] Framework: ${framework}`);
  console.log(`🌊 [CHUNKING STRATEGY] Matrix: ${matrixSize}x${matrixSize}, Block: ${blockSize}x${blockSize}`);
  console.log(`🌊 [CHUNKING STRATEGY] Expected chunks: ${blocksPerDim * blocksPerDim * blocksPerDim}`);

  // Get input data (same logic as batch mode)
  const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'combined_matrix')
                    || (plan.inputRefs || []).find(r => r.name === 'input');

  let inlineCombinedBuffer = null;
  if (!inputFileRef && plan.metadata?.inputData) {
    inlineCombinedBuffer = Buffer.from(plan.metadata.inputData, 'base64');
  }

  if (!inputFileRef && !inlineCombinedBuffer) {
    throw new Error('BlockMatrix streaming requires input file or inline inputData');
  }

  if (inputFileRef) {
    const expectedSize = 4 + matrixSize * matrixSize * 2 * 4;
    if (inputFileRef.size !== expectedSize) {
      throw new Error(`Input file size mismatch: expected ${expectedSize} bytes, got ${inputFileRef.size} bytes`);
    }
  }

  let fileHandle = null;
  let chunkIndex = 0;
  let dispatchedCount = 0;
  let errors = [];

  try {
    // Open file if using file input
    if (inputFileRef?.path) {
      fileHandle = await fs.open(inputFileRef.path, 'r');
      console.log(`📂 Opened input file: ${inputFileRef.path}`);
    }

    // Create and dispatch chunks one by one
    for (let i = 0; i < blocksPerDim; i++) {
      for (let j = 0; j < blocksPerDim; j++) {
        for (let k = 0; k < blocksPerDim; k++) {
          try {
            // Read matrix blocks (same as batch mode)
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

            // Create framework-specific descriptor
            const descriptor = this.createFrameworkSpecificDescriptor(
              framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, plan.parentId, blockByteSize
            );

            // Add streaming-specific metadata
            descriptor.streamingMetadata = {
              createdAt: Date.now(),
              blockCoords: { i, j, k },
              isStreaming: true,
              totalChunks: blocksPerDim * blocksPerDim * blocksPerDim
            };

            console.log(`🚀 Dispatching chunk ${descriptor.chunkId} (${chunkIndex + 1}/${blocksPerDim * blocksPerDim * blocksPerDim})`);

            // IMMEDIATELY dispatch this chunk
            await dispatchCallback(descriptor);

            dispatchedCount++;
            chunkIndex++;

            // Small delay to prevent overwhelming the system
            if (chunkIndex % 10 === 0) {
              await new Promise(resolve => setTimeout(resolve, 10));
            }

          } catch (chunkError) {
            const errorMsg = `Failed to create/dispatch chunk (${i},${j},k${k}): ${chunkError.message}`;
            console.error(`❌ ${errorMsg}`);
            errors.push(errorMsg);

            // Continue with next chunk instead of failing completely
            chunkIndex++;
            continue;
          }
        }
      }
    }

    const totalExpected = blocksPerDim * blocksPerDim * blocksPerDim;

    if (errors.length > 0) {
      console.warn(`⚠️  Streaming completed with ${errors.length} errors out of ${totalExpected} chunks`);
    }

    console.log(`✅ Streaming chunk creation completed: ${dispatchedCount}/${totalExpected} chunks dispatched`);

    return {
      success: true,
      mode: 'streaming',
      totalChunks: totalExpected,
      dispatchedChunks: dispatchedCount,
      errors: errors.length > 0 ? errors : undefined,
      framework,
      strategy: this.name,
      metadata: {
        matrixSize,
        blockSize,
        blocksPerDim,
        dispatchDuration: Date.now() - (plan.startTime || Date.now())
      }
    };

  } catch (error) {
    console.error(`💥 Streaming chunk creation failed:`, error);

    return {
      success: false,
      error: `Streaming chunk creation failed: ${error.message}`,
      partialResults: {
        dispatchedChunks: dispatchedCount,
        totalExpected: blocksPerDim * blocksPerDim * blocksPerDim,
        errors
      }
    };

  } finally {
    // Always close file handle
    if (fileHandle) {
      try {
        await fileHandle.close();
        console.log(`📂 Closed input file`);
      } catch (closeError) {
        console.warn(`⚠️  Failed to close file handle:`, closeError.message);
      }
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
      case 'javascript':
      return {
        ...baseDescriptor,
        kernel: this.getJavaScriptKernel(),
        entry: 'blockMatrixMultiply',
        workgroupCount: [1, 1, 1], // Not used for JS, but kept for consistency
        jsExecutionHints: {
          algorithm: 'block_matrix_multiply',
          parallelizable: false,
          memoryAccess: 'sequential'
        }
      };
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

  getJavaScriptKernel() {
  return `
    // JavaScript CPU kernel for block matrix multiplication
    // This function will be executed in the browser's JavaScript engine
    function blockMatrixMultiply(blockA, blockB, blockSize) {
      const result = new Float32Array(blockSize * blockSize);

      // Standard matrix multiplication algorithm
      for (let i = 0; i < blockSize; i++) {
        for (let j = 0; j < blockSize; j++) {
          let sum = 0;
          for (let k = 0; k < blockSize; k++) {
            const aVal = blockA[i * blockSize + k];
            const bVal = blockB[k * blockSize + j];
            sum += aVal * bVal;
          }
          result[i * blockSize + j] = sum;
        }
      }

      return result;
    }

    // Metadata for the kernel
    // block_size: Size of the matrix block (N x N)
    // matrix_size: Size of the full matrix (for reference)
  `;
  }
  // NEW: CUDA kernel
  getCUDAKernel() {
    return `
      extern "C" __global__ void block_matrix_multiply(
          int block_size,            // Uniform 0: Block size
          int matrix_size,           // Uniform 1: Matrix size
          const float* block_a,      // Input 0: A block data
          const float* block_b,      // Input 1: B block data
          float* partial_result      // Output 0: Result block
      ) {
          int row = blockIdx.x * blockDim.x + threadIdx.x;
          int col = blockIdx.y * blockDim.y + threadIdx.y;

          // Bounds checking
          if (row >= block_size || col >= block_size) return;

          // Debug: Print thread info for first few threads
          if (row == 0 && col == 0) {
              printf("CUDA kernel: block_size=%d, matrix_size=%d\\n", block_size, matrix_size);
              printf("CUDA kernel: gridDim=(%d,%d,%d), blockDim=(%d,%d,%d)\\n",
                    gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
          }

          float sum = 0.0f;
          for (int k = 0; k < block_size; k++) {
              float a_val = block_a[row * block_size + k];
              float b_val = block_b[k * block_size + col];
              sum += a_val * b_val;
          }

          int output_idx = row * block_size + col;
          partial_result[output_idx] = sum;

          // Debug: Print first result
          if (row == 0 && col == 0) {
              printf("CUDA kernel: first result = %f (a[0]=%f, b[0]=%f)\\n",
                    sum, block_a[0], block_b[0]);
          }
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
  static buildChunks({ A, B, N, blockSize }) {
    if (!(A instanceof Float32Array) || !(B instanceof Float32Array)) {
      throw new Error("A and B must be Float32Array");
    }
    if (A.length !== N * N || B.length !== N * N) {
      throw new Error("A and B must be N*N in length");
    }
    if (N % blockSize !== 0) {
      throw new Error("N must be divisible by blockSize");
    }

    const tiles = N / blockSize;
    const kernelSrc = this.getCUDAKernel();
    const entry = "block_matrix_multiply";
    const outBytes = blockSize * blockSize * 4; // float32

    const chunks = [];
    let chunkCounter = 0;

    // helper: copy a tile (r0..r0+bs-1, c0..c0+bs-1) into a compact Float32Array
    const sliceTile = (src, r0, c0) => {
      const bs = blockSize;
      const tile = new Float32Array(bs * bs);
      for (let r = 0; r < bs; r++) {
        const srcOff = (r0 + r) * N + c0;
        tile.set(src.subarray(srcOff, srcOff + bs), r * bs);
      }
      return tile;
    };

    // Produce chunks for every (i,j,k)
    for (let bi = 0; bi < tiles; bi++) {
      for (let bj = 0; bj < tiles; bj++) {
        for (let bk = 0; bk < tiles; bk++) {
          // Tiles: A(i,k) and B(k,j)
          const aTile = sliceTile(A, bi * blockSize, bk * blockSize);
          const bTile = sliceTile(B, bk * blockSize, bj * blockSize);

          // Pack as bytes for transport (executors receive raw bytes)
          const aBytes = new Uint8Array(aTile.buffer.slice(0)); // copy
          const bBytes = new Uint8Array(bTile.buffer.slice(0)); // copy

          // ID: block-i-j-kk (k-index tagged)
          const chunkId = `block-${bi}-${bj}-k${bk}`;

          // Metadata: declare uniforms explicitly (order + types), plus launch dims.
          // - CUDA uses blockDim/gridDim
          // - OpenCL/WebGPU can use workgroupCount = [blockSize, blockSize, 1]
          const metadata = {
            strategy: "block_matrix",
            // Explicit ABI for uniforms (no special-casing needed in executors)
            schema: {
              uniforms: [
                { name: "block_size",  type: "int32" },
                { name: "matrix_size", type: "int32" }
              ]
            },
            // Values for the uniforms declared above:
            block_size: blockSize,
            matrix_size: N,

            // Launch hints:
            blockDim: [blockSize, blockSize, 1],
            gridDim:  [1, 1, 1],

            // For OpenCL/WebGPU-style executors that use "global size":
            workgroupCount: [blockSize, blockSize, 1],

            // Optional descriptive fields (ignored by executors but helpful upstream)
            tile: { row: bi, col: bj, k: bk, size: blockSize }
          };

          // Chunk payload
          const chunk = {
            id: chunkId,

            // Kernel source & entry symbol:
            kernel: kernelSrc,
            entry,

            // Multi-input & multi-output model:
            inputs: [aBytes, bBytes],
            outputs: [],                // not used; we specify sizes instead
            outputSizes: [outBytes],    // one output buffer (partial C tile) in bytes

            // Executor-independent metadata:
            metadata
          };

          chunks.push(chunk);
          chunkCounter++;
        }
      }
    }

    return { chunks, tiles, chunkCount: chunkCounter, blockSize, N };
  }

  // Convenience: build from JS arrays
  static buildFromArrays({ A, B, N, blockSize }) {
    return this.buildChunks({
      A: A instanceof Float32Array ? A : new Float32Array(A),
      B: B instanceof Float32Array ? B : new Float32Array(B),
      N,
      blockSize
    });
  }
}

