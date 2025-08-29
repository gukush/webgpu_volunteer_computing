// ENHANCED: BlockMatrixChunkingStrategy.js - Now provides framework-specific shaders
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';
import { info } from '../logger.js';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';


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
    if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] createChunkDescriptors called with framework: ${framework}`);
    if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Plan framework: ${plan.framework}`);
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
            if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] About to create descriptor for framework: ${framework}, chunk: ${i}-${j}-k${k}`);
            // NEW: Framework-specific descriptor creation
            const descriptor = await this.createFrameworkSpecificDescriptor(
              framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, plan.parentId, blockByteSize
            );
            if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Created descriptor keys:`, Object.keys(descriptor));
            if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Descriptor has webglVertexShader:`, !!descriptor.webglVertexShader);
            if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Descriptor has kernel:`, !!descriptor.kernel);
            if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Descriptor framework:`, descriptor.framework);
            if (descriptor.webglVertexShader) {
              if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] WebGL vertex shader length:`, descriptor.webglVertexShader.length);
              if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] WebGL vertex shader preview:`, descriptor.webglVertexShader.substring(0, 100) + '...');
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

  if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Starting streaming chunk creation`);
  if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Framework: ${framework}`);
  if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Matrix: ${matrixSize}x${matrixSize}, Block: ${blockSize}x${blockSize}`);
  if (__DEBUG_ON__) console.log(` [CHUNKING STRATEGY] Expected chunks: ${blocksPerDim * blocksPerDim * blocksPerDim}`);

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
      console.log(` Opened input file: ${inputFileRef.path}`);
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
            const descriptor = await this.createFrameworkSpecificDescriptor(
              framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, plan.parentId, blockByteSize
            );

            // Add streaming-specific metadata
            descriptor.streamingMetadata = {
              createdAt: Date.now(),
              blockCoords: { i, j, k },
              isStreaming: true,
              totalChunks: blocksPerDim * blocksPerDim * blocksPerDim
            };

            console.log(` Dispatching chunk ${descriptor.chunkId} (${chunkIndex + 1}/${blocksPerDim * blocksPerDim * blocksPerDim})`);

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
            console.error(` ${errorMsg}`);
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
      console.warn(`️  Streaming completed with ${errors.length} errors out of ${totalExpected} chunks`);
    }

    if (__DEBUG_ON__) console.log(` Streaming chunk creation completed: ${dispatchedCount}/${totalExpected} chunks dispatched`);

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
    console.error(` Streaming chunk creation failed:`, error);

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
        console.log(` Closed input file`);
      } catch (closeError) {
        console.warn(`️  Failed to close file handle:`, closeError.message);
      }
    }
  }
}

  // NEW: Create framework-specific chunk descriptors
  async createFrameworkSpecificDescriptor(framework, chunkIndex, i, j, k, blockA, blockB, blockSize, matrixSize, parentId, blockByteSize) {
    if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] createFrameworkSpecificDescriptor called with framework: ${framework}`);
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
    if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Base descriptor created, framework: ${framework}`);
    let kernelSource;
    switch (framework) {
      case 'webgpu':
        kernelSource = await this.getKernelFromFile('webgpu','compute');
        return {
          ...baseDescriptor,
          kernel: kernelSource,
          entry: 'main',
          workgroupCount: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1]
        };

      case 'webgl':
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Creating WebGL descriptor`);
        const webglVertexShader = await this.getKernelFromFile('webgl', 'vertex');
        const webglFragmentShader = await this.getKernelFromFile('webgl', 'fragment');
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Retrieved WebGL shaders:`);
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] - Vertex shader length:`, webglVertexShader ? webglVertexShader.length : 'undefined');
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] - Fragment shader length:`, webglFragmentShader ? webglFragmentShader.length : 'undefined');

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

        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Final descriptor keys:`, Object.keys(descriptor));
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Has webglVertexShader:`, !!descriptor.webglVertexShader);
        return descriptor;
      case 'javascript':
        kernelSource = await this.getKernelFromFile('javascript', 'kernel');
      return {
        ...baseDescriptor,
        kernel: kernelSource,
        entry: 'blockMatrixMultiply',
        workgroupCount: [1, 1, 1], // Not used for JS, but kept for consistency
        jsExecutionHints: {
          algorithm: 'block_matrix_multiply',
          parallelizable: false,
          memoryAccess: 'sequential'
        }
      };
      case 'vulkan':  // NEW: Add Vulkan support
         kernelSource = await this.getKernelFromFile('vulkan', 'compute');
        if (__DEBUG_ON__) console.log(`[STRATEGY DEBUG] Creating Vulkan descriptor`);
        return {
          ...baseDescriptor,
          kernel: kernelSource,
          entry: 'main',
          workgroupCount: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1],
          shaderType: 'glsl'  // Indicate this is GLSL for Vulkan
        };

      case 'cuda':
        kernelSource = await this.getKernelFromFile('cuda', 'kernel');
        return {
          ...baseDescriptor,
          kernel: kernelSource,
          entry: 'block_matrix_multiply',
          blockDim: [16, 16, 1],
          gridDim: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1]
        };

      case 'opencl':
        kernelSource = await this.getKernelFromFile('opencl', 'kernel');
        return {
          ...baseDescriptor,
          kernel: kernelSource,
          entry: 'block_matrix_multiply',
          globalWorkSize: [blockSize, blockSize],
          localWorkSize: [16, 16]
        };

      default:
        throw new Error(`Unsupported framework: ${framework}`);
    }
  }

  buildPJA(plan, desc) {
  const wgsl =
    desc.kernel || desc.wgsl || (desc.kernels?.webgpu?.source) || '';
  const blockSize = Number(desc?.metadata?.block_size ?? 16);
  // Your kernel uses @workgroup_size(16,16,1); dispatch is in tiles
  const tiles = Math.ceil(blockSize / 16);
  const workgroupCount = [tiles, tiles, 1];

  return {
    schemaVersion: '1.0',
    taskType: 'block_matrix_multiply',
    capabilitiesRequired: { compute: true, storageBuffers: true },
    resourceHints: { workgroupCount },
    kernels: wgsl ? { webgpu: { entry: desc.entry || 'main', source: wgsl } } : {},
    script: `
      // BlockMM PJA (single pass)
      const spec = {
        lang: 'webgpu',
        source: (pja.kernels.webgpu && pja.kernels.webgpu.source) || (ctx.chunk.kernel || ctx.chunk.wgsl),
        entry: (pja.kernels.webgpu && pja.kernels.webgpu.entry) || (ctx.chunk.entry || 'main'),
        workgroupCount: (pja.resourceHints && pja.resourceHints.workgroupCount) || (ctx.chunk.workgroupCount || [1,1,1]),
        inputs: ctx.chunk.inputs,
        outputs: ctx.chunk.outputs,
        metadata: ctx.chunk.metadata
      };
      const results = await rt.executeOnGPU(spec);
      return { results };
    `
  };
}

    async getKernelFromFile(framework, type) {
    const filename = `block_matrix_multiply_${framework}_${type}`;
    let extension;
    switch (framework) {
      case 'webgpu':
        extension = 'wgsl';
        break;
      case 'vulkan':
        extension = 'glsl';
        break;
      case 'webgl':
        extension = 'wgsl';
        break;
      case 'javascript':
        extension = 'js';
        break;
      case 'cuda':
        extension = 'cu';
        break;
      case 'opencl':
        extension = 'cl';
        break;
      default:
        throw new Error(`Unknown extension for framework: ${framework}`);
    }
    const kernelUrl = new URL(`../kernels/block_matrix_multiply/${filename}.${extension}`, import.meta.url);
    const kernelPath = fileURLToPath(kernelUrl);
    if (__DEBUG_ON__) console.log(`[FILE IO] Attempting to read kernel from: ${kernelPath}`);
    try {
      const content = await fs.readFile(kernelPath, 'utf8');
      return content;
    } catch (err) {
      console.error(`Error reading kernel file from ${kernelPath}:`, err);
      throw new Error(`Failed to load kernel for ${framework} from file: ${kernelPath}`);
    }
  }

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

