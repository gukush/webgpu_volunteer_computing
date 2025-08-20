// FIXED: BlockMatrixChunkingStrategy.js - Handle deferred file processing
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

  /**
   * FIXED: Plan execution - validate metadata and return plan WITHOUT processing files
   */
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
    const totalChunks = blocksPerDim * blocksPerDim * blocksPerDim; // blocksPerDim³
    const blockElementCount = blockSize * blockSize;
    const blockByteSize = blockElementCount * 4;

    // Return plan without creating chunk descriptors yet
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

  /**
   * NEW: Create chunk descriptors - called AFTER files are uploaded
   */
  async createChunkDescriptors(plan) {
    const { matrixSize, blockSize, blocksPerDim, blockByteSize } = plan.metadata;

    // NOW we can safely access input files
    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'combined_matrix')
                      || (plan.inputRefs || []).find(r => r.name === 'input');

    let inlineCombinedBuffer = null;
    if (!inputFileRef && plan.metadata?.inputData) {
      inlineCombinedBuffer = Buffer.from(plan.metadata.inputData, 'base64');
    }

    if (!inputFileRef && !inlineCombinedBuffer) {
      throw new Error('BlockMatrix requires input file or inline inputData. Upload files first via POST /api/workloads/:id/inputs');
    }

    // Validate file size if using file input
    if (inputFileRef) {
      const expectedSize = 4 + matrixSize * matrixSize * 2 * 4; // header + 2 matrices
      if (inputFileRef.size !== expectedSize) {
        throw new Error(
          `Input file size mismatch: expected ${expectedSize} bytes ` +
          `(header + 2×${matrixSize}×${matrixSize} float32 matrices), got ${inputFileRef.size} bytes`
        );
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

            descriptors.push({
              chunkId: `block-${i}-${j}-k${k}`,
              chunkIndex: chunkIndex++,
              parentId: plan.parentId,
              framework: 'webgpu',
              kernel: this.getBlockMultiplyShader(),
              entry: 'main',
              workgroupCount: [Math.ceil(blockSize / 16), Math.ceil(blockSize / 16), 1],
              inputs: [
                { name: 'matrix_a_block', data: blockA.toString('base64') },
                { name: 'matrix_b_block', data: blockB.toString('base64') }
              ],
              outputs: [{ name: 'partial_result', size: blockByteSize }],
              metadata: { block_size: blockSize, matrix_size: matrixSize },
              assemblyMetadata: { outputBlockRow: i, outputBlockCol: j, kIndex: k }
            });
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
   * Read a block from an open file handle
   */
  async readBlockFromHandle(fileHandle, matrixType, blockCoords, blockSize, matrixSize) {
    const floatSize = 4;
    const matrixAOffset = 4; // Skip 4-byte size header
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

  /**
   * Extract a block from an in-memory buffer
   */
  extractBlockFromBuffer(combinedBuffer, matrixType, blockCoords, blockSize, matrixSize) {
    const floatSize = 4;
    const matrixAOffset = 4; // Skip 4-byte size header
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

  /**
   * Validate input files match expected format
   */
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

    const expectedSize = 4 + matrixSize * matrixSize * 2 * 4; // header + 2 matrices
    if (matrixFile.size !== expectedSize) {
      return {
        valid: false,
        errors: [
          `Matrix file size mismatch: expected ${expectedSize} bytes ` +
          `(header + 2×${matrixSize}×${matrixSize} float32 matrices), got ${matrixFile.size} bytes`
        ]
      };
    }

    return { valid: true };
  }

  getBlockMultiplyShader() {
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
}