// strategies/BlockMatrixAssemblyStrategy.js
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class BlockMatrixAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('block_matrix_assembly');
  }

  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'result_matrix',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  assembleResults(completedChunks, plan) {
    const validation = this.validateChunks(completedChunks, plan);
    if (!validation.valid) {
      return {
        success: false,
        error: validation.error,
        missing: validation.missing
      };
    }

    try {
      const { matrixSize, blockSize, blocksPerDim } = plan.metadata;
      
      // Group chunks by output block
      const outputBlocks = this.groupChunksByOutputBlock(completedChunks);
      
      // Sum partial results for each output block
      const assembledBlocks = this.assembleOutputBlocks(outputBlocks, blockSize);
      
      // Reconstruct full matrix from blocks
      const resultMatrix = this.reconstructMatrix(assembledBlocks, matrixSize, blockSize, blocksPerDim);
      
      const resultBuffer = Buffer.from(resultMatrix.buffer);
      
      return {
        success: true,
        outputs: { result_matrix: resultBuffer.toString('base64') },
        data: resultBuffer.toString('base64'),
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          algorithm: 'block_matrix_multiplication',
          blockSize: blockSize,
          blocksPerDim: blocksPerDim,
          totalPartialResults: completedChunks.length,
          totalOutputBlocks: blocksPerDim * blocksPerDim
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Block matrix assembly failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  /**
   * Group chunks by their target output block
   */
  groupChunksByOutputBlock(completedChunks) {
    const groups = new Map();

    for (const chunk of completedChunks) {
      const { outputBlockRow, outputBlockCol, kIndex } = chunk.assemblyMetadata;
      const blockId = `${outputBlockRow}-${outputBlockCol}`;

      if (!groups.has(blockId)) {
        groups.set(blockId, []);
      }

      groups.get(blockId).push({
        chunk,
        kIndex,
        result: chunk.results ? chunk.results[0] : chunk.result
      });
    }

    return groups;
  }

  /**
   * Sum partial results for each output block
   */
  assembleOutputBlocks(outputBlocks, blockSize) {
    const assembledBlocks = new Map();

    for (const [blockId, partialResults] of outputBlocks) {
      // Sort by k index to ensure consistent ordering
      partialResults.sort((a, b) => a.kIndex - b.kIndex);

      // Initialize result block with zeros
      const blockResult = new Float32Array(blockSize * blockSize);

      // Sum all partial results for this block
      for (const partial of partialResults) {
        const partialBuffer = Buffer.from(partial.result, 'base64');
        const partialArray = new Float32Array(partialBuffer.buffer);

        for (let i = 0; i < blockResult.length; i++) {
          blockResult[i] += partialArray[i];
        }
      }

      assembledBlocks.set(blockId, blockResult);
    }

    return assembledBlocks;
  }

  /**
   * Reconstruct full matrix from assembled blocks
   */
  reconstructMatrix(assembledBlocks, matrixSize, blockSize, blocksPerDim) {
    const resultMatrix = new Float32Array(matrixSize * matrixSize);

    for (let blockRow = 0; blockRow < blocksPerDim; blockRow++) {
      for (let blockCol = 0; blockCol < blocksPerDim; blockCol++) {
        const blockId = `${blockRow}-${blockCol}`;
        const block = assembledBlocks.get(blockId);

        if (!block) {
          throw new Error(`Missing assembled block ${blockId}`);
        }

        // Copy block data to correct position in result matrix
        const startRow = blockRow * blockSize;
        const startCol = blockCol * blockSize;

        for (let i = 0; i < blockSize; i++) {
          for (let j = 0; j < blockSize; j++) {
            const blockIdx = i * blockSize + j;
            const matrixIdx = (startRow + i) * matrixSize + (startCol + j);
            resultMatrix[matrixIdx] = block[blockIdx];
          }
        }
      }
    }

    return resultMatrix;
  }

  /**
   * Enhanced validation for block matrix assembly
   */
  validateChunks(completedChunks, plan) {
    const baseValidation = super.validateChunks(completedChunks, plan);
    if (!baseValidation.valid) return baseValidation;

    const { blocksPerDim } = plan.metadata;
    const expectedChunks = blocksPerDim * blocksPerDim * blocksPerDim; // blocksPerDim³

    if (completedChunks.length !== expectedChunks) {
      return {
        valid: false,
        error: `Expected ${expectedChunks} chunks (${blocksPerDim}³), got ${completedChunks.length}`,
        missing: this.findMissingChunks(completedChunks, expectedChunks)
      };
    }

    // Validate we have all required partial results for each output block
    const outputBlocks = this.groupChunksByOutputBlock(completedChunks);
    const expectedOutputBlocks = blocksPerDim * blocksPerDim;

    if (outputBlocks.size !== expectedOutputBlocks) {
      return {
        valid: false,
        error: `Expected ${expectedOutputBlocks} output blocks, got ${outputBlocks.size}`
      };
    }

    // Validate each output block has correct number of partial results
    for (const [blockId, partials] of outputBlocks) {
      if (partials.length !== blocksPerDim) {
        return {
          valid: false,
          error: `Output block ${blockId} has ${partials.length} partial results, expected ${blocksPerDim}`
        };
      }

      // Validate k indices are complete (0 to blocksPerDim-1)
      const kIndices = new Set(partials.map(p => p.kIndex));
      for (let k = 0; k < blocksPerDim; k++) {
        if (!kIndices.has(k)) {
          return {
            valid: false,
            error: `Output block ${blockId} missing partial result for k=${k}`
          };
        }
      }
    }

    return { valid: true };
  }

  /**
   * Find missing chunks for error reporting
   */
  findMissingChunks(completedChunks, expectedTotal) {
    const receivedIndices = new Set(completedChunks.map(c => c.chunkIndex).filter(idx => idx !== undefined));
    const missing = [];

    for (let i = 0; i < expectedTotal; i++) {
      if (!receivedIndices.has(i)) {
        missing.push(i);
      }
    }

    return missing;
  }
}
