import { BaseAssemblyStrategy } from '../../strategies/base/BaseAssemblyStrategy.js';

export default class MatrixTiledAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('matrix_tiled_assembly');
  }

  /**
   * Get schema for matrix tiled assembly
   */
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
      const schema = plan.schema || this.getDefaultSchema();
      const matrixSize = completedChunks[0].assemblyMetadata.matrixSize;
      const resultMatrix = new Float32Array(matrixSize * matrixSize);

      // Place each tile in the correct position
      for (const chunk of completedChunks) {
        const {
          startRow,
          startCol,
          tileRows,
          tileCols
        } = chunk.assemblyMetadata;

        // NEW: Handle both single result and multi-result format
        let tileResult;
        if (chunk.results && Array.isArray(chunk.results)) {
          // Multi-result format - take first result
          tileResult = chunk.results[0];
        } else {
          // Single result format (backward compatibility)
          tileResult = chunk.result;
        }

        // Decode the tile result
        const tileBuffer = Buffer.from(tileResult, 'base64');
        const tileData = new Float32Array(tileBuffer.buffer);

        // Copy tile data to the correct position in the result matrix
        for (let i = 0; i < tileRows; i++) {
          for (let j = 0; j < tileCols; j++) {
            const tileIndex = i * tileCols + j;
            const matrixIndex = (startRow + i) * matrixSize + (startCol + j);

            if (tileIndex < tileData.length && matrixIndex < resultMatrix.length) {
              resultMatrix[matrixIndex] = tileData[tileIndex];
            }
          }
        }
      }

      // Convert back to buffer and then base64
      const resultBuffer = Buffer.from(resultMatrix.buffer);
      const outputName = schema.outputs[0].name;

      return {
        success: true,
        outputs: {
          [outputName]: resultBuffer.toString('base64')
        },
        data: resultBuffer.toString('base64'), // Backward compatibility
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          matrixSize,
          format: 'float32_matrix',
          shape: [matrixSize, matrixSize]
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Matrix assembly failed: ${error.message}`
      };
    }
  }

  /**
   * Override validation to check for matrix-specific metadata
   */
  validateChunks(completedChunks, plan) {
    // First run base validation
    const baseValidation = super.validateChunks(completedChunks, plan);
    if (!baseValidation.valid) {
      return baseValidation;
    }

    // Matrix-specific validation
    const expectedTiles = plan.totalChunks;
    const receivedTiles = completedChunks.length;

    if (receivedTiles !== expectedTiles) {
      return {
        valid: false,
        error: `Expected ${expectedTiles} tiles, got ${receivedTiles}`,
        missing: this.findMissingTiles(completedChunks, expectedTiles)
      };
    }

    // Check that we have all tile positions and required metadata
    const tilePositions = new Set();
    let matrixSize = null;

    for (const chunk of completedChunks) {
      if (!chunk.assemblyMetadata) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} missing assembly metadata`
        };
      }

      const { tileRow, tileCol, matrixSize: chunkMatrixSize } = chunk.assemblyMetadata;

      if (tileRow === undefined || tileCol === undefined) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} missing tile position metadata`
        };
      }

      const positionKey = `${tileRow}-${tileCol}`;
      if (tilePositions.has(positionKey)) {
        return {
          valid: false,
          error: `Duplicate tile at position (${tileRow}, ${tileCol})`
        };
      }

      tilePositions.add(positionKey);

      // Validate consistent matrix size
      if (matrixSize === null) {
        matrixSize = chunkMatrixSize;
      } else if (matrixSize !== chunkMatrixSize) {
        return {
          valid: false,
          error: `Inconsistent matrix size: expected ${matrixSize}, got ${chunkMatrixSize} in chunk ${chunk.chunkId}`
        };
      }
    }

    return { valid: true };
  }

  /**
   * Find missing tiles for better error reporting
   */
  findMissingTiles(completedChunks, expectedTiles) {
    const receivedIndices = new Set(
      completedChunks.map(chunk => chunk.chunkIndex ?? chunk.assemblyMetadata?.tileIndex ?? -1)
    );

    const missing = [];
    for (let i = 0; i < expectedTiles; i++) {
      if (!receivedIndices.has(i)) {
        missing.push(i);
      }
    }

    return missing;
  }

  /**
   * Override single output buffer assembly for matrix reconstruction
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    // For matrix tiled assembly, we need to reconstruct the full matrix
    // This method shouldn't be called directly for matrix assembly
    // since we override assembleResults completely
    return Buffer.concat(buffers);
  }

  /**
   * Sort chunks by tile position for proper assembly
   */
  sortChunks(chunks) {
    return chunks.sort((a, b) => {
      // Sort by tile index if available
      if (a.chunkIndex !== undefined && b.chunkIndex !== undefined) {
        return a.chunkIndex - b.chunkIndex;
      }

      // Sort by tile position
      if (a.assemblyMetadata && b.assemblyMetadata) {
        const aRow = a.assemblyMetadata.tileRow || 0;
        const aCol = a.assemblyMetadata.tileCol || 0;
        const bRow = b.assemblyMetadata.tileRow || 0;
        const bCol = b.assemblyMetadata.tileCol || 0;

        if (aRow !== bRow) {
          return aRow - bRow;
        }
        return aCol - bCol;
      }

      // Fallback to base sorting
      return super.sortChunks([a, b])[0] === a ? -1 : 1;
    });
  }
}