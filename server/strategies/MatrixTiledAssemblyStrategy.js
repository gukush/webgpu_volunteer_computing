// strategies/MatrixTiledAssemblyStrategy.js
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import fs from 'fs';
import path from 'path';

export default class MatrixTiledAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('matrix_tiled_assembly');
  }

  /**
   * FIXED: Get schema that matches chunking strategy
   */
  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'output', // FIXED: Match chunking strategy output name
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  assembleResults(completedChunks, plan) {
    // Optional: write assembled result directly to a file if plan.metadata.outputPath is set
    const wantFile = plan && plan.metadata && plan.metadata.outputPath;

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

      // Extract matrix size from the first chunk's metadata
      let matrixSize;
      if (completedChunks[0].assemblyMetadata) {
        matrixSize = completedChunks[0].assemblyMetadata.matrixSize;
      } else {
        // Fallback: extract from plan metadata
        matrixSize = plan.metadata?.matrixSize;
      }

      if (!matrixSize) {
        return {
          success: false,
          error: 'Could not determine matrix size from chunks or plan metadata'
        };
      }

    let outFd = null;
    if (wantFile) {
      try {
        const outPath = path.resolve(plan.metadata.outputPath);
        outFd = fs.openSync(outPath, 'w');
        // Preallocate file size
        fs.ftruncateSync(outFd, matrixSize * matrixSize * 4);
      } catch (e) {
        console.warn('Failed to open outputPath for writing, falling back to in-memory:', e.message);
      }
    }

      const resultMatrix = new Float32Array(matrixSize * matrixSize);

      // Sort chunks to ensure proper assembly order
      const sortedChunks = this.sortChunks(completedChunks);

      // Place each tile in the correct position
      for (const chunk of sortedChunks) {
        const {
          startRow,
          startCol,
          tileRows,
          tileCols
        } = chunk.assemblyMetadata;

        // FIXED: Handle both single result and multi-result format
        let tileResult;
        if (chunk.results && Array.isArray(chunk.results)) {
          // Multi-result format - take first result
          tileResult = chunk.results[0];
        } else {
          // Single result format (backward compatibility)
          tileResult = chunk.result;
        }

        if (!tileResult) {
          return {
            success: false,
            error: `Chunk ${chunk.chunkId || chunk.chunkIndex} has no result data`
          };
        }

        // Decode the tile result
        const tileBuffer = Buffer.from(tileResult, 'base64');
        const tileData = new Float32Array(tileBuffer.buffer);

        // FIXED: Copy tile data to the correct position in the result matrix
        for (let i = 0; i < tileRows; i++) {
          for (let j = 0; j < tileCols; j++) {
            const tileIndex = i * tileCols + j;
            const globalRow = startRow + i;
            const globalCol = startCol + j;
            const matrixIndex = globalRow * matrixSize + globalCol;

            if (tileIndex < tileData.length &&
                matrixIndex < resultMatrix.length &&
                globalRow < matrixSize &&
                globalCol < matrixSize) {
            if (resultMatrix) {
              resultMatrix[matrixIndex] = tileData[tileIndex];
            } else if (wantFile && outFd) {
              const byteOffset = matrixIndex * 4;
              // Write one float32 at the correct position
              const tmpBuf = Buffer.allocUnsafe(4);
              tmpBuf.writeFloatLE(tileData[tileIndex], 0);
              fs.writeSync(outFd, tmpBuf, 0, 4, byteOffset);
            }
            }
          }
        }
      }

      // Convert back to buffer and then base64
      if (wantFile && outFd) {
        fs.closeSync(outFd);
        return {
          success: true,
          finalResult: { path: path.resolve(plan.metadata.outputPath) }
        };
      }
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
          shape: [matrixSize, matrixSize],
          totalTiles: completedChunks.length,
          assemblyStrategy: this.name
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Matrix assembly failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  /**
   * FIXED: Enhanced validation to check for matrix-specific metadata
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
      // Check for result data
      const hasResult = (chunk.results && Array.isArray(chunk.results) && chunk.results.length > 0) ||
                       chunk.result;

      if (!hasResult) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} has no result data`
        };
      }

      if (!chunk.assemblyMetadata) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} missing assembly metadata`
        };
      }

      const {
        tileRow,
        tileCol,
        matrixSize: chunkMatrixSize,
        startRow,
        startCol,
        tileRows,
        tileCols
      } = chunk.assemblyMetadata;

      // Validate required metadata fields
      if (tileRow === undefined || tileCol === undefined) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} missing tile position metadata`
        };
      }

      if (startRow === undefined || startCol === undefined ||
          tileRows === undefined || tileCols === undefined) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId || chunk.chunkIndex} missing tile dimension metadata`
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
    const receivedIndices = new Set();

    for (const chunk of completedChunks) {
      if (chunk.chunkIndex !== undefined) {
        receivedIndices.add(chunk.chunkIndex);
      } else if (chunk.assemblyMetadata?.tileIndex !== undefined) {
        receivedIndices.add(chunk.assemblyMetadata.tileIndex);
      }
    }

    const missing = [];
    for (let i = 0; i < expectedTiles; i++) {
      if (!receivedIndices.has(i)) {
        missing.push(i);
      }
    }

    return missing;
  }

  /**
   * FIXED: Sort chunks by tile position for proper assembly
   */
  sortChunks(chunks) {
    return chunks.sort((a, b) => {
      // Sort by chunk index if available
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

  /**
   * Override single output buffer assembly for matrix reconstruction
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    // For matrix tiled assembly, we override assembleResults completely
    // This method is called by the base class but not used for tiled matrix assembly
    console.warn('assembleSingleOutputBuffers called for matrix_tiled_assembly - this should not happen');
    return Buffer.concat(buffers);
  }

  /**
   * NEW: Handle multiple outputs (for future extension)
   */
  assembleMultipleOutputs(sortedChunks, plan, schema) {
    if (schema.outputs.length > 1) {
      console.warn('Matrix tiled assembly does not yet support multiple outputs, using first output only');
    }

    // For now, treat as single output
    return this.assembleSingleOutput(sortedChunks, plan, schema);
  }

  /**
   * Create enhanced assembly metadata
   */
  createAssemblyMetadata(plan, chunks) {
    const baseMetadata = super.createAssemblyMetadata(plan, chunks);

    return {
      ...baseMetadata,
      matrixStrategy: 'tiled',
      tilesProcessed: chunks.length,
      averageProcessingTime: chunks.reduce((sum, chunk) =>
        sum + (chunk.processingTime || 0), 0) / chunks.length
    };
  }
}