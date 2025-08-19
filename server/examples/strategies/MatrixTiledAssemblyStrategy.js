import { BaseAssemblyStrategy } from '../../strategies/base/BaseAssemblyStrategy.js';

export default class MatrixTiledAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('matrix_tiled_assembly');
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

        // Decode the tile result
        const tileBuffer = Buffer.from(chunk.result, 'base64');
        const tileData = new Float32Array(tileBuffer.buffer);

        // Copy tile data to the correct position in the result matrix
        for (let i = 0; i < tileRows; i++) {
          for (let j = 0; j < tileCols; j++) {
            const tileIndex = i * tileCols + j;
            const matrixIndex = (startRow + i) * matrixSize + (startCol + j);
            resultMatrix[matrixIndex] = tileData[tileIndex];
          }
        }
      }

      // Convert back to buffer and then base64
      const resultBuffer = Buffer.from(resultMatrix.buffer);

      return {
        success: true,
        data: resultBuffer.toString('base64'),
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

  validateChunks(completedChunks, plan) {
    const expectedTiles = plan.totalChunks;
    const receivedTiles = completedChunks.length;

    if (receivedTiles !== expectedTiles) {
      return {
        valid: false,
        error: `Expected ${expectedTiles} tiles, got ${receivedTiles}`
      };
    }

    // Check that we have all tile positions
    const tilePositions = new Set();

    for (const chunk of completedChunks) {
      const { tileRow, tileCol } = chunk.assemblyMetadata;
      const positionKey = `${tileRow}-${tileCol}`;

      if (tilePositions.has(positionKey)) {
        return {
          valid: false,
          error: `Duplicate tile at position (${tileRow}, ${tileCol})`
        };
      }

      tilePositions.add(positionKey);
    }

    return { valid: true };
  }
}