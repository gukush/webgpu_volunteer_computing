// DistributedConvolutionAssemblyStrategy.js - Assembly strategy for distributed convolution
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class DistributedConvolutionAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('distributed_convolution_assembly');
    this.outputTensor = null;
    this.completedTiles = new Map();
    this.streamingCallbacks = new Map();
  }

  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'output_tensor',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  /**
   * Initialize output tensor for streaming assembly
   */
  async initOutputStore(plan) {
    const {
      outputHeight,
      outputWidth,
      outputChannels,
      tilesY,
      tilesX
    } = plan.metadata;

    const totalOutputSize = outputHeight * outputWidth * outputChannels;

    console.log(`️ Initializing convolution output tensor: ${outputHeight}×${outputWidth}×${outputChannels} = ${totalOutputSize} elements`);

    // Initialize output tensor buffer
    this.outputTensor = new Float32Array(totalOutputSize);
    this.outputHeight = outputHeight;
    this.outputWidth = outputWidth;
    this.outputChannels = outputChannels;
    this.tilesY = tilesY;
    this.tilesX = tilesX;
    this.totalExpectedTiles = tilesY * tilesX;
    this.completedTileCount = 0;

    console.log(` Convolution assembly initialized: expecting ${this.totalExpectedTiles} tiles (${tilesY}×${tilesX})`);
  }

  /**
   * Register callback for streaming assembly
   */
  onBlockComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('tile_complete', callback);
    }
  }

  onAssemblyComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('assembly_complete', callback);
    }
  }

  /**
   * Process single tile result immediately (streaming mode)
   */
  async processChunkResult(chunkResult) {
    if (!this.outputTensor) {
      throw new Error('Output tensor not initialized. Call initOutputStore first.');
    }

    const { assemblyMetadata } = chunkResult;
    if (!assemblyMetadata) {
      throw new Error('Chunk result missing assembly metadata for convolution');
    }

    const {
      tileY,
      tileX,
      outputStartY,
      outputStartX,
      tileOutputHeight,
      tileOutputWidth,
      outputChannels
    } = assemblyMetadata;

    console.log(` Processing convolution tile (${tileY},${tileX}): ${tileOutputHeight}×${tileOutputWidth}×${outputChannels}`);

    // Decode the tile result
    const resultBuffer = Buffer.from(chunkResult.result || chunkResult.results[0], 'base64');
    const tileData = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

    // Copy tile data into correct position in output tensor
    this.copyTileToOutput(
      tileData,
      outputStartY, outputStartX,
      tileOutputHeight, tileOutputWidth,
      outputChannels
    );

    // Mark tile as completed
    const tileId = `${tileY}-${tileX}`;
    this.completedTiles.set(tileId, {
      tileY, tileX,
      completedAt: Date.now(),
      tileSize: tileData.length
    });

    this.completedTileCount++;

    console.log(` Convolution tile (${tileY},${tileX}) completed (${this.completedTileCount}/${this.totalExpectedTiles} total)`);

    // Fire streaming callback
    const callback = this.streamingCallbacks.get('tile_complete');
    if (callback) {
      const progress = (this.completedTileCount / this.totalExpectedTiles) * 100;
      await callback({
        tileY, tileX,
        completedTiles: this.completedTileCount,
        totalTiles: this.totalExpectedTiles,
        progress: progress
      });
    }

    // Check if entire convolution is complete
    if (this.completedTileCount === this.totalExpectedTiles) {
      console.log(` Convolution assembly completed! All ${this.totalExpectedTiles} tiles processed.`);

      const finalResult = await this.finalize();

      // Fire completion callback
      const completeCallback = this.streamingCallbacks.get('assembly_complete');
      if (completeCallback) {
        await completeCallback({
          result: finalResult,
          outputShape: [this.outputHeight, this.outputWidth, this.outputChannels],
          outputTensor: this.outputTensor
        });
      }

      return {
        success: true,
        complete: true,
        result: finalResult
      };
    }

    return {
      success: true,
      complete: false,
      progress: (this.completedTileCount / this.totalExpectedTiles) * 100
    };
  }

  /**
   * Copy tile data to correct position in output tensor
   */
  copyTileToOutput(tileData, startY, startX, tileHeight, tileWidth, channels) {
    let tileIdx = 0;

    for (let y = 0; y < tileHeight; y++) {
      for (let x = 0; x < tileWidth; x++) {
        const outputY = startY + y;
        const outputX = startX + x;

        // Bounds check
        if (outputY < this.outputHeight && outputX < this.outputWidth) {
          const outputBaseIdx = (outputY * this.outputWidth + outputX) * this.outputChannels;

          // Copy all channels for this spatial position
          for (let c = 0; c < channels; c++) {
            this.outputTensor[outputBaseIdx + c] = tileData[tileIdx + c];
          }
        }

        tileIdx += channels;
      }
    }
  }

  /**
   * Finalize assembly and return result
   */
  async finalize() {
    if (!this.outputTensor) {
      throw new Error('Output tensor not initialized');
    }

    console.log(` Finalizing convolution result: ${this.outputHeight}×${this.outputWidth}×${this.outputChannels}`);

    // Convert Float32Array to buffer and then base64
    const resultBuffer = Buffer.from(this.outputTensor.buffer);
    const resultBase64 = resultBuffer.toString('base64');

    return {
      type: 'tensor',
      data: resultBase64,
      shape: [this.outputHeight, this.outputWidth, this.outputChannels],
      dtype: 'float32'
    };
  }

  /**
   * Traditional batch assembly (for compatibility)
   */
  async assembleResults(completedChunks, plan) {
    const validation = this.validateChunks(completedChunks, plan);
    if (!validation.valid) {
      return {
        success: false,
        error: validation.error,
        missing: validation.missing
      };
    }

    try {
      // Initialize if not already done
      if (!this.outputTensor) {
        await this.initOutputStore(plan);
      }

      console.log(` Batch assembling ${completedChunks.length} convolution tiles...`);

      // Sort chunks by tile position
      const sortedChunks = this.sortTileChunks(completedChunks);

      // Process each tile
      for (const chunk of sortedChunks) {
        const { assemblyMetadata } = chunk;
        const {
          outputStartY, outputStartX,
          tileOutputHeight, tileOutputWidth,
          outputChannels
        } = assemblyMetadata;

        const resultBuffer = Buffer.from(chunk.result, 'base64');
        const tileData = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

        this.copyTileToOutput(
          tileData,
          outputStartY, outputStartX,
          tileOutputHeight, tileOutputWidth,
          outputChannels
        );
      }

      // Finalize result
      const finalResult = await this.finalize();

      return {
        success: true,
        outputs: { output_tensor: finalResult.data },
        data: finalResult.data,
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          algorithm: 'distributed_convolution',
          outputShape: finalResult.shape,
          outputDtype: finalResult.dtype,
          totalTiles: completedChunks.length,
          tilesShape: [this.tilesY, this.tilesX]
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Distributed convolution assembly failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  /**
   * Sort chunks by tile position for proper assembly order
   */
  sortTileChunks(chunks) {
    return chunks.sort((a, b) => {
      const aTileY = a.assemblyMetadata?.tileY || 0;
      const aTileX = a.assemblyMetadata?.tileX || 0;
      const bTileY = b.assemblyMetadata?.tileY || 0;
      const bTileX = b.assemblyMetadata?.tileX || 0;

      if (aTileY !== bTileY) {
        return aTileY - bTileY;
      }
      return aTileX - bTileX;
    });
  }

  /**
   * Enhanced validation for convolution tiles
   */
  validateChunks(completedChunks, plan) {
    const baseValidation = super.validateChunks(completedChunks, plan);
    if (!baseValidation.valid) return baseValidation;

    const { tilesY, tilesX } = plan.metadata;
    const expectedTiles = tilesY * tilesX;

    if (completedChunks.length !== expectedTiles) {
      return {
        valid: false,
        error: `Expected ${expectedTiles} tiles (${tilesY}×${tilesX}), got ${completedChunks.length}`,
        missing: this.findMissingTiles(completedChunks, tilesY, tilesX)
      };
    }

    // Validate each tile has proper metadata
    for (const chunk of completedChunks) {
      const { assemblyMetadata } = chunk;
      if (!assemblyMetadata) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId} missing assembly metadata`
        };
      }

      const requiredFields = ['tileY', 'tileX', 'outputStartY', 'outputStartX', 'tileOutputHeight', 'tileOutputWidth'];
      for (const field of requiredFields) {
        if (assemblyMetadata[field] === undefined) {
          return {
            valid: false,
            error: `Chunk ${chunk.chunkId} missing required metadata field: ${field}`
          };
        }
      }

      // Validate tile position is within bounds
      if (assemblyMetadata.tileY >= tilesY || assemblyMetadata.tileX >= tilesX) {
        return {
          valid: false,
          error: `Chunk ${chunk.chunkId} has invalid tile position (${assemblyMetadata.tileY},${assemblyMetadata.tileX}), expected within (${tilesY},${tilesX})`
        };
      }
    }

    return { valid: true };
  }

  /**
   * Find missing tiles for error reporting
   */
  findMissingTiles(completedChunks, tilesY, tilesX) {
    const receivedTiles = new Set();

    completedChunks.forEach(chunk => {
      const { tileY, tileX } = chunk.assemblyMetadata || {};
      if (tileY !== undefined && tileX !== undefined) {
        receivedTiles.add(`${tileY}-${tileX}`);
      }
    });

    const missing = [];
    for (let y = 0; y < tilesY; y++) {
      for (let x = 0; x < tilesX; x++) {
        const tileId = `${y}-${x}`;
        if (!receivedTiles.has(tileId)) {
          missing.push({ tileY: y, tileX: x });
        }
      }
    }

    return missing;
  }

  /**
   * Create enhanced metadata for convolution assembly
   */
  createAssemblyMetadata(plan, chunks) {
    const baseMetadata = super.createAssemblyMetadata(plan, chunks);

    return {
      ...baseMetadata,
      convolutionType: '2d',
      inputShape: [plan.metadata.inputHeight, plan.metadata.inputWidth, plan.metadata.inputChannels],
      outputShape: [plan.metadata.outputHeight, plan.metadata.outputWidth, plan.metadata.outputChannels],
      filterShape: [plan.metadata.filterHeight, plan.metadata.filterWidth],
      stride: [plan.metadata.strideY, plan.metadata.strideX],
      padding: [plan.metadata.paddingY, plan.metadata.paddingX],
      tiling: {
        tilesY: plan.metadata.tilesY,
        tilesX: plan.metadata.tilesX,
        tileHeight: plan.metadata.tileHeight,
        tileWidth: plan.metadata.tileWidth,
        halo: [plan.metadata.haloY, plan.metadata.haloX]
      },
      totalComputeTime: this.calculateTotalComputeTime(chunks)
    };
  }

  /**
   * Calculate total processing time across all tiles
   */
  calculateTotalComputeTime(chunks) {
    return chunks.reduce((total, chunk) => {
      return total + (chunk.processingTime || 0);
    }, 0);
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.outputTensor) {
      this.outputTensor = null;
    }
    this.completedTiles.clear();
    this.streamingCallbacks.clear();
  }
}