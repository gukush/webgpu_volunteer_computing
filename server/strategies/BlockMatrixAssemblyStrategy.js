// ENHANCED: BlockMatrixAssemblyStrategy.js - Complete version with streaming assembly and smart memory management
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import { selectStore } from './storage/MatrixStore.js';
import path from 'path';
import os from 'os';
import { info } from './logger.js';
const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';


export default class BlockMatrixAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('block_matrix_assembly');
    this.outputStore = null;
    this.blockCompleteness = null;
    this.streamingCallbacks = new Map(); // For streaming assembly
    this.MEMORY_THRESHOLD_BYTES = 512 * 1024 * 1024; // 512MB
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

  /**
   * ENHANCED: Initialize output store with smart memory allocation
   */
  async initOutputStore(plan) {
    const { matrixSize, blockSize, blocksPerDim, memoryStrategy } = plan.metadata;
    const matrixBytes = matrixSize * matrixSize * 4;

    // Determine output file path
    const workdir = process.env.VOLUNTEER_STORAGE || path.join(os.tmpdir(), 'volunteer');
    const outputPath = path.join(workdir, `${plan.parentId}_result.bin`);

    // Use memory strategy from planning phase, or determine here
    const strategy = memoryStrategy?.outputStrategy || this.determineOutputStrategy(matrixBytes);
    const preferMemory = strategy === 'memory';

    if (__DEBUG_ON__) console.log(` Assembly strategy: ${strategy} for ${Math.round(matrixBytes/1024/1024)}MB result matrix`);

    this.outputStore = selectStore({
      filePath: outputPath,
      rows: matrixSize,
      cols: matrixSize,
      thresholdBytes: this.MEMORY_THRESHOLD_BYTES,
      preferOutputInMemory: preferMemory,
      writable: true,
      initialize: true
    });

    this.blocksPerDim = blocksPerDim;
    this.blockSize = blockSize;
    this.matrixSize = matrixSize;
    this.blockCompleteness = new Map(); // `${i}-${j}` -> k-count
    this.totalExpectedBlocks = blocksPerDim * blocksPerDim; // Unique output blocks
    this.completedBlocks = 0;

    info.bind(null, 'ASSEMBLY')(`Output store initialized: ${this.outputStore.kind} (${this.totalExpectedBlocks} unique blocks expected)`);
  }

  determineOutputStrategy(matrixBytes) {
    return matrixBytes <= this.MEMORY_THRESHOLD_BYTES ? 'memory' : 'mmap';
  }

  /**
   * ENHANCED: Register callback for streaming assembly
   */
  onBlockComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('block_complete', callback);
    }
  }

  onAssemblyComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('assembly_complete', callback);
    }
  }

  /**
   * ENHANCED: Process single chunk result immediately (streaming mode)
   */
  async processChunkResult(chunkResult) {
    if (!this.outputStore) {
      throw new Error('Output store not initialized. Call initOutputStore first.');
    }

    const { assemblyMetadata } = chunkResult;
    if (!assemblyMetadata) {
      throw new Error('Chunk result missing assembly metadata');
    }

    const { outputBlockRow, outputBlockCol, kIndex } = assemblyMetadata;

    // Decode the partial result
    const resultBuffer = Buffer.from(chunkResult.result || chunkResult.results[0], 'base64');
    const partialBlock = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

    // Accumulate into output store
    this.outputStore.addBlock(outputBlockRow, outputBlockCol, this.blockSize, partialBlock);

    // Track block completion
    const blockId = `${outputBlockRow}-${outputBlockCol}`;
    const currentK = this.blockCompleteness.get(blockId) || 0;
    const newK = currentK + 1;
    this.blockCompleteness.set(blockId, newK);

    if (__DEBUG_ON__) console.log(` Block (${outputBlockRow},${outputBlockCol}) += k${kIndex} result (${newK}/${this.blocksPerDim} k-values)`);

    // Check if this output block is complete
    const blockComplete = this.isBlockComplete(outputBlockRow, outputBlockCol);
    if (blockComplete && this.blockCompleteness.get(blockId) === this.blocksPerDim) {
      // This is the first time this block completed
      this.completedBlocks++;
      if (__DEBUG_ON__) console.log(` Block (${outputBlockRow},${outputBlockCol}) completed (${this.completedBlocks}/${this.totalExpectedBlocks} total)`);

      // Fire streaming callback
      const callback = this.streamingCallbacks.get('block_complete');
      if (callback) {
        const progress = (this.completedBlocks / this.totalExpectedBlocks) * 100;
        await callback({
          blockRow: outputBlockRow,
          blockCol: outputBlockCol,
          completedBlocks: this.completedBlocks,
          totalBlocks: this.totalExpectedBlocks,
          progress: progress
        });
      }
    }

    // Check if entire matrix is complete
    if (this.completedBlocks === this.totalExpectedBlocks) {
      info.bind(null, 'ASSEMBLY')(`Matrix assembly completed! All ${this.totalExpectedBlocks} blocks done.`);

      const finalResult = await this.finalize();

      // Fire completion callback
      const completeCallback = this.streamingCallbacks.get('assembly_complete');
      if (completeCallback) {
        await completeCallback({
          result: finalResult,
          matrixSize: this.matrixSize,
          outputStore: this.outputStore
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
      progress: (this.completedBlocks / this.totalExpectedBlocks) * 100
    };
  }

  /**
   * Check if output block has received all k-contributions
   */
  isBlockComplete(blockRow, blockCol) {
    const blockId = `${blockRow}-${blockCol}`;
    return (this.blockCompleteness.get(blockId) || 0) === this.blocksPerDim;
  }

  /**
   * ENHANCED: Finalize assembly and return result
   */
  async finalize() {
    if (!this.outputStore) {
      throw new Error('Output store not initialized');
    }

    if (__DEBUG_ON__) console.log(` Finalizing ${this.outputStore.kind} result matrix...`);

    if (this.outputStore.kind === 'memory') {
      // Return the Float32Array directly
      return {
        type: 'memory',
        data: this.outputStore.view,
        matrixSize: this.matrixSize
      };
    } else if (this.outputStore.kind === 'mmap') {
      // Return file path and metadata
      return {
        type: 'file',
        path: this.outputStore.filePath,
        matrixSize: this.matrixSize,
        rows: this.matrixSize,
        cols: this.matrixSize
      };
    }

    throw new Error(`Unknown output store type: ${this.outputStore.kind}`);
  }

  /**
   * TRADITIONAL: Batch assembly (for compatibility)
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
      // Initialize store if not already done
      if (!this.outputStore) {
        await this.initOutputStore(plan);
      }

      if (__DEBUG_ON__) console.log(` Batch assembling ${completedChunks.length} chunks...`);

      // Group chunks by output block and process
      const outputBlocks = this.groupChunksByOutputBlock(completedChunks);

      // Process each output block
      for (const [blockId, partialResults] of outputBlocks) {
        const [blockRow, blockCol] = blockId.split('-').map(Number);

        // Sum all partial results for this block
        let blockSum = null;
        for (const partial of partialResults) {
          const resultBuffer = Buffer.from(partial.result, 'base64');
          const partialArray = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

          if (!blockSum) {
            blockSum = new Float32Array(partialArray);
          } else {
            for (let i = 0; i < blockSum.length; i++) {
              blockSum[i] += partialArray[i];
            }
          }
        }

        // Store the accumulated block
        if (blockSum) {
          this.outputStore.putBlock(blockRow, blockCol, this.blockSize, blockSum);
        }
      }

      // Finalize and return result
      const finalResult = await this.finalize();

      return {
        success: true,
        outputs: { result_matrix: this.convertResultToBase64(finalResult) },
        data: this.convertResultToBase64(finalResult),
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          algorithm: 'block_matrix_multiplication',
          blockSize: this.blockSize,
          blocksPerDim: this.blocksPerDim,
          totalPartialResults: completedChunks.length,
          totalOutputBlocks: this.blocksPerDim * this.blocksPerDim,
          outputStore: this.outputStore.kind
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
   * Convert result to base64 for transmission
   */
  convertResultToBase64(result) {
    if (result.type === 'memory') {
      return Buffer.from(result.data.buffer).toString('base64');
    } else if (result.type === 'file') {
      // For file-based results, return null - client should use download endpoint
      return null;
    }
    throw new Error(`Unknown result type: ${result.type}`);
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

    // Sort partial results by k index within each block
    for (const [blockId, partials] of groups) {
      partials.sort((a, b) => a.kIndex - b.kIndex);
    }

    return groups;
  }

  /**
   * Enhanced validation for streaming assembly
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

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.outputStore && typeof this.outputStore.close === 'function') {
      this.outputStore.close();
    }
    this.outputStore = null;
    this.blockCompleteness = null;
    this.streamingCallbacks.clear();
  }
}