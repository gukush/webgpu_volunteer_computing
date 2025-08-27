// SimpleNeuralNetworkAssemblyStrategy.js - Assembly strategy for simple 2-layer neural network
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import { selectStore } from './storage/MatrixStore.js';
import path from 'path';
import os from 'os';
import { info } from '../logger.js';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';

export default class SimpleNeuralNetworkAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('simple_neural_network_assembly');
    this.outputStore = null;
    this.chunkCompleteness = null;
    this.streamingCallbacks = new Map();
    this.MEMORY_THRESHOLD_BYTES = 512 * 1024 * 1024; // 512MB
  }

  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'neural_network_output',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  async initOutputStore(plan) {
    const { outputSize, batchSize, memoryStrategy } = plan.metadata;
    const outputBytes = batchSize * outputSize * 4;

    // Determine output file path
    const workdir = process.env.VOLUNTEER_STORAGE || path.join(os.tmpdir(), 'volunteer');
    const outputPath = path.join(workdir, `${plan.parentId}_neural_output.bin`);

    // Use memory strategy from planning phase, or determine here
    const strategy = memoryStrategy?.outputStrategy || this.determineOutputStrategy(outputBytes);
    const preferMemory = strategy === 'memory';

    if (__DEBUG_ON__) console.log(` Assembly strategy: ${strategy} for ${Math.round(outputBytes/1024/1024)}MB output`);

    this.outputStore = selectStore({
      filePath: outputPath,
      rows: batchSize,
      cols: outputSize,
      thresholdBytes: this.MEMORY_THRESHOLD_BYTES,
      preferOutputInMemory: preferMemory,
      writable: true,
      initialize: true
    });

    this.outputSize = outputSize;
    this.batchSize = batchSize;
    this.totalChunks = plan.metadata.totalChunks;
    this.chunkCompleteness = new Map(); // chunkIndex -> completion status
    this.completedChunks = 0;
    this.startTime = Date.now(); // Track start time for streaming progress
    this.chunkCompletionTimes = new Map(); // Track completion times for estimation

    info.bind(null, 'ASSEMBLY')(`Output store initialized: ${this.outputStore.kind} (${this.totalChunks} chunks expected)`);
  }

  determineOutputStrategy(outputBytes) {
    return outputBytes <= this.MEMORY_THRESHOLD_BYTES ? 'memory' : 'mmap';
  }

  onChunkComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('chunk_complete', callback);
    }
  }

  onAssemblyComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('assembly_complete', callback);
    }
  }

  async processChunkResult(chunkResult) {
    if (!this.outputStore) {
      throw new Error('Output store not initialized. Call initOutputStore first.');
    }

    const { assemblyMetadata } = chunkResult;
    if (!assemblyMetadata) {
      throw new Error('Chunk result missing assembly metadata');
    }

    const { chunkIndex, startBatch, chunkSize, outputSize } = assemblyMetadata;

    // Decode the result
    const resultBuffer = Buffer.from(chunkResult.result || chunkResult.results[0], 'base64');
    const outputArray = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

    // Store the chunk result in the output store
    // Each chunk contains a subset of the batch
    for (let batchIdx = 0; batchIdx < chunkSize; batchIdx++) {
      const globalBatchIdx = startBatch + batchIdx;
      const chunkStart = batchIdx * outputSize;
      const chunkEnd = chunkStart + outputSize;
      const batchOutput = outputArray.slice(chunkStart, chunkEnd);

      // Use putBlock with block size 1 × outputSize (treating each batch sample as a block)
      this.outputStore.putBlock(globalBatchIdx, 0, 1, batchOutput);
    }

    // Mark chunk as complete
    this.chunkCompleteness.set(chunkIndex, true);
    this.completedChunks++;

    if (__DEBUG_ON__) console.log(` Chunk ${chunkIndex} completed (${this.completedChunks}/${this.totalChunks} total)`);

    // Fire streaming callback for real-time progress
    const callback = this.streamingCallbacks.get('chunk_complete');
    if (callback) {
      const progress = (this.completedChunks / this.totalChunks) * 100;
      await callback({
        chunkIndex,
        startBatch,
        chunkSize,
        completedChunks: this.completedChunks,
        totalChunks: this.totalChunks,
        progress: progress,
        estimatedTimeRemaining: this.estimateTimeRemaining()
      });
    }

    // Check if all chunks are complete
    if (this.completedChunks === this.totalChunks) {
      info.bind(null, 'ASSEMBLY')(`Neural network assembly completed! All ${this.totalChunks} chunks done.`);

      const finalResult = await this.finalize();

      // Fire completion callback
      const completeCallback = this.streamingCallbacks.get('assembly_complete');
      if (completeCallback) {
        await completeCallback({
          result: finalResult,
          batchSize: this.batchSize,
          outputSize: this.outputSize,
          outputStore: this.outputStore,
          totalProcessingTime: Date.now() - (this.startTime || Date.now())
        });
      }

      return {
        success: true,
        complete: true,
        result: finalResult,
        metadata: {
          totalChunks: this.totalChunks,
          completedChunks: this.completedChunks,
          processingTime: Date.now() - (this.startTime || Date.now())
        }
      };
    }

    return {
      success: true,
      complete: false,
      progress: (this.completedChunks / this.totalChunks) * 100,
      metadata: {
        completedChunks: this.completedChunks,
        totalChunks: this.totalChunks,
        estimatedTimeRemaining: this.estimateTimeRemaining()
      }
    };
  }

  async finalize() {
    if (!this.outputStore) {
      throw new Error('Output store not initialized');
    }

    if (__DEBUG_ON__) console.log(` Finalizing ${this.outputStore.kind} neural network output...`);

    if (this.outputStore.kind === 'memory') {
      // Return the Float32Array directly
      return {
        type: 'memory',
        data: this.outputStore.view,
        batchSize: this.batchSize,
        outputSize: this.outputSize
      };
    } else if (this.outputStore.kind === 'mmap') {
      // Return file path and metadata
      return {
        type: 'file',
        path: this.outputStore.filePath,
        batchSize: this.batchSize,
        outputSize: this.outputSize
      };
    }

    throw new Error(`Unknown output store type: ${this.outputStore.kind}`);
  }

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

      // Sort chunks by chunkIndex to ensure proper ordering
      const sortedChunks = completedChunks.sort((a, b) => a.chunkIndex - b.chunkIndex);

      // Process each chunk
      for (const chunk of sortedChunks) {
        const { assemblyMetadata } = chunk;
        const { startBatch, chunkSize, outputSize } = assemblyMetadata;

        // Decode the result
        const resultBuffer = Buffer.from(chunk.result || chunk.results[0], 'base64');
        const outputArray = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

                // Store the chunk result
        for (let batchIdx = 0; batchIdx < chunkSize; batchIdx++) {
          const globalBatchIdx = startBatch + batchIdx;
          const chunkStart = batchIdx * outputSize;
          const chunkEnd = chunkStart + outputSize;
          const batchOutput = outputArray.slice(chunkStart, chunkEnd);

          // Use putBlock with block size 1 × outputSize (treating each batch sample as a block)
          this.outputStore.putBlock(globalBatchIdx, 0, 1, batchOutput);
        }
      }

      // Finalize and return result
      const finalResult = await this.finalize();

      return {
        success: true,
        outputs: { neural_network_output: this.convertResultToBase64(finalResult) },
        data: this.convertResultToBase64(finalResult),
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          algorithm: 'simple_neural_network',
          batchSize: this.batchSize,
          outputSize: this.outputSize,
          totalChunks: this.totalChunks,
          outputStore: this.outputStore.kind
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Simple neural network assembly failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  convertResultToBase64(result) {
    if (result.type === 'memory') {
      return Buffer.from(result.data.buffer).toString('base64');
    } else if (result.type === 'file') {
      // For file-based results, return null - client should use download endpoint
      return null;
    }
    throw new Error(`Unknown result type: ${result.type}`);
  }

  validateChunks(completedChunks, plan) {
    const baseValidation = super.validateChunks(completedChunks, plan);
    if (!baseValidation.valid) return baseValidation;

    const { totalChunks } = plan.metadata;

    if (completedChunks.length !== totalChunks) {
      return {
        valid: false,
        error: `Expected ${totalChunks} chunks, got ${completedChunks.length}`,
        missing: this.findMissingChunks(completedChunks, totalChunks)
      };
    }

    // Validate we have all required chunk indices
    const chunkIndices = new Set(completedChunks.map(c => c.chunkIndex).filter(idx => idx !== undefined));
    for (let i = 0; i < totalChunks; i++) {
      if (!chunkIndices.has(i)) {
        return {
          valid: false,
          error: `Missing chunk with index ${i}`
        };
      }
    }

    return { valid: true };
  }

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
   * Estimate time remaining based on completed chunks
   */
  estimateTimeRemaining() {
    if (this.completedChunks === 0) return null;

    const elapsed = Date.now() - this.startTime;
    const avgTimePerChunk = elapsed / this.completedChunks;
    const remainingChunks = this.totalChunks - this.completedChunks;

    return Math.round(avgTimePerChunk * remainingChunks);
  }

  /**
   * Get streaming progress information
   */
  getStreamingProgress() {
    return {
      completedChunks: this.completedChunks,
      totalChunks: this.totalChunks,
      progress: (this.completedChunks / this.totalChunks) * 100,
      estimatedTimeRemaining: this.estimateTimeRemaining(),
      elapsedTime: Date.now() - this.startTime,
      throughput: this.completedChunks / ((Date.now() - this.startTime) / 1000) // chunks per second
    };
  }

  async cleanup() {
    if (this.outputStore && typeof this.outputStore.close === 'function') {
      this.outputStore.close();
    }
    this.outputStore = null;
    this.chunkCompleteness = null;
    this.streamingCallbacks.clear();
    this.chunkCompletionTimes = null;
  }
}
