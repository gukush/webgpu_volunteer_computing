// Enhanced chunking manager with streaming support
class StreamingChunkingManager {
  
  async processStreamingWorkload(workload) {
    try {
      // Get the chunking strategy
      const strategy = this.getStrategy(workload.chunkingStrategy);
      
      // Check if strategy supports streaming
      if (strategy.supportsStreaming && strategy.supportsStreaming()) {
        console.log(`Using streaming mode for strategy ${workload.chunkingStrategy}`);
        return await this.processStreamingChunks(workload, strategy);
      } else {
        console.log(`Using batch mode for strategy ${workload.chunkingStrategy}`);
        return await this.processBatchChunks(workload, strategy);
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async processStreamingChunks(workload, strategy) {
    const streamingPlan = await strategy.createStreamingPlan(workload);
    const chunkStore = this.createChunkStore(workload.id, streamingPlan);
    
    // Start streaming chunk generation in the background
    this.startStreamingGeneration(workload, strategy, streamingPlan, chunkStore);
    
    return {
      success: true,
      plan: streamingPlan,
      streaming: true,
      message: 'Streaming chunk generation started'
    };
  }

  async startStreamingGeneration(workload, strategy, plan, chunkStore) {
    try {
      console.log(`Starting streaming generation for ${workload.id}`);
      
      // Create async generator for chunks
      const chunkGenerator = strategy.generateChunksStream(workload, plan);
      
      let chunkIndex = 0;
      for await (const chunkDescriptor of chunkGenerator) {
        // Add chunk to store as soon as it's ready
        chunkStore.allChunkDefs.push({
          ...chunkDescriptor,
          chunkIndex,
          status: 'queued',
          dispatchesMade: 0,
          submissions: [],
          activeAssignments: new Set(),
          assignedClients: new Set(),
          verified_results: null
        });
        
        console.log(`Stream: Generated chunk ${chunkDescriptor.chunkId} (${chunkIndex + 1}/${plan.totalChunks})`);
        
        // Notify that new chunks are available for assignment
        this.notifyChunksAvailable(workload.id);
        
        chunkIndex++;
      }
      
      console.log(`Stream: Completed generation of ${chunkIndex} chunks for ${workload.id}`);
      chunkStore.generationComplete = true;
      
    } catch (error) {
      console.error(`Stream generation failed for ${workload.id}:`, error);
      chunkStore.generationError = error.message;
    }
  }

  notifyChunksAvailable(workloadId) {
    // Trigger chunk assignment check
    process.nextTick(() => {
      // This would be called by the main server loop
      global.assignCustomChunkToAvailableClients?.();
    });
  }
}

// Enhanced BlockMatrixChunkingStrategy with streaming support
export default class BlockMatrixChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('block_matrix');
  }

  supportsStreaming() {
    return true;
  }

  async createStreamingPlan(workload) {
    const { matrixSize, blockSize } = workload.metadata;
    const blocksPerDim = matrixSize / blockSize;
    const totalChunks = blocksPerDim * blocksPerDim * blocksPerDim;

    return {
      strategy: this.name,
      totalChunks,
      metadata: {
        ...workload.metadata,
        blocksPerDim,
        streamingEnabled: true
      },
      assemblyStrategy: 'block_matrix_assembly',
      schema: this.defineInputSchema()
    };
  }

  async *generateChunksStream(workload, plan) {
    const { matrixSize, blockSize, blocksPerDim } = plan.metadata;
    
    // Create a streaming file reader for large matrices
    const matrixReader = await this.createMatrixReader(workload);
    
    let chunkIndex = 0;

    // Generate chunks as matrix blocks become available
    for (let outputRow = 0; outputRow < blocksPerDim; outputRow++) {
      for (let outputCol = 0; outputCol < blocksPerDim; outputCol++) {
        for (let k = 0; k < blocksPerDim; k++) {
          
          // Load only the required blocks for this chunk
          const blockA = await matrixReader.getBlock('A', outputRow, k, blockSize);
          const blockB = await matrixReader.getBlock('B', k, outputCol, blockSize);
          
          const outputSize = blockSize * blockSize * 4;

          yield {
            chunkId: `block-${outputRow}-${outputCol}-k${k}`,
            parentId: workload.id,
            framework: 'webgpu',
            kernel: this.getBlockMultiplyShader(),
            entry: 'main',
            workgroupCount: [
              Math.ceil(blockSize / 16),
              Math.ceil(blockSize / 16),
              1
            ],

            inputs: [
              { 
                name: 'matrix_a_block', 
                data: Buffer.from(blockA.buffer).toString('base64') 
              },
              { 
                name: 'matrix_b_block', 
                data: Buffer.from(blockB.buffer).toString('base64') 
              }
            ],

            outputs: [
              { name: 'partial_result', size: outputSize }
            ],

            metadata: {
              block_size: blockSize,
              matrix_size: matrixSize,
              outputBlockRow: outputRow,
              outputBlockCol: outputCol,
              kIndex: k
            },

            schema: this.defineInputSchema()
          };

          chunkIndex++;
          
          // Log progress periodically
          if (chunkIndex % 10 === 0) {
            console.log(`Stream: Generated ${chunkIndex}/${plan.totalChunks} chunks...`);
          }
        }
      }
    }

    // Cleanup reader
    await matrixReader.close();
  }

  async createMatrixReader(workload) {
    if (workload.inputRefs && workload.inputRefs[0]) {
      // File-based input - use streaming reader
      return new StreamingMatrixReader(workload.inputRefs[0].path, workload.metadata.matrixSize);
    } else if (workload.loadedInputs && workload.loadedInputs.input) {
      // In-memory input - use memory reader
      return new MemoryMatrixReader(workload.loadedInputs.input, workload.metadata.matrixSize);
    } else {
      throw new Error('No valid matrix input source found');
    }
  }

  // ... rest of existing methods ...
}

// Streaming file reader for large matrix files
class StreamingMatrixReader {
  constructor(filePath, matrixSize) {
    this.filePath = filePath;
    this.matrixSize = matrixSize;
    this.blockCache = new Map(); // Cache for recently accessed blocks
    this.maxCacheSize = 50; // Limit cache size
  }

  async getBlock(matrixName, blockRow, blockCol, blockSize) {
    const blockKey = `${matrixName}-${blockRow}-${blockCol}`;
    
    // Check cache first
    if (this.blockCache.has(blockKey)) {
      return this.blockCache.get(blockKey);
    }

    // Calculate file offset for this block
    const offset = this.calculateBlockOffset(matrixName, blockRow, blockCol, blockSize);
    
    // Read only the required block data from file
    const blockData = await this.readBlockFromFile(offset, blockSize);
    
    // Add to cache (with LRU eviction)
    this.addToCache(blockKey, blockData);
    
    return blockData;
  }

  calculateBlockOffset(matrixName, blockRow, blockCol, blockSize) {
    const headerSize = 4; // 4-byte size header
    const matrixSizeBytes = this.matrixSize * this.matrixSize * 4; // float32
    
    let baseOffset = headerSize;
    if (matrixName === 'B') {
      baseOffset += matrixSizeBytes; // Skip matrix A
    }

    // Calculate offset within the matrix for this block
    const blockStartRow = blockRow * blockSize;
    const blockStartCol = blockCol * blockSize;
    
    // For now, we need to read row by row since matrix is stored row-major
    // In a more sophisticated implementation, we could read the entire block at once
    // if the file format supported it
    return {
      baseOffset,
      blockStartRow,
      blockStartCol,
      blockSize
    };
  }

  async readBlockFromFile(offsetInfo, blockSize) {
    const fs = await import('fs/promises');
    const fileHandle = await fs.open(this.filePath, 'r');
    
    try {
      const block = new Float32Array(blockSize * blockSize);
      const rowSizeBytes = this.matrixSize * 4; // One row in bytes
      
      // Read block row by row
      for (let i = 0; i < blockSize; i++) {
        const rowOffset = offsetInfo.baseOffset + 
          (offsetInfo.blockStartRow + i) * rowSizeBytes + 
          offsetInfo.blockStartCol * 4;
        
        const rowBuffer = Buffer.allocUnsafe(blockSize * 4);
        await fileHandle.read(rowBuffer, 0, blockSize * 4, rowOffset);
        
        // Copy to block array
        const rowData = new Float32Array(rowBuffer.buffer);
        for (let j = 0; j < blockSize; j++) {
          block[i * blockSize + j] = rowData[j];
        }
      }
      
      return block;
    } finally {
      await fileHandle.close();
    }
  }

  addToCache(key, data) {
    // Simple LRU cache implementation
    if (this.blockCache.size >= this.maxCacheSize) {
      const firstKey = this.blockCache.keys().next().value;
      this.blockCache.delete(firstKey);
    }
    this.blockCache.set(key, data);
  }

  async close() {
    this.blockCache.clear();
  }
}

// Memory reader for in-memory matrices (fallback)
class MemoryMatrixReader {
  constructor(buffer, matrixSize) {
    this.matrices = this.parseMatrixBuffer(buffer, matrixSize);
    this.matrixSize = matrixSize;
  }

  async getBlock(matrixName, blockRow, blockCol, blockSize) {
    const matrix = this.matrices[matrixName];
    return this.extractBlock(matrix, blockRow, blockCol, blockSize);
  }

  extractBlock(matrix, blockRow, blockCol, blockSize) {
    const block = new Float32Array(blockSize * blockSize);
    const startRow = blockRow * blockSize;
    const startCol = blockCol * blockSize;

    for (let i = 0; i < blockSize; i++) {
      for (let j = 0; j < blockSize; j++) {
        const matrixIdx = (startRow + i) * this.matrixSize + (startCol + j);
        const blockIdx = i * blockSize + j;
        block[blockIdx] = matrix[matrixIdx];
      }
    }

    return block;
  }

  parseMatrixBuffer(buffer, matrixSize) {
    const headerSize = buffer.readUInt32LE(0);
    if (headerSize !== matrixSize) {
      throw new Error(`Matrix size mismatch: header says ${headerSize}, expected ${matrixSize}`);
    }

    const matrixA = new Float32Array(buffer.buffer, buffer.byteOffset + 4, matrixSize * matrixSize);
    const matrixB = new Float32Array(buffer.buffer, buffer.byteOffset + 4 + matrixSize * matrixSize * 4, matrixSize * matrixSize);

    return { A: matrixA, B: matrixB };
  }

  async close() {
    // Nothing to cleanup for memory reader
  }
}