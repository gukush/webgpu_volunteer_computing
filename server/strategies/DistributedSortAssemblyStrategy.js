// DistributedSortAssemblyStrategy.js - Assembly strategy for distributed sorting
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class DistributedSortAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('distributed_sort_assembly');
  }

  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'sorted_data',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  /**
   * For iterative workloads, this isn't called - the chunking strategy handles assembly
   * via updateGlobalState and assembleFinalResult. This method is here for compatibility.
   */
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
      // Since this is an iterative strategy, we should have a single final result
      if (completedChunks.length !== 1) {
        throw new Error(`Expected single final sorted result, got ${completedChunks.length} chunks`);
      }

      const finalChunk = completedChunks[0];
      const sortedData = finalChunk.results ? finalChunk.results[0] : finalChunk.result;

      return {
        success: true,
        outputs: { sorted_data: sortedData },
        data: sortedData,
        metadata: {
          ...this.createAssemblyMetadata(plan, completedChunks),
          algorithm: 'distributed_merge_tree',
          sortedElements: this.estimateElementCount(sortedData),
          totalPhases: plan.totalPhases
        }
      };

    } catch (error) {
      return {
        success: false,
        error: `Distributed sort assembly failed: ${error.message}`
      };
    }
  }

  /**
   * Validate chunks for distributed sorting
   */
  validateChunks(completedChunks, plan) {
    const baseValidation = super.validateChunks(completedChunks, plan);
    if (!baseValidation.valid) return baseValidation;

    // For iterative workloads, we expect phases to complete sequentially
    // The final assembly should only have one chunk (the completely sorted result)
    
    if (plan.executionModel === 'iterative_refinement') {
      // During iterative execution, individual phases may have multiple chunks
      // The final result should be a single sorted array
      if (completedChunks.length > 1) {
        // Check if this is an intermediate phase or the final result
        const hasPhaseMetadata = completedChunks.some(chunk => 
          chunk.assemblyMetadata && chunk.assemblyMetadata.phaseType
        );
        
        if (!hasPhaseMetadata) {
          return {
            valid: false,
            error: `Final distributed sort result should be a single chunk, got ${completedChunks.length}`
          };
        }
      }
    }

    return { valid: true };
  }

  /**
   * Estimate element count from base64 data
   */
  estimateElementCount(base64Data) {
    if (!base64Data) return 0;
    
    try {
      const byteLength = (base64Data.length * 3) / 4;
      return Math.floor(byteLength / 4); // Assuming 4-byte floats
    } catch (e) {
      return 0;
    }
  }

  /**
   * Create enhanced metadata for distributed sort assembly
   */
  createAssemblyMetadata(plan, chunks) {
    const baseMetadata = super.createAssemblyMetadata(plan, chunks);
    
    return {
      ...baseMetadata,
      sortingAlgorithm: 'distributed_merge_tree',
      localSortMethod: 'gpu_radix_sort',
      mergeMethod: 'parallel_gpu_merge',
      distributedPhases: plan.totalPhases,
      originalDataSize: plan.metadata?.dataSize,
      chunkSize: plan.metadata?.chunkSize,
      totalComputeTime: this.calculateTotalComputeTime(chunks)
    };
  }

  /**
   * Calculate total processing time across all chunks
   */
  calculateTotalComputeTime(chunks) {
    return chunks.reduce((total, chunk) => {
      return total + (chunk.processingTime || 0);
    }, 0);
  }

  /**
   * Sort chunks by their assembly order (for multi-phase results)
   */
  sortChunks(chunks) {
    return chunks.sort((a, b) => {
      // For distributed sort, chunks should be ordered by phase and then by chunk index
      const aPhase = a.assemblyMetadata?.mergeLevel || 0;
      const bPhase = b.assemblyMetadata?.mergeLevel || 0;
      
      if (aPhase !== bPhase) {
        return aPhase - bPhase;
      }
      
      // Within same phase, sort by chunk index
      const aIndex = a.chunkIndex !== undefined ? a.chunkIndex : this.extractChunkIndex(a.chunkId);
      const bIndex = b.chunkIndex !== undefined ? b.chunkIndex : this.extractChunkIndex(b.chunkId);
      return aIndex - bIndex;
    });
  }

  /**
   * Verify sorting correctness (optional validation)
   */
  verifySortedResult(sortedData) {
    try {
      const buffer = Buffer.from(sortedData, 'base64');
      const floatArray = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
      
      // Check if array is sorted
      for (let i = 1; i < floatArray.length; i++) {
        if (floatArray[i] < floatArray[i - 1]) {
          return {
            valid: false,
            error: `Sort verification failed at index ${i}: ${floatArray[i]} < ${floatArray[i - 1]}`
          };
        }
      }
      
      return {
        valid: true,
        elementCount: floatArray.length,
        minValue: floatArray[0],
        maxValue: floatArray[floatArray.length - 1]
      };
      
    } catch (error) {
      return {
        valid: false,
        error: `Sort verification error: ${error.message}`
      };
    }
  }
}
