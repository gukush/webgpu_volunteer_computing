// strategies/base/BaseAssemblyStrategy.js
// Base class for all assembly strategies with multi-output support

export class BaseAssemblyStrategy {
  constructor(name) {
    this.name = name;
  }

  /**
   * Assemble results from completed chunks
   * @param {Array} completedChunks - Array of chunk results with metadata
   * @param {Object} plan - The original execution plan
   * @returns {Object} - { success: boolean, outputs: Object, metadata: Object, error?: string }
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
      const schema = plan.schema || this.getDefaultSchema();
      const sortedChunks = this.sortChunks(completedChunks);

      // NEW: Multi-output assembly
      if (schema.outputs.length > 1) {
        return this.assembleMultipleOutputs(sortedChunks, plan, schema);
      } else {
        return this.assembleSingleOutput(sortedChunks, plan, schema);
      }
    } catch (error) {
      return {
        success: false,
        error: `Assembly failed: ${error.message}`
      };
    }
  }

  /**
   * Assemble multiple outputs from chunks
   * @param {Array} sortedChunks - Chunks sorted by index
   * @param {Object} plan - Original execution plan
   * @param {Object} schema - Output schema
   * @returns {Object} - Assembly result with named outputs
   */
  assembleMultipleOutputs(sortedChunks, plan, schema) {
    const outputs = {};

    // Transpose results: group by output index
    const outputsByIdx = Array(schema.outputs.length).fill().map(() => []);

    for (const chunk of sortedChunks) {
      // Each chunk should have results array
      const chunkResults = chunk.results || [chunk.result]; // Backward compatibility

      if (!Array.isArray(chunkResults)) {
        throw new Error(`Chunk ${chunk.chunkId} results must be an array for multi-output assembly`);
      }

      if (chunkResults.length !== schema.outputs.length) {
        throw new Error(`Chunk ${chunk.chunkId} has ${chunkResults.length} results, expected ${schema.outputs.length}`);
      }

      // Add each output to its respective group
      chunkResults.forEach((result, outputIdx) => {
        if (outputIdx < outputsByIdx.length) {
          outputsByIdx[outputIdx].push(Buffer.from(result, 'base64'));
        }
      });
    }

    // Assemble each output separately
    for (let i = 0; i < schema.outputs.length; i++) {
      const outputDef = schema.outputs[i];
      const outputBuffers = outputsByIdx[i];
      const assembledOutput = this.assembleSingleOutputBuffers(outputBuffers, plan, outputDef);
      outputs[outputDef.name] = assembledOutput.toString('base64');
    }

    return {
      success: true,
      outputs: outputs,
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };
  }

  /**
   * Assemble single output from chunks (backward compatibility)
   * @param {Array} sortedChunks - Chunks sorted by index
   * @param {Object} plan - Original execution plan
   * @param {Object} schema - Output schema
   * @returns {Object} - Assembly result
   */
  assembleSingleOutput(sortedChunks, plan, schema) {
    const outputDef = schema.outputs[0];
    const outputBuffers = sortedChunks.map(chunk => {
      // Handle both single result and array format
      const result = chunk.results ? chunk.results[0] : chunk.result;
      return this.decodeResult(result);
    });

    const assembledOutput = this.assembleSingleOutputBuffers(outputBuffers, plan, outputDef);
    const outputName = outputDef.name;

    return {
      success: true,
      outputs: { [outputName]: assembledOutput.toString('base64') },
      data: assembledOutput.toString('base64'), // Backward compatibility
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };
  }

  /**
   * Assemble buffers for a single output
   * @param {Array} buffers - Array of Buffer objects
   * @param {Object} plan - Original execution plan
   * @param {Object} outputDef - Output definition from schema
   * @returns {Buffer} - Assembled buffer
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    // Default: concatenate buffers
    return Buffer.concat(buffers);
  }

  /**
   * Validate that all required chunks are present
   * @param {Array} completedChunks - Array of chunk results
   * @param {Object} plan - The original execution plan
   * @returns {Object} - { valid: boolean, missing?: Array, error?: string }
   */
  validateChunks(completedChunks, plan) {
    const expectedChunks = plan.totalChunks;
    const receivedChunks = completedChunks.length;

    if (receivedChunks !== expectedChunks) {
      const received = completedChunks.map(c => c.chunkIndex || c.chunkId);
      const missing = [];

      for (let i = 0; i < expectedChunks; i++) {
        if (!received.includes(i)) {
          missing.push(i);
        }
      }

      return {
        valid: false,
        missing,
        error: `Expected ${expectedChunks} chunks, got ${receivedChunks}`
      };
    }

    // NEW: Validate multi-output chunks
    const schema = plan.schema || this.getDefaultSchema();
    if (schema.outputs.length > 1) {
      for (const chunk of completedChunks) {
        const chunkResults = chunk.results || (chunk.result ? [chunk.result] : []);
        if (chunkResults.length !== schema.outputs.length) {
          return {
            valid: false,
            error: `Chunk ${chunk.chunkId || chunk.chunkIndex} has ${chunkResults.length} results, expected ${schema.outputs.length}`
          };
        }
      }
    }

    return { valid: true };
  }

  /**
   * Sort chunks by their intended order
   * @param {Array} chunks - Array of chunk results
   * @returns {Array} - Sorted array of chunks
   */
  sortChunks(chunks) {
    return chunks.sort((a, b) => {
      // Try chunkIndex first, fall back to parsing chunkId
      const indexA = a.chunkIndex !== undefined ? a.chunkIndex : this.extractChunkIndex(a.chunkId);
      const indexB = b.chunkIndex !== undefined ? b.chunkIndex : this.extractChunkIndex(b.chunkId);
      return indexA - indexB;
    });
  }

  /**
   * Extract chunk index from chunk ID
   * @param {string} chunkId - Chunk identifier
   * @returns {number} - Extracted index
   */
  extractChunkIndex(chunkId) {
    // Handle formats like "chunk-5", "tile-2-3", etc.
    const matches = chunkId.match(/(\d+)$/);
    return matches ? parseInt(matches[1], 10) : 0;
  }

  /**
   * Convert base64 result to buffer
   * @param {string} base64Result - Base64 encoded result
   * @returns {Buffer} - Decoded buffer
   */
  decodeResult(base64Result) {
    try {
      return Buffer.from(base64Result, 'base64');
    } catch (e) {
      throw new Error(`Invalid base64 result data: ${e.message}`);
    }
  }

  /**
   * Concatenate results in order (default assembly strategy)
   * @param {Array} sortedChunks - Chunks sorted by index
   * @returns {Buffer} - Concatenated result
   */
  concatenateResults(sortedChunks) {
    const buffers = sortedChunks.map(chunk => {
      const result = chunk.results ? chunk.results[0] : chunk.result;
      return this.decodeResult(result);
    });
    return Buffer.concat(buffers);
  }

  /**
   * Get default schema when none is provided
   * @returns {Object} - Default schema
   */
  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'output',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
  }

  /**
   * Create assembly metadata
   * @param {Object} plan - Original execution plan
   * @param {Array} chunks - Processed chunks
   * @returns {Object} - Assembly metadata
   */
  createAssemblyMetadata(plan, chunks) {
    return {
      assemblyStrategy: this.name,
      totalChunks: chunks.length,
      assembledAt: Date.now(),
      originalPlan: {
        strategy: plan.strategy,
        totalChunks: plan.totalChunks
      },
      outputCount: plan.schema ? plan.schema.outputs.length : 1
    };
  }
}