// strategies/base/BaseAssemblyStrategy.js
// Base class for all assembly strategies

export class BaseAssemblyStrategy {
  constructor(name) {
    this.name = name;
  }

  /**
   * Assemble results from completed chunks
   * @param {Array} completedChunks - Array of chunk results with metadata
   * @param {Object} plan - The original execution plan
   * @returns {Object} - { success: boolean, data: any, metadata: Object, error?: string }
   */
  assembleResults(completedChunks, plan) {
    throw new Error(`${this.name}: assembleResults must be implemented by subclass`);
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
    const buffers = sortedChunks.map(chunk => this.decodeResult(chunk.result));
    return Buffer.concat(buffers);
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
      }
    };
  }
}