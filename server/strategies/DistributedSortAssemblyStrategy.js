// strategies/DistributedSortAssemblyStrategy.js
// Assembly strategy for a simplified, single-phase distributed sort.
// This strategy receives multiple sorted chunks and merges them on the server.

import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class DistributedSortAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('distributed_sort_assembly');
  }

  /**
   * Defines the final, single output schema for the sorted data.
   */
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
   * Overrides the default buffer assembly (which is just concatenation).
   * This method performs the final merge-sort of all sorted chunks from clients.
   * @param {Array<Buffer>} buffers - An array of buffers, each containing a sorted Float32 array.
   * @param {Object} plan - The original execution plan.
   * @param {Object} outputDef - The output definition from the schema.
   * @returns {Buffer} A single buffer containing the fully sorted data.
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    console.log(`[Sort Assembly] Starting final merge of ${buffers.length} sorted chunks.`);
    const startTime = Date.now();

    if (buffers.length === 0) {
      return Buffer.alloc(0);
    }

    // 1. Convert all chunk buffers into a single large Float32Array.
    // This is memory-intensive and assumes the final sorted array can fit in the server's memory.
    const totalElements = buffers.reduce((sum, buf) => sum + buf.length, 0) / 4;
    const combinedArray = new Float32Array(totalElements);

    let offset = 0;
    for (const buffer of buffers) {
      // Create a Float32Array view of the buffer without copying its underlying ArrayBuffer
      const chunkArray = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
      combinedArray.set(chunkArray, offset);
      offset += chunkArray.length;
    }

    // 2. Sort the combined array.
    // The built-in sort is highly optimized for numeric types.
    combinedArray.sort();

    // 3. Convert the final sorted Float32Array back to a Buffer.
    const finalBuffer = Buffer.from(combinedArray.buffer);

    const duration = Date.now() - startTime;
    console.log(`[Sort Assembly] Final merge completed in ${duration}ms.`);

    return finalBuffer;
  }
}