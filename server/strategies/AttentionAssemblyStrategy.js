// strategies/AttentionAssemblyStrategy.js
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class AttentionAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() { super('multi_head_attention_assembly'); }

  getDefaultSchema() {
    return { outputs: [ { name: 'context', type: 'storage_buffer', elementType: 'f32' } ] };
  }

  initOutputStore(plan) {
    const { seqLength, dModel } = plan.metadata || {};
    const bytes = (seqLength || 0) * (dModel || 0) * 4;
    this.outputStore = Buffer.alloc(bytes);
    this.map = new Map();
  }

  processChunkResult(chunkResult, plan) {
    if (!this.outputStore) this.initOutputStore(plan);
    const { seqStart = 0, seqLen = 0 } = chunkResult.assemblyMetadata || {};
    const out = Buffer.from(chunkResult.output || chunkResult.outputs?.[0] || '', 'base64');
    const rowBytes = plan.metadata.dModel * 4;
    for (let t = 0; t < seqLen; t++) {
      const srcStart = t * rowBytes;
      const dstStart = (seqStart + t) * rowBytes;
      out.copy(this.outputStore, dstStart, srcStart, srcStart + rowBytes);
    }
    return null; // streaming aggregation; final object returned in assembleResults
  }

  assembleResults(chunks, plan) {
    if (!this.outputStore) this.initOutputStore(plan);
    // In case of batch mode (no processChunkResult), concatenate by seqStart
    const sorted = this.sortChunks(chunks);
    if (sorted.length && !this.outputStore.length) {
      const bufs = sorted.map(c => Buffer.from(c.output || c.outputs?.[0] || '', 'base64'));
      this.outputStore = Buffer.concat(bufs);
    }
    return this.outputStore;
  }

  async cleanup() { this.outputStore = null; }
}
