// strategies/FeedForwardAssemblyStrategy.js
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

export default class FeedForwardAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() { super('feed_forward_assembly'); }

  getDefaultSchema() {
    return { outputs: [ { name: 'Y', type: 'storage_buffer', elementType: 'f32' } ] };
  }

  assembleResults(completedChunks, plan) {
    const sorted = this.sortChunks(completedChunks);
    const buffers = sorted.map(c => Buffer.from(c.output || c.outputs?.[0] || '', 'base64'));
    return Buffer.concat(buffers);
  }
}
