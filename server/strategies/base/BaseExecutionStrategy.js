// strategies/base/BaseExecutionStrategy.js
export class BaseExecutionStrategy {
  constructor(name) { this.name = name; }
  // Must return a Portable Jobscript Artifact (PJA) for a given chunk descriptor
  // shape: { schemaVersion, taskType, kernels, script, resourceHints, callbacks? }
  // where `script` is a string: async (rt, pja, ctx) => { ... } that returns {results?:string[]}
  buildPJA(plan, chunkDescriptor) {
    throw new Error('buildPJA not implemented');
  }
}
export default BaseExecutionStrategy;
