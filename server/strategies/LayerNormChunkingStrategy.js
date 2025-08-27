// strategies/LayerNormChunkingStrategy.js
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';

export default class LayerNormChunkingStrategy extends BaseChunkingStrategy {
  constructor() { super('layer_normalization'); }

  defineInputSchema() {
    // X, Residual (optional), gamma, beta
    return {
      inputs: [
        { name: 'X', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'Residual', type: 'storage_buffer', binding: 2, elementType: 'f32', optional: true },
        { name: 'gamma', type: 'storage_buffer', binding: 3, elementType: 'f32' },
        { name: 'beta', type: 'storage_buffer', binding: 4, elementType: 'f32' },
      ],
      outputs: [ { name: 'Y', type: 'storage_buffer', binding: 5, elementType: 'f32' } ]
    };
  }

  planExecution(workload) {
    const { seqLength, dModel, framework = 'webgpu', chunking = { seqChunk: 256 }, epsilon = 1e-5 } = workload.metadata || {};
    const schema = this.defineInputSchema();
    this.parseMultipleInputs(workload.input, schema);
    const totalChunks = Math.ceil(seqLength / (chunking.seqChunk || 256));
    return {
      strategy: this.name,
      totalChunks,
      schema,
      metadata: { seqLength, dModel, framework, chunking, epsilon },
      assemblyStrategy: 'layer_norm_assembly',
      shaderTemplate: 'layer_norm'
    };
  }

  createChunkDescriptors(plan) {
    const { seqLength, dModel, framework, chunking, epsilon } = plan.metadata;
    const seqChunk = chunking.seqChunk || 256;
    const descriptors = [];
    const chunks = Math.ceil(seqLength / seqChunk);
    for (let i = 0; i < chunks; i++) {
      const seqStart = i * seqChunk;
      const seqLen = Math.min(seqChunk, seqLength - seqStart);
      descriptors.push({
        chunkId: `ln-${i}`,
        chunkIndex: i,
        parentId: plan.parentId,
        framework,
        kernel: framework === 'webgpu' ? this.getWebGPUShader() : this.getJavaScriptKernel(),
        entry: 'main',
        workgroupCount: [Math.ceil(seqLen / 8), Math.ceil(dModel / 8), 1],
        inputs: [],
        outputSizes: [seqLen * dModel * 4],
        uniforms: { seq_length: seqLength, d_model: dModel, seq_start: seqStart, seq_len: seqLen, epsilon },
        assemblyMetadata: { seqStart, seqLen }
      });
    }
    return descriptors;
  }

  getJavaScriptKernel() {
    return `export function run(payload) {
      const { uniforms, inputs } = payload;
      const X = new Float32Array(inputs[0]);
      const Residual = inputs[1] ? new Float32Array(inputs[1]) : null;
      const gamma = new Float32Array(inputs[2]);
      const beta = new Float32Array(inputs[3]);

      const seq = uniforms.seq_length >>> 0;
      const dmodel = uniforms.d_model >>> 0;
      const s0 = uniforms.seq_start >>> 0;
      const sl = uniforms.seq_len >>> 0;
      const eps = Number(uniforms.epsilon);

      const out = new Float32Array(sl * dmodel);

      for (let t = 0; t < sl; t++) {
        // compute mean/var over d_model for row X[t+s0]+Residual[t+s0] if present
        let mean = 0.0;
        for (let j = 0; j < dmodel; j++) {
          const v = X[(t+s0)*dmodel + j] + (Residual ? Residual[(t+s0)*dmodel + j] : 0.0);
          mean += v;
        }
        mean /= dmodel;
        let varsum = 0.0;
        for (let j = 0; j < dmodel; j++) {
          const v = X[(t+s0)*dmodel + j] + (Residual ? Residual[(t+s0)*dmodel + j] : 0.0);
          const d = v - mean;
          varsum += d*d;
        }
        const invstd = 1.0 / Math.sqrt(varsum / dmodel + eps);
        for (let j = 0; j < dmodel; j++) {
          const v = X[(t+s0)*dmodel + j] + (Residual ? Residual[(t+s0)*dmodel + j] : 0.0);
          const z = (v - mean) * invstd;
          out[t*dmodel + j] = z * gamma[j] + beta[j];
        }
      }
      return [out.buffer];
    }`;
  }

  getWebGPUShader() {
    return `
      struct Params { seq_length: u32, d_model: u32, seq_start: u32, seq_len: u32, epsilon: f32 };
      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read> X: array<f32>;
      @group(0) @binding(2) var<storage, read> RES: array<f32>;
      @group(0) @binding(3) var<storage, read> GAMMA: array<f32>;
      @group(0) @binding(4) var<storage, read> BETA: array<f32>;
      @group(0) @binding(5) var<storage, read_write> OUT: array<f32>;

      @compute @workgroup_size(8,8,1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let t = gid.x;
        let j = gid.y;
        if (t >= params.seq_len || j >= params.d_model) { return; }

        let dmodel = params.d_model;
        let seq_idx = params.seq_start + t;

        // compute mean/var per row (serially per invocation's j==0 is typical, but here a simple version)
        var mean: f32 = 0.0;
        for (var m: u32 = 0u; m < dmodel; m = m + 1u) {
          mean = mean + (X[seq_idx*dmodel + m] + RES[seq_idx*dmodel + m]);
        }
        mean = mean / f32(dmodel);
        var varsum: f32 = 0.0;
        for (var m: u32 = 0u; m < dmodel; m = m + 1u) {
          let v = X[seq_idx*dmodel + m] + RES[seq_idx*dmodel + m];
          let d = v - mean;
          varsum = varsum + d*d;
        }
        let invstd = inverseSqrt(varsum / f32(dmodel) + params.epsilon);
        let v = X[seq_idx*dmodel + j] + RES[seq_idx*dmodel + j];
        let z = (v - mean) * invstd;
        OUT[t*dmodel + j] = z * GAMMA[j] + BETA[j];
      }`;
  }
}
