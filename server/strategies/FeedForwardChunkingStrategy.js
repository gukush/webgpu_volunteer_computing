// strategies/FeedForwardChunkingStrategy.js
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';

export default class FeedForwardChunkingStrategy extends BaseChunkingStrategy {
  constructor() { super('feed_forward_network'); }

  defineInputSchema() {
    // Input X (seq x d_model), weights W1 (d_model x d_ff), b1 (d_ff),
    // W2 (d_ff x d_model), b2 (d_model)
    return {
      inputs: [
        { name: 'X', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'W1', type: 'storage_buffer', binding: 2, elementType: 'f32' },
        { name: 'b1', type: 'storage_buffer', binding: 3, elementType: 'f32' },
        { name: 'W2', type: 'storage_buffer', binding: 4, elementType: 'f32' },
        { name: 'b2', type: 'storage_buffer', binding: 5, elementType: 'f32' },
      ],
      outputs: [ { name: 'Y', type: 'storage_buffer', binding: 6, elementType: 'f32' } ]
    };
  }

  planExecution(workload) {
    const { seqLength, dModel, dFF, framework = 'webgpu', chunking = { seqChunk: 128 } } = workload.metadata || {};
    const schema = this.defineInputSchema();
    this.parseMultipleInputs(workload.input, schema);
    const totalChunks = Math.ceil(seqLength / (chunking.seqChunk || 128));
    return {
      strategy: this.name,
      totalChunks,
      schema,
      metadata: { seqLength, dModel, dFF, framework, chunking },
      assemblyStrategy: 'feed_forward_assembly',
      shaderTemplate: 'ffn_dense'
    };
  }

  createChunkDescriptors(plan) {
    const { seqLength, dModel, dFF, framework, chunking } = plan.metadata;
    const seqChunk = chunking.seqChunk || 128;
    const descriptors = [];
    const chunks = Math.ceil(seqLength / seqChunk);
    for (let i = 0; i < chunks; i++) {
      const seqStart = i * seqChunk;
      const seqLen = Math.min(seqChunk, seqLength - seqStart);
      descriptors.push({
        chunkId: `ffn-${i}`,
        chunkIndex: i,
        parentId: plan.parentId,
        framework,
        kernel: framework === 'webgpu' ? this.getWebGPUShader() : this.getJavaScriptKernel(),
        entry: 'main',
        workgroupCount: [Math.ceil(seqLen / 8), Math.ceil(dModel / 8), 1],
        inputs: [],
        outputSizes: [seqLen * dModel * 4],
        uniforms: { seq_length: seqLength, d_model: dModel, d_ff: dFF, seq_start: seqStart, seq_len: seqLen },
        assemblyMetadata: { seqStart, seqLen }
      });
    }
    return descriptors;
  }

  getJavaScriptKernel() {
    return `export function run(payload) {
      const { uniforms, inputs } = payload;
      const X = new Float32Array(inputs[0]); // [seq, d_model]
      const W1 = new Float32Array(inputs[1]); // [d_model, d_ff]
      const b1 = new Float32Array(inputs[2]); // [d_ff]
      const W2 = new Float32Array(inputs[3]); // [d_ff, d_model]
      const b2 = new Float32Array(inputs[4]); // [d_model]

      const seq = uniforms.seq_length >>> 0;
      const dmodel = uniforms.d_model >>> 0;
      const dff = uniforms.d_ff >>> 0;
      const s0 = uniforms.seq_start >>> 0;
      const sl = uniforms.seq_len >>> 0;

      const out = new Float32Array(sl * dmodel);

      function gelu(x){ const k = 0.7978845608; return 0.5*x*(1.0 + Math.tanh(k*(x + 0.044715*x*x*x))); }

      // naive dense(X[s0:s0+sl], W1)+b1 -> gelu -> dense + b2
      for (let t = 0; t < sl; t++) {
        // hidden = X[t] * W1 + b1
        const hidden = new Float32Array(dff);
        for (let j = 0; j < dff; j++) {
          let acc = b1[j];
          for (let k = 0; k < dmodel; k++) {
            acc += X[(t+s0)*dmodel + k] * W1[k*dff + j];
          }
          hidden[j] = gelu(acc);
        }
        // out = hidden * W2 + b2
        for (let j = 0; j < dmodel; j++) {
          let acc = b2[j];
          for (let k = 0; k < dff; k++) {
            acc += hidden[k] * W2[k*dmodel + j];
          }
          out[t*dmodel + j] = acc;
        }
      }
      return [out.buffer];
    }`;
  }

  getWebGPUShader() {
    return `
      struct Params { seq_length: u32, d_model: u32, d_ff: u32, seq_start: u32, seq_len: u32 };
      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read> X: array<f32>;
      @group(0) @binding(2) var<storage, read> W1: array<f32>;
      @group(0) @binding(3) var<storage, read> B1: array<f32>;
      @group(0) @binding(4) var<storage, read> W2: array<f32>;
      @group(0) @binding(5) var<storage, read> B2: array<f32>;
      @group(0) @binding(6) var<storage, read_write> OUT: array<f32>;

      fn gelu(x: f32) -> f32 {
        let k = 0.7978845608;
        return 0.5 * x * (1.0 + tanh(k*(x + 0.044715*x*x*x)));
      }

      @compute @workgroup_size(8,8,1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let t = gid.x;
        let j = gid.y;
        if (t >= params.seq_len || j >= params.d_model) { return; }

        let dmodel = params.d_model;
        let dff = params.d_ff;
        let seq_idx = params.seq_start + t;

        // hidden[j] computed on the fly per output j
        var acc: f32 = B2[j];
        for (var k: u32 = 0u; k < dff; k = k + 1u) {
          // (X * W1 + b1)[k]
          var h: f32 = B1[k];
          for (var m: u32 = 0u; m < dmodel; m = m + 1u) {
            h = h + X[seq_idx*dmodel + m] * W1[m*dff + k];
          }
          h = gelu(h);
          acc = acc + h * W2[k*dmodel + j];
        }
        OUT[t*dmodel + j] = acc;
      }`;
  }
}
