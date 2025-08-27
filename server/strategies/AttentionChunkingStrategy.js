// strategies/AttentionChunkingStrategy.js
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';

export default class AttentionChunkingStrategy extends BaseChunkingStrategy {
  constructor() { super('multi_head_attention'); }

  defineInputSchema() {
    // Expect pre-projected Q, K, V and optional mask; shapes are flattened 1D buffers.
    return {
      inputs: [
        { name: 'Q', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'K', type: 'storage_buffer', binding: 2, elementType: 'f32' },
        { name: 'V', type: 'storage_buffer', binding: 3, elementType: 'f32' },
      ],
      outputs: [
        { name: 'context', type: 'storage_buffer', binding: 4, elementType: 'f32' }
      ]
    };
  }

  planExecution(workload) {
    const { seqLength, dModel, numHeads, framework = 'webgpu',
            chunking = { mode: 'heads', seqChunk: 64 } } = workload.metadata || {};
    const headDim = Math.floor(dModel / numHeads);
    const schema = this.defineInputSchema();
    // Parse to ensure inputs exist
    this.parseMultipleInputs(workload.input, schema);

    let totalChunks = 0;
    let mode = chunking.mode || 'heads';
    if (mode === 'heads') totalChunks = numHeads;
    else if (mode === 'sequence') totalChunks = Math.ceil(seqLength / (chunking.seqChunk || 64));
    else totalChunks = numHeads * Math.ceil(seqLength / (chunking.seqChunk || 64));

    return {
      strategy: this.name,
      totalChunks,
      schema,
      metadata: {
        seqLength, dModel, numHeads, headDim, framework, chunking
      },
      assemblyStrategy: 'multi_head_attention_assembly',
      shaderTemplate: 'attention_spda' // not used; JavaScript fallback or client-provided shaders
    };
  }

  createChunkDescriptors(plan) {
    const { seqLength, dModel, numHeads, headDim, framework, chunking } = plan.metadata;
    const mode = chunking.mode || 'heads';
    const descriptors = [];
    const schema = plan.schema;
    const inputs = this.parseMultipleInputs(plan.input || plan.metadata.inputData || {}, schema);

    if (mode === 'heads' || mode === 'hybrid') {
      const seqChunks = mode === 'hybrid' ? Math.ceil(seqLength / (chunking.seqChunk || 64)) : 1;
      for (let h = 0; h < numHeads; h++) {
        for (let s = 0; s < seqChunks; s++) {
          const seqStart = s * (chunking.seqChunk || 64);
          const seqLen = Math.min((chunking.seqChunk || 64), seqLength - seqStart);
          descriptors.push(this._mkDescriptor(plan, descriptors.length, { head: h, seqStart, seqLen }, framework));
        }
      }
    } else {
      const seqChunks = Math.ceil(seqLength / (chunking.seqChunk || 64));
      for (let s = 0; s < seqChunks; s++) {
        const seqStart = s * (chunking.seqChunk || 64);
        const seqLen = Math.min((chunking.seqChunk || 64), seqLength - seqStart);
        descriptors.push(this._mkDescriptor(plan, descriptors.length, { head: -1, seqStart, seqLen }, framework));
      }
    }
    return descriptors;
  }

  _mkDescriptor(plan, idx, { head, seqStart, seqLen }, framework) {
    const uniforms = {
      seq_length: plan.metadata.seqLength,
      d_model: plan.metadata.dModel,
      num_heads: plan.metadata.numHeads,
      head_dim: plan.metadata.headDim,
      head_index: head,
      seq_start: seqStart,
      seq_len: seqLen
    };
    return {
      chunkId: `attn-${idx}`,
      chunkIndex: idx,
      parentId: plan.parentId,
      framework,
      kernel: framework === 'webgpu' ? this.getWebGPUShader() : this.getJavaScriptKernel(),
      entry: 'main',
      workgroupCount: [Math.ceil(seqLen / 8), Math.ceil(plan.metadata.headDim / 8), 1],
      inputs: [], // manager will bind Q,K,V based on schema
      outputSizes: [seqLen * plan.metadata.dModel * 4],
      uniforms,
      assemblyMetadata: { head, seqStart, seqLen }
    };
  }

  getJavaScriptKernel() {
    // Naive CPU SPDA for a (seqLen x headDim) per head; expects packed Q,K,V for entire layer.
    return `export function run(payload) {
      const { uniforms, inputs } = payload;
      const Q = new Float32Array(inputs[0]);
      const K = new Float32Array(inputs[1]);
      const V = new Float32Array(inputs[2]);
      const seq = uniforms.seq_length >>> 0;
      const dmodel = uniforms.d_model >>> 0;
      const heads = uniforms.num_heads >>> 0;
      const hdim = uniforms.head_dim >>> 0;
      const h = uniforms.head_index >>> 0;
      const s0 = uniforms.seq_start >>> 0;
      const sl = uniforms.seq_len >>> 0;

      const out = new Float32Array(sl * dmodel);
      const scale = 1.0 / Math.sqrt(hdim);

      function idxQ(t, hh, j){ return (t*heads + hh)*hdim + j; }
      function idxK(t, hh, j){ return (t*heads + hh)*hdim + j; }
      function idxV(t, hh, j){ return (t*heads + hh)*hdim + j; }

      for (let t = 0; t < sl; t++) {
        // logits over all T for this token t+s0
        const logits = new Float32Array(seq);
        let maxv = -1e38;
        for (let u = 0; u < seq; u++) {
          let dot = 0.0;
          for (let j = 0; j < hdim; j++) {
            dot += Q[idxQ(t+s0, h, j)] * K[idxK(u, h, j)];
          }
          dot *= scale;
          logits[u] = dot;
          if (dot > maxv) maxv = dot;
        }
        // softmax
        let denom = 0.0;
        for (let u = 0; u < seq; u++) { logits[u] = Math.exp(logits[u] - maxv); denom += logits[u]; }
        for (let u = 0; u < seq; u++) logits[u] /= denom;
        // weighted sum of V across all heads then place into correct slice
        for (let j = 0; j < hdim; j++) {
          let acc = 0.0;
          for (let u = 0; u < seq; u++) acc += logits[u] * V[idxV(u, h, j)];
          // write into head slice
          out[t*dmodel + h*hdim + j] = acc;
        }
      }
      return [out.buffer];
    }`;
  }

  getWebGPUShader() {
    // Lightweight kernel that just copies the JavaScript-calculated slice if present;
    // real GPU implementation can be swapped later. Keep signature valid.
    return `
      struct Params {
        seq_length: u32,
        d_model: u32,
        num_heads: u32,
        head_dim: u32,
        head_index: i32,
        seq_start: u32,
        seq_len: u32,
      };
      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read> Q: array<f32>;
      @group(0) @binding(2) var<storage, read> K: array<f32>;
      @group(0) @binding(3) var<storage, read> V: array<f32>;
      @group(0) @binding(4) var<storage, read_write> OUT: array<f32>;

      @compute @workgroup_size(8,8,1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let t = gid.x; // local token within chunk
        let j = gid.y; // feature index within d_model
        if (t >= params.seq_len || j >= params.d_model) { return; }
        // Placeholder: zero output (CPU fallback fills real values via JS kernel on CPU clients)
        OUT[t*params.d_model + j] = 0.0;
      }`;
  }
}
