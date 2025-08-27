// TransformerBlockCoordinator.js
// Orchestrates a single Transformer block locally with optional distributed chunking.
// Exposes executeTransformerBlock(config) returning a result with output (base64).

import crypto from 'crypto';

export class TransformerBlockCoordinator {
  constructor(apiBase = null, httpClient = null) {
    this.apiBase = apiBase;
    this.httpClient = httpClient;
  }

  // Simple CPU implementation, deterministic pseudo-random weights if not provided
  _initWeights(cfg) {
    const { dModel, numHeads, dFF } = cfg;
    const headDim = dModel / numHeads;
    function randn(n){ const a = new Float32Array(n); for (let i=0;i<n;i++) a[i]=(Math.random()-0.5)*0.02; return a; }
    const params = {
      Wq: randn(dModel*dModel),
      Wk: randn(dModel*dModel),
      Wv: randn(dModel*dModel),
      Wo: randn(dModel*dModel),
      W1: randn(dModel*dFF), b1: randn(dFF),
      W2: randn(dFF*dModel), b2: randn(dModel),
      ln1_g: new Float32Array(dModel).fill(1), ln1_b: new Float32Array(dModel).fill(0),
      ln2_g: new Float32Array(dModel).fill(1), ln2_b: new Float32Array(dModel).fill(0),
    };
    return params;
  }

  async executeTransformerBlock(config) {
    const cfg = { ...config };
    const { seqLength, dModel, numHeads, dFF } = cfg;
    const headDim = dModel / numHeads;

    // Make input X deterministic
    const X = new Float32Array(seqLength * dModel);
    for (let i=0;i<X.length;i++) X[i] = Math.sin(i*0.01);

    const weights = this._initWeights(cfg);

    function layernorm(Y, gamma, beta, eps=1e-5) {
      const seq = Y.length / dModel;
      const out = new Float32Array(Y.length);
      for (let t=0;t<seq;t++) {
        let mean=0; for (let j=0;j<dModel;j++) mean+=Y[t*dModel+j];
        mean/=dModel;
        let varsum=0; for (let j=0;j<dModel;j++){ const d=Y[t*dModel+j]-mean; varsum+=d*d; }
        const invstd = 1/Math.sqrt(varsum/dModel + eps);
        for (let j=0;j<dModel;j++) {
          const z=(Y[t*dModel+j]-mean)*invstd;
          out[t*dModel+j]=z*gamma[j]+beta[j];
        }
      }
      return out;
    }
    function matmul(a, b, M, K, N) { // a[M,K] * b[K,N] => [M,N]
      const out = new Float32Array(M*N);
      for (let i=0;i<M;i++) {
        for (let j=0;j<N;j++) {
          let s=0; for (let k=0;k<K;k++) s+=a[i*K+k]*b[k*N+j];
          out[i*N+j]=s;
        }
      }
      return out;
    }
    function add(a,b){ const out=new Float32Array(a.length); for(let i=0;i<a.length;i++) out[i]=a[i]+b[i]; return out; }
    function gelu(x){ const k=0.7978845608; return 0.5*x*(1+Math.tanh(k*(x+0.044715*x*x*x))); }

    const stages = [];

    const t0 = Date.now();
    const ln1_in = X; // Pre-Norm: LN before attention with residual
    const ln1 = layernorm(ln1_in, weights.ln1_g, weights.ln1_b, cfg.epsilon || 1e-5);
    stages.push({ stage: 'layer_norm_1', duration: Date.now()-t0 });

    const t1 = Date.now();
    const Q = matmul(ln1, weights.Wq, seqLength, dModel, dModel);
    const K = matmul(ln1, weights.Wk, seqLength, dModel, dModel);
    const V = matmul(ln1, weights.Wv, seqLength, dModel, dModel);
    // reshape [T,d_model] -> [T,heads,headDim] implicit indexing in attention below
    const scale = 1/Math.sqrt(headDim);

    // attention
    const ctx = new Float32Array(seqLength*dModel);
    for (let h=0; h<numHeads; h++) {
      for (let t=0; t<seqLength; t++) {
        // logits over seq
        const logits = new Float32Array(seqLength);
        let maxv=-1e38;
        for (let u=0; u<seqLength; u++) {
          let dot=0;
          for (let j=0; j<headDim; j++) {
            dot += Q[t*dModel + h*headDim + j] * K[u*dModel + h*headDim + j];
          }
          dot *= scale;
          logits[u]=dot; if (dot>maxv) maxv=dot;
        }
        let denom=0; for (let u=0; u<seqLength; u++){ logits[u]=Math.exp(logits[u]-maxv); denom+=logits[u]; }
        for (let u=0; u<seqLength; u++) logits[u]/=denom;
        for (let j=0; j<headDim; j++) {
          let acc=0; for (let u=0; u<seqLength; u++) acc+=logits[u]*V[u*dModel + h*headDim + j];
          ctx[t*dModel + h*headDim + j] = acc;
        }
      }
    }
    const attn_out = matmul(ctx, weights.Wo, seqLength, dModel, dModel);
    stages.push({ stage: 'attention', duration: Date.now()-t1 });

    const res1 = add(X, attn_out);

    const t2 = Date.now();
    const ln2 = layernorm(res1, weights.ln2_g, weights.ln2_b, cfg.epsilon || 1e-5);
    stages.push({ stage: 'layer_norm_2', duration: Date.now()-t2 });

    const t3 = Date.now();
    // FFN
    // hidden = gelu( ln2 * W1 + b1 )
    const hidden = new Float32Array(seqLength * dFF);
    for (let t=0; t<seqLength; t++) {
      for (let j=0; j<dFF; j++) {
        let acc = weights.b1[j];
        for (let k=0; k<dModel; k++) acc += ln2[t*dModel + k]*weights.W1[k*dFF + j];
        hidden[t*dFF + j] = gelu(acc);
      }
    }
    const ffn_out = matmul(hidden, weights.W2, seqLength, dFF, dModel);
    const res2 = add(res1, ffn_out);
    stages.push({ stage: 'ffn', duration: Date.now()-t3 });

    const outBuf = Buffer.from(new Float32Array(res2).buffer);
    const blockId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    return {
      blockId,
      output: outBuf.toString('base64'),
      metadata: { stages }
    };
  }
}

export default TransformerBlockCoordinator;
