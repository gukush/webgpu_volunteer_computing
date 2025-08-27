// strategies/LayerNormChunkingStrategy.js - Fixed to handle file inputs directly
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';

export default class LayerNormChunkingStrategy extends BaseChunkingStrategy {
  constructor() { super('layer_normalization'); }

  defineInputSchema() {
    return {
      inputs: [
        { name: 'input', type: 'storage_buffer', binding: 1, elementType: 'f32' },
        { name: 'residual', type: 'storage_buffer', binding: 2, elementType: 'f32', optional: true },
        { name: 'gamma', type: 'storage_buffer', binding: 3, elementType: 'f32' },
        { name: 'beta', type: 'storage_buffer', binding: 4, elementType: 'f32' },
      ],
      outputs: [ { name: 'Y', type: 'storage_buffer', binding: 5, elementType: 'f32' } ]
    };
  }

  planExecution(workload) {
    const { seqLength, dModel, framework = 'webgpu', chunking = { seqChunk: 256 }, epsilon = 1e-5 } = workload.metadata || {};

    // Don't try to parse inputs here - we'll handle files per chunk
    const totalChunks = Math.ceil(seqLength / (chunking.seqChunk || 256));

    return {
      strategy: this.name,
      totalChunks,
      schema: this.defineInputSchema(),
      metadata: { seqLength, dModel, framework, chunking, epsilon },
      assemblyStrategy: 'layer_norm_assembly',
      shaderTemplate: 'layer_norm'
    };
  }

  async createChunkDescriptors(plan) {
    const { seqLength, dModel, framework, chunking, epsilon } = plan.metadata;
    const seqChunk = chunking.seqChunk || 256;
    const descriptors = [];
    const chunks = Math.ceil(seqLength / seqChunk);

    // Handle file-based inputs (like BlockMatrix strategy)
    const inputFiles = this.organizeInputFiles(plan.inputRefs);
    let fileHandles = null;

    // Handle inline inputs (fallback)
    let inlineInputs = null;
    if (!inputFiles.hasFiles && plan.metadata?.inputData) {
      inlineInputs = this.parseInlineInputs(plan.metadata.inputData);
    }

    try {
      // Open file handles if using files
      if (inputFiles.hasFiles) {
        fileHandles = await this.openInputFiles(inputFiles);
      }

      for (let i = 0; i < chunks; i++) {
        const seqStart = i * seqChunk;
        const seqLen = Math.min(seqChunk, seqLength - seqStart);

        // Read chunk inputs either from files or inline data
        let chunkInputs;
        if (fileHandles) {
          chunkInputs = await this.readChunkInputsFromFiles(fileHandles, seqStart, seqLen, dModel);
        } else if (inlineInputs) {
          chunkInputs = this.extractChunkInputsFromInline(inlineInputs, seqStart, seqLen, dModel);
        } else {
          throw new Error('No input data available - neither files nor inline data found');
        }

        descriptors.push({
          chunkId: `ln-${i}`,
          chunkIndex: i,
          parentId: plan.parentId,
          framework,
          kernel: framework === 'webgpu' ? this.getWebGPUShader() : this.getJavaScriptKernel(),
          entry: 'main',
          workgroupCount: [Math.ceil(seqLen / 8), Math.ceil(dModel / 8), 1],
          inputs: chunkInputs,
          outputs: [{ name: 'Y', binding: 5, size: seqLen * dModel * 4 }],
          outputSizes: [seqLen * dModel * 4],
          uniforms: { binding: 0, order: ['seq_length','d_model','seq_start','seq_len','epsilon'], seq_length: seqLength, d_model: dModel, seq_start: seqStart, seq_len: seqLen, epsilon },
          assemblyMetadata: { seqStart, seqLen }
        });
      }

      return descriptors;

    } finally {
      // Always close file handles
      if (fileHandles) {
        await this.closeInputFiles(fileHandles);
      }
    }
  }

  // NEW: Organize input file references by name
  organizeInputFiles(inputRefs) {
    if (!inputRefs || inputRefs.length === 0) {
      return { hasFiles: false };
    }

    const files = {};
    for (const ref of inputRefs) {
      files[ref.name] = ref;
    }

    return {
      hasFiles: true,
      input: files.input,
      residual: files.residual,
      gamma: files.gamma,
      beta: files.beta
    };
  }

  // NEW: Open file handles for reading
  async openInputFiles(inputFiles) {
    const handles = {};

    if (inputFiles.input) {
      handles.input = await fs.open(inputFiles.input.path, 'r');
    }
    if (inputFiles.residual) {
      handles.residual = await fs.open(inputFiles.residual.path, 'r');
    }
    if (inputFiles.gamma) {
      handles.gamma = await fs.open(inputFiles.gamma.path, 'r');
    }
    if (inputFiles.beta) {
      handles.beta = await fs.open(inputFiles.beta.path, 'r');
    }

    return handles;
  }

  // NEW: Read chunk inputs directly from files (memory efficient)
  async readChunkInputsFromFiles(fileHandles, seqStart, seqLen, dModel) {
    const chunkInputs = [];
    const floatSize = 4;

    // Input tensor: read seqLen rows starting from seqStart
    if (fileHandles.input) {
      const inputSize = seqLen * dModel * floatSize;
      const inputBuffer = Buffer.alloc(inputSize);
      const offset = seqStart * dModel * floatSize;
      await fileHandles.input.read(inputBuffer, 0, inputSize, offset);
      chunkInputs.push({
        name: 'input', binding: 1,
        data: inputBuffer.toString('base64')
      });
    }

    // Residual tensor: read seqLen rows starting from seqStart (optional)
    if (fileHandles.residual) {
      const residualSize = seqLen * dModel * floatSize;
      const residualBuffer = Buffer.alloc(residualSize);
      const offset = seqStart * dModel * floatSize;
      await fileHandles.residual.read(residualBuffer, 0, residualSize, offset);
      chunkInputs.push({
        name: 'residual', binding: 2,
        data: residualBuffer.toString('base64')
      });
    }

    // Gamma: read entire parameter vector (dModel elements)
    if (fileHandles.gamma) {
      const gammaSize = dModel * floatSize;
      const gammaBuffer = Buffer.alloc(gammaSize);
      await fileHandles.gamma.read(gammaBuffer, 0, gammaSize, 0);
      chunkInputs.push({
        name: 'gamma', binding: 3,
        data: gammaBuffer.toString('base64')
      });
    }

    // Beta: read entire parameter vector (dModel elements)
    if (fileHandles.beta) {
      const betaSize = dModel * floatSize;
      const betaBuffer = Buffer.alloc(betaSize);
      await fileHandles.beta.read(betaBuffer, 0, betaSize, 0);
      chunkInputs.push({
        name: 'beta', binding: 4,
        data: betaBuffer.toString('base64')
      });
    }

    return chunkInputs;
  }

  // NEW: Close file handles
  async closeInputFiles(fileHandles) {
    const closePromises = [];

    for (const [name, handle] of Object.entries(fileHandles)) {
      if (handle) {
        closePromises.push(handle.close());
      }
    }

    await Promise.all(closePromises);
  }

  // NEW: Parse inline inputs (fallback for non-file mode)
  parseInlineInputs(inputData) {
    let parsedInputs = {};
    try {
      parsedInputs = (typeof inputData === 'string') ? JSON.parse(inputData) : inputData;
    } catch (e) {
      parsedInputs = { input: inputData };
    }
    return parsedInputs;
  }

  // NEW: Extract chunk from inline inputs (fallback)
  extractChunkInputsFromInline(inlineInputs, seqStart, seqLen, dModel) {
    const chunkInputs = [];
    const floatSize = 4;

    for (const [name, base64Data] of Object.entries(inlineInputs)) {
      const fullBuffer = Buffer.from(base64Data, 'base64');

      if (name === 'input' || name === 'residual') {
        // Extract sequence slice for input/residual tensors
        const startByte = seqStart * dModel * floatSize;
        const chunkSize = seqLen * dModel * floatSize;
        const chunkBuffer = fullBuffer.slice(startByte, startByte + chunkSize);
        chunkInputs.push({
          name,
          data: chunkBuffer.toString('base64')
        });
      } else {
        // Use full parameter vectors for gamma/beta
        chunkInputs.push({
          name,
          data: base64Data
        });
      }
    }

    return chunkInputs;
  }

  // Existing shader methods remain the same...
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
        let mean = 0.0;
        for (let j = 0; j < dmodel; j++) {
          const v = X[t*dmodel + j] + (Residual ? Residual[t*dmodel + j] : 0.0);
          mean += v;
        }
        mean /= dmodel;
        let varsum = 0.0;
        for (let j = 0; j < dmodel; j++) {
          const v = X[t*dmodel + j] + (Residual ? Residual[t*dmodel + j] : 0.0);
          const d = v - mean;
          varsum += d*d;
        }
        const invstd = 1.0 / Math.sqrt(varsum / dmodel + eps);
        for (let j = 0; j < dmodel; j++) {
          const v = X[t*dmodel + j] + (Residual ? Residual[t*dmodel + j] : 0.0);
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

        var mean: f32 = 0.0;
        for (var m: u32 = 0u; m < dmodel; m = m + 1u) {
          mean = mean + (X[t*dmodel + m] + RES[t*dmodel + m]);
        }
        mean = mean / f32(dmodel);
        var varsum: f32 = 0.0;
        for (var m: u32 = 0u; m < dmodel; m = m + 1u) {
          let v = X[t*dmodel + m] + RES[t*dmodel + m];
          let d = v - mean;
          varsum = varsum + d*d;
        }
        let invstd = inverseSqrt(varsum / f32(dmodel) + params.epsilon);
        let v = X[t*dmodel + j] + RES[t*dmodel + j];
        let z = (v - mean) * invstd;
        OUT[t*dmodel + j] = z * GAMMA[j] + BETA[j];
      }`;
  }
}