// ECMStage1ChunkingStrategy.js
// Strategy for ECM Stage 1 (Montgomery curves) across multiple frameworks.
// - Generates random curves per chunk (size = chunk_size) with unique seeds
// - Precomputes Montgomery constants for N (R^2 mod N, mont_one, n0inv32)
// - Builds prime powers list up to B1
// - Emits framework-specific descriptors (WGSL for WebGPU, JS kernel fallback, others pass source + buffers)
// NOTE: This file assumes the corresponding kernels exist under server/kernels/ECMStage1/*

import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import { info } from '../logger.js';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KERNEL_DIR = path.resolve(__dirname, '../kernels/ECMStage1');

function toHex(bi) {
  let s = bi.toString(16);
  if (s.length % 2) s = '0' + s;
  return s;
}

function fromHex(hex) {
  return BigInt('0x' + hex.replace(/^0x/, ''));
}

function u32(n) { return Number(n & 0xFFFFFFFFn) >>> 0; }

// Pack a BigInt into 8 x u32 little-endian limbs
function packU256LE(bi) {
  const limbs = new Uint32Array(8);
  let x = bi;
  for (let i = 0; i < 8; i++) {
    limbs[i] = u32(x);
    x >>= 32n;
  }
  return limbs;
}

// Unpack 8 x u32 LE to BigInt
function unpackU256LE(u32arr, offsetWords = 0) {
  let x = 0n;
  for (let i = 0; i < 8; i++) {
    x |= BigInt(u32arr[offsetWords + i]) << (32n * BigInt(i));
  }
  return x;
}

// Compute montgomery parameters for 256-bit (word=32 bits)
function montParams(N) {
  const R = 1n << 256n; // 2^256
  const mont_one = R % N;
  const R2 = (R * R) % N;

  // n0inv32 = -N^{-1} mod 2^32, depends only on least-significant 32-bit word of N
  const n0 = Number(N & 0xFFFFFFFFn) >>> 0;
  if ((n0 & 1) === 0) {
    throw new Error('N must be odd for Montgomery arithmetic (n0 even)');
  }
  // Compute modular inverse of n0 mod 2^32 using extended Euclid on 32-bit integers
  let t = 0n, newT = 1n;
  let r = 1n << 32n, newR = BigInt(n0);
  while (newR !== 0n) {
    const q = r / newR;
    [t, newT] = [newT, t - q * newT];
    [r, newR] = [newR, r - q * newR];
  }
  if (r !== 1n) throw new Error('n0 inverse does not exist');
  if (t < 0n) t += 1n << 32n;
  const inv = Number(t) >>> 0;
  const n0inv32 = ((0x100000000 - inv) >>> 0); // -inv mod 2^32
  return { R2, mont_one, n0inv32 };
}

// Simple deterministic RNG from seed using SHA256(counter || seed)
function* drbg(seedStr) {
  let counter = 0;
  const seed = Buffer.isBuffer(seedStr) ? seedStr : Buffer.from(String(seedStr));
  while (true) {
    const h = crypto.createHash('sha256');
    const cbuf = Buffer.alloc(8);
    cbuf.writeUInt32BE(counter >>> 0, 4);
    h.update(cbuf);
    h.update(seed);
    const digest = h.digest();
    counter++;
    yield digest; // 32 bytes
  }
}

function gcd(a, b) {
  a = a < 0n ? -a : a; b = b < 0n ? -b : b;
  while (b !== 0n) { const t = b; b = a % b; a = t; }
  return a;
}

// Sample random value in [1, N-1] with gcd(val,N)=1
function sampleCoprime(drbgGen, N) {
  while (true) {
    const d = drbgGen.next().value; // 32 bytes
    let x = BigInt('0x' + d.toString('hex'));
    x = (x % (N - 1n)) + 1n;
    if (gcd(x, N) === 1n) return x;
  }
}

// Prime sieve up to B1, return prime powers list (u32)
function primePowersUpTo(B1) {
  const n = B1 >>> 0;
  const sieve = new Uint8Array(n + 1);
  const primes = [];
  for (let i = 2; i <= n; i++) {
    if (!sieve[i]) {
      primes.push(i);
      for (let j = i + i; j <= n; j += i) sieve[j] = 1;
    }
  }
  const list = [];
  for (const p of primes) {
    let pe = p;
    while (pe * p <= n) pe *= p;
    list.push(pe >>> 0);
  }
  return list;
}

// Build ConstsBuffer bytes (binding 0)
function buildConstsBuffer(N, R2, mont_one, n0inv32) {
  const buf = new Uint32Array(28); // 8N + 8R2 + 8mont_one + 1 + 3 pad
  buf.set(packU256LE(N), 0);
  buf.set(packU256LE(R2), 8);
  buf.set(packU256LE(mont_one), 16);
  buf[24] = n0inv32 >>> 0;
  buf[25] = 0; buf[26] = 0; buf[27] = 0;
  return Buffer.from(buf.buffer);
}

// Build CurvesInBuffer bytes (binding 1) for an array of {A24, X1} BigInt
function buildCurvesInBuffer(curves) {
  const out = new Uint32Array(curves.length * 16); // 16 words per curve (A24 + X1)
  let w = 0;
  for (const c of curves) {
    out.set(packU256LE(c.A24), w); w += 8;
    out.set(packU256LE(c.X1),  w); w += 8;
  }
  return Buffer.from(out.buffer);
}

// Build primes buffer (binding 2)
function buildPrimesBuffer(list) {
  const arr = new Uint32Array(list.length);
  for (let i = 0; i < list.length; i++) arr[i] = list[i] >>> 0;
  return Buffer.from(arr.buffer);
}

// Helper to base64-encode a Buffer
function b64(buf) { return Buffer.from(buf).toString('base64'); }

// ---- Strategy class ----

export default class ECMStage1ChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('ecm_stage1');
  }

    defineInputSchema() {
      return {
        inputs: [
          { name: 'packed', type: 'storage_buffer', binding: 1, elementType: 'u32' }, // consts+primes
          { name: 'curves', type: 'storage_buffer', binding: 2, elementType: 'u32' },
        ],
        outputs: [
          { name: 'curve_out', type: 'storage_buffer', binding: 3, elementType: 'u32' }
        ],
        uniforms: [
          {
            name: 'params', type: 'uniform_buffer', binding: 0,
            fields: [
              { name: 'pp_count', type: 'u32' },
              { name: 'num_curves', type: 'u32' },
              { name: 'compute_gcd', type: 'u32' },
              { name: '_upad', type: 'u32' },
            ]
          }
        ]
      };
    }

  // Phase 1: Validate inputs and construct a plan (no heavy work here)
  planExecution(plan) {
    const md = plan.metadata || {};
    let N;
    if (md.N_hex) N = fromHex(md.N_hex);
    else if (md.N_dec) N = BigInt(md.N_dec);
    else throw new Error('Provide N_hex or N_dec');

    if ((N & 1n) === 0n) throw new Error('N must be odd');
    const B1 = md.B1 >>> 0;
    if (!B1 || B1 < 5) throw new Error('B1 must be >= 5');
    const totalCurves = Math.max(1, Number(md.total_curves|0));
    const chunkSize = Math.max(1, Number(md.chunk_size|0));
    const framework = md.framework || 'webgpu';
    const compute_gcd = md.compute_gcd !== false;

    const totalChunks = Math.ceil(totalCurves / chunkSize);

    // Precompute prime powers list
    const ppList = primePowersUpTo(B1);

    // Precompute montgomery constants
    const { R2, mont_one, n0inv32 } = montParams(N);

    // Stash minimal metadata for streaming/batch creation
    const strategyMd = {
      N_hex: toHex(N),
      B1,
      totalCurves,
      chunkSize,
      totalChunks,
      framework,
      compute_gcd,
      seed: md.seed || 'ecm',
      // Strides in bytes for buffers (match WGSL std layout assumptions)
      sizes: {
        consts: 112,        // 28 * 4
        curveIn: 64,        // (A24 + X1) = 16 u32
        outStride: 48       // (U256 + u32) padded to 48 (WGSL std)
      },
      pp_count: ppList.length
    };
    const schema = this.defineInputSchema();
    // Don't compute curve data yet (can be large); do that in createChunkDescriptors / streaming.
    return {
      strategy: this.name,
      assemblyStrategy: 'ecm_stage1_assembly',
      schema,
      metadata: strategyMd,
      shared: {
        constsB64: b64(buildConstsBuffer(N, R2, mont_one, n0inv32)),
        primesB64: b64(buildPrimesBuffer(ppList))
      }
    };
  }

  // Phase 2: Materialize chunk descriptors (batch mode)
async createChunkDescriptors(plan) {
  const parentId = plan.parentId;           // manager sets this
  const md       = plan.metadata;
  const N        = fromHex(md.N_hex);
  const B1       = md.B1 >>> 0;
  const totalCurves = md.totalCurves;
  const chunkSize   = md.chunkSize;
  const totalChunks = Math.ceil(totalCurves / chunkSize);
  const pp_count    = md.pp_count;
  const seed        = md.seed || 'ecm';

  const constsB64  = plan.shared?.constsB64;
  const primesB64  = plan.shared?.primesB64;
  if (!constsB64 || !primesB64) {
    throw new Error('Missing shared buffers: constsB64 or primesB64');
  }
  const constsBuf  = Buffer.from(constsB64, 'base64');
  const primesBuf  = Buffer.from(primesB64, 'base64');
  const packedB64  = Buffer.concat([constsBuf, primesBuf]).toString('base64');



  const kernels = await this.loadKernels();
  const descriptors = [];

  const gen = drbg(`${seed}:${parentId}`);

  let curveIndex = 0;
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    const count = Math.min(chunkSize, totalCurves - curveIndex);
    const curves = [];

    for (let i = 0; i < count; i++) {
      const A24 = sampleCoprime(gen, N);
      const X1  = sampleCoprime(gen, N);
      curves.push({ A24, X1 });
    }

    const curveBuf = buildCurvesInBuffer(curves);
    const outStride = md.sizes?.outStride || 48;
    const outBytes  = count * outStride;

    // Pack uniforms as a fixed 16-byte blob (u32[4]) to match WGSL layout
    const u = new Uint32Array([pp_count, count, md.compute_gcd ? 1 : 0, 0]);
    const uniforms = { binding: 4, data: b64(Buffer.from(u.buffer)) };
    const uMeta = {
      pp_count: pp_count,
      num_curves: count,
      compute_gcd: md.compute_gcd ? 1 : 0,
      pad0: 0
    };
    const base = {
      chunkId: `ecm-${chunkIndex}`,
      id: `ecm-${chunkIndex}`,
      chunkIndex,
      parentId,
      framework: plan.framework || md.framework || 'webgpu',

    // Explicit bindings for inputs to match WGSL @binding(0..2)
    inputs: [
      { name: 'packed', data: packedB64,   binding: 1 }, // matches WGSL @group(1) @binding(1)
      { name: 'curves', data: b64(curveBuf), binding: 2 }, // @group(2) @binding(2)
    ],

      // Outputs + sizes (validator + client)
      outputs:     [{ name: 'curve_out', size: outBytes, binding: 3 }], // binding was 3
      outputSizes: [outBytes],

      // (remove the old binary uniforms blob — the client uses metadata)
      //uniforms,

      // Free-form metadata if you want it
      metadata: {
        chunk_index: chunkIndex,
        total_chunks: totalChunks,
        B1,
        pp_count,
        num_curves: count,
         uniforms: { pp_count, num_curves: count, compute_gcd: md.compute_gcd ? 1 : 0, pad0: 0 }, // ???
      },

      // One thread per curve, @workgroup_size(256) => dispatch ceil(count/256)
      workgroupCount: [Math.ceil(count / 256), 1, 1]
    };

    // Attach the kernel per framework
    const desc = this.createFrameworkSpecificDescriptor(base, kernels, base.framework);
    descriptors.push(desc);
    curveIndex += count;
  }

  return descriptors;
}


async createChunkDescriptorsStreaming(plan, emit) {
  const descs = await this.createChunkDescriptors(plan);
  if (typeof emit === 'function') for (const d of descs) emit(d);
  return { totalChunks: descs.length };
}
async streamChunkDescriptors(plan) {
  const descs = await this.createChunkDescriptors(plan);
  async function* gen() { for (const d of descs) yield d; }
  return { totalChunks: descs.length, stream: gen() };
}

  createFrameworkSpecificDescriptor(base, kernels, framework) {
    const fw = (framework || 'webgpu').toLowerCase();

    switch (fw) {
      case 'webgpu':
        return {
          ...base,
          wgsl: kernels.webgpu.wgsl,
          entry: 'main'
          //workgroup: { x: 256, y: 1, z: 1 } // kernel uses 1 thread per curve; dispatch count handled on client
        };

      case 'javascript':
        return {
          ...base,
          framework: 'javascript',
          kernel: kernels.javascript.src,
          entry: kernels.javascript.entry || 'ecm_stage1'
        };

      case 'webgl':
        // Provide shaders; executor will map metadata/uniforms/inputs via generic helpers
        return {
          ...base,
          framework: 'webgl',
          webglVertexShader: kernels.webgl.vertex,
          webglFragmentShader: kernels.webgl.fragment,
          webglShaderType: 'transform', // hint for client
          webglNumElements: base.metadata.num_curves
        };

      case 'cuda':
      case 'opencl':
      case 'vulkan':
        return {
          ...base,
          framework: fw,
          kernel: kernels[fw].src,
          entry: kernels[fw].entry || 'ecm_stage1'
        };

      default:
        return { ...base, framework: 'webgpu', wgsl: kernels.webgpu.wgsl, entry: 'main' };
    }
  }

  async loadKernels() {
    const read = async (p) => {
      try { return await fs.readFile(p, 'utf8'); }
      catch { return null; }
    };

    const webgpu = { wgsl: await read(path.join(KERNEL_DIR, 'ecm_stage1_webgpu_compute.wgsl')) };
    const javascript = { src: await read(path.join(KERNEL_DIR, 'ecm_stage1_javascript_kernel.js')), entry: 'ecm_stage1' };
    const webgl = {
      vertex: await read(path.join(KERNEL_DIR, 'ecm_stage1_webgl_vertex.glsl')),
      fragment: await read(path.join(KERNEL_DIR, 'ecm_stage1_webgl_fragment.glsl'))
    };
    const opencl = { src: await read(path.join(KERNEL_DIR, 'ecm_stage1_opencl_kernel.cl')), entry: 'ecm_stage1' };
    const cuda   = { src: await read(path.join(KERNEL_DIR, 'ecm_stage1_cuda_kernel.cu')), entry: 'ecm_stage1' };
    const vulkan = { src: await read(path.join(KERNEL_DIR, 'ecm_stage1_vulkan_compute.glsl')), entry: 'main' };

    return { webgpu, javascript, webgl, opencl, cuda, vulkan };
  }
}
