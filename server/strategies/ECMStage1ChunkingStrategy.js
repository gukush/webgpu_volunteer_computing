// strategies/ECMStage1ChunkingStrategy.js
// ECM Stage-1 (Montgomery curves) — WebGPU kernel, 64-bit n
// One thread per curve; each thread runs ladder for all p^e (p ≤ B1), then gcd(Z, n).
// If any thread finds 1 < g < n, it atomically publishes g and the chunk is "successful".

import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import { info, warn } from '../logger.js';
import crypto from 'crypto';

const LOG = info('ECM');

function toBase64U64LE(arrBig) {
  const buf = Buffer.allocUnsafe(arrBig.length * 8);
  for (let i = 0; i < arrBig.length; i++) buf.writeBigUInt64LE(BigInt.asUintN(64, arrBig[i]), i * 8);
  return buf.toString('base64');
}
function toBase64PairsU64LE(pairs) {
  const buf = Buffer.allocUnsafe(pairs.length * 16);
  let off = 0n;
  for (const [a, b] of pairs) {
    buf.writeBigUInt64LE(BigInt.asUintN(64, a), Number(off)); off += 8n;
    buf.writeBigUInt64LE(BigInt.asUintN(64, b), Number(off)); off += 8n;
  }
  return buf.toString('base64');
}

// Basic math helpers (BigInt; 64-bit only)
function egcd(a, b) {
  let old_r = a, r = b;
  let old_s = 1n, s = 0n;
  let old_t = 0n, t = 1n;
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q * r];
    [old_s, s] = [s, old_s - q * s];
    [old_t, t] = [t, old_t - q * t];
  }
  return { g: old_r, x: old_s, y: old_t };
}
function mod(a, n) { a %= n; return a < 0n ? a + n : a; }
function modInv(a, n) {
  const { g, x } = egcd(mod(a, n), n);
  if (g !== 1n) return null; // no inverse
  return mod(x, n);
}
function gcd(a, b) { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a,b]=[b, a%b]; return a; }

function primesUpTo(limit) {
  const L = Number(limit);
  const sieve = new Uint8Array(L + 1);
  const out = [];
  for (let i = 2; i <= L; i++) {
    if (!sieve[i]) {
      out.push(i);
      for (let j = i * 2; j <= L; j += i) sieve[j] = 1;
    }
  }
  return out;
}
function primePowersUpTo(B1) {
  const ps = primesUpTo(B1);
  const list = [];
  for (const p of ps) {
    let e = 1n, pe = BigInt(p);
    while (pe * BigInt(p) <= BigInt(B1)) { pe *= BigInt(p); e++; }
    list.push({ p, e: Number(e), pe }); // we’ll multiply by pe using ladder
  }
  return list;
}

/**
 * Suyama parameterization to derive (A24, x1) from σ.
 * u = σ^2 − 5, v = 4σ
 * x1 = (u/v)^2 mod n
 * A = ((v − u)^3 * (3u + v) * inv(4 u^3 v)) − 2  (mod n)
 * A24 = (A+2)/4 = ((v − u)^3 * (3u + v) * inv(16 u^3 v))  (mod n)
 * If any denominator shares gcd with n, that gcd is a factor; we skip such σ here.
 */
function curveFromSigma64(n, sigma) {
  const N = BigInt(n);
  const sig = BigInt(sigma);
  const u = mod(sig*sig - 5n, N);
  const v = mod(4n*sig, N);
  const three_u_plus_v = mod(3n*u + v, N);
  const v_minus_u = mod(v - u, N);
  const v_minus_u_cu = mod(v_minus_u * v_minus_u % N * v_minus_u, N);
  // denom = 16*u^3*v
  const u3 = mod(u * u % N * u, N);
  let denom = mod(16n * u3 % N * v, N);
  let g = gcd(denom, N);
  if (g !== 1n) return { earlyFactor: g };
  const inv = modInv(denom, N);
  if (inv === null) return { skip: true }; // should be handled by g != 1 earlier
  const A24 = mod(v_minus_u_cu * three_u_plus_v % N * inv, N);
  // x1 = (u/v)^2
  const g2 = gcd(v, N);
  if (g2 !== 1n) return { earlyFactor: g2 };
  const vInv = modInv(v, N);
  if (!vInv) return { skip: true };
  const x1 = mod((u * vInv) % N * ((u * vInv) % N), N);
  return { A24, x1 };
}

// WGSL kernel (u64, shift-add mulmod, Montgomery ladder x-only)
const WGSL_ECM_STAGE1 = /* wgsl */`
struct Uniforms {
  n: u64,
  numCurves: u32,
  numPrimePowers: u32,
  _pad: u32
}
@group(0) @binding(0) var<uniform> U : Uniforms;

// pairs of (A24, X1) per curve (Montgomery form uses A24=(A+2)/4 mod n).
struct Curve {
  A24: u64,
  X1: u64
}
@group(0) @binding(1) var<storage, read> CURVES : array<Curve>;

// prime powers p^e as u64
@group(0) @binding(2) var<storage, read> PPS : array<u64>;

// result: atomic flag + factor (u64 split in two u32)
struct Result {
  flag: u32,
  fac_lo: u32,
  fac_hi: u32
}
@group(0) @binding(3) var<storage, read_write> OUT : Result;

fn pack_u64(x: u64) -> vec2<u32> {
  let lo = u32(x & u64(0xffffffffu));
  let hi = u32(x >> 32u);
  return vec2<u32>(lo, hi);
}
fn unpack_u64(v: vec2<u32>) -> u64 {
  return (u64(v.y) << 32u) | u64(v.x);
}

fn addmod(a: u64, b: u64, m: u64) -> u64 {
  if (a >= m - b) { return a - (m - b); }
  return a + b;
}
fn submod(a: u64, b: u64, m: u64) -> u64 {
  if (a >= b) { return a - b; }
  return m - (b - a);
}
fn dblmod(a: u64, m: u64) -> u64 {
  if (a >= m - a) { return a - (m - a); }
  return a + a;
}

// Shift-add (no 128-bit) multiplication: (a*b) mod m
fn mulmod(a0: u64, b0: u64, m: u64) -> u64 {
  var a = a0 % m;
  var b = b0;
  var res: u64 = 0u;
  while (b != 0u) {
    if ((b & 1u) != 0u) {
      if (res >= m - a) { res = res - (m - a); } else { res = res + a; }
    }
    if (a >= m - a) { a = a - (m - a); } else { a = a + a; }
    b = b >> 1u;
  }
  return res;
}
fn sqrmod(a: u64, m: u64) -> u64 {
  return mulmod(a, a, m);
}

fn gcd_u64(a0: u64, b0: u64) -> u64 {
  var a = a0;
  var b = b0;
  if (a == 0u) { return b; }
  if (b == 0u) { return a; }
  loop {
    let r = a % b;
    if (r == 0u) { return b; }
    a = b;
    b = r;
  }
}

// Modular inverse via extended Euclid with modularized coefficients.
// Returns 0 if gcd(a, m) != 1.
fn invmod(a_in: u64, m: u64) -> u64 {
  var a = a_in % m;
  if (a == 0u) { return 0u; }
  var t: u64 = 0u;
  var newt: u64 = 1u;
  var r: u64 = m;
  var newr: u64 = a;
  // Standard EEA with coefficients kept modulo m
  loop {
    if (newr == 0u) { break; }
    let q: u64 = r / newr;
    // (t, newt) = (newt, t - q*newt mod m)
    let tmp_t = newt;
    let qn = mulmod(q, newt, m);
    newt = submod(t, qn, m);
    t = tmp_t;
    // (r, newr) = (newr, r - q*newr)  (exact, no underflow)
    let tmp_r = newr;
    r = r - q * newr;
    newr = tmp_r;
  }
  if (r != 1u) { return 0u; } // not invertible
  return t;                   // 0..m-1
}

// Montgomery x-only doubling with A24 = (A+2)/4 mod n
fn mdbl(X: u64, Z: u64, A24: u64, n: u64) -> vec2<u64> {
  let XPZ = addmod(X, Z, n);
  let XMZ = submod(X, Z, n);
  let AA  = sqrmod(XPZ, n);
  let BB  = sqrmod(XMZ, n);
  let C   = submod(AA, BB, n);
  let X2  = mulmod(AA, BB, n);
  let tmp = addmod(BB, mulmod(A24, C, n), n);
  let Z2  = mulmod(C, tmp, n);
  return vec2<u64>(X2, Z2);
}

// Differential addition: given P=(X1,Z1), Q=(X2,Z2), and XD = X(P-Q)
fn madd(X1: u64, Z1: u64, X2: u64, Z2: u64, XD: u64, n: u64) -> vec2<u64> {
  let t1 = mulmod(addmod(X1, Z1, n), submod(X2, Z2, n), n);
  let t2 = mulmod(submod(X1, Z1, n), addmod(X2, Z2, n), n);
  let X3 = sqrmod(addmod(t1, t2, n), n);
  let Z3 = mulmod(XD, sqrmod(submod(t1, t2, n), n), n);
  return vec2<u64>(X3, Z3);
}

// Montgomery ladder by 64-bit k; base is affine X1 (Z=1)
fn ladder(k: u64, X1: u64, A24: u64, n: u64) -> vec2<u64> {
  var X2: u64 = 1u; var Z2: u64 = 0u;  // [0]P
  var X3: u64 = X1; var Z3: u64 = 1u;  // [1]P
  var started = false;
  var i: i32 = 63;
  loop {
    if (i < 0) { break; }
    let bit: u64 = (k >> u32(i)) & 1u;
    if (!started) {
      if (bit == 1u) { started = true; }
      i = i - 1;
      continue;
    }
    if (bit == 0u) {
      let a = madd(X2, Z2, X3, Z3, X1, n);
      let d = mdbl(X2, Z2, A24, n);
      X3 = a.x; Z3 = a.y; X2 = d.x; Z2 = d.y;
    } else {
      let a = madd(X2, Z2, X3, Z3, X1, n);
      let d = mdbl(X3, Z3, A24, n);
      X2 = a.x; Z2 = a.y; X3 = d.x; Z3 = d.y;
    }
    i = i - 1;
  }
  return vec2<u64>(X2, Z2); // [k]P
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.numCurves) { return; }

  // Early stop if some other thread found a factor
  if (OUT.flag != 0u) { return; }

  let n = U.n;
  let c = CURVES[idx];

  // We'll keep the current base as *affine* X1_aff, and the last projective (X,Z).
  var X1_aff: u64 = c.X1;
  var X: u64 = 0u;
  var Z: u64 = 1u;

  for (var i: u32 = 0u; i < U.numPrimePowers; i = i + 1u) {
    let pe = PPS[i];

    // Multiply current base by pe
    let XZ = ladder(pe, X1_aff, c.A24, n);
    X = XZ.x;
    Z = XZ.y;

    // Optional early poll to bail quickly
    if (((i & 31u) == 0u) && (OUT.flag != 0u)) { return; }

    // If this isn't the last prime power, normalize to affine for the next step.
    if (i + 1u < U.numPrimePowers) {
      // If Z shares a factor with n, we already succeeded.
      let g = gcd_u64(Z % n, n);
      if (g > 1u && g < n) {
        if (OUT.flag == 0u) {
          OUT.flag = 1u;
          let v = pack_u64(g);
          OUT.fac_lo = v.x;
          OUT.fac_hi = v.y;
        }
        return;
      }
      // Otherwise Z is invertible modulo n; convert to affine X/Z
      let zinvy = invmod(Z % n, n);
      // If for some reason inverse failed, try gcd again and stop.
      if (zinvy == 0u) {
        let g2 = gcd_u64(Z % n, n);
        if (g2 > 1u && g2 < n) {
                  if (OUT.flag == 0u) {
          OUT.flag = 1u;
            let v2 = pack_u64(g2);
            OUT.fac_lo = v2.x;
            OUT.fac_hi = v2.y;
          }
        }
        return;
      }
      X1_aff = mulmod(X, zinvy, n);
      // reset Z to 1 (affine base for the next ladder call)
      Z = 1u;
    }
  }

  // Final GCD using the last projective Z
  let g = gcd_u64(Z % n, n);
  if (g > 1u && g < n) {
    if (OUT.flag == 0u) {
      OUT.flag = 1u;
      let v = pack_u64(g);
      OUT.fac_lo = v.x;
      OUT.fac_hi = v.y;
    }
  }
}

`;

// --------------------------------------------

export default class ECMStage1ChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('ecm_stage1');
  }

  validateWorkload(workload) {
    const n = BigInt(workload?.metadata?.n ?? 0);
    if (n <= 1n) return { valid: false, error: `Invalid 'n' (must be > 1)` };
    if (n > 0xffff_ffff_ffff_ffffn) return { valid: false, error: `This MVP supports up to 64-bit n.` };
    const B1 = Number(workload?.metadata?.B1 ?? 50000);
    if (!Number.isFinite(B1) || B1 < 1000) return { valid: false, error: `B1 too small (>=1000).` };
    const curvesPerChunk = Number(workload?.metadata?.curvesPerChunk ?? 256);
    if (!Number.isFinite(curvesPerChunk) || curvesPerChunk < 1) return { valid: false, error: `curvesPerChunk must be >= 1` };
    return { valid: true };
  }

  planExecution(workload) {
    const n = BigInt(workload.metadata.n);
    const B1 = Number(workload.metadata.B1 ?? 50000);
    const curvesTotal = Number(workload.metadata.curvesTotal ?? 8192);
    const curvesPerChunk = Number(workload.metadata.curvesPerChunk ?? 256);
    const framework = workload.framework || 'webgpu';

    const totalChunks = Math.ceil(curvesTotal / curvesPerChunk);

    return {
      chunkingStrategy: this.name,
      assemblyStrategy: 'ecm_stage1_assembly',
      framework,
      totalChunks,
      metadata: {
        n: n.toString(), // keep BigInt; we serialize later
        B1, curvesTotal, curvesPerChunk
      }
    };
  }

  async createCommonInputs(plan) {
    const n = BigInt(plan.metadata.n);
    const B1 = plan.metadata.B1;
    const pps = primePowersUpTo(B1); // array of {p, e, pe: BigInt}
    const peVals = pps.map(x => x.pe);
    const peBase64 = toBase64U64LE(peVals);

    // The OUT buffer: flag (4 bytes) + 8 bytes factor + 4 bytes padding (16 total is fine)
    const initOut = Buffer.alloc(16);
    initOut.writeUInt32LE(0, 0); // flag
    initOut.writeUInt32LE(0, 4); // fac_lo (we zero both halves)
    initOut.writeUInt32LE(0, 8); // fac_hi
    const outInitB64 = initOut.toString('base64');

    // Uniforms are supplied by metadata in descriptor (n, counts). OUT is shared per chunk.
    return { peBase64, numPrimePowers: peVals.length, outInitB64 };
  }

  async createChunkDescriptorsStreaming(plan, dispatch) {
    // Shared inputs for all chunks:
    const common = await this.createCommonInputs(plan);

    const n = BigInt(plan.metadata.n);
    const curvesTotal = plan.metadata.curvesTotal;
    const curvesPerChunk = plan.metadata.curvesPerChunk;

    let emitted = 0;
    let producedCurves = 0;

    while (producedCurves < curvesTotal) {
      const thisCount = Math.min(curvesPerChunk, curvesTotal - producedCurves);

      // Build curve seeds for this chunk
      const curves = [];
      while (curves.length < thisCount) {
        const sigma = 6n + (BigInt.asUintN(53, BigInt(crypto.randomInt(1 << 30))) % 1_000_000_000n);
        const c = curveFromSigma64(n, sigma);
        if (c?.earlyFactor && c.earlyFactor !== n && c.earlyFactor !== 1n) {
          // In principle we could short-circuit here and finish the workload without dispatching,
          // but to keep server plumbing simple, just skip this sigma and pick another.
          continue;
        }
        if (!c?.A24 || !c?.x1) continue; // bad sigma, try again
        curves.push([c.A24, c.x1]);
      }

      const curvesB64 = toBase64PairsU64LE(curves);

      const descriptor = {
        parentId: plan.parentId,
        chunkId: `${plan.parentId}:${emitted}`,
        chunkIndex: emitted,
        framework: plan.framework || 'webgpu',
        kernel: WGSL_ECM_STAGE1,
        entry: 'main',
        metadata: {
          n_lo: Number(n & 0xffff_ffffn),
          n_hi: Number(n >> 32n),
          numCurves: thisCount,
          numPrimePowers: common.numPrimePowers
        },
        inputs: [
          // binding(1): curves buffer
          { name: 'curves', data: curvesB64, size: thisCount * 16 },
          // binding(2): prime powers
          { name: 'prime_powers', data: common.peBase64, size: common.numPrimePowers * 8 }
        ],
        outputs: [
          // binding(3): single result struct (flag + u64 factor)
          { name: 'result', data: common.outInitB64, size: 16 }
        ],
        // binding(0) uniforms are auto-packed from metadata by your framework
        workgroupCount: [Math.ceil(thisCount / 128), 1, 1],
        assemblyMetadata: { curvesInChunk: thisCount }
      };

      await dispatch(descriptor);
      emitted++;
      producedCurves += thisCount;
    }

    return { totalChunks: emitted };
  }

  // Batch mode (not used in streaming by default)
  async createChunkDescriptors(plan) {
    const descs = [];
    await this.createChunkDescriptorsStreaming(plan, d => descs.push(d));
    return descs;
  }
}
