// strategies/ECMStage1AssemblyStrategy.js
// Early-exit assembler: finishes on first non-trivial factor reported by any chunk.

import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import { info } from '../logger.js';

const LOG = info('ECM');

function readU64LEfromB64(b64) {
  const buf = Buffer.from(b64, 'base64');
  if (buf.length < 16) return 0n;
  // layout: flag (4) + fac_lo (4) + fac_hi (4) + pad(4)
  const lo = buf.readUInt32LE(4);
  const hi = buf.readUInt32LE(8);
  return (BigInt(hi) << 32n) | BigInt(lo);
}

export default class ECMStage1AssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('ecm_stage1_assembly');
    this.expected = 0;
    this.seen = 0;
    this.done = false;
    this.n = null;
  }

  async initOutputStore(plan) {
    this.expected = Number(plan.totalChunks || 0);
    this.seen = 0;
    this.done = false;
    this.n = BigInt(plan.metadata.n);
  }

  onBlockComplete(cb) { this._progress = cb; }
  onAssemblyComplete(cb) { this._complete = cb; }

  async processChunkResult(chunkResult) {
    if (this.done) return { status: 'ignored' };

    this.seen++;
    const outB64 = Array.isArray(chunkResult.results) ? chunkResult.results[0] : chunkResult.result;
    const factor = readU64LEfromB64(outB64);
    let payload;

    if (factor && factor > 1n && factor < this.n && (this.n % factor) === 0n) {
      this.done = true;
      payload = {
        result: { n: this.n.toString(), factors: [factor.toString(), (this.n / factor).toString()], method: 'ecm_stage1', complete: true },
        message: `ECM Stage-1 factor found: ${factor}`
      };
      if (this._complete) await this._complete(payload);
      return { success: true, done: true, ...payload };
    }

    if (this._progress) await this._progress({
      completedChunks: this.seen,
      totalChunks: this.expected,
      progress: this.expected ? (100 * this.seen / this.expected) : 0
    });

    if (this.seen >= this.expected && !this.done) {
      this.done = true;
      payload = { result: { n: this.n.toString(), method: 'ecm_stage1', complete: true, found: false } };
      if (this._complete) await this._complete(payload);
      return { success: true, done: true, ...payload };
    }

    return { success: true, done: false };
  }

  async assembleResults(chunks, plan) {
    await this.initOutputStore(plan);
    for (const c of chunks) {
      const r = await this.processChunkResult({ chunkId: c.chunkId, results: Array.isArray(c.results) ? c.results : [c.result] });
      if (r.done) return r;
    }
    return { success: true, result: { n: this.n.toString(), method: 'ecm_stage1', complete: true, found: false } };
  }
}
