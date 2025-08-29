// ECMStage1AssemblyStrategy.js
// Streaming-friendly assembler for ECM Stage 1
// - Parses per-curve outputs (U256 result + status)
// - If any curve reports status==2 (non-trivial factor), finalizes early with that factor
// - Otherwise tracks progress; upon completion, returns a JSON summary

import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import { info } from '../logger.js';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';

const _streamingProgress = new Map();
// Helpers to parse outputs
function readU32LE(buf, o) {
  return buf[o] | (buf[o+1]<<8) | (buf[o+2]<<16) | (buf[o+3]<<24);
}
function limbsToBigInt(buf, byteOffset) {
  let x = 0n;
  for (let i = 0; i < 8; i++) {
    const w = readU32LE(buf, byteOffset + i*4) >>> 0;
    x |= BigInt(w) << (32n * BigInt(i));
  }
  return x;
}
function bigIntToHex(bi) {
  let s = bi.toString(16);
  if (s.length % 2) s = '0' + s;
  return s;
}

/**
 * Helper: get numeric totalChunks from parent metadata if present, otherwise undefined.
 * Accepts whatever parent object you have available in your strategy.
 */
function _getTotalChunksFromParent(parent) {
  if (!parent) return undefined;
  // Common places where total chunk count might live:
  return parent.totalChunks
      || parent.metadata?.totalChunks
      || parent.metadata?.streamTotalChunks
      || parent.plan?.totalChunks
      || undefined;
}


export default class ECMStage1AssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('ecm_stage1_assembly');
    this.totalChunks = 0;
    this.completed = 0;
    this.factor = null;
    this.curvesChecked = 0;
    this.outStride = 48; // bytes per element (U256 + u32 status, padded to 48)
    // ADD: Streaming callbacks infrastructure
    this.streamingCallbacks = new Map();
  }

  async initOutputStore(plan) {
    // Nothing to persist on disk for ECM stage 1
    this.totalChunks = plan.metadata?.totalChunks || 0;
    this.outStride = plan.metadata?.sizes?.outStride || 48;
    this.factor = null;
    this.completed = 0;
    this.curvesChecked = 0;
  }

  // ADD: Streaming callback registration methods
  onBlockComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('block_complete', callback);
    }
  }

  onAssemblyComplete(callback) {
    if (typeof callback === 'function') {
      this.streamingCallbacks.set('assembly_complete', callback);
    }
  }

  // Streaming assembly entry point
  async processChunkResult(chunkResult, chunkDescriptor) {
    try {
      // chunkResult contains .result (base64) or .results[0]
      const base64 = chunkResult.result || (chunkResult.results && chunkResult.results[0]);
      if (!base64) {
        return { success: false, status: 'error', error: 'Missing result' };
      }
      const raw = Buffer.from(base64, 'base64');
      const num = chunkDescriptor?.metadata?.num_curves || 0;

      // Parse each element
      let found = null;
      for (let i = 0; i < num; i++) {
        const off = i * this.outStride;
        const g = limbsToBigInt(raw, off);  // 32 bytes
        const status = readU32LE(raw, off + 32) >>> 0;
        this.curvesChecked++;

        if (__DEBUG_ON__) {
          if (status !== 0) {
            console.log(`[ECM ASM] Curve #${i} status=${status}, g=${g.toString(16)}`);
          }
        }

        // status: 2 => non-trivial gcd, 1 => gcd=1/no factor, 0 => raw Z returned
        if (status === 2) {
          found = g;
          break;
        }
      }

      this.completed++;

      // ADD: Fire progress callback after each chunk
      const progressCallback = this.streamingCallbacks.get('block_complete');
      if (progressCallback && this.totalChunks > 0) {
        const progress = (this.completed / this.totalChunks) * 100;
        await progressCallback({
          completedBlocks: this.completed,
          totalBlocks: this.totalChunks,
          progress: progress,
          curvesChecked: this.curvesChecked
        });
      }

      if (found && !this.factor) {
        this.factor = found;
        const payload = Buffer.from(JSON.stringify({
          status: 'factor_found',
          factor_hex: bigIntToHex(found),
          factor_dec: found.toString(10),
          curves_checked: this.curvesChecked
        }), 'utf8').toString('base64');

        // ADD: Fire completion callback for early termination
        const completeCallback = this.streamingCallbacks.get('assembly_complete');
        if (completeCallback) {
          await completeCallback({
            result: payload,
            status: 'factor_found',
            curvesChecked: this.curvesChecked
          });
        }

        return {
          success: true,
          status: 'complete',
          finalResult: payload,
          stats: {
            assemblyStrategy: this.name,
            completedChunks: this.completed,
            totalChunks: this.totalChunks,
            curvesChecked: this.curvesChecked
          }
        };
      }

      // If all chunks processed and no factor
      if (this.completed >= this.totalChunks) {
        const payload = Buffer.from(JSON.stringify({
          status: 'no_factor',
          curves_checked: this.curvesChecked
        }), 'utf8').toString('base64');

        // ADD: Fire completion callback for normal completion
        const completeCallback = this.streamingCallbacks.get('assembly_complete');
        if (completeCallback) {
          await completeCallback({
            result: payload,
            status: 'no_factor',
            curvesChecked: this.curvesChecked
          });
        }

        return {
          success: true,
          status: 'complete',
          finalResult: payload,
          stats: {
            assemblyStrategy: this.name,
            completedChunks: this.completed,
            totalChunks: this.totalChunks,
            curvesChecked: this.curvesChecked
          }
        };
      }

      // Still in progress
      return {
        success: true,
        status: 'in_progress',
        stats: {
          assemblyStrategy: this.name,
          completedChunks: this.completed,
          totalChunks: this.totalChunks,
          curvesChecked: this.curvesChecked
        }
      };

    } catch (e) {
      return { success: false, status: 'error', error: e.message };
    }
  }

  async cleanup() {
    // ADD: Clear streaming callbacks
    this.streamingCallbacks.clear();
  }
}