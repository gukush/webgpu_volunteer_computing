// COMPLETE: strategies/base/BaseAssemblyStrategy.js - Enhanced base class with multi-output + optional file output

import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import crypto from 'crypto';

function sha256(buf) { return crypto.createHash('sha256').update(buf).digest('hex'); }
function looksLikeDir(p) { return p && (p.endsWith('/') || !path.extname(p)); }

export class BaseAssemblyStrategy {
  /**
   * @param {string} name
   * @param {Object} [opts] - optional, backward compatible
   *   { workloadId?, metadata?, storageRoot?, outputMode?, outputPath?, outputFilename?,
   *     splitOutputsAsFiles?, suppressInMemoryOutputs? }
   */
  constructor(name, opts = {}) {
    this.name = name;
    // Non-breaking: all optional; defaults preserve old behavior (memory-only).
    this._opts = {
      workloadId: opts.workloadId,
      metadata: opts.metadata || {},
      storageRoot: opts.storageRoot || process.env.VOLUNTEER_STORAGE || path.join(os.tmpdir(), 'volunteer'),

      // File-output controls (can also be provided via metadata)
      outputMode: opts.outputMode ?? opts.metadata?.outputMode,                // 'file' | 'memory' | undefined
      outputPath: opts.outputPath ?? opts.metadata?.outputPath,               // file or dir
      outputFilename: opts.outputFilename ?? opts.metadata?.outputFilename ?? 'final.bin',
      splitOutputsAsFiles: !!(opts.splitOutputsAsFiles ?? opts.metadata?.splitOutputsAsFiles),
      suppressInMemoryOutputs: !!(opts.suppressInMemoryOutputs ?? opts.metadata?.suppressInMemoryOutputs),
    };
  }

  /**
   * Assemble results from completed chunks
   * @param {Array} completedChunks - Array of chunk results with metadata
   * @param {Object} plan - The original execution plan
   * @returns {Object} - { success, outputs, metadata, data?, artifact?, error? }
   */
  async assembleResults(completedChunks, plan) {
    const validation = this.validateChunks(completedChunks, plan);
    if (!validation.valid) {
      return {
        success: false,
        error: validation.error,
        missing: validation.missing
      };
    }

    try {
      const schema = plan.schema || this.getDefaultSchema();
      const sortedChunks = this.sortChunks(completedChunks);

      // Multi-output path
      if (schema.outputs && schema.outputs.length > 1) {
        return await this._assembleMultiWithOptionalFile(sortedChunks, plan, schema);
      } else {
        return await this._assembleSingleWithOptionalFile(sortedChunks, plan, schema);
      }
    } catch (error) {
      return {
        success: false,
        error: `Assembly failed: ${error.message}`
      };
    }
  }

  // ---------- NEW: wrappers that keep old returns but add file output when requested ----------

  async _assembleMultiWithOptionalFile(sortedChunks, plan, schema) {
    // Build per-output buffers (already in-memory) and base64 map (back-compat)
    const outputs = {};
    const buffersByOutputIdx = Array(schema.outputs.length).fill().map(() => []);

    for (const chunk of sortedChunks) {
      const chunkResults = chunk.results || [chunk.result];
      if (!Array.isArray(chunkResults)) {
        throw new Error(`Chunk ${chunk.chunkId} results must be an array for multi-output assembly`);
      }
      if (chunkResults.length !== schema.outputs.length) {
        throw new Error(`Chunk ${chunk.chunkId} has ${chunkResults.length} results, expected ${schema.outputs.length}`);
      }
      chunkResults.forEach((result, idx) => {
        buffersByOutputIdx[idx].push(this.decodeResult(result));
      });
    }

    // Assemble each output (default: concatenation, override in subclasses)
    const assembledBuffers = {};
    for (let i = 0; i < schema.outputs.length; i++) {
      const def = schema.outputs[i];
      const assembled = this.assembleSingleOutputBuffers(buffersByOutputIdx[i], plan, def);
      assembledBuffers[def.name] = assembled;
      outputs[def.name] = assembled.toString('base64');
    }

    // Default return (pure memory) if no file mode requested
    const base = {
      success: true,
      outputs: this._opts.suppressInMemoryOutputs ? undefined : outputs,
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };

    // Optional file output
    const artifact = await this._maybeWriteFiles(schema, assembledBuffers);
    if (artifact) {
      base.artifact = artifact;
      // For convenience, expose single-output "data" only when not suppressed and only if exactly 1 output
      if (!this._opts.suppressInMemoryOutputs && schema.outputs.length === 1) {
        const firstName = schema.outputs[0].name;
        base.data = outputs[firstName];
      }
    } else {
      // Pure memory path (back-compat): include `data` only when single output (not this branch)
    }

    return base;
  }

  async _assembleSingleWithOptionalFile(sortedChunks, plan, schema) {
    const outputDef = schema.outputs[0];
    const outputBuffers = sortedChunks.map(chunk => {
      const result = chunk.results ? chunk.results[0] : chunk.result;
      return this.decodeResult(result);
    });

    const assembled = this.assembleSingleOutputBuffers(outputBuffers, plan, outputDef);
    const outputs = { [outputDef.name]: assembled.toString('base64') };

    const base = {
      success: true,
      outputs: this._opts.suppressInMemoryOutputs ? undefined : outputs,
      // Back-compat: keep single-output `data`
      data: this._opts.suppressInMemoryOutputs ? undefined : outputs[outputDef.name],
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };

    const artifact = await this._maybeWriteFiles(schema, { [outputDef.name]: assembled });
    if (artifact) base.artifact = artifact;

    return base;
  }

  /**
   * If requested, write output(s) to disk and return an artifact descriptor.
   * Returns null when not in file mode.
   *
   * For multi-output with splitOutputsAsFiles=true or outputPath as directory,
   * writes one file per named output; otherwise writes a single combined file
   * with outputs concatenated in schema order.
   *
   * @param {Object} schema - output schema ({ outputs: [{name,...}, ...] })
   * @param {Object<string,Buffer>} assembledBuffers - map name -> Buffer
   * @returns {Promise<null | {type:'file'|'file_group', path?, files?, bytes?, sha256?}>}
   */
  async _maybeWriteFiles(schema, assembledBuffers) {
    const mode = this._opts.outputMode || this._opts.metadata?.outputMode;
    if (mode !== 'file' && !this._opts.outputPath) return null;

    const isMulti = (schema.outputs?.length || 0) > 1;
    const outPath = this._opts.outputPath;
    const writePerOutput = isMulti && (this._opts.splitOutputsAsFiles || looksLikeDir(outPath));

    const baseDir = looksLikeDir(outPath)
      ? outPath
      : path.join(this._opts.storageRoot, this._opts.workloadId || 'unknown', 'final');

    await fs.mkdir(baseDir, { recursive: true });

    if (writePerOutput) {
      const files = [];
      for (const def of schema.outputs) {
        const name = def.name;
        const buf = assembledBuffers[name];
        if (!buf) continue;
        const fname = `${name}.bin`;
        const finalPath = path.join(baseDir, fname);
        const tmp = `${finalPath}.part`;
        await fs.writeFile(tmp, buf);
        await fs.rename(tmp, finalPath);
        files.push({ name, path: finalPath, bytes: buf.length, sha256: sha256(buf) });
      }
      return { type: 'file_group', files };
    }

    // single file path resolution
    const finalPath = looksLikeDir(outPath)
      ? path.join(baseDir, this._opts.outputFilename)
      : (outPath || path.join(baseDir, this._opts.outputFilename));

    // concatenate all outputs in schema order
    const ordered = (schema.outputs || [{ name: 'output' }]).map(d => assembledBuffers[d.name]).filter(Boolean);
    const combined = Buffer.concat(ordered);

    const tmp = `${finalPath}.part`;
    await fs.writeFile(tmp, combined);
    await fs.rename(tmp, finalPath);

    return { type: 'file', path: finalPath, bytes: combined.length, sha256: sha256(combined) };
  }

  // ---------- ORIGINAL METHODS (unchanged behavior) ----------

  /**
   * Assemble multiple outputs from chunks
   * (kept for subclass overrides; base class calls the new wrapper above)
   */
  assembleMultipleOutputs(sortedChunks, plan, schema) {
    // Not used directly anymore by base — kept for compatibility if subclasses call it.
    const outputs = {};
    const outputsByIdx = Array(schema.outputs.length).fill().map(() => []);

    for (const chunk of sortedChunks) {
      const chunkResults = chunk.results || [chunk.result]; // Backward compatibility
      if (!Array.isArray(chunkResults)) {
        throw new Error(`Chunk ${chunk.chunkId} results must be an array for multi-output assembly`);
      }
      if (chunkResults.length !== schema.outputs.length) {
        throw new Error(`Chunk ${chunk.chunkId} has ${chunkResults.length} results, expected ${schema.outputs.length}`);
      }
      chunkResults.forEach((result, outputIdx) => {
        if (outputIdx < outputsByIdx.length) {
          outputsByIdx[outputIdx].push(Buffer.from(result, 'base64'));
        }
      });
    }

    for (let i = 0; i < schema.outputs.length; i++) {
      const outputDef = schema.outputs[i];
      const outputBuffers = outputsByIdx[i];
      const assembledOutput = this.assembleSingleOutputBuffers(outputBuffers, plan, outputDef);
      outputs[outputDef.name] = assembledOutput.toString('base64');
    }

    return {
      success: true,
      outputs: outputs,
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };
  }

  /**
   * Assemble single output from chunks (backward compatibility)
   * (kept for subclass overrides; base class calls the new wrapper above)
   */
  assembleSingleOutput(sortedChunks, plan, schema) {
    const outputDef = schema.outputs[0];
    const outputBuffers = sortedChunks.map(chunk => {
      const result = chunk.results ? chunk.results[0] : chunk.result;
      return this.decodeResult(result);
    });

    const assembledOutput = this.assembleSingleOutputBuffers(outputBuffers, plan, outputDef);
    const outputName = outputDef.name;

    return {
      success: true,
      outputs: { [outputName]: assembledOutput.toString('base64') },
      data: assembledOutput.toString('base64'), // Backward compatibility
      metadata: this.createAssemblyMetadata(plan, sortedChunks)
    };
  }

  /**
   * Assemble buffers for a single output
   * @param {Array<Buffer>} buffers
   * @param {Object} plan
   * @param {Object} outputDef
   * @returns {Buffer}
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    // Default: concatenate buffers
    return Buffer.concat(buffers);
  }

  validateChunks(completedChunks, plan) {
    const expectedChunks = plan.totalChunks;
    const receivedChunks = completedChunks.length;

    if (receivedChunks !== expectedChunks) {
      const received = completedChunks.map(c => c.chunkIndex || c.chunkId);
      const missing = [];
      for (let i = 0; i < expectedChunks; i++) {
        if (!received.includes(i)) missing.push(i);
      }
      return { valid: false, missing, error: `Expected ${expectedChunks} chunks, got ${receivedChunks}` };
    }

    // Validate multi-output chunks
    const schema = plan.schema || this.getDefaultSchema();
    if (schema.outputs && schema.outputs.length > 1) {
      for (const chunk of completedChunks) {
        const chunkResults = chunk.results || (chunk.result ? [chunk.result] : []);
        if (chunkResults.length !== schema.outputs.length) {
          return {
            valid: false,
            error: `Chunk ${chunk.chunkId || chunk.chunkIndex} has ${chunkResults.length} results, expected ${schema.outputs.length}`
          };
        }
      }
    }

    return { valid: true };
  }

  sortChunks(chunks) {
    return chunks.sort((a, b) => {
      const indexA = a.chunkIndex !== undefined ? a.chunkIndex : this.extractChunkIndex(a.chunkId);
      const indexB = b.chunkIndex !== undefined ? b.chunkIndex : this.extractChunkIndex(b.chunkId);
      return indexA - indexB;
    });
  }

  extractChunkIndex(chunkId) {
    const matches = (chunkId || '').match(/(\d+)$/);
    return matches ? parseInt(matches[1], 10) : 0;
  }

  decodeResult(base64Result) {
    try {
      return Buffer.from(base64Result, 'base64');
    } catch (e) {
      throw new Error(`Invalid base64 result data: ${e.message}`);
    }
    }

  concatenateResults(sortedChunks) {
    const buffers = sortedChunks.map(chunk => {
      const result = chunk.results ? chunk.results[0] : chunk.result;
      return this.decodeResult(result);
    });
    return Buffer.concat(buffers);
  }

  getDefaultSchema() {
    return {
      outputs: [{ name: 'output', type: 'storage_buffer', elementType: 'f32' }]
    };
  }

  createAssemblyMetadata(plan, chunks) {
    return {
      assemblyStrategy: this.name,
      totalChunks: chunks.length,
      assembledAt: Date.now(),
      originalPlan: { strategy: plan.strategy, totalChunks: plan.totalChunks },
      outputCount: plan.schema ? plan.schema.outputs.length : 1
    };
  }
}
