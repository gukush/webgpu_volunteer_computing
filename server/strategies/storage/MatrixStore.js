// strategies/storage/MatrixStore.js
import fs from 'fs';
let mmap;
try {
  // Optional native dep; fall back to plain fs if not installed
  mmap = await import('mmap-io').then(m => m.default || m).catch(() => null);
} catch { mmap = null; }

const BYTE_PER_F32 = 4;

export function bytesForF32(rows, cols) {
  return rows * cols * BYTE_PER_F32;
}

export class InMemoryMatrixStore {
  constructor({ rows, cols }) {
    this.rows = rows; this.cols = cols;
    this.kind = 'memory';
    this.view = new Float32Array(rows * cols);
  }
  // returns a view (not a copy) of a contiguous block
  getBlockView(blockRow, blockCol, blockSize) {
    const out = new Float32Array(blockSize * blockSize);
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const src = (baseRow + r) * this.cols + baseCol;
      out.set(this.view.subarray(src, src + blockSize), r * blockSize);
    }
    return out;
  }
  // copy data into the right place
  putBlock(blockRow, blockCol, blockSize, blockData) {
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const dst = (baseRow + r) * this.cols + baseCol;
      this.view.set(blockData.subarray(r * blockSize, (r + 1) * blockSize), dst);
    }
  }
  // accumulate (C += partial)
  addBlock(blockRow, blockCol, blockSize, partial) {
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const dst = (baseRow + r) * this.cols + baseCol;
      for (let c = 0; c < blockSize; c++) {
        this.view[dst + c] += partial[r * blockSize + c];
      }
    }
  }
  close() {}
}

export class MmapMatrixStore {
  constructor({ filePath, rows, cols, writable = true, initialize = false }) {
    if (!mmap) {
      throw new Error('mmap-io not installed; run `npm i mmap-io` or use InMemoryMatrixStore');
    }
    this.kind = 'mmap';
    this.rows = rows; this.cols = cols; this.filePath = filePath;
    const byteLen = bytesForF32(rows, cols);
    const fd = fs.openSync(filePath, writable ? 'w+' : 'r');
    if (initialize) fs.ftruncateSync(fd, byteLen);
    const prot = (writable ? (mmap.PROT_READ | mmap.PROT_WRITE) : mmap.PROT_READ);
    const flags = mmap.MAP_SHARED;
    this.buf = mmap.map(byteLen, prot, flags, fd, 0);
    this.fd = fd;
    // Note: node Buffer wraps mapped memory; use Float32Array over its ArrayBuffer
    this.view = new Float32Array(
      this.buf.buffer,
      this.buf.byteOffset,
      byteLen / BYTE_PER_F32
    );
  }
  // Views over mmap must be copies to avoid aliasing confusion
  getBlockView(blockRow, blockCol, blockSize) {
    const out = new Float32Array(blockSize * blockSize);
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const src = (baseRow + r) * this.cols + baseCol;
      out.set(this.view.subarray(src, src + blockSize), r * blockSize);
    }
    return out;
  }
  putBlock(blockRow, blockCol, blockSize, blockData) {
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const dst = (baseRow + r) * this.cols + baseCol;
      this.view.set(blockData.subarray(r * blockSize, (r + 1) * blockSize), dst);
    }
  }
  addBlock(blockRow, blockCol, blockSize, partial) {
    const baseRow = blockRow * blockSize;
    const baseCol = blockCol * blockSize;
    for (let r = 0; r < blockSize; r++) {
      const dst = (baseRow + r) * this.cols + baseCol;
      for (let c = 0; c < blockSize; c++) {
        this.view[dst + c] += partial[r * blockSize + c];
      }
    }
  }
  close() {
    try { mmap && mmap.unmap(this.buf); } catch {}
    try { fs.closeSync(this.fd); } catch {}
  }
}

// Decide RAM vs mmap, with “prefer output in memory” heuristic
export function selectStore({ filePath, rows, cols, thresholdBytes, preferOutputInMemory = false, writable = true, initialize = false }) {
  const bytes = bytesForF32(rows, cols);
  const useMemory = bytes <= thresholdBytes || (!writable && bytes <= thresholdBytes); // inputs can be smol
  if (preferOutputInMemory && writable && bytes <= thresholdBytes) {
    return new InMemoryMatrixStore({ rows, cols });
  }
  if (useMemory) return new InMemoryMatrixStore({ rows, cols });
  if (!filePath) throw new Error('filePath required for mmap store');
  return new MmapMatrixStore({ filePath, rows, cols, writable, initialize });
}
