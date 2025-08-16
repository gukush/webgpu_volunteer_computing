#!/usr/bin/env node
// submit-task.mjs
// Tiny CLI to submit tasks to your server (matrix start, WGSL workloads, set K, start queued WGSL)
// Requires Node 18+ (for global fetch) or Node 16 with `node --experimental-fetch`.
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function printHelp() {
  console.log(`
Usage:
  node submit-task.mjs matrix --size <int> --chunk <int> [--host https://localhost:3000]
  node submit-task.mjs wgsl --wgsl <file.wgsl> --label <name> --workgroups <x,y,z> --output-size <bytes> [--entry main] [--bind storage-in-storage-out] [--input <file.bin>] [--chunkable [--chunk-type elements|bytes --chunk-size <n> --elem-size <bytes> --agg concatenate]] [--host https://localhost:3000]
  node submit-task.mjs wgsl-start [--host https://localhost:3000]
  node submit-task.mjs set-k --k <int> [--host https://localhost:3000]

Examples:
  node submit-task.mjs matrix --size 512 --chunk 32
  node submit-task.mjs set-k --k 2
  node submit-task.mjs wgsl --wgsl ./mandelbrot.wgsl --label "Mandelbrot" --workgroups 64,1,1 --output-size 1048576
  node submit-task.mjs wgsl --wgsl ./kernel.wgsl --label "VecAdd" --workgroups 1024,1,1 --output-size 4096 --input ./input.bin --chunkable --chunk-type elements --chunk-size 1024 --elem-size 4 --agg concatenate
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const val = (i + 1 < argv.length && !argv[i+1].startsWith('--')) ? argv[++i] : true;
      args[key] = val;
    } else {
      args._.push(a);
    }
  }
  return args;
}

function hostBase(args) {
  return (args.host || process.env.TASK_HOST || 'https://localhost:3000').replace(/\/$/, '');
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function del(url) {
  const res = await fetch(url, { method: 'DELETE' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  const args = parseArgs(rest);
  const base = hostBase(args);

  if (!cmd || args.help || args.h) return printHelp();

  if (cmd === 'matrix') {
    const size = parseInt(args.size, 10);
    const chunk = parseInt(args.chunk, 10);
    if (!Number.isInteger(size) || !Number.isInteger(chunk)) {
      console.error('matrix: --size and --chunk are required integers');
      process.exit(1);
    }
    const out = await postJSON(`${base}/api/matrix/start`, { matrixSize: size, chunkSize: chunk });
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  if (cmd === 'set-k') {
    const k = parseInt(args.k, 10);
    if (!Number.isInteger(k) || k < 1) {
      console.error('set-k: --k must be integer >= 1');
      process.exit(1);
    }
    const out = await postJSON(`${base}/api/system/k`, { k });
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  if (cmd === 'wgsl') {
    if (!args.wgsl) {
      console.error('wgsl: --wgsl <file.wgsl> is required');
      process.exit(1);
    }
    const wgslPath = path.resolve(args.wgsl);
    const wgsl = await fs.readFile(wgslPath, 'utf8');
    const label = args.label || path.basename(wgslPath);
    const entry = args.entry || 'main';
    const workgroups = (args.workgroups || '1,1,1').split(',').map(x => parseInt(x, 10));
    if (workgroups.length !== 3 || workgroups.some(n => !Number.isInteger(n) || n <= 0)) {
      console.error('wgsl: --workgroups must be like "64,1,1"');
      process.exit(1);
    }
    const outputSize = parseInt(args['output-size'], 10);
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      console.error('wgsl: --output-size is required and must be > 0');
      process.exit(1);
    }
    const bindLayout = args.bind || 'storage-in-storage-out';
    let inputBase64 = undefined;
    if (args.input) {
      const buf = await fs.readFile(path.resolve(args.input));
      inputBase64 = buf.toString('base64');
    }
    const payload = {
      label,
      wgsl,
      entry,
      workgroupCount: workgroups,
      bindLayout,
      outputSize,
      input: inputBase64,
      chunkable: !!args.chunkable
    };
    if (args.chunkable) {
      payload.inputChunkProcessingType = args['chunk-type'] || 'elements';
      payload.inputChunkSize = parseInt(args['chunk-size'] || '1024', 10);
      payload.inputElementSizeBytes = parseInt(args['elem-size'] || '4', 10);
      payload.outputAggregationMethod = args.agg || 'concatenate';
    }
    const out = await postJSON(`${base}/api/workloads`, payload);
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  if (cmd === 'wgsl-start') {
    const out = await postJSON(`${base}/api/workloads/startQueued`, {});
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  if (cmd === 'workloads-rm') {
    const id = args.id;
    if (!id) {
      console.error('workloads-rm: --id <workloadId> is required');
      process.exit(1);
    }
    const out = await del(`${base}/api/workloads/${encodeURIComponent(id)}`);
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  printHelp();
}

main().catch(err => {
  console.error(err.message || err);
  process.exit(1);
});
