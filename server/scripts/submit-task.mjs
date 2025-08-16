#!/usr/bin/env node
// submit-task.mjs - Multi-framework extension
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SUPPORTED_FRAMEWORKS = {
  'webgpu': { extension: '.wgsl', defaultEntry: 'main' },
  'webgl': { extension: '.glsl', defaultEntry: 'main' },
  'cuda': { extension: '.cu', defaultEntry: 'kernel' },
  'opencl': { extension: '.cl', defaultEntry: 'kernel' },
  'vulkan': { extension: '.comp', defaultEntry: 'main' }
};

function printHelp() {
  console.log(`
Usage:
  node submit-task.mjs matrix --size <int> --chunk <int> [--host https://localhost:3000]
  node submit-task.mjs compute --framework <framework> --kernel <file> --label <name> --workgroups <x,y,z> --output-size <bytes> [options...] [--host https://localhost:3000]
  node submit-task.mjs wgsl --wgsl <file.wgsl> --label <name> --workgroups <x,y,z> --output-size <bytes> [options...] [--host https://localhost:3000]
  node submit-task.mjs compute-start [--host https://localhost:3000]
  node submit-task.mjs set-k --k <int> [--host https://localhost:3000]

Frameworks:
  ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}

Compute Command Options:
  --framework <framework>     Computing framework to use (required)
  --kernel <file>            Kernel source file (required)
  --label <name>             Human-readable label for the workload
  --workgroups <x,y,z>       Workgroup dispatch dimensions
  --output-size <bytes>      Expected total output buffer size in bytes
  --entry <name>             Kernel entry point function name
  --bind <layout>            Buffer binding layout convention
  --input <file.bin>         Input data file (binary)
  --compilation-opts <json>  Framework-specific compilation options

Chunking Options (for large workloads):
  --chunkable                Enable input chunking
  --chunk-type <type>        Chunking type: elements|bytes
  --chunk-size <n>           Size per chunk
  --chunk-output-size <bytes> Output size per chunk (required for chunking)
  --elem-size <bytes>        Element size in bytes (for element chunking)
  --agg <method>             Output aggregation method: concatenate

Examples:
  # WebGPU compute shader
  node submit-task.mjs compute --framework webgpu --kernel ./mandelbrot.wgsl --label "Mandelbrot" --workgroups 64,1,1 --output-size 1048576

  # CUDA kernel
  node submit-task.mjs compute --framework cuda --kernel ./vecadd.cu --label "Vector Add" --workgroups 1024,1,1 --output-size 4096 --input ./data.bin

  # OpenCL with chunking
  node submit-task.mjs compute --framework opencl --kernel ./process.cl --label "Data Processing" --workgroups 256,1,1 --output-size 8192 --input ./large_data.bin --chunkable --chunk-type elements --chunk-size 1024 --chunk-output-size 512 --elem-size 4

  # WebGL compute
  node submit-task.mjs compute --framework webgl --kernel ./shader.glsl --label "WebGL Compute" --workgroups 32,32,1 --output-size 2048

  # Vulkan compute
  node submit-task.mjs compute --framework vulkan --kernel ./compute.comp --label "Vulkan Compute" --workgroups 128,1,1 --output-size 4096 --compilation-opts '{"spirvVersion": "1.5"}'

  # Legacy WGSL command (backward compatibility)
  node submit-task.mjs wgsl --wgsl ./mandelbrot.wgsl --label "Mandelbrot" --workgroups 64,1,1 --output-size 1048576

  # Matrix multiplication
  node submit-task.mjs matrix --size 512 --chunk 32

  # Set redundancy factor
  node submit-task.mjs set-k --k 2

  # Start all queued compute workloads
  node submit-task.mjs compute-start
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

async function validateFramework(framework) {
  if (!SUPPORTED_FRAMEWORKS[framework]) {
    console.error(`Unsupported framework: ${framework}`);
    console.error(`Supported frameworks: ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}`);
    process.exit(1);
  }
}

async function readKernelFile(filePath, framework) {
  try {
    const resolvedPath = path.resolve(filePath);
    const kernel = await fs.readFile(resolvedPath, 'utf8');

    // Validate file extension matches framework
    const expectedExt = SUPPORTED_FRAMEWORKS[framework].extension;
    const actualExt = path.extname(filePath);

    if (actualExt !== expectedExt) {
      console.warn(`Warning: File extension ${actualExt} doesn't match expected ${expectedExt} for ${framework}`);
    }

    return kernel;
  } catch (err) {
    console.error(`Failed to read kernel file ${filePath}: ${err.message}`);
    process.exit(1);
  }
}

async function parseCompilationOptions(optsString) {
  if (!optsString) return {};

  try {
    return JSON.parse(optsString);
  } catch (err) {
    console.error(`Invalid compilation options JSON: ${err.message}`);
    process.exit(1);
  }
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

  if (cmd === 'compute') {
    const framework = args.framework;
    if (!framework) {
      console.error('compute: --framework is required');
      console.error(`Supported frameworks: ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}`);
      process.exit(1);
    }

    await validateFramework(framework);

    if (!args.kernel) {
      console.error('compute: --kernel <file> is required');
      process.exit(1);
    }

    const kernel = await readKernelFile(args.kernel, framework);
    const label = args.label || path.basename(args.kernel);
    const entry = args.entry || SUPPORTED_FRAMEWORKS[framework].defaultEntry;
    const workgroups = (args.workgroups || '1,1,1').split(',').map(x => parseInt(x, 10));

    if (workgroups.length !== 3 || workgroups.some(n => !Number.isInteger(n) || n <= 0)) {
      console.error('compute: --workgroups must be like "64,1,1"');
      process.exit(1);
    }

    const outputSize = parseInt(args['output-size'], 10);
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      console.error('compute: --output-size is required and must be > 0');
      process.exit(1);
    }

    // Framework-specific bind layout defaults
    const defaultBindLayouts = {
      webgpu: 'storage-in-storage-out',
      webgl: 'webgl-transform-feedback',
      cuda: 'cuda-global-memory',
      opencl: 'opencl-global-memory',
      vulkan: 'vulkan-storage-buffer'
    };

    const bindLayout = args.bind || defaultBindLayouts[framework];
    const compilationOptions = await parseCompilationOptions(args['compilation-opts']);

    let inputBase64 = undefined;
    if (args.input) {
      const buf = await fs.readFile(path.resolve(args.input));
      inputBase64 = buf.toString('base64');
    }

    const payload = {
      label,
      framework,
      kernel,
      entry,
      workgroupCount: workgroups,
      bindLayout,
      outputSize,
      input: inputBase64,
      chunkable: !!args.chunkable,
      compilationOptions
    };

    if (args.chunkable) {
      // Validate chunk output size is provided for chunking
      const chunkOutputSize = parseInt(args['chunk-output-size'], 10);
      if (!Number.isInteger(chunkOutputSize) || chunkOutputSize <= 0) {
        console.error('compute: --chunk-output-size is required for chunkable workloads and must be > 0');
        process.exit(1);
      }

      payload.inputChunkProcessingType = args['chunk-type'] || 'elements';
      payload.inputChunkSize = parseInt(args['chunk-size'] || '1024', 10);
      payload.chunkOutputSize = chunkOutputSize;
      payload.inputElementSizeBytes = parseInt(args['elem-size'] || '4', 10);
      payload.outputAggregationMethod = args.agg || 'concatenate';

      if (!args.input) {
        console.error('compute: --input is required for chunkable workloads');
        process.exit(1);
      }
    }

    const out = await postJSON(`${base}/api/workloads`, payload);
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  // Backward compatibility: 'wgsl' command
  if (cmd === 'wgsl') {
    console.warn('WARNING: "wgsl" command is deprecated. Use "compute --framework webgpu" instead.');

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
      framework: 'webgpu', // Force WebGPU for backward compatibility
      kernel: wgsl,
      wgsl, // Keep for server backward compatibility
      entry,
      workgroupCount: workgroups,
      bindLayout,
      outputSize,
      input: inputBase64,
      chunkable: !!args.chunkable
    };

    if (args.chunkable) {
      // Also validate chunk output size for legacy WGSL command
      const chunkOutputSize = parseInt(args['chunk-output-size'], 10);
      if (!Number.isInteger(chunkOutputSize) || chunkOutputSize <= 0) {
        console.error('wgsl: --chunk-output-size is required for chunkable workloads and must be > 0');
        process.exit(1);
      }

      payload.inputChunkProcessingType = args['chunk-type'] || 'elements';
      payload.inputChunkSize = parseInt(args['chunk-size'] || '1024', 10);
      payload.chunkOutputSize = chunkOutputSize;
      payload.inputElementSizeBytes = parseInt(args['elem-size'] || '4', 10);
      payload.outputAggregationMethod = args.agg || 'concatenate';
    }

    const out = await postJSON(`${base}/api/workloads`, payload);
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  if (cmd === 'compute-start' || cmd === 'wgsl-start') {
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

  if (cmd === 'frameworks') {
    try {
      const res = await fetch(`${base}/api/frameworks`);
      const data = await res.json();
      console.log(JSON.stringify(data, null, 2));
    } catch (err) {
      console.error('Failed to fetch framework info:', err.message);
      process.exit(1);
    }
    return;
  }

  printHelp();
}

main().catch(err => {
  console.error(err.message || err);
  process.exit(1);
});