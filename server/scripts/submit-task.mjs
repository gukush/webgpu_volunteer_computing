#!/usr/bin/env node
// submit-task.mjs - Multi-framework extension with enhanced chunking system
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
Multi-Framework Volunteer Computing Task Submission

Usage:
  # Traditional commands
  node submit-task.mjs matrix --size <int> --chunk <int> [--host <url>]
  node submit-task.mjs compute --framework <fw> --kernel <file> --label <name> --workgroups <x,y,z> --output-size <bytes> [options...]
  node submit-task.mjs wgsl --wgsl <file.wgsl> --label <name> --workgroups <x,y,z> --output-size <bytes> [options...]

  # Enhanced commands (new)
  node submit-task.mjs compute-advanced --framework <fw> --kernel <file> --chunking <strategy> --assembly <strategy> --metadata <json> [options...]
  node submit-task.mjs matrix-tiled --size <int> --tile-size <int> [--label <name>]
  node submit-task.mjs sort --algorithm <alg> --array-size <int> [--chunk-size <int>] [--input <file>]
  node submit-task.mjs strategy-upload --type <chunking|assembly> --name <name> --file <strategy.js>
  node submit-task.mjs strategies-list
  node submit-task.mjs frameworks

  # System commands
  node submit-task.mjs compute-start
  node submit-task.mjs set-k --k <int>
  node submit-task.mjs workloads-rm --id <workloadId>

Frameworks:
  ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}

Enhanced Compute Options:
  --framework <framework>     Computing framework (required)
  --kernel <file>            Kernel source file (required)
  --chunking <strategy>      Chunking strategy name or .js file (required)
  --assembly <strategy>      Assembly strategy name or .js file (required)
  --metadata <json>          Strategy-specific metadata (required)
  --label <name>             Human-readable label
  --input <file>             Input data file (binary, required for chunking)
  --compilation-opts <json>  Framework-specific options

Built-in Strategies:
  Chunking: linear, matrix_tiled, bitonic_sort
  Assembly: linear_assembly, matrix_tiled_assembly

Sorting Algorithms:
  bitonic    - Bitonic sort (requires power-of-2 array size)
  odd-even   - Odd-even sort (any size, often faster convergence)
  sample     - Sample sort (good for very large arrays)

Examples:
  # Enhanced tiled matrix multiplication
  node submit-task.mjs matrix-tiled --size 1024 --tile-size 64 --label "Large Matrix"

  # Custom chunking strategy upload
  node submit-task.mjs strategy-upload --type chunking --name my_chunking --file ./my_strategy.js

  # Advanced workload with custom strategies
  node submit-task.mjs compute-advanced \\
    --framework webgpu \\
    --kernel ./image_blur.wgsl \\
    --chunking image_tiled \\
    --assembly image_tiled_assembly \\
    --metadata '{"imageWidth": 1920, "imageHeight": 1080, "tileSize": 128}' \\
    --input ./image_data.bin \\
    --label "Image Blur Processing"

  # Multi-framework support
  node submit-task.mjs compute-advanced \\
    --framework cuda \\
    --kernel ./fluid_sim.cu \\
    --chunking ./custom_chunking.js \\
    --assembly ./custom_assembly.js \\
    --metadata '{"gridSize": 512, "timeSteps": 100}' \\
    --input ./initial_conditions.bin \\
    --compilation-opts '{"deviceId": 0, "computeCapability": "7.5"}'

  # Iterative sorting
  node submit-task.mjs sort --algorithm bitonic --array-size 2048 --chunk-size 64

  # List available strategies and frameworks
  node submit-task.mjs strategies-list
  node submit-task.mjs frameworks

Chunking Options (traditional):
  --chunkable                Enable input chunking
  --chunk-type <type>        Chunking type: elements|bytes
  --chunk-size <n>           Size per chunk
  --chunk-output-size <bytes> Output size per chunk (required for chunking)
  --elem-size <bytes>        Element size in bytes (for element chunking)
  --agg <method>             Output aggregation method: concatenate
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

async function getJSON(url) {
  const res = await fetch(url);
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

// Enhanced helper functions
async function readStrategyFile(filePath, strategyType) {
  try {
    const resolvedPath = path.resolve(filePath);
    const strategyCode = await fs.readFile(resolvedPath, 'utf8');

    if (!strategyCode.includes('class') || !strategyCode.includes('extends')) {
      console.warn(`Warning: ${filePath} doesn't appear to contain a class extending Base${strategyType === 'chunking' ? 'Chunking' : 'Assembly'}Strategy`);
    }

    return strategyCode;
  } catch (err) {
    console.error(`Failed to read strategy file ${filePath}: ${err.message}`);
    process.exit(1);
  }
}

async function parseJSON(jsonString, context) {
  if (!jsonString) return {};

  try {
    return JSON.parse(jsonString);
  } catch (err) {
    console.error(`Invalid ${context} JSON: ${err.message}`);
    process.exit(1);
  }
}

async function resolveStrategy(strategyInput, strategyType, base) {
  if (strategyInput.endsWith('.js')) {
    console.log(`Will upload custom ${strategyType} strategy from ${strategyInput}...`);
    return `custom_${path.basename(strategyInput, '.js')}_${Date.now()}`;
  } else {
    return strategyInput;
  }
}

async function generateTestData(arraySize, outputFile, dataType = 'float32') {
  console.log(`Generating ${arraySize} ${dataType} test values...`);

  let buffer;

  if (dataType === 'float32') {
    const testArray = new Float32Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
      testArray[i] = Math.random() * 1000;
    }
    buffer = Buffer.from(testArray.buffer);
  } else if (dataType === 'int32') {
    const testArray = new Int32Array(arraySize);
    for (let i = 0; i < arraySize; i++) {
      testArray[i] = Math.floor(Math.random() * 1000);
    }
    buffer = Buffer.from(testArray.buffer);
  }

  await fs.writeFile(outputFile, buffer);
  console.log(`Test data written to ${outputFile}`);
  return buffer;
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  const args = parseArgs(rest);
  const base = hostBase(args);

  if (!cmd || args.help || args.h) return printHelp();

  // Traditional commands
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

  // Enhanced: Matrix tiled computation
  if (cmd === 'matrix-tiled') {
    const size = parseInt(args.size, 10);
    const tileSize = parseInt(args['tile-size'], 10);

    if (!Number.isInteger(size) || !Number.isInteger(tileSize)) {
      console.error('matrix-tiled: --size and --tile-size are required integers');
      process.exit(1);
    }

    if (tileSize > size) {
      console.error('matrix-tiled: --tile-size cannot be larger than --size');
      process.exit(1);
    }

    const label = args.label || `Tiled Matrix ${size}×${size} (${tileSize}×${tileSize} tiles)`;

    const out = await postJSON(`${base}/api/matrix/tiled-advanced`, {
      matrixSize: size,
      tileSize: tileSize,
      label: label
    });
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  // Enhanced: Strategy upload
  if (cmd === 'strategy-upload') {
    const type = args.type;
    const name = args.name;
    const filePath = args.file;

    if (!type || !['chunking', 'assembly'].includes(type)) {
      console.error('strategy-upload: --type must be "chunking" or "assembly"');
      process.exit(1);
    }

    if (!name) {
      console.error('strategy-upload: --name is required');
      process.exit(1);
    }

    if (!filePath) {
      console.error('strategy-upload: --file is required');
      process.exit(1);
    }

    const strategyCode = await readStrategyFile(filePath, type);

    try {
      const out = await postJSON(`${base}/api/strategies/register`, {
        strategyCode,
        type,
        name
      });
      console.log(JSON.stringify(out, null, 2));
    } catch (err) {
      console.error('Failed to upload strategy:', err.message);
      process.exit(1);
    }
    return;
  }

  // Enhanced: List strategies
  if (cmd === 'strategies-list') {
    try {
      const out = await getJSON(`${base}/api/strategies`);
      console.log(JSON.stringify(out, null, 2));
    } catch (err) {
      console.error('Failed to fetch strategies:', err.message);
      process.exit(1);
    }
    return;
  }

  // Enhanced: Advanced compute workload
  if (cmd === 'compute-advanced') {
    const framework = args.framework;
    if (!framework) {
      console.error('compute-advanced: --framework is required');
      console.error(`Supported frameworks: ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}`);
      process.exit(1);
    }

    await validateFramework(framework);

    if (!args.kernel) {
      console.error('compute-advanced: --kernel <file> is required');
      process.exit(1);
    }

    if (!args.chunking) {
      console.error('compute-advanced: --chunking <strategy> is required');
      process.exit(1);
    }

    if (!args.assembly) {
      console.error('compute-advanced: --assembly <strategy> is required');
      process.exit(1);
    }

    if (!args.metadata) {
      console.error('compute-advanced: --metadata <json> is required');
      process.exit(1);
    }

    if (!args.input) {
      console.error('compute-advanced: --input <file> is required for advanced chunked workloads');
      process.exit(1);
    }

    const kernel = await readKernelFile(args.kernel, framework);
    const metadata = await parseJSON(args.metadata, 'metadata');
    const compilationOptions = await parseJSON(args['compilation-opts'], 'compilation options');

    const chunkingStrategy = await resolveStrategy(args.chunking, 'chunking', base);
    const assemblyStrategy = await resolveStrategy(args.assembly, 'assembly', base);

    // Handle different input formats
    let inputData;
    try {
      if (args.input.endsWith('.json')) {
        // Multi-input JSON format
        const inputContent = await fs.readFile(path.resolve(args.input), 'utf8');
        inputData = inputContent;
      } else {
        // Binary input file
        const inputBuffer = await fs.readFile(path.resolve(args.input));
        inputData = inputBuffer.toString('base64');
      }
    } catch (err) {
      console.error(`Failed to read input file ${args.input}: ${err.message}`);
      process.exit(1);
    }

    const label = args.label || `Advanced ${framework.toUpperCase()} Workload`;
    const entry = args.entry || SUPPORTED_FRAMEWORKS[framework].defaultEntry;

    const payload = {
      label,
      chunkingStrategy,
      assemblyStrategy,
      framework,
      input: inputData,
      metadata: {
        ...metadata,
        customShader: kernel,
        entry,
        compilationOptions
      }
    };

    // Add custom strategy files if provided
    if (args.chunking.endsWith('.js')) {
      payload.customChunkingFile = await readStrategyFile(args.chunking, 'chunking');
    }

    if (args.assembly.endsWith('.js')) {
      payload.customAssemblyFile = await readStrategyFile(args.assembly, 'assembly');
    }

    console.log(`Submitting advanced workload with strategies: ${chunkingStrategy} → ${assemblyStrategy}`);

    try {
      const out = await postJSON(`${base}/api/workloads/advanced`, payload);
      console.log(JSON.stringify(out, null, 2));
    } catch (err) {
      console.error('Failed to submit advanced workload:', err.message);
      process.exit(1);
    }
    return;
  }

  // Enhanced: Iterative sorting
  if (cmd === 'sort') {
    const algorithm = args.algorithm;
    const arraySize = parseInt(args['array-size'], 10);
    const chunkSize = parseInt(args['chunk-size'] || '64', 10);

    if (!algorithm || !['bitonic', 'odd-even', 'sample'].includes(algorithm)) {
      console.error('sort: --algorithm must be bitonic, odd-even, or sample');
      process.exit(1);
    }

    if (!Number.isInteger(arraySize) || arraySize <= 0) {
      console.error('sort: --array-size must be positive integer');
      process.exit(1);
    }

    if (algorithm === 'bitonic' && (arraySize & (arraySize - 1)) !== 0) {
      console.error('sort: bitonic sort requires power-of-2 array size');
      process.exit(1);
    }

    let inputData;
    if (args.input) {
      const buf = await fs.readFile(path.resolve(args.input));
      inputData = buf.toString('base64');
    } else {
      console.log('No input file provided, generating random test data...');
      const testBuffer = await generateTestData(arraySize, `test_array_${arraySize}.bin`);
      inputData = testBuffer.toString('base64');
    }

    const strategyMap = {
      'bitonic': 'bitonic_sort',
      'odd-even': 'odd_even_sort',
      'sample': 'sample_sort'
    };

    const payload = {
      label: args.label || `${algorithm} sort ${arraySize} elements`,
      chunkingStrategy: strategyMap[algorithm],
      assemblyStrategy: 'linear_assembly',
      framework: 'webgpu',
      input: inputData,
      metadata: {
        arraySize,
        chunkSize,
        algorithm
      }
    };

    const out = await postJSON(`${base}/api/workloads/iterative`, payload);
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