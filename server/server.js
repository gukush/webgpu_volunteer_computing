// server.js - Complete Enhanced Version with Multi-Input/Output Support
import express from 'express';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import http from 'http';
import https from 'https';
import { Server as SocketIOServer } from 'socket.io';
import crypto from 'crypto';

// Enhanced: Import chunking system
import { EnhancedChunkingManager } from './strategies/EnhancedChunkingManager.js';

const app = express();

function sha256Hex(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function checksumMatrixRowsFloat32LE(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return sha256Hex(Buffer.alloc(0));
  const cols = Array.isArray(rows[0]) ? rows[0].length : 0;
  const buf = Buffer.allocUnsafe(rows.length * cols * 4);
  let o = 0;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    for (let j = 0; j < cols; j++) {
      buf.writeFloatLE(row[j], o);
      o += 4;
    }
  }
  return sha256Hex(buf);
}

function tallyKByChecksum(submissions, expectedByteLength, k) {
  const voteMap = new Map(); // checksum -> Set(clientId)
  for (const s of submissions) {
    const sizeOk = (expectedByteLength == null) || (s.byteLength === expectedByteLength);
    const selfConsistent = s.serverChecksum && (s.serverChecksum === s.reportedChecksum);
    if (sizeOk && selfConsistent) {
      if (!voteMap.has(s.serverChecksum)) voteMap.set(s.serverChecksum, new Set());
      voteMap.get(s.serverChecksum).add(s.clientId); // same client counted once
    }
  }
  for (const [chk, voters] of voteMap.entries()) {
    if (voters.size >= k) return { ok: true, winningChecksum: chk, voters };
  }
  return { ok: false };
}

// NEW: Handle multi-result checksums (concatenate all outputs for per-chunk verification)
function checksumFromResults(results) {
  // results is array of base64 strings
  if (!Array.isArray(results)) {
    results = [results]; // Backward compatibility
  }

  const buffers = results.map(b64 => Buffer.from(b64, 'base64'));
  const combinedBuf = Buffer.concat(buffers);

  return {
    serverChecksum: sha256Hex(combinedBuf),
    byteLength: combinedBuf.length,
    buffer: combinedBuf,
    buffers: buffers // Keep individual buffers for assembly
  };
}

function checksumFromBase64(base64) {
  const buf = Buffer.from(base64, 'base64');
  return { serverChecksum: sha256Hex(buf), byteLength: buf.length, buffer: buf };
}

const SUPPORTED_FRAMEWORKS = {
  'webgpu': {
    browserBased: true,
    shaderExtension: '.wgsl',
    defaultBindLayout: 'storage-in-storage-out'
  },
  'webgl': {
    browserBased: true,
    shaderExtension: '.glsl',
    defaultBindLayout: 'webgl-transform-feedback'
  },
  'cuda': {
    browserBased: false,
    shaderExtension: '.cu',
    defaultBindLayout: 'cuda-global-memory'
  },
  'opencl': {
    browserBased: false,
    shaderExtension: '.cl',
    defaultBindLayout: 'opencl-global-memory'
  },
  'vulkan': {
    browserBased: false,
    shaderExtension: '.comp',
    defaultBindLayout: 'vulkan-storage-buffer'
  }
};

// Enhanced: Initialize chunking manager
const chunkingManager = new EnhancedChunkingManager();

let server;
let useHttps = false;
const PORT = process.env.PORT || 3000;

try {
  const privateKey = fs.readFileSync(path.join(path.resolve(), 'certificates/key.pem'), 'utf8');
  const certificate = fs.readFileSync(path.join(path.resolve(), 'certificates/cert.pem'), 'utf8');
  const credentials = { key: privateKey, cert: certificate };
  server = https.createServer(credentials, app);
  useHttps = true;
  console.log('Using HTTPS server with self-signed certificates');
} catch (error) {
  console.warn('SSL certificates not found or unreadable, falling back to HTTP. Error:', error.message);
  server = http.createServer(app);
}

const io = new SocketIOServer(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
    credentials: true
  }
});

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

app.use(express.static(path.join(path.resolve(), 'public')));
app.use(express.json({ limit: '50mb' }));

let ADMIN_K_PARAMETER = 1;

const matrixState = {
  isRunning: false,
  problem: null,
  tasks: [],
  activeTasks: new Map(),
  completedTasks: new Map(),
  clients: new Map(),
  startTime: null,
  endTime: null,
  stats: { totalClients: 0, activeClients: 0, completedTasks: 0, totalTasks: 0 }
};

const matrixResultBuffer = new Map();
const MATRIX_TASK_TIMEOUT = 3 * 60 * 1000;

const customWorkloads = new Map();
const customWorkloadChunks = new Map();
const pendingCustomChunks = [];
const CUSTOM_TASKS_FILE = 'custom_tasks.json';
const CUSTOM_CHUNK_TIMEOUT = 5 * 60 * 1000;

function saveCustomWorkloads() {
  const workloadsArray = Array.from(customWorkloads.values()).map(wl => {
    const workloadToSave = { ...wl };
    if (workloadToSave.activeAssignments instanceof Set) {
      workloadToSave.activeAssignments = Array.from(workloadToSave.activeAssignments);
    }
    return workloadToSave;
  });
  fs.writeFile(CUSTOM_TASKS_FILE, JSON.stringify(workloadsArray, null, 2), err => {
    if (err) console.error('Error saving custom workloads:', err);
    else console.log(`Custom workloads saved to ${CUSTOM_TASKS_FILE}`);
  });
}

function loadCustomWorkloads() {
  try {
    if (fs.existsSync(CUSTOM_TASKS_FILE)) {
      const data = fs.readFileSync(CUSTOM_TASKS_FILE, 'utf8');
      const workloadsArray = JSON.parse(data);
      workloadsArray.forEach(workload => {
        if (workload.status === undefined) {
          if (workload.finalResult) workload.status = 'complete';
          else if (workload.results && workload.results.length > 0) workload.status = 'processing';
          else workload.status = 'queued';
        }
        workload.chunkable = workload.chunkable || false;
        workload.isChunkParent = workload.isChunkParent || workload.chunkable;
        workload.processingTimes = workload.processingTimes || [];
        workload.results = workload.results || [];
        workload.dispatchesMade = workload.dispatchesMade || 0;
        workload.activeAssignments = new Set(workload.activeAssignments || []);

        // NEW: Handle backward compatibility for output sizes
        if (!workload.outputSizes && workload.outputSize) {
          workload.outputSizes = [workload.outputSize];
        }
        if (!workload.chunkOutputSizes && workload.chunkOutputSize) {
          workload.chunkOutputSizes = [workload.chunkOutputSize];
        }

        customWorkloads.set(workload.id, workload);
      });
      console.log(`Loaded ${customWorkloads.size} custom workloads from ${CUSTOM_TASKS_FILE}`);
    }
  } catch (err) {
    console.error('Error loading custom workloads:', err);
  }
}

function broadcastCustomWorkloadList() {
  io.emit('workloads:list_update', Array.from(customWorkloads.values()));
}

// Enhanced: Advanced workload API with pluggable strategies
app.post('/api/workloads/advanced', async (req, res) => {
  const {
    label,
    chunkingStrategy,
    assemblyStrategy,
    framework = 'webgpu',
    input,
    metadata,
    customShader,
    customChunkingFile,
    customAssemblyFile
  } = req.body;

  try {
    // Load custom strategies if provided
    if (customChunkingFile) {
      const result = chunkingManager.registerCustomStrategy(
        customChunkingFile,
        'chunking',
        chunkingStrategy
      );
      if (!result.success) {
        return res.status(400).json({ error: `Custom chunking strategy failed: ${result.error}` });
      }
    }

    if (customAssemblyFile) {
      const result = chunkingManager.registerCustomStrategy(
        customAssemblyFile,
        'assembly',
        assemblyStrategy
      );
      if (!result.success) {
        return res.status(400).json({ error: `Custom assembly strategy failed: ${result.error}` });
      }
    }

    const workloadId = uuidv4();

    const enhancedWorkload = {
      id: workloadId,
      label: label || `Enhanced Workload ${workloadId.substring(0, 6)}`,
      chunkingStrategy,
      assemblyStrategy,
      framework,
      input,
      metadata: {
        ...metadata,
        customShader
      },
      createdAt: Date.now()
    };

    const result = await chunkingManager.processChunkedWorkload(enhancedWorkload);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    customWorkloads.set(workloadId, {
      ...enhancedWorkload,
      status: 'chunking_ready',
      isChunkParent: true,
      enhanced: true,
      chunkDescriptors: result.chunkDescriptors,
      plan: result.plan
    });

    const chunksForParent = {
      parentId: workloadId,
      allChunkDefs: result.chunkDescriptors.map((desc, index) => ({
        ...desc,
        status: 'queued',
        dispatchesMade: 0,
        submissions: [],
        activeAssignments: new Set(),
        verified_results: null // Will be array of base64 strings
      })),
      completedChunksData: new Map(),
      expectedChunks: result.plan.totalChunks,
      status: 'awaiting_start',
      aggregationMethod: result.plan.assemblyStrategy,
      enhanced: true
    };

    customWorkloadChunks.set(workloadId, chunksForParent);

    res.json({
      success: true,
      id: workloadId,
      message: `Enhanced workload created with ${result.plan.totalChunks} chunks`,
      strategy: chunkingStrategy,
      assemblyStrategy: assemblyStrategy,
      chunks: result.plan.totalChunks
    });

  } catch (error) {
    console.error('Error creating advanced workload:', error);
    res.status(500).json({ error: error.message });
  }
});

// Enhanced: Strategy registration API
app.post('/api/strategies/register', (req, res) => {
  const { strategyCode, type, name } = req.body;

  if (!strategyCode || !type || !name) {
    return res.status(400).json({
      success: false,
      error: 'strategyCode, type, and name are required'
    });
  }

  if (!['chunking', 'assembly'].includes(type)) {
    return res.status(400).json({
      success: false,
      error: 'type must be "chunking" or "assembly"'
    });
  }

  try {
    const result = chunkingManager.registerCustomStrategy(strategyCode, type, name);

    if (result.success) {
      res.json({
        success: true,
        message: `Custom ${type} strategy '${name}' registered successfully`,
        strategyName: result.strategyName
      });
    } else {
      res.status(400).json(result);
    }
  } catch (error) {
    res.status(400).json({
      success: false,
      error: `Failed to register strategy: ${error.message}`
    });
  }
});

// Enhanced: List available strategies
app.get('/api/strategies', (req, res) => {
  const strategies = chunkingManager.getAvailableStrategies();

  res.json({
    available: strategies,
    examples: {
      matrix_tiled: {
        description: "Divide matrix computation into rectangular tiles",
        requiredMetadata: ["matrixSize", "tileSize"],
        optionalMetadata: ["workgroupSize"],
        example: {
          chunkingStrategy: "matrix_tiled",
          assemblyStrategy: "matrix_tiled_assembly",
          metadata: { matrixSize: 512, tileSize: 64 }
        }
      },
      linear: {
        description: "Linear element-wise processing",
        requiredMetadata: ["elementSize", "chunkSize"],
        example: {
          chunkingStrategy: "linear",
          assemblyStrategy: "linear_assembly",
          metadata: { elementSize: 4, chunkSize: 1024 }
        }
      }
    }
  });
});

// Enhanced: Tiled matrix API
app.post('/api/matrix/tiled-advanced', (req, res) => {
  const { matrixSize, tileSize, label } = req.body || {};

  if (!matrixSize || !tileSize || matrixSize <= 0 || tileSize <= 0) {
    return res.status(400).json({
      error: 'matrixSize and tileSize must be positive integers'
    });
  }

  try {
    // Generate random matrices A and B
    const A = generateRandomMatrix(matrixSize);
    const B = generateRandomMatrix(matrixSize);

    // Serialize matrices in the format expected by the tiled strategy
    const matrixBuffer = new ArrayBuffer(4 + matrixSize * matrixSize * 8);
    const view = new DataView(matrixBuffer);

    // Header: matrix size
    view.setUint32(0, matrixSize, true);

    // Matrix A data
    const aOffset = 4;
    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        const idx = aOffset + (i * matrixSize + j) * 4;
        view.setFloat32(idx, A[i][j], true);
      }
    }

    // Matrix B data
    const bOffset = 4 + matrixSize * matrixSize * 4;
    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        const idx = bOffset + (i * matrixSize + j) * 4;
        view.setFloat32(idx, B[i][j], true);
      }
    }

    const inputBase64 = Buffer.from(matrixBuffer).toString('base64');

    const workloadRequest = {
      id: uuidv4(),
      label: label || `Tiled Matrix Multiplication ${matrixSize}×${matrixSize} (${tileSize}×${tileSize} tiles)`,
      chunkingStrategy: 'matrix_tiled',
      assemblyStrategy: 'matrix_tiled_assembly',
      framework: 'webgpu',
      input: inputBase64,
      metadata: {
        matrixSize,
        tileSize,
        operation: 'matrix_multiply'
      }
    };

    // Process with chunking manager
    chunkingManager.processChunkedWorkload(workloadRequest)
      .then(result => {
        if (result.success) {
          const tilesPerDim = Math.ceil(matrixSize / tileSize);
          res.json({
            success: true,
            id: workloadRequest.id,
            message: `Tiled matrix multiplication started: ${tilesPerDim}×${tilesPerDim} = ${tilesPerDim * tilesPerDim} tiles`,
            matrixSize,
            tileSize,
            totalTiles: tilesPerDim * tilesPerDim
          });
        } else {
          res.status(400).json(result);
        }
      })
      .catch(error => {
        console.error('Error creating tiled matrix workload:', error);
        res.status(500).json({ error: 'Internal server error' });
      });

  } catch (error) {
    console.error('Error in tiled matrix API:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/workloads', (req, res) => {
  const {
    label,
    framework = 'webgpu',
    kernel,
    wgsl,
    entry = 'main',
    workgroupCount,
    bindLayout,
    input,
    outputSize, // Backward compatibility
    outputSizes, // NEW: Array of output sizes
    chunkable = false,
    inputChunkProcessingType = 'elements',
    inputChunkSize,
    chunkOutputSize, // Backward compatibility
    chunkOutputSizes, // NEW: Array of chunk output sizes
    inputElementSizeBytes = 4,
    outputAggregationMethod = 'concatenate',
    compilationOptions = {}
  } = req.body;

  if (!SUPPORTED_FRAMEWORKS[framework]) {
    return res.status(400).json({
      error: `Unsupported framework: ${framework}. Supported: ${Object.keys(SUPPORTED_FRAMEWORKS).join(', ')}`
    });
  }

  const kernelCode = kernel || wgsl;
  if (!kernelCode || !workgroupCount) {
    return res.status(400).json({
      error: 'Missing required fields: kernel/wgsl, workgroupCount'
    });
  }

  // NEW: Handle output sizes (backward compatibility)
  let finalOutputSizes = outputSizes;
  if (!finalOutputSizes && outputSize) {
    finalOutputSizes = [outputSize];
  }
  if (!finalOutputSizes || finalOutputSizes.length === 0) {
    return res.status(400).json({
      error: 'outputSizes (or outputSize for backward compatibility) is required'
    });
  }

  // NEW: Validate max inputs/outputs
  if (finalOutputSizes.length > 3) {
    return res.status(400).json({
      error: 'Maximum 3 outputs supported'
    });
  }

  // NEW: Parse inputs if provided
  let parsedInputs = {};
  if (input) {
    try {
      // Try to parse as JSON (multi-input format)
      if (typeof input === 'string' && input.startsWith('{')) {
        parsedInputs = JSON.parse(input);
      } else {
        // Single input (backward compatibility)
        parsedInputs = { input: input };
      }
    } catch (e) {
      parsedInputs = { input: input }; // Fallback to single input
    }
  }

  // NEW: Validate max inputs
  if (Object.keys(parsedInputs).length > 4) {
    return res.status(400).json({
      error: 'Maximum 4 inputs supported'
    });
  }

  if (chunkable) {
    // NEW: Handle chunk output sizes (backward compatibility)
    let finalChunkOutputSizes = chunkOutputSizes;
    if (!finalChunkOutputSizes && chunkOutputSize) {
      finalChunkOutputSizes = [chunkOutputSize];
    }
    if (!finalChunkOutputSizes || finalChunkOutputSizes.length === 0) {
      return res.status(400).json({
        error: 'chunkOutputSizes (or chunkOutputSize for backward compatibility) is required for chunkable workloads'
      });
    }
    if (finalChunkOutputSizes.length !== finalOutputSizes.length) {
      return res.status(400).json({
        error: 'chunkOutputSizes must match number of outputs'
      });
    }
  }

  const effectiveBindLayout = bindLayout || SUPPORTED_FRAMEWORKS[framework].defaultBindLayout;

  const id = uuidv4();
  const workloadMeta = {
    id,
    label: label || `${framework.toUpperCase()} Workload ${id.substring(0, 6)}`,
    framework,
    kernel: kernelCode,
    wgsl: framework === 'webgpu' ? kernelCode : undefined,
    entry,
    workgroupCount,
    bindLayout: effectiveBindLayout,
    input: Object.keys(parsedInputs).length > 0 ? JSON.stringify(parsedInputs) : undefined,
    outputSizes: finalOutputSizes, // NEW: Array
    status: 'queued',
    results: [],
    processingTimes: [],
    createdAt: Date.now(),
    chunkable,
    inputChunkProcessingType,
    inputChunkSize,
    chunkOutputSizes: chunkable ? (chunkOutputSizes || [chunkOutputSize]) : undefined, // NEW: Array
    inputElementSizeBytes,
    outputAggregationMethod,
    isChunkParent: chunkable,
    dispatchesMade: 0,
    activeAssignments: new Set(),
    compilationOptions,

    // Backward compatibility fields
    outputSize: finalOutputSizes[0],
    chunkOutputSize: chunkable ? (chunkOutputSizes?.[0] || chunkOutputSize) : undefined
  };

  if (framework === 'cuda' && !compilationOptions.deviceId) {
    compilationOptions.deviceId = 0;
  }

  customWorkloads.set(id, workloadMeta);

  if (chunkable) {
    const prepResult = prepareAndQueueChunks(workloadMeta);
    if (!prepResult.success) {
      customWorkloads.delete(id);
      return res.status(400).json({ error: `Chunk preparation failed: ${prepResult.error}` });
    }
  }

  saveCustomWorkloads();
  broadcastCustomWorkloadList();
  console.log(`📡 ${framework.toUpperCase()} workload ${id} (${workloadMeta.label}) queued.`);
  res.json({ ok: true, id, message: `${framework.toUpperCase()} workload "${workloadMeta.label}" queued.` });
});

app.post('/api/matrix/start', (req, res) => {
  const { matrixSize, chunkSize } = req.body || {};
  if (!Number.isInteger(matrixSize) || !Number.isInteger(chunkSize) || matrixSize <= 0 || chunkSize <= 0) {
    return res.status(400).json({ error: 'matrixSize and chunkSize must be positive integers' });
  }
  const problem = prepareMatrixMultiplication(matrixSize, chunkSize);
  res.json({
    ok: true,
    problem: { size: problem.size, chunkSize: problem.chunkSize },
    totalTasks: matrixState.stats.totalTasks
  });
});

app.post('/api/system/k', (req, res) => {
  const { k } = req.body || {};
  if (!Number.isInteger(k) || k < 1) return res.status(400).json({ error: 'k must be integer ≥ 1' });
  ADMIN_K_PARAMETER = k;
  io.emit('admin:k_update', ADMIN_K_PARAMETER);
  res.json({ ok: true, k: ADMIN_K_PARAMETER });
});

app.post('/api/workloads/startQueued', (req, res) => {
  let startedNonChunked = 0, activatedChunkParents = 0;
  customWorkloads.forEach(wl => {
    if (wl.status === 'queued') {
      wl.startedAt = Date.now();
      if (wl.isChunkParent) {
        const prep = prepareAndQueueChunks(wl);
        if (!prep.success) return;
        const store = customWorkloadChunks.get(wl.id);
        wl.status = 'assigning_chunks';
        store.status = 'assigning_chunks';
        store.allChunkDefs.forEach(cd => {
          cd.status = 'queued';
          cd.dispatchesMade = 0;
          cd.submissions = [];
          cd.activeAssignments.clear();
          cd.verified_results = null;
        });
        io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
        activatedChunkParents++;
      } else {
        wl.status = 'pending_dispatch';
        startedNonChunked++;
      }
    }
  });
  saveCustomWorkloads();
  broadcastCustomWorkloadList();
  res.json({ ok: true, activatedChunkParents, startedNonChunked });
});

app.delete('/api/workloads/:id', (req, res) => {
  const { id } = req.params;
  if (!customWorkloads.has(id)) return res.status(404).json({ error: 'Not found' });
  customWorkloadChunks.delete(id);
  customWorkloads.delete(id);
  saveCustomWorkloads();
  broadcastCustomWorkloadList();
  res.json({ ok: true });
});

app.get('/api/status', (req, res) => {
  res.json({
    k: ADMIN_K_PARAMETER,
    matrix: {
      isRunning: matrixState.isRunning,
      stats: matrixState.stats,
      problem: matrixState.problem ? { size: matrixState.problem.size, chunkSize: matrixState.problem.chunkSize } : null
    },
    workloads: Array.from(customWorkloads.values()).map(w => ({
      id: w.id, label: w.label, status: w.status, isChunkParent: w.isChunkParent, startedAt: w.startedAt, completedAt: w.completedAt
    })),
    clients: Array.from(matrixState.clients.values()).map(c => ({
      id: c.id, connected: c.connected, completedTasks: c.completedTasks, usingCpu: c.usingCpu, hasWebGPU: c.hasWebGPU
    }))
  });
});

app.get('/api/frameworks', (req, res) => {
  const stats = {};

  Object.keys(SUPPORTED_FRAMEWORKS).forEach(framework => {
    const clients = Array.from(matrixState.clients.values())
      .filter(c => c.supportedFrameworks && c.supportedFrameworks.includes(framework));

    const workloads = Array.from(customWorkloads.values())
      .filter(w => w.framework === framework);

    stats[framework] = {
      availableClients: clients.length,
      activeWorkloads: workloads.filter(w => w.status !== 'complete').length,
      completedWorkloads: workloads.filter(w => w.status === 'complete').length
    };
  });

  res.json({ frameworks: SUPPORTED_FRAMEWORKS, stats });
});

function prepareAndQueueChunks(parentWorkload) {
  const parentId = parentWorkload.id;

  if (customWorkloadChunks.has(parentId)) {
    customWorkloadChunks.delete(parentId);
  }

  // NEW: Parse input data (single or multi-input)
  let parsedInputs = {};
  if (parentWorkload.input) {
    try {
      if (typeof parentWorkload.input === 'string' && parentWorkload.input.startsWith('{')) {
        parsedInputs = JSON.parse(parentWorkload.input);
      } else {
        parsedInputs = { input: parentWorkload.input };
      }
    } catch (e) {
      parsedInputs = { input: parentWorkload.input };
    }
  }

  // Calculate chunk info based on the first input (or assume empty if no inputs)
  const firstInputKey = Object.keys(parsedInputs)[0];
  const inputData = firstInputKey ? Buffer.from(parsedInputs[firstInputKey], 'base64') : Buffer.alloc(0);
  const totalInputBytes = inputData.length;

  let actualChunkSizeBytes;
  if (parentWorkload.inputChunkProcessingType === 'elements') {
    actualChunkSizeBytes = parentWorkload.inputChunkSize * parentWorkload.inputElementSizeBytes;
  } else {
    actualChunkSizeBytes = parentWorkload.inputChunkSize;
  }

  if (actualChunkSizeBytes <= 0 && parentWorkload.chunkable && totalInputBytes > 0) {
    return { success: false, error: `Invalid actualChunkSizeBytes ${actualChunkSizeBytes} for ${parentId}` };
  }
  const numChunks = (totalInputBytes > 0 && actualChunkSizeBytes > 0)
    ? Math.ceil(totalInputBytes / actualChunkSizeBytes)
    : 0;

  const chunksForParent = {
    parentId,
    allChunkDefs: [],
    completedChunksData: new Map(),
    expectedChunks: numChunks,
    status: 'awaiting_start',
    aggregationMethod: parentWorkload.outputAggregationMethod,
    finalOutputSizes: parentWorkload.outputSizes, // NEW: Array
    chunkOutputSizes: parentWorkload.chunkOutputSizes // NEW: Array
  };

  for (let i = 0; i < numChunks; i++) {
    const chunkId = `${parentId}-chunk-${i}`;
    const byteOffset = i * actualChunkSizeBytes;
    const currentChunkByteLength = Math.min(actualChunkSizeBytes, totalInputBytes - byteOffset);
    if (currentChunkByteLength <= 0) continue;

    // NEW: Create input chunks for all inputs
    const chunkInputs = {};
    for (const [inputName, inputBase64] of Object.entries(parsedInputs)) {
      const inputBuffer = Buffer.from(inputBase64, 'base64');
      const chunkInputSlice = inputBuffer.slice(byteOffset, byteOffset + currentChunkByteLength);
      chunkInputs[inputName] = chunkInputSlice.toString('base64');
    }

    const chunkDef = {
      parentId,
      chunkId,
      chunkOrderIndex: i,
      status: 'queued',
      framework: parentWorkload.framework,
      kernel: parentWorkload.kernel,
      wgsl: parentWorkload.wgsl,
      entry: parentWorkload.entry,
      workgroupCount: parentWorkload.workgroupCount,
      bindLayout: parentWorkload.bindLayout,
      outputSizes: parentWorkload.chunkOutputSizes, // NEW: Array for this chunk
      inputData: Object.keys(chunkInputs).length === 1 ? Object.values(chunkInputs)[0] : JSON.stringify(chunkInputs), // Backward compatibility
      inputs: Object.keys(chunkInputs).length > 1 ? Object.values(chunkInputs) : [Object.values(chunkInputs)[0] || ''], // NEW: Array for client
      chunkUniforms: {},
      dispatchesMade: 0,
      submissions: [],
      activeAssignments: new Set(),
      assignedClients: new Set(),
      verified_results: null, // Will be array of base64 strings
      compilationOptions: parentWorkload.compilationOptions
    };

    if (parentWorkload.inputChunkProcessingType === 'elements') {
      chunkDef.chunkUniforms.chunkOffsetElements = Math.floor(byteOffset / parentWorkload.inputElementSizeBytes);
      chunkDef.chunkUniforms.chunkInputSizeElements = Math.floor(currentChunkByteLength / parentWorkload.inputElementSizeBytes);
      chunkDef.chunkUniforms.totalOriginalInputSizeElements = Math.floor(totalInputBytes / parentWorkload.inputElementSizeBytes);
    }
    chunkDef.chunkUniforms.chunkOffsetBytes = byteOffset;
    chunkDef.chunkUniforms.chunkInputSizeBytes = currentChunkByteLength;
    chunkDef.chunkUniforms.totalOriginalInputSizeBytes = totalInputBytes;

    chunksForParent.allChunkDefs.push(chunkDef);
  }

  customWorkloadChunks.set(parentId, chunksForParent);
  console.log(`Workload ${parentId}: Prepared ${chunksForParent.allChunkDefs.length} chunks with ${parentWorkload.outputSizes?.length || 1} outputs each.`);
  return { success: true };
}

function generateRandomMatrix(size) {
  const matrix = new Array(size);
  for (let i = 0; i < size; i++) {
    matrix[i] = new Array(size);
    for (let j = 0; j < size; j++) {
      matrix[i][j] = Math.random();
    }
  }
  return matrix;
}

function prepareMatrixMultiplication(size, chunkSize) {
  console.log(`Preparing matrix multiplication: ${size}×${size}, chunk ${chunkSize}`);
  matrixState.activeTasks.clear();
  matrixState.completedTasks.clear();
  matrixResultBuffer.clear();
  matrixState.tasks = [];
  matrixState.stats.completedTasks = 0;
  matrixState.stats.totalTasks = 0;
  matrixState.startTime = Date.now();

  const A = generateRandomMatrix(size);
  const B = generateRandomMatrix(size);
  matrixState.problem = { type: 'matrixMultiply', matrixA: A, matrixB: B, size, chunkSize };

  const numChunks = Math.ceil(size / chunkSize);
  for (let i = 0; i < numChunks; i++) {
    const startRow = i * chunkSize;
    const endRow = Math.min((i + 1) * chunkSize, size);
    matrixState.tasks.push({
      id: `task-${i}`, startRow, endRow, status: 'pending', dispatchesMade: 0
    });
  }

  matrixState.stats.totalTasks = matrixState.tasks.length;
  matrixState.isRunning = true;
  console.log(`Created ${matrixState.tasks.length} matrix tasks`);
  broadcastStatus();
  return matrixState.problem;
}

function assignMatrixTask(clientId) {
  if (!matrixState.isRunning) return null;
  const client = matrixState.clients.get(clientId);
  if (!client || client.isBusyWithMatrixTask || client.isBusyWithCustomChunk) return null;

  for (const t of matrixState.tasks) {
    if (t.status === 'pending' && t.dispatchesMade < ADMIN_K_PARAMETER) {
      const already = Array.from(matrixState.activeTasks.values())
        .some(a => a.logicalTaskId === t.id && a.assignedTo === clientId);
      if (already) continue;

      t.dispatchesMade++;
      const instanceId = uuidv4();
      matrixState.activeTasks.set(instanceId, {
        logicalTaskId: t.id,
        assignedTo: clientId,
        startTime: Date.now()
      });
      client.isBusyWithMatrixTask = true;
      console.log(`Dispatching matrix task ${t.id} (#${t.dispatchesMade}/${ADMIN_K_PARAMETER}) to ${clientId}`);
      return {
        assignmentId: instanceId,
        id: t.id,
        startRow: t.startRow,
        endRow: t.endRow,
        matrixA: matrixState.problem.matrixA,
        matrixB: matrixState.problem.matrixB,
        size: matrixState.problem.size,
        type: 'matrixMultiply'
      };
    }
  }
  return null;
}

function processMatrixTaskResult(logicalTaskId, verifiedResultData, contributingClientIds) {
  const taskDef = matrixState.tasks.find(t => t.id === logicalTaskId);
  if (!taskDef || taskDef.status === 'completed') return false;

  taskDef.status = 'completed';
  taskDef.result = verifiedResultData;

  const submissions = matrixResultBuffer.get(logicalTaskId) || [];
  const rep = submissions.find(s => JSON.stringify(s.result) === JSON.stringify(verifiedResultData)) || {};

  matrixState.completedTasks.set(logicalTaskId, {
    id: logicalTaskId,
    startRow: taskDef.startRow,
    endRow: taskDef.endRow,
    status: 'completed',
    result: verifiedResultData,
    assignedTo: contributingClientIds.join(', '),
    processingTime: rep.processingTime || 0,
    verifiedAt: Date.now()
  });
  matrixState.stats.completedTasks++;
  console.log(`Matrix ${logicalTaskId} verified (${matrixState.stats.completedTasks}/${matrixState.stats.totalTasks})`);

  contributingClientIds.forEach(cid => {
    const cl = matrixState.clients.get(cid);
    if (cl) {
      cl.completedTasks = (cl.completedTasks || 0) + 1;
      cl.lastActive = Date.now();
    }
  });

  if (matrixState.stats.completedTasks === matrixState.stats.totalTasks && matrixState.isRunning) {
    finalizeMatrixComputation();
  }
  return true;
}

function finalizeMatrixComputation() {
  matrixState.endTime = Date.now();
  matrixState.isRunning = false;
  const elapsed = (matrixState.endTime - matrixState.startTime) / 1000;

  const size = matrixState.problem.size;
  const full = Array.from({ length: size }, () => Array(size).fill(0));
  for (const ct of matrixState.completedTasks.values()) {
    const rows = ct.result;
    for (let i = ct.startRow; i < ct.endRow; i++) {
      full[i] = rows[i - ct.startRow];
    }
  }

  io.emit('computation:complete', {
    type: 'matrixMultiply',
    totalTime: elapsed,
    result: full
  });
  broadcastStatus();
  console.log(`Matrix computation done in ${elapsed.toFixed(2)}s`);
}

function broadcastStatus() {
  const elapsed = matrixState.startTime
    ? (Date.now() - matrixState.startTime) / 1000
    : 0;
  io.emit('state:update', {
    stats: matrixState.stats,
    elapsedTime: elapsed
  });
}

function assignCustomChunkToAvailableClients() {
  const availableClients = Array.from(matrixState.clients.entries()).filter(([clientId, client]) =>
    client.connected &&
    client.gpuInfo &&
    !client.isBusyWithCustomChunk &&
    !client.isBusyWithMatrixTask &&
    client.socket
  );

  for (const [clientId, client] of availableClients) {
    for (const parent of customWorkloads.values()) {
      if (!parent.isChunkParent || !['assigning_chunks', 'processing_chunks'].includes(parent.status)) continue;

      if (!client.supportedFrameworks.includes(parent.framework)) {
        continue;
      }

      const store = customWorkloadChunks.get(parent.id);
      if (!store) continue;

      for (const cd of store.allChunkDefs) {
        if (
          cd.status !== 'completed' &&
          !cd.verified_results &&
          cd.dispatchesMade < ADMIN_K_PARAMETER &&
          !cd.assignedClients.has(clientId)
        ) {
          cd.dispatchesMade++;
          cd.activeAssignments.add(clientId);
          cd.assignedClients.add(clientId);
          cd.status = 'active';
          cd.assignedTo = clientId;
          cd.assignedAt = Date.now();
          client.isBusyWithCustomChunk = true;

          const taskData = {
            ...cd,
            framework: parent.framework,
            compilationOptions: parent.compilationOptions,
            enhanced: parent.enhanced,
            // NEW: Send inputs and outputSizes as arrays
            inputs: cd.inputs || [cd.inputData || ''],
            outputSizes: cd.outputSizes || [cd.outputSize]
          };

          client.socket.emit('workload:chunk_assign', taskData);
          console.log(`Assigned ${parent.framework} chunk ${cd.chunkId} to ${clientId} (${cd.inputs?.length || 1} inputs, ${cd.outputSizes?.length || 1} outputs)`);
          break;
        }
      }

      if (client.isBusyWithCustomChunk) break;
    }
  }
}

function tryDispatchNonChunkedWorkloads() {
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (!client.connected || !client.gpuInfo || client.isBusyWithCustomChunk ||
        client.isBusyWithMatrixTask || client.isBusyWithNonChunkedWGSL || !client.socket) {
      continue;
    }

    for (const wl of customWorkloads.values()) {
      if (!wl.isChunkParent && ['pending_dispatch', 'pending'].includes(wl.status)
        && !wl.finalResult && wl.dispatchesMade < ADMIN_K_PARAMETER
        && !wl.activeAssignments.has(clientId)) {

        if (!client.supportedFrameworks.includes(wl.framework)) {
          continue;
        }

        wl.dispatchesMade++;
        wl.activeAssignments.add(clientId);
        client.isBusyWithNonChunkedWGSL = true;
        console.log(`Dispatching ${wl.framework} workload ${wl.label} to ${clientId}`);

        // NEW: Parse inputs for non-chunked workloads
        let parsedInputs = {};
        if (wl.input) {
          try {
            if (typeof wl.input === 'string' && wl.input.startsWith('{')) {
              parsedInputs = JSON.parse(wl.input);
            } else {
              parsedInputs = { input: wl.input };
            }
          } catch (e) {
            parsedInputs = { input: wl.input };
          }
        }

        const taskData = {
          ...wl,
          compilationOptions: wl.compilationOptions,
          // NEW: Send as arrays for consistency
          inputs: Object.values(parsedInputs),
          outputSizes: wl.outputSizes || [wl.outputSize]
        };

        client.socket.emit('workload:new', taskData);
        break;
      }
    }
  }
}

function handleClientDisconnect(clientId) {
  const client = matrixState.clients.get(clientId);
  if (client) {
    client.connected = false;
    client.isBusyWithMatrixTask = false;
    client.isBusyWithCustomChunk = false;
    client.isBusyWithNonChunkedWGSL = false;
  }

  for (const [assignId, inst] of matrixState.activeTasks.entries()) {
    if (inst.assignedTo === clientId) {
      console.log(`Matrix assignment ${assignId} for ${clientId} timed out on disconnect.`);
      matrixState.activeTasks.delete(assignId);
    }
  }

  customWorkloadChunks.forEach(store => {
    store.allChunkDefs.forEach(cd => {
      if (cd.activeAssignments.has(clientId)) {
        cd.activeAssignments.delete(clientId);
        if (cd.assignedTo === clientId) {
          cd.assignedTo = null;
          cd.assignedAt = null;
        }
        if (!cd.verified_results && cd.activeAssignments.size === 0 && cd.status === 'active') {
          cd.status = 'queued';
        }
      }
    });
  });

  customWorkloads.forEach(wl => {
    if (!wl.isChunkParent && wl.activeAssignments.has(clientId)) {
      wl.activeAssignments.delete(clientId);
    }
  });

  matrixState.clients.delete(clientId);
  matrixState.stats.activeClients = matrixState.clients.size;
}

function checkTaskTimeouts() {
  const now = Date.now();

  for (const [assignId, inst] of matrixState.activeTasks.entries()) {
    if (now - inst.startTime > MATRIX_TASK_TIMEOUT) {
      console.log(`Matrix assignment ${assignId} timed out.`);
      const cl = matrixState.clients.get(inst.assignedTo);
      if (cl) cl.isBusyWithMatrixTask = false;
      matrixState.activeTasks.delete(assignId);
    }
  }

  customWorkloadChunks.forEach(store => {
    store.allChunkDefs.forEach(cd => {
      if (cd.activeAssignments.size > 0 && cd.assignedAt && (now - cd.assignedAt > CUSTOM_CHUNK_TIMEOUT) && cd.assignedTo) {
        const timedOutClient = cd.assignedTo;
        console.log(`Chunk ${cd.chunkId} for ${cd.parentId} timed out on ${timedOutClient}`);
        const cl = matrixState.clients.get(timedOutClient);
        if (cl) cl.isBusyWithCustomChunk = false;
        cd.activeAssignments.delete(timedOutClient);
        cd.assignedTo = null;
        cd.assignedAt = null;
        const parent = customWorkloads.get(cd.parentId);
        if (parent) {
          parent.processingTimes.push({
            chunkId: cd.chunkId,
            error: 'timeout',
            assignedTo: timedOutClient,
            timedOutAt: now
          });
        }
      }
    });
  });
}

setInterval(() => {
  checkTaskTimeouts();
}, 30000);

io.on('connection', socket => {
  console.log(`New client: ${socket.id}`);
  matrixState.clients.set(socket.id, {
    id: socket.id,
    socket,
    connected: true,
    joinedAt: Date.now(),
    lastActive: Date.now(),
    completedTasks: 0,
    gpuInfo: null,
    supportedFrameworks: [],
    isPuppeteer: socket.handshake.query.mode === 'headless',
    clientType: 'browser',
    isBusyWithMatrixTask: false,
    isBusyWithCustomChunk: false,
    isBusyWithNonChunkedWGSL: false
  });
  matrixState.stats.totalClients++;
  matrixState.stats.activeClients = matrixState.clients.size;

  socket.emit('register', { clientId: socket.id });
  socket.emit('admin:k_update', ADMIN_K_PARAMETER);

  socket.on('client:join', (data) => {
    const c = matrixState.clients.get(socket.id);
    if (!c) return;
    if (c.hasJoined) return;

    c.hasJoined = true;
    c.gpuInfo = data.gpuInfo;
    c.hasWebGPU = !!data.hasWebGPU;
    c.supportedFrameworks = data.supportedFrameworks || ['webgpu'];
    c.clientType = data.clientType || 'browser';

    console.log(`Client ${socket.id} joined; supports frameworks: ${c.supportedFrameworks.join(', ')}`);
    broadcastClientList();
  });

  socket.on('task:request', () => {
    const client = matrixState.clients.get(socket.id);
    if (client && matrixState.isRunning && !client.isBusyWithCustomChunk && !client.isBusyWithMatrixTask) {
      const task = assignMatrixTask(socket.id);
      if (task) socket.emit('task:assign', task);
    }
  });

  socket.on('task:complete', data => {
    const client = matrixState.clients.get(socket.id);
    if (!client || !client.connected) return;

    client.isBusyWithMatrixTask = false;
    client.lastActive = Date.now();

    const { assignmentId, taskId, result: received, processingTime, reportedChecksum } = data;
    const inst = matrixState.activeTasks.get(assignmentId);
    if (!inst || inst.logicalTaskId !== taskId || inst.assignedTo !== socket.id) {
      return;
    }
    matrixState.activeTasks.delete(assignmentId);

    const tdef = matrixState.tasks.find(t => t.id === taskId);
    if (!tdef || tdef.status === 'completed') return;

    const serverChecksum = checksumMatrixRowsFloat32LE(received);

    // store the submission (now also tracking checksums)
    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    matrixResultBuffer.get(taskId).push({
      clientId: socket.id,
      result: received,
      processingTime,
      submissionTime: Date.now(),
      // NEW fields
      reportedChecksum: reportedChecksum,
      serverChecksum
    });

    // tally votes by checksum (distinct clients; require self-consistency)
    const entries = matrixResultBuffer.get(taskId);
    const tally = tallyKByChecksum(
      entries.map(e => ({
        clientId: e.clientId,
        serverChecksum: e.serverChecksum,
        reportedChecksum: e.reportedChecksum
      })),
      /* expectedByteLength */ undefined,  // size implicit in Float32 packing
      ADMIN_K_PARAMETER
    );

    let verified = false;
    let finalData = null;
    let contributors = [];
    if (tally.ok) {
      // pick a representative result from the winning checksum bucket
      const winner = entries.find(e => e.serverChecksum === tally.winningChecksum);
      finalData = winner?.result;
      contributors = Array.from(tally.voters).slice(0, ADMIN_K_PARAMETER);
      verified = !!finalData;
    }

    if (verified) {
      const ok = processMatrixTaskResult(taskId, finalData, contributors.slice(0, ADMIN_K_PARAMETER));
      if (ok) {
        contributors.forEach(cid => {
          const cl = matrixState.clients.get(cid);
          if (cl) cl.socket.emit('task:verified', { taskId, type: 'matrixMultiply' });
        });
        broadcastStatus();
      }
    } else {
      socket.emit('task:submitted', { taskId, type: 'matrixMultiply' });
    }
  });

  socket.on('workload:done', ({ id, result, results, processingTime, reportedChecksum }) => {
    const wl = customWorkloads.get(id);
    if (!wl || wl.isChunkParent) {
      socket.emit('workload:error', { id, message: 'Invalid workload ID or is chunk parent.' });
      return;
    }
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithNonChunkedWGSL = false;

    wl.activeAssignments.delete(socket.id);

    // NEW: Handle single result or array of results
    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) {
      finalResults = [finalResults];
    }

    const checksumData = checksumFromResults(finalResults);
    wl.results.push({
      clientId: socket.id,
      results: finalResults, // NEW: Array of results
      result: finalResults[0], // Backward compatibility
      submissionTime: Date.now(),
      processingTime,
      // NEW fields
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength
    });
    wl.processingTimes.push({ clientId: socket.id, timeMs: processingTime });

    // Verify by checksum with K unique voters; also check size if outputSizes exists
    const expectedSize = wl.outputSizes ? wl.outputSizes.reduce((a, b) => a + b, 0) : wl.outputSize;
    const tally = tallyKByChecksum(
      wl.results.map(r => ({
        clientId: r.clientId,
        serverChecksum: r.serverChecksum,
        reportedChecksum: r.reportedChecksum,
        byteLength: r.byteLength
      })),
      expectedSize,
      ADMIN_K_PARAMETER
    );

    if (tally.ok) {
      const winner = wl.results.find(r => r.serverChecksum === tally.winningChecksum);
      wl.status = 'complete';
      wl.finalResults = winner.results; // NEW: Array
      wl.finalResult = Array.from(Buffer.concat(winner.results.map(r => Buffer.from(r, 'base64')))); // Concatenated for backward compatibility
      wl.completedAt = Date.now();
      console.log(`✅ ${wl.framework} workload ${id} VERIFIED & COMPLETE.`);
      io.emit('workload:complete', {
        id,
        label: wl.label,
        finalResults: wl.finalResults, // NEW: Array
        finalResult: wl.finalResult // Backward compatibility
      });
    } else {
      wl.status = 'processing';
      console.log(`${wl.framework} ${id}: ${wl.results.length} submissions, awaiting ${ADMIN_K_PARAMETER}.`);
    }
    saveCustomWorkloads();
    broadcastCustomWorkloadList();
  });

  // Enhanced: Handle enhanced chunk completion
  socket.on('workload:chunk_done_enhanced', ({ parentId, chunkId, results, result, processingTime, strategy, metadata, reportedChecksum }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;

    const workloadState = customWorkloads.get(parentId);
    const chunkStore = customWorkloadChunks.get(parentId);

    if (workloadState && chunkStore && chunkStore.enhanced) {
      // Find the chunk descriptor
      const cd = chunkStore.allChunkDefs.find(c => c.chunkId === chunkId);
      if (!cd) {
        console.warn(`Enhanced chunk ${chunkId} not found in store for parent ${parentId}`);
        return;
      }

      // NEW: Handle single result or array of results
      let finalResults = results || [result];
      if (!Array.isArray(finalResults)) {
        finalResults = [finalResults];
      }

      // Compute server checksum and record submission, then try to verify using shared helper.
      let checksumData;
      try {
        checksumData = checksumFromResults(finalResults);
      } catch (err) {
        console.error(`Enhanced chunk ${chunkId} from ${socket.id} invalid base64`);
        return;
      }

      const submission = {
        clientId: socket.id,
        results: finalResults, // NEW: Array
        processingTime,
        reportedChecksum: reportedChecksum,
        serverChecksum: checksumData.serverChecksum,
        byteLength: checksumData.byteLength
      };

      const verifyRes = verifyAndRecordChunkSubmission(workloadState, chunkStore, cd, submission, cd.chunkOrderIndex, ADMIN_K_PARAMETER);

      // If verification succeeded, hand the verified chunk to chunkingManager for its assembly logic.
      if (verifyRes.verified) {
        // Use the verified_results as the canonical chunk payload
        const verifiedResults = cd.verified_results;
        const assemblyResult = chunkingManager.handleChunkCompletion(
          parentId,
          chunkId,
          verifiedResults,
          processingTime
        );

        if (assemblyResult.success && assemblyResult.status === 'complete') {
          workloadState.status = 'complete';
          workloadState.finalResult = assemblyResult.finalResult.data;
          workloadState.completedAt = Date.now();
          workloadState.assemblyStats = assemblyResult.stats;

          customWorkloadChunks.delete(parentId);

          console.log(`✅ Enhanced workload ${parentId} completed with ${assemblyResult.stats.chunkingStrategy}/${assemblyResult.stats.assemblyStrategy}`);

          io.emit('workload:complete', {
            id: parentId,
            label: workloadState.label,
            finalResult: Array.from(Buffer.from(assemblyResult.finalResult.data, 'base64')),
            enhanced: true,
            stats: assemblyResult.stats
          });

          saveCustomWorkloads();
          broadcastCustomWorkloadList();
        } else if (!assemblyResult.success) {
          console.error(`Enhanced chunk processing failed: ${assemblyResult.error}`);
          workloadState.status = 'error';
          workloadState.error = assemblyResult.error;
          saveCustomWorkloads();
          broadcastCustomWorkloadList();
        } else {
          // Partial success - assembly not complete yet.
          workloadState.status = 'processing_chunks';
          saveCustomWorkloads();
          broadcastCustomWorkloadList();
        }
      } else {
        // Not enough matching submissions yet — leave it to future submissions.
        workloadState.status = 'processing_chunks';
        saveCustomWorkloads();
        broadcastCustomWorkloadList();
      }
    } else {
      // Fall back to regular chunk processing
      handleRegularChunkCompletion(parentId, chunkId, results || [result], processingTime, reportedChecksum);
    }
  });

  socket.on('workload:chunk_done', ({ parentId, chunkId, chunkOrderIndex, results, result, processingTime, reportedChecksum }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;
    const parent = customWorkloads.get(parentId);
    const store = customWorkloadChunks.get(parentId);
    if (!parent || !store) return;

    const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd || cd.verified_results || cd.status === 'completed') return;

    cd.activeAssignments.delete(socket.id);

    // NEW: Handle single result or array of results
    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) {
      finalResults = [finalResults];
    }

    // Validate base64 and compute checksum
    let checksumData;
    try {
      checksumData = checksumFromResults(finalResults);
    } catch (err) {
      console.error(`Invalid base64 for chunk ${chunkId} from ${socket.id}`);
      return;
    }

    const submission = {
      clientId: socket.id,
      results: finalResults, // NEW: Array
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength,
      buffers: checksumData.buffers // Individual buffers for assembly
    };

    parent.processingTimes.push({ clientId: socket.id, chunkId, timeMs: processingTime });

    // Use shared verification helper
    const verifyRes = verifyAndRecordChunkSubmission(parent, store, cd, submission, chunkOrderIndex, ADMIN_K_PARAMETER);

    if (verifyRes.verified) {
      console.log(`Chunk ${chunkId} VERIFIED by K=${ADMIN_K_PARAMETER} (checksum ${verifyRes.winningChecksum.slice(0,8)}…) ` +
                  `(${store.completedChunksData.size}/${store.expectedChunks})`);
    }

    // After per-chunk verification, aggregation will happen if all chunks are present.
    if (
      !parent.finalResult &&
      store.completedChunksData.size === store.expectedChunks
    ) {
      console.log(`All chunks for ${parentId} verified, aggregating...`);
      parent.status = 'aggregating';

      // NEW: Handle multi-output aggregation
      if (parent.outputSizes && parent.outputSizes.length > 1) {
        // Multi-output: aggregate by output index
        const finalOutputs = {};
        for (let outputIdx = 0; outputIdx < parent.outputSizes.length; outputIdx++) {
          const outputBuffers = [];
          for (let chunkIdx = 0; chunkIdx < store.expectedChunks; chunkIdx++) {
            const chunkBuffers = store.completedChunksData.get(chunkIdx);
            if (chunkBuffers && chunkBuffers[outputIdx]) {
              outputBuffers.push(chunkBuffers[outputIdx]);
            }
          }
          finalOutputs[`output_${outputIdx}`] = Buffer.concat(outputBuffers);
        }
        parent.finalOutputs = finalOutputs; // NEW: Object with named outputs
        parent.finalResult = Array.from(Buffer.concat(Object.values(finalOutputs))); // Concatenated for backward compatibility
      } else {
        // Single output: concatenate all chunks
        parent.finalResult = Array.from(Buffer.concat(
          Array.from({ length: store.expectedChunks }, (_, i) => store.completedChunksData.get(i))
        ));
      }

      parent.status = 'complete';
      parent.completedAt = Date.now();
      io.emit('workload:complete', {
        id: parentId,
        label: parent.label,
        finalResult: parent.finalResult,
        finalOutputs: parent.finalOutputs // NEW: Multi-output results
      });

      saveCustomWorkloads();
      broadcastCustomWorkloadList();
    }

    saveCustomWorkloads();
    broadcastCustomWorkloadList();
  });

  function verifyAndRecordChunkSubmission(parent, store, cd, submission, chunkOrderIndex, k) {
    // defensive early return if already verified
    if (cd.verified_results) {
      return { verified: true, winningChecksum: cd._winningChecksum || null, winnerSubmission: null };
    }

    // append submission if not already present from same client + checksum
    // (avoid double-pushing identical entries)
    if (!cd.submissions) cd.submissions = [];
    const dup = cd.submissions.some(s => s.clientId === submission.clientId && s.serverChecksum === submission.serverChecksum);
    if (!dup) cd.submissions.push(submission);

    // compute expected size: prefer per-chunk descriptor, fallback to parent-level hint
    const expectedSize = (cd.outputSizes && cd.outputSizes.length > 0)
      ? cd.outputSizes.reduce((a, b) => a + b, 0)
      : (parent.chunkOutputSizes && parent.chunkOutputSizes.length > 0)
        ? parent.chunkOutputSizes.reduce((a, b) => a + b, 0)
        : parent.chunkOutputSize;

    // run the generic tally routine
    const tallyResult = tallyKByChecksum(
      cd.submissions.map(s => ({
        clientId: s.clientId,
        serverChecksum: s.serverChecksum,
        reportedChecksum: s.reportedChecksum,
        byteLength: s.byteLength
      })),
      expectedSize,
      k
    );

    if (!tallyResult.ok) {
      // Not yet reached K matching, nothing to change
      return { verified: false, winningChecksum: null, winnerSubmission: null };
    }

    // If we reach here, a checksum has K distinct voters and we must mark chunk verified.
    const winningChecksum = tallyResult.winningChecksum;

    // Avoid race: if another request verified between our tally and now, respect it.
    if (cd.verified_results) {
      return { verified: true, winningChecksum: cd._winningChecksum || winningChecksum, winnerSubmission: null };
    }

    // find a submission object that matches the winning checksum
    const winner = cd.submissions.find(s => s.serverChecksum === winningChecksum);
    if (!winner) {
      // Unlikely, but defensive
      return { verified: false, winningChecksum: null, winnerSubmission: null };
    }

    // mark verified, persist winner results and store buffers for assembly
    cd.verified_results = winner.results; // NEW: Array of results
    cd.status = 'completed';
    cd._winningChecksum = winningChecksum; // keep for debugging / idempotence

    // store buffers for assembly at the chunkOrderIndex
    if (!store.completedChunksData) store.completedChunksData = new Map();
    store.completedChunksData.set(chunkOrderIndex, winner.buffers || winner.results.map(r => Buffer.from(r, 'base64')));

    return { verified: true, winningChecksum, winnerSubmission: winner };
  }

  function handleRegularChunkCompletion(parentId, chunkId, results, processingTime, reportedChecksum) {
    // Default regular chunk completion path uses the same verification helper.
    const parent = customWorkloads.get(parentId);
    const store = customWorkloadChunks.get(parentId);
    if (!parent || !store) return;

    const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd || cd.verified_results || cd.status === 'completed') return;

    // Validate base64 and compute checksum
    let checksumData;
    try {
      checksumData = checksumFromResults(results);
    } catch (err) {
      console.error(`Invalid base64 for chunk ${chunkId} in regular handler`);
      return;
    }

    const submission = {
      clientId: 'regular-handler', // anonymous (or could be replaced by actual sender if known)
      results: results, // NEW: Array
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength,
      buffers: checksumData.buffers
    };

    parent.processingTimes.push({ clientId: submission.clientId, chunkId, timeMs: processingTime });

    const verifyRes = verifyAndRecordChunkSubmission(parent, store, cd, submission, cd.chunkOrderIndex, ADMIN_K_PARAMETER);

    if (verifyRes.verified) {
      console.log(`Regular chunk ${chunkId} VERIFIED by K=${ADMIN_K_PARAMETER} (checksum ${verifyRes.winningChecksum.slice(0,8)}…) ` +
                  `(${store.completedChunksData.size}/${store.expectedChunks})`);
    }

    // If all chunks done, assemble
    if (!parent.finalResult && store.completedChunksData.size === store.expectedChunks) {
      parent.status = 'aggregating';
      parent.finalResult = Array.from(Buffer.concat(
        Array.from({ length: store.expectedChunks }, (_, i) => Buffer.concat(store.completedChunksData.get(i) || []))
      ));
      parent.status = 'complete';
      parent.completedAt = Date.now();
      io.emit('workload:complete', {
        id: parentId,
        label: parent.label,
        finalResult: parent.finalResult
      });
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
    } else {
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
    }
  }

  socket.on('workload:busy', ({ id, reason }) => {
    const c = matrixState.clients.get(socket.id);
    if (c) c.isBusyWithNonChunkedWGSL = false;
    const wl = customWorkloads.get(id);
    if (wl && !wl.isChunkParent) {
      wl.activeAssignments.delete(socket.id);
      if (wl.status !== 'complete') wl.status = 'pending_dispatch';
      console.warn(`WGSL ${id} declined by ${socket.id} (${reason||'busy'})`);
      saveCustomWorkloads(); broadcastCustomWorkloadList();
    }
    tryDispatchNonChunkedWorkloads();
  });

  socket.on('workload:error', ({ id, message }) => {
    const c = matrixState.clients.get(socket.id);
    if (c) c.isBusyWithNonChunkedWGSL = false;
    const wl = customWorkloads.get(id);
    if (wl && !wl.isChunkParent) {
      wl.activeAssignments.delete(socket.id);
      if (wl.status !== 'complete') wl.status = 'pending_dispatch';
      console.warn(`WGSL ${id} errored on ${socket.id}: ${message}`);
      saveCustomWorkloads(); broadcastCustomWorkloadList();
    }
    tryDispatchNonChunkedWorkloads();
  });

  socket.on('workload:chunk_error', ({ parentId, chunkId, message }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;
    console.warn(`Chunk error ${chunkId} from ${socket.id}: ${message}`);
    const store = customWorkloadChunks.get(parentId);
    if (store) {
      const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
      if (cd) {
        cd.activeAssignments.delete(socket.id);
        const parent = customWorkloads.get(parentId);
        if (parent) {
          parent.processingTimes.push({ chunkId, clientId: socket.id, error: message });
        }
      }
    }
  });

  socket.on('admin:set_k_parameter', newK => {
    if (typeof newK === 'number' && newK >= 1 && Number.isInteger(newK)) {
      ADMIN_K_PARAMETER = newK;
      console.log(`K set to ${ADMIN_K_PARAMETER} by ${socket.id}`);
      socket.emit('admin:feedback', { success: true, message: `K = ${ADMIN_K_PARAMETER}`, panelType: 'system' });
      io.emit('admin:k_update', ADMIN_K_PARAMETER);
    } else {
      socket.emit('admin:feedback', { success: false, message: 'Invalid K, must be integer ≥1.', panelType: 'system' });
    }
  });

  socket.on('admin:startQueuedCustomWorkloads', () => {
    let startedNonChunked = 0;
    let activatedChunkParents = 0;
    customWorkloads.forEach(wl => {
      if (wl.status === 'queued') {
        wl.startedAt = Date.now();
        if (wl.isChunkParent) {
          const prep = prepareAndQueueChunks(wl);
          if (!prep.success) return;
          const store = customWorkloadChunks.get(wl.id);
          wl.status = 'assigning_chunks';
          store.status = 'assigning_chunks';
          store.allChunkDefs.forEach(cd => {
            cd.status = 'queued';
            cd.dispatchesMade = 0;
            cd.submissions = [];
            cd.activeAssignments.clear();
            cd.verified_results = null;
          });
          console.log(`Activated chunk parent ${wl.id} with ${store.allChunkDefs.length} chunks.`);
          activatedChunkParents++;
          io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
        } else {
          wl.status = 'pending_dispatch';
          wl.dispatchesMade = 0;
          wl.activeAssignments.clear();
          wl.results = [];
          console.log(`Queued non-chunk WGSL ${wl.id} for K-dispatch.`);
          startedNonChunked++;
        }
      }
    });
    saveCustomWorkloads();
    broadcastCustomWorkloadList();
    socket.emit('admin:feedback', {
      success: true,
      message: `${activatedChunkParents} chunk parents activated, ${startedNonChunked} non-chunked queued.`,
      panelType: 'wgsl'
    });
  });

  socket.on('admin:removeCustomWorkload', ({ workloadId }) => {
    if (customWorkloads.has(workloadId)) {
      customWorkloads.delete(workloadId);
      customWorkloadChunks.delete(workloadId);
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
      io.emit('workload:removed', { id: workloadId });
      socket.emit('admin:feedback', {
        success: true,
        message: `Workload ${workloadId} removed.`,
        panelType: 'wgsl'
      });
    } else {
      socket.emit('admin:feedback', {
        success: false,
        message: `Workload ${workloadId} not found.`,
        panelType: 'wgsl'
      });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
    handleClientDisconnect(socket.id);
    broadcastClientList();
    broadcastStatus();
  });
});

function broadcastClientList() {
  const list = Array.from(matrixState.clients.values()).map(c => ({
    id: c.id,
    joinedAt: c.joinedAt,
    completedTasks: c.completedTasks,
    gpuInfo: c.gpuInfo,
    supportedFrameworks: c.supportedFrameworks,
    clientType: c.clientType,
    lastActive: c.lastActive,
    connected: c.connected,
    usingCpu: c.gpuInfo?.isCpuComputation || false,
    isPuppeteer: c.isPuppeteer,
    isBusyWithMatrixTask: c.isBusyWithMatrixTask,
    isBusyWithCustomChunk: c.isBusyWithCustomChunk,
    isBusyWithNonChunkedWGSL: c.isBusyWithNonChunkedWGSL
  }));
  io.emit('clients:update', { clients: list });
}

function assignTasksToAvailableClients() {
  if (!matrixState.isRunning) return;
  for (const [cid, client] of matrixState.clients.entries()) {
    if (client.connected && client.gpuInfo && !client.isBusyWithMatrixTask && !client.isBusyWithCustomChunk && !client.isBusyWithNonChunkedWGSL) {
      const task = assignMatrixTask(cid);
      if (task) {
        client.socket.emit('task:assign', task);
      }
    }
  }
}

server.listen(PORT, () => {
  loadCustomWorkloads();
  console.log(`Server on ${useHttps ? 'HTTPS' : 'HTTP'}://localhost:${PORT}`);
  console.log(`Enhanced chunking system initialized with ${chunkingManager.registry.listStrategies().chunking.length} chunking strategies`);

  setInterval(() => {
    assignTasksToAvailableClients();
    assignCustomChunkToAvailableClients();
    tryDispatchNonChunkedWorkloads();
  }, 5000);
});