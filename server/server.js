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

// For multi-output or single-output chunk results
function checksumFromResults(results) {
  if (!Array.isArray(results)) results = [results];
  const buffers = results.map(b64 => Buffer.from(b64, 'base64'));
  const combinedBuf = Buffer.concat(buffers);
  return {
    serverChecksum: sha256Hex(combinedBuf),
    byteLength: combinedBuf.length,
    buffer: combinedBuf,
    buffers: buffers
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
app.use(express.json({ limit: '200mb' })); // increase if needed

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

const CUSTOM_TASKS_FILE = 'custom_tasks.json';
const CUSTOM_CHUNKS_FILE = 'custom_chunks.json';
const CUSTOM_CHUNK_TIMEOUT = 5 * 60 * 1000;

function saveCustomWorkloads() {
  /*
  const workloadsArray = Array.from(customWorkloads.values()).map(wl => {
    const workloadToSave = { ...wl };
    if (workloadToSave.activeAssignments instanceof Set) {
      workloadToSave.activeAssignments = Array.from(workloadToSave.activeAssignments);
    }
    // remove socket references etc if present
    return workloadToSave;
  });
  try {
    fs.writeFileSync(CUSTOM_TASKS_FILE, JSON.stringify(workloadsArray, null, 2));
  } catch (err) {
    console.error('Error saving custom workloads:', err);
  }*/
  console.log('Workload saving disabled for proof of concept');
}

function saveCustomWorkloadChunks() {
  /*
  const serialized = {};
  customWorkloadChunks.forEach((store, id) => {
    serialized[id] = {
      parentId: store.parentId,
      status: store.status,
      expectedChunks: store.expectedChunks,
      aggregationMethod: store.aggregationMethod,
      enhanced: store.enhanced,
      allChunkDefs: store.allChunkDefs.map(cd => ({
        ...cd,
        activeAssignments: Array.from(cd.activeAssignments || []),
        assignedClients: Array.from(cd.assignedClients || [])
      })),
      completedChunksData: Array.from((store.completedChunksData || new Map()).entries()).map(([idx, bufs]) => [
        idx,
        (bufs || []).map(b => b.toString('base64'))
      ])
    };
  });
  try {
    fs.writeFileSync(CUSTOM_CHUNKS_FILE, JSON.stringify(serialized, null, 2));
  } catch (err) {
    console.error('Error saving custom workload chunks:', err);
  }
  */
 console.log('Chunk saving disabled for proof of concept');
}

function loadCustomWorkloads() {
  /*
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
        workload.processingTimes = workload.processingTimes || [];
        workload.results = workload.results || [];
        workload.dispatchesMade = workload.dispatchesMade || 0;
        workload.activeAssignments = new Set(workload.activeAssignments || []);
        if (!workload.outputSizes && workload.outputSize) workload.outputSizes = [workload.outputSize];
        if (!workload.chunkOutputSizes && workload.chunkOutputSize) workload.chunkOutputSizes = [workload.chunkOutputSize];
        customWorkloads.set(workload.id, workload);
      });
      console.log(`Loaded ${customWorkloads.size} custom workloads from ${CUSTOM_TASKS_FILE}`);
    }
  } catch (err) {
    console.error('Error loading custom workloads:', err);
  }
  */
 console.log('Workload loading disabled for proof of concept - starting fresh');
}

function loadCustomWorkloadChunks() {
  /*
  try {
    if (!fs.existsSync(CUSTOM_CHUNKS_FILE)) return;
    const raw = fs.readFileSync(CUSTOM_CHUNKS_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    Object.entries(parsed).forEach(([id, store]) => {
      const reconstructed = {
        parentId: store.parentId,
        allChunkDefs: (store.allChunkDefs || []).map(cd => ({
          ...cd,
          activeAssignments: new Set(cd.activeAssignments || []),
          assignedClients: new Set(cd.assignedClients || [])
        })),
        completedChunksData: new Map((store.completedChunksData || []).map(([idx, arr]) => [
          parseInt(idx, 10),
          (arr || []).map(b64 => Buffer.from(b64, 'base64'))
        ])),
        expectedChunks: store.expectedChunks,
        status: store.status,
        aggregationMethod: store.aggregationMethod,
        enhanced: store.enhanced
      };
      customWorkloadChunks.set(id, reconstructed);
    });
    console.log(`Loaded ${customWorkloadChunks.size} custom chunk stores from ${CUSTOM_CHUNKS_FILE}`);
  } catch (err) {
    console.error('Error loading custom workload chunks:', err);
  }
  */
  console.log('Chunk loading disabled for proof of concept - starting fresh');
}

function broadcastCustomWorkloadList() {
  io.emit('workloads:list_update', Array.from(customWorkloads.values()));
}

// --- Advanced route: create parent + chunk store ---
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
    customAssemblyFile,
    customChunkingCode,
    customAssemblyCode
  } = req.body;

  try {
    // If paths provided, read files into code fields
    if (customChunkingFile && !customChunkingCode) {
      try {
        req.body.customChunkingCode = fs.readFileSync(customChunkingFile, 'utf8');
      } catch (err) {
        return res.status(400).json({ error: `Failed to read customChunkingFile: ${err.message}` });
      }
    }
    if (customAssemblyFile && !customAssemblyCode) {
      try {
        req.body.customAssemblyCode = fs.readFileSync(customAssemblyFile, 'utf8');
      } catch (err) {
        return res.status(400).json({ error: `Failed to read customAssemblyFile: ${err.message}` });
      }
    }

    if (req.body.customChunkingCode) {
      const result = chunkingManager.registerCustomStrategy(req.body.customChunkingCode, 'chunking', chunkingStrategy);
      if (!result.success) return res.status(400).json({ error: `Custom chunking strategy failed: ${result.error}` });
    }
    if (req.body.customAssemblyCode) {
      const result = chunkingManager.registerCustomStrategy(req.body.customAssemblyCode, 'assembly', assemblyStrategy);
      if (!result.success) return res.status(400).json({ error: `Custom assembly strategy failed: ${result.error}` });
    }

    const workloadId = uuidv4();
    const enhancedWorkload = {
      id: workloadId,
      label: label || `Enhanced Workload ${workloadId.substring(0, 6)}`,
      chunkingStrategy,
      assemblyStrategy,
      framework,
      input,
      metadata: { ...metadata, customShader },
      createdAt: Date.now()
    };

    const result = await chunkingManager.processChunkedWorkload(enhancedWorkload);
    if (!result.success) return res.status(400).json(result);

    const firstCd = result.chunkDescriptors && result.chunkDescriptors[0];
    customWorkloads.set(workloadId, {
      ...enhancedWorkload,
      isChunkParent: true,
      enhanced: true,
      chunkDescriptors: result.chunkDescriptors,
      status: 'queued',
      dispatchesMade: 0,
      activeAssignments: new Set(),
      processingTimes: [],
      results: [],
      plan: result.plan,
      outputSizes: firstCd?.outputSizes || []
    });

    const store = {
      parentId: workloadId,
      allChunkDefs: result.chunkDescriptors.map((cd, idx) => ({
        ...cd,
        status: 'queued',
        dispatchesMade: 0,
        submissions: [],
        activeAssignments: new Set(),
        assignedClients: new Set(),
        verified_results: null,
        chunkOrderIndex: idx
      })),
      completedChunksData: new Map(),
      expectedChunks: result.plan.totalChunks,
      status: 'awaiting_start',
      aggregationMethod: result.plan.assemblyStrategy,
      enhanced: true
    };
    customWorkloadChunks.set(workloadId, store);

    saveCustomWorkloads();
    saveCustomWorkloadChunks();
    broadcastCustomWorkloadList();

    return res.json({
      success: true,
      id: workloadId,
      totalChunks: result.plan.totalChunks,
      message: `Queued enhanced workload '${label || workloadId.slice(0,6)}'`
    });
  } catch (error) {
    console.error('Error creating advanced workload:', error);
    res.status(500).json({ error: error.message });
  }
});

// Strategy registration
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

app.get('/api/strategies', (req, res) => {
  let strategies = {};
  try {
    strategies = chunkingManager.getAvailableStrategies();
  } catch (e) {
    strategies = { chunking: [], assembly: [] };
  }

  res.json({
    available: strategies,
    examples: {
      matrix_tiled: {
        description: "Divide matrix computation into rectangular tiles",
        requiredMetadata: ["matrixSize", "tileSize"],
        example: {
          chunkingStrategy: "matrix_tiled",
          assemblyStrategy: "matrix_tiled_assembly",
          metadata: { matrixSize: 512, tileSize: 64 }
        }
      }
    }
  });
});

app.post('/api/system/k', (req, res) => {
  const { k } = req.body || {};
  if (!Number.isInteger(k) || k < 1) return res.status(400).json({ error: 'k must be integer ≥ 1' });
  ADMIN_K_PARAMETER = k;
  io.emit('admin:k_update', ADMIN_K_PARAMETER);
  res.json({ ok: true, k: ADMIN_K_PARAMETER });
});

// Activate queued parents - only enhanced parents are started without prepareAndQueueChunks
app.post('/api/workloads/startQueued', (req, res) => {
  let activatedChunkParents = 0;
  let startedNonChunked = 0;

  customWorkloads.forEach(wl => {
    if (wl.status === 'queued') {
      wl.startedAt = Date.now();
      if (wl.isChunkParent) {
        if (wl.enhanced) {
          const store = customWorkloadChunks.get(wl.id);
          if (!store) return;
          wl.status = 'assigning_chunks';
          store.status = 'assigning_chunks';
          for (const cd of store.allChunkDefs) {
            cd.status = 'queued';
            cd.dispatchesMade = 0;
            cd.submissions = [];
            cd.activeAssignments?.clear?.();
            cd.assignedClients?.clear?.();
            cd.verified_results = null;
          }
          io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
          activatedChunkParents++;
        } else {
          const prep = prepareAndQueueChunks(wl);
          if (!prep.success) return;
          const store = customWorkloadChunks.get(wl.id);
          wl.status = 'assigning_chunks';
          store.status = 'assigning_chunks';
          store.allChunkDefs.forEach(cd => {
            cd.status = 'queued';
            cd.dispatchesMade = 0;
            cd.submissions = [];
            cd.activeAssignments.clear && cd.activeAssignments.clear();
            cd.verified_results = null;
          });
          io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
          activatedChunkParents++;
        }
      } else {
        wl.status = 'pending_dispatch';
        startedNonChunked++;
      }
    }
  });

  saveCustomWorkloads();
  saveCustomWorkloadChunks();
  broadcastCustomWorkloadList();

  res.json({ ok: true, activatedChunkParents, startedNonChunked });
});

app.delete('/api/workloads/:id', (req, res) => {
  const { id } = req.params;
  if (!customWorkloads.has(id)) return res.status(404).json({ error: 'Not found' });
  customWorkloadChunks.delete(id);
  customWorkloads.delete(id);
  saveCustomWorkloads();
  saveCustomWorkloadChunks();
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

// --- Legacy-style chunk preparation (kept for fallback) ---
function prepareAndQueueChunks(parentWorkload) {
  const parentId = parentWorkload.id;

  if (customWorkloadChunks.has(parentId)) {
    customWorkloadChunks.delete(parentId);
  }

  // Parse input data (single or multi-input)
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
    finalOutputSizes: parentWorkload.outputSizes,
    chunkOutputSizes: parentWorkload.chunkOutputSizes
  };

  for (let i = 0; i < numChunks; i++) {
    const chunkId = `${parentId}-chunk-${i}`;
    const byteOffset = i * actualChunkSizeBytes;
    const currentChunkByteLength = Math.min(actualChunkSizeBytes, totalInputBytes - byteOffset);
    if (currentChunkByteLength <= 0) continue;

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
      outputSizes: parentWorkload.chunkOutputSizes,
      inputData: Object.keys(chunkInputs).length === 1 ? Object.values(chunkInputs)[0] : JSON.stringify(chunkInputs),
      inputs: Object.keys(chunkInputs).length > 1 ? Object.values(chunkInputs) : [Object.values(chunkInputs)[0] || ''],
      chunkUniforms: {},
      dispatchesMade: 0,
      submissions: [],
      activeAssignments: new Set(),
      assignedClients: new Set(),
      verified_results: null,
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

// --- Matrix helpers (full implementations) ---
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
  const elapsed = matrixState.startTime ? (Date.now() - matrixState.startTime) / 1000 : 0;
  io.emit('state:update', {
    stats: matrixState.stats,
    elapsedTime: elapsed
  });
}

// --- Dispatchers ---
function parseWorkloadInputs(inputString) {
  if (!inputString) return { inputs: {}, schema: null };

  try {
    // Try to parse as JSON first (multi-input format)
    if (inputString.startsWith('{')) {
      const parsed = JSON.parse(inputString);

      // Generate schema from parsed inputs
      const schema = generateInputSchema(parsed);

      return {
        inputs: parsed,
        schema,
        isMultiInput: Object.keys(parsed).length > 1
      };
    } else {
      // Single input (backward compatibility)
      return {
        inputs: { input: inputString },
        schema: generateInputSchema({ input: inputString }),
        isMultiInput: false
      };
    }
  } catch (e) {
    // If JSON parsing fails, treat as single input
    return {
      inputs: { input: inputString },
      schema: generateInputSchema({ input: inputString }),
      isMultiInput: false
    };
  }
}

// Generate WebGPU binding schema from inputs
function generateInputSchema(inputs) {
  const schema = {
    uniforms: [],
    inputs: [],
    outputs: []
  };

  // Add standard uniforms (these will be common across strategies)
  schema.uniforms.push({
    name: 'params',
    type: 'uniform_buffer',
    size: 64, // Reserve space for up to 16 u32 values
    binding: 0
  });

  let bindingIndex = 1;

  // Add input storage buffers
  Object.keys(inputs).forEach((inputName, index) => {
    if (index >= 4) return; // Max 4 inputs

    schema.inputs.push({
      name: inputName,
      type: 'storage_buffer',
      usage: 'read',
      binding: bindingIndex++
    });
  });

  return schema;
}

// Enhanced chunk assignment with proper schema
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

          // Parse parent inputs and generate schema
          const parsedData = parseWorkloadInputs(parent.input);
          const inputCount = Object.keys(parsedData.inputs).length;
          const outputCount = cd.outputSizes?.length || 1;

          // Create unified task data with proper schema
          const taskData = {
            ...cd,
            framework: parent.framework,
            compilationOptions: parent.compilationOptions,
            enhanced: parent.enhanced,

            // Unified input/output format
            inputSchema: generateTaskSchema(parsedData.inputs, cd.outputSizes || [cd.outputSize]),
            chunkInputs: parsedData.inputs,
            outputSizes: cd.outputSizes || [cd.outputSize],

            // Legacy compatibility
            inputs: cd.inputs || [cd.inputData || ''],
            outputSize: cd.outputSize // Keep for backward compatibility
          };

          client.socket.emit('workload:chunk_assign', taskData);

          // Fixed logging to show actual input/output counts
          console.log(`Assigned ${parent.framework} chunk ${cd.chunkId} to ${clientId} (${inputCount} inputs, ${outputCount} outputs)`);
          break;
        }
      }

      if (client.isBusyWithCustomChunk) break;
    }
  }
}

// Generate complete task schema including outputs
function generateTaskSchema(inputs, outputSizes) {
  const schema = {
    uniforms: [{
      name: 'params',
      type: 'uniform_buffer',
      size: 64,
      binding: 0
    }],
    inputs: [],
    outputs: []
  };

  let bindingIndex = 1;

  // Add input storage buffers
  Object.keys(inputs).forEach((inputName, index) => {
    if (index >= 4) return; // Max 4 inputs

    schema.inputs.push({
      name: inputName,
      type: 'storage_buffer',
      usage: 'read',
      binding: bindingIndex++
    });
  });

  // Add output storage buffers
  outputSizes.forEach((size, index) => {
    if (index >= 3) return; // Max 3 outputs

    schema.outputs.push({
      name: `output_${index}`,
      type: 'storage_buffer',
      usage: 'write',
      size: size,
      binding: bindingIndex++
    });
  });

  return schema;
}

function tryDispatchNonChunkedWorkloads() {
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (!client.connected || !client.gpuInfo || client.isBusyWithCustomChunk ||
        client.isBusyWithMatrixTask || client.isBusyWithNonChunkedWGSL || !client.socket) {
      continue;
    }

    for (const wl of customWorkloads.values()) {
      if (!wl.isChunkParent && ['pending_dispatch', 'pending'].includes(wl.status)
        && !wl.finalResultBase64 && wl.dispatchesMade < ADMIN_K_PARAMETER
        && !wl.activeAssignments.has(clientId)) {

        if (!client.supportedFrameworks.includes(wl.framework)) {
          continue;
        }

        wl.dispatchesMade++;
        wl.activeAssignments.add(clientId);
        client.isBusyWithNonChunkedWGSL = true;
        console.log(`Dispatching ${wl.framework} workload ${wl.label} to ${clientId}`);

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

  // Matrix task completion: 'task:complete'
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

    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    matrixResultBuffer.get(taskId).push({
      clientId: socket.id,
      result: received,
      processingTime,
      submissionTime: Date.now(),
      reportedChecksum: reportedChecksum,
      serverChecksum
    });

    const entries = matrixResultBuffer.get(taskId);
    const tally = tallyKByChecksum(
      entries.map(e => ({
        clientId: e.clientId,
        serverChecksum: e.serverChecksum,
        reportedChecksum: e.reportedChecksum
      })),
      undefined,
      ADMIN_K_PARAMETER
    );

    let verified = false;
    let finalData = null;
    let contributors = [];
    if (tally.ok) {
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

  // Non-chunked workload done: 'workload:done'
  socket.on('workload:done', ({ id, result, results, processingTime, reportedChecksum }) => {
    const wl = customWorkloads.get(id);
    if (!wl || wl.isChunkParent) {
      socket.emit('workload:error', { id, message: 'Invalid workload ID or is chunk parent.' });
      return;
    }
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithNonChunkedWGSL = false;

    wl.activeAssignments.delete(socket.id);

    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) finalResults = [finalResults];

    const checksumData = checksumFromResults(finalResults);
    wl.results.push({
      clientId: socket.id,
      results: finalResults,
      result: finalResults[0],
      submissionTime: Date.now(),
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength
    });
    wl.processingTimes.push({ clientId: socket.id, timeMs: processingTime });

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
      wl.finalResults = winner.results;
      wl.finalResultBase64 = Buffer.concat(winner.results.map(r => Buffer.from(r, 'base64'))).toString('base64');
      wl.completedAt = Date.now();
      console.log(`✅ ${wl.framework} workload ${id} VERIFIED & COMPLETE.`);
      io.emit('workload:complete', {
        id,
        label: wl.label,
        finalResults: wl.finalResults,
        finalResultBase64: wl.finalResultBase64
      });
    } else {
      wl.status = 'processing';
      console.log(`${wl.framework} ${id}: ${wl.results.length} submissions, awaiting ${ADMIN_K_PARAMETER}.`);
    }
    saveCustomWorkloads();
    broadcastCustomWorkloadList();
  });

  // Enhanced chunk completion
  socket.on('workload:chunk_done_enhanced', ({ parentId, chunkId, results, result, processingTime, strategy, metadata, reportedChecksum }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;

    const workloadState = customWorkloads.get(parentId);
    const chunkStore = customWorkloadChunks.get(parentId);

    if (workloadState && chunkStore && chunkStore.enhanced) {
      const cd = chunkStore.allChunkDefs.find(c => c.chunkId === chunkId);
      if (!cd) {
        console.warn(`Enhanced chunk ${chunkId} not found in store for parent ${parentId}`);
        return;
      }

      let finalResults = results || [result];
      if (!Array.isArray(finalResults)) finalResults = [finalResults];

      let checksumData;
      try {
        checksumData = checksumFromResults(finalResults);
      } catch (err) {
        console.error(`Enhanced chunk ${chunkId} from ${socket.id} invalid base64`);
        return;
      }

      const submission = {
        clientId: socket.id,
        results: finalResults,
        processingTime,
        reportedChecksum: reportedChecksum,
        serverChecksum: checksumData.serverChecksum,
        byteLength: checksumData.byteLength,
        buffers: checksumData.buffers
      };

      const verifyRes = verifyAndRecordChunkSubmission(workloadState, chunkStore, cd, submission, cd.chunkOrderIndex, ADMIN_K_PARAMETER);

      if (verifyRes.verified) {
        const verifiedResults = cd.verified_results;
        const assemblyResult = chunkingManager.handleChunkCompletion(parentId, chunkId, verifiedResults, processingTime);

        if (assemblyResult.success && assemblyResult.status === 'complete') {
          workloadState.status = 'complete';
          let finalBase64 = null;
          if (assemblyResult.finalResult && assemblyResult.finalResult.data) {
            finalBase64 = typeof assemblyResult.finalResult.data === 'string'
              ? assemblyResult.finalResult.data
              : Buffer.from(assemblyResult.finalResult.data).toString('base64');
          }
          workloadState.finalResultBase64 = finalBase64;
          workloadState.completedAt = Date.now();
          workloadState.assemblyStats = assemblyResult.stats;

          customWorkloadChunks.delete(parentId);
          saveCustomWorkloads();
          saveCustomWorkloadChunks();

          console.log(`✅ Enhanced workload ${parentId} completed with ${assemblyResult.stats.chunkingStrategy}/${assemblyResult.stats.assemblyStrategy}`);

          io.emit('workload:complete', {
            id: parentId,
            label: workloadState.label,
            finalResultBase64: finalBase64,
            enhanced: true,
            stats: assemblyResult.stats
          });
        } else if (!assemblyResult.success) {
          console.error(`Enhanced chunk processing failed: ${assemblyResult.error}`);
          workloadState.status = 'error';
          workloadState.error = assemblyResult.error;
          saveCustomWorkloads(); saveCustomWorkloadChunks();
          broadcastCustomWorkloadList();
        } else {
          workloadState.status = 'processing_chunks';
          saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
        }
      } else {
        workloadState.status = 'processing_chunks';
        saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
      }
    } else {
      handleRegularChunkCompletion(parentId, chunkId, results || [result], processingTime, reportedChecksum);
    }
  });

  // Regular chunk completion
  socket.on('workload:chunk_done', ({ parentId, chunkId, chunkOrderIndex, results, result, processingTime, reportedChecksum }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;
    const parent = customWorkloads.get(parentId);
    const store = customWorkloadChunks.get(parentId);
    if (!parent || !store) return;

    const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd || cd.verified_results || cd.status === 'completed') return;

    cd.activeAssignments.delete(socket.id);

    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) finalResults = [finalResults];

    let checksumData;
    try {
      checksumData = checksumFromResults(finalResults);
    } catch (err) {
      console.error(`Invalid base64 for chunk ${chunkId} from ${socket.id}`);
      return;
    }

    const submission = {
      clientId: socket.id,
      results: finalResults,
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength,
      buffers: checksumData.buffers
    };

    parent.processingTimes.push({ clientId: socket.id, chunkId, timeMs: processingTime });

    const verifyRes = verifyAndRecordChunkSubmission(parent, store, cd, submission, chunkOrderIndex, ADMIN_K_PARAMETER);

    if (verifyRes.verified) {
      console.log(`Chunk ${chunkId} VERIFIED by K=${ADMIN_K_PARAMETER} (checksum ${verifyRes.winningChecksum.slice(0,8)}…) ` +
                  `(${store.completedChunksData.size}/${store.expectedChunks})`);
    }

    if (!parent.finalResultBase64 && store.completedChunksData.size === store.expectedChunks) {
      console.log(`All chunks for ${parentId} verified, aggregating...`);
      parent.status = 'aggregating';

      if (parent.outputSizes && parent.outputSizes.length > 1) {
        const finalOutputs = {};
        for (let outputIdx = 0; outputIdx < parent.outputSizes.length; outputIdx++) {
          const outputBuffers = [];
          for (let chunkIdx = 0; chunkIdx < store.expectedChunks; chunkIdx++) {
            const chunkBuffers = store.completedChunksData.get(chunkIdx);
            if (chunkBuffers && chunkBuffers[outputIdx]) {
              outputBuffers.push(chunkBuffers[outputIdx]);
            } else {
              console.error(`Missing buffer for chunk ${chunkIdx} output ${outputIdx}`);
            }
          }
          finalOutputs[`output_${outputIdx}`] = Buffer.concat(outputBuffers).toString('base64');
        }
        parent.finalOutputs = finalOutputs;
        parent.finalResultBase64 = Buffer.concat(Object.values(finalOutputs).map(b64 => Buffer.from(b64, 'base64'))).toString('base64');
      } else {
        const perChunkBuffers = [];
        for (let i = 0; i < store.expectedChunks; i++) {
          const arr = store.completedChunksData.get(i);
          if (!arr || arr.length === 0) {
            console.error(`Missing chunk ${i} during aggregation for ${parentId}`);
          } else {
            perChunkBuffers.push(arr[0]);
          }
        }
        const finalBuffer = Buffer.concat(perChunkBuffers);
        parent.finalResultBase64 = finalBuffer.toString('base64');
      }

      parent.status = 'complete';
      parent.completedAt = Date.now();
      io.emit('workload:complete', {
        id: parentId,
        label: parent.label,
        finalResultBase64: parent.finalResultBase64,
        finalOutputs: parent.finalOutputs
      });

      customWorkloadChunks.delete(parentId);
      saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
    }

    saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
  });

  function verifyAndRecordChunkSubmission(parent, store, cd, submission, chunkOrderIndex, k) {
    if (cd.verified_results) {
      return { verified: true, winningChecksum: cd._winningChecksum || null, winnerSubmission: null };
    }

    if (!cd.submissions) cd.submissions = [];
    const dup = cd.submissions.some(s => s.clientId === submission.clientId && s.serverChecksum === submission.serverChecksum);
    if (!dup) cd.submissions.push(submission);

    const expectedSize = (cd.outputSizes && cd.outputSizes.length > 0)
      ? cd.outputSizes.reduce((a, b) => a + b, 0)
      : (parent.chunkOutputSizes && parent.chunkOutputSizes.length > 0)
        ? parent.chunkOutputSizes.reduce((a, b) => a + b, 0)
        : parent.chunkOutputSize;

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
      return { verified: false, winningChecksum: null, winnerSubmission: null };
    }

    const winningChecksum = tallyResult.winningChecksum;

    if (cd.verified_results) {
      return { verified: true, winningChecksum: cd._winningChecksum || winningChecksum, winnerSubmission: null };
    }

    const winner = cd.submissions.find(s => s.serverChecksum === winningChecksum);
    if (!winner) {
      return { verified: false, winningChecksum: null, winnerSubmission: null };
    }

    cd.verified_results = winner.results;
    cd.status = 'completed';
    cd._winningChecksum = winningChecksum;

    if (!store.completedChunksData) store.completedChunksData = new Map();
    store.completedChunksData.set(chunkOrderIndex, winner.buffers || winner.results.map(r => Buffer.from(r, 'base64')));

    return { verified: true, winningChecksum, winnerSubmission: winner };
  }

  function handleRegularChunkCompletion(parentId, chunkId, results, processingTime, reportedChecksum) {
    const parent = customWorkloads.get(parentId);
    const store = customWorkloadChunks.get(parentId);
    if (!parent || !store) return;

    const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd || cd.verified_results || cd.status === 'completed') return;

    let checksumData;
    try {
      checksumData = checksumFromResults(results);
    } catch (err) {
      console.error(`Invalid base64 for chunk ${chunkId} in regular handler`);
      return;
    }

    const submission = {
      clientId: 'regular-handler',
      results: results,
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

    if (!parent.finalResultBase64 && store.completedChunksData.size === store.expectedChunks) {
      parent.status = 'aggregating';
      const perChunkBuffers = [];
      for (let i = 0; i < store.expectedChunks; i++) {
        const arr = store.completedChunksData.get(i);
        if (!arr || arr.length === 0) {
          console.error(`Missing chunk ${i} during aggregation for ${parentId}`);
        } else {
          perChunkBuffers.push(arr[0]);
        }
      }
      parent.finalResultBase64 = Buffer.concat(perChunkBuffers).toString('base64');
      parent.status = 'complete';
      parent.completedAt = Date.now();
      io.emit('workload:complete', {
        id: parentId,
        label: parent.label,
        finalResultBase64: parent.finalResultBase64
      });

      customWorkloadChunks.delete(parentId);
      saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
    } else {
      saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
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
      saveCustomWorkloads(); saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
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
      saveCustomWorkloads(); saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
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
          saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
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
          if (wl.enhanced) {
            const store = customWorkloadChunks.get(wl.id);
            if (!store) return;
            wl.status = 'assigning_chunks';
            store.status = 'assigning_chunks';
            store.allChunkDefs.forEach(cd => {
              cd.status = 'queued';
              cd.dispatchesMade = 0;
              cd.submissions = [];
              cd.activeAssignments = new Set();
              cd.verified_results = null;
            });
            console.log(`Activated enhanced chunk parent ${wl.id} with ${store.allChunkDefs.length} chunks.`);
            activatedChunkParents++;
            io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
          } else {
            const prep = prepareAndQueueChunks(wl);
            if (!prep.success) return;
            const store = customWorkloadChunks.get(wl.id);
            wl.status = 'assigning_chunks';
            store.status = 'assigning_chunks';
            store.allChunkDefs.forEach(cd => {
              cd.status = 'queued';
              cd.dispatchesMade = 0;
              cd.submissions = [];
              cd.activeAssignments = new Set();
              cd.verified_results = null;
            });
            console.log(`Activated legacy chunk parent ${wl.id} with ${store.allChunkDefs.length} chunks.`);
            activatedChunkParents++;
            io.emit('workload:parent_started', { id: wl.id, label: wl.label, status: wl.status });
          }
        } else {
          wl.status = 'pending_dispatch';
          wl.dispatchesMade = 0;
          wl.activeAssignments = new Set();
          wl.results = [];
          console.log(`Queued non-chunk WGSL ${wl.id} for K-dispatch.`);
          startedNonChunked++;
        }
      }
    });
    saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
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
      saveCustomWorkloads(); saveCustomWorkloadChunks(); broadcastCustomWorkloadList();
      io.emit('workload:removed', { id: workloadId });
      socket.emit('admin:feedback', { success: true, message: `Workload ${workloadId} removed.`, panelType: 'wgsl' });
    } else {
      socket.emit('admin:feedback', { success: false, message: `Workload ${workloadId} not found.`, panelType: 'wgsl' });
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
  loadCustomWorkloadChunks();
  console.log(`Server on ${useHttps ? 'HTTPS' : 'HTTP'}://localhost:${PORT}`);
  try {
    const counts = chunkingManager.registry && chunkingManager.registry.listStrategies ? chunkingManager.registry.listStrategies() : chunkingManager.getAvailableStrategies();
    const nChunking = (counts && counts.chunking) ? counts.chunking.length : (counts.chunking ? counts.chunking.length : 0);
    console.log(`Enhanced chunking system initialized with ${nChunking} chunking strategies`);
  } catch (e) {
    console.log('Enhanced chunking system initialized');
  }

  setInterval(() => {
    assignTasksToAvailableClients();
    assignCustomChunkToAvailableClients();
    tryDispatchNonChunkedWorkloads();
  }, 5000);
});

