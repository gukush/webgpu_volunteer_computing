// server.js
import express from 'express';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import http from 'http';
import https from 'https';
import { Server as SocketIOServer } from 'socket.io';

const app = express();

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

        if (workload.isChunkParent && workload.status !== 'complete' && workload.status !== 'error') {
          if (['chunking_queued', 'chunking', 'assigning_chunks', 'aggregating', 'processing_chunks', 'pending_dispatch'].includes(workload.status)
            || (workload.status === 'pending' && workload.isChunkParent)) {
            console.log(`Chunked workload ${workload.id} (${workload.label}) was in intermediate state ${workload.status}; resetting to 'queued'.`);
            workload.status = 'queued';
          }
        } else if (!workload.isChunkParent && ['pending', 'processing'].includes(workload.status) && !workload.finalResult) {
          console.log(`Non-chunked workload ${workload.id} (${workload.label}) was ${workload.status}; resetting to 'queued'.`);
          workload.status = 'queued';
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

app.post('/api/workloads', (req, res) => {
  const {
    label, wgsl, entry = 'main', workgroupCount, bindLayout = "storage-in-storage-out",
    input, outputSize,
    chunkable = false, inputChunkProcessingType = 'elements', inputChunkSize,
    inputElementSizeBytes = 4, outputAggregationMethod = 'concatenate'
  } = req.body;

  if (!wgsl || !workgroupCount || !outputSize) {
    return res.status(400).json({ error: 'Missing required fields: wgsl, workgroupCount, outputSize' });
  }
  if (!Array.isArray(workgroupCount) || workgroupCount.length !== 3 || !workgroupCount.every(n => typeof n === 'number' && n > 0)) {
    return res.status(400).json({ error: 'workgroupCount must be an array of 3 positive numbers.' });
  }
  if (typeof outputSize !== 'number' || outputSize <= 0) {
    return res.status(400).json({ error: 'outputSize must be a positive number.' });
  }

  const id = uuidv4();
  const workloadMeta = {
    id,
    label: label || `Custom Workload ${id.substring(0, 6)}`,
    wgsl, entry,
    workgroupCount, bindLayout, input, outputSize,
    status: 'queued',
    results: [], processingTimes: [],
    createdAt: Date.now(),
    chunkable, inputChunkProcessingType, inputChunkSize,
    inputElementSizeBytes, outputAggregationMethod,
    isChunkParent: chunkable,
    dispatchesMade: 0,
    activeAssignments: new Set()
  };

  if (chunkable) {
    if (!inputChunkSize || inputChunkSize <= 0) {
      return res.status(400).json({ error: 'Missing or invalid inputChunkSize for chunkable workload.' });
    }
    if (inputChunkProcessingType === 'elements' && (!inputElementSizeBytes || inputElementSizeBytes <= 0)) {
      return res.status(400).json({ error: 'Missing or invalid inputElementSizeBytes for chunkable workload with element processing type.' });
    }
    if (!input) {
      return res.status(400).json({ error: 'Input data (base64) is required for chunkable workloads.' });
    }
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
  const messageVerb = chunkable
    ? "accepted for chunking. Chunks prepared."
    : "queued.";
  console.log(`📡 Workload ${id} (${workloadMeta.label}) ${messageVerb}`);
  res.json({ ok: true, id, message: `Workload "${workloadMeta.label}" ${messageVerb}` });
});

function prepareAndQueueChunks(parentWorkload) {
  const parentId = parentWorkload.id;

  // clear any old chunks for this parent
  if (customWorkloadChunks.has(parentId)) {
    customWorkloadChunks.delete(parentId);
  }

  const inputData = Buffer.from(parentWorkload.input || '', 'base64');
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
    finalOutputSize: parentWorkload.outputSize
  };

  for (let i = 0; i < numChunks; i++) {
    const chunkId = `${parentId}-chunk-${i}`;
    const byteOffset = i * actualChunkSizeBytes;
    const currentChunkByteLength = Math.min(actualChunkSizeBytes, totalInputBytes - byteOffset);
    if (currentChunkByteLength <= 0) continue;

    const chunkInputDataSlice = inputData.slice(byteOffset, byteOffset + currentChunkByteLength);

    const chunkDef = {
      parentId,
      chunkId,
      chunkOrderIndex: i,
      status: 'queued',
      wgsl: parentWorkload.wgsl,
      entry: parentWorkload.entry,
      workgroupCount: parentWorkload.workgroupCount,
      bindLayout: parentWorkload.bindLayout,
      inputData: chunkInputDataSlice.toString('base64'),
      chunkUniforms: {},
      dispatchesMade: 0,
      submissions: [],
      activeAssignments: new Set(),
      assignedClients: new Set(),
      verified_result_base64: null
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
  console.log(`Workload ${parentId}: Prepared ${chunksForParent.allChunkDefs.length} chunks.`);
  return { success: true };
}

function generateRandomMatrix(size) {
  const m = new Array(size);
  for (let i = 0; i < size; i++) {
    m[i] = new Array(size);
    for (let j = 0; j < size; j++) {
      m[i][j] = Math.random();
    }
  }
  return m;
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
      // avoid reassigning same client to same logical task
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

      const store = customWorkloadChunks.get(parent.id);
      if (!store) continue;

      for (const cd of store.allChunkDefs) {
        if (
          cd.status !== 'completed' &&
          !cd.verified_result_base64 &&
          cd.dispatchesMade < ADMIN_K_PARAMETER &&
          !cd.assignedClients.has(clientId)
        ) {
          // Assign this chunk to this client
          cd.dispatchesMade++;
          cd.activeAssignments.add(clientId);
          cd.assignedClients.add(clientId);
          cd.status = 'active';
          cd.assignedTo = clientId;
          cd.assignedAt = Date.now();
          client.isBusyWithCustomChunk = true;

          client.socket.emit('workload:chunk_assign', { ...cd });
          console.log(`Assigned chunk ${cd.chunkId} (${parent.id}) to ${clientId}`);
          break;
        }
      }

      if (client.isBusyWithCustomChunk) break;
    }
  }
}

function tryDispatchNonChunkedWorkloads() {
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (!client.connected || !client.gpuInfo || client.isBusyWithCustomChunk || client.isBusyWithMatrixTask || client.isBusyWithNonChunkedWGSL || !client.socket) {
      continue;
    }
    for (const wl of customWorkloads.values()) {
      if (!wl.isChunkParent && ['pending_dispatch', 'pending'].includes(wl.status)
        && !wl.finalResult && wl.dispatchesMade < ADMIN_K_PARAMETER && !wl.activeAssignments.has(clientId)) {
        wl.dispatchesMade++;
        wl.activeAssignments.add(clientId);
        client.isBusyWithNonChunkedWGSL = true;
        console.log(`Dispatching WGSL ${wl.label} (ID ${wl.id}) to ${clientId}`);
        client.socket.emit('workload:new', wl);
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
        if (!cd.verified_result_base64 && cd.activeAssignments.size === 0 && cd.status === 'active') {
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
    isPuppeteer: socket.handshake.query.mode === 'headless',
    isBusyWithMatrixTask: false,
    isBusyWithCustomChunk: false,
    isBusyWithNonChunkedWGSL: false
  });
  matrixState.stats.totalClients++;
  matrixState.stats.activeClients = matrixState.clients.size;

  socket.emit('register', { clientId: socket.id });
  socket.emit('admin:k_update', ADMIN_K_PARAMETER);

  socket.on('client:join', data => {
    const client = matrixState.clients.get(socket.id);
    if (client) {
      client.gpuInfo = data.gpuInfo;
      console.log(`Client ${socket.id} joined; GPU info recorded.`);
      broadcastClientList();
    }
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

    const { assignmentId, taskId, result: received, processingTime } = data;
    const inst = matrixState.activeTasks.get(assignmentId);
    if (!inst || inst.logicalTaskId !== taskId || inst.assignedTo !== socket.id) {
      return;
    }
    matrixState.activeTasks.delete(assignmentId);

    const tdef = matrixState.tasks.find(t => t.id === taskId);
    if (!tdef || tdef.status === 'completed') return;

    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    matrixResultBuffer.get(taskId).push({
      clientId: socket.id,
      result: received,
      processingTime,
      submissionTime: Date.now()
    });

    const entries = matrixResultBuffer.get(taskId);
    const counts = entries.reduce((acc, e) => {
      const k = JSON.stringify(e.result);
      acc[k] = (acc[k] || 0) + 1;
      return acc;
    }, {});

    let verified = false;
    let finalData = null;
    let contributors = [];

    for (const [k, c] of Object.entries(counts)) {
      if (c >= ADMIN_K_PARAMETER) {
        verified = true;
        finalData = JSON.parse(k);
        contributors = entries.filter(e => JSON.stringify(e.result) === k).map(e => e.clientId);
        break;
      }
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

  socket.on('workload:done', ({ id, result, processingTime }) => {
    const wl = customWorkloads.get(id);
    if (!wl || wl.isChunkParent) {
      socket.emit('workload:error', { id, message: 'Invalid workload ID or is chunk parent.' });
      return;
    }
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithNonChunkedWGSL = false;

    wl.activeAssignments.delete(socket.id);
    wl.results.push({ clientId: socket.id, result, submissionTime: Date.now(), processingTime });
    wl.processingTimes.push({ clientId: socket.id, timeMs: processingTime });

    const counts = wl.results.reduce((acc, e) => {
      acc[e.result] = (acc[e.result] || 0) + 1;
      return acc;
    }, {});
    let verifiedKey = null;
    for (const [k, c] of Object.entries(counts)) {
      if (c >= ADMIN_K_PARAMETER) { verifiedKey = k; break; }
    }
    if (verifiedKey) {
      wl.status = 'complete';
      wl.finalResult = Array.from(Buffer.from(verifiedKey, 'base64'));
      wl.completedAt = Date.now();
      console.log(`✅ WGSL workload ${id} VERIFIED & COMPLETE.`);
      io.emit('workload:complete', { id, label: wl.label, finalResult: wl.finalResult });
    } else {
      wl.status = 'processing';
      console.log(`WGSL ${id}: ${wl.results.length} submissions, awaiting ${ADMIN_K_PARAMETER}.`);
    }
    saveCustomWorkloads();
    broadcastCustomWorkloadList();
  });

  socket.on('workload:chunk_done', ({ parentId, chunkId, chunkOrderIndex, result, processingTime }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;
    const parent = customWorkloads.get(parentId);
    const store = customWorkloadChunks.get(parentId);
    if (!parent || !store) return;

    const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd || cd.verified_result_base64 || cd.status === 'completed') return;

    cd.activeAssignments.delete(socket.id);
    try { Buffer.from(result, 'base64'); } catch {
      console.error(`Invalid base64 for chunk ${chunkId} from ${socket.id}`);
      return;
    }
    cd.submissions.push({ clientId: socket.id, result_base64: result, processingTime });
    parent.processingTimes.push({ clientId: socket.id, chunkId, timeMs: processingTime });
/*
    const counts = cd.submissions.reduce((acc, s) => {
      acc[s.result_base64] = (acc[s.result_base64] || 0) + 1;
      return acc;
    }, {});
    let vk = null;
    for (const [k, c] of Object.entries(counts)) {
      if (c >= ADMIN_K_PARAMETER) { vk = k; break; }
    }
    if (vk && !cd.verified_result_base64) {
  cd.verified_result_base64 = vk;
  cd.status = 'completed';
  store.completedChunksData.set(chunkOrderIndex, Buffer.from(vk, 'base64'));
  console.log(`Chunk ${chunkId} VERIFIED (${store.completedChunksData.size}/${store.expectedChunks})`);
}
*/
    if (cd.submissions.length >= ADMIN_K_PARAMETER && !cd.verified_result_base64) {
      const firstResult = cd.submissions[0].result_base64;
      cd.verified_result_base64 = firstResult;
      cd.status = 'completed';
      store.completedChunksData.set(chunkOrderIndex, Buffer.from(firstResult, 'base64'));
      console.log(`Chunk ${chunkId} accepted after ${ADMIN_K_PARAMETER} submissions (${store.completedChunksData.size}/${store.expectedChunks})`);
    }


if (
  !parent.finalResult &&                                // prevent double finalizing
  store.completedChunksData.size === store.expectedChunks
) {
  console.log(`All chunks for ${parentId} verified, aggregating...`);
  parent.status = 'aggregating';
  parent.finalResult = Array.from(Buffer.concat(
    Array.from({ length: store.expectedChunks }, (_, i) => store.completedChunksData.get(i))
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
}


    saveCustomWorkloads();
    broadcastCustomWorkloadList();
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
            cd.verified_result_base64 = null;
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
      // remove both the parent workload and any chunk‐definitions
      customWorkloads.delete(workloadId);
      customWorkloadChunks.delete(workloadId);
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
      // inform **all** clients so they can drop that card from the UI
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
  setInterval(() => {
    assignTasksToAvailableClients();
    assignCustomChunkToAvailableClients();
    tryDispatchNonChunkedWorkloads();
  }, 5000);
});
