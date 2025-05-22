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
app.use(express.json({ limit: '50mb' })); // Increased limit for potentially larger base64 inputs

const matrixState = {
  isRunning: false, problem: null, tasks: [], activeTasks: new Map(),
  completedTasks: new Map(), clients: new Map(), startTime: null, endTime: null,
  stats: { totalClients: 0, activeClients: 0, completedTasks: 0, totalTasks: 0 }
};
const matrixResultBuffer = new Map();
const MATRIX_MIN_VERIFICATIONS = 2;
const MATRIX_TASK_TIMEOUT = 3 * 60 * 1000; // 3 minutes

const customWorkloads = new Map(); // Stores parent workloads
const CUSTOM_WGSL_REQUIRED_VOTES = 1; // For non-chunked workloads
const CUSTOM_TASKS_FILE = 'custom_tasks.json';

// NEW data structures for chunking
const customWorkloadChunks = new Map(); // Key: parentWorkloadId, Value: { allChunkDefs: [], completedChunksData: Map, expectedChunks: number, status: string, aggregationMethod: string, finalOutputSize: number }
const pendingCustomChunks = []; // Array of { parentId, chunkId, ... chunkDef ... } for assignment
const CUSTOM_CHUNK_TIMEOUT = 5 * 60 * 1000; // 5 minutes for a chunk to complete

// --- Persistence for Custom Workloads ---
function saveCustomWorkloads() {
  const workloadsArray = Array.from(customWorkloads.values());
  fs.writeFile(CUSTOM_TASKS_FILE, JSON.stringify(workloadsArray, null, 2), (err) => {
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
        workload.processingTimes = workload.processingTimes || []; // Ensure this exists

        if (workload.isChunkParent && workload.status !== 'complete' && workload.status !== 'error') {
            if (['chunking_queued', 'chunking', 'assigning_chunks', 'aggregating', 'processing_chunks'].includes(workload.status) ||
                (workload.status === 'pending' && workload.isChunkParent)) {
                 console.log(`Chunked workload ${workload.id} (${workload.label}) was in an intermediate state (${workload.status}). Resetting to 'queued'. Admin must restart to re-prepare and assign chunks.`);
                 workload.status = 'queued';
            }
        }
        customWorkloads.set(workload.id, workload);
      });
      console.log(`Loaded ${customWorkloads.size} custom workloads from ${CUSTOM_TASKS_FILE}`);
    }
  } catch (err) { console.error('Error loading custom workloads:', err); }
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
    if (wgsl.length > 100 * 1024) { // Approx 100KB, adjust as needed
        return res.status(400).json({ error: 'WGSL source code too large (max 100KB).' });
    }
    if (input && input.length > 30 * 1024 * 1024 * 4/3) { // Approx 30MB base64 limit
        return res.status(400).json({ error: 'Input data too large (max approx 30MB raw).' });
    }


    const id = uuidv4();
    const workloadMeta = {
        id, label: label || `Custom Workload ${id.substring(0,6)}`, wgsl, entry,
        workgroupCount, bindLayout, input, outputSize, // outputSize is for the final aggregated result
        status: 'queued', results: [], processingTimes: [], createdAt: Date.now(),
        // Chunking related meta
        chunkable, inputChunkProcessingType, inputChunkSize, inputElementSizeBytes, outputAggregationMethod,
        isChunkParent: chunkable,
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

        customWorkloads.set(id, workloadMeta);
        const prepResult = prepareAndQueueChunks(workloadMeta);

        if (!prepResult.success) {
            customWorkloads.delete(id);
            console.error(`Failed to prepare chunks for ${id}: ${prepResult.error}`);
            return res.status(400).json({ error: `Chunk preparation failed: ${prepResult.error}` });
        }
        saveCustomWorkloads();
        broadcastCustomWorkloadList();
        console.log(`📡 Chunkable custom WGSL workload ${id} (${workloadMeta.label}) accepted. Chunks prepared. Needs admin to start.`);
        res.json({ ok: true, id, message: `Workload "${workloadMeta.label}" accepted for chunking. Chunks prepared. Needs admin start.` });

    } else {
        customWorkloads.set(id, workloadMeta);
        saveCustomWorkloads();
        broadcastCustomWorkloadList();
        console.log(`📡 Queued non-chunkable WGSL workload ${id} (${workloadMeta.label}). Needs admin to start.`);
        res.json({ ok: true, id, message: `Workload "${workloadMeta.label}" queued. Needs admin start.` });
    }
});

function prepareAndQueueChunks(parentWorkload) {
    const parentId = parentWorkload.id;
    if (customWorkloadChunks.has(parentId)) {
        console.log(`Chunks for ${parentId} already exist or were attempted. Clearing previous chunk definitions for re-preparation.`);
        customWorkloadChunks.delete(parentId);
        for (let i = pendingCustomChunks.length - 1; i >= 0; i--) {
            if (pendingCustomChunks[i].parentId === parentId) {
                pendingCustomChunks.splice(i, 1);
            }
        }
    }

    const inputData = Buffer.from(parentWorkload.input, 'base64');
    const totalInputBytes = inputData.length;
    let actualChunkSizeBytes;

    if (parentWorkload.inputChunkProcessingType === 'elements') {
        actualChunkSizeBytes = parentWorkload.inputChunkSize * parentWorkload.inputElementSizeBytes;
    } else { // bytes
        actualChunkSizeBytes = parentWorkload.inputChunkSize;
    }

    if (actualChunkSizeBytes <= 0) {
        return { success: false, error: `Invalid actualChunkSizeBytes ${actualChunkSizeBytes} for ${parentId}` };
    }
    if (totalInputBytes === 0 && parentWorkload.chunkable) {
         console.warn(`Workload ${parentId}: Input data is empty after base64 decoding. Zero chunks will be created, workload may not proceed unless shader handles empty input.`);
    }

    const numChunks = totalInputBytes > 0 ? Math.ceil(totalInputBytes / actualChunkSizeBytes) : 0;

    const chunksForParent = {
        parentId: parentId,
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
            parentId: parentId,
            chunkId: chunkId,
            chunkOrderIndex: i,
            status: 'queued',
            wgsl: parentWorkload.wgsl,
            entry: parentWorkload.entry,
            workgroupCount: parentWorkload.workgroupCount, // Client recalculates dispatch
            bindLayout: parentWorkload.bindLayout,
            inputData: chunkInputDataSlice.toString('base64'),
            chunkUniforms: {
                chunkOffsetBytes: byteOffset,
                chunkInputSizeBytes: currentChunkByteLength,
                totalOriginalInputSizeBytes: totalInputBytes,
            },
        };
        if (parentWorkload.inputChunkProcessingType === 'elements') {
            chunkDef.chunkUniforms.chunkOffsetElements = Math.floor(byteOffset / parentWorkload.inputElementSizeBytes);
            chunkDef.chunkUniforms.chunkInputSizeElements = Math.floor(currentChunkByteLength / parentWorkload.inputElementSizeBytes);
            chunkDef.chunkUniforms.totalOriginalInputSizeElements = Math.floor(totalInputBytes / parentWorkload.inputElementSizeBytes);
        }
        chunksForParent.allChunkDefs.push(chunkDef);
    }
    customWorkloadChunks.set(parentId, chunksForParent);
    console.log(`Workload ${parentId}: Prepared ${chunksForParent.allChunkDefs.length} chunks.`);
    return { success: true };
}

function generateRandomMatrix(size) {
  const matrix = [];
  for (let i = 0; i < size; i++) { const row = []; for (let j = 0; j < size; j++) row.push(Math.random() * 2 - 1); matrix.push(row); }
  return matrix;
}

function prepareMatrixMultiplication(size, chunkSize) {
  console.log(`Preparing matrix multiplication problem: ${size}x${size} with chunk size ${chunkSize}`);
  matrixState.activeTasks.clear(); matrixState.completedTasks.clear(); matrixResultBuffer.clear();
  matrixState.problem = null; matrixState.tasks = []; matrixState.stats.completedTasks = 0;
  matrixState.stats.totalTasks = 0; matrixState.startTime = null; matrixState.endTime = null; matrixState.isRunning = false;

  const matrixA = generateRandomMatrix(size); const matrixB = generateRandomMatrix(size);
  matrixState.problem = { type: 'matrixMultiply', matrixA, matrixB, size, chunkSize };
  const numChunks = Math.ceil(size / chunkSize);
  for (let i = 0; i < numChunks; i++) {
    const startRow = i * chunkSize; const endRow = Math.min((i + 1) * chunkSize, size);
    matrixState.tasks.push({ id: `task-${i}`, startRow, endRow, status: 'pending' });
  }
  matrixState.stats.totalTasks = matrixState.tasks.length; matrixState.startTime = Date.now();
  matrixState.isRunning = true; console.log(`Created ${matrixState.tasks.length} matrix tasks`);
  return matrixState.problem;
}

function assignMatrixTask(clientId) {
  const taskDefinition = matrixState.tasks.find(task => task.status === 'pending' && !matrixState.completedTasks.has(task.id));
  if (!taskDefinition) return null;
  taskDefinition.status = 'active';
  matrixState.activeTasks.set(taskDefinition.id, { id: taskDefinition.id, assignedTo: clientId, startTime: Date.now() });
  console.log(`Assigning matrix task ${taskDefinition.id} to client ${clientId}`);
  return {
    id: taskDefinition.id, startRow: taskDefinition.startRow, endRow: taskDefinition.endRow,
    matrixA: matrixState.problem.matrixA, matrixB: matrixState.problem.matrixB,
    size: matrixState.problem.size, type: 'matrixMultiply'
  };
}

function processMatrixTaskResult(taskId, verifiedResultData, contributingClientIds = []) {
  const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
  if (!taskDefinition || taskDefinition.status === 'completed') return false;

  taskDefinition.status = 'completed'; taskDefinition.result = verifiedResultData;
  matrixState.activeTasks.delete(taskId);

  const submissions = matrixResultBuffer.get(taskId) || [];
  const relevantCorrectSubmissions = submissions.filter(s => JSON.stringify(s.result) === JSON.stringify(verifiedResultData));
  const representativeSubmission = relevantCorrectSubmissions.length > 0 ? relevantCorrectSubmissions[0] : {};

  matrixState.completedTasks.set(taskId, {
    id: taskId, startRow: taskDefinition.startRow, endRow: taskDefinition.endRow, status: 'completed',
    result: verifiedResultData, assignedTo: contributingClientIds.join(', '),
    processingTime: representativeSubmission.processingTime || 0, verifiedAt: Date.now()
  });
  matrixState.stats.completedTasks++;
  console.log(`Matrix task ${taskId} marked as completed. Total completed: ${matrixState.stats.completedTasks}/${matrixState.stats.totalTasks}`);

  contributingClientIds.forEach(cId => {
    const client = matrixState.clients.get(cId);
    if (client) { client.completedTasks = (client.completedTasks || 0) + 1; client.lastActive = Date.now(); }
  });
  matrixResultBuffer.delete(taskId);

  if (matrixState.stats.completedTasks === matrixState.stats.totalTasks && matrixState.isRunning) {
    finalizeMatrixComputation();
  }
  return true;
}

function finalizeMatrixComputation() {
    console.log('All matrix tasks completed, finalizing computation.');
    matrixState.endTime = Date.now(); matrixState.isRunning = false;
    const totalTime = (matrixState.endTime - matrixState.startTime) / 1000;
    console.log(`Matrix computation completed in ${totalTime.toFixed(2)} seconds`);
    const results = [];
    matrixState.completedTasks.forEach(task => {
        results.push({ id: task.id, startRow: task.startRow, endRow: task.endRow, processingTime: task.processingTime, client: task.assignedTo });
    });
    results.sort((a,b) => parseInt(a.id.split('-')[1]) - parseInt(b.id.split('-')[1]));
    io.emit('computation:complete', { totalTime, results, stats: matrixState.stats, type: 'matrixMultiply' });
}

function assignCustomChunkToAvailableClients() {
    if (pendingCustomChunks.length === 0) return;

    for (const [clientId, client] of matrixState.clients.entries()) {
        if (client.connected && client.gpuInfo && !client.isBusyWithCustomChunk && client.socket) {
            const chunkIndex = pendingCustomChunks.findIndex(c => c.status === 'pending');
            if (chunkIndex > -1) {
                const chunkToAssign = pendingCustomChunks[chunkIndex];

                chunkToAssign.status = 'active';
                chunkToAssign.assignedTo = clientId;
                chunkToAssign.assignedAt = Date.now();
                client.isBusyWithCustomChunk = true;

                pendingCustomChunks.splice(chunkIndex, 1);

                client.socket.emit('workload:chunk_assign', chunkToAssign);
                console.log(`Assigned chunk ${chunkToAssign.chunkId} (Parent: ${chunkToAssign.parentId}) to client ${clientId}`);
                if (pendingCustomChunks.length === 0) break;
            } else {
                break;
            }
        }
    }
}

function handleClientDisconnect(clientId) {
  const client = matrixState.clients.get(clientId);
  if (client && client.isPuppeteer) { console.log(`Puppeteer client ${clientId} disconnected.`); }

  for (const [taskId, activeTaskInstance] of matrixState.activeTasks.entries()) {
    if (activeTaskInstance.assignedTo === clientId) {
      console.log(`Matrix task ${taskId} was assigned to disconnected client ${clientId}. Reverting to pending.`);
      const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
      if (taskDefinition && taskDefinition.status !== 'completed') taskDefinition.status = 'pending';
      matrixState.activeTasks.delete(taskId);
    }
  }

  customWorkloadChunks.forEach(chunkStore => {
    if (['assigning_chunks', 'processing_chunks'].includes(chunkStore.status)) {
        chunkStore.allChunkDefs.forEach(chunkDef => {
            if (chunkDef.status === 'active' && chunkDef.assignedTo === clientId) {
                console.log(`Custom chunk ${chunkDef.chunkId} (Parent: ${chunkDef.parentId}) assigned to disconnected ${clientId}. Reverting to pending.`);
                chunkDef.status = 'pending'; // Mark for re-assignment
                chunkDef.assignedTo = null;
                chunkDef.assignedAt = null;
                if (!pendingCustomChunks.find(pc => pc.chunkId === chunkDef.chunkId)) {
                    pendingCustomChunks.unshift(chunkDef);
                }
            }
        });
    }
  });

  matrixState.clients.delete(clientId);
  matrixState.stats.activeClients = matrixState.clients.size;
  if (pendingCustomChunks.length > 0) assignCustomChunkToAvailableClients();
}

function checkTaskTimeouts() {
  const now = Date.now();
  for (const [taskId, activeTaskInstance] of matrixState.activeTasks.entries()) {
    if (now - activeTaskInstance.startTime > MATRIX_TASK_TIMEOUT) {
      console.log(`Matrix task ${taskId} assigned to ${activeTaskInstance.assignedTo} timed out. Reverting to pending.`);
      const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
      if (taskDefinition && taskDefinition.status !== 'completed') taskDefinition.status = 'pending';
      matrixState.activeTasks.delete(taskId);
    }
  }

  customWorkloadChunks.forEach(chunkStore => {
     if (['assigning_chunks', 'processing_chunks'].includes(chunkStore.status)) {
        chunkStore.allChunkDefs.forEach(chunkDef => {
            if (chunkDef.status === 'active' && chunkDef.assignedAt && (now - chunkDef.assignedAt > CUSTOM_CHUNK_TIMEOUT)) {
                console.log(`Custom chunk ${chunkDef.chunkId} (Parent: ${chunkDef.parentId}) assigned to ${chunkDef.assignedTo} timed out. Reverting to pending.`);
                const client = matrixState.clients.get(chunkDef.assignedTo);
                if (client) client.isBusyWithCustomChunk = false;

                const originalAssignedTo = chunkDef.assignedTo;
                chunkDef.status = 'pending'; // Mark for re-assignment
                chunkDef.assignedTo = null;
                chunkDef.assignedAt = null;
                if (!pendingCustomChunks.find(pc => pc.chunkId === chunkDef.chunkId)) {
                    pendingCustomChunks.unshift(chunkDef);
                }
                const parentWorkload = customWorkloads.get(chunkDef.parentId);
                if(parentWorkload){
                    parentWorkload.processingTimes = parentWorkload.processingTimes || [];
                    parentWorkload.processingTimes.push({chunkId: chunkDef.chunkId, error: 'timeout', assignedTo: originalAssignedTo, timedOutAt: now});
                    // Optionally mark parent as error if too many timeouts, for now just re-queue chunk
                    saveCustomWorkloads(); // Save processingTimes update
                }
            }
        });
     }
  });
  if (pendingCustomChunks.length > 0) assignCustomChunkToAvailableClients();
}
setInterval(checkTaskTimeouts, 30000);


io.on('connection', (socket) => {
  const clientId = uuidv4();
  console.log(`New client connected: ${clientId}`);
  const isPuppeteerWorker = socket.handshake.query.mode === 'headless';

  matrixState.clients.set(clientId, {
    id: clientId, socket: socket, connected: true, joinedAt: Date.now(),
    lastActive: Date.now(), completedTasks: 0, gpuInfo: null, isPuppeteer: isPuppeteerWorker,
    isBusyWithCustomChunk: false
  });
  matrixState.stats.totalClients++; matrixState.stats.activeClients = matrixState.clients.size;

  socket.emit('register', { clientId });
  socket.emit('state:update', { isRunning: matrixState.isRunning, stats: matrixState.stats, problem: matrixState.problem ? { type: matrixState.problem.type, size: matrixState.problem.size, chunkSize: matrixState.problem.chunkSize } : null });
  broadcastClientList();
  broadcastCustomWorkloadList();

  customWorkloads.forEach(workload => {
    if (workload.status === 'pending' && !workload.isChunkParent) {
        socket.emit('workload:new', workload);
    }
    if (workload.isChunkParent && customWorkloadChunks.has(workload.id)){
        const chunkData = customWorkloadChunks.get(workload.id);
        if(chunkData.status === 'assigning_chunks' || chunkData.status === 'processing_chunks'){
            socket.emit('workload:parent_started', {id: workload.id, label: workload.label, status: workload.status});
        }
    }
  });

  socket.on('client:join', (data) => {
    const client = matrixState.clients.get(clientId);
    if (client) {
      const isCpuClient = !data.gpuInfo || data.gpuInfo.vendor === 'CPU Fallback' || data.gpuInfo.device === 'CPU Computation';
      client.gpuInfo = isCpuClient ? { vendor: 'CPU Fallback', device: 'CPU Computation', isCpuComputation: true } : { ...data.gpuInfo, isCpuComputation: false };
      console.log(`Client ${clientId} joined. GPU: ${client.gpuInfo.vendor || 'N/A'} ${client.gpuInfo.device || 'N/A'}${client.isPuppeteer ? ' (Puppeteer)' : ''}`);
      client.lastActive = Date.now();
      broadcastClientList();
      if (matrixState.isRunning) {
        const task = assignMatrixTask(clientId);
        if (task) socket.emit('task:assign', task); else socket.emit('task:wait', {type: 'matrixMultiply'});
      }
      assignCustomChunkToAvailableClients();
    }
  });

  socket.on('task:request', () => {
    const client = matrixState.clients.get(clientId);
    if (client) {
      client.lastActive = Date.now();
      if (matrixState.isRunning && !client.isBusyWithCustomChunk) { // Only assign matrix if not busy with chunk
        const task = assignMatrixTask(clientId);
        if (task) socket.emit('task:assign', task); else socket.emit('task:wait', {type: 'matrixMultiply'});
      } else if (!matrixState.isRunning) {
        socket.emit('state:update', { isRunning: false, stats: matrixState.stats });
      }
      // No explicit request for custom chunks, they are pushed by server
    }
  });

  socket.on('task:complete', (data) => { // Matrix task completion
    const client = matrixState.clients.get(clientId);
    if (!client || !client.connected) return;
    client.lastActive = Date.now();
    // ... (rest of matrix task completion logic from original file, unchanged) ...
    const { taskId, result: receivedResult, processingTime } = data;

    if (!taskId || receivedResult === undefined) {
      socket.emit('task:error', { taskId, message: 'Incomplete submission data.', type: 'matrixMultiply' }); return;
    }
    const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
    if (!taskDefinition) {
      socket.emit('task:error', { taskId, message: 'Unknown task ID.', type: 'matrixMultiply' }); return;
    }
    if (taskDefinition.status === 'completed' || matrixState.completedTasks.has(taskId)) {
      if (matrixState.isRunning && !client.isBusyWithCustomChunk) {
        const newTask = assignMatrixTask(clientId); if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait', {type: 'matrixMultiply'});
      } return;
    }

    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    const entries = matrixResultBuffer.get(taskId);
    if (entries.some(entry => entry.clientId === clientId)) {
        if (matrixState.isRunning && !client.isBusyWithCustomChunk) {
            const newTask = assignMatrixTask(clientId); if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait', {type: 'matrixMultiply'});
        } return;
    }
    entries.push({ clientId, result: receivedResult, processingTime, submissionTime: Date.now() });

    const resultCounts = entries.reduce((acc, entry) => {
      const resultKey = JSON.stringify(entry.result); acc[resultKey] = (acc[resultKey] || 0) + 1; return acc;
    }, {});

    let verified = false, finalResultData = null, contributingClientIds = [];
    for (const [resultKey, count] of Object.entries(resultCounts)) {
      if (count >= MATRIX_MIN_VERIFICATIONS) {
        try { finalResultData = JSON.parse(resultKey); verified = true;
              contributingClientIds = entries.filter(e => JSON.stringify(e.result) === resultKey).map(e => e.clientId);
              break;
        } catch (e) { console.error("Error parsing resultKey for matrix task", e); }
      }
    }

    if (verified) {
      const success = processMatrixTaskResult(taskId, finalResultData, contributingClientIds);
      contributingClientIds.forEach(cId => {
        matrixState.clients.get(cId)?.socket?.emit('task:verified', { taskId, type: 'matrixMultiply' });
      });
      if (success && matrixState.isRunning && !client.isBusyWithCustomChunk) {
        const newTask = assignMatrixTask(clientId); if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait', {type: 'matrixMultiply'});
      }
      broadcastStatus();
    } else {
      socket.emit('task:submitted', { taskId, type: 'matrixMultiply' });
    }
    assignCustomChunkToAvailableClients(); // Check for custom chunks after matrix task
  });

  socket.on('workload:done', ({ id, result, processingTime }) => { // For NON-CHUNKED custom workloads
    const workload = customWorkloads.get(id);
    if (!workload) {
      console.warn(`Client ${clientId} submitted result for unknown custom workload ${id}`);
      socket.emit('workload:error', { id, message: 'Unknown workload ID.' }); return;
    }
    if (workload.isChunkParent) {
        console.warn(`Received 'workload:done' for a chunk parent ${id}. Chunks should use 'workload:chunk_done'. Ignoring.`);
        return;
    }
    if (workload.status === 'complete' || workload.status === 'queued') {
      console.log(`Custom workload ${id} (${workload.label}) is ${workload.status}. Ignoring submission from ${clientId}.`); return;
    }

    console.log(`Client ${clientId} submitted result for non-chunked custom workload ${id} (${workload.label}).`);
    // Result is now base64 from client, server handles as string key
    workload.results.push({ socketId: clientId, clientReportedId: socket.id, result, submissionTime: Date.now(), processingTime });
    workload.processingTimes.push({ clientId: socket.id, timeMs: processingTime });

    const resultCounts = workload.results.reduce((acc, resEntry) => {
        const resultKey = resEntry.result; // result is already base64 string
        acc[resultKey] = (acc[resultKey] || 0) + 1; return acc;
    }, {});

    let verifiedResultKey = null;
    for (const [key, count] of Object.entries(resultCounts)) {
        if (count >= CUSTOM_WGSL_REQUIRED_VOTES) {
            verifiedResultKey = key;
            break;
        }
    }

    if (verifiedResultKey) {
        if (workload.status !== 'complete') {
            workload.status = 'complete';
            try {
                // For non-chunked, finalResult can be stored as is (base64) or decoded if server needs raw bytes.
                // The prompt's original `finalResult` was `Array.from(new Uint8Array(finalResultBuffer))`
                // Let's store the base64 string directly, or decode if needed for consistency.
                // For simplicity with JSON, let's try to store the array form.
                const finalResultBytes = Buffer.from(verifiedResultKey, 'base64');
                workload.finalResult = Array.from(finalResultBytes);
            } catch (e) {
                console.error(`Error decoding base64 final result for non-chunked workload ${id}: ${e.message}`);
                workload.status = 'error';
                workload.error = 'Final result decode error';
                saveCustomWorkloads();
                broadcastCustomWorkloadList();
                return;
            }
            workload.completedAt = Date.now();
            saveCustomWorkloads();
            const avgTime = workload.processingTimes.reduce((sum, pt) => sum + pt.timeMs, 0) / (workload.processingTimes.length || 1);
            console.log(`✅ Non-chunked WGSL Workload ${id} (${workload.label}) finished. Avg time: ${avgTime.toFixed(0)} ms`);
            io.emit('workload:complete', { id, label: workload.label, finalResult: workload.finalResult });
            broadcastCustomWorkloadList();
        }
    } else {
        workload.status = 'processing';
        saveCustomWorkloads();
        broadcastCustomWorkloadList();
        console.log(`Workload ${id} (${workload.label}) has ${workload.results.length} submissions. Waiting for ${CUSTOM_WGSL_REQUIRED_VOTES} identical results.`);
    }
  });

  socket.on('workload:chunk_done', ({ parentId, chunkId, chunkOrderIndex, result, processingTime }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;

    const parentWorkload = customWorkloads.get(parentId);
    const chunkDataStore = customWorkloadChunks.get(parentId);

    if (!parentWorkload || !chunkDataStore) {
        console.warn(`Received chunk_done for unknown parent ${parentId} or missing chunkDataStore for chunk ${chunkId} from client ${socket.id}`);
        assignCustomChunkToAvailableClients();
        return;
    }

    const chunkDef = chunkDataStore.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!chunkDef || chunkDef.status === 'completed' || chunkDef.status === 'error') {
        console.warn(`Received chunk_done for ${chunkId} (Parent ${parentId}) from ${socket.id}, but chunk status is ${chunkDef?.status} or not found. Assigned: ${chunkDef?.assignedTo}`);
        assignCustomChunkToAvailableClients();
        return;
    }
    if (chunkDef.assignedTo !== socket.id && chunkDef.status === 'active') {
         console.warn(`Chunk ${chunkId} completed by ${socket.id}, but was last assigned to ${chunkDef.assignedTo}. Accepting due to 'active' status (potential race/reassignment).`);
    } else if (chunkDef.assignedTo !== socket.id) {
         console.warn(`Chunk ${chunkId} completed by ${socket.id}, but not assigned to them (${chunkDef.assignedTo}). Ignoring stale completion.`);
         assignCustomChunkToAvailableClients(); // Free up current client just in case
         return;
    }


    chunkDef.status = 'completed';
    chunkDef.result_base64 = result; // Store base64 directly from client
    chunkDef.processingTime = processingTime;
    chunkDef.completedBy = socket.id;
    chunkDef.completedAt = Date.now();

    try {
      chunkDataStore.completedChunksData.set(chunkOrderIndex, Buffer.from(result, 'base64'));
    } catch (e) {
      console.error(`Error decoding base64 result for chunk ${chunkId} (Parent ${parentId}) from client ${socket.id}: ${e.message}`);
      chunkDef.status = 'error';
      chunkDef.error = 'Result decoding failed on server';
      parentWorkload.status = 'error';
      parentWorkload.error = `Failed to decode result for chunk ${chunkId}.`;
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
      assignCustomChunkToAvailableClients();
      return;
    }

    parentWorkload.processingTimes = parentWorkload.processingTimes || [];
    parentWorkload.processingTimes.push({ clientId: socket.id, chunkId, timeMs: processingTime });
    chunkDataStore.status = 'processing_chunks';

    console.log(`Chunk ${chunkId} (Parent: ${parentId}) completed by ${socket.id}. Progress: ${chunkDataStore.completedChunksData.size}/${chunkDataStore.expectedChunks}`);

    if (chunkDataStore.completedChunksData.size === chunkDataStore.expectedChunks) {
        if (chunkDataStore.expectedChunks === 0 && totalInputBytes > 0) { // Should have been caught earlier.
            console.error(`Error for ${parentId}: Expected 0 chunks but received completion. This indicates a logic flaw in chunk preparation or counting.`);
            parentWorkload.status = 'error';
            parentWorkload.error = 'Logic error: 0 chunks expected with non-zero input for aggregation.';
            saveCustomWorkloads(); broadcastCustomWorkloadList(); assignCustomChunkToAvailableClients(); return;
        }
        if (chunkDataStore.expectedChunks === 0 && totalInputBytes === 0) { // Valid case: 0 input, 0 chunks.
            console.log(`Workload ${parentId} had 0 expected chunks and 0 completed. Marking as complete with empty result.`);
            parentWorkload.finalResult = []; // Empty result
            parentWorkload.status = 'complete';
            parentWorkload.completedAt = Date.now();
            chunkDataStore.status = 'completed';
            saveCustomWorkloads();
            io.emit('workload:complete', { id: parentId, label: parentWorkload.label, finalResult: parentWorkload.finalResult });
            broadcastCustomWorkloadList();
            assignCustomChunkToAvailableClients();
            return;
        }


        console.log(`All ${chunkDataStore.expectedChunks} chunks for ${parentId} received. Aggregating...`);
        parentWorkload.status = 'aggregating';
        broadcastCustomWorkloadList();

        let finalResultBuffer;
        if (chunkDataStore.aggregationMethod === 'concatenate') {
            const sortedResults = [];
            let aggregationOk = true;
            for (let i = 0; i < chunkDataStore.expectedChunks; i++) {
                if (chunkDataStore.completedChunksData.has(i)) {
                    sortedResults.push(chunkDataStore.completedChunksData.get(i));
                } else {
                    const missingChunkDef = chunkDataStore.allChunkDefs.find(cd => cd.chunkOrderIndex === i);
                    console.error(`Error aggregating ${parentId}: Missing chunk data for order index ${i} (ChunkId: ${missingChunkDef?.chunkId}, Status: ${missingChunkDef?.status})`);
                    parentWorkload.status = 'error';
                    parentWorkload.error = `Aggregation failed: Missing data for chunk ${i} (ID: ${missingChunkDef?.chunkId})`;
                    aggregationOk = false;
                    break;
                }
            }
            if (aggregationOk) {
                 finalResultBuffer = Buffer.concat(sortedResults);
            } else {
                saveCustomWorkloads(); broadcastCustomWorkloadList(); assignCustomChunkToAvailableClients(); return;
            }
        } else {
            console.warn(`Unsupported aggregation method: ${chunkDataStore.aggregationMethod} for ${parentId}`);
            parentWorkload.status = 'error';
            parentWorkload.error = `Unsupported aggregation method: ${chunkDataStore.aggregationMethod}`;
            saveCustomWorkloads(); broadcastCustomWorkloadList(); assignCustomChunkToAvailableClients(); return;
        }

        if (parentWorkload.outputSize && finalResultBuffer.length !== parentWorkload.outputSize) {
            console.warn(`Warning for ${parentId}: Aggregated output size ${finalResultBuffer.length} bytes does not match expected ${parentWorkload.outputSize} bytes.`);
            // Not treating as critical error for now, but could.
        }

        parentWorkload.finalResult = Array.from(new Uint8Array(finalResultBuffer));
        parentWorkload.status = 'complete';
        parentWorkload.completedAt = Date.now();
        chunkDataStore.status = 'completed';
        saveCustomWorkloads();

        const totalChunkProcessingTime = parentWorkload.processingTimes.reduce((sum, pt) => sum + (pt.timeMs || 0), 0);
        const numProcessedChunks = parentWorkload.processingTimes.filter(pt => pt.timeMs !== undefined).length;
        const avgTime = numProcessedChunks > 0 ? totalChunkProcessingTime / numProcessedChunks : 0;

        console.log(`✅ Chunked WGSL Workload ${parentId} (${parentWorkload.label}) finished. Total chunks: ${chunkDataStore.expectedChunks}. Avg chunk time: ${avgTime.toFixed(0)} ms`);
        io.emit('workload:complete', { id: parentId, label: parentWorkload.label, finalResult: parentWorkload.finalResult });
        broadcastCustomWorkloadList();
    }
    assignCustomChunkToAvailableClients();
  });

  socket.on('workload:chunk_error', ({ parentId, chunkId, message }) => {
    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;

    console.warn(`Client ${socket.id} reported error for chunk ${chunkId} (Parent: ${parentId}): ${message}`);
    const chunkDataStore = customWorkloadChunks.get(parentId);
    if (chunkDataStore) {
        const chunkDef = chunkDataStore.allChunkDefs.find(c => c.chunkId === chunkId);
        if (chunkDef && chunkDef.status === 'active' && chunkDef.assignedTo === socket.id) {
            chunkDef.status = 'error';
            chunkDef.error = `Client error: ${message}`;
            chunkDef.erroredAt = Date.now();

            const parentWorkload = customWorkloads.get(parentId);
            if (parentWorkload) {
                parentWorkload.processingTimes.push({ chunkId, error: message, clientId: socket.id});
                if (parentWorkload.status !== 'error' && parentWorkload.status !== 'complete') {
                    parentWorkload.status = 'error';
                    parentWorkload.error = `Error on chunk ${chunkId}: ${message}`;
                    console.error(`Parent workload ${parentId} (${parentWorkload.label}) failed due to error in chunk ${chunkId}.`);
                    saveCustomWorkloads();
                    broadcastCustomWorkloadList();
                }
            }
        } else {
            console.warn(`Chunk error for ${chunkId} from ${socket.id}, but chunk not found active for this client or status is ${chunkDef?.status}.`);
        }
    }
    assignCustomChunkToAvailableClients();
  });

  socket.on('workload:error', ({ id, message }) => { // For non-chunked workload errors
    const workload = customWorkloads.get(id);
    console.warn(`Client ${clientId} reported error for non-chunked custom workload ${id}: ${message}`);
    if (workload && !workload.isChunkParent && workload.status !== 'complete' && workload.status !== 'error') {
      workload.status = 'error';
      workload.error = `Client error: ${message}`;
      workload.processingTimes.push({clientId: socket.id, error: message});
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
    }
  });

  socket.on('admin:start', (data) => {
    const adminClient = matrixState.clients.get(clientId);
    if (!adminClient ) {
        socket.emit('admin:feedback', { success: false, message: "Unauthorized matrix start."}); return;
    }
    if (matrixState.isRunning && matrixState.stats.completedTasks < matrixState.stats.totalTasks) {
      socket.emit('admin:feedback', { success: false, message: 'Matrix computation is already running.' }); return;
    }
    const { matrixSize, chunkSize } = data;
    if (!Number.isInteger(matrixSize) || !Number.isInteger(chunkSize) || matrixSize <=0 || chunkSize <=0) {
      socket.emit('admin:feedback', { success: false, message: 'Invalid matrixSize or chunkSize.' }); return;
    }
    prepareMatrixMultiplication(matrixSize, chunkSize);
    broadcastStatus(); assignTasksToAvailableClients(); // For matrix
    socket.emit('admin:feedback', { success: true, message: `Matrix computation ${matrixSize}x${matrixSize} (chunk: ${chunkSize}) started.` });
  });

  socket.on('admin:startQueuedCustomWorkloads', () => {
    const adminClient = matrixState.clients.get(clientId);
    if (!adminClient ) {
         console.warn(`Unauthorized client ${clientId} attempted to start queued workloads.`);
         socket.emit('admin:feedback', { success: false, message: "Unauthorized action."});
         return;
    }
    console.log(`Admin request from ${clientId} to start queued custom workloads.`);
    let startedCount = 0;
    let chunkedParentsActivated = 0;

    customWorkloads.forEach(workload => {
      if (workload.status === 'queued') {
        workload.status = 'pending';
        workload.startedAt = Date.now();
        startedCount++;

        if (workload.isChunkParent) {
            if (!customWorkloadChunks.has(workload.id) || customWorkloadChunks.get(workload.id).allChunkDefs.length === 0) { // Re-prepare if not found or empty (e.g. empty input case)
                console.log(`Chunks for ${workload.id} not found/empty in memory, preparing now...`);
                const prepResult = prepareAndQueueChunks(workload); // This also clears old pendingCustomChunks for this parent
                if (!prepResult.success) {
                    console.error(`Failed to re-prepare chunks for ${workload.id}: ${prepResult.error}`);
                    workload.status = 'error';
                    workload.error = `Failed to re-prepare chunks: ${prepResult.error}`;
                    return;
                }
            }

            const chunkData = customWorkloadChunks.get(workload.id);
            if (chunkData) {
                if (chunkData.expectedChunks === 0) { // If 0 chunks (e.g. empty input)
                    console.log(`Parent workload ${workload.id} (${workload.label}) has 0 chunks. Marking as complete.`);
                    workload.status = 'complete';
                    workload.finalResult = []; // Empty result
                    workload.completedAt = Date.now();
                    chunkData.status = 'completed';
                    // No chunks to assign
                } else {
                    chunkData.status = 'assigning_chunks';
                    let pendingChunkCountThisParent = 0;
                    chunkData.allChunkDefs.forEach(chunkDef => {
                        if (['queued', 'pending', 'error', 'timeout'].includes(chunkDef.status)) { // Pick up various states for re-attempt
                            chunkDef.status = 'pending';
                            chunkDef.assignedTo = null; chunkDef.assignedAt = null; chunkDef.error = null; // Reset for re-assignment
                            if (!pendingCustomChunks.find(pc => pc.chunkId === chunkDef.chunkId)) {
                               pendingCustomChunks.push(chunkDef);
                               pendingChunkCountThisParent++;
                            } else { // If already in pendingCustomChunks, ensure its status is 'pending'
                                const existing = pendingCustomChunks.find(pc => pc.chunkId === chunkDef.chunkId);
                                if(existing) existing.status = 'pending';
                            }
                        }
                    });
                    console.log(`Parent workload ${workload.id} (${workload.label}) started, ${pendingChunkCountThisParent} chunk(s) made pending/re-pending (total ${chunkData.allChunkDefs.length}).`);
                    io.emit('workload:parent_started', { id: workload.id, label: workload.label, status: 'assigning_chunks' });
                    chunkedParentsActivated++;
                }
            } else {
                 console.error(`Chunk data store not found for chunkable parent ${workload.id} during start.`);
                 workload.status = 'error';
                 workload.error = 'Chunk data store missing during start.';
            }
        } else {
            io.emit('workload:new', workload); // For non-chunked tasks
            console.log(`Non-chunked custom workload ${workload.id} (${workload.label}) status changed to 'pending'.`);
        }
      }
    });

    if (startedCount > 0) {
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
      if (chunkedParentsActivated > 0) assignCustomChunkToAvailableClients();

      socket.emit('admin:feedback', { success: true, message: `${startedCount} workload(s) set to 'pending'. ${chunkedParentsActivated} chunked parent(s) activated.` });
    } else {
      socket.emit('admin:feedback', { success: false, message: 'No queued workloads found.' });
    }
  });

  socket.on('admin:removeCustomWorkload', ({ workloadId }) => {
    const adminClient = matrixState.clients.get(clientId);
    if (!adminClient ) {
        socket.emit('admin:feedback', { success: false, message: "Unauthorized to remove workload." });
        return;
    }
    if (!workloadId) {
        socket.emit('admin:feedback', { success: false, message: "Workload ID required." });
        return;
    }
    const workloadToRemove = customWorkloads.get(workloadId);
    if (workloadToRemove) {
        const removedLabel = workloadToRemove.label;
        customWorkloads.delete(workloadId);
        if (workloadToRemove.isChunkParent) {
            customWorkloadChunks.delete(workloadId);
            for (let i = pendingCustomChunks.length - 1; i >= 0; i--) {
                if (pendingCustomChunks[i].parentId === workloadId) {
                    pendingCustomChunks.splice(i, 1);
                }
            }
        }
        saveCustomWorkloads();
        console.log(`Admin ${clientId} removed workload ${workloadId} ("${removedLabel}").`);
        io.emit('workload:removed', { id: workloadId, label: removedLabel });
        broadcastCustomWorkloadList();
        socket.emit('admin:feedback', { success: true, message: `Workload "${removedLabel}" removed.` });
    } else {
        socket.emit('admin:feedback', { success: false, message: `Workload ID ${workloadId} not found.` });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${clientId}`);
    const client = matrixState.clients.get(clientId);
    if(client) {
        client.connected = false;
    }
    handleClientDisconnect(clientId);
    broadcastClientList(); broadcastStatus();
  });
});

function broadcastStatus() {
  const statusPayload = {
    isRunning: matrixState.isRunning,
    stats: { ...matrixState.stats, activeTasks: matrixState.activeTasks.size, bufferedTasks: matrixResultBuffer.size,
             elapsedTime: matrixState.startTime && matrixState.isRunning ? (Date.now() - matrixState.startTime) / 1000 : (matrixState.endTime && matrixState.startTime ? (matrixState.endTime - matrixState.startTime) / 1000 : 0)
           },
    problem: matrixState.problem ? { type: matrixState.problem.type, size: matrixState.problem.size, chunkSize: matrixState.problem.chunkSize } : null
  };
  io.emit('state:update', statusPayload);
}

function broadcastClientList() {
  const clientList = Array.from(matrixState.clients.values()).map(client => ({
    id: client.id, joinedAt: client.joinedAt, completedTasks: client.completedTasks,
    gpuInfo: client.gpuInfo, lastActive: client.lastActive, connected: client.connected,
    usingCpu: client.gpuInfo?.isCpuComputation || false, isPuppeteer: client.isPuppeteer,
    isBusyWithCustomChunk: client.isBusyWithCustomChunk
  }));
  io.emit('clients:update', { clients: clientList });
}

function assignTasksToAvailableClients() { // For matrix tasks only
  if (!matrixState.isRunning) return;
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (client.connected && client.gpuInfo && client.socket && !client.isBusyWithCustomChunk) {
      let alreadyHasActiveMatrixTask = false;
      for(const activeTask of matrixState.activeTasks.values()){ if(activeTask.assignedTo === clientId){ alreadyHasActiveMatrixTask = true; break; }}
      if(!alreadyHasActiveMatrixTask){
        const task = assignMatrixTask(clientId);
        if (task) client.socket.emit('task:assign', task); else break;
      }
    }
  }
}

if (process.env.HEADLESS_POOL) {
  const poolSize = Number(process.env.HEADLESS_POOL);
  if (poolSize > 0) {
    console.log(`HEADLESS_POOL environment variable set. Spawning ${poolSize} Puppeteer workers...`);
    import('./headless-client.js')
      .then(module => {
        const serverUrl = `http${useHttps ? 's' : ''}://localhost:${PORT}/`;
        module.spawnPuppeteerWorkers(serverUrl, poolSize, false)
          .catch(err => console.error("Error spawning headless pool:", err));
      })
      .catch(err => console.error("Failed to import headless-client.js:", err));
  }
}

server.listen(PORT, () => {
  loadCustomWorkloads();
  console.log(`Server running on ${useHttps ? 'HTTPS' : 'HTTP'} at ${useHttps ? 'https' : 'http'}://localhost:${PORT}`);
  setInterval(() => {
      if (pendingCustomChunks.length > 0) {
          assignCustomChunkToAvailableClients();
      }
      // Also attempt to assign matrix tasks if any are pending and computation is running
      if (matrixState.isRunning && matrixState.tasks.some(t => t.status === 'pending')) {
          assignTasksToAvailableClients();
      }
  }, 7000); // Check for assignments periodically
});