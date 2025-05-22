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
app.use(express.json({ limit: '10mb' }));

const matrixState = {
  isRunning: false, problem: null, tasks: [], activeTasks: new Map(),
  completedTasks: new Map(), clients: new Map(), startTime: null, endTime: null,
  stats: { totalClients: 0, activeClients: 0, completedTasks: 0, totalTasks: 0 }
};
const matrixResultBuffer = new Map();
const MATRIX_MIN_VERIFICATIONS = 2;
const MATRIX_TASK_TIMEOUT = 3 * 60 * 1000; // 3 minutes

const customWorkloads = new Map();
const CUSTOM_WGSL_REQUIRED_VOTES = 1;
const CUSTOM_TASKS_FILE = 'custom_tasks.json';

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
            if (workload.finalResult) {
                workload.status = 'complete';
            } else if (workload.results && workload.results.length > 0) {
                workload.status = 'processing';
            } else {
                workload.status = 'pending';
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
  const { label, wgsl, entry = 'main', workgroupCount, bindLayout = "storage-in-storage-out", input, outputSize } = req.body;

  if (!wgsl || !workgroupCount || !outputSize) {
    return res.status(400).json({ error: 'Missing required fields: wgsl, workgroupCount, outputSize' });
  }
  if (!Array.isArray(workgroupCount) || workgroupCount.length !== 3 || !workgroupCount.every(n => typeof n === 'number' && n > 0)) {
    return res.status(400).json({ error: 'workgroupCount must be an array of 3 positive numbers.' });
  }
  if (typeof outputSize !== 'number' || outputSize <= 0) {
    return res.status(400).json({ error: 'outputSize must be a positive number.' });
  }
  if (wgsl.length > 100 * 1024) { // 100KB limit
      return res.status(400).json({ error: 'WGSL source code too large (max 100KB).' });
  }

  const id = uuidv4();
  const meta = {
    id, label: label || `Custom Workload ${id.substring(0,6)}`, wgsl, entry,
    workgroupCount, bindLayout, input, outputSize,
    status: 'queued', // Initial status is 'queued'
    results: [], processingTimes: [], createdAt: Date.now()
  };
  customWorkloads.set(id, meta);
  saveCustomWorkloads();

  io.emit('workload:new', meta); // Broadcast so UIs/clients are aware (clients must filter by status)
  broadcastCustomWorkloadList(); // Send updated full list

  console.log(`📡 Queued custom WGSL workload ${id} (${meta.label}). Needs admin to start.`);
  res.json({ ok: true, id, message: `Workload "${meta.label}" queued. Needs admin start.` });
});


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
  matrixState.clients.delete(clientId);
  matrixState.stats.activeClients = matrixState.clients.size;
}

function checkMatrixTaskTimeouts() {
  const now = Date.now();
  for (const [taskId, activeTaskInstance] of matrixState.activeTasks.entries()) {
    if (now - activeTaskInstance.startTime > MATRIX_TASK_TIMEOUT) {
      console.log(`Matrix task ${taskId} assigned to ${activeTaskInstance.assignedTo} timed out. Reverting to pending.`);
      const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
      if (taskDefinition && taskDefinition.status !== 'completed') taskDefinition.status = 'pending';
      matrixState.activeTasks.delete(taskId);
    }
  }
}
setInterval(checkMatrixTaskTimeouts, 30000); // Check every 30 seconds

io.on('connection', (socket) => {
  const clientId = uuidv4(); // This clientId is unique per connection
  console.log(`New client connected: ${clientId}`);
  const isPuppeteerWorker = socket.handshake.query.mode === 'headless';

  matrixState.clients.set(clientId, {
    id: clientId, socket: socket, connected: true, joinedAt: Date.now(),
    lastActive: Date.now(), completedTasks: 0, gpuInfo: null, isPuppeteer: isPuppeteerWorker
    // Add an 'isAdmin' flag here if you have an admin login system
    // isAdmin: false // example
  });
  matrixState.stats.totalClients++; matrixState.stats.activeClients = matrixState.clients.size;

  socket.emit('register', { clientId });
  socket.emit('state:update', { isRunning: matrixState.isRunning, stats: matrixState.stats, problem: matrixState.problem ? { type: matrixState.problem.type, size: matrixState.problem.size, chunkSize: matrixState.problem.chunkSize } : null });
  broadcastClientList();
  broadcastCustomWorkloadList(); // Send full list of WGSL tasks to new client

  customWorkloads.forEach(workload => {
    if (workload.status === 'pending' || workload.status === 'processing') {
        socket.emit('workload:new', workload);
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
    }
  });

  socket.on('task:request', () => {
    const client = matrixState.clients.get(clientId);
    if (client) {
      client.lastActive = Date.now();
      if (matrixState.isRunning) {
        const task = assignMatrixTask(clientId);
        if (task) socket.emit('task:assign', task); else socket.emit('task:wait', {type: 'matrixMultiply'});
      } else {
        socket.emit('state:update', { isRunning: false, stats: matrixState.stats });
      }
    }
  });

  socket.on('task:complete', (data) => {
    const client = matrixState.clients.get(clientId);
    if (!client || !client.connected) return;
    client.lastActive = Date.now();
    const { taskId, result: receivedResult, processingTime } = data;

    if (!taskId || receivedResult === undefined) {
      socket.emit('task:error', { taskId, message: 'Incomplete submission data.', type: 'matrixMultiply' }); return;
    }
    const taskDefinition = matrixState.tasks.find(t => t.id === taskId);
    if (!taskDefinition) {
      socket.emit('task:error', { taskId, message: 'Unknown task ID.', type: 'matrixMultiply' }); return;
    }
    if (taskDefinition.status === 'completed' || matrixState.completedTasks.has(taskId)) {
      if (matrixState.isRunning) {
        const newTask = assignMatrixTask(clientId); if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait', {type: 'matrixMultiply'});
      } return;
    }

    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    const entries = matrixResultBuffer.get(taskId);
    if (entries.some(entry => entry.clientId === clientId)) {
        if (matrixState.isRunning) {
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
      if (success && matrixState.isRunning) {
        const newTask = assignMatrixTask(clientId); if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait', {type: 'matrixMultiply'});
      }
      broadcastStatus();
    } else {
      socket.emit('task:submitted', { taskId, type: 'matrixMultiply' });
    }
  });

  socket.on('workload:done', ({ id, result, processingTime }) => {
    const workload = customWorkloads.get(id);
    if (!workload) {
      console.warn(`Client ${clientId} submitted result for unknown custom workload ${id}`);
      socket.emit('workload:error', { id, message: 'Unknown workload ID.' }); return;
    }
    if (workload.status === 'complete' || workload.status === 'queued') {
      console.log(`Custom workload ${id} is ${workload.status}. Ignoring submission from ${clientId}.`); return;
    }

    console.log(`Client ${clientId} submitted result for custom workload ${id} (${workload.label}). Result length: ${result?.length}`);
    workload.results.push({ socketId: clientId, clientReportedId: socket.id, result, submissionTime: Date.now(), processingTime });

    const resultCounts = workload.results.reduce((acc, resEntry) => {
        const resultKey = JSON.stringify(resEntry.result);
        acc[resultKey] = (acc[resultKey] || 0) + 1; return acc;
    }, {});

    let verifiedResultKey = null; let contributingSubmissions = [];
    for (const [key, count] of Object.entries(resultCounts)) {
        if (count >= CUSTOM_WGSL_REQUIRED_VOTES) {
            verifiedResultKey = key;
            contributingSubmissions = workload.results.filter(r => JSON.stringify(r.result) === verifiedResultKey);
            break;
        }
    }

    if (verifiedResultKey) {
        if (workload.status !== 'complete') {
            workload.status = 'complete'; workload.finalResult = JSON.parse(verifiedResultKey);
            workload.completedAt = Date.now();
            workload.processingTimes = workload.processingTimes || [];
            contributingSubmissions.forEach(sub => {
                if (sub.processingTime !== undefined) {
                    workload.processingTimes.push({ clientId: sub.socketId, timeMs: sub.processingTime });
                }
            });
            saveCustomWorkloads();
            const avgTime = workload.processingTimes?.reduce((sum, pt) => sum + pt.timeMs, 0) / (workload.processingTimes?.length || 1);
            console.log(`✅ Custom WGSL Workload ${id} (${workload.label}) finished and verified. Avg time (approx): ${avgTime.toFixed(0)} ms`);
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

  socket.on('workload:error', ({ id, message }) => {
    const workload = customWorkloads.get(id);
    console.warn(`Client ${clientId} reported error for custom workload ${id}: ${message}`);
    if (workload && workload.status !== 'complete' && workload.status !== 'error') {
      // Future: error handling logic
    }
  });

  socket.on('admin:start', (data) => { // For matrix tasks
    // Add admin check here if needed
    if (matrixState.isRunning && matrixState.stats.completedTasks < matrixState.stats.totalTasks) {
      socket.emit('error', { message: 'Matrix computation is already running.' }); return;
    }
    const { matrixSize, chunkSize } = data;
    if (!Number.isInteger(matrixSize) || !Number.isInteger(chunkSize) || matrixSize <=0 || chunkSize <=0) {
      socket.emit('error', { message: 'Invalid matrixSize or chunkSize.' }); return;
    }
    prepareMatrixMultiplication(matrixSize, chunkSize);
    broadcastStatus(); assignTasksToAvailableClients();
  });

  socket.on('admin:startQueuedCustomWorkloads', () => {
    const adminClient = matrixState.clients.get(clientId);
    // Basic Admin Check (improve with roles/permissions for production)
    if (!adminClient) { // Example: or if (!adminClient.isAdmin)
         console.warn(`Unauthorized client ${clientId} attempted to start queued workloads.`);
         socket.emit('admin:feedback', { success: false, message: "Unauthorized action."});
         return;
    }
    console.log(`Admin request from ${clientId} to start queued custom workloads.`);
    let startedCount = 0;
    customWorkloads.forEach(workload => {
      if (workload.status === 'queued') {
        workload.status = 'pending';
        workload.startedAt = Date.now();
        io.emit('workload:new', workload);
        console.log(`Custom workload ${workload.id} (${workload.label}) status changed to 'pending' and broadcasted.`);
        startedCount++;
      }
    });

    if (startedCount > 0) {
      saveCustomWorkloads();
      broadcastCustomWorkloadList();
      socket.emit('admin:feedback', { success: true, message: `${startedCount} custom workload(s) transitioned to 'pending' and broadcasted.` });
    } else {
      socket.emit('admin:feedback', { success: false, message: 'No custom workloads found in "queued" state.' });
    }
  });

  // --- NEW: Handler for removing a custom WGSL workload ---
  socket.on('admin:removeCustomWorkload', ({ workloadId }) => {
    const adminClient = matrixState.clients.get(clientId);
    // Basic Admin Check (improve with roles/permissions for production)
    if (!adminClient ) { // Example: or if (!adminClient.isAdmin)
        console.warn(`Unauthorized client ${clientId} attempted to remove workload ${workloadId}.`);
        socket.emit('admin:feedback', { success: false, message: "Unauthorized action to remove workload." });
        return;
    }

    if (!workloadId) {
        console.warn(`Admin ${clientId} attempted to remove workload without providing an ID.`);
        socket.emit('admin:feedback', { success: false, message: "Workload ID is required for removal." });
        return;
    }

    const workloadToRemove = customWorkloads.get(workloadId);

    if (workloadToRemove) {
        const removedLabel = workloadToRemove.label;
        customWorkloads.delete(workloadId);
        saveCustomWorkloads(); // Persist the change

        console.log(`Admin ${clientId} removed custom WGSL workload ${workloadId} ("${removedLabel}").`);

        io.emit('workload:removed', { id: workloadId, label: removedLabel });
        broadcastCustomWorkloadList(); // Send updated full list

        socket.emit('admin:feedback', { success: true, message: `Workload "${removedLabel}" (ID: ${workloadId}) removed successfully.` });
    } else {
        console.warn(`Admin ${clientId} attempted to remove non-existent workload ID: ${workloadId}`);
        socket.emit('admin:feedback', { success: false, message: `Workload with ID ${workloadId} not found.` });
    }
  });
  // --- END NEW Handler ---

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${clientId}`);
    const client = matrixState.clients.get(clientId);
    if(client) client.connected = false;
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
    usingCpu: client.gpuInfo?.isCpuComputation || false, isPuppeteer: client.isPuppeteer
  }));
  io.emit('clients:update', { clients: clientList });
}

function assignTasksToAvailableClients() { // For matrix tasks
  if (!matrixState.isRunning) return;
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (client.connected && client.gpuInfo) {
      let alreadyHasActiveTask = false;
      for(const activeTask of matrixState.activeTasks.values()){ if(activeTask.assignedTo === clientId){ alreadyHasActiveTask = true; break; }}
      if(!alreadyHasActiveTask){
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
});