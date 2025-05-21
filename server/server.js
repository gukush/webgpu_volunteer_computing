const express = require('express');
const fs = require('fs');
const path = require('path'); // Corrected: require path
const { v4: uuidv4 } = require('uuid');

// Import both http and https modules
const http = require('http');
const https = require('https');

const app = express();

// Read SSL certificates
let server;
let useHttps = false;

try {
  const privateKey = fs.readFileSync(path.join(__dirname, 'certificates/key.pem'), 'utf8');
  const certificate = fs.readFileSync(path.join(__dirname, 'certificates/cert.pem'), 'utf8');
  const credentials = { key: privateKey, cert: certificate };
  server = https.createServer(credentials, app);
  useHttps = true;
  console.log('Using HTTPS server with self-signed certificates');
} catch (error) {
  console.warn('SSL certificates not found, falling back to HTTP:', error.message);
  server = http.createServer(app);
}

// Initialize Socket.io with CORS settings
const { Server } = require('socket.io');
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
    credentials: true
  }
});

// CORS middleware for Express
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

// Serve static files and JSON parser
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json()); // Added from the second snippet

// State management
const state = {
  isRunning: false,
  problem: null,
  tasks: [], // Stores task definitions, status can change
  activeTasks: new Map(), // taskId -> taskObject (tasks currently assigned and being processed by a client)
  completedTasks: new Map(), // taskId -> taskObject (tasks that are fully VERIFIED and processed)
  clients: new Map(), // clientId -> clientObject { id, socket, connected, joinedAt, lastActive, completedTasks, gpuInfo }
  startTime: null,
  endTime: null,
  stats: {
    totalClients: 0,
    activeClients: 0,
    completedTasks: 0, // Number of VERIFIED tasks
    totalTasks: 0
  }
};

// Result buffer for verification
const resultBuffer = new Map(); // taskId -> Array of { clientId, result, processingTime, submissionTime, submittedVia? }
const MIN_VERIFICATIONS = 2; // Configurable: number of identical results needed for verification

// Task timeout in milliseconds (3 minutes)
const TASK_TIMEOUT = 3 * 60 * 1000;

// Generate random matrix of given size
function generateRandomMatrix(size) {
  const matrix = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) {
      row.push(Math.random() * 2 - 1); // Random values between -1 and 1
    }
    matrix.push(row);
  }
  return matrix;
}

// Prepare matrix multiplication problem
function prepareMatrixMultiplication(size, chunkSize) {
  console.log(`Preparing matrix multiplication problem: ${size}x${size} with chunk size ${chunkSize}`);

  // Reset states for a new problem
  state.activeTasks.clear();
  state.completedTasks.clear();
  resultBuffer.clear(); // Clear verification buffer
  state.problem = null;
  state.tasks = [];
  state.stats.completedTasks = 0;
  state.stats.totalTasks = 0;
  state.startTime = null;
  state.endTime = null;
  state.isRunning = false; // Temporarily set to false

  const matrixA = generateRandomMatrix(size);
  const matrixB = generateRandomMatrix(size);

  state.problem = {
    type: 'matrixMultiply',
    matrixA,
    matrixB,
    size,
    chunkSize
  };

  const numChunks = Math.ceil(size / chunkSize);
  for (let i = 0; i < numChunks; i++) {
    const startRow = i * chunkSize;
    const endRow = Math.min((i + 1) * chunkSize, size);
    state.tasks.push({
      id: `task-${i}`,
      startRow,
      endRow,
      status: 'pending' // Initial status
    });
  }

  state.stats.totalTasks = state.tasks.length;
  state.startTime = Date.now();
  state.isRunning = true; // Set to true now that setup is complete

  console.log(`Created ${state.tasks.length} tasks`);
  return state.problem;
}

// Assign a task to a client
function assignTask(clientId) {
  // Find a task that is 'pending' and not yet completed (verified).
  const taskDefinition = state.tasks.find(task => task.status === 'pending' && !state.completedTasks.has(task.id));

  if (!taskDefinition) {
    return null; // No assignable tasks currently
  }

  // Mark task as active in the main task list
  taskDefinition.status = 'active';
  // taskDefinition.assignedTo = clientId; // This might be better tracked in activeTasks for specific assignment instance
  // taskDefinition.assignTime = Date.now();

  // Add to activeTasks map (tracks specific client assignments for timeout purposes)
  state.activeTasks.set(taskDefinition.id, {
    id: taskDefinition.id,
    assignedTo: clientId,
    startTime: Date.now() // Start time of this specific client's work on this task
  });

  console.log(`Assigning task ${taskDefinition.id} to client ${clientId}`);
  return { // Data sent to client for computation
    id: taskDefinition.id,
    startRow: taskDefinition.startRow,
    endRow: taskDefinition.endRow,
    matrixA: state.problem.matrixA,
    matrixB: state.problem.matrixB,
    size: state.problem.size
  };
}

// Process a task result AFTER it has been VERIFIED
function processTaskResult(taskId, verifiedResultData, contributingClientIds = []) {
  const taskDefinition = state.tasks.find(t => t.id === taskId);
  if (!taskDefinition) {
    console.error(`[ProcessTaskResult] Critical: Task definition for ${taskId} not found.`);
    return false;
  }

  if (taskDefinition.status === 'completed') {
    console.warn(`[ProcessTaskResult] Task ${taskId} is already marked as completed. Ignoring.`);
    return false; // Already processed
  }

  taskDefinition.status = 'completed';
  taskDefinition.result = verifiedResultData; // Store verified result in main task list

  // Remove from activeTasks (it might have been assigned to one or more clients who timed out or submitted)
  // This simply removes the general "active" tracking for timeout; specific client submissions are in resultBuffer.
  state.activeTasks.delete(taskId);

  // Find relevant submissions for processing time (e.g., take the first one)
  const submissions = resultBuffer.get(taskId) || [];
  const relevantCorrectSubmissions = submissions.filter(s => JSON.stringify(s.result) === JSON.stringify(verifiedResultData));
  const representativeSubmission = relevantCorrectSubmissions.length > 0 ? relevantCorrectSubmissions[0] : {};

  state.completedTasks.set(taskId, {
    id: taskId,
    startRow: taskDefinition.startRow,
    endRow: taskDefinition.endRow,
    status: 'completed',
    result: verifiedResultData,
    assignedTo: contributingClientIds.join(', ') || 'verified_by_system', // List contributors
    processingTime: representativeSubmission.processingTime || 0, // Example: use one processing time
    verifiedAt: Date.now()
  });

  state.stats.completedTasks++;
  console.log(`Task ${taskId} marked as completed and verified. Total completed: ${state.stats.completedTasks}/${state.stats.totalTasks}`);

  // Update stats for all contributing clients
  contributingClientIds.forEach(cId => {
    const client = state.clients.get(cId);
    if (client) {
      client.completedTasks = (client.completedTasks || 0) + 1;
      client.lastActive = Date.now();
    }
  });

  resultBuffer.delete(taskId); // Clean up verified task from buffer

  if (state.stats.completedTasks === state.stats.totalTasks && state.isRunning) {
    finalizeComputation();
  }
  return true;
}

// Handle client disconnection
function handleClientDisconnect(clientId) {
  // Reassign tasks that were actively assigned to this client and not yet verified
  for (const [taskId, activeTaskInstance] of state.activeTasks.entries()) {
    if (activeTaskInstance.assignedTo === clientId) {
      console.log(`Task ${taskId} was assigned to disconnected client ${clientId}. Reverting to pending.`);
      const taskDefinition = state.tasks.find(t => t.id === taskId);
      if (taskDefinition && taskDefinition.status !== 'completed') {
        taskDefinition.status = 'pending'; // Make it available again
      }
      state.activeTasks.delete(taskId); // Remove this specific assignment
    }
  }
  // Submissions in resultBuffer from this client remain for potential verification.
  state.clients.delete(clientId);
  state.stats.activeClients = state.clients.size;
}

// Check for timed-out tasks
function checkTaskTimeouts() {
  const now = Date.now();
  for (const [taskId, activeTaskInstance] of state.activeTasks.entries()) {
    if (now - activeTaskInstance.startTime > TASK_TIMEOUT) {
      console.log(`Task ${taskId} assigned to ${activeTaskInstance.assignedTo} timed out. Reverting to pending.`);
      const taskDefinition = state.tasks.find(t => t.id === taskId);
      if (taskDefinition && taskDefinition.status !== 'completed') {
        taskDefinition.status = 'pending'; // Make it available again
      }
      state.activeTasks.delete(taskId); // Remove this assignment
      // broadcastStatus(); // Optional: notify clients status changed
    }
  }
}

// Finalize the computation
function finalizeComputation() {
  console.log('All tasks completed, finalizing computation.');
  state.endTime = Date.now();
  state.isRunning = false;

  const totalTime = (state.endTime - state.startTime) / 1000;
  console.log(`Computation completed in ${totalTime.toFixed(2)} seconds`);

  const results = [];
  state.completedTasks.forEach(task => {
    results.push({
      id: task.id,
      startRow: task.startRow,
      endRow: task.endRow,
      // result: task.result, // Optional: include actual result data
      processingTime: task.processingTime, // This is from one of the contributors
      client: task.assignedTo // String of contributing client IDs
    });
  });
  results.sort((a,b) => parseInt(a.id.split('-')[1]) - parseInt(b.id.split('-')[1]));


  io.emit('computation:complete', {
    totalTime,
    results, // Contains summarized results per task
    stats: state.stats
  });
}

// Start periodic task timeout check
setInterval(checkTaskTimeouts, 30000); // Every 30 seconds

// Socket.io connection handling
io.on('connection', (socket) => {
  const clientId = uuidv4();
  console.log(`New client connected: ${clientId}`);

  state.clients.set(clientId, {
    id: clientId,
    socket: socket,
    connected: true,
    joinedAt: Date.now(),
    lastActive: Date.now(),
    completedTasks: 0,
    gpuInfo: null
  });

  state.stats.totalClients++;
  state.stats.activeClients = state.clients.size;

  socket.emit('register', { clientId });
  socket.emit('state:update', {
    isRunning: state.isRunning,
    stats: state.stats,
    problem: state.problem ? { type: state.problem.type, size: state.problem.size, chunkSize: state.problem.chunkSize } : null
  });
  broadcastClientList();

  socket.on('client:join', (data) => {
    const client = state.clients.get(clientId);
    if (client) {
      const isCpuClient = !data.gpuInfo || data.gpuInfo.vendor === 'CPU Fallback' || data.gpuInfo.device === 'CPU Computation';
      client.gpuInfo = isCpuClient ? { vendor: 'CPU Fallback', device: 'CPU Computation', isCpuComputation: true } : { ...data.gpuInfo, isCpuComputation: false };
      console.log(`Client ${clientId} joined. GPU: ${client.gpuInfo.vendor || 'Unknown'} ${client.gpuInfo.device || ''}`);
      client.lastActive = Date.now();
      broadcastClientList();
      if (state.isRunning) {
        const task = assignTask(clientId);
        if (task) socket.emit('task:assign', task);
        else socket.emit('task:wait');
      }
    }
  });

  socket.on('task:request', () => {
    const client = state.clients.get(clientId);
    if (client) {
      client.lastActive = Date.now();
      if (state.isRunning) {
        const task = assignTask(clientId);
        if (task) socket.emit('task:assign', task);
        else socket.emit('task:wait');
      } else {
        socket.emit('state:update', { isRunning: false, stats: state.stats });
      }
    }
  });

  socket.on('task:complete', (data) => { // Result submitted by a client
    const client = state.clients.get(clientId);
    if (!client || !client.connected) return;

    client.lastActive = Date.now();
    const { taskId, result: receivedResult, processingTime } = data;

    if (!taskId || receivedResult === undefined) {
      console.error(`Client ${clientId} submitted incomplete data: ${JSON.stringify(data)}`);
      socket.emit('task:error', { taskId, message: 'Incomplete submission data.' });
      return;
    }

    const taskDefinition = state.tasks.find(t => t.id === taskId);
    if (!taskDefinition) {
        console.warn(`Client ${clientId} submitted result for unknown task ${taskId}.`);
        socket.emit('task:error', { taskId, message: 'Unknown task ID.' });
        return;
    }

    console.log(`Client ${clientId} submitted result for task ${taskId} (took ${processingTime || 'N/A'}ms).`);

    if (taskDefinition.status === 'completed' || state.completedTasks.has(taskId)) {
      console.log(`Task ${taskId} already verified. Ignoring submission from ${clientId}.`);
      socket.emit('task:acknowledged', { taskId, message: 'Task already completed.' });
      // Still try to assign a new task if running
      if (state.isRunning) {
          const newTask = assignTask(clientId);
          if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait');
      }
      return;
    }

    if (!resultBuffer.has(taskId)) resultBuffer.set(taskId, []);
    const entries = resultBuffer.get(taskId);

    if (entries.some(entry => entry.clientId === clientId)) {
      console.log(`Client ${clientId} already submitted for task ${taskId}. Ignoring duplicate.`);
      socket.emit('task:error', { taskId, message: 'Duplicate submission for this task.'});
      // Try to assign new task to avoid client getting stuck
      if (state.isRunning) {
        const newTask = assignTask(clientId);
        if (newTask) socket.emit('task:assign', newTask); else socket.emit('task:wait');
      }
      return;
    }

    entries.push({ clientId, result: receivedResult, processingTime, submissionTime: Date.now(), submittedVia: 'socket' });

    const resultCounts = entries.reduce((acc, entry) => {
      const resultKey = JSON.stringify(entry.result);
      acc[resultKey] = (acc[resultKey] || 0) + 1;
      return acc;
    }, {});

    let verified = false;
    let finalResultData = null;
    let contributingClientIds = [];

    for (const [resultKey, count] of Object.entries(resultCounts)) {
      if (count >= MIN_VERIFICATIONS) {
        try {
            finalResultData = JSON.parse(resultKey);
            verified = true;
            contributingClientIds = entries.filter(e => JSON.stringify(e.result) === resultKey).map(e => e.clientId);
            console.log(`Task ${taskId} verified with result: ${resultKey}. Contributors: ${contributingClientIds.join(', ')}`);
            break;
        } catch (e) {
            console.error(`Error parsing result key ${resultKey} for task ${taskId}: ${e.message}`);
            // Potentially remove malformed entry or handle error
        }
      }
    }

    if (verified) {
      const success = processTaskResult(taskId, finalResultData, contributingClientIds);
      contributingClientIds.forEach(cId => {
        const c = state.clients.get(cId);
        c?.socket?.emit('task:verified', { taskId, result: finalResultData, message: 'Your submission contributed to task verification.' });
      });

      if (success && state.isRunning) {
        const newTask = assignTask(clientId); // Assign new task to current submitter
        if (newTask) socket.emit('task:assign', newTask);
        else socket.emit('task:wait');
      }
      broadcastStatus();
    } else {
      const currentSubmissionsForThisResult = resultCounts[JSON.stringify(receivedResult)] || 0;
      const needed = MIN_VERIFICATIONS - currentSubmissionsForThisResult;
      console.log(`Task ${taskId} from ${clientId} received (${entries.length} total subs, ${currentSubmissionsForThisResult}/${MIN_VERIFICATIONS} for this result). Awaiting more.`);
      socket.emit('task:submitted', { taskId, message: `Result received. Awaiting ${needed > 0 ? needed : 'further'} matching submission(s).` });
    }
  });

  socket.on('admin:start', (data) => {
    if (state.isRunning && state.stats.completedTasks < state.stats.totalTasks) {
      socket.emit('error', { message: 'Computation is already running and in progress.' });
      return;
    }
    const { matrixSize, chunkSize } = data;
    if (!Number.isInteger(matrixSize) || !Number.isInteger(chunkSize) || matrixSize <=0 || chunkSize <=0) {
        socket.emit('error', { message: 'Invalid matrixSize or chunkSize.' });
        return;
    }
    console.log(`Admin ${clientId} initiated new computation: ${matrixSize}x${matrixSize}, chunk ${chunkSize}`);
    prepareMatrixMultiplication(matrixSize, chunkSize);
    broadcastStatus(); // Includes problem info
    assignTasksToAvailableClients();
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${clientId}`);
    const client = state.clients.get(clientId);
    if(client) client.connected = false;

    handleClientDisconnect(clientId); // This will remove client from state.clients
    broadcastClientList();
    broadcastStatus();
  });
});

// HTTP Endpoint for result submission (from second snippet, adapted)
app.post('/submit-result', (req, res) => {
  const { taskId, result: receivedResult, clientId: httpClientId } = req.body;

  if (!taskId || receivedResult === undefined || !httpClientId) {
    return res.status(400).send({ error: 'Missing taskId, result, or clientId in request body' });
  }

  const taskDefinition = state.tasks.find(t => t.id === taskId);
  if (!taskDefinition) {
      return res.status(404).send({ error: `Task ${taskId} not found.`});
  }

  console.log(`HTTP submission for task ${taskId} from client ${httpClientId}.`);

  if (taskDefinition.status === 'completed' || state.completedTasks.has(taskId)) {
    return res.status(200).send({ verified: true, result: state.completedTasks.get(taskId)?.result, message: 'Task already completed.' });
  }

  if (!resultBuffer.has(taskId)) resultBuffer.set(taskId, []);
  const entries = resultBuffer.get(taskId);

  if (entries.some(entry => entry.clientId === httpClientId && entry.submittedVia === 'http')) { // Prevent http dupes from same ID
    return res.status(409).send({ error: 'Duplicate HTTP submission for this task by this client ID.' });
  }

  entries.push({ clientId: httpClientId, result: receivedResult, processingTime: 0, submissionTime: Date.now(), submittedVia: 'http' });

  const resultCounts = entries.reduce((acc, entry) => {
    const resultKey = JSON.stringify(entry.result);
    acc[resultKey] = (acc[resultKey] || 0) + 1;
    return acc;
  }, {});

  let verified = false;
  let finalResultData = null;
  let contributingClientIds = [];

  for (const [resultKey, count] of Object.entries(resultCounts)) {
    if (count >= MIN_VERIFICATIONS) {
      try {
        finalResultData = JSON.parse(resultKey);
        verified = true;
        contributingClientIds = entries.filter(e => JSON.stringify(e.result) === resultKey).map(e => e.clientId);
        console.log(`Task ${taskId} verified via HTTP stream. Result: ${resultKey}. Contributors: ${contributingClientIds.join(', ')}`);
        break;
      } catch (e) {
        console.error(`Error parsing result key ${resultKey} for task ${taskId} from HTTP: ${e.message}`);
      }
    }
  }

  if (verified) {
    const success = processTaskResult(taskId, finalResultData, contributingClientIds);
    if (success) {
      broadcastStatus();
      // If httpClientId is a known socket client, try to assign new task
      const clientObj = state.clients.get(httpClientId);
      if (clientObj && clientObj.socket && clientObj.connected && state.isRunning) {
        const newTask = assignTask(httpClientId);
        if (newTask) clientObj.socket.emit('task:assign', newTask);
        else clientObj.socket.emit('task:wait');
      }
      return res.status(200).send({ verified: true, result: finalResultData, message: 'Task verified successfully.' });
    } else {
      return res.status(500).send({ error: 'Task verification reported but internal processing failed.' });
    }
  } else {
    const currentSubmissionsForThisResult = resultCounts[JSON.stringify(receivedResult)] || 0;
    res.status(202).send({ verified: false, message: `Result for ${taskId} accepted. ${currentSubmissionsForThisResult}/${MIN_VERIFICATIONS} for this result. Total submissions for task: ${entries.length}.` });
  }
});

// Broadcast current status to all clients
function broadcastStatus() {
  const statusPayload = {
    isRunning: state.isRunning,
    stats: {
      ...state.stats,
      activeTasks: state.activeTasks.size, // More accurate count of tasks being actively worked on
      bufferedTasks: resultBuffer.size, // Tasks with at least one submission
      elapsedTime: state.startTime && state.isRunning ? (Date.now() - state.startTime) / 1000 : (state.endTime && state.startTime ? (state.endTime - state.startTime) / 1000 : 0)
    },
    problem: state.problem ? { type: state.problem.type, size: state.problem.size, chunkSize: state.problem.chunkSize } : null
  };
  io.emit('state:update', statusPayload);
}

// Broadcast client list to all clients
function broadcastClientList() {
  const clientList = Array.from(state.clients.values()).map(client => ({
    id: client.id,
    joinedAt: client.joinedAt,
    completedTasks: client.completedTasks,
    gpuInfo: client.gpuInfo,
    lastActive: client.lastActive,
    connected: client.connected, // Add connected status
    usingCpu: client.gpuInfo?.isCpuComputation || false
  }));
  io.emit('clients:update', { clients: clientList });
}

// Assign tasks to available clients (e.g., on start or if many tasks become pending)
function assignTasksToAvailableClients() {
  if (!state.isRunning) return;
  console.log("Attempting to assign tasks to all available and joined clients...");
  for (const [clientId, client] of state.clients.entries()) {
    // Check if client is connected, joined (has GPU info), and not already processing too many tasks (if such a limit exists)
    if (client.connected && client.gpuInfo) { // gpuInfo implies they've "joined"
      // Check if this client already has an active task to prevent double assignment from this loop
      let alreadyHasActiveTask = false;
      for(const activeTask of state.activeTasks.values()){
          if(activeTask.assignedTo === clientId){
              alreadyHasActiveTask = true;
              break;
          }
      }
      if(!alreadyHasActiveTask){
          const task = assignTask(clientId);
          if (task) {
            client.socket.emit('task:assign', task);
          } else {
            // No tasks left for this client or any client
            // client.socket.emit('task:wait'); // Can be noisy if no tasks
            break; // Stop trying if no tasks are assignable
          }
      }
    }
  }
}

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on ${useHttps ? 'HTTPS' : 'HTTP'} at ${useHttps ? 'https' : 'http'}://localhost:${PORT}`);
});