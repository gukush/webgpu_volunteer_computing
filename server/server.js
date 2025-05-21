const express = require('express');
const fs = require('fs');
const path = require('path');
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

  // Create HTTPS server
  server = https.createServer(credentials, app);
  useHttps = true;
  console.log('Using HTTPS server with self-signed certificates');
} catch (error) {
  console.warn('SSL certificates not found, falling back to HTTP:', error.message);
  // Fallback to HTTP if certificates are not available
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

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// State management
const state = {
  isRunning: false,
  problem: null,
  tasks: [],
  activeTasks: new Map(),
  completedTasks: new Map(),
  clients: new Map(),
  startTime: null,
  endTime: null,
  stats: {
    totalClients: 0,
    activeClients: 0,
    completedTasks: 0,
    totalTasks: 0
  }
};

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

  // Generate random matrices
  const matrixA = generateRandomMatrix(size);
  const matrixB = generateRandomMatrix(size);

  state.problem = {
    type: 'matrixMultiply',
    matrixA,
    matrixB,
    size,
    chunkSize
  };

  // Divide the problem into tasks
  const numChunks = Math.ceil(size / chunkSize);
  state.tasks = [];

  for (let i = 0; i < numChunks; i++) {
    const startRow = i * chunkSize;
    const endRow = Math.min((i + 1) * chunkSize, size);

    state.tasks.push({
      id: `task-${i}`,
      startRow,
      endRow,
      status: 'pending'
    });
  }

  state.stats.totalTasks = state.tasks.length;
  state.startTime = Date.now();
  state.isRunning = true;

  console.log(`Created ${state.tasks.length} tasks`);

  return state.problem;
}

// Assign a task to a client
function assignTask(clientId) {
  // Check if there are pending tasks
  const pendingTask = state.tasks.find(task => task.status === 'pending');

  if (!pendingTask) {
    // No pending tasks
    return null;
  }

  // Mark task as active
  pendingTask.status = 'active';
  pendingTask.assignedTo = clientId;
  pendingTask.startTime = Date.now();

  // Add to active tasks map
  state.activeTasks.set(pendingTask.id, pendingTask);

  // Return task with problem data
  return {
    ...pendingTask,
    matrixA: state.problem.matrixA,
    matrixB: state.problem.matrixB,
    size: state.problem.size
  };
}

// Process a completed task
function processTaskResult(clientId, taskId, resultData) {
  // Check if task is in active tasks
  if (!state.activeTasks.has(taskId)) {
    console.log(`Ignoring result for unknown task ${taskId}`);
    return false;
  }

  const task = state.activeTasks.get(taskId);

  // Verify the task was assigned to this client
  if (task.assignedTo !== clientId) {
    console.log(`Task ${taskId} was assigned to ${task.assignedTo}, not ${clientId}`);
    return false;
  }

  // Mark task as completed
  task.status = 'completed';
  task.endTime = Date.now();
  task.processingTime = task.endTime - task.startTime;
  task.result = resultData;

  // Move from active to completed
  state.activeTasks.delete(taskId);
  state.completedTasks.set(taskId, task);

  // Update statistics
  state.stats.completedTasks++;

  // Update client stats
  const client = state.clients.get(clientId);
  if (client) {
    client.completedTasks++;
    client.lastActive = Date.now();
  }

  // Check if all tasks are completed
  if (state.stats.completedTasks === state.stats.totalTasks) {
    finalizeComputation();
  }

  return true;
}

// Handle client disconnection
function handleClientDisconnect(clientId) {
  // Check for any tasks assigned to this client
  for (const [taskId, task] of state.activeTasks.entries()) {
    if (task.assignedTo === clientId) {
      console.log(`Reassigning task ${taskId} from disconnected client ${clientId}`);

      // Reset task to pending
      task.status = 'pending';
      task.assignedTo = null;
      state.activeTasks.delete(taskId);

      // Add back to pending tasks
      state.tasks.push(task);
    }
  }

  // Remove client from active clients
  state.clients.delete(clientId);
  state.stats.activeClients = state.clients.size;
}

// Check for timed-out tasks
function checkTaskTimeouts() {
  const now = Date.now();

  for (const [taskId, task] of state.activeTasks.entries()) {
    // Check if task has been running too long
    if (now - task.startTime > TASK_TIMEOUT) {
      console.log(`Task ${taskId} timed out, reassigning`);

      // Reset task to pending
      task.status = 'pending';
      task.assignedTo = null;
      state.activeTasks.delete(taskId);

      // Add back to pending tasks
      state.tasks.push(task);
    }
  }
}

// Finalize the computation
function finalizeComputation() {
  console.log('All tasks completed, finalizing computation');

  state.endTime = Date.now();
  state.isRunning = false;

  const totalTime = (state.endTime - state.startTime) / 1000;
  console.log(`Computation completed in ${totalTime.toFixed(2)} seconds`);

  // Gather all results
  const results = [];
  for (let i = 0; i < state.stats.totalTasks; i++) {
    const task = state.completedTasks.get(`task-${i}`);
    if (task && task.result) {
      results.push({
        id: task.id,
        startRow: task.startRow,
        endRow: task.endRow,
        processingTime: task.processingTime,
        client: task.assignedTo
      });
    }
  }

  // Broadcast completion to all clients
  io.emit('computation:complete', {
    totalTime,
    results,
    stats: state.stats
  });
}

// Start periodic task timeout check
setInterval(checkTaskTimeouts, 30000); // Every 30 seconds

// Socket.io connection handling
io.on('connection', (socket) => {
  const clientId = uuidv4();
  console.log(`New client connected: ${clientId}`);

  // Register the client
  state.clients.set(clientId, {
    id: clientId,
    socket: socket,
    connected: true,
    joinedAt: Date.now(),
    lastActive: Date.now(),
    completedTasks: 0,
    gpuInfo: null
  });

  // Update stats
  state.stats.totalClients++;
  state.stats.activeClients = state.clients.size;

  // Send client ID to the client
  socket.emit('register', { clientId });

  // Send current state to the client
  socket.emit('state:update', {
    isRunning: state.isRunning,
    stats: state.stats
  });

  // Client requests to join computation
  socket.on('client:join', (data) => {
    const client = state.clients.get(clientId);
    if (client) {
      // Check if this is a CPU client or GPU client
      const isCpuClient = !data.gpuInfo ||
                         data.gpuInfo.vendor === 'CPU Fallback' ||
                         data.gpuInfo.device === 'CPU Computation';

      if (isCpuClient) {
        console.log(`Client ${clientId} joined with CPU fallback computation`);
        client.gpuInfo = {
          vendor: 'CPU Fallback',
          device: 'CPU Computation',
          isCpuComputation: true
        };
      } else {
        console.log(`Client ${clientId} joined with GPU: ${data.gpuInfo?.vendor || 'Unknown'} ${data.gpuInfo?.architecture || ''}`);
        client.gpuInfo = data.gpuInfo;
        client.gpuInfo.isCpuComputation = false;
      }

      client.lastActive = Date.now();

      // Broadcast updated client list
      broadcastClientList();

      // If computation is running, assign a task
      if (state.isRunning) {
        const task = assignTask(clientId);
        if (task) {
          socket.emit('task:assign', task);
        } else {
          socket.emit('task:wait');
        }
      }
    }
  });

  // Client requests a task
  socket.on('task:request', () => {
    const client = state.clients.get(clientId);
    if (client) {
      client.lastActive = Date.now();

      if (state.isRunning) {
        const task = assignTask(clientId);
        if (task) {
          socket.emit('task:assign', task);
        } else {
          socket.emit('task:wait');
        }
      } else {
        socket.emit('state:update', {
          isRunning: false,
          stats: state.stats
        });
      }
    }
  });

  // Client submits a task result
  socket.on('task:complete', (data) => {
    const client = state.clients.get(clientId);
    if (client) {
      client.lastActive = Date.now();

      console.log(`Client ${clientId} completed task ${data.taskId} in ${data.processingTime}ms`);

      // Process the result
      const success = processTaskResult(clientId, data.taskId, data.result);

      // If computation is still running, assign a new task
      if (success && state.isRunning) {
        const task = assignTask(clientId);
        if (task) {
          socket.emit('task:assign', task);
        } else {
          socket.emit('task:wait');
        }
      }

      // Broadcast updated status
      broadcastStatus();
    }
  });

  // Admin starts a new computation
  socket.on('admin:start', (data) => {
    if (!state.isRunning) {
      const { matrixSize, chunkSize } = data;
      prepareMatrixMultiplication(matrixSize, chunkSize);

      // Broadcast to all clients
      broadcastStatus();

      // Assign tasks to connected clients
      assignTasksToAvailableClients();
    } else {
      socket.emit('error', { message: 'Computation already running' });
    }
  });

  // Client disconnects
  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${clientId}`);
    handleClientDisconnect(clientId);
    broadcastClientList();
    broadcastStatus();
  });
});

// Broadcast current status to all clients
function broadcastStatus() {
  io.emit('state:update', {
    isRunning: state.isRunning,
    stats: {
      ...state.stats,
      elapsedTime: state.startTime ? (Date.now() - state.startTime) / 1000 : 0
    }
  });
}

// Broadcast client list to all clients
function broadcastClientList() {
  const clientList = Array.from(state.clients.values()).map(client => ({
    id: client.id,
    joinedAt: client.joinedAt,
    completedTasks: client.completedTasks,
    gpuInfo: client.gpuInfo,
    lastActive: client.lastActive,
    usingCpu: client.gpuInfo?.isCpuComputation || false
  }));

  io.emit('clients:update', { clients: clientList });
}

// Assign tasks to available clients
function assignTasksToAvailableClients() {
  // Find all connected clients
  for (const [clientId, client] of state.clients.entries()) {
    if (client.connected) {
      const task = assignTask(clientId);
      if (task) {
        client.socket.emit('task:assign', task);
      }
    }
  }
}

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Access your application at ${useHttps ? 'https' : 'http'}://localhost:${PORT}`);
});