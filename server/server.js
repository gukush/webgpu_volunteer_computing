import express from 'express';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import http from 'http';
import https from 'https';
import { Server as SocketIOServer } from 'socket.io';
import crypto from 'crypto';
import os from 'os';
import Busboy from 'busboy';
import { pipeline } from 'stream/promises';
import { WebSocketServer } from 'ws';
import url from 'url';

// Enhanced: Import chunking system
import { EnhancedChunkingManager } from './strategies/EnhancedChunkingManager.js';
const streamingChunkQueues = new Map(); // workloadId -> queue of pending chunks
const maxQueueSize = 1000; // Prevent memory bloat
const app = express();


const STORAGE_ROOT = process.env.VOLUNTEER_STORAGE || path.join(os.tmpdir(), 'volunteer');
async function ensureDir(p) { await fs.promises.mkdir(p, { recursive: true }); }


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
  'javascript': {
    browserBased: true,
    shaderExtension: '.js',
    defaultBindLayout: 'cpu-direct'
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

io.engine.on("connection_error", (err) => {
  console.warn("[EIO] connection_error", {
    code: err.code, message: err.message, context: err.context
  });
});
io.on("connection", (s) => console.log("[IO] connected", s.id));

// RAW WEBSOCKET SERVER FOR NATIVE CLIENTS
const wss = new WebSocketServer({
  noServer: true, // Important: Don't auto-attach to the HTTP server
  path: '/ws-native',
  verifyClient: (info) => {
    console.log(`[WS-NATIVE] Connection attempt from ${info.origin || 'unknown'}`);
    return true;
  }
});

// Manually handle the upgrade for raw WebSocket connections
server.on('upgrade', (request, socket, head) => {
  const pathname = url.parse(request.url).pathname;

  // Only handle /ws-native path for raw WebSocket
  if (pathname === '/ws-native') {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  } else {
    // Let Socket.IO handle all other upgrade requests
    // Don't destroy the socket here - let Socket.IO's upgrade handler take over
    return;
  }
});


// WebSocket client wrapper to mimic Socket.IO client interface
class WebSocketClientWrapper {
  constructor(ws, clientId) {
    this.ws = ws;
    this.id = clientId;
    this.connected = true;
    this.joinedAt = Date.now();
    this.lastActive = Date.now();
    this.completedTasks = 0;
    this.gpuInfo = null;
    this.supportedFrameworks = [];
    this.clientType = 'native';
    this.isBusyWithMatrixTask = false;
    this.isBusyWithCustomChunk = false;
    this.isBusyWithNonChunkedWGSL = false;
    this.hasJoined = false;

    // Message handlers
    this.eventHandlers = new Map();

    // Set up WebSocket event handlers
    this.ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        console.log(`[WS-NATIVE] Received from ${this.id}:`, message.type);

        if (message.type && this.eventHandlers.has(message.type)) {
          this.eventHandlers.get(message.type)(message.data || {});
        }
      } catch (err) {
        console.error(`[WS-NATIVE] Failed to parse message from ${this.id}:`, err);
      }
    });

    this.ws.on('close', () => {
      console.log(`[WS-NATIVE] Client ${this.id} disconnected`);
      this.connected = false;
      handleClientDisconnect(this.id);
      broadcastClientList();
      broadcastStatus();
    });

    this.ws.on('error', (err) => {
      console.error(`[WS-NATIVE] WebSocket error for ${this.id}:`, err);
    });
  }

  // Mimic Socket.IO interface
  on(event, handler) {
    this.eventHandlers.set(event, handler);
  }

  emit(event, data = {}) {
    if (this.connected && this.ws.readyState === this.ws.OPEN) {
      try {
        const message = JSON.stringify({ type: event, data });
        this.ws.send(message);
        console.log(`[WS-NATIVE] Sent to ${this.id}:`, event);
      } catch (err) {
        console.error(`[WS-NATIVE] Failed to send to ${this.id}:`, err);
      }
    }
  }

  off(event) {
    this.eventHandlers.delete(event);
  }
}

// Handle WebSocket connections
wss.on('connection', (ws, request) => {
  const clientId = `ws-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  console.log(`[WS-NATIVE] New native client connected: ${clientId}`);

  // Create wrapper that mimics Socket.IO client
  const client = new WebSocketClientWrapper(ws, clientId);

  // Add to clients map (same as Socket.IO clients)
  matrixState.clients.set(clientId, client);
  matrixState.stats.totalClients++;
  matrixState.stats.activeClients = matrixState.clients.size;

  // Send initial registration
  client.emit('register', { clientId });
  client.emit('admin:k_update', ADMIN_K_PARAMETER);

  // === EVENT HANDLERS (same as Socket.IO) ===

  // Client join with capabilities
  client.on('client:join', (data) => {
    if (client.hasJoined) return;

    client.hasJoined = true;
    client.gpuInfo = data.gpuInfo;
    client.hasWebGPU = !!data.hasWebGPU;
    client.supportedFrameworks = data.supportedFrameworks || ['vulkan'];
    client.clientType = data.clientType || 'native';

    console.log(`[WS-NATIVE] Client ${clientId} joined; supports frameworks: ${client.supportedFrameworks.join(', ')}`);
    broadcastClientList();
  });

  // Matrix task request
  client.on('task:request', () => {
    if (client && matrixState.isRunning && !client.isBusyWithCustomChunk && !client.isBusyWithMatrixTask) {
      const task = assignMatrixTask(clientId);
      if (task) client.emit('task:assign', task);
    }
  });

  // Matrix task completion
  client.on('task:complete', (data) => {
    if (!client || !client.connected) return;

    client.isBusyWithMatrixTask = false;
    client.lastActive = Date.now();

    const { assignmentId, taskId, result: received, processingTime, reportedChecksum } = data;
    const inst = matrixState.activeTasks.get(assignmentId);
    if (!inst || inst.logicalTaskId !== taskId || inst.assignedTo !== clientId) {
      return;
    }
    matrixState.activeTasks.delete(assignmentId);

    const tdef = matrixState.tasks.find(t => t.id === taskId);
    if (!tdef || tdef.status === 'completed') return;

    const serverChecksum = checksumMatrixRowsFloat32LE(received);

    if (!matrixResultBuffer.has(taskId)) matrixResultBuffer.set(taskId, []);
    matrixResultBuffer.get(taskId).push({
      clientId: clientId,
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
          if (cl) cl.emit('task:verified', { taskId, type: 'matrixMultiply' });
        });
        broadcastStatus();
      }
    } else {
      client.emit('task:submitted', { taskId, type: 'matrixMultiply' });
    }
  });

  // Non-chunked workload completion
  client.on('workload:done', ({ id, result, results, processingTime, reportedChecksum }) => {
    const wl = customWorkloads.get(id);
    if (!wl || wl.isChunkParent) {
      client.emit('workload:error', { id, message: 'Invalid workload ID or is chunk parent.' });
      return;
    }

    client.isBusyWithNonChunkedWGSL = false;
    wl.activeAssignments.delete(clientId);

    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) finalResults = [finalResults];

    const checksumData = checksumFromResults(finalResults);
    wl.results.push({
      clientId: clientId,
      results: finalResults,
      result: finalResults[0],
      submissionTime: Date.now(),
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength
    });
    wl.processingTimes.push({ clientId: clientId, timeMs: processingTime });

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

      // Broadcast to ALL clients (Socket.IO and WebSocket)
      io.emit('workload:complete', {
        id,
        label: wl.label,
        finalResults: wl.finalResults,
        finalResultBase64: wl.finalResultBase64
      });

      // Also broadcast to native WebSocket clients
      matrixState.clients.forEach(cl => {
        if (cl.clientType === 'native') {
          cl.emit('workload:complete', {
            id,
            label: wl.label,
            finalResults: wl.finalResults,
            finalResultBase64: wl.finalResultBase64
          });
        }
      });
    } else {
      wl.status = 'processing';
      console.log(`${wl.framework} ${id}: ${wl.results.length} submissions, awaiting ${ADMIN_K_PARAMETER}.`);
    }
    saveCustomWorkloads();
    broadcastCustomWorkloadList();
  });

  // Enhanced chunk completion
  client.on('workload:chunk_done_enhanced', ({ parentId, chunkId, results, result, processingTime, strategy, metadata, reportedChecksum }) => {
    console.log(`[CHUNK RESULT] Enhanced chunk ${chunkId} completed by native client ${clientId}`);

    client.isBusyWithCustomChunk = false;

    const workloadState = customWorkloads.get(parentId);
    const chunkStore = customWorkloadChunks.get(parentId);

    if (!workloadState || !chunkStore) {
      console.error(`[CHUNK RESULT] Workload ${parentId} not found for chunk ${chunkId}`);
      return;
    }

    if (!chunkStore.enhanced) {
      console.error(`[CHUNK RESULT] Chunk ${chunkId} received enhanced completion but store is not enhanced`);
      return;
    }

    const cd = chunkStore.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd) {
      console.warn(`[CHUNK RESULT] Enhanced chunk ${chunkId} not found in store for parent ${parentId}`);
      return;
    }

    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) finalResults = [finalResults];

    console.log(`[CHUNK RESULT] Processing ${finalResults.length} results for chunk ${chunkId}`);

    let checksumData;
    try {
      checksumData = checksumFromResults(finalResults);
      console.log(`[CHUNK RESULT] Chunk ${chunkId} checksum: ${checksumData.serverChecksum.slice(0, 8)}... (${checksumData.byteLength} bytes)`);
    } catch (err) {
      console.error(`[CHUNK RESULT] Enhanced chunk ${chunkId} from ${clientId} invalid base64:`, err);
      return;
    }

    const submission = {
      clientId: clientId,
      results: finalResults,
      processingTime,
      reportedChecksum: reportedChecksum,
      serverChecksum: checksumData.serverChecksum,
      byteLength: checksumData.byteLength,
      buffers: checksumData.buffers
    };

    const verifyRes = verifyAndRecordChunkSubmission(workloadState, chunkStore, cd, submission, cd.chunkOrderIndex, ADMIN_K_PARAMETER);

    if (verifyRes.verified) {
      console.log(`✅ Enhanced chunk ${chunkId} VERIFIED by K=${ADMIN_K_PARAMETER} (checksum ${verifyRes.winningChecksum.slice(0,8)}…)`);

      const verifiedResults = cd.verified_results;

      console.log(`[CHUNK RESULT] Calling chunkingManager.handleChunkCompletion for ${chunkId}`);

      try {
        const assemblyResult = chunkingManager.handleChunkCompletion(parentId, chunkId, verifiedResults, processingTime);
        console.log(`[CHUNK RESULT] Assembly result for ${chunkId}:`, {
          success: assemblyResult.success,
          status: assemblyResult.status,
          error: assemblyResult.error
        });

        if (assemblyResult.success && assemblyResult.status === 'complete') {
          console.log(`🎉 Enhanced workload ${parentId} COMPLETED!`);

          workloadState.status = 'complete';
          workloadState.completedAt = Date.now();
          workloadState.assemblyStats = assemblyResult.stats;

          let finalBase64 = null;
          if (assemblyResult.finalResult) {
            if (typeof assemblyResult.finalResult === 'string') {
              finalBase64 = assemblyResult.finalResult;
            } else if (assemblyResult.finalResult.data) {
              finalBase64 = typeof assemblyResult.finalResult.data === 'string'
                ? assemblyResult.finalResult.data
                : Buffer.from(assemblyResult.finalResult.data).toString('base64');
            }
          }

          workloadState.finalResultBase64 = finalBase64;

          console.log(`[CHUNK RESULT] Cleaning up workload ${parentId}`);
          try {
            chunkingManager.cleanupWorkload(parentId);
          } catch (cleanupError) {
            console.warn(`[CHUNK RESULT] Cleanup warning for ${parentId}:`, cleanupError.message);
          }

          customWorkloadChunks.delete(parentId);
          saveCustomWorkloads();
          saveCustomWorkloadChunks();

          console.log(`✅ Enhanced workload ${parentId} completed with ${assemblyResult.stats?.chunkingStrategy}/${assemblyResult.stats?.assemblyStrategy}`);

          // Broadcast to both Socket.IO and native WebSocket clients
          const completionData = {
            id: parentId,
            label: workloadState.label,
            finalResultBase64: finalBase64,
            finalResultUrl: finalBase64 ? null : `/api/workloads/${parentId}/download/final`,
            enhanced: true,
            stats: assemblyResult.stats
          };

          io.emit('workload:complete', completionData);

          matrixState.clients.forEach(cl => {
            if (cl.clientType === 'native') {
              cl.emit('workload:complete', completionData);
            }
          });
        } else if (!assemblyResult.success) {
          console.error(`❌ Enhanced chunk processing failed for ${parentId}: ${assemblyResult.error}`);
          workloadState.status = 'error';
          workloadState.error = assemblyResult.error;
          saveCustomWorkloads();
          saveCustomWorkloadChunks();
          broadcastCustomWorkloadList();
        } else {
          console.log(`⏳ Enhanced workload ${parentId} still processing (${assemblyResult.status})`);
          workloadState.status = 'processing_chunks';
          saveCustomWorkloads();
          saveCustomWorkloadChunks();
          broadcastCustomWorkloadList();
        }
      } catch (assemblyError) {
        console.error(`❌ Assembly error for chunk ${chunkId}:`, assemblyError);
        workloadState.status = 'error';
        workloadState.error = assemblyError.message;
        saveCustomWorkloads();
        saveCustomWorkloadChunks();
        broadcastCustomWorkloadList();
      }
    } else {
      console.log(`⏳ Enhanced chunk ${chunkId} waiting for more submissions (${cd.submissions?.length || 0}/${ADMIN_K_PARAMETER})`);
      workloadState.status = 'processing_chunks';
      saveCustomWorkloads();
      saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
    }
  });

  // Error handlers
  client.on('workload:busy', ({ id, reason }) => {
    client.isBusyWithNonChunkedWGSL = false;
    const wl = customWorkloads.get(id);
    if (wl && !wl.isChunkParent) {
      wl.activeAssignments.delete(clientId);
      if (wl.status !== 'complete') wl.status = 'pending_dispatch';
      console.warn(`WGSL ${id} declined by native client ${clientId} (${reason||'busy'})`);
      saveCustomWorkloads();
      saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
    }
    tryDispatchNonChunkedWorkloads();
  });

  client.on('workload:error', ({ id, message }) => {
    client.isBusyWithNonChunkedWGSL = false;
    const wl = customWorkloads.get(id);
    if (wl && !wl.isChunkParent) {
      wl.activeAssignments.delete(clientId);
      if (wl.status !== 'complete') wl.status = 'pending_dispatch';
      console.warn(`WGSL ${id} errored on native client ${clientId}: ${message}`);
      saveCustomWorkloads();
      saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
    }
    tryDispatchNonChunkedWorkloads();
  });

  client.on('workload:chunk_error', ({ parentId, chunkId, message }) => {
    client.isBusyWithCustomChunk = false;
    console.warn(`Chunk error ${chunkId} from native client ${clientId}: ${message}`);
    const store = customWorkloadChunks.get(parentId);
    if (store) {
      const cd = store.allChunkDefs.find(c => c.chunkId === chunkId);
      if (cd) {
        cd.activeAssignments.delete(clientId);
        const parent = customWorkloads.get(parentId);
        addProcessingTime(parent, {
          chunkId,
          clientId: clientId,
          error: message
        });
        saveCustomWorkloads();
        saveCustomWorkloadChunks();
        broadcastCustomWorkloadList();
      }
    }
  });

  console.log(`[WS-NATIVE] Client ${clientId} event handlers registered`);
});

// Update broadcastClientList to include native clients
function broadcastClientListEnhanced() {
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
    isPuppeteer: c.isPuppeteer || false,
    isBusyWithMatrixTask: c.isBusyWithMatrixTask,
    isBusyWithCustomChunk: c.isBusyWithCustomChunk,
    isBusyWithNonChunkedWGSL: c.isBusyWithNonChunkedWGSL
  }));

  // Broadcast to Socket.IO clients
  io.emit('clients:update', { clients: list });

  // Broadcast to native WebSocket clients
  matrixState.clients.forEach(client => {
    if (client.clientType === 'native' && client.connected) {
      client.emit('clients:update', { clients: list });
    }
  });
}

// Replace the original broadcastClientList function
const originalBroadcastClientList = broadcastClientList;
broadcastClientList = broadcastClientListEnhanced;

console.log('🔌 Raw WebSocket server initialized on /ws-native endpoint');


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
  function redactWorkload(w) {
    return {
      id: w.id,
      label: w.label,
      framework: w.framework,
      chunkingStrategy: w.chunkingStrategy,
      assemblyStrategy: w.assemblyStrategy,
      createdAt: w.createdAt,
      status: w.status,
      totalChunks: w.expectedChunks || (w.plan && w.plan.totalChunks) || 0,
      dispatched: w.dispatchesMade || 0,
      active: (w.activeAssignments && w.activeAssignments.size) || 0,
      enhanced: !!w.enhanced
    };
  }
  const summary = Array.from(customWorkloads.values()).map(redactWorkload);
  io.emit('workloads:list_update', summary);
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
    customAssemblyCode,
    streamingMode = false
  } = req.body;

  try {
    // Register custom strategies if provided
    if (req.body.customChunkingCode) {
      const result = chunkingManager.registerCustomStrategy(req.body.customChunkingCode, 'chunking', chunkingStrategy);
      if (!result.success) return res.status(400).json({ error: `Custom chunking strategy failed: ${result.error}` });
    }
    if (req.body.customAssemblyCode) {
      const result = chunkingManager.registerCustomStrategy(req.body.customAssemblyCode, 'assembly', assemblyStrategy);
      if (!result.success) return res.status(400).json({ error: `Custom assembly strategy failed: ${result.error}` });
    }

    const workloadId = uuidv4();

    // Create enhanced workload metadata
    const enhancedWorkload = {
      id: workloadId,
      label: label || `Enhanced Workload ${workloadId.substring(0, 6)}`,
      chunkingStrategy,
      assemblyStrategy,
      framework,
      metadata: {
        ...metadata,
        customShader,
        streamingMode,
        memoryThreshold: metadata?.memoryThreshold || 512 * 1024 * 1024,
        batchSize: metadata?.batchSize || 10
      },
      createdAt: Date.now(),
      isChunkParent: true,
      enhanced: true,
      streamingMode,
      inputRefs: [],
      outputSizes: req.body.outputSizes || []
    };

    // PHASE 1: ONLY validate and plan workload (no file access, no processing)
    const validationResult = await chunkingManager.validateAndPlanWorkload(enhancedWorkload);
    if (!validationResult.success) {
      return res.status(400).json(validationResult);
    }

    // Handle inline input if provided
    if (input) {
      try {
        const inputsDir = path.join(STORAGE_ROOT, workloadId, 'inputs');
        await ensureDir(inputsDir);
        const refs = [];

        let parsedInputs = {};
        try {
          parsedInputs = (typeof input === 'string') ? JSON.parse(input) : input;
        } catch (e) {
          parsedInputs = { input };
        }

        for (const [name, val] of Object.entries(parsedInputs || {})) {
          let buf = null;
          if (typeof val === 'string') {
            try { buf = Buffer.from(val, 'base64'); } catch { buf = Buffer.from(val); }
          } else if (val && val.type === 'Buffer' && Array.isArray(val.data)) {
            buf = Buffer.from(val.data);
          }
          if (!buf) continue;

          const sha = sha256Hex(buf);
          const fp = path.join(inputsDir, `${name}-${sha}.bin`);
          await fs.promises.writeFile(fp, buf);
          refs.push({ name, path: fp, sha256: sha, size: buf.length });
        }

        enhancedWorkload.inputRefs = refs;
        enhancedWorkload.status = 'queued';
      } catch (e) {
        console.warn('Failed to persist inline inputs:', e.message);
      }
    } else {
      enhancedWorkload.status = validationResult.requiresFileUpload ? 'awaiting_input' : 'queued';
    }

    // Store the plan for later use
    enhancedWorkload.plan = validationResult.plan;

    // Store workload (metadata only - no processing yet!)
    customWorkloads.set(workloadId, enhancedWorkload);
    saveCustomWorkloads();
    broadcastCustomWorkloadList();

    console.log(`✅ Created workload ${workloadId} (streaming: ${streamingMode}) - ready for compute-start`);

    return res.json({
      success: true,
      id: workloadId,
      status: enhancedWorkload.status,
      streamingMode,
      requiresFileUpload: validationResult.requiresFileUpload,
      message: enhancedWorkload.status === 'awaiting_input'
        ? `Workload created. Upload input files via POST /api/workloads/${workloadId}/inputs then start computation.`
        : `Workload created and ready to start.`
    });

  } catch (error) {
    console.error('Error creating advanced workload:', error);
    res.status(500).json({ error: error.message });
  }
});


app.get('/api/streaming/queues', (req, res) => {
  const queueInfo = {};

  for (const [workloadId, queue] of streamingChunkQueues.entries()) {
    const workload = customWorkloads.get(workloadId);
    queueInfo[workloadId] = {
      label: workload?.label || 'Unknown',
      queueSize: queue.length,
      oldestChunk: queue.length > 0 ? {
        chunkId: queue[0].descriptor.chunkId,
        queuedAt: queue[0].queuedAt,
        attempts: queue[0].attempts
      } : null,
      framework: queue.length > 0 ? queue[0].descriptor.framework : null
    };
  }

  res.json({
    totalQueues: streamingChunkQueues.size,
    totalQueuedChunks: Array.from(streamingChunkQueues.values()).reduce((sum, queue) => sum + queue.length, 0),
    queues: queueInfo
  });
});

// Enhanced compute-start endpoint - Process chunks AFTER files uploaded
app.post('/api/workloads/:id/compute-start', async (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);

  if (!wl) {
    return res.status(404).json({ error: 'Workload not found' });
  }

  if (!['queued', 'awaiting_input'].includes(wl.status)) {
    return res.status(400).json({ error: `Cannot start workload in status: ${wl.status}` });
  }

  // Extract streaming mode from request or workload
  const streamingMode = req.body.streamingMode || wl.streamingMode || false;
  const batchSize = req.body.batchSize || wl.metadata?.batchSize || 10;

  console.log(`🚀 [COMPUTE START] Starting workload ${wid} (streaming: ${streamingMode})`);

  try {
    // For enhanced workloads, process chunks
    if (wl.enhanced && wl.isChunkParent) {
      console.log(`📄 [COMPUTE START] Processing chunks for workload ${wid}...`);

      // CRITICAL: Set up streaming callbacks BEFORE processing if streaming mode
      if (streamingMode) {
        console.log(`🌊 [COMPUTE START] Setting up streaming callbacks for workload ${wid}`);

        // Set up dispatch callback for streaming chunk creation
        chunkingManager.setDispatchCallback(wid, async (chunkDescriptor) => {
          console.log(`📤 [STREAMING] Immediate dispatch: ${chunkDescriptor.chunkId} (framework: ${chunkDescriptor.framework})`);
          return await dispatchStreamingChunkEnhanced(wid, chunkDescriptor);
        });

        // Set up server callbacks for streaming events
        chunkingManager.setServerCallbacks({
          onWorkloadComplete: async (workloadId, result) => {
            await handleStreamingWorkloadComplete(workloadId, result);
          },
          onAssemblyProgress: (workloadId, progress) => {
            console.log(`📊 [STREAMING] Assembly progress: ${progress.completedBlocks}/${progress.totalBlocks} blocks`);

            io.emit('assembly:progress', {
              workloadId,
              completedBlocks: progress.completedBlocks,
              totalBlocks: progress.totalBlocks,
              progress: progress.progress
            });

            matrixState.clients.forEach(client => {
              if (client.clientType === 'native' && client.connected) {
                client.emit('assembly:progress', {
                  workloadId,
                  completedBlocks: progress.completedBlocks,
                  totalBlocks: progress.totalBlocks,
                  progress: progress.progress
                });
              }
            });
          }
        });

        console.log(`✅ [COMPUTE START] Streaming callbacks configured for ${wid}`);
      }

      // NOW process the chunks (with streaming setup ready)
      console.log(`⚡ [COMPUTE START] Calling processChunkedWorkload(streamingMode: ${streamingMode})`);
      const result = await chunkingManager.processChunkedWorkload(wl, streamingMode);

      if (!result.success) {
        console.error(`❌ [COMPUTE START] Processing failed:`, result.error);
        wl.status = 'error';
        wl.error = result.error;
        saveCustomWorkloads();
        return res.status(400).json(result);
      }

      console.log(`🎯 [COMPUTE START] Processing result:`, {
        success: result.success,
        totalChunks: result.totalChunks,
        streamingMode: result.streamingMode,
        message: result.message
      });

      // Update workload with results
      wl.chunkDescriptors = result.chunkDescriptors;
      wl.plan = result.plan;
      wl.streamingMode = streamingMode;

      if (streamingMode && result.streamingMode) {
        // Streaming mode: chunks are being created and dispatched dynamically
        wl.status = 'streaming_chunks';
        console.log(`🌊 [COMPUTE START] Streaming mode active - chunks will be dispatched as created`);
      } else {
        // Batch mode: create chunk store
        const store = {
          parentId: wid,
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
          expectedChunks: result.totalChunks,
          status: 'assigning_chunks',
          aggregationMethod: result.plan.assemblyStrategy,
          enhanced: true
        };

        customWorkloadChunks.set(wid, store);
        wl.status = 'assigning_chunks';
        console.log(`✅ [COMPUTE START] Created ${store.allChunkDefs.length} chunk descriptors (batch mode)`);
      }
    } else {
      // Non-chunked workload
      wl.status = 'pending_dispatch';
    }

    wl.startedAt = Date.now();
    saveCustomWorkloads();
    saveCustomWorkloadChunks();
    broadcastCustomWorkloadList();

    // Emit event
    io.emit('workload:started', {
      id: wl.id,
      label: wl.label,
      status: wl.status,
      startedAt: wl.startedAt,
      streamingMode,
      totalChunks: wl.plan?.totalChunks || wl.chunkDescriptors?.length || 0
    });

    console.log(`🎉 [COMPUTE START] Workload ${wid} started successfully (${streamingMode ? 'streaming' : 'batch'} mode)`);

    res.json({
      success: true,
      workloadId: wid,
      status: wl.status,
      streamingMode,
      totalChunks: wl.plan?.totalChunks || wl.chunkDescriptors?.length || 0,
      message: streamingMode
        ? `Streaming workload started - chunks dispatched as created`
        : `Batch workload started successfully`
    });

  } catch (error) {
    console.error('❌ [COMPUTE START] Error:', error);
    wl.status = 'error';
    wl.error = error.message;
    saveCustomWorkloads();

    res.status(500).json({ error: 'Failed to start computation: ' + error.message });
  }
});

async function dispatchStreamingChunk(workloadId, chunkDescriptor) {
  try {
    // Find available clients that support the framework
    const availableClients = Array.from(matrixState.clients.entries()).filter(([clientId, client]) =>
      client.connected &&
      client.gpuInfo &&
      !client.isBusyWithCustomChunk &&
      !client.isBusyWithMatrixTask &&
      client.supportedFrameworks &&
      client.supportedFrameworks.includes(chunkDescriptor.framework)
    );

    if (availableClients.length === 0) {
      console.log(`⏳ No available clients for streaming chunk ${chunkDescriptor.chunkId}, queuing...`);
      // Queue the chunk for later dispatch
      return await queueStreamingChunk(workloadId, chunkDescriptor);
    }

    // Dispatch to first available client
    const [clientId, client] = availableClients[0];
    return await dispatchChunkToClient(client, chunkDescriptor, workloadId);

  } catch (error) {
    console.error(`Failed to dispatch streaming chunk ${chunkDescriptor.chunkId}:`, error);
    throw error;
  }
}

async function dispatchChunkToClient(client, chunkDescriptor, workloadId) {
  console.log(`[CHUNK DISPATCH] === SENDING TO CLIENT ===`);
  console.log(`[CHUNK DISPATCH] Client ID: ${client.id.substring(0, 8)}...`);
  console.log(`[CHUNK DISPATCH] Client Type: ${client.clientType || 'browser'}`);
  console.log(`[CHUNK DISPATCH] Chunk ID: ${chunkDescriptor.chunkId}`);
  console.log(`[CHUNK DISPATCH] Framework: ${chunkDescriptor.framework}`);

  try {
    client.isBusyWithCustomChunk = true;

    // Create unified task data for the client (same as before but with better logging)
    const taskData = {
      parentId: chunkDescriptor.parentId,
      chunkId: chunkDescriptor.chunkId,
      chunkOrderIndex: chunkDescriptor.chunkIndex,
      framework: chunkDescriptor.framework,
      enhanced: true,
      streaming: true,

      // Shader code and execution parameters
      kernel: chunkDescriptor.kernel,
      wgsl: chunkDescriptor.kernel,
      entry: chunkDescriptor.entry || 'main',
      workgroupCount: chunkDescriptor.workgroupCount || [1, 1, 1],

      // Framework-specific properties
      webglShaderType: chunkDescriptor.webglShaderType,
      webglVertexShader: chunkDescriptor.webglVertexShader,
      webglFragmentShader: chunkDescriptor.webglFragmentShader,
      webglVaryings: chunkDescriptor.webglVaryings,
      webglNumElements: chunkDescriptor.webglNumElements,
      webglInputSpec: chunkDescriptor.webglInputSpec,

      blockDim: chunkDescriptor.blockDim,
      gridDim: chunkDescriptor.gridDim,
      globalWorkSize: chunkDescriptor.globalWorkSize,
      localWorkSize: chunkDescriptor.localWorkSize,
      shaderType: chunkDescriptor.shaderType,

      // Data and metadata
      inputs: chunkDescriptor.inputs || [],
      outputs: chunkDescriptor.outputs || [],
      metadata: chunkDescriptor.metadata || {},
      outputSizes: chunkDescriptor.outputs ? chunkDescriptor.outputs.map(o => o.size) : [],

      // Assembly metadata for streaming
      assemblyMetadata: chunkDescriptor.assemblyMetadata,

      // Streaming metadata
      streamingMetadata: chunkDescriptor.streamingMetadata
    };

    console.log(`🎯 [DISPATCH] Sending ${chunkDescriptor.framework} chunk ${chunkDescriptor.chunkId} to ${client.clientType || 'browser'} client ${client.id}`);

    // Send to appropriate client type with timeout
    const dispatchPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Dispatch timeout'));
      }, 30000); // 30 second timeout

      try {
        if (client.socket) {
          client.socket.emit('workload:chunk_assign', taskData);
          clearTimeout(timeout);
          resolve({ success: true });
        } else if (client.emit) {
          client.emit('workload:chunk_assign', taskData);
          clearTimeout(timeout);
          resolve({ success: true });
        } else {
          clearTimeout(timeout);
          reject(new Error(`Client ${client.id} has no dispatch method`));
        }
      } catch (error) {
        clearTimeout(timeout);
        reject(error);
      }
    });

    const result = await dispatchPromise;

    // Log successful dispatch
    updateStreamingWorkloadProgress(workloadId, chunkDescriptor.chunkId, 'dispatched');

    return result;

  } catch (error) {
    client.isBusyWithCustomChunk = false;
    console.error(`❌ [DISPATCH] Failed to dispatch to client ${client.id}:`, error);
    throw error;
  }
}

function assignCustomChunkToAvailableClientsEnhanced() {
  const availableClients = Array.from(matrixState.clients.entries()).filter(([clientId, client]) =>
    client.connected &&
    client.gpuInfo &&
    !client.isBusyWithCustomChunk &&
    !client.isBusyWithMatrixTask &&
    (client.socket || client.emit)
  );

  for (const [clientId, client] of availableClients) {
    // NEW: First check for streaming workloads
    let assigned = false;

    for (const parent of customWorkloads.values()) {
      if (parent.streamingMode && parent.status === 'streaming_chunks') {
        // For streaming workloads, chunks are dispatched via the callback system
        // This loop mainly handles any queued chunks that couldn't be dispatched immediately
        continue;
      }

      // Handle regular batch workloads (existing logic)
      if (!parent.isChunkParent || !['assigning_chunks', 'processing_chunks'].includes(parent.status)) continue;

      // Check if client supports the framework
      if (!client.supportedFrameworks || !client.supportedFrameworks.includes(parent.framework)) {
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

          // Create task data (existing logic)
          const taskData = {
            parentId: cd.parentId,
            chunkId: cd.chunkId,
            chunkOrderIndex: cd.chunkOrderIndex,
            framework: parent.framework,
            enhanced: parent.enhanced,
            streaming: false,  // Mark as batch chunk

            kernel: cd.kernel || parent.metadata?.customShader,
            wgsl: cd.wgsl || cd.kernel || parent.metadata?.customShader,
            entry: cd.entry || 'main',
            workgroupCount: cd.workgroupCount || [1, 1, 1],

            // Framework-specific properties (existing)
            webglShaderType: cd.webglShaderType,
            webglVertexShader: cd.webglVertexShader,
            webglFragmentShader: cd.webglFragmentShader,
            webglVaryings: cd.webglVaryings,
            webglNumElements: cd.webglNumElements,
            webglInputSpec: cd.webglInputSpec,

            blockDim: cd.blockDim,
            gridDim: cd.gridDim,
            globalWorkSize: cd.globalWorkSize,
            localWorkSize: cd.localWorkSize,
            shaderType: cd.shaderType,

            inputs: cd.inputs || [],
            outputs: cd.outputs || [],
            metadata: cd.metadata || {},
            outputSizes: cd.outputs ? cd.outputs.map(o => o.size) : [cd.outputSize],
            outputSize: cd.outputs ? cd.outputs[0]?.size : cd.outputSize,

            chunkingStrategy: parent.chunkingStrategy,
            assemblyStrategy: parent.assemblyStrategy
          };

          // Dispatch to client
          if (client.socket) {
            client.socket.emit('workload:chunk_assign', taskData);
          } else if (client.emit) {
            client.emit('workload:chunk_assign', taskData);
          }

          console.log(`✅ Assigned batch chunk ${cd.chunkId} to ${client.clientType || 'browser'} client ${clientId}`);
          assigned = true;
          break;
        }
      }

      if (assigned) break;
    }
  }
}

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


// --- Binary input upload (multipart) ---
app.post('/api/workloads/:id/inputs', async (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);

  if (!wl) {
    return res.status(404).json({ error: 'Workload not found' });
  }

  if (wl.status !== 'awaiting_input' && wl.status !== 'queued') {
    return res.status(400).json({ error: `Cannot upload files to workload in status: ${wl.status}` });
  }

  const inputsDir = path.join(STORAGE_ROOT, wid, 'inputs');
  await ensureDir(inputsDir);

  const uploadedFiles = [];
  const metadata = {};
  let totalBytesReceived = 0;
  const maxFileSize = 1024 * 1024 * 1024; // 1GB limit per file

  try {
    const bb = Busboy({
      headers: req.headers,
      limits: {
        fileSize: maxFileSize,
        files: 10 // Max 10 files
      }
    });

    // Handle file uploads with streaming
    bb.on('file', (fieldname, fileStream, info) => {
      const { filename, encoding, mimeType } = info;
      const sanitizedFilename = path.basename(filename || `${fieldname}_${Date.now()}.bin`);
      const outputPath = path.join(inputsDir, sanitizedFilename);

      console.log(`📁 Receiving file: ${sanitizedFilename} (${encoding}, ${mimeType})`);

      const hash = crypto.createHash('sha256');
      const writeStream = fs.createWriteStream(outputPath);
      let fileSize = 0;

      fileStream.on('data', (chunk) => {
        totalBytesReceived += chunk.length;
        fileSize += chunk.length;
        hash.update(chunk);

        // Check file size limit
        if (fileSize > maxFileSize) {
          fileStream.destroy();
          writeStream.destroy();
          throw new Error(`File ${sanitizedFilename} exceeds size limit`);
        }
      });

      fileStream.on('error', (err) => {
        console.error(`File stream error for ${sanitizedFilename}:`, err);
        writeStream.destroy();
      });

      fileStream.on('end', () => {
        const sha256 = hash.digest('hex');
        uploadedFiles.push({
          name: fieldname,
          filename: sanitizedFilename,
          path: outputPath,
          sha256,
          size: fileSize,
          encoding,
          mimeType
        });
        console.log(`✅ File saved: ${sanitizedFilename} (${fileSize} bytes, SHA256: ${sha256.slice(0, 8)}...)`);
      });

      // Pipe with error handling
      fileStream.pipe(writeStream);
    });

    // Handle form fields (metadata)
    bb.on('field', (fieldname, value) => {
      metadata[fieldname] = value;
    });

    // Handle completion
    bb.on('close', async () => {
      try {
        // Update workload with file references
        wl.inputRefs = (wl.inputRefs || []).concat(uploadedFiles);
        wl.uploadMetadata = { ...wl.uploadMetadata, ...metadata };

        // Clean up inline input data to save memory
        delete wl.input;

        // Validate uploaded files using the chunking manager
        if (wl.enhanced && chunkingManager) {
          try {
            const validation = await chunkingManager.validateInputFiles?.(wl, uploadedFiles);
            if (validation && !validation.valid) {
              return res.status(400).json({
                error: 'Input validation failed',
                details: validation.errors
              });
            }
          } catch (validationError) {
            console.warn('Input validation error:', validationError.message);
          }
        }

        // Update status to ready for computation
        if (wl.status === 'awaiting_input') {
          wl.status = 'queued';
        }

        // Save workload state
        saveCustomWorkloads();
        broadcastCustomWorkloadList();

        res.json({
          success: true,
          workloadId: wid,
          files: uploadedFiles.map(f => ({
            name: f.name,
            filename: f.filename,
            size: f.size,
            sha256: f.sha256
          })),
          metadata,
          totalBytes: totalBytesReceived,
          status: wl.status,
          message: `${uploadedFiles.length} files uploaded successfully. Ready to start computation.`
        });

        console.log(`📦 Upload complete for workload ${wid}: ${uploadedFiles.length} files, ${totalBytesReceived} total bytes`);

      } catch (error) {
        console.error('Upload completion error:', error);
        res.status(500).json({ error: 'Failed to process uploaded files: ' + error.message });
      }
    });

    // Handle errors
    bb.on('error', (error) => {
      console.error('Busboy error:', error);
      res.status(400).json({ error: 'Upload failed: ' + error.message });
    });

    // Start processing the upload
    req.pipe(bb);

  } catch (error) {
    console.error('Upload setup error:', error);
    res.status(500).json({ error: 'Failed to setup file upload: ' + error.message });
  }
});

// --- Stream chunk input bytes (binary) ---
app.get('/api/workloads/:wid/chunks/:cid/input/:idx', async (req, res) => {
  const { wid, cid, idx } = req.params;
  const wl = customWorkloads.get(wid);
  if (!wl) return res.status(404).end();
  // For matrix_tiled, we have a single combined input named 'input'
  const ref = (wl.inputRefs || []).find(r => r.name === 'input') || (wl.inputRefs || [])[0];
  if (!ref || !ref.path) return res.status(404).end();
  res.setHeader('Content-Type', 'application/octet-stream');
  const rs = fs.createReadStream(ref.path);
  rs.on('error', () => res.status(500).end());
  rs.pipe(res);
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
// STEP 3: Update the startQueued endpoint to handle the new flow
app.post('/api/workloads/startQueued', (req, res) => {
  let activatedChunkParents = 0;
  let startedNonChunked = 0;
  let awaitingInput = 0;

  customWorkloads.forEach(async (wl) => {
    if (wl.status === 'queued') {
      try {
        // Trigger compute-start for each queued workload
        const computeStartUrl = `/api/workloads/${wl.id}/compute-start`;
        const startResult = await fetch(`http://localhost:${PORT}${computeStartUrl}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });

        if (startResult.ok) {
          if (wl.isChunkParent) {
            activatedChunkParents++;
          } else {
            startedNonChunked++;
          }
        }
      } catch (error) {
        console.error(`Failed to start workload ${wl.id}:`, error.message);
      }
    } else if (wl.status === 'awaiting_input') {
      awaitingInput++;
    }
  });

  res.json({
    ok: true,
    activatedChunkParents,
    startedNonChunked,
    awaitingInput,
    message: awaitingInput > 0 ? `${awaitingInput} workloads need input files uploaded first` : 'All queued workloads started'
  });
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
    (client.socket || client.emit) // Support both Socket.IO and native WebSocket clients
  );

  for (const [clientId, client] of availableClients) {
    for (const parent of customWorkloads.values()) {
      if (!parent.isChunkParent || !['assigning_chunks', 'processing_chunks'].includes(parent.status)) continue;

      // Check if client supports the framework
      if (!client.supportedFrameworks || !client.supportedFrameworks.includes(parent.framework)) {
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

          // Debug: Log what we're about to send
          console.log(`[CHUNK DEBUG] Chunk ${cd.chunkId} structure:`, {
            hasInputs: !!cd.inputs,
            inputsLength: cd.inputs ? cd.inputs.length : 0,
            hasOutputs: !!cd.outputs,
            outputsLength: cd.outputs ? cd.outputs.length : 0,
            hasSchema: !!cd.schema,
            hasMetadata: !!cd.metadata,
            clientType: client.clientType || 'unknown'
          });

          // Create unified task data that the client expects
          const taskData = {
            // Basic chunk info
            parentId: cd.parentId,
            chunkId: cd.chunkId,
            chunkOrderIndex: cd.chunkOrderIndex,
            framework: parent.framework,
            enhanced: parent.enhanced,

            // Shader code (WebGPU)
            kernel: cd.kernel || parent.metadata?.customShader,
            wgsl: cd.wgsl || cd.kernel || parent.metadata?.customShader,
            entry: cd.entry || 'main',
            workgroupCount: cd.workgroupCount || [1, 1, 1],

            // WebGL-specific properties (copy from chunk descriptor)
            webglShaderType: cd.webglShaderType,
            webglVertexShader: cd.webglVertexShader,
            webglFragmentShader: cd.webglFragmentShader,
            webglVaryings: cd.webglVaryings,
            webglNumElements: cd.webglNumElements,
            webglInputSpec: cd.webglInputSpec,

            // CUDA-specific properties
            blockDim: cd.blockDim,
            gridDim: cd.gridDim,
            cudaKernel: cd.cudaKernel,

            // OpenCL-specific properties
            globalWorkSize: cd.globalWorkSize,
            localWorkSize: cd.localWorkSize,
            openclKernel: cd.openclKernel,

            // Vulkan-specific properties
            vulkanShader: cd.vulkanShader,
            computeShader: cd.computeShader,
            specializationConstants: cd.specializationConstants,
            pushConstants: cd.pushConstants,

            // Data and outputs
            inputs: cd.inputs || [],           // Array of {name, data} objects
            outputs: cd.outputs || [],         // Array of {name, size} objects
            metadata: cd.metadata || {},       // Strategy-specific uniforms data
            schema: cd.schema,                 // Binding schema

            // Legacy compatibility
            outputSizes: cd.outputs ? cd.outputs.map(o => o.size) : [cd.outputSize],
            outputSize: cd.outputs ? cd.outputs[0]?.size : cd.outputSize,

            // Debug info
            chunkingStrategy: parent.chunkingStrategy,
            assemblyStrategy: parent.assemblyStrategy
          };

          // Debug: Log the complete task data being sent
          console.log(`[CHUNK DEBUG] Sending chunk ${cd.chunkId} to ${clientId} (${client.clientType || 'browser'}):`, {
            inputs: taskData.inputs?.length,
            outputs: taskData.outputs?.length,
            hasKernel: !!taskData.kernel,
            hasSchema: !!taskData.schema,
            workgroupCount: taskData.workgroupCount,
            framework: parent.framework
          });

          console.log(`[SERVER DEBUG] Sending chunk ${cd.chunkId} to ${clientId}:`);
          console.log(`[SERVER DEBUG] Framework: ${parent.framework}`);
          console.log(`[SERVER DEBUG] Client type: ${client.clientType || 'browser'}`);
          console.log(`[SERVER DEBUG] TaskData keys:`, Object.keys(taskData));

          // Framework-specific debug logging
          if (parent.framework === 'webgl') {
            console.log(`[SERVER DEBUG] Has webglVertexShader:`, !!taskData.webglVertexShader);
            console.log(`[SERVER DEBUG] Has webglFragmentShader:`, !!taskData.webglFragmentShader);
            if (taskData.webglVertexShader) {
              console.log(`[SERVER DEBUG] WebGL vertex shader length:`, taskData.webglVertexShader.length);
              console.log(`[SERVER DEBUG] WebGL vertex shader preview:`, taskData.webglVertexShader.substring(0, 100) + '...');
            }
          } else if (parent.framework === 'vulkan') {
            console.log(`[SERVER DEBUG] Has vulkanShader:`, !!taskData.vulkanShader);
            console.log(`[SERVER DEBUG] Has computeShader:`, !!taskData.computeShader);
          } else if (parent.framework === 'cuda') {
            console.log(`[SERVER DEBUG] Has cudaKernel:`, !!taskData.cudaKernel);
            console.log(`[SERVER DEBUG] blockDim:`, taskData.blockDim);
            console.log(`[SERVER DEBUG] gridDim:`, taskData.gridDim);
          } else if (parent.framework === 'opencl') {
            console.log(`[SERVER DEBUG] Has openclKernel:`, !!taskData.openclKernel);
            console.log(`[SERVER DEBUG] globalWorkSize:`, taskData.globalWorkSize);
          }

          console.log(`[SERVER DEBUG] Has kernel:`, !!taskData.kernel);

          // Send chunk assignment based on client type
          if (client.socket) {
            // Socket.IO client (browser)
            console.log(`[DISPATCH] Sending to Socket.IO client ${clientId}`);
            client.socket.emit('workload:chunk_assign', taskData);
          } else if (client.emit) {
            // Native WebSocket client (C++/native)
            console.log(`[DISPATCH] Sending to native WebSocket client ${clientId}`);
            client.emit('workload:chunk_assign', taskData);
          } else {
            // Fallback: try both methods (shouldn't happen with updated filter)
            console.warn(`[DISPATCH] Client ${clientId} has no emit method, attempting fallback`);
            if (typeof client.send === 'function') {
              // Try direct WebSocket send with JSON message format
              const message = JSON.stringify({
                type: 'workload:chunk_assign',
                data: taskData
              });
              client.send(message);
            } else {
              console.error(`[DISPATCH] No viable dispatch method for client ${clientId}`);
              // Revert chunk assignment state
              cd.dispatchesMade--;
              cd.activeAssignments.delete(clientId);
              cd.assignedClients.delete(clientId);
              cd.status = 'queued';
              cd.assignedTo = null;
              cd.assignedAt = null;
              client.isBusyWithCustomChunk = false;
              continue;
            }
          }

          const inputCount = taskData.inputs?.length || 0;
          const outputCount = taskData.outputs?.length || 0;
          const clientTypeStr = client.clientType || 'browser';

          console.log(`✅ Assigned ${parent.framework} chunk ${cd.chunkId} to ${clientTypeStr} client ${clientId} (${inputCount} inputs, ${outputCount} outputs)`);

          // Emit assignment event for monitoring
          try {
            io.emit('chunk:assigned', {
              chunkId: cd.chunkId,
              parentId: cd.parentId,
              clientId: clientId,
              clientType: clientTypeStr,
              framework: parent.framework,
              timestamp: Date.now()
            });
          } catch (emitError) {
            console.warn(`[DISPATCH] Failed to emit assignment event:`, emitError.message);
          }

          break;
        }
      }

      if (client.isBusyWithCustomChunk) break;
    }
  }
}

// Enhanced function to dispatch to both client types with framework filtering
function dispatchToAllClients(eventName, eventData, frameworkFilter = null) {
  let socketIoCount = 0;
  let nativeWsCount = 0;

  matrixState.clients.forEach((client, clientId) => {
    if (!client.connected) return;

    // Apply framework filter if specified
    if (frameworkFilter && (!client.supportedFrameworks || !client.supportedFrameworks.includes(frameworkFilter))) {
      return;
    }

    try {
      if (client.socket) {
        // Socket.IO client
        client.socket.emit(eventName, eventData);
        socketIoCount++;
      } else if (client.emit) {
        // Native WebSocket client
        client.emit(eventName, eventData);
        nativeWsCount++;
      }
    } catch (error) {
      console.warn(`[DISPATCH] Failed to send ${eventName} to client ${clientId}:`, error.message);
    }
  });

  if (frameworkFilter) {
    console.log(`[DISPATCH] Sent ${eventName} to ${socketIoCount} Socket.IO + ${nativeWsCount} native clients supporting ${frameworkFilter}`);
  } else {
    console.log(`[DISPATCH] Sent ${eventName} to ${socketIoCount} Socket.IO + ${nativeWsCount} native clients`);
  }
}

// Helper function to get client statistics
function getClientStatistics() {
  const stats = {
    total: 0,
    socketIO: 0,
    nativeWS: 0,
    byFramework: {},
    busy: { matrix: 0, chunk: 0, workload: 0 }
  };

  matrixState.clients.forEach(client => {
    if (!client.connected) return;

    stats.total++;

    if (client.socket) {
      stats.socketIO++;
    } else if (client.emit) {
      stats.nativeWS++;
    }

    // Count by framework
    if (client.supportedFrameworks) {
      client.supportedFrameworks.forEach(framework => {
        stats.byFramework[framework] = (stats.byFramework[framework] || 0) + 1;
      });
    }

    // Count busy clients
    if (client.isBusyWithMatrixTask) stats.busy.matrix++;
    if (client.isBusyWithCustomChunk) stats.busy.chunk++;
    if (client.isBusyWithNonChunkedWGSL) stats.busy.workload++;
  });

  return stats;
}

// Enhanced tryDispatchNonChunkedWorkloads for dual client support
function tryDispatchNonChunkedWorkloads() { // Enhanced
  for (const [clientId, client] of matrixState.clients.entries()) {
    if (!client.connected || !client.gpuInfo || client.isBusyWithCustomChunk ||
        client.isBusyWithMatrixTask || client.isBusyWithNonChunkedWGSL ||
        (!client.socket && !client.emit)) {
      continue;
    }

    for (const wl of customWorkloads.values()) {
      if (!wl.isChunkParent && ['pending_dispatch', 'pending'].includes(wl.status)
        && !wl.finalResultBase64 && wl.dispatchesMade < ADMIN_K_PARAMETER
        && !wl.activeAssignments.has(clientId)) {

        if (!client.supportedFrameworks || !client.supportedFrameworks.includes(wl.framework)) {
          continue;
        }

        wl.dispatchesMade++;
        wl.activeAssignments.add(clientId);
        client.isBusyWithNonChunkedWGSL = true;

        const clientTypeStr = client.clientType || 'browser';
        console.log(`Dispatching ${wl.framework} workload ${wl.label} to ${clientTypeStr} client ${clientId}`);

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

        // Send based on client type
        if (client.socket) {
          client.socket.emit('workload:new', taskData);
        } else if (client.emit) {
          client.emit('workload:new', taskData);
        }

        break;
      }
    }
  }
}

// Replace the original function
//const originalTryDispatchNonChunkedWorkloads = tryDispatchNonChunkedWorkloads;
//function tryDispatchNonChunkedWorkloads = tryDispatchNonChunkedWorkloadsEnhanced;


function debugMatrixTiledChunk(chunkDef, parentWorkload) {
  console.log(`[MATRIX TILED DEBUG] Chunk ${chunkDef.chunkId}:`, {
    hasInputs: !!chunkDef.inputs,
    inputNames: chunkDef.inputs?.map(i => i.name) || [],
    hasOutputs: !!chunkDef.outputs,
    outputSizes: chunkDef.outputs?.map(o => o.size) || [],
    hasMetadata: !!chunkDef.metadata,
    metadataKeys: Object.keys(chunkDef.metadata || {}),
    hasKernel: !!(chunkDef.kernel || parentWorkload.metadata?.customShader)
  });
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
/*
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
*/

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
function ensureProcessingTimes(workload) {
  if (!workload.processingTimes) {
    workload.processingTimes = [];
  }
  return workload;
}

// Helper function to safely add processing time
function addProcessingTime(workload, timeData) {
  if (!workload) return;
  ensureProcessingTimes(workload);
  workload.processingTimes.push(timeData);
}

// Fixed checkTaskTimeouts function
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

        // FIXED: Use helper function to safely add processing time
        const parent = customWorkloads.get(cd.parentId);
        addProcessingTime(parent, {
          chunkId: cd.chunkId,
          error: 'timeout',
          assignedTo: timedOutClient,
          timedOutAt: now
        });
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
    console.log(`[CHUNK RESULT] Enhanced chunk ${chunkId} completed by ${socket.id}`);

    const client = matrixState.clients.get(socket.id);
    if (client) client.isBusyWithCustomChunk = false;
    updateStreamingWorkloadProgress(parentId, chunkId, 'completed');
    const workloadState = customWorkloads.get(parentId);
    const chunkStore = customWorkloadChunks.get(parentId);

    if (!workloadState || !chunkStore) {
      console.error(`[CHUNK RESULT] Workload ${parentId} not found for chunk ${chunkId}`);
      return;
    }

    if (!chunkStore.enhanced) {
      console.error(`[CHUNK RESULT] Chunk ${chunkId} received enhanced completion but store is not enhanced`);
      return;
    }

    const cd = chunkStore.allChunkDefs.find(c => c.chunkId === chunkId);
    if (!cd) {
      console.warn(`[CHUNK RESULT] Enhanced chunk ${chunkId} not found in store for parent ${parentId}`);
      return;
    }

    let finalResults = results || [result];
    if (!Array.isArray(finalResults)) finalResults = [finalResults];

    console.log(`[CHUNK RESULT] Processing ${finalResults.length} results for chunk ${chunkId}`);

    let checksumData;
    try {
      checksumData = checksumFromResults(finalResults);
      console.log(`[CHUNK RESULT] Chunk ${chunkId} checksum: ${checksumData.serverChecksum.slice(0, 8)}... (${checksumData.byteLength} bytes)`);
    } catch (err) {
      console.error(`[CHUNK RESULT] Enhanced chunk ${chunkId} from ${socket.id} invalid base64:`, err);
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
      console.log(`✅ Enhanced chunk ${chunkId} VERIFIED by K=${ADMIN_K_PARAMETER} (checksum ${verifyRes.winningChecksum.slice(0,8)}…)`);

      const verifiedResults = cd.verified_results;

      // Use the enhanced chunking manager for completion handling
      console.log(`[CHUNK RESULT] Calling chunkingManager.handleChunkCompletion for ${chunkId}`);

      try {
        const assemblyResult = chunkingManager.handleChunkCompletion(parentId, chunkId, verifiedResults, processingTime);
        console.log(`[CHUNK RESULT] Assembly result for ${chunkId}:`, {
          success: assemblyResult.success,
          status: assemblyResult.status,
          error: assemblyResult.error
        });

        if (assemblyResult.success && assemblyResult.status === 'complete') {
          console.log(`🎉 Enhanced workload ${parentId} COMPLETED!`);

          workloadState.status = 'complete';
          workloadState.completedAt = Date.now();
          workloadState.assemblyStats = assemblyResult.stats;

          // Extract final result data
          let finalBase64 = null;
          if (assemblyResult.finalResult) {
            if (typeof assemblyResult.finalResult === 'string') {
              finalBase64 = assemblyResult.finalResult;
            } else if (assemblyResult.finalResult.data) {
              finalBase64 = typeof assemblyResult.finalResult.data === 'string'
                ? assemblyResult.finalResult.data
                : Buffer.from(assemblyResult.finalResult.data).toString('base64');
            }
          }

          workloadState.finalResultBase64 = finalBase64;

          // Clean up
          console.log(`[CHUNK RESULT] Cleaning up workload ${parentId}`);
          try {
            chunkingManager.cleanupWorkload(parentId);
          } catch (cleanupError) {
            console.warn(`[CHUNK RESULT] Cleanup warning for ${parentId}:`, cleanupError.message);
          }

          customWorkloadChunks.delete(parentId);
          saveCustomWorkloads();
          saveCustomWorkloadChunks();

          console.log(`✅ Enhanced workload ${parentId} completed with ${assemblyResult.stats?.chunkingStrategy}/${assemblyResult.stats?.assemblyStrategy}`);

          io.emit('workload:complete', {
            id: parentId,
            label: workloadState.label,
            finalResultBase64: finalBase64,
            finalResultUrl: finalBase64 ? null : `/api/workloads/${parentId}/download/final`,
            enhanced: true,
            stats: assemblyResult.stats
          });
        } else if (!assemblyResult.success) {
          console.error(`❌ Enhanced chunk processing failed for ${parentId}: ${assemblyResult.error}`);
          workloadState.status = 'error';
          workloadState.error = assemblyResult.error;
          saveCustomWorkloads();
          saveCustomWorkloadChunks();
          broadcastCustomWorkloadList();
        } else {
          console.log(`⏳ Enhanced workload ${parentId} still processing (${assemblyResult.status})`);
          workloadState.status = 'processing_chunks';
          saveCustomWorkloads();
          saveCustomWorkloadChunks();
          broadcastCustomWorkloadList();
        }
      } catch (assemblyError) {
        console.error(`❌ Assembly error for chunk ${chunkId}:`, assemblyError);
        workloadState.status = 'error';
        workloadState.error = assemblyError.message;
        saveCustomWorkloads();
        saveCustomWorkloadChunks();
        broadcastCustomWorkloadList();
      }
    } else {
      console.log(`⏳ Enhanced chunk ${chunkId} waiting for more submissions (${cd.submissions?.length || 0}/${ADMIN_K_PARAMETER})`);
      workloadState.status = 'processing_chunks';
      saveCustomWorkloads();
      saveCustomWorkloadChunks();
      broadcastCustomWorkloadList();
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

    addProcessingTime(parent, { clientId: socket.id, chunkId, timeMs: processingTime });

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

    //parent.processingTimes.push();
    addProcessingTime(parent, { clientId: submission.clientId, chunkId, timeMs: processingTime })
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

        // FIXED: Use helper function to safely add processing time
        addProcessingTime(parent, {
          chunkId,
          clientId: socket.id,
          error: message
        });

        saveCustomWorkloads();
        saveCustomWorkloadChunks();
        broadcastCustomWorkloadList();
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

  setInterval(async () => {
  assignTasksToAvailableClients();
  assignCustomChunkToAvailableClientsEnhanced();
  tryDispatchNonChunkedWorkloads();

  // Process queued streaming chunks (now async)
  try {
    await processStreamingQueues();
  } catch (error) {
    console.error('Error processing streaming queues:', error);
  }
}, 1000);
});

app.get('/api/streaming/status', (req, res) => {
  const streamingStatus = {
    activeQueues: streamingChunkQueues.size,
    totalQueuedChunks: Array.from(streamingChunkQueues.values()).reduce((sum, queue) => sum + queue.length, 0),
    activeStreamingWorkloads: Array.from(customWorkloads.values()).filter(w => w.streamingMode && w.status !== 'complete').length,
    queueDetails: {}
  };

  // Add detailed queue information
  for (const [workloadId, queue] of streamingChunkQueues.entries()) {
    const workload = customWorkloads.get(workloadId);
    streamingStatus.queueDetails[workloadId] = {
      label: workload?.label || 'Unknown',
      framework: workload?.framework || 'Unknown',
      queueSize: queue.length,
      oldestChunkAge: queue.length > 0 ? Date.now() - Math.min(...queue.map(item => item.queuedAt)) : 0,
      averageAttempts: queue.length > 0 ? queue.reduce((sum, item) => sum + item.attempts, 0) / queue.length : 0,
      progress: workload?.streamingProgress || null
    };
  }

  res.json(streamingStatus);
});

app.post('/api/streaming/process-queues', async (req, res) => {
  try {
    const startTime = Date.now();
    await processStreamingQueues();
    const processingTime = Date.now() - startTime;

    res.json({
      success: true,
      processingTime,
      remainingQueues: streamingChunkQueues.size,
      totalQueued: Array.from(streamingChunkQueues.values()).reduce((sum, queue) => sum + queue.length, 0)
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// --- Download final result if stored on disk ---
app.get('/api/workloads/:id/download/final', (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);
  if (!wl) return res.status(404).json({ error: 'not found' });
  const outPath = wl.finalResultPath || (wl.metadata && wl.metadata.outputPath);
  if (!outPath || !fs.existsSync(outPath)) return res.status(404).json({ error: 'no final file' });
  res.setHeader('Content-Type', 'application/octet-stream');
  fs.createReadStream(outPath).pipe(res);
});

app.get('/api/workloads/:id/status', (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);

  if (!wl) {
    return res.status(404).json({ error: 'Workload not found' });
  }

  const store = customWorkloadChunks.get(wid);
  const status = {
    id: wid,
    label: wl.label,
    status: wl.status,
    framework: wl.framework,
    chunkingStrategy: wl.chunkingStrategy,
    assemblyStrategy: wl.assemblyStrategy,
    createdAt: wl.createdAt,
    startedAt: wl.startedAt,
    completedAt: wl.completedAt,
    enhanced: wl.enhanced,
    streamingMode: wl.streamingMode,  // NEW: Include streaming mode
    isChunkParent: wl.isChunkParent,
    inputFiles: (wl.inputRefs || []).map(ref => ({
      name: ref.name,
      size: ref.size,
      sha256: ref.sha256
    })),
    requiresInput: wl.status === 'awaiting_input',
    canStart: wl.status === 'queued',
    error: wl.error
  };

  if (store) {
    status.chunks = {
      total: store.expectedChunks,
      completed: store.completedChunksData.size,
      active: Array.from(store.allChunkDefs).filter(cd => cd.status === 'active').length,
      progress: store.expectedChunks > 0 ? (store.completedChunksData.size / store.expectedChunks * 100) : 0
    };
  }

  // NEW: Add streaming-specific information
  if (wl.streamingMode) {
    const workloadProgress = chunkingManager.getWorkloadProgress(wid);
    if (workloadProgress) {
      status.streaming = {
        dispatchedChunks: workloadProgress.dispatchedChunks || 0,
        completedChunks: workloadProgress.completedChunks || 0,
        totalChunks: workloadProgress.totalChunks || 0,
        progress: workloadProgress.progress || 0
      };
    }
  }

  res.json(status);
});

async function queueStreamingChunk(workloadId, chunkDescriptor) {
  console.log(`📋 [QUEUE] Queuing chunk ${chunkDescriptor.chunkId} - no available clients`);

  // Initialize queue for this workload if it doesn't exist
  if (!streamingChunkQueues.has(workloadId)) {
    streamingChunkQueues.set(workloadId, []);
  }

  const queue = streamingChunkQueues.get(workloadId);

  // Check queue size limit with dynamic scaling
  const workload = customWorkloads.get(workloadId);
  const dynamicMaxQueueSize = workload?.metadata?.maxQueueSize || Math.min(maxQueueSize, 100);

  if (queue.length >= dynamicMaxQueueSize) {
    // Remove oldest items to make room (FIFO)
    const removed = queue.splice(0, Math.ceil(dynamicMaxQueueSize * 0.1)); // Remove 10%
    console.warn(`⚠️  [QUEUE] Queue full for ${workloadId}, removed ${removed.length} oldest chunks`);
  }

  // Add to queue with enhanced metadata
  queue.push({
    descriptor: chunkDescriptor,
    queuedAt: Date.now(),
    attempts: 0,
    workloadId,
    priority: chunkDescriptor.streamingMetadata?.tileProgress || 0
  });

  console.log(`📊 [QUEUE] Workload ${workloadId} queue: ${queue.length} chunks waiting`);

  return {
    success: true,
    queued: true,
    queueSize: queue.length,
    position: queue.length
  };
}

// ADD: Function to process queued streaming chunks
async function processStreamingQueues() {
  if (streamingChunkQueues.size === 0) return;

  const startTime = Date.now();
  let totalProcessed = 0;

  console.log(`🔄 [QUEUE] Processing ${streamingChunkQueues.size} streaming queues...`);

  // Get available clients by framework
  const clientsByFramework = new Map();
  matrixState.clients.forEach((client, clientId) => {
    if (!client.connected || !client.gpuInfo ||
        client.isBusyWithCustomChunk || client.isBusyWithMatrixTask ||
        (!client.socket && !client.emit)) {
      return;
    }

    if (client.supportedFrameworks) {
      client.supportedFrameworks.forEach(framework => {
        if (!clientsByFramework.has(framework)) {
          clientsByFramework.set(framework, []);
        }
        clientsByFramework.get(framework).push({ clientId, client });
      });
    }
  });

  // Process each workload's queue with priority
  const workloadQueues = Array.from(streamingChunkQueues.entries())
    .sort(([aWorkloadId, aQueue], [bWorkloadId, bQueue]) => {
      // Prioritize workloads with older chunks
      const aOldest = Math.min(...aQueue.map(item => item.queuedAt));
      const bOldest = Math.min(...bQueue.map(item => item.queuedAt));
      return aOldest - bOldest;
    });

  for (const [workloadId, queue] of workloadQueues) {
    if (queue.length === 0) continue;

    let queueProcessed = 0;
    const maxDispatchPerQueue = Math.min(queue.length, 5); // Limit per cycle

    for (let i = 0; i < maxDispatchPerQueue && queue.length > 0; i++) {
      const queuedItem = queue.shift();
      const { descriptor } = queuedItem;

      // Find available client for this framework
      const frameworkClients = clientsByFramework.get(descriptor.framework) || [];
      const availableClient = frameworkClients.find(({ client }) =>
        !client.isBusyWithCustomChunk && !client.isBusyWithMatrixTask
      );

      if (!availableClient) {
        // No client available, put back at front of queue
        queue.unshift(queuedItem);
        break;
      }

      try {
        // Dispatch the queued chunk
        await dispatchChunkToClient(availableClient.client, descriptor, workloadId);

        // Mark client as busy and remove from available pool
        availableClient.client.isBusyWithCustomChunk = true;
        const frameworkList = clientsByFramework.get(descriptor.framework);
        const clientIndex = frameworkList.indexOf(availableClient);
        if (clientIndex > -1) {
          frameworkList.splice(clientIndex, 1);
        }

        queueProcessed++;
        totalProcessed++;

        console.log(`✅ [QUEUE] Dispatched queued chunk ${descriptor.chunkId} to ${availableClient.clientId}`);

      } catch (error) {
        // Re-queue the chunk with incremented attempt count
        queuedItem.attempts++;
        if (queuedItem.attempts < 3) {
          queue.push(queuedItem);
          console.warn(`⚠️  [QUEUE] Requeued chunk ${descriptor.chunkId} (attempt ${queuedItem.attempts}/3)`);
        } else {
          console.error(`❌ [QUEUE] Dropped chunk ${descriptor.chunkId} after 3 failed attempts`);
          updateStreamingWorkloadProgress(workloadId, descriptor.chunkId, 'failed');
        }
        break;
      }
    }

    if (queueProcessed > 0) {
      console.log(`📤 [QUEUE] Processed ${queueProcessed} chunks for workload ${workloadId} (${queue.length} remaining)`);
    }

    // Clean up empty queues
    if (queue.length === 0) {
      streamingChunkQueues.delete(workloadId);
      console.log(`🧹 [QUEUE] Cleared empty queue for workload ${workloadId}`);
    }
  }

  const processingTime = Date.now() - startTime;
  if (totalProcessed > 0) {
    console.log(`⚡ [QUEUE] Processed ${totalProcessed} total chunks in ${processingTime}ms`);
  }
}

async function dispatchStreamingChunkEnhanced(workloadId, chunkDescriptor) {
  console.log(`[STREAMING DISPATCH] === STARTING DISPATCH ===`);
  console.log(`[STREAMING DISPATCH] Workload: ${workloadId}`);
  console.log(`[STREAMING DISPATCH] Chunk: ${chunkDescriptor.chunkId}`);
  console.log(`[STREAMING DISPATCH] Framework: ${chunkDescriptor.framework}`);
  console.log(`[STREAMING DISPATCH] Has inputs: ${!!chunkDescriptor.inputs} (${chunkDescriptor.inputs?.length || 0})`);
  console.log(`[STREAMING DISPATCH] Has outputs: ${!!chunkDescriptor.outputs} (${chunkDescriptor.outputs?.length || 0})`);

  try {
    // Find available clients that support the framework
    const availableClients = Array.from(matrixState.clients.entries()).filter(([clientId, client]) =>
      client.connected &&
      client.gpuInfo &&
      !client.isBusyWithCustomChunk &&
      !client.isBusyWithMatrixTask &&
      client.supportedFrameworks &&
      client.supportedFrameworks.includes(chunkDescriptor.framework) &&
      (client.socket || client.emit)
    );

    if (availableClients.length === 0) {
      console.log(`⏳ No available clients for streaming chunk ${chunkDescriptor.chunkId} (framework: ${chunkDescriptor.framework}), queuing...`);
      return await queueStreamingChunk(workloadId, chunkDescriptor);
    }

    // Dispatch to first available client
    const [clientId, client] = availableClients[0];
    const result = await dispatchChunkToClient(client, chunkDescriptor, workloadId);

    console.log(`🚀 Immediately dispatched streaming chunk ${chunkDescriptor.chunkId} to ${clientId}`);
    return result;

  } catch (error) {
    console.error(`Failed to dispatch streaming chunk ${chunkDescriptor.chunkId}:`, error);
    // Fallback to queuing if immediate dispatch fails
    return await queueStreamingChunk(workloadId, chunkDescriptor);
  }
}

function updateStreamingWorkloadProgress(workloadId, chunkId, status) {
  const workload = customWorkloads.get(workloadId);
  if (!workload) return;

  if (!workload.streamingProgress) {
    workload.streamingProgress = {
      dispatched: new Map(),
      completed: new Map(),
      failed: new Map(),
      totalDispatched: 0,
      totalCompleted: 0,
      totalFailed: 0
    };
  }

  const progress = workload.streamingProgress;

  switch (status) {
    case 'dispatched':
      progress.dispatched.set(chunkId, Date.now());
      progress.totalDispatched++;
      console.log(`📈 [PROGRESS] Workload ${workloadId}: ${progress.totalDispatched} chunks dispatched`);
      break;

    case 'completed':
      progress.completed.set(chunkId, Date.now());
      progress.totalCompleted++;
      progress.dispatched.delete(chunkId);
      console.log(`📈 [PROGRESS] Workload ${workloadId}: ${progress.totalCompleted} chunks completed`);
      break;

    case 'failed':
      progress.failed.set(chunkId, Date.now());
      progress.totalFailed++;
      progress.dispatched.delete(chunkId);
      console.log(`📈 [PROGRESS] Workload ${workloadId}: ${progress.totalFailed} chunks failed`);
      break;
  }

  // Emit progress update to clients
  io.emit('streaming:progress', {
    workloadId,
    dispatched: progress.totalDispatched,
    completed: progress.totalCompleted,
    failed: progress.totalFailed,
    active: progress.dispatched.size,
    timestamp: Date.now()
  });
}


function cleanupStreamingQueue(workloadId) {
  const queue = streamingChunkQueues.get(workloadId);
  if (queue) {
    console.log(`🧹 Cleaning up streaming queue for ${workloadId}: ${queue.length} chunks discarded`);
    streamingChunkQueues.delete(workloadId);
  }
}

function cleanupStreamingWorkload(workloadId) {
  console.log(`🧹 [CLEANUP] Cleaning up streaming workload ${workloadId}`);

  // Remove from streaming queues
  const queue = streamingChunkQueues.get(workloadId);
  if (queue) {
    console.log(`🗑️  [CLEANUP] Removing ${queue.length} queued chunks for ${workloadId}`);
    streamingChunkQueues.delete(workloadId);
  }

  // Clean up progress tracking
  const workload = customWorkloads.get(workloadId);
  if (workload && workload.streamingProgress) {
    const progress = workload.streamingProgress;
    console.log(`📊 [CLEANUP] Final stats for ${workloadId}: ${progress.totalCompleted} completed, ${progress.totalFailed} failed`);
    delete workload.streamingProgress;
  }

  // Clean up from chunking manager
  if (typeof chunkingManager.cleanupWorkload === 'function') {
    try {
      chunkingManager.cleanupWorkload(workloadId);
    } catch (error) {
      console.warn(`⚠️  [CLEANUP] ChunkingManager cleanup warning for ${workloadId}:`, error.message);
    }
  }
}

function handleStreamingError(workloadId, chunkId, error, clientId) {
  console.error(`❌ [STREAMING ERROR] Workload ${workloadId}, chunk ${chunkId}, client ${clientId}: ${error.message}`);

  // Update progress tracking
  updateStreamingWorkloadProgress(workloadId, chunkId, 'failed');

  // Emit error to monitoring clients
  io.emit('streaming:error', {
    workloadId,
    chunkId,
    clientId,
    error: error.message,
    timestamp: Date.now()
  });

  // Check if we should abort the workload due to too many failures
  const workload = customWorkloads.get(workloadId);
  if (workload && workload.streamingProgress) {
    const progress = workload.streamingProgress;
    const failureRate = progress.totalFailed / (progress.totalCompleted + progress.totalFailed + progress.totalDispatched);

    if (failureRate > 0.5 && progress.totalFailed > 10) {
      console.error(`🛑 [STREAMING ERROR] High failure rate (${(failureRate * 100).toFixed(1)}%) for workload ${workloadId}, considering abort`);

      // Optionally auto-abort workloads with high failure rates
      if (process.env.STREAMING_AUTO_ABORT === 'true') {
        workload.status = 'error';
        workload.error = `High failure rate: ${progress.totalFailed} failed chunks (${(failureRate * 100).toFixed(1)}%)`;
        cleanupStreamingWorkload(workloadId);

        io.emit('workload:aborted', {
          id: workloadId,
          reason: workload.error,
          timestamp: Date.now()
        });
      }
    }
  }
}

setInterval(() => {
  // Log streaming statistics every 30 seconds
  if (streamingChunkQueues.size > 0) {
    const totalQueued = Array.from(streamingChunkQueues.values()).reduce((sum, queue) => sum + queue.length, 0);
    const activeStreaming = Array.from(customWorkloads.values()).filter(w => w.streamingMode && w.status !== 'complete').length;

    console.log(`📈 [STREAMING STATS] ${activeStreaming} active streaming workloads, ${totalQueued} chunks queued`);

    // Check for stalled queues
    streamingChunkQueues.forEach((queue, workloadId) => {
      const stalledChunks = queue.filter(item => Date.now() - item.queuedAt > 300000); // 5 minutes
      if (stalledChunks.length > 0) {
        console.warn(`⚠️  [STREAMING STATS] Workload ${workloadId} has ${stalledChunks.length} chunks stalled for >5 minutes`);
      }
    });
  }
}, 30000);

async function handleStreamingWorkloadComplete(workloadId, result) {
  console.log(`🎉 Streaming workload ${workloadId} completed!`);

  const workload = customWorkloads.get(workloadId);
  if (workload) {
    workload.status = 'complete';
    workload.completedAt = Date.now();

    // Handle result storage based on type
    if (result.type === 'memory') {
      workload.finalResultBase64 = Buffer.from(result.data.buffer).toString('base64');
    } else if (result.type === 'file') {
      workload.finalResultPath = result.path;
      workload.finalResultBase64 = null;
    }

    // NEW: Clean up streaming queue
    cleanupStreamingQueue(workloadId);

    saveCustomWorkloads();
    broadcastCustomWorkloadList();

    // Broadcast completion
    const completionData = {
      id: workloadId,
      label: workload.label,
      finalResultBase64: workload.finalResultBase64,
      finalResultUrl: workload.finalResultPath ? `/api/workloads/${workloadId}/download/final` : null,
      enhanced: true,
      streaming: true,
      matrixSize: result.matrixSize
    };

    io.emit('workload:complete', completionData);
    matrixState.clients.forEach(cl => {
      if (cl.clientType === 'native') {
        cl.emit('workload:complete', completionData);
      }
    });
  }
}


// --- Upload chunk result (binary) ---
app.post('/api/workloads/:wid/chunks/:cid/result/:idx', async (req, res) => {
  const { wid, cid, idx } = req.params;
  const wl = customWorkloads.get(wid);
  if (!wl) return res.status(404).json({ error: 'workload not found' });
  const dir = path.join(STORAGE_ROOT, wid, 'results', cid);
  await ensureDir(dir);
  const fp = path.join(dir, `out-${idx}.bin`);
  try {
    await pipeline(req, fs.createWriteStream(fp));
    res.json({ success: true, path: fp });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});


app.post('/api/workloads/debug/:id', (req, res) => {
  const wl = customWorkloads.get(req.params.id);
  const store = customWorkloadChunks.get(req.params.id);

  if (!wl || !store) {
    return res.status(404).json({ error: 'Workload not found' });
  }

  const debugInfo = {
    workload: {
      id: wl.id,
      status: wl.status,
      enhanced: wl.enhanced,
      hasInputRefs: !!wl.inputRefs,
      inputRefsCount: wl.inputRefs?.length || 0
    },
    store: {
      expectedChunks: store.expectedChunks,
      actualChunks: store.allChunkDefs.length,
      chunkSample: store.allChunkDefs[0] ? {
        chunkId: store.allChunkDefs[0].chunkId,
        hasInputs: !!store.allChunkDefs[0].inputs,
        inputsLength: store.allChunkDefs[0].inputs?.length || 0,
        hasOutputs: !!store.allChunkDefs[0].outputs,
        outputsLength: store.allChunkDefs[0].outputs?.length || 0,
        hasKernel: !!store.allChunkDefs[0].kernel,
        hasMetadata: !!store.allChunkDefs[0].metadata
      } : null
    }
  };

  res.json(debugInfo);
  debugMatrixTiledChunk(store,wl);
});

app.post('/api/workloads/:id/chunks/append', express.json({ limit: '2mb' }), (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);
  const store = customWorkloadChunks.get(wid);
  if (!wl || !store) return res.status(404).json({ error: 'Workload not found' });

  const { chunks = [] } = req.body;
  for (const cd of chunks) {
    cd.status = 'pending';
    store.allChunkDefs.push(cd);
  }
  // optional: kick the dispatcher to assign immediately if workers idle
  if (typeof store.dispatchNext === 'function') store.dispatchNext();

  res.json({ ok: true, queued: store.allChunkDefs.length });
});

app.get('/api/workloads/:id/chunks/status', (req, res) => {
  const wid = req.params.id;
  const wl = customWorkloads.get(wid);
  const store = customWorkloadChunks.get(wid);

  if (!wl || !store) {
    return res.status(404).json({ error: 'Workload not found' });
  }

  const chunkStatus = store.allChunkDefs.map(cd => ({
    chunkId: cd.chunkId,
    status: cd.status,
    submissions: cd.submissions ? cd.submissions.length : 0,
    verified: !!cd.verified_results,
    dispatchesMade: cd.dispatchesMade,
    activeAssignments: cd.activeAssignments ? cd.activeAssignments.size : 0
  }));

  res.json({
    workloadId: wid,
    totalChunks: store.expectedChunks,
    completedChunks: store.completedChunksData.size,
    chunks: chunkStatus,
    progress: store.expectedChunks > 0 ? (store.completedChunksData.size / store.expectedChunks * 100) : 0
  });
});


function verifyAndRecordChunkSubmission(parent, store, cd, submission, chunkOrderIndex, k) {
  if (cd.verified_results) {
    return { verified: true, winningChecksum: cd._winningChecksum || null, winnerSubmission: null };
  }

  if (!cd.submissions) cd.submissions = [];
  const dup = cd.submissions.some(s => s.clientId === submission.clientId && s.serverChecksum === submission.serverChecksum);
  if (!dup) cd.submissions.push(submission);

  console.log(`[SUBMISSION] Chunk ${cd.chunkId} received submission from ${submission.clientId} (${cd.submissions.length} total)`);

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
    console.log(`[SUBMISSION] Chunk ${cd.chunkId} needs more submissions (${cd.submissions.length}/${k})`);
    return { verified: false, winningChecksum: null, winnerSubmission: null };
  }

  const winningChecksum = tallyResult.winningChecksum;

  if (cd.verified_results) {
    return { verified: true, winningChecksum: cd._winningChecksum || winningChecksum, winnerSubmission: null };
  }

  const winner = cd.submissions.find(s => s.serverChecksum === winningChecksum);
  if (!winner) {
    console.error(`[SUBMISSION] No winning submission found for chunk ${cd.chunkId} with checksum ${winningChecksum.slice(0,8)}...`);
    return { verified: false, winningChecksum: null, winnerSubmission: null };
  }

  cd.verified_results = winner.results;
  cd.status = 'completed';
  cd._winningChecksum = winningChecksum;

  if (!store.completedChunksData) store.completedChunksData = new Map();
  store.completedChunksData.set(chunkOrderIndex, winner.buffers || winner.results.map(r => Buffer.from(r, 'base64')));

  console.log(`✅ Chunk ${cd.chunkId} verified with checksum ${winningChecksum.slice(0,8)}...`);

  // Log progress after verification
  logChunkProgress(parent.id);

  return { verified: true, winningChecksum, winnerSubmission: winner };
}

  function logChunkProgress(parentId) {
  const store = customWorkloadChunks.get(parentId);
  if (!store) return;

  const completed = store.completedChunksData.size;
  const total = store.expectedChunks;
  const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;

  console.log(`📊 Chunk progress for ${parentId}: ${completed}/${total} chunks (${percentage}%)`);

  // Emit progress update to clients
  io.emit('workload:progress', {
    id: parentId,
    completed,
    total,
    percentage
  });
}