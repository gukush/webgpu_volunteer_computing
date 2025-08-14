// client.js
const state = {
  webgpuSupported: false,
  device: null,
  adapterInfo: null,
  connected: false,
  clientId: null,
  isComputingMatrix: false,
  isComputingWgsl: false,
  isComputingChunk: false,
  currentTask: null,
  completedTasks: 0,
  statistics: { processingTime: 0 }
};

const inFlightWorkloads = new Set();
let workloadListenerBound = false;
let hasJoinedOnce = false;


const elements = {
  webgpuStatus: document.getElementById('webgpu-status'),
  gpuInfo: document.getElementById('gpu-info'),
  computationStatus: document.getElementById('computation-status'),
  clientStatus: document.getElementById('client-status'),
  taskStatus: document.getElementById('task-status'),
  joinComputation: document.getElementById('join-computation'),
  leaveComputation: document.getElementById('leave-computation'),
  startComputation: document.getElementById('start-computation'),
  toggleAdmin: document.getElementById('toggle-admin'),
  adminPanel: document.getElementById('admin-panel'),
  matrixSize: document.getElementById('matrix-size'),
  chunkSize: document.getElementById('chunk-size'),
  taskLog: document.getElementById('task-log'),
  adminLogMatrix: document.getElementById('admin-log-matrix'),
  adminLogWgsl: document.getElementById('admin-log-wgsl'),
  adminLogSystem: document.getElementById('admin-log-system'),
  clientGrid: document.getElementById('client-grid'),
  activeClients: document.getElementById('active-clients'),
  totalTasks: document.getElementById('total-tasks'),
  completedTasks: document.getElementById('completed-tasks'),
  elapsedTime: document.getElementById('elapsed-time'),
  myTasks: document.getElementById('my-tasks'),
  processingTime: document.getElementById('processing-time'),
  wgslLabel: document.getElementById('wgsl-label'),
  wgslEntryPoint: document.getElementById('wgsl-entry-point'),
  wgslSrc: document.getElementById('wgsl-src'),
  wgslGroupsX: document.getElementById('wgsl-groups-x'),
  wgslGroupsY: document.getElementById('wgsl-groups-y'),
  wgslGroupsZ: document.getElementById('wgsl-groups-z'),
  wgslBindLayout: document.getElementById('wgsl-bind-layout'),
  wgslOutputSize: document.getElementById('wgsl-output-size'),
  wgslInputData: document.getElementById('wgsl-input-data'),
  pushWgslWorkloadButton: document.getElementById('push-wgsl-workload'),
  activeWgslWorkloadsGrid: document.getElementById('active-wgsl-workloads-grid'),
  startQueuedWgslButton: document.getElementById('startQueuedWgslButton'),
  adminKValueInput: document.getElementById('admin-k-value'),
  setKButton: document.getElementById('set-k-button'),
  currentKDisplay: document.getElementById('current-k-display')
};

const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';
const socket = io({ query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {} });


function bindWorkloadListener() {
  if (workloadListenerBound) return;
  workloadListenerBound = true;

  // Remove any anonymous handlers that may have been added previously
  socket.off('workload:new');

  function onWorkloadNew(meta) {
    console.log('[HEADLESS] workload:new', meta.id, meta.label,
                'wgslBusy=', !!state.isComputingWgsl,
                'matrixBusy=', !!state.matrixBusy,
                'chunkBusy=', !!state.isComputingChunk);

    // drop duplicate deliveries of the same workload id
    if (inFlightWorkloads.has(meta.id)) {
      console.warn('[HEADLESS] duplicate workload event, declining', meta.id);
      socket.emit('workload:busy', { id: meta.id, reason: 'duplicate-event' });
      return;
    }

    // decline if any local lock is held
    if (state.isComputingWgsl || state.matrixBusy || state.isComputingChunk) {
      console.warn('[HEADLESS] Busy, rejecting workload', meta.id);
      socket.emit('workload:busy', { id: meta.id, reason: 'local-busy' });
      return;
    }

    // accept: set lock synchronously, before any await
    state.isComputingWgsl = true;
    inFlightWorkloads.add(meta.id);
    console.log('[HEADLESS] Accepting WGSL workload', meta.id);

    (async () => {
      try {
        const result = await executeWGSL(meta.code, meta.inputData);
        socket.emit('workload:done', { id: meta.id, result });
      } catch (err) {
        console.error('[HEADLESS] WGSL failed', err);
        socket.emit('workload:error', { id: meta.id, message: err?.message || String(err) });
      } finally {
        state.isComputingWgsl = false;
        inFlightWorkloads.delete(meta.id);
        console.log('[HEADLESS] WGSL finished', meta.id);
      }
    })();
  }

  socket.on('workload:new', onWorkloadNew);
}



async function initWebGPU() {
  console.log(`[HEADLESS] Checking WebGPU support...`);
  console.log(`[HEADLESS] isSecureContext: ${window.isSecureContext}`);
  console.log(`[HEADLESS] navigator.gpu available: ${!!navigator.gpu}`);
  if (!window.isSecureContext) {
    console.error(`[HEADLESS] Not a secure context`);
    elements.webgpuStatus.innerHTML = `WebGPU requires a secure context.`;
    elements.webgpuStatus.className = 'status error';
    elements.joinComputation.disabled = false;
    return false;
  }
  if (!navigator.gpu) {
    elements.webgpuStatus.textContent = 'WebGPU not supported – CPU fallback.';
    elements.webgpuStatus.className = 'status warning';
    elements.joinComputation.disabled = false;
    return false;
  }

  try {
    const rawAdapters = [];
    const high = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (high) rawAdapters.push(high);
    const low = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
    if (low && !rawAdapters.includes(low)) rawAdapters.push(low);
    const def = await navigator.gpu.requestAdapter();
    if (def && !rawAdapters.includes(def)) rawAdapters.push(def);

    if (!rawAdapters.length) throw new Error('No GPU adapters found.');

    const adaptersWithInfo = [];
    for (const adapter of rawAdapters) {
      let info = { vendor: 'Unknown', architecture: 'Unknown', device: 'Unknown', description: 'N/A' };
      if (typeof adapter.requestAdapterInfo === 'function') {
        try {
          const di = await adapter.requestAdapterInfo();
          info = { ...info, ...di };
        } catch { /* ignore */ }
      }
      adaptersWithInfo.push({ adapter, info });
    }

    const discrete = ['nvidia','amd','apple','intel','qualcomm','arm'];
    let chosen = adaptersWithInfo.find(a =>
      discrete.some(v =>
        a.info.vendor.toLowerCase().includes(v) &&
        !a.info.description.toLowerCase().includes('swiftshader')
      )
    ) || adaptersWithInfo[0];

    state.adapter = chosen.adapter;
    state.adapterInfo = chosen.info;
    state.device = await chosen.adapter.requestDevice();
    state.webgpuSupported = true;

    elements.webgpuStatus.textContent = `WebGPU: ${state.adapterInfo.vendor}`;
    elements.webgpuStatus.className = 'status success';
    elements.gpuInfo.innerHTML = `
      <strong>Vendor:</strong> ${state.adapterInfo.vendor}<br>
      <strong>Arch:</strong> ${state.adapterInfo.architecture}<br>
      <strong>Device:</strong> ${state.adapterInfo.device}
    `;
    elements.gpuInfo.className = 'status success';
    elements.joinComputation.disabled = false;
    return true;
  } catch (e) {
    elements.webgpuStatus.textContent = `WebGPU init error: ${e.message}`;
    elements.webgpuStatus.className = 'status warning';
    elements.gpuInfo.textContent = '';
    elements.gpuInfo.className = 'status error';
    state.webgpuSupported = false;
    elements.joinComputation.disabled = false;
    return false;
  }
}

async function multiplyMatricesGPU(A, B, size, startRow, endRow) {
  logTaskActivity(`GPU: computing rows ${startRow}–${endRow}`);
  const t0 = performance.now();
  const flatA = new Float32Array(size*size);
  const flatB = new Float32Array(size*size);
  for (let i=0; i<size; i++) {
    for (let j=0; j<size; j++) {
      flatA[i*size+j] = A[i][j];
      flatB[i*size+j] = B[i][j];
    }
  }

  const aBuf = state.device.createBuffer({
    size: flatA.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const bBuf = state.device.createBuffer({
    size: flatB.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const rowsToCalc = endRow - startRow;
  const resultSize = rowsToCalc * size * Float32Array.BYTES_PER_ELEMENT;
  const resBuf = state.device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const uniBuf = state.device.createBuffer({
    size: 3 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  state.device.queue.writeBuffer(aBuf, 0, flatA);
  state.device.queue.writeBuffer(bBuf, 0, flatB);
  state.device.queue.writeBuffer(uniBuf, 0, new Uint32Array([size, startRow, endRow]));

  const module = state.device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, write> R: array<f32>;
      struct U { size: u32, start: u32, end: u32 };
      @group(0) @binding(3) var<uniform> u: U;
      @compute @workgroup_size(8,8)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row = g.x + u.start;
        let col = g.y;
        if (row >= u.end || col >= u.size) { return; }
        var sum: f32 = 0.0;
        for (var i: u32 = 0; i < u.size; i++) {
          sum += A[row*u.size + i] * B[i*u.size + col];
        }
        R[(row - u.start)*u.size + col] = sum;
      }`
  });

  const pipeline = state.device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' }
  });

  const bind = state.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: resBuf } },
      { binding: 3, resource: { buffer: uniBuf } }
    ]
  });

  const enc = state.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(
    Math.ceil(rowsToCalc/8),
    Math.ceil(size/8)
  );
  pass.end();

  const readBuf = state.device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  enc.copyBufferToBuffer(resBuf, 0, readBuf, 0, resultSize);
  state.device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readBuf.getMappedRange().slice(0));
  const result = [];
  for (let i=0; i<rowsToCalc; i++) {
    result[i] = Array.from(arr.slice(i*size, (i+1)*size));
  }
  readBuf.unmap();
  aBuf.destroy(); bBuf.destroy(); resBuf.destroy(); readBuf.destroy(); uniBuf.destroy();

  const dt = performance.now() - t0;
  logTaskActivity(`GPU done in ${dt.toFixed(0)}ms`);
  return { result, processingTime: dt };
}

async function multiplyMatricesCPU(A, B, size, startRow, endRow) {
  logTaskActivity(`CPU: computing rows ${startRow}–${endRow}`);
  const t0 = performance.now();
  const rowsToCalc = endRow - startRow;
  const result = [];
  for (let i=0; i<rowsToCalc; i++) {
    const rowIdx = startRow + i;
    result[i] = [];
    for (let j=0; j<size; j++) {
      let sum = 0;
      for (let k=0; k<size; k++) {
        sum += A[rowIdx][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  const dt = performance.now() - t0;
  logTaskActivity(`CPU done in ${dt.toFixed(0)}ms`);
  return { result, processingTime: dt };
}

async function processMatrixTask(task) {
  state.isComputingMatrix = true;
  state.currentTask = task;
  elements.taskStatus.textContent = `Matrix ${task.id} rows ${task.startRow}-${task.endRow}`;
  elements.taskStatus.className = 'status info';
  updateComputationStatusDisplay();

  try {
    const out = state.webgpuSupported
      ? await multiplyMatricesGPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow)
      : await multiplyMatricesCPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);

    state.completedTasks++;
    state.statistics.processingTime += out.processingTime;
    elements.myTasks.textContent = state.completedTasks;
    elements.processingTime.textContent = `${out.processingTime.toFixed(0)}ms`;
    elements.taskStatus.textContent = `Matrix ${task.id} complete`;
    elements.taskStatus.className = 'status success';
    return out;
  } catch (e) {
    elements.taskStatus.textContent = `Error: ${e.message}`;
    elements.taskStatus.className = 'status error';
    socket.emit('task:error', {
      assignmentId: task.assignmentId,
      taskId: task.id,
      message: e.message,
      type: 'matrixMultiply'
    });
    throw e;
  } finally {
    state.isComputingMatrix = false;
    state.currentTask = null;
    updateComputationStatusDisplay();
  }
}

function joinComputation() {
  elements.joinComputation.disabled = true;
  elements.leaveComputation.disabled = false;
  socket.emit('client:join', {
    gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback' }
  });
}

function leaveComputation() {
  elements.joinComputation.disabled = false;
  elements.leaveComputation.disabled = true;
  state.isComputingMatrix = false;
  state.isComputingWgsl = false;
  state.isComputingChunk = false;
  state.currentTask = null;
  elements.taskStatus.textContent = 'No matrix task';
  elements.taskStatus.className = 'status info';
  socket.emit('client:leave');
}

function requestMatrixTask() {
  if (!state.connected || state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) return;
  socket.emit('task:request');
}

function startMatrixComputation() {
  const size = +elements.matrixSize.value;
  const chunk = +elements.chunkSize.value;
  socket.emit('admin:start', { matrixSize: size, chunkSize: chunk });
  logAdminActivity(`Start matrix ${size}×${size}, chunk ${chunk}`, 'matrix');
}

function logTaskActivity(msg, type='info') {
  if (IS_HEADLESS) return;
  const d = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = `status ${type}`;
  div.textContent = `[${d}] ${msg}`;
  elements.taskLog.appendChild(div);
  elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

function logAdminActivity(msg, panel='matrix', type='info') {
  if (IS_HEADLESS) { console.log(`[ADMIN] ${msg}`); return; }
  let container = elements.adminLogMatrix;
  if (panel === 'wgsl') container = elements.adminLogWgsl;
  if (panel === 'system') container = elements.adminLogSystem;
  const d = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = `status ${type}`;
  div.textContent = `[${d}] ${msg}`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function updateClientDisplay(clients) {
  if (IS_HEADLESS) return;
  elements.clientGrid.innerHTML = '';
  clients.forEach(c => {
    const el = document.createElement('div');
    el.className = `client-card ${!c.connected?'client-inactive':''}`;
    el.innerHTML = `
      <div>${c.id.substring(0,8)}...</div>
      <div>Tasks: ${c.completedTasks}</div>
      <div>${c.usingCpu?'CPU':'GPU'}</div>
      <div>${c.isBusyWithMatrixTask?'Matrix':c.isBusyWithCustomChunk?'Chunk':c.isBusyWithNonChunkedWGSL?'WGSL':'Idle'}</div>
    `;
    elements.clientGrid.appendChild(el);
  });
}

function updateStatsDisplay(stats) {
  if (IS_HEADLESS) return;
  elements.activeClients.textContent = stats.activeClients;
  elements.totalTasks.textContent = stats.totalTasks;
  elements.completedTasks.textContent = stats.completedTasks;
  if (stats.elapsedTime !== undefined) {
    elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`;
  }
}

function updateComputationStatusDisplay() {
  if (IS_HEADLESS) return;
  let txt = '';
  let cls = 'status info';
  if (state.isComputingMatrix) txt = 'Processing matrix…';
  else if (state.isComputingWgsl) txt = 'Processing WGSL…';
  else if (state.isComputingChunk) txt = 'Processing chunk…';
  else if (state.connected) txt = 'Idle, waiting';
  else { txt = 'Disconnected'; cls = 'status error'; }
  elements.computationStatus.textContent = txt;
  elements.computationStatus.className = cls;
}
/*
socket.on('connect', () => {
  state.connected = true;
  elements.clientStatus.textContent = 'Connected';
  elements.clientStatus.className = 'status success';
  logTaskActivity('Connected to server');
});
*/


socket.on('connect', () => {
  if (!hasJoinedOnce) {
    hasJoinedOnce = true;
    socket.emit('client:join', {
      gpuInfo: state.adapterInfo || { vendor: 'unknown' },
      hasWebGPU: !!state.device
    });
  }
  bindWorkloadListener();
});

socket.on('register', data => {
  state.clientId = data.clientId;
  elements.clientStatus.textContent = `You: ${data.clientId.substring(0,8)}…`;
});

socket.on('clients:update', data => {
  updateClientDisplay(data.clients);
});

socket.on('state:update', data => {
  updateStatsDisplay(data.stats);
  updateComputationStatusDisplay();
});

socket.on('task:assign', async task => {
  state.matrixBusy = true;
  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) return;
  try {
    const out = await processMatrixTask(task);
    socket.emit('task:complete', {
      assignmentId: task.assignmentId,
      taskId: task.id,
      ...out
    });
  } catch {}
});

socket.on('task:submitted', data => {
  logTaskActivity(`Submitted ${data.taskId}, awaiting verification`);
});

socket.on('task:verified', data => {
  logTaskActivity(`Your result for ${data.taskId} verified!`, 'success');
});

socket.on('task:wait', () => {
  if (!state.isComputingMatrix) {
    elements.taskStatus.textContent = 'No tasks, waiting…';
    elements.taskStatus.className = 'status warning';
    setTimeout(requestMatrixTask, 5000 + Math.random()*3000);
  }
});

socket.on('task:error', d => {
  logTaskActivity(`Server error on ${d.taskId}: ${d.message}`, 'error');
  if (state.currentTask && state.currentTask.assignmentId === d.assignmentId) {
    state.isComputingMatrix = false;
    state.currentTask = null;
    updateComputationStatusDisplay();
    requestMatrixTask();
  }
});

socket.on('computation:complete', d => {
  logTaskActivity(`Matrix finished in ${d.totalTime.toFixed(1)}s`, 'success');
  updateComputationStatusDisplay();
});

socket.on('workload:removed', ({ id }) => {
  const card = document.getElementById(`wgsl-card-${id}`);
  if (card) card.remove();
});

socket.on('workload:new', async meta => {
  console.log(`[HEADLESS] Received WGSL workload: ${meta.id}, label: ${meta.label}`);
  console.log(`[HEADLESS] State check:`);
  console.log(`  - isComputingMatrix: ${state.isComputingMatrix}`);
  console.log(`  - isComputingWgsl: ${state.isComputingWgsl}`);
  console.log(`  - isComputingChunk: ${state.isComputingChunk}`);
  console.log(`  - currentTask: ${state.currentTask ? JSON.stringify(state.currentTask) : 'null'}`);
  console.log(`  - device available: ${!!state.device}`);
  if (meta.isChunkParent) return;
  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
    socket.emit('workload:error', { id: meta.id, message: 'Busy' });
    return;
  }
  if (!state.device) {
    socket.emit('workload:error', { id: meta.id, message: 'No GPU' });
    return;
  }
  state.isComputingWgsl = true;
  state.currentTask = meta;
  updateComputationStatusDisplay();
  logTaskActivity(`Processing WGSL ${meta.label}`);

  try {
    const t0 = performance.now();
    const shader = state.device.createShaderModule({ code: meta.wgsl });
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      throw new Error(ci.messages.filter(m=>m.type==='error').map(m=>m.message).join('\n'));
    }
    const pipeline = state.device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: meta.entry } });

    const inputBytes = meta.input
      ? Uint8Array.from(atob(meta.input), c => c.charCodeAt(0))
      : new Uint8Array();

    let binding = 0;
    const entries = [];
    if (inputBytes.length) {
      const inBuf = state.device.createBuffer({
        size: Math.max(16, inputBytes.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint8Array(inBuf.getMappedRange()).set(inputBytes);
      inBuf.unmap();
      entries.push({ binding: binding++, resource: { buffer: inBuf } });
    }
    const outBuf = state.device.createBuffer({
      size: Math.max(16, meta.outputSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    entries.push({ binding: binding++, resource: { buffer: outBuf } });

    const bg = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
    entries
    });

    const enc = state.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(...meta.workgroupCount);
    pass.end();

    const readBuf = state.device.createBuffer({
      size: outBuf.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
    state.device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const resultBytes = new Uint8Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    const dt = performance.now() - t0;
    logTaskActivity(`WGSL done in ${dt.toFixed(0)}ms, ${resultBytes.length} bytes`);
    const resultBase64 = btoa(String.fromCharCode(...resultBytes));
    socket.emit('workload:done', { id: meta.id, result: resultBase64, processingTime: dt });
  } catch (err) {
    logTaskActivity(`WGSL error: ${err.message}`, 'error');
    socket.emit('workload:error', { id: meta.id, message: err.message });
  } finally {
    state.isComputingWgsl = false;
    state.currentTask = null;
    updateComputationStatusDisplay();
    requestMatrixTask();
  }
});

socket.on('workload:chunk_assign', async chunk => {
  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
    socket.emit('workload:chunk_error', { parentId: chunk.parentId, chunkId: chunk.chunkId, message: 'Busy' });
    return;
  }
  if (!state.device) {
    socket.emit('workload:chunk_error', { parentId: chunk.parentId, chunkId: chunk.chunkId, message: 'No GPU' });
    return;
  }
  state.isComputingChunk = true;
  state.currentTask = chunk;
  updateComputationStatusDisplay();
  logTaskActivity(`Processing chunk ${chunk.chunkId}`);

  try {
    const t0 = performance.now();
    const shader = state.device.createShaderModule({ code: chunk.wgsl });
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      throw new Error(ci.messages.filter(m=>m.type==='error').map(m=>m.message).join('\n'));
    }
    const pipeline = state.device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: chunk.entry } });

    const inBytes = Uint8Array.from(atob(chunk.inputData), c => c.charCodeAt(0));
    const inBuf = state.device.createBuffer({
      size: Math.max(16, inBytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint8Array(inBuf.getMappedRange()).set(inBytes);
    inBuf.unmap();

    const uniformVals = [
      chunk.chunkUniforms.chunkOffsetBytes,
      chunk.chunkUniforms.chunkInputSizeBytes,
      chunk.chunkUniforms.totalOriginalInputSizeBytes
    ];
    if ('chunkOffsetElements' in chunk.chunkUniforms) {
      uniformVals.push(
        chunk.chunkUniforms.chunkOffsetElements,
        chunk.chunkUniforms.chunkInputSizeElements,
        chunk.chunkUniforms.totalOriginalInputSizeElements
      );
    }
    const uniBuf = state.device.createBuffer({
      size: Math.max(16, new Uint32Array(uniformVals).byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    state.device.queue.writeBuffer(uniBuf, 0, new Uint32Array(uniformVals));

    let estOut = chunk.chunkUniforms.chunkInputSizeBytes || 16;
    const outBuf = state.device.createBuffer({
      size: Math.max(16, estOut),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bg = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } }
      ]
    });

    const enc = state.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);

    const wgX = Math.ceil((chunk.chunkUniforms.chunkInputSizeBytes / 4) / 64);
    pass.dispatchWorkgroups(Math.max(1, wgX));
    pass.end();

    const readBuf = state.device.createBuffer({
      size: outBuf.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
    state.device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const resultBytes = new Uint8Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    const dt = performance.now() - t0;
    logTaskActivity(`Chunk ${chunk.chunkId} done in ${dt.toFixed(0)}ms`);
    const base64 = btoa(String.fromCharCode(...resultBytes));
    socket.emit('workload:chunk_done', {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      chunkOrderIndex: chunk.chunkOrderIndex,
      result: base64,
      processingTime: dt
    });
  } catch (err) {
    logTaskActivity(`Chunk error: ${err.message}`, 'error');
    socket.emit('workload:chunk_error', {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: err.message
    });
  } finally {
    state.isComputingChunk = false;
    state.currentTask = null;
    updateComputationStatusDisplay();
    requestMatrixTask();
  }
});

socket.on('workloads:list_update', all => {
  if (IS_HEADLESS) return;

  // clear out the grid
  elements.activeWgslWorkloadsGrid.innerHTML = '';

  if (!all.length) {
    elements.activeWgslWorkloadsGrid.innerHTML = '<p>No WGSL workloads.</p>';
    return;
  }

  all.forEach(wl => {
    const card = document.createElement('div');
    card.className = `wgsl-card status-${wl.status}`;
    card.id = `wgsl-card-${wl.id}`;

    // Build the inner HTML, including status, optional elapsed time, and Remove button
    let html = `
      <h4>${wl.label} (${wl.id.substring(0,6)})${wl.isChunkParent?' (Chunked)':''}</h4>
      <p>Status: ${wl.status}</p>
    `;

    // If complete, show total wall-clock
    if (wl.status === 'complete' && wl.startedAt && wl.completedAt) {
      const elapsedMs = wl.completedAt - wl.startedAt;
      html += `<p><small>Completed in ${(elapsedMs/1000).toFixed(2)} s (incl. dispatch & collection)</small></p>`;
    }

    // Add the Remove button
    html += `
      <div class="wgsl-card-actions">
        <button class="remove-wgsl-button danger" data-workload-id="${wl.id}">
          Remove (X)
        </button>
      </div>
    `;

    card.innerHTML = html;
    elements.activeWgslWorkloadsGrid.appendChild(card);

    // Wire up the click handler
    card.querySelector('.remove-wgsl-button').addEventListener('click', () => {
      const id = wl.id;
      if (confirm(`Remove workload "${wl.label}" (ID ${id.substring(0,6)})?`)) {
        socket.emit('admin:removeCustomWorkload', { workloadId: id });
      }
    });
  });
});


socket.on('workload:new', async meta => {
  console.log(`[HEADLESS] Received WGSL workload: ${meta.id}`);

  if (meta.isChunkParent) return;
  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
    console.log(`[HEADLESS] Busy, rejecting workload`);
    socket.emit('workload:error', { id: meta.id, message: 'Busy' });
    return;
  }
  if (!state.device) {
    console.log(`[HEADLESS] No GPU device available`);
    socket.emit('workload:error', { id: meta.id, message: 'No GPU' });
    return;
  }

  state.isComputingWgsl = true;
  state.currentTask = meta;
  updateComputationStatusDisplay();
  console.log(`[HEADLESS] Processing WGSL ${meta.label}`);

  try {
    const t0 = performance.now();
    console.log(`[HEADLESS] Creating shader module...`);
    const shader = state.device.createShaderModule({ code: meta.wgsl });

    console.log(`[HEADLESS] Getting compilation info...`);
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      const errors = ci.messages.filter(m=>m.type==='error').map(m=>m.message).join('\n');
      console.error(`[HEADLESS] Shader compilation errors: ${errors}`);
      throw new Error(errors);
    }

    console.log(`[HEADLESS] Creating compute pipeline...`);
    // ... rest of the code ...

    console.log(`[HEADLESS] WGSL complete, sending result`);
    socket.emit('workload:done', { id: meta.id, result: resultBase64, processingTime: dt });
  } catch (err) {
    console.error(`[HEADLESS] WGSL error: ${err.message}`, err);
    socket.emit('workload:error', { id: meta.id, message: err.message });
  } finally {
    state.isComputingWgsl = false;
    state.currentTask = null;
    updateComputationStatusDisplay();
  }
});

socket.on('workload:complete', data => {
  logTaskActivity(`Workload ${data.label||data.id} complete!`, 'success');
});

socket.on('workload:parent_started', data => {
  logTaskActivity(`Parent ${data.label} started (${data.status})`);
});

socket.on('admin:feedback', d => {
  logAdminActivity(d.message, d.panelType||'wgsl', d.success?'success':'error');
});

socket.on('admin:k_update', newK => {
  elements.adminKValueInput.value = newK;
  elements.currentKDisplay.textContent = newK;
  logAdminActivity(`K = ${newK}`, 'system', 'info');
});

elements.joinComputation.addEventListener('click', joinComputation);
elements.leaveComputation.addEventListener('click', leaveComputation);
elements.startComputation.addEventListener('click', startMatrixComputation);
elements.toggleAdmin.addEventListener('click', () => {
  elements.adminPanel.style.display = elements.adminPanel.style.display==='block'?'none':'block';
});
elements.setKButton.addEventListener('click', () => {
  const k = parseInt(elements.adminKValueInput.value);
  if (!isNaN(k) && k>=1) socket.emit('admin:set_k_parameter', k);
  else logAdminActivity('Invalid K', 'system', 'error');
});
elements.pushWgslWorkloadButton.addEventListener('click', async () => {
  const chunkable = document.getElementById('wgsl-chunkable')?.checked;
  const payload = {
    label: elements.wgslLabel.value || 'Untitled',
    wgsl: elements.wgslSrc.value,
    entry: elements.wgslEntryPoint.value || 'main',
    workgroupCount: [
      +elements.wgslGroupsX.value || 1,
      +elements.wgslGroupsY.value || 1,
      +elements.wgslGroupsZ.value || 1
    ],
    bindLayout: elements.wgslBindLayout.value,
    outputSize: +elements.wgslOutputSize.value,
    input: elements.wgslInputData.value || undefined,
    chunkable
  };

  if (chunkable) {
    payload.inputChunkProcessingType = document.getElementById('wgsl-chunk-processing-type').value;
    payload.inputChunkSize = +document.getElementById('wgsl-input-chunk-size').value;
    payload.inputElementSizeBytes = +document.getElementById('wgsl-input-element-size-bytes').value;
    payload.outputAggregationMethod = document.getElementById('wgsl-output-aggregation-method').value;
  }

  logAdminActivity(`Pushing WGSL "${payload.label}"…`, 'wgsl');
  const res = await fetch('/api/workloads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const j = await res.json();
  if (res.ok && j.ok) {
    logAdminActivity(`ID ${j.id.substring(0,6)}: ${j.message}`, 'wgsl', 'success');
  } else {
    logAdminActivity(`Error: ${j.error || res.statusText}`, 'wgsl', 'error');
  }
});

elements.startQueuedWgslButton.addEventListener('click', () => {
  logAdminActivity('Starting queued WGSL…', 'wgsl');
  socket.emit('admin:startQueuedCustomWorkloads');
});

async function init() {
  if (IS_HEADLESS) {
    document.documentElement.style.display = 'none';
    console.log(`Headless worker ${WORKER_ID}`);
  }
  await initWebGPU();
  updateComputationStatusDisplay();
  if (IS_HEADLESS) {
    joinComputation();
  } else {
    if (new URLSearchParams(location.search).has('admin')) {
      elements.adminPanel.style.display = 'block';
    }
  }
}

init();
