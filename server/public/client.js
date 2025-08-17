// client.js - Complete Enhanced Version
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

// === NEW: checksum helpers (browser) ===
async function sha256HexFromU8(u8) {
  const view = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', view);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}
function base64ToU8(b64) {
  const s = atob(b64);
  const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}
async function checksumBase64(b64) {
  return sha256HexFromU8(base64ToU8(b64));
}
async function checksumMatrixRowsFloat32LE(rows) {
  const r = rows.length, c = r ? rows[0].length : 0;
  const buf = new ArrayBuffer(r * c * 4);
  const view = new DataView(buf);
  let o = 0;
  for (let i = 0; i < r; i++) {
    const row = rows[i];
    for (let j = 0; j < c; j++) { view.setFloat32(o, row[j], true); o += 4; }
  }
  return sha256HexFromU8(new Uint8Array(buf));
}

// Enhanced: Multi-framework state tracking
const frameworkState = {
  webgpu: { supported: false, device: null, adapterInfo: null },
  webgl: { supported: false, context: null, extensions: null }
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
  pushWgslWorkloadButton: document.getElementById('push-compute-workload'),
  activeWgslWorkloadsGrid: document.getElementById('active-wgsl-workloads-grid'),
  startQueuedWgslButton: document.getElementById('start-queued-compute-button'),
  adminKValueInput: document.getElementById('admin-k-value'),
  setKButton: document.getElementById('set-k-button'),
  currentKDisplay: document.getElementById('current-k-display')
};

const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';
const socket = io({ query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {} });

// Enhanced: Multi-framework capability detection
async function detectFrameworkCapabilities() {
  const capabilities = {
    supportedFrameworks: [],
    clientType: 'browser'
  };

  // WebGPU detection (existing logic enhanced)
  if (window.isSecureContext && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        let adapterInfo = { vendor: 'Unknown', architecture: 'Unknown' };

        if (typeof adapter.requestAdapterInfo === 'function') {
          try {
            adapterInfo = await adapter.requestAdapterInfo();
          } catch { /* ignore */ }
        }

        frameworkState.webgpu = { supported: true, device, adapterInfo };
        capabilities.supportedFrameworks.push('webgpu');
        console.log('WebGPU available!')
      }
    } catch (e) {
      console.warn('WebGPU not available:', e.message);
    }
  }

  // WebGL detection
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      preserveDrawingBuffer: false
    });

    if (gl) {
      const extensions = {
        transformFeedback: gl.getExtension('OES_texture_float'),
        vertexArrayObject: gl.getExtension('OES_vertex_array_object'),
      };

      frameworkState.webgl = { supported: true, context: gl, extensions };
      capabilities.supportedFrameworks.push('webgl');
    }
  } catch (e) {
    console.warn('WebGL2 not available:', e.message);
  }

  return capabilities;
}

// Enhanced: Multi-input, multi-framework execution
async function executeEnhancedChunk(chunk) {
  console.log(`[ENHANCED] Starting execution for ${chunk.chunkId} with strategy ${chunk.chunkingStrategy || 'unknown'}`);

  if (!state.device) {
    throw new Error('No GPU device available');
  }

  if (!chunk.outputSize || chunk.outputSize <= 0) {
    throw new Error(`Invalid chunk output size: ${chunk.outputSize}`);
  }

  const t0 = performance.now();

  try {
    console.log(`[ENHANCED] Creating shader module...`);
    const shader = state.device.createShaderModule({
      code: chunk.kernel || chunk.wgsl
    });

    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      const errors = ci.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
      throw new Error(`Shader compilation failed: ${errors}`);
    }

    const pipeline = state.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shader,
        entryPoint: chunk.entry || 'main'
      }
    });

    let binding = 0;
    const entries = [];

    // Enhanced: Handle multiple inputs based on schema
    if (chunk.inputSchema && chunk.chunkInputs) {
      console.log(`[MULTI-INPUT] Processing chunk with ${chunk.inputSchema.inputs.length} inputs`);

      // Handle uniforms first
      for (const uniformDef of chunk.inputSchema.uniforms || []) {
        const uniformBuffer = createUniformBuffer(chunk.uniforms, uniformDef);
        entries.push({ binding: binding++, resource: { buffer: uniformBuffer } });
      }

      // Handle inputs
      for (const inputDef of chunk.inputSchema.inputs) {
        const inputData = chunk.chunkInputs[inputDef.name];

        if (inputDef.type === 'storage_buffer') {
          const storageBuffer = createStorageBuffer(inputData, inputDef);
          entries.push({ binding: binding++, resource: { buffer: storageBuffer } });
        }
      }

      // Add output buffers
      for (const outputDef of chunk.inputSchema.outputs) {
        const outputBuffer = state.device.createBuffer({
          size: Math.max(16, chunk.outputSize),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        entries.push({ binding: binding++, resource: { buffer: outputBuffer } });
      }

    } else {
      // Handle legacy single-input format with uniforms

      // Add uniforms buffer if present
      if (chunk.uniforms && Object.keys(chunk.uniforms).length > 0) {
        console.log(`[ENHANCED] Setting up uniforms:`, chunk.uniforms);

        const uniformArray = createUniformArray(chunk.uniforms, chunk.chunkingStrategy);

        const uniformBuffer = state.device.createBuffer({
          size: Math.max(16, uniformArray.byteLength),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        state.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
        entries.push({ binding: binding++, resource: { buffer: uniformBuffer } });
      }

      // Handle input data
      const inputBytes = chunk.inputData
        ? Uint8Array.from(atob(chunk.inputData), c => c.charCodeAt(0))
        : new Uint8Array();

      if (inputBytes.length > 0) {
        const inputBuffer = state.device.createBuffer({
          size: Math.max(16, inputBytes.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true
        });
        new Uint8Array(inputBuffer.getMappedRange()).set(inputBytes);
        inputBuffer.unmap();
        entries.push({ binding: binding++, resource: { buffer: inputBuffer } });
      }

      // Create output buffer
      console.log(`[ENHANCED] Creating output buffer: ${chunk.outputSize} bytes`);
      const outputBuffer = state.device.createBuffer({
        size: Math.max(16, parseInt(chunk.outputSize)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      entries.push({ binding: binding++, resource: { buffer: outputBuffer } });
    }

    // Create bind group and execute
    const bindGroup = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    console.log(`[ENHANCED] Dispatching workgroups:`, chunk.workgroupCount);
    const encoder = state.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(...chunk.workgroupCount);
    computePass.end();

    // Read back results
    const outputBuffer = entries[entries.length - 1].resource.buffer;
    const readBuffer = state.device.createBuffer({
      size: outputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);
    state.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const resultBytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    const dt = performance.now() - t0;
    console.log(`[ENHANCED] Chunk ${chunk.chunkId} completed in ${dt.toFixed(0)}ms, ${resultBytes.length} bytes output`);

    const resultBase64 = btoa(String.fromCharCode(...resultBytes));

    return {
      result: resultBase64,
      processingTime: dt
    };

  } catch (err) {
    console.error(`[ENHANCED] Chunk execution error:`, err);
    throw err;
  }
}

// Helper functions for multi-input support
function createStorageBuffer(inputData, inputDef) {
  let bytes;

  if (typeof inputData === 'string') {
    bytes = Uint8Array.from(atob(inputData), c => c.charCodeAt(0));
  } else if (inputData instanceof Buffer) {
    bytes = new Uint8Array(inputData);
  } else {
    bytes = new Uint8Array(inputData);
  }

  const buffer = state.device.createBuffer({
    size: Math.max(16, bytes.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });

  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();

  return buffer;
}

function createUniformBuffer(inputData, inputDef) {
  let uniformBytes;

  if (typeof inputData === 'object' && !Buffer.isBuffer(inputData)) {
    uniformBytes = serializeUniforms(inputData, inputDef);
  } else {
    uniformBytes = new Uint8Array(inputData);
  }

  const buffer = state.device.createBuffer({
    size: Math.max(16, uniformBytes.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  state.device.queue.writeBuffer(buffer, 0, uniformBytes);
  return buffer;
}

function createUniformArray(uniforms, strategy) {
  switch (strategy) {
    case 'matrix_tiled':
      return new Uint32Array([
        uniforms.matrix_n || 0,
        uniforms.tile_start_row || 0,
        uniforms.tile_start_col || 0,
        uniforms.tile_rows || 0,
        uniforms.tile_cols || 0,
        uniforms.tile_size || 0
      ]);

    case 'matrix_rows':
      return new Uint32Array([
        uniforms.matrix_n || 0,
        uniforms.output_start_row || 0,
        uniforms.rows_to_compute || 0
      ]);

    default:
      const values = Object.values(uniforms).filter(v => typeof v === 'number');
      return new Uint32Array(values);
  }
}

function serializeUniforms(uniformObj, inputDef) {
  // Simple serialization based on field definitions
  if (inputDef && inputDef.fields) {
    const values = inputDef.fields.map(field => uniformObj[field] || 0);
    return new Uint32Array(values);
  }

  // Fallback: serialize all numeric values
  const values = Object.values(uniformObj).filter(v => typeof v === 'number');
  return new Uint32Array(values);
}

// Enhanced: Framework-agnostic kernel execution
async function executeFrameworkKernel(meta) {
  const framework = meta.framework || 'webgpu';

  switch (framework) {
    case 'webgpu':
      return await executeWGSL(meta);
    case 'webgl':
      return await executeWebGLCompute(meta);
    default:
      throw new Error(`Unsupported framework: ${framework}`);
  }
}

async function executeWGSL(meta) {
  console.log(`[HEADLESS] Starting WGSL execution for ${meta.id || meta.chunkId}`);

  if (!state.device) {
    throw new Error('No GPU device available');
  }

  if (!meta.outputSize || meta.outputSize <= 0) {
    throw new Error(`Invalid output size: ${meta.outputSize}`);
  }

  const t0 = performance.now();

  try {
    console.log(`[HEADLESS] Creating shader module...`);
    const shader = state.device.createShaderModule({ code: meta.wgsl || meta.kernel });

    console.log(`[HEADLESS] Getting compilation info...`);
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      const errors = ci.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
      console.error(`[HEADLESS] Shader compilation errors: ${errors}`);
      throw new Error(`Shader compilation failed: ${errors}`);
    }

    console.log(`[HEADLESS] Creating compute pipeline...`);
    const pipeline = state.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shader,
        entryPoint: meta.entry || 'main'
      }
    });

    const inputBytes = meta.input || meta.inputData
      ? Uint8Array.from(atob(meta.input || meta.inputData), c => c.charCodeAt(0))
      : new Uint8Array();

    let binding = 0;
    const entries = [];

    if (meta.chunkUniforms) {
      const uniformVals = [
        meta.chunkUniforms.chunkOffsetBytes || 0,
        meta.chunkUniforms.chunkInputSizeBytes || 0,
        meta.chunkUniforms.totalOriginalInputSizeBytes || 0
      ];
      if ('chunkOffsetElements' in meta.chunkUniforms) {
        uniformVals.push(
          meta.chunkUniforms.chunkOffsetElements || 0,
          meta.chunkUniforms.chunkInputSizeElements || 0,
          meta.chunkUniforms.totalOriginalInputSizeElements || 0
        );
      }

      const uniBuf = state.device.createBuffer({
        size: Math.max(16, new Uint32Array(uniformVals).byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      state.device.queue.writeBuffer(uniBuf, 0, new Uint32Array(uniformVals));
      entries.push({ binding: binding++, resource: { buffer: uniBuf } });
    }

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

    console.log(`[HEADLESS] Creating output buffer with size: ${meta.outputSize} bytes`);
    const outBuf = state.device.createBuffer({
      size: Math.max(16, parseInt(meta.outputSize)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    entries.push({ binding: binding++, resource: { buffer: outBuf } });

    const bg = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    console.log(`[HEADLESS] Dispatching workgroups: ${meta.workgroupCount}`);
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
    console.log(`[HEADLESS] WGSL completed in ${dt.toFixed(0)}ms, ${resultBytes.length} bytes output`);

    const resultBase64 = btoa(String.fromCharCode(...resultBytes));

    logTaskActivity(`WGSL ${meta.label || meta.chunkId} done in ${dt.toFixed(0)}ms`);

    return {
      result: resultBase64,
      processingTime: dt
    };

  } catch (err) {
    console.error(`[HEADLESS] WGSL execution error:`, err);
    logTaskActivity(`WGSL error: ${err.message}`, 'error');
    throw err;
  }
}

// Basic WebGL compute support (simplified)
async function executeWebGLCompute(meta) {
  console.log(`[WebGL] Starting compute execution for ${meta.id}`);

  if (!frameworkState.webgl.supported) {
    throw new Error('WebGL not available');
  }

  // Simplified WebGL execution - would need more sophisticated implementation
  throw new Error('WebGL compute execution not yet fully implemented');
}

function bindWorkloadListener() {
  if (workloadListenerBound) return;
  workloadListenerBound = true;

  socket.off('workload:new');

  function onWorkloadNew(meta) {
    const framework = meta.framework || 'webgpu';
    console.log(`[FRAMEWORK] workload:new ${meta.id} (${framework})`, meta.label);

    if (inFlightWorkloads.has(meta.id)) {
      console.warn(`[FRAMEWORK] duplicate workload event, declining ${meta.id}`);
      socket.emit('workload:busy', { id: meta.id, reason: 'duplicate-event' });
      return;
    }

    if (state.isComputingWgsl || state.matrixBusy || state.isComputingChunk) {
      console.warn(`[FRAMEWORK] Busy, rejecting workload ${meta.id}`);
      socket.emit('workload:busy', { id: meta.id, reason: 'local-busy' });
      return;
    }

    // Check if we support this framework
    if (!frameworkState[framework]?.supported) {
      console.warn(`[FRAMEWORK] Framework ${framework} not supported, rejecting ${meta.id}`);
      socket.emit('workload:busy', { id: meta.id, reason: 'framework-not-supported' });
      return;
    }

    state.isComputingWgsl = true;
    inFlightWorkloads.add(meta.id);
    console.log(`[FRAMEWORK] Accepting ${framework} workload ${meta.id}`);

    (async () => {
      try {
        const { result, processingTime } = await executeFrameworkKernel(meta);
        const reportedChecksum = await checksumBase64(result);
        socket.emit('workload:done', {
          id: meta.id,
          result,
          processingTime,
          reportedChecksum
        });
      } catch (err) {
        console.error(`[FRAMEWORK] ${framework} workload failed`, err);
        socket.emit('workload:error', {
          id: meta.id,
          message: `${framework}: ${err?.message || String(err)}`
        });
      } finally {
        state.isComputingWgsl = false;
        inFlightWorkloads.delete(meta.id);
        console.log(`[FRAMEWORK] ${framework} workload finished ${meta.id}`);
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
    return false;
  }

  if (!navigator.gpu) {
    elements.webgpuStatus.textContent = 'WebGPU not supported — CPU fallback.';
    elements.webgpuStatus.className = 'status warning';
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

    // Update framework state
    frameworkState.webgpu = {
      supported: true,
      device: state.device,
      adapterInfo: state.adapterInfo
    };

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

async function init() {
  if (IS_HEADLESS) {
    document.documentElement.style.display = 'none';
    console.log(`Headless worker ${WORKER_ID}`);
  }

  await initWebGPU();
  const capabilities = await detectFrameworkCapabilities();
  updateFrameworkDisplay(capabilities);
  updateComputationStatusDisplay();

  if (IS_HEADLESS) {
    joinComputation();
  } else {
    if (new URLSearchParams(location.search).has('admin')) {
      elements.adminPanel.style.display = 'block';
    }
  }
}

function updateFrameworkDisplay(capabilities) {
  if (IS_HEADLESS) return;

  const frameworkInfo = document.getElementById('framework-info') ||
    document.createElement('div');
  frameworkInfo.id = 'framework-info';
  frameworkInfo.className = 'status info';
  frameworkInfo.style.marginTop = '15px';

  frameworkInfo.innerHTML = `
    <strong>Supported Frameworks:</strong><br>
    ${capabilities.supportedFrameworks.map(fw =>
      `<span class="framework-badge">${fw.toUpperCase()}</span>`
    ).join(' ')}
  `;

  const gpuInfo = elements.gpuInfo;
  if (gpuInfo.nextSibling) {
    gpuInfo.parentNode.insertBefore(frameworkInfo, gpuInfo.nextSibling);
  } else {
    gpuInfo.parentNode.appendChild(frameworkInfo);
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

  detectFrameworkCapabilities().then(capabilities => {
    socket.emit('client:join', {
      gpuInfo: frameworkState.webgpu.adapterInfo || frameworkState.webgl.extensions || { vendor: 'CPU Fallback' },
      hasWebGPU: frameworkState.webgpu.supported,
      supportedFrameworks: capabilities.supportedFrameworks,
      clientType: capabilities.clientType
    });
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
    // === NEW: include reportedChecksum for matrix rows ===
    const reportedChecksum = await checksumMatrixRowsFloat32LE(out.result);
    socket.emit('task:complete', {
      assignmentId: task.assignmentId,
      taskId: task.id,
      result: out.result,
      processingTime: out.processingTime,
      reportedChecksum
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

// Enhanced: Enhanced chunk assignment handler
socket.on('workload:chunk_assign', async chunk => {
  const framework = chunk.framework || 'webgpu';

  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: 'Client busy'
    });
    return;
  }

  if (!frameworkState[framework]?.supported) {
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: `Framework ${framework} not supported`
    });
    return;
  }

  if (!chunk.outputSize || chunk.outputSize <= 0) {
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: `Invalid chunk output size: ${chunk.outputSize}`
    });
    return;
  }

  state.isComputingChunk = true;
  state.currentTask = chunk;
  updateComputationStatusDisplay();

  const strategy = chunk.chunkingStrategy || 'unknown';
  logTaskActivity(`Processing ${framework} chunk ${chunk.chunkId} (${strategy}, output: ${chunk.outputSize} bytes)`);

  try {
    let result;

    if (chunk.enhanced || chunk.inputSchema) {
      result = await executeEnhancedChunk(chunk);
    } else {
      result = await executeFrameworkKernel(chunk);
    }

    const eventName = chunk.enhanced ? 'workload:chunk_done_enhanced' : 'workload:chunk_done';
    const eventData = {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      chunkOrderIndex: chunk.chunkOrderIndex,
      result: result.result,
      processingTime: result.processingTime
    };

    if (chunk.enhanced) {
      eventData.strategy = chunk.chunkingStrategy;
      eventData.metadata = chunk.metadata;
    }

    // === NEW: include per-chunk reportedChecksum ===
    eventData.reportedChecksum = await checksumBase64(result.result);

    socket.emit(eventName, eventData);

  } catch (err) {
    logTaskActivity(`${framework} chunk error: ${err.message}`, 'error');
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: `${framework}: ${err.message}`
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

  elements.activeWgslWorkloadsGrid.innerHTML = '';

  if (!all.length) {
    elements.activeWgslWorkloadsGrid.innerHTML = '<p>No WGSL workloads.</p>';
    return;
  }

  all.forEach(wl => {
    const card = document.createElement('div');
    card.className = `wgsl-card status-${wl.status}`;
    card.id = `wgsl-card-${wl.id}`;

    let html = `
      <h4>${wl.label} (${wl.id.substring(0,6)})${wl.isChunkParent?' (Chunked)':''}</h4>
      <p>Status: ${wl.status}</p>
    `;

    if (wl.status === 'complete' && wl.startedAt && wl.completedAt) {
      const elapsedMs = wl.completedAt - wl.startedAt;
      html += `<p><small>Completed in ${(elapsedMs/1000).toFixed(2)} s (incl. dispatch & collection)</small></p>`;
    }

    html += `
      <div class="wgsl-card-actions">
        <button class="remove-wgsl-button danger" data-workload-id="${wl.id}">
          Remove (X)
        </button>
      </div>
    `;

    card.innerHTML = html;
    elements.activeWgslWorkloadsGrid.appendChild(card);

    card.querySelector('.remove-wgsl-button').addEventListener('click', () => {
      const id = wl.id;
      if (confirm(`Remove workload "${wl.label}" (ID ${id.substring(0,6)})?`)) {
        socket.emit('admin:removeCustomWorkload', { workloadId: id });
      }
    });
  });
});

// Duplicate 'workload:new' path (non-bindWorkloadListener) — keep in sync with above.
socket.on('workload:new', async meta => {
  console.log(`[HEADLESS] Received WGSL workload: ${meta.id}, label: ${meta.label}`);
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

    // === NEW: include reportedChecksum for whole-workload ===
    const reportedChecksum = await checksumBase64(resultBase64);
    socket.emit('workload:done', { id: meta.id, result: resultBase64, processingTime: dt, reportedChecksum });
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

socket.on('workload:complete', data => {
  if (data.enhanced) {
    logTaskActivity(`✅ Enhanced workload ${data.label} complete! (${data.stats?.chunkingStrategy}/${data.stats?.assemblyStrategy})`, 'success');

    if (data.stats) {
      logTaskActivity(`   Strategy: ${data.stats.chunkingStrategy} → ${data.stats.assemblyStrategy}`, 'info');
      logTaskActivity(`   Chunks: ${data.stats.totalChunks}`, 'info');
    }
  } else {
    logTaskActivity(`Workload ${data.label||data.id} complete!`, 'success');
  }
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

const style = document.createElement('style');
style.textContent = `
  .framework-badge {
    display: inline-block;
    padding: 2px 6px;
    margin: 2px;
    background: #007bff;
    color: white;
    border-radius: 3px;
    font-size: 0.8em;
    font-weight: bold;
  }
`;
document.head.appendChild(style);

init();
