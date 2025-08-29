// --- lightweight client logger (debug guard) ---
(function(){
  const q = new URLSearchParams(location.search);
  const lvl = (window.LOG_LEVEL || q.get('log') || 'info').toLowerCase();
  window.__DEBUG_ON__ = lvl === 'debug';
})();
// -----------------------------------------------
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

const shaderModuleCache = new Map();    // key -> GPUShaderModule
const computePipelineCache = new Map(); // key -> GPUComputePipeline


function fnv1a(str) { // tiny sync hash fallback
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h >>> 0) * 0x01000193;
  }
  return (h >>> 0).toString(16);
}

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
const socket = io({
  path: "/socket.io",
  //withCredentials: true,
  // transports: ["websocket"],   // optional; omit to allow polling->websocket
  query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {}
});
socket.on("connect_error", (err) => {
  console.error("[client] connect_error", err);
});
socket.io.on("reconnect_attempt", n => (__DEBUG_ON__ ? console.log : function(){})("reconnect_attempt", n));
socket.io.on("upgrade", t => (__DEBUG_ON__ ? console.log : function(){})("upgraded to", t && t.name));
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
        (__DEBUG_ON__ ? console.log : function(){})('WebGPU available!')
      }
    } catch (e) {
      console.warn('WebGPU not available:', e.message);
    }
  }

  // WebGL detection
  // WebGL detection (CORRECTED)
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      preserveDrawingBuffer: false
    });

  if (gl) {
    // Check for actual WebGL2 compute capabilities
    const extensions = {
      // Check for extensions we actually need for compute
      colorBufferFloat: gl.getExtension('EXT_color_buffer_float') !== null,
      textureFloatLinear: gl.getExtension('OES_texture_float_linear') !== null,

      // Check WebGL2 built-in features (these should always be true in WebGL2)
      hasTransformFeedback: typeof gl.createTransformFeedback === 'function',
      hasVertexArrayObjects: typeof gl.createVertexArray === 'function',
      hasFloatTextures: gl.getParameter(gl.MAX_TEXTURE_SIZE) >= 1024,
      hasFramebuffers: typeof gl.createFramebuffer === 'function'
    };

    // WebGL2 compute capability check
    const canCompute = extensions.hasTransformFeedback &&
                      extensions.hasFramebuffers &&
                      extensions.hasFloatTextures;

    if (canCompute) {
      frameworkState.webgl = {
        supported: true,
        context: gl,
        extensions,
        version: gl.getParameter(gl.VERSION),
        vendor: gl.getParameter(gl.VENDOR),
        renderer: gl.getParameter(gl.RENDERER)
      };
      capabilities.supportedFrameworks.push('webgl');
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Detection successful:`, {
        version: frameworkState.webgl.version,
        vendor: frameworkState.webgl.vendor,
        canCompute: true,
        extensions: extensions
      });
    } else {
      console.warn(`[WebGL] WebGL2 available but missing compute capabilities:`, extensions);
    }
  } else {
    console.warn(`[WebGL] Failed to create WebGL2 context`);
  }
  } catch (e) {
    console.warn('WebGL2 detection error:', e.message);
  }

  // JavaScript CPU framework (always available)
  capabilities.supportedFrameworks.push('javascript');
  frameworkState.javascript = {
    supported: true,
    context: 'cpu',
    version: 'ES2020+',
    vendor: navigator.userAgent,
    cores: navigator.hardwareConcurrency || 4
  };

  (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] CPU framework available with ${frameworkState.javascript.cores} logical cores`);

  return capabilities;
}

async function executeUnifiedChunk(chunk) {
  (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Starting execution for ${chunk.chunkId}`);
  (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Chunk structure:`, {
    hasInputs: !!chunk.inputs,
    inputsLength: chunk.inputs?.length || 0,
    hasOutputs: !!chunk.outputs,
    outputsLength: chunk.outputs?.length || 0,
    hasSchema: !!chunk.schema,
    hasMetadata: !!chunk.metadata,
    hasKernel: !!(chunk.kernel || chunk.wgsl)
  });

  if (!state.device) {
    throw new Error('No GPU device available');
  }

  // Validate chunk has required data
  if (!chunk.kernel && !chunk.wgsl) {
    throw new Error('No shader code provided');
  }

  if (!chunk.outputs || chunk.outputs.length === 0) {
    throw new Error('No output specifications provided');
  }

  const t0 = performance.now();
  const buffers = [];
  const entries = [];

  try {
    // Create shader module
    (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Creating shader module...`);
    const shaderCode = chunk.kernel || chunk.wgsl;
    const shader = state.device.createShaderModule({
      code: shaderCode
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

    let bindingIndex = 0;

    // === Explicit binding & uniforms support (keeps legacy fallback) ===
    const __usedBindings = new Set();
    function __pushEntryWithBinding(bindingMaybe, buffer) {
      let binding;
      if (typeof bindingMaybe === 'number' && Number.isFinite(bindingMaybe)) {
        binding = bindingMaybe >>> 0;
      } else {
        while (__usedBindings.has(bindingIndex)) bindingIndex++;
        binding = bindingIndex++;
      }
      __usedBindings.add(binding);
      entries.push({ binding, resource: { buffer } });
      buffers.push(buffer);
      return binding;
    }
    function __buildUniformBufferFromUniforms(uniforms) {
      if (uniforms && typeof uniforms.data === 'string') {
        const bytes = Uint8Array.from(atob(uniforms.data), c => c.charCodeAt(0));
        const size = Math.max(16, bytes.byteLength);
        const buf = state.device.createBuffer({
          size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true
        });
        new Uint8Array(buf.getMappedRange()).set(bytes);
        buf.unmap();
        return buf;
      }
      const keys = Array.isArray(uniforms?.order) && uniforms.order.length
        ? uniforms.order
        : Object.keys(uniforms || {}).filter(k => k !== 'binding' && k !== 'order');
      const BYTES_PER = 4;
      const size = Math.max(16, keys.length * BYTES_PER);
      const ab = new ArrayBuffer(size);
      const dv = new DataView(ab);
      let off = 0;
      for (const k of keys) {
        if (k === 'binding' || k === 'order') continue;
        const v = uniforms[k];
        if (Number.isInteger(v)) {
          dv.setUint32(off, v >>> 0, true);
        } else {
          dv.setFloat32(off, Number(v) || 0, true);
        }
        off += BYTES_PER;
      }
      const buf = state.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      state.device.queue.writeBuffer(buf, 0, ab);
      return buf;
    }

    // Prefer explicit uniforms provided by strategy (with optional explicit binding)
    if (chunk.uniforms) {
      try {
        const ubuf = __buildUniformBufferFromUniforms(chunk.uniforms);
        const ubinding = (typeof chunk.uniforms.binding === 'number') ? chunk.uniforms.binding : 0;
        __pushEntryWithBinding(ubinding, ubuf);
        (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Added explicit uniforms at binding ${ubinding}`);
      } catch (e) {
        console.warn('[UNIFIED] Failed to build explicit uniforms, falling back to metadata if available:', e);
      }
    }

    // Handle uniforms/metadata
    if (!chunk.uniforms && chunk.metadata && Object.keys(chunk.metadata).length > 0) {
      (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Setting up uniforms from metadata:`, chunk.metadata);

      // Create uniform array from metadata
      const uniformValues = [];

      // Add common uniform fields that strategies might use
      const commonFields = [
        'matrix_n', 'matrixSize', 'tile_start_row', 'tile_start_col',
        'tile_rows', 'tile_cols', 'tile_size', 'tileSize'
      ];

      for (const field of commonFields) {
        if (chunk.metadata[field] !== undefined) {
          uniformValues.push(chunk.metadata[field]);
        }
      }

      // Add any remaining numeric values
      for (const [key, value] of Object.entries(chunk.metadata)) {
        if (typeof value === 'number' && !commonFields.includes(key)) {
          uniformValues.push(value);
        }
      }

      if (uniformValues.length > 0) {
        const uniformArray = new Uint32Array(uniformValues);
        const uniformBuffer = state.device.createBuffer({
          size: Math.max(16, uniformArray.byteLength),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        state.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
        entries.push({
          binding: bindingIndex++,
          resource: { buffer: uniformBuffer }
        });
        buffers.push(uniformBuffer);

        (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Created uniform buffer with ${uniformValues.length} values at binding ${bindingIndex - 1}`);
      }
    }

    // Handle inputs
    if (chunk.inputs && chunk.inputs.length > 0) {
      (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Processing ${chunk.inputs.length} inputs`);

      for (let i = 0; i < chunk.inputs.length; i++) {
        const input = chunk.inputs[i];

        if (!input || !input.data) {
          console.warn(`[UNIFIED] Input ${i} has no data, skipping`);
          continue;
        }

        try {
          const inputBytes = Uint8Array.from(atob(input.data), c => c.charCodeAt(0));

          const inputBuffer = state.device.createBuffer({
            size: Math.max(16, inputBytes.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
          });

          new Uint8Array(inputBuffer.getMappedRange()).set(inputBytes);
          inputBuffer.unmap();

          __pushEntryWithBinding((input && typeof input.binding === 'number') ? input.binding : undefined, inputBuffer);

          (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Input ${input.name || i}: ${inputBytes.length} bytes at binding ${bindingIndex - 1}`);
        } catch (err) {
          console.error(`[UNIFIED] Failed to process input ${i}:`, err);
          throw new Error(`Failed to decode input ${input.name || i}: ${err.message}`);
        }
      }
    }

    // Handle outputs
    const outputBuffers = [];
    (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Creating ${chunk.outputs.length} output buffers`);

    for (let i = 0; i < chunk.outputs.length; i++) {
      const output = chunk.outputs[i];

      if (!output || !output.size || output.size <= 0) {
        throw new Error(`Invalid output ${i}: missing or invalid size`);
      }

      const outputBuffer = state.device.createBuffer({
        size: Math.max(16, output.size),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      __pushEntryWithBinding((output && typeof output.binding === 'number') ? output.binding : undefined, outputBuffer);
      outputBuffers.push(outputBuffer);

      (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Output ${output.name || i}: ${output.size} bytes at binding ${bindingIndex - 1}`);
    }

    // Create bind group and execute
    const bindGroup = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    const workgroupCount = chunk.workgroupCount || [1, 1, 1];
    (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Dispatching workgroups:`, workgroupCount);

    const encoder = state.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(...workgroupCount);
    computePass.end();

    // Read back all outputs
    const readBuffers = [];
    const results = [];

    for (let i = 0; i < outputBuffers.length; i++) {
      const outputBuffer = outputBuffers[i];
      const readBuffer = state.device.createBuffer({
        size: outputBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);
      readBuffers.push(readBuffer);
    }

    state.device.queue.submit([encoder.finish()]);

    // Map and read all outputs
    for (let i = 0; i < readBuffers.length; i++) {
      const readBuffer = readBuffers[i];
      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultBytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();

      const resultBase64 = btoa(String.fromCharCode(...resultBytes));
      results.push(resultBase64);

      (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Output ${i}: ${resultBytes.length} bytes -> ${resultBase64.length} chars base64`);
    }

    // Cleanup buffers
    buffers.forEach(buffer => buffer.destroy());
    readBuffers.forEach(buffer => buffer.destroy());

    const dt = performance.now() - t0;
    (__DEBUG_ON__ ? console.log : function(){})(`[UNIFIED] Chunk ${chunk.chunkId} completed in ${dt.toFixed(0)}ms`);

    return {
      results: results,
      result: results[0], // For backward compatibility
      processingTime: dt
    };

  } catch (err) {
    console.error(`[UNIFIED] Chunk execution error:`, err);
    // Cleanup any created buffers
    buffers.forEach(buffer => {
      try { buffer.destroy(); } catch {}
    });
    throw err;
  }
}

function createUniformArrayFromSchema(metadata, uniformDef) {
  if (uniformDef.fields) {
    return new Uint32Array(uniformDef.fields.map(field => metadata[field.name] || 0));
  }
  // Fallback for strategies that don't define fields
  return new Uint32Array(Object.values(metadata).filter(v => typeof v === 'number'));
}
// Enhanced: Multi-input, multi-framework execution

// Add performance.now() polyfill for older browsers
if (typeof performance === 'undefined') {
  window.performance = {
    now: function() { return Date.now(); }
  };
}

/*
async function executeEnhancedChunk(chunk) {
  (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Starting execution for ${chunk.chunkId} with strategy ${chunk.chunkingStrategy || 'unknown'}`);

  if (!state.device) {
    throw new Error('No GPU device available');
  }

  if (!chunk.outputSize || chunk.outputSize <= 0) {
    throw new Error(`Invalid chunk output size: ${chunk.outputSize}`);
  }

  const t0 = performance.now();

  try {
    (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Creating shader module...`);
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
      (__DEBUG_ON__ ? console.log : function(){})(`[MULTI-INPUT] Processing chunk with ${chunk.inputSchema.inputs.length} inputs`);

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
        (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Setting up uniforms:`, chunk.uniforms);

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
      (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Creating output buffer: ${chunk.outputSize} bytes`);
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

    (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Dispatching workgroups:`, chunk.workgroupCount);
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
    (__DEBUG_ON__ ? console.log : function(){})(`[ENHANCED] Chunk ${chunk.chunkId} completed in ${dt.toFixed(0)}ms, ${resultBytes.length} bytes output`);

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
*/
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
    case 'javascript':
      return await executeJavaScriptCompute(meta);
    default:
      throw new Error(`Unsupported framework: ${framework}`);
  }
}

async function executeJavaScriptCompute(chunk) {
  (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Starting CPU execution for ${chunk.chunkId || chunk.id}`);

  const outputs = chunk.outputs || [];
  if (outputs.length === 0) {
    throw new Error('No output specifications provided');
  }

  const t0 = performance.now();

  try {
    // Parse inputs
    const inputs = [];
    if (chunk.inputs && chunk.inputs.length > 0) {
      for (let i = 0; i < chunk.inputs.length; i++) {
        const input = chunk.inputs[i];
        if (!input || !input.data) continue;

        const inputBytes = Uint8Array.from(atob(input.data), c => c.charCodeAt(0));
        const inputFloats = new Float32Array(inputBytes.buffer);
        inputs.push(inputFloats);
      }
    }

    // Extract metadata
    const metadata = chunk.metadata || {};
    const blockSize = metadata.block_size || metadata.blockSize || 64;
    const matrixSize = metadata.matrix_size || metadata.matrixSize || 512;

    (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Processing block: ${blockSize}x${blockSize}, matrix: ${matrixSize}x${matrixSize}`);

    let results = [];

    // Execute dynamic kernel if provided by server
    if (chunk.kernel && chunk.entry) {
      (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Executing dynamic kernel: ${chunk.entry}`);

      try {
        // Create execution context with inputs and metadata available
        const kernelFunction = new Function('inputs', 'metadata', 'Float32Array', `
          ${chunk.kernel}

          // Call the entry point function
          if (typeof ${chunk.entry} === 'function') {
            return ${chunk.entry}(inputs[0], inputs[1], metadata.block_size || metadata.blockSize);
          } else {
            throw new Error('Entry point function ${chunk.entry} not found in kernel');
          }
        `);

        const result = kernelFunction(inputs, metadata, Float32Array);

        if (result instanceof Float32Array) {
          results.push(result);
        } else {
          throw new Error('Kernel must return Float32Array');
        }

        (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Dynamic kernel executed successfully`);

      } catch (kernelError) {
        console.error(`[JavaScript] Dynamic kernel execution failed:`, kernelError);
        (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Falling back to built-in implementation`);

        // Fallback to built-in implementation
        if (inputs.length >= 2) {
          const result = executeBlockMatrixMultiply(inputs[0], inputs[1], blockSize);
          results.push(result);
        } else {
          throw new Error('Dynamic kernel failed and insufficient inputs for fallback');
        }
      }
    } else {
      // No dynamic kernel - use built-in implementations
      (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Using built-in computation (no dynamic kernel provided)`);

      if (inputs.length >= 2) {
        // Matrix block multiplication
        const result = executeBlockMatrixMultiply(inputs[0], inputs[1], blockSize);
        results.push(result);
      } else if (inputs.length === 1) {
        // Single input processing
        const result = processSingleInput(inputs[0], metadata);
        results.push(result);
      } else {
        // Generate test output
        const size = outputs[0].size / 4; // Assume float32
        const result = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          result[i] = Math.random();
        }
        results.push(result);
      }
    }

    // Convert results to base64
    const base64Results = results.map(result => {
      const bytes = new Uint8Array(result.buffer);
      return btoa(String.fromCharCode(...bytes));
    });

    const dt = performance.now() - t0;
    (__DEBUG_ON__ ? console.log : function(){})(`[JavaScript] Computation completed in ${dt.toFixed(0)}ms`);

    return {
      results: base64Results,
      result: base64Results[0], // Backward compatibility
      processingTime: dt
    };

  } catch (error) {
    console.error(`[JavaScript] Execution error:`, error);
    throw error;
  }
}

async function executeWGSL(meta) {
  (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Starting WGSL execution for ${meta.id || meta.chunkId}`);

  if (!state.device) {
    throw new Error('No GPU device available');
  }

  if (!meta.outputSize || meta.outputSize <= 0) {
    throw new Error(`Invalid output size: ${meta.outputSize}`);
  }

  const t0 = performance.now();

  try {
    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Creating shader module...`);
    const shader = state.device.createShaderModule({ code: meta.wgsl || meta.kernel });

    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Getting compilation info...`);
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      const errors = ci.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
      console.error(`[HEADLESS] Shader compilation errors: ${errors}`);
      throw new Error(`Shader compilation failed: ${errors}`);
    }

    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Creating compute pipeline...`);
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

    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Creating output buffer with size: ${meta.outputSize} bytes`);
    const outBuf = state.device.createBuffer({
      size: Math.max(16, parseInt(meta.outputSize)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    entries.push({ binding: binding++, resource: { buffer: outBuf } });

    const bg = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Dispatching workgroups: ${meta.workgroupCount}`);
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
    (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] WGSL completed in ${dt.toFixed(0)}ms, ${resultBytes.length} bytes output`);

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


async function executeWebGPUCompute(chunk) {
  (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Starting compute execution for ${chunk.chunkId || chunk.id}`);

  if (!state.device) {
    throw new Error('No WebGPU device available');
  }

  const outputs = chunk.outputs || [];
  if (outputs.length === 0) {
    throw new Error('No output specifications provided');
  }

  const t0 = performance.now();
  const buffers = [];
  const entries = [];

  try {
    // Create shader module from strategy-provided WGSL
    (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Creating shader module...`);
    const shaderCode = chunk.kernel || chunk.wgsl;
    const shaderChecksum = chunk.shaderChecksum || (shaderCode ? fnv1a(shaderCode) : undefined);
    if (!shaderCode) {
      if (shaderChecksum && shaderModuleCache.has(shaderChecksum)) {
        // ok, we can proceed with cached module
      } else {
        throw new Error('No WGSL shader code provided by strategy');
      }
    }


    let shader;
    if (shaderChecksum && shaderModuleCache.has(shaderChecksum)) {
      shader = shaderModuleCache.get(shaderChecksum);
    } else {
      shader = state.device.createShaderModule({ code: shaderCode });
      const ci = await shader.getCompilationInfo();
      if (ci.messages.some(m => m.type === 'error')) {
        const errs = ci.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
        throw new Error(`Shader compilation failed: ${errs}`);
      }
      if (shaderChecksum) shaderModuleCache.set(shaderChecksum, shader);
    }

    // Check compilation
    const ci = await shader.getCompilationInfo();
    if (ci.messages.some(m => m.type === 'error')) {
      const errors = ci.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
      throw new Error(`WGSL compilation failed: ${errors}`);
    }

    // Create compute pipeline
    const entry = chunk.entry || 'main';
    const layoutKey = 'auto'; // or derive from your explicit bindings if you have them
    const pipeKey = `${shaderChecksum || fnv1a(shaderCode)}|${entry}|${layoutKey}`;

    let pipeline = computePipelineCache.get(pipeKey);
    if (!pipeline) {
      pipeline = state.device.createComputePipeline({
        layout: 'auto',
        compute: { module: shader, entryPoint: entry }
      });
      computePipelineCache.set(pipeKey, pipeline);
    }

    let bindingIndex = 0;

    // === Explicit binding & uniforms support (keeps legacy fallback) ===
    const __usedBindings = new Set();
    function __pushEntryWithBinding(bindingMaybe, buffer) {
      let binding;
      if (typeof bindingMaybe === 'number' && Number.isFinite(bindingMaybe)) {
        binding = bindingMaybe >>> 0;
      } else {
        while (__usedBindings.has(bindingIndex)) bindingIndex++;
        binding = bindingIndex++;
      }
      __usedBindings.add(binding);
      entries.push({ binding, resource: { buffer } });
      buffers.push(buffer);
      return binding;
    }
    function __buildUniformBufferFromUniforms(uniforms) {
      if (uniforms && typeof uniforms.data === 'string') {
        const bytes = Uint8Array.from(atob(uniforms.data), c => c.charCodeAt(0));
        const size = Math.max(16, bytes.byteLength);
        const buf = state.device.createBuffer({
          size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true
        });
        new Uint8Array(buf.getMappedRange()).set(bytes);
        buf.unmap();
        return buf;
      }
      const keys = Array.isArray(uniforms?.order) && uniforms.order.length
        ? uniforms.order
        : Object.keys(uniforms || {}).filter(k => k !== 'binding' && k !== 'order');
      const BYTES_PER = 4;
      const size = Math.max(16, keys.length * BYTES_PER);
      const ab = new ArrayBuffer(size);
      const dv = new DataView(ab);
      let off = 0;
      for (const k of keys) {
        if (k === 'binding' || k === 'order') continue;
        const v = uniforms[k];
        if (Number.isInteger(v)) {
          dv.setUint32(off, v >>> 0, true);
        } else {
          dv.setFloat32(off, Number(v) || 0, true);
        }
        off += BYTES_PER;
      }
      const buf = state.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      state.device.queue.writeBuffer(buf, 0, ab);
      return buf;
    }

    // Prefer explicit uniforms provided by strategy (with optional explicit binding)
    if (chunk.uniforms) {
      try {
        const ubuf = __buildUniformBufferFromUniforms(chunk.uniforms);
        const ubinding = (typeof chunk.uniforms.binding === 'number') ? chunk.uniforms.binding : 0;
        __pushEntryWithBinding(ubinding, ubuf);
        (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Added explicit uniforms at binding ${ubinding}`);
      } catch (e) {
        console.warn('[WebGPU] Failed to build explicit uniforms, falling back to metadata if available:', e);
      }
    }

    // Handle uniforms/metadata from strategy
    if (!chunk.uniforms && chunk.metadata && Object.keys(chunk.metadata).length > 0) {
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Setting up uniforms from metadata:`, chunk.metadata);

      // Create uniform array from metadata fields
      const uniformValues = [];

      // Add metadata fields in a consistent order for WebGPU
      const fieldOrder = [
        'block_size', 'matrix_size', 'matrix_n', 'matrixSize',
        'tile_start_row', 'tile_start_col', 'tile_rows', 'tile_cols', 'tile_size', 'tileSize'
      ];

      for (const field of fieldOrder) {
        if (chunk.metadata[field] !== undefined) {
          uniformValues.push(chunk.metadata[field]);
        }
      }

      // Add any remaining numeric values not already included
      for (const [key, value] of Object.entries(chunk.metadata)) {
        if (typeof value === 'number' && !fieldOrder.includes(key)) {
          uniformValues.push(value);
        }
      }

      if (uniformValues.length > 0) {
        const uniformArray = new Uint32Array(uniformValues);
        const uniformBuffer = state.device.createBuffer({
          size: Math.max(16, uniformArray.byteLength),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        state.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
        entries.push({
          binding: bindingIndex++,
          resource: { buffer: uniformBuffer }
        });
        buffers.push(uniformBuffer);

        (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Created uniform buffer with ${uniformValues.length} values at binding ${bindingIndex - 1}`);
      }
    }

    // Handle inputs from strategy
    if (chunk.inputs && chunk.inputs.length > 0) {
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Processing ${chunk.inputs.length} inputs`);

      for (let i = 0; i < chunk.inputs.length; i++) {
        const input = chunk.inputs[i];

        if (!input || !input.data) {
          console.warn(`[WebGPU] Input ${i} has no data, skipping`);
          continue;
        }

        try {
          const inputBytes = Uint8Array.from(atob(input.data), c => c.charCodeAt(0));

          const inputBuffer = state.device.createBuffer({
            size: Math.max(16, inputBytes.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
          });

          new Uint8Array(inputBuffer.getMappedRange()).set(inputBytes);
          inputBuffer.unmap();

          __pushEntryWithBinding((input && typeof input.binding === 'number') ? input.binding : undefined, inputBuffer);

          (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Input ${input.name || i}: ${inputBytes.length} bytes at binding ${bindingIndex - 1}`);
        } catch (err) {
          console.error(`[WebGPU] Failed to process input ${i}:`, err);
          throw new Error(`Failed to decode input ${input.name || i}: ${err.message}`);
        }
      }
    }

    // Handle outputs from strategy
    const outputBuffers = [];
    (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Creating ${chunk.outputs.length} output buffers`);

    for (let i = 0; i < chunk.outputs.length; i++) {
      const output = chunk.outputs[i];

      if (!output || !output.size || output.size <= 0) {
        throw new Error(`Invalid output ${i}: missing or invalid size`);
      }

      const outputBuffer = state.device.createBuffer({
        size: Math.max(16, output.size),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      __pushEntryWithBinding((output && typeof output.binding === 'number') ? output.binding : undefined, outputBuffer);
      outputBuffers.push(outputBuffer);

      (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Output ${output.name || i}: ${output.size} bytes at binding ${bindingIndex - 1}`);
    }

    // Create bind group and execute
    const bindGroup = state.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    // Get workgroup count from strategy
    const workgroupCount = chunk.workgroupCount || [1, 1, 1];
    (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Dispatching workgroups:`, workgroupCount);

    const encoder = state.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(...workgroupCount);
    computePass.end();

    // Read back all outputs
    const readBuffers = [];
    const results = [];

    for (let i = 0; i < outputBuffers.length; i++) {
      const outputBuffer = outputBuffers[i];
      const readBuffer = state.device.createBuffer({
        size: outputBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);
      readBuffers.push(readBuffer);
    }

    state.device.queue.submit([encoder.finish()]);

    // Map and read all outputs
    for (let i = 0; i < readBuffers.length; i++) {
      const readBuffer = readBuffers[i];
      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultBytes = new Uint8Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();

      const resultBase64 = btoa(String.fromCharCode(...resultBytes));
      results.push(resultBase64);

      (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Output ${i}: ${resultBytes.length} bytes -> ${resultBase64.length} chars base64`);
    }

    // Cleanup buffers
    buffers.forEach(buffer => buffer.destroy());
    readBuffers.forEach(buffer => buffer.destroy());

    const dt = performance.now() - t0;
    (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Chunk ${chunk.chunkId} completed in ${dt.toFixed(0)}ms`);

    return {
      results: results,
      result: results[0], // For backward compatibility
      processingTime: dt
    };

  } catch (err) {
    console.error(`[WebGPU] Chunk execution error:`, err);
    // Cleanup any created buffers
    buffers.forEach(buffer => {
      try { buffer.destroy(); } catch {}
    });
    throw err;
  }
}


// Basic WebGL compute support (simplified)
// Basic WebGL compute support (task-agnostic, multi-output)
async function executeWebGLCompute(chunk) {
  (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Starting compute execution for ${chunk.chunkId || chunk.id}`);

  if (!frameworkState.webgl.supported) {
    throw new Error('WebGL not available');
  }

  const outputs = chunk.outputs || [];
  if (outputs.length === 0) {
    throw new Error('No output sizes specified');
  }

  const t0 = performance.now();

  try {
    // Create a dedicated canvas for this computation
    const canvas = document.createElement('canvas');
    const computeGL = canvas.getContext('webgl2', {
      antialias: false,
      preserveDrawingBuffer: false,
      alpha: false
    });

    if (!computeGL) {
      throw new Error('Failed to create WebGL2 context for compute');
    }

    // Use strategy-provided shaders for WebGL
    if (chunk.webglVertexShader) {
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Using strategy-provided WebGL shaders`);
      return await executeWebGLTransformFeedback(computeGL, chunk, t0);
    } else {
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] No WebGL shaders provided, attempting WGSL->GLSL conversion`);
      // Fallback: try to convert WGSL to GLSL (limited support)
      return await executeWebGLWithWGSLFallback(computeGL, chunk, t0);
    }

  } catch (err) {
    console.error(`[WebGL] Compute execution error:`, err);
    throw err;
  }
}

// WebGL Transform Feedback execution with strategy-provided shaders (multi-output)
async function executeWebGLTransformFeedback(gl, chunk, t0) {
  (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Executing transform feedback compute...`);

  // Use strategy-provided shaders
  const vertexShaderSource   = chunk.webglVertexShader;
  const fragmentShaderSource = chunk.webglFragmentShader || getDefaultFragmentShader();
  if (!vertexShaderSource) {
    throw new Error('No WebGL vertex shader provided by strategy');
  }

  // Compile + link
  const vertexShader   = compileShader(gl, gl.VERTEX_SHADER,   vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);

  // Configure transform feedback varyings (one per output)
  const varyings = chunk.webglVaryings || ['v_result'];
  if ((chunk.outputs?.length || 0) !== varyings.length) {
    throw new Error(`Outputs (${chunk.outputs?.length || 0}) must match webglVaryings (${varyings.length})`);
  }
  gl.transformFeedbackVaryings(program, varyings, gl.SEPARATE_ATTRIBS);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program);
    throw new Error(`Program linking failed: ${error}`);
  }
  gl.useProgram(program);

  // Set uniforms from metadata
  const metadata = chunk.metadata || {};
  setWebGLUniforms(gl, program, metadata);

  // ---- Inputs (task-agnostic) ----
  // Route inputs by spec (default: texture-based)
  const resources = setupWebGLInputs(gl, program, chunk);

  // If textures were created, bind them to texture units and set sampler uniforms u_input_0..N
  if (resources.textures && resources.textures.length > 0) {
    resources.textures.forEach((tex, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const loc = gl.getUniformLocation(program, `u_input_${i}`);
      if (loc) gl.uniform1i(loc, i);
    });
  }

  // ---- Geometry / dispatch ----
  // Number of logical "points" to process (prefer strategy hint, then matrix fallback, then bytes/4)
  const outputs = chunk.outputs || [];
  const fallbackElems =
    (metadata.block_size && metadata.block_size * metadata.block_size) ||
    Math.max(1, Math.floor((outputs[0]?.size || 4) / 4));
  const numElements = chunk.webglNumElements ?? fallbackElems;

  // Index stream (as before)
  const indices = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) indices[i] = i;

  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  const a_index = gl.getAttribLocation(program, 'a_index');
  if (a_index !== -1) {
    gl.enableVertexAttribArray(a_index);
    gl.vertexAttribPointer(a_index, 1, gl.FLOAT, false, 0, 0);
  }

  (__DEBUG_ON__ ? console.log : function(){})(`[WebGL DEBUG] Execution parameters:`, {
    numElements,
    outputs: outputs.map(o => o.size),
    inputsLength: (chunk.inputs || []).length,
    varyings
  });

  // ---- Transform Feedback output buffers (one per output) ----
  const tf = gl.createTransformFeedback();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);

  const tfBuffers = [];
  const outArrays = [];

  for (let i = 0; i < outputs.length; i++) {
    const size = outputs[i].size;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, buf);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, size, gl.DYNAMIC_READ);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, buf);
    tfBuffers.push(buf);
    outArrays.push(new Uint8Array(size));
  }

  // ---- Draw (points) with rasterizer discard ----
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, numElements);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);

  /*
  // Add fence synchronization before reading buffers
   // Create a sync object to wait for GPU operations to complete
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);

  // Flush commands to ensure the sync object is processed
  gl.flush();

  // Wait for the GPU to complete all operations (with timeout)
  const timeoutNs = 1000000000; // 1 second in nanoseconds
  const result = gl.clientWaitSync(sync, gl.SYNC_FLUSH_COMMANDS_BIT, timeoutNs);

  if (result === gl.TIMEOUT_EXPIRED) {
    console.warn('[WebGL] Sync timeout - operations may not have completed');
  } else if (result === gl.WAIT_FAILED) {
    console.warn('[WebGL] Sync wait failed');
  }

  // Clean up the sync object
  gl.deleteSync(sync);
  */
  // ---- CRITICAL FIX: Properly unbind transform feedback before reading ----
  // First, unbind the transform feedback object
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

  // Then, explicitly unbind each buffer from its indexed transform feedback binding point
  for (let i = 0; i < tfBuffers.length; i++) {
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null);
  }

  // Ensure transform feedback buffer binding point is unbound
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);

  // ---- Read back results (now safe to bind to COPY_READ_BUFFER) ----
  const results = [];
  for (let i = 0; i < tfBuffers.length; i++) {
    gl.bindBuffer(gl.COPY_READ_BUFFER, tfBuffers[i]);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, outArrays[i]);
    const base64 = btoa(String.fromCharCode(...outArrays[i]));
    results.push(base64);
  }

  // Unbind the copy read buffer
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);

  // ---- Cleanup ----
  if (resources.textures) {
    resources.textures.forEach(tex => gl.deleteTexture(tex));
  }
  if (resources.attributeBuffers) {
    resources.attributeBuffers.forEach(({ buffer, location }) => {
      try { if (location !== -1) gl.disableVertexAttribArray(location); } catch {}
      gl.deleteBuffer(buffer);
    });
  }

  gl.deleteBuffer(indexBuffer);
  tfBuffers.forEach(b => gl.deleteBuffer(b));
  gl.deleteTransformFeedback(tf);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  gl.deleteProgram(program);

  const dt = performance.now() - t0;
  (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Transform feedback completed in ${dt.toFixed(0)}ms, results: ${results.length}`);

  return {
    results,
    result: results[0], // backward compatibility
    processingTime: dt
  };
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader);
    const shaderType = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
    console.error(`[WebGL] ${shaderType} shader source:`, source);
    gl.deleteShader(shader);
    throw new Error(`${shaderType} shader compilation failed: ${error}`);
  }

  return shader;
}

function setWebGLUniforms(gl, program, uniforms) {
  Object.entries(uniforms).forEach(([name, value]) => {
    const location = gl.getUniformLocation(program, `u_${name}`);
    if (location === null) return;

    if (typeof value === 'number') {
      if (Number.isInteger(value)) {
        gl.uniform1i(location, value);
      } else {
        gl.uniform1f(location, value);
      }
    }
  });
}
// Fallback for WGSL chunks without WebGL shaders (basic conversion)
async function executeWebGLWithWGSLFallback(gl, chunk, t0) {
  console.warn(`[WebGL] WGSL fallback not fully implemented - WebGL requires GLSL shaders`);
  throw new Error('WebGL execution requires WebGL-specific shaders from strategy. WGSL->GLSL conversion not implemented.');
}

/**
 * Task-agnostic WebGL input setup.
 * Supports:
 *  - spec.type === 'texture'  (default): uploads each input as an R32F texture, sets u_input_i samplers
 *  - spec.type === 'attribute': uploads Float32 buffers and enables attributes as specified
 *
 * Strategy can pass:
 *   chunk.webglInputSpec = {
 *     type: 'texture' | 'attribute',
 *     internalFormat: 'R32F',   // optional, defaults to R32F
 *     textureShape: [w, h],     // optional; if omitted, uses metadata.block_size for square
 *     attributes: [             // for 'attribute' type
 *       { name: 'a_input0', size: 1 }, // gl.vertexAttribPointer(size, FLOAT)
 *       ...
 *     ]
 *   }
 */
function setupWebGLInputs(gl, program, chunk) {
  const spec = chunk.webglInputSpec || { type: 'texture', internalFormat: 'R32F' };
  const inputs = chunk.inputs || [];
  const resources = { textures: null, attributeBuffers: null };

  if (spec.type === 'texture') {
    const textures = [];
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      if (!input?.data) continue;

      const bytes  = Uint8Array.from(atob(input.data), c => c.charCodeAt(0));
      // If this is float data, interpret as Float32Array; you may extend to other formats as needed
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));

      // Choose width/height: from spec.textureShape, metadata.block_size, or a square fallback
      let width, height;

      (__DEBUG_ON__ ? console.log : function(){})(`[WebGL INPUT DEBUG] Input ${i} processing:`, {
        hasTextureShape: Array.isArray(spec.textureShape),
        textureShapeLength: spec.textureShape?.length,
        hasBlockSize: !!chunk.metadata?.block_size,
        blockSize: chunk.metadata?.block_size,
        floatsLength: floats.length
      });

      if (Array.isArray(spec.textureShape) && spec.textureShape.length === 2) {
        [width, height] = spec.textureShape;
      } else if (chunk.metadata?.block_size) {
        width = height = chunk.metadata.block_size;
      } else {
        const n = Math.floor(Math.sqrt(floats.length));
        width = height = Math.max(1, n);
      }

      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, floats);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      textures.push(tex);
    }
    resources.textures = textures;
  } else if (spec.type === 'attribute') {
    // Map each declared attribute to a corresponding input[i]
    const attributeBuffers = [];
    const attributes = spec.attributes || [];
    for (let i = 0; i < attributes.length; i++) {
      const attr = attributes[i];
      const input = inputs[i];
      if (!attr?.name || !input?.data) continue;

      const loc = gl.getAttribLocation(program, attr.name);
      if (loc === -1) continue;

      const bytes  = Uint8Array.from(atob(input.data), c => c.charCodeAt(0));
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));

      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);

      const comps = Math.max(1, Math.min(4, attr.size || 1));
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, comps, gl.FLOAT, false, 0, 0);

      attributeBuffers.push({ buffer: buf, location: loc });
    }
    resources.attributeBuffers = attributeBuffers;
  }

  return resources;
}

function bindWorkloadListener() {
  if (workloadListenerBound) return;
  workloadListenerBound = true;

  socket.off('workload:new');

  function onWorkloadNew(meta) {
    const framework = meta.framework || 'webgpu';
    (__DEBUG_ON__ ? console.log : function(){})(`[FRAMEWORK] workload:new ${meta.id} (${framework})`, meta.label);

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
    (__DEBUG_ON__ ? console.log : function(){})(`[FRAMEWORK] Accepting ${framework} workload ${meta.id}`);

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
        (__DEBUG_ON__ ? console.log : function(){})(`[FRAMEWORK] ${framework} workload finished ${meta.id}`);
      }
    })();
  }

  socket.on('workload:new', onWorkloadNew);
}

async function initWebGPU() {
  (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Checking WebGPU support...`);
  (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] isSecureContext: ${window.isSecureContext}`);
  (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] navigator.gpu available: ${!!navigator.gpu}`);

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

    if (state.device && typeof state.device.lost?.then === 'function') {
    state.device.lost.then(() => {
      shaderModuleCache.clear();
      computePipelineCache.clear();
      console.warn('[WebGPU] device lost — shader/pipeline caches cleared');
    }).catch((e) => {
      console.warn('[WebGPU] device.lost promise rejected:', e);
    });
  }

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
    (__DEBUG_ON__ ? console.log : function(){})(`Headless worker ${WORKER_ID}`);
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

  const frameworkBadges = capabilities.supportedFrameworks.map(fw => {
    const badge = `<span class="framework-badge framework-${fw}">${fw.toUpperCase()}</span>`;
    return badge;
  }).join(' ');

  frameworkInfo.innerHTML = `
    <strong>Supported Frameworks:</strong><br>
    ${frameworkBadges}
  `;

  // Add JavaScript-specific styling
  const style = document.createElement('style');
  style.textContent += `
    .framework-badge.framework-javascript {
      background: #f7df1e;
      color: #000;
    }
  `;
  if (!document.head.querySelector('style[data-framework-styles]')) {
    style.setAttribute('data-framework-styles', 'true');
    document.head.appendChild(style);
  }

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
  if (IS_HEADLESS) { (__DEBUG_ON__ ? console.log : function(){})(`[ADMIN] ${msg}`); return; }
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

socket.on('connect', async () => {
  if (!hasJoinedOnce) {
    hasJoinedOnce = true;

    // Wait for framework detection to complete
    const capabilities = await detectFrameworkCapabilities();

    socket.emit('client:join', {
      gpuInfo: frameworkState.webgpu.adapterInfo || frameworkState.webgl.extensions || { vendor: 'CPU Fallback' },
      hasWebGPU: frameworkState.webgpu.supported,
      supportedFrameworks: capabilities.supportedFrameworks, //  Now included!
      clientType: capabilities.clientType
    });

    (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT] Auto-joined with frameworks: ${capabilities.supportedFrameworks.join(', ')}`);
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
  (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] Received chunk assignment:`, {
    chunkId: chunk.chunkId,
    framework: chunk.framework,
    hasInputs: !!chunk.inputs,
    inputsLength: chunk.inputs?.length || 0,
    hasOutputs: !!chunk.outputs,
    outputsLength: chunk.outputs?.length || 0,
    hasKernel: !!(chunk.kernel || chunk.wgsl),
    enhanced: chunk.enhanced
  });
  (__DEBUG_ON__ ? console.log : function(){})(`[CHUNK RECEIVED] Streaming mode: ${chunk.streaming}`);
  // NEW: Debug WebGL-specific properties
  (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] WebGL properties:`, {
    hasWebglVertexShader: !!chunk.webglVertexShader,
    hasWebglFragmentShader: !!chunk.webglFragmentShader,
    webglShaderType: chunk.webglShaderType,
    webglVaryings: chunk.webglVaryings,
    webglNumElements: chunk.webglNumElements
  });

  if (chunk.webglVertexShader) {
    (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] WebGL vertex shader length:`, chunk.webglVertexShader.length);
    (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] WebGL vertex shader preview:`, chunk.webglVertexShader.substring(0, 100) + '...');
  }

  if (chunk.webglFragmentShader) {
    (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] WebGL fragment shader length:`, chunk.webglFragmentShader.length);
  }

  (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] All chunk properties:`, Object.keys(chunk));

  const framework = chunk.framework || 'webgpu';

  if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
    console.warn(`[CHUNK] Client busy, rejecting chunk ${chunk.chunkId}`);
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: 'Client busy'
    });
    return;
  }

  if (!frameworkState[framework]?.supported) {
    console.warn(`[CHUNK] Framework ${framework} not supported for chunk ${chunk.chunkId}`);
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: `Framework ${framework} not supported`
    });
    return;
  }

  // Validate outputs
  const outputs = chunk.outputs || [];
  if (outputs.length === 0) {
    console.error(`[CHUNK] No outputs specified for chunk ${chunk.chunkId}`);
    const eventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(eventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: 'No outputs specified'
    });
    return;
  }
  // idk if this belongs here
  state.isComputingChunk = true;
  state.currentTask = chunk;
  if (!IS_HEADLESS) {
  elements.taskStatus.textContent = `${framework.toUpperCase()} ${chunk.chunkId} ${strategy}`;
  elements.taskStatus.className = 'status info';

  // Update framework badge
  const currentFramework = document.getElementById('current-framework');
  if (currentFramework) {
    currentFramework.textContent = framework.toUpperCase();
    currentFramework.className = `stat-value framework-${framework}`;
  }

  // Update strategy display
  const currentStrategy = document.getElementById('current-strategy');
  if (currentStrategy) {
    currentStrategy.textContent = strategy;
  }
}

  updateComputationStatusDisplay();

  const inputCount = chunk.inputs?.length || 0;
  const outputCount = outputs.length;
  const strategy = chunk.chunkingStrategy || 'unknown';

  logTaskActivity(`[${framework.toUpperCase()}] Processing chunk ${chunk.chunkId} (${strategy}, ${inputCount}→${outputCount})`);

  try {
  // Framework-specific execution routing
  let result;

  switch (framework) {
    case 'webgpu':
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGPU] Routing chunk ${chunk.chunkId} to WebGPU execution`);
      result = await executeWebGPUCompute(chunk);
      break;

    case 'webgl':
      (__DEBUG_ON__ ? console.log : function(){})(`[WebGL] Routing chunk ${chunk.chunkId} to WebGL execution`);
      result = await executeWebGLCompute(chunk);
      break;
    case 'javascript':
      (__DEBUG_ON__ ? console.log : function(){})(`[JS] Routing chunk ${chunk.chunkId} to JavaScript execution`);
      result = await executeJavaScriptCompute(chunk);
      break;
    case 'cuda':
      (__DEBUG_ON__ ? console.log : function(){})(`[CUDA] Routing chunk ${chunk.chunkId} to CUDA execution`);
      // TODO: Implement CUDA execution
      throw new Error('CUDA execution not yet implemented');

    case 'opencl':
      (__DEBUG_ON__ ? console.log : function(){})(`[OpenCL] Routing chunk ${chunk.chunkId} to OpenCL execution`);
      // TODO: Implement OpenCL execution
      throw new Error('OpenCL execution not yet implemented');

    case 'vulkan':
      (__DEBUG_ON__ ? console.log : function(){})(`[Vulkan] Routing chunk ${chunk.chunkId} to Vulkan execution`);
      // TODO: Implement Vulkan execution
      throw new Error('Vulkan execution not yet implemented');

    default:
      throw new Error(`Unknown framework: ${framework}`);
  }

  // Handle successful execution...

  // Record client processing time for timing analysis
  if (window.timingManager && chunk.chunkId) {
    try {
      window.timingManager.recordClientProcessingTime(chunk.chunkId, result.processingTime);
    } catch (e) {
      console.warn('[TIMING] Failed to record client processing time:', e);
    }
  }

  const eventName = chunk.enhanced ? 'workload:chunk_done_enhanced' : 'workload:chunk_done';
  const eventData = {
    parentId: chunk.parentId,
    chunkId: chunk.chunkId,
    chunkOrderIndex: chunk.chunkOrderIndex,
    results: result.results,  // Multiple outputs
    result: result.result,    // First output (backward compatibility)
    processingTime: result.processingTime
  };

  if (chunk.enhanced) {
    eventData.strategy = chunk.chunkingStrategy;
    eventData.metadata = chunk.metadata;
  }

  // Include checksums for all outputs
  if (result.results && result.results.length > 1) {
    eventData.reportedChecksums = await Promise.all(
      result.results.map(r => checksumBase64(r))
    );
  } else {
    eventData.reportedChecksum = await checksumBase64(result.result);
  }

  (__DEBUG_ON__ ? console.log : function(){})(`[${framework.toUpperCase()}] Sending results for ${chunk.chunkId}:`, {
    resultsCount: result.results.length,
    resultSizes: result.results.map(r => Math.round(r.length * 0.75))
  });

  socket.emit(eventName, eventData);

  logTaskActivity(`[${framework.toUpperCase()}] Chunk ${chunk.chunkId} complete: ${outputCount} outputs`, 'success');
  (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT DEBUG] Chunk ${chunk.chunkId} completion:`, {
  enhanced: chunk.enhanced,
  streaming: chunk.streaming,
  eventName: eventName,
  framework: framework
});
 } catch (err) {
    console.error(`[${framework.toUpperCase()}] Execution error for ${chunk.chunkId}:`, err);
    logTaskActivity(`[${framework.toUpperCase()}] Chunk ${chunk.chunkId} error: ${err.message}`, 'error');

    // Send ONLY the appropriate error event
    const errorEventName = chunk.enhanced ? 'workload:chunk_error_enhanced' : 'workload:chunk_error';
    socket.emit(errorEventName, {
      parentId: chunk.parentId,
      chunkId: chunk.chunkId,
      message: `${framework}: ${err.message}`
    });

    (__DEBUG_ON__ ? console.log : function(){})(`[CLIENT] Sent ${errorEventName} for chunk ${chunk.chunkId}`);

  } finally {
    state.isComputingChunk = false;
    state.currentTask = null;

    // Reset framework display
    if (!IS_HEADLESS) {
      const currentFramework = document.getElementById('current-framework');
      const currentStrategy = document.getElementById('current-strategy');

      if (currentFramework) {
        currentFramework.textContent = '-';
        currentFramework.className = 'stat-value';
      }

      if (currentStrategy) {
        currentStrategy.textContent = '-';
      }
    }

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
  (__DEBUG_ON__ ? console.log : function(){})(`[HEADLESS] Received WGSL workload: ${meta.id}, label: ${meta.label}`);
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
    logTaskActivity(` Enhanced workload ${data.label} complete! (${data.stats?.chunkingStrategy}/${data.stats?.assemblyStrategy})`, 'success');

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



/* === NVML listener bridge (no modules, no imports) ===========================
   This block connects to local NVML listener and sends messages:
   { type:"chunk_status", chunk_id, status (0|1|-1), gpu_index, pid? }
   It also auto-wraps a common chunk handler to emit start/end/error.
   To force a specific listener URL, add to page URL: ?nvml=wss://host/nvml&nvmlGpu=0
   Defaults:
     - if page is https => wss://<host>/nvml (proxy this path to ws://127.0.0.1:8765)
     - if page is http  => ws://127.0.0.1:8765
============================================================================= */
(function () {
  if (typeof window === 'undefined') return; // browser only

  // --- enableNvml gate ------------------------------------------------------
  try {
    const usp = new URLSearchParams(window.location.search);
    const val = usp.get('enableNvml');
    const enabled = !!val && ['1','true','yes','on'].includes(String(val).toLowerCase());
    if (!enabled) {
      console.log('[NVML] Disabled (add ?enableNvml=1 to enable)');
      return;
    }
    console.log('[NVML] Enabled via ?enableNvml');
  } catch (e) {
    console.warn('[NVML] Could not parse URL params; disabling.', e);
    return;
  }

  // --- tiny helper ----------------------------------------------------------
  function qs(name, fallback = null) {
    try {
      const v = new URLSearchParams(window.location.search).get(name);
      return v !== null ? v : fallback;
    } catch { return fallback; }
  }

  // --- minimal bridge class (no imports) ------------------------------------
  class NvmlBridgeBrowser {
    constructor(opts) {
      this.url = (opts && opts.url) || '';
      // Validate URL override if provided
      if (this.url) {
        try { new URL(this.url); }
        catch(e){ console.warn('[NVML] Invalid ?nvml URL override:', this.url, e); this.url = ''; }
      }
      this.gpuIndex = Number.isFinite(opts && opts.gpuIndex) ? Number(opts.gpuIndex) : 0;
      this.ws = null;
      this.pid = (typeof window !== 'undefined' && window.__PID__) || undefined;
      this._connected = false;
      this._queue = [];
      this._reconnTimer = null;
      this._helloSent = false;
    }

    _resolveDefaultUrl() {
      try {
        if (location.protocol === 'https:') return `wss://${location.host}/nvml`;
        return 'ws://127.0.0.1:8765';
      } catch {
        return 'ws://127.0.0.1:8765';
      }
    }

    _connect() {
      const target = this.url || this._resolveDefaultUrl();
      console.log('[NVML] Connecting to', target, `(GPU ${this.gpuIndex})`);
      try {
        this.ws = new WebSocket(target);
      } catch (e) {
        console.error('[NVML] WebSocket ctor failed', e);
        this._scheduleReconnect();
        return;
      }

      this.ws.onopen = () => {
        this._connected = true;
        this._helloSent = true;
        console.log('[NVML] Connected');
        try {
          this.ws.send(JSON.stringify({ type: 'hello', gpu: this.gpuIndex, pid: this.pid }));
        } catch (e) {
          console.warn('[NVML] hello send failed', e);
        }
        // flush queued
        if (this._queue.length) {
          for (const m of this._queue.splice(0)) this._send(m);
        }
      };

      this.ws.onmessage = (ev) => {
        let msg = ev.data;
        try { msg = JSON.parse(ev.data); } catch {}
        if (window.__DEBUG_ON__) console.log('[NVML] Message', msg);
      };

      this.ws.onerror = (e) => {
        console.error('[NVML] Error', e);
      };

      this.ws.onclose = () => {
        this._connected = false;
        console.log('[NVML] Closed');
        this._scheduleReconnect();
      };
    }

    _scheduleReconnect() {
      if (this._reconnTimer) return;
      this._reconnTimer = setTimeout(() => {
        this._reconnTimer = null;
        this._connect();
      }, 1500);
    }

    _send(obj) {
      if (!this.ws || this.ws.readyState !== 1) {
        this._queue.push(obj);
        return;
      }
      try {
        this.ws.send(JSON.stringify(obj));
      } catch (e) {
        console.warn('[NVML] send failed', e);
      }
    }

    start() {
      if (this.ws) return;
      this._connect();
    }

    notifyChunkStart(chunk_id) {
      this._send({ type: 'chunk_status', chunk_id, status: 1, gpu_index: this.gpuIndex, pid: this.pid });
    }
    notifyChunkEnd(chunk_id) {
      this._send({ type: 'chunk_status', chunk_id, status: 0, gpu_index: this.gpuIndex, pid: this.pid });
    }
    notifyChunkError(chunk_id) {
      this._send({ type: 'chunk_status', chunk_id, status: -1, gpu_index: this.gpuIndex, pid: this.pid });
    }
  }

  // --- bootstrap the bridge based on URL ------------------------------------
  const gpuIndex = parseInt(qs('nvmlGpu', '0'), 10) || 0;
  const overrideUrl = qs('nvml', '');
  const bridge = new NvmlBridgeBrowser({ gpuIndex, url: overrideUrl });

  // Start the connection now
  bridge.start();

  // Expose for manual calls / debugging
  window.__nvmlBridge = bridge;

  // --- optional instrumentation ---------------------------------------------
  // If your app has a central "runChunk" or similar, wrap it so the bridge
  // emits start/end/error automatically. This is a best-effort search:
  (function autoInstrument() {
    const candidates = ['runChunk', 'executeChunk', 'processChunk', 'runWorkload'];
    for (const name of candidates) {
      const host = (window.app && typeof window.app[name] === 'function')
                ? window.app
                : (typeof window[name] === 'function' ? window : null);
      if (host) {
        const fn = host[name];
        host[name] = async function wrappedChunk(chunk, ...args) {
          const chunkId = (chunk && (chunk.id || chunk.chunk_id)) || (Date.now() + ':' + Math.random().toString(16).slice(2));
          try {
            __nvmlBridge.notifyChunkStart(chunkId);
            const res = await fn.apply(this, [chunk, ...args]);
            __nvmlBridge.notifyChunkEnd(chunkId);
            return res;
          } catch (e) {
            __nvmlBridge.notifyChunkError(chunkId);
            throw e;
          }
        };
        break; // instrument only the first match
      }
    }
  })();
})();
/* === end NVML listener bridge ============================================== */
