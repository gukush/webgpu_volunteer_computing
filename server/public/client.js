// client.js

// State management
const state = {
    webgpuSupported: false,
    device: null,
    adapter: null,
    adapterInfo: null,
    connected: false,
    clientId: null,
    isComputing: false, // For matrix tasks primarily
    currentTask: null, // For matrix tasks primarily
    completedTasks: 0, // For matrix tasks primarily
    statistics: {
        processingTime: 0 // For matrix tasks primarily
    }
};

const elements = {
    webgpuStatus: document.getElementById('webgpu-status'),
    gpuInfo: document.getElementById('gpu-info'),
    computationStatus: document.getElementById('computation-status'),
    clientStatus: document.getElementById('client-status'),
    taskStatus: document.getElementById('task-status'), // For matrix tasks
    joinComputation: document.getElementById('join-computation'),
    leaveComputation: document.getElementById('leave-computation'),
    startComputation: document.getElementById('start-computation'), // Matrix computation
    toggleAdmin: document.getElementById('toggle-admin'),
    adminPanel: document.getElementById('admin-panel'),
    matrixSize: document.getElementById('matrix-size'),
    chunkSize: document.getElementById('chunk-size'),
    taskLog: document.getElementById('task-log'), // General client log
    adminLogMatrix: document.getElementById('admin-log-matrix'),
    adminLogWgsl: document.getElementById('admin-log-wgsl'),
    clientGrid: document.getElementById('client-grid'),
    activeClients: document.getElementById('active-clients'),
    totalTasks: document.getElementById('total-tasks'),
    completedTasks: document.getElementById('completed-tasks'),
    elapsedTime: document.getElementById('elapsed-time'),
    myTasks: document.getElementById('my-tasks'), // For matrix tasks
    processingTime: document.getElementById('processing-time'), // For matrix tasks

    // WGSL Admin UI
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
    activeWgslWorkloadsGrid: document.getElementById('active-wgsl-workloads-grid')
};

// Headless mode detection
const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';


// Connect to Socket.io server
const socket = io({
    query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {}
});

async function initWebGPU() {
    try {
        if (!window.isSecureContext) {
            elements.webgpuStatus.innerHTML = `<div>WebGPU requires a secure context (HTTPS or localhost).</div>`;
            elements.webgpuStatus.className = 'status error';
            elements.joinComputation.disabled = false; return false; // Allow CPU fallback join
        }
        if (!navigator.gpu) {
            elements.webgpuStatus.textContent = 'WebGPU is not supported - CPU computation fallback.';
            elements.webgpuStatus.className = 'status warning';
            elements.joinComputation.disabled = false; return false; // Allow CPU fallback join
        }

        logTaskActivity("Initializing WebGPU...");
        let selectedAdapter = null;
        let adapters = [];
        try { const highPerfAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (highPerfAdapter) adapters.push({ adapter: highPerfAdapter, type: 'high-performance' }); } catch (e) { console.warn("No high-perf adapter:", e.message); }
        try { const lowPowerAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' }); if (lowPowerAdapter && !adapters.some(a => a.adapter === lowPowerAdapter)) adapters.push({ adapter: lowPowerAdapter, type: 'low-power' }); } catch (e) { console.warn("No low-power adapter:", e.message); }
        try { const defaultAdapter = await navigator.gpu.requestAdapter(); if (defaultAdapter && !adapters.some(a => a.adapter === defaultAdapter)) adapters.push({ adapter: defaultAdapter, type: 'default' }); } catch (e) { console.warn("No default adapter:", e.message); }

        if (adapters.length === 0) throw new Error("No WebGPU adapters found.");

        for (let i = 0; i < adapters.length; i++) {
            try { adapters[i].info = await adapters[i].adapter.requestAdapterInfo(); } catch (e) { adapters[i].info = { vendor: 'Unknown', architecture: 'Unknown', device: 'Unknown', description: e.message }; }
        }

        // Prefer discrete GPU (NVIDIA/AMD), then by type preference
        const discreteVendors = ['nvidia', 'amd', 'advanced micro devices']; // common vendor names
        selectedAdapter = adapters.find(a => discreteVendors.some(v => a.info.vendor?.toLowerCase().includes(v) || a.info.description?.toLowerCase().includes(v)))?.adapter ||
                          adapters.find(a => a.type === 'high-performance')?.adapter ||
                          adapters[0].adapter;

        state.adapter = selectedAdapter;
        state.adapterInfo = await state.adapter.requestAdapterInfo();

        state.device = await state.adapter.requestDevice();
        state.webgpuSupported = true;

        elements.webgpuStatus.textContent = `WebGPU Ready: ${state.adapterInfo.vendor} - ${state.adapterInfo.architecture}`;
        elements.webgpuStatus.className = 'status success';
        let gpuInfoHTML = `<strong>Vendor:</strong> ${state.adapterInfo.vendor} <br><strong>Arch:</strong> ${state.adapterInfo.architecture}`;
        if (state.adapterInfo.device) gpuInfoHTML += `<br><strong>Device:</strong> ${state.adapterInfo.device}`;
        if (state.adapterInfo.description && state.adapterInfo.description !== state.adapterInfo.device) gpuInfoHTML += `<br><strong>Desc:</strong> ${state.adapterInfo.description}`;
        elements.gpuInfo.innerHTML = gpuInfoHTML;
        elements.gpuInfo.className = 'status success';

        elements.joinComputation.disabled = false;
        logTaskActivity(`WebGPU initialized with: ${state.adapterInfo.vendor} ${state.adapterInfo.architecture}`);
        return true;

    } catch (error) {
        console.error('Error initializing WebGPU:', error);
        elements.webgpuStatus.textContent = `WebGPU error: ${error.message} - CPU fallback.`;
        elements.webgpuStatus.className = 'status warning';
        elements.gpuInfo.innerHTML = `WebGPU initialization failed.`;
        elements.gpuInfo.className = 'status error';
        elements.joinComputation.disabled = false; // Allow CPU fallback join
        return false;
    }
}

// Matrix multiplication using WebGPU (existing)
async function multiplyMatricesGPU(matrixA, matrixB, size, startRow, endRow) {
    logTaskActivity(`GPU: Matrix task rows ${startRow}-${endRow}`);
    const startTime = performance.now();
    try {
        const flatMatrixA = new Float32Array(size * size);
        const flatMatrixB = new Float32Array(size * size);
        for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) { flatMatrixA[i * size + j] = matrixA[i][j]; flatMatrixB[i * size + j] = matrixB[i][j]; }

        const shaderModule = state.device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
                @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
                @group(0) @binding(2) var<storage, write> resultMatrix: array<f32>;
                struct Uniforms { size: u32, startRow: u32, endRow: u32, }
                @group(0) @binding(3) var<uniform> uniforms: Uniforms;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let row = global_id.x + uniforms.startRow;
                    let col = global_id.y;
                    if (row >= uniforms.endRow || col >= uniforms.size) { return; }
                    var sum: f32 = 0.0;
                    for (var i: u32 = 0; i < uniforms.size; i = i + 1) {
                        sum = sum + matrixA[row * uniforms.size + i] * matrixB[i * uniforms.size + col];
                    }
                    resultMatrix[(row - uniforms.startRow) * uniforms.size + col] = sum;
                }`
        });
        const aBuffer = state.device.createBuffer({ size: flatMatrixA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bBuffer = state.device.createBuffer({ size: flatMatrixB.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const rowsToCalculate = endRow - startRow;
        const resultBufferSize = rowsToCalculate * size * Float32Array.BYTES_PER_ELEMENT;
        const resultBuffer = state.device.createBuffer({ size: resultBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const uniformBuffer = state.device.createBuffer({ size: 3 * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const uniformData = new Uint32Array([size, startRow, endRow]);

        state.device.queue.writeBuffer(aBuffer, 0, flatMatrixA);
        state.device.queue.writeBuffer(bBuffer, 0, flatMatrixB);
        state.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const computePipeline = state.device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
        const bindGroup = state.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [ { binding: 0, resource: { buffer: aBuffer } }, { binding: 1, resource: { buffer: bBuffer } }, { binding: 2, resource: { buffer: resultBuffer } }, { binding: 3, resource: { buffer: uniformBuffer } }, ]
        });
        const commandEncoder = state.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline); passEncoder.setBindGroup(0, bindGroup);
        const rowWorkgroups = Math.ceil((endRow - startRow) / 8);
        const colWorkgroups = Math.ceil(size / 8);
        passEncoder.dispatchWorkgroups(rowWorkgroups, colWorkgroups); passEncoder.end();

        const readBuffer = state.device.createBuffer({ size: resultBufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultBufferSize);
        state.device.queue.submit([commandEncoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Float32Array(readBuffer.getMappedRange());
        const result = new Array(rowsToCalculate);
        for (let i = 0; i < rowsToCalculate; i++) { result[i] = new Array(size); for (let j = 0; j < size; j++) result[i][j] = resultArray[i * size + j]; }
        readBuffer.unmap();
        aBuffer.destroy(); bBuffer.destroy(); resultBuffer.destroy(); readBuffer.destroy(); uniformBuffer.destroy(); // Cleanup

        const processingTime = performance.now() - startTime;
        logTaskActivity(`GPU: Matrix task completed in ${processingTime.toFixed(0)}ms`);
        return { result, processingTime };
    } catch (error) { logTaskActivity(`GPU: Matrix task ERROR: ${error.message}`, 'error'); throw error; }
}

// CPU fallback for matrix multiplication (existing)
async function multiplyMatricesCPU(matrixA, matrixB, size, startRow, endRow) {
    logTaskActivity(`CPU: Matrix task rows ${startRow}-${endRow}`);
    const startTime = performance.now();
    try {
        const rowsToCalculate = endRow - startRow; const result = new Array(rowsToCalculate);
        for (let i = 0; i < rowsToCalculate; i++) {
            result[i] = new Array(size); const rowIndex = startRow + i;
            for (let j = 0; j < size; j++) { let sum = 0; for (let k = 0; k < size; k++) sum += matrixA[rowIndex][k] * matrixB[k][j]; result[i][j] = sum; }
        }
        const processingTime = performance.now() - startTime;
        logTaskActivity(`CPU: Matrix task completed in ${processingTime.toFixed(0)}ms`);
        return { result, processingTime };
    } catch (error) { logTaskActivity(`CPU: Matrix task ERROR: ${error.message}`, 'error'); throw error; }
}

// Process a matrix task (existing)
async function processMatrixTask(task) {
    state.currentTask = task;
    elements.taskStatus.textContent = `Matrix task ${task.id} (rows ${task.startRow}-${task.endRow})`;
    elements.taskStatus.className = 'status info';
    try {
        let result;
        if (state.webgpuSupported && state.device) {
            result = await multiplyMatricesGPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        } else {
            result = await multiplyMatricesCPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        }
        state.completedTasks++; state.statistics.processingTime += result.processingTime;
        elements.myTasks.textContent = state.completedTasks;
        elements.processingTime.textContent = `${result.processingTime.toFixed(0)}ms`;
        elements.taskStatus.textContent = `Matrix task ${task.id} completed`; elements.taskStatus.className = 'status success';
        return { taskId: task.id, result: result.result, processingTime: result.processingTime };
    } catch (error) {
        elements.taskStatus.textContent = `Matrix task error: ${error.message}`; elements.taskStatus.className = 'status error';
        // Emit error to server for this task if needed
        socket.emit('task:error', { taskId: task.id, message: error.message, type: 'matrixMultiply' });
        throw error; // Rethrow to be caught by task:assign handler
    }
}

// Join the computation (existing)
function joinComputation() {
    elements.joinComputation.disabled = true; elements.leaveComputation.disabled = false;
    const mode = (state.webgpuSupported && state.device) ? 'WebGPU' : 'CPU';
    logTaskActivity(`Joining computation network (${mode})...`);
    socket.emit('client:join', { gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback', device: 'CPU Computation' } });
    state.isComputing = true; // For matrix tasks
    elements.computationStatus.textContent = `Joined computation network (${mode}), waiting for tasks`;
    elements.computationStatus.className = 'status info';
}

// Leave the computation (existing)
function leaveComputation() {
    elements.joinComputation.disabled = false; elements.leaveComputation.disabled = true;
    state.isComputing = false; state.currentTask = null; // For matrix tasks
    logTaskActivity('Left computation network');
    elements.taskStatus.textContent = 'No matrix task assigned'; elements.taskStatus.className = 'status info';
    elements.computationStatus.textContent = 'Not participating in computation'; elements.computationStatus.className = 'status warning';
    socket.emit('client:leave'); // Inform server
}

// Request a matrix task from the server (existing)
function requestMatrixTask() {
    if (state.isComputing && !state.currentTask) { socket.emit('task:request'); }
}

// Start a new matrix computation (admin - existing)
function startMatrixComputation() {
    const matrixSize = parseInt(elements.matrixSize.value);
    const chunkSize = parseInt(elements.chunkSize.value);
    socket.emit('admin:start', { matrixSize, chunkSize });
    logAdminActivity(`Starting new matrix computation: ${matrixSize}x${matrixSize} with chunk size ${chunkSize}`, 'matrix');
}

// Add a log entry to the task log (existing)
function logTaskActivity(message, type = 'info') {
    if (IS_HEADLESS) { // For headless, log to console and optionally update title
        console.log(`[Worker ${WORKER_ID} Log]: ${message}`);
        if (type === 'error') console.error(`[Worker ${WORKER_ID} ErrorLog]: ${message}`);
        document.title = `Worker ${WORKER_ID} | ${message.substring(0,50)}`;
        return;
    }
    const logItem = document.createElement('div'); logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.taskLog.appendChild(logItem); elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

// Add a log entry to the admin log (existing, adapted for different logs)
function logAdminActivity(message, panelType = 'matrix', type = 'info') {
    if (IS_HEADLESS) { console.log(`[Admin Log]: ${message}`); return; }
    const logContainer = panelType === 'wgsl' ? elements.adminLogWgsl : elements.adminLogMatrix;
    const logItem = document.createElement('div'); logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.appendChild(logItem); logContainer.scrollTop = logContainer.scrollHeight;
}

// Update client display (existing)
function updateClientDisplay(clients) {
    if (IS_HEADLESS || !elements.clientGrid) return;
    elements.clientGrid.innerHTML = '';
    if (!clients || clients.length === 0) { elements.clientGrid.innerHTML = '<div>No clients connected</div>'; return; }
    clients.forEach(client => {
        const isCurrentClient = client.id === state.clientId;
        const timeSinceActive = Date.now() - client.lastActive;
        const isActive = client.connected && timeSinceActive < 60000; // 60 seconds threshold
        const clientEl = document.createElement('div');
        clientEl.className = `client-card ${!isActive ? 'client-inactive' : ''} ${client.usingCpu ? 'client-cpu' : 'client-gpu'} ${client.isPuppeteer ? 'client-puppeteer' : ''}`;
        let clientHTML = `<div>${isCurrentClient ? '<strong>You</strong>' : 'Client'} ${client.isPuppeteer ? '(Puppeteer)' : ''}</div>
                          <div><small>${client.id.substring(0, 8)}...</small></div>`;
        if (client.gpuInfo) {
            clientHTML += `<div><small>${client.gpuInfo.isCpuComputation ? 'CPU' : (client.gpuInfo.vendor?.split(' ')[0] || 'GPU')}</small></div>`;
        }
        clientHTML += `<div>Tasks: ${client.completedTasks || 0}</div>`;
        clientHTML += `<div><small>${isActive ? 'Active' : 'Inactive'}</small></div>`;
        clientEl.innerHTML = clientHTML;
        elements.clientGrid.appendChild(clientEl);
    });
}

// Update stats display (existing)
function updateStatsDisplay(stats) {
    if (IS_HEADLESS || !elements.activeClients) return;
    elements.activeClients.textContent = stats.activeClients || 0;
    elements.totalTasks.textContent = stats.totalTasks || 0;
    elements.completedTasks.textContent = stats.completedTasks || 0;
    if (stats.elapsedTime) { elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`; }
}


// --- Custom WGSL Workload Handling ---
socket.on('workload:new', async meta => {
    if (!state.device) {
        logTaskActivity(`Received custom workload "${meta.label}" but WebGPU device not ready. Skipping.`, 'warning');
        socket.emit('workload:error', { id: meta.id, message: 'WebGPU device not available on client.' });
        return;
    }
    logTaskActivity(`Received custom workload "${meta.label}" (ID: ${meta.id.substring(0,6)}). Processing...`);
    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder; // Declare here for finally block

    try {
        const startTime = performance.now();
        // (1) Compile shader
        shader = state.device.createShaderModule({ code: meta.wgsl });
        const info = await shader.getCompilationInfo();
        if (info.messages.some(m => m.type === 'error')) {
            const errorMessages = info.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed: ${errorMessages}`);
        }

        // (2) Create pipeline (simple "storage-in-storage-out" for now)
        // TODO: Extend based on meta.bindLayout for more complex scenarios
        if (meta.bindLayout !== "storage-in-storage-out") {
            throw new Error(`Unsupported bindLayout: ${meta.bindLayout}. Only 'storage-in-storage-out' is currently implemented.`);
        }
        pipeline = state.device.createComputePipeline({
            layout: 'auto', // This is convenient but can be explicit for performance/validation
            compute: { module: shader, entryPoint: meta.entry || 'main' }
        });

        // (3) Input -> GPU (if provided)
        if (meta.input) {
            // Assuming meta.input is base64 encoded binary data
            const inputDataBytes = Uint8Array.from(atob(meta.input), c => c.charCodeAt(0));
            inputBuf = state.device.createBuffer({
                size: inputDataBytes.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true // Create mapped for easier writing
            });
            new Uint8Array(inputBuf.getMappedRange()).set(inputDataBytes);
            inputBuf.unmap();
        }

        // (4) Output buffer
        outBuf = state.device.createBuffer({
            size: meta.outputSize, // Bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // (5) Bind group, dispatch, readback
        const entries = [];
        if (inputBuf) { // If there's an input buffer, it's binding 0
            entries.push({ binding: 0, resource: { buffer: inputBuf } });
            entries.push({ binding: 1, resource: { buffer: outBuf } }); // Output is binding 1
        } else { // If no input buffer, output buffer might be used as input (e.g. for GOL) or is just output
             // This assumes shader expects binding 0 for input (even if unused) and binding 1 for output.
             // A more robust system would detail bindings in meta.
            const placeholderInputSize = meta.outputSize; // Or some other default if shader reads from binding 0
            inputBuf = state.device.createBuffer({ size: placeholderInputSize, usage: GPUBufferUsage.STORAGE }); // Dummy if not read
            entries.push({ binding: 0, resource: { buffer: inputBuf } }); // Placeholder if no real input, or shader handles it.
            entries.push({ binding: 1, resource: { buffer: outBuf } });
        }

        const bindGroup = state.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0), // Assumes a single bind group at index 0
            entries: entries
        });

        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...meta.workgroupCount);
        pass.end();

        // (6) Copy result to readable buffer
        readBuf = state.device.createBuffer({
            size: meta.outputSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, meta.outputSize);
        state.device.queue.submit([commandEncoder.finish()]);

        await state.device.queue.onSubmittedWorkDone(); // Wait for GPU to finish
        await readBuf.mapAsync(GPUMapMode.READ);

        const resultData = new Uint8Array(readBuf.getMappedRange().slice(0)); // Get a copy
        const result = Array.from(resultData); // Convert Uint8Array to a plain array of numbers

        readBuf.unmap(); // Unmap after copying

        const processingTime = performance.now() - startTime;
        logTaskActivity(`Custom workload "${meta.label}" completed in ${processingTime.toFixed(0)}ms. Result size: ${result.length} bytes.`);
        socket.emit('workload:done', { id: meta.id, result });

    } catch (err) {
        logTaskActivity(`Error processing custom workload "${meta.label}": ${err.message}`, 'error');
        console.error(`Custom workload error for ${meta.id}:`, err);
        socket.emit('workload:error', { id: meta.id, message: err.message });
    } finally {
        // Cleanup GPU resources
        if (inputBuf) inputBuf.destroy();
        if (outBuf) outBuf.destroy();
        if (readBuf) readBuf.destroy();
        // Shader modules and pipelines are often cached or reused, but can be destroyed if truly one-off.
    }
});

function updateActiveWgslWorkloadsDisplay(workloadsFromServer) {
    if (IS_HEADLESS || !elements.activeWgslWorkloadsGrid) return;
    elements.activeWgslWorkloadsGrid.innerHTML = '';
    if (!workloadsFromServer || workloadsFromServer.length === 0) {
        elements.activeWgslWorkloadsGrid.innerHTML = '<p>No custom WGSL workloads currently active.</p>';
        return;
    }
    workloadsFromServer.forEach(wl => {
        const card = document.createElement('div');
        card.className = 'wgsl-card';
        card.innerHTML = `
            <h4>${wl.label} <small>(${wl.id.substring(0,6)})</small></h4>
            <p>Status: <strong>${wl.status || 'N/A'}</strong></p>
            <p>Dispatch: ${wl.workgroupCount.join('x')} groups</p>
            <p>Output: ${wl.outputSize} bytes</p>
            ${wl.wgsl ? `<p>Shader: <pre>${wl.wgsl.substring(0,100)}${wl.wgsl.length > 100 ? '...' : ''}</pre></p>` : ''}
            ${wl.results ? `<p>Submissions: ${wl.results.length}</p>` : ''}
            ${wl.finalResult ? `<p>Final Result (first 10 bytes): <pre>${JSON.stringify(wl.finalResult.slice(0,10))}...</pre></p>` : ''}
        `;
        elements.activeWgslWorkloadsGrid.appendChild(card);
    });
}


// Socket.io event handlers (existing + new)
socket.on('connect', () => {
    state.connected = true;
    elements.clientStatus.textContent = 'Connected to server'; elements.clientStatus.className = 'status success';
    logTaskActivity('Connected to computation server');
    if (elements.joinComputation) elements.joinComputation.disabled = false;
});

socket.on('disconnect', () => {
    state.connected = false;
    elements.clientStatus.textContent = 'Disconnected from server'; elements.clientStatus.className = 'status error';
    logTaskActivity('Disconnected from server', 'error');
    if (elements.joinComputation) elements.joinComputation.disabled = true;
    if (elements.leaveComputation) elements.leaveComputation.disabled = true;
    state.isComputing = false;
});

socket.on('register', (data) => {
    state.clientId = data.clientId;
    elements.clientStatus.textContent = `Connected as: ${data.clientId.substring(0, 8)}... ${IS_HEADLESS ? '(Headless Worker '+WORKER_ID+')' : ''}`;
    if (elements.joinComputation) elements.joinComputation.disabled = false;
});

socket.on('state:update', (data) => { // For matrix computation status
    if (IS_HEADLESS) return;
    if (data.isRunning) {
        elements.computationStatus.textContent = 'Matrix computation in progress'; elements.computationStatus.className = 'status info';
    } else {
        elements.computationStatus.textContent = 'No matrix computation in progress'; elements.computationStatus.className = 'status warning';
    }
    updateStatsDisplay(data.stats);
});

socket.on('clients:update', (data) => { updateClientDisplay(data.clients); });

socket.on('task:assign', async (task) => { // For matrix tasks
    if (!state.isComputing && !IS_HEADLESS) return; // Regular clients must join. Headless auto-processes.
    if (task.type && task.type !== 'matrixMultiply') {
        logTaskActivity(`Received task of unknown type: ${task.type}. Ignoring.`, 'warning');
        return;
    }
    try {
        const result = await processMatrixTask(task);
        socket.emit('task:complete', result); state.currentTask = null; requestMatrixTask();
    } catch (error) {
        logTaskActivity(`Error processing matrix task: ${error.message}`, 'error');
        state.currentTask = null; setTimeout(requestMatrixTask, 2000); // Retry after delay
    }
});

socket.on('task:wait', (data) => { // For matrix tasks
    if (data && data.type && data.type !== 'matrixMultiply') return;
    elements.taskStatus.textContent = 'Waiting for available matrix tasks'; elements.taskStatus.className = 'status warning';
    logTaskActivity('No matrix tasks available, waiting...'); state.currentTask = null;
    setTimeout(requestMatrixTask, 5000 + Math.random() * 5000); // Longer, randomized wait
});

socket.on('computation:complete', (data) => { // For matrix computation
    if (data && data.type && data.type !== 'matrixMultiply') return;
    elements.computationStatus.textContent = `Matrix computation completed in ${data.totalTime.toFixed(1)}s`;
    elements.computationStatus.className = 'status success';
    logTaskActivity(`Matrix computation completed in ${data.totalTime.toFixed(1)}s`); state.currentTask = null;
});

socket.on('task:error', (data) => { // For matrix tasks
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Server error for matrix task ${data.taskId}: ${data.message}`, 'error');
});

socket.on('task:submitted', (data) => { // For matrix tasks
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Matrix task ${data.taskId} submitted. Awaiting verification.`);
});
socket.on('task:verified', (data) => { // For matrix tasks
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Your submission for matrix task ${data.taskId} was verified!`, 'success');
});

// For custom WGSL workload status updates from server (e.g. if server maintains a list)
socket.on('workloads:active_list', (workloads) => {
    if (IS_HEADLESS || !elements.activeWgslWorkloadsGrid) return;
    updateActiveWgslWorkloadsDisplay(workloads);
});
socket.on('workload:complete', (data) => {
    logTaskActivity(`Server confirmed custom workload "${data.label || data.id.substring(0,6)}" is complete!`, 'success');
    // Potentially update a specific workload card in the UI if displaying them
});


// Event listeners
if (!IS_HEADLESS) {
    elements.joinComputation.addEventListener('click', joinComputation);
    elements.leaveComputation.addEventListener('click', leaveComputation);
    elements.startComputation.addEventListener('click', startMatrixComputation);
    elements.toggleAdmin.addEventListener('click', () => {
        elements.adminPanel.style.display = (elements.adminPanel.style.display === 'block') ? 'none' : 'block';
    });

    elements.pushWgslWorkloadButton.addEventListener('click', async () => {
        const payload = {
            label: elements.wgslLabel.value || 'Untitled WGSL Workload',
            wgsl: elements.wgslSrc.value,
            entry: elements.wgslEntryPoint.value || 'main',
            workgroupCount: [
                +elements.wgslGroupsX.value,
                +elements.wgslGroupsY.value,
                +elements.wgslGroupsZ.value
            ],
            bindLayout: elements.wgslBindLayout.value,
            outputSize: +elements.wgslOutputSize.value,
            input: elements.wgslInputData.value.trim() || undefined // Optional input
        };

        if (!payload.wgsl || !payload.workgroupCount.every(c => c > 0) || !payload.outputSize) {
            logAdminActivity('WGSL Upload: Missing WGSL source, valid workgroup counts, or output size.', 'wgsl', 'error');
            return;
        }

        logAdminActivity(`Pushing WGSL workload: "${payload.label}"...`, 'wgsl');
        try {
            const res = await fetch('/api/workloads', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const json = await res.json();
            if (res.ok && json.ok) {
                logAdminActivity(`WGSL Workload ${json.id} (${payload.label}) queued successfully.`, 'wgsl', 'success');
            } else {
                logAdminActivity(`WGSL Upload Error: ${json.error || 'Unknown server error'} (Status: ${res.status})`, 'wgsl', 'error');
            }
        } catch (err) {
            logAdminActivity(`WGSL Upload Fetch Error: ${err.message}`, 'wgsl', 'error');
            console.error("WGSL Push error:", err);
        }
    });
}


// Initialize the application
async function init() {
    if (IS_HEADLESS) {
        document.documentElement.style.display = 'none';
        console.log(`Puppeteer worker #${WORKER_ID} initializing. UI hidden.`);
        document.title = `Headless Worker ${WORKER_ID} - Initializing`;
    }

    await initWebGPU(); // Initialize WebGPU, enable join button on success

    if (IS_HEADLESS) {
        if (state.webgpuSupported && elements.joinComputation && !elements.joinComputation.disabled) {
            logTaskActivity("Headless mode: Auto-joining computation...");
            joinComputation(); // Auto-join for headless clients
        } else if (!state.webgpuSupported && elements.joinComputation && !elements.joinComputation.disabled) {
            logTaskActivity("Headless mode: WebGPU not fully supported/failed, attempting to join for CPU tasks...");
            joinComputation(); // Still join if CPU fallback is an option
        } else {
            logTaskActivity("Headless mode: Cannot auto-join. WebGPU init might have failed or button disabled.", 'warning');
        }
    } else { // For regular browser clients
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('admin')) {
            elements.adminPanel.style.display = 'block';
        }
        // Ensure join button is enabled after a small delay if not already by initWebGPU
        setTimeout(() => {
            if (elements.joinComputation && elements.joinComputation.disabled && (!navigator.gpu || !window.isSecureContext)) {
                elements.joinComputation.disabled = false; // Force enable if initial checks failed but user might still want to try CPU
            }
        }, 2000);
    }
}

init();