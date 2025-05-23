// client.js

// State management
const state = {
    webgpuSupported: false, device: null, adapter: null, adapterInfo: null,
    connected: false, clientId: null,
    isComputingMatrix: false,    // Busy with a matrix task instance
    isComputingWgsl: false,      // Busy with a non-chunked WGSL task instance
    isComputingChunk: false,     // Busy with a WGSL chunk instance
    currentTask: null,           // Holds current matrix task { assignmentId, id, ... } or WGSL task meta
    completedTasks: 0,
    statistics: { processingTime: 0 }
};

const elements = {
    webgpuStatus: document.getElementById('webgpu-status'),
    gpuInfo: document.getElementById('gpu-info'),
    computationStatus: document.getElementById('computation-status'),
    clientStatus: document.getElementById('client-status'),
    taskStatus: document.getElementById('task-status'),
    joinComputation: document.getElementById('join-computation'),
    leaveComputation: document.getElementById('leave-computation'),
    startComputation: document.getElementById('start-computation'), // Admin
    toggleAdmin: document.getElementById('toggle-admin'), // Admin
    adminPanel: document.getElementById('admin-panel'), // Admin
    matrixSize: document.getElementById('matrix-size'), // Admin
    chunkSize: document.getElementById('chunk-size'), // Admin
    taskLog: document.getElementById('task-log'),
    adminLogMatrix: document.getElementById('admin-log-matrix'), // Admin
    adminLogWgsl: document.getElementById('admin-log-wgsl'), // Admin
    adminLogSystem: document.getElementById('admin-log-system'), // Admin for K param
    clientGrid: document.getElementById('client-grid'),
    activeClients: document.getElementById('active-clients'),
    totalTasks: document.getElementById('total-tasks'),
    completedTasks: document.getElementById('completed-tasks'),
    elapsedTime: document.getElementById('elapsed-time'),
    myTasks: document.getElementById('my-tasks'),
    processingTime: document.getElementById('processing-time'),
    wgslLabel: document.getElementById('wgsl-label'), // Admin
    wgslEntryPoint: document.getElementById('wgsl-entry-point'), // Admin
    wgslSrc: document.getElementById('wgsl-src'), // Admin
    wgslGroupsX: document.getElementById('wgsl-groups-x'), // Admin
    wgslGroupsY: document.getElementById('wgsl-groups-y'), // Admin
    wgslGroupsZ: document.getElementById('wgsl-groups-z'), // Admin
    wgslBindLayout: document.getElementById('wgsl-bind-layout'), // Admin
    wgslOutputSize: document.getElementById('wgsl-output-size'), // Admin
    wgslInputData: document.getElementById('wgsl-input-data'), // Admin
    pushWgslWorkloadButton: document.getElementById('push-wgsl-workload'), // Admin
    activeWgslWorkloadsGrid: document.getElementById('active-wgsl-workloads-grid'), // Admin
    startQueuedWgslButton: document.getElementById('startQueuedWgslButton'), // Admin
    // Admin K parameter UI
    adminKValueInput: document.getElementById('admin-k-value'),
    setKButton: document.getElementById('set-k-button'),
    currentKDisplay: document.getElementById('current-k-display'),
};

const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';
const socket = io({ query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {} });

async function initWebGPU() {
    // ... (initWebGPU - unchanged from your provided version)
    try {
        if (!window.isSecureContext) {
            elements.webgpuStatus.innerHTML = `<div>WebGPU requires a secure context (HTTPS or localhost).</div>`;
            elements.webgpuStatus.className = 'status error';
            if (elements.joinComputation) elements.joinComputation.disabled = false; return false;
        }
        if (!navigator.gpu) {
            elements.webgpuStatus.textContent = 'WebGPU is not supported - CPU computation fallback.';
            elements.webgpuStatus.className = 'status warning';
            if (elements.joinComputation) elements.joinComputation.disabled = false; return false;
        }

        logTaskActivity("Initializing WebGPU...");
        let selectedAdapter = null;
        const rawAdapters = [];

        try { const highPerfAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (highPerfAdapter) rawAdapters.push(highPerfAdapter); } catch (e) { console.warn("No high-perf adapter:", e.message); }
        try { const lowPowerAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' }); if (lowPowerAdapter && !rawAdapters.find(a => a === lowPowerAdapter)) rawAdapters.push(lowPowerAdapter); } catch (e) { console.warn("No low-power adapter:", e.message); }
        try { const defaultAdapter = await navigator.gpu.requestAdapter(); if (defaultAdapter && !rawAdapters.find(a => a === defaultAdapter)) rawAdapters.push(defaultAdapter); } catch (e) { console.warn("No default adapter found or request failed:", e.message); }


        if (rawAdapters.length === 0) throw new Error("No WebGPU adapters found.");

        const adaptersWithInfo = [];
        for (const adapter of rawAdapters) {
            let info = { vendor: 'Unknown', architecture: 'Unknown', device: 'Unknown', description: 'Info not available' };
            try {
                if (adapter && typeof adapter.requestAdapterInfo === 'function') {
                    const detailedInfo = await adapter.requestAdapterInfo();
                    info.vendor = detailedInfo.vendor || info.vendor;
                    info.architecture = detailedInfo.architecture || info.architecture;
                    info.device = detailedInfo.device || info.device;
                    info.description = detailedInfo.description || info.description;
                } else {
                    console.warn("adapter.requestAdapterInfo() is not available on this adapter. Using default info.");
                }
            } catch (e) {
                console.warn(`Error fetching adapter info: ${e.message}. Using default info.`);
                info.description = `Error fetching info: ${e.message}`;
            }
            adaptersWithInfo.push({ adapter, info });
        }

        const discreteVendors = ['nvidia', 'amd', 'advanced micro devices', 'apple', 'qualcomm', 'arm'];
        let foundAdapterEntry = adaptersWithInfo.find(a =>
            discreteVendors.some(v =>
                (a.info.vendor?.toLowerCase().includes(v) ||
                 a.info.description?.toLowerCase().includes(v)) &&
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver') &&
                !a.info.description?.toLowerCase().includes('llvmpipe')
            )
        );

        if (!foundAdapterEntry && adaptersWithInfo.length > 0) {
            foundAdapterEntry = adaptersWithInfo.find(a =>
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver') &&
                !a.info.description?.toLowerCase().includes('llvmpipe')
            ) || adaptersWithInfo[0];
        }


        if (!foundAdapterEntry || !foundAdapterEntry.adapter) {
            throw new Error("No suitable WebGPU adapter found after attempting to get info.");
        }

        state.adapter = foundAdapterEntry.adapter;
        state.adapterInfo = foundAdapterEntry.info;

        state.device = await state.adapter.requestDevice();
        state.webgpuSupported = true;

        let vendorDisplay = state.adapterInfo.vendor !== 'Unknown' && state.adapterInfo.vendor !== 'N/A' ? state.adapterInfo.vendor : "GPU";
        let archDisplay = state.adapterInfo.architecture !== 'Unknown' && state.adapterInfo.architecture !== 'N/A' ? state.adapterInfo.architecture : "";

        elements.webgpuStatus.textContent = `WebGPU Ready: ${vendorDisplay}${archDisplay ? ' - ' + archDisplay : ''}`;
        elements.webgpuStatus.className = 'status success';

        let gpuInfoHTML = `<strong>Vendor:</strong> ${state.adapterInfo.vendor} <br><strong>Arch:</strong> ${state.adapterInfo.architecture}`;
        if (state.adapterInfo.device && state.adapterInfo.device !== 'Unknown') gpuInfoHTML += `<br><strong>Device ID:</strong> ${state.adapterInfo.device}`;
        if (state.adapterInfo.description && state.adapterInfo.description !== 'Info not available' && !state.adapterInfo.description.startsWith('Error fetching info')) {
            gpuInfoHTML += `<br><strong>Desc:</strong> ${state.adapterInfo.description}`;
        } else if (state.adapterInfo.description !== 'Info not available') {
             gpuInfoHTML += `<br><strong>Desc (Debug):</strong> (${state.adapterInfo.description})`;
        }
        elements.gpuInfo.innerHTML = gpuInfoHTML;
        elements.gpuInfo.className = 'status success';

        if(elements.joinComputation) elements.joinComputation.disabled = false;
        logTaskActivity(`WebGPU initialized. Adapter: ${vendorDisplay} ${archDisplay}. Detail: ${state.adapterInfo.description}`);
        return true;

    } catch (error) {
        console.error('Error initializing WebGPU:', error);
        elements.webgpuStatus.textContent = `WebGPU error: ${error.message} - CPU fallback.`;
        elements.webgpuStatus.className = 'status warning';
        elements.gpuInfo.innerHTML = `WebGPU initialization failed.`;
        elements.gpuInfo.className = 'status error';
        state.webgpuSupported = false;
        state.adapter = null;
        state.device = null;
        state.adapterInfo = { vendor: 'N/A', architecture: 'N/A', device: 'N/A', description: error.message };
        if(elements.joinComputation) elements.joinComputation.disabled = false;
        return false;
    }
}

async function multiplyMatricesGPU(matrixA, matrixB, size, startRow, endRow) {
    // ... (multiplyMatricesGPU - unchanged from your provided version)
    if (!state.device) throw new Error("WebGPU device not available for GPU computation.");
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
        const resultArrayBuffer = readBuffer.getMappedRange();
        const resultArray = new Float32Array(resultArrayBuffer.slice(0)); // Create a copy
        const result = new Array(rowsToCalculate);
        for (let i = 0; i < rowsToCalculate; i++) { result[i] = new Array(size); for (let j = 0; j < size; j++) result[i][j] = resultArray[i * size + j]; }
        readBuffer.unmap(); // Unmap after copying
        aBuffer.destroy(); bBuffer.destroy(); resultBuffer.destroy(); readBuffer.destroy(); uniformBuffer.destroy();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`GPU: Matrix task completed in ${processingTime.toFixed(0)}ms`);
        return { result, processingTime };
    } catch (error) { logTaskActivity(`GPU: Matrix task ERROR: ${error.message}`, 'error'); throw error; }
}

async function multiplyMatricesCPU(matrixA, matrixB, size, startRow, endRow) {
    // ... (multiplyMatricesCPU - unchanged from your provided version)
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

async function processMatrixTask(task) { // task contains { assignmentId, id (logical), ... }
    state.currentTask = task;
    state.isComputingMatrix = true;
    elements.taskStatus.textContent = `Matrix task ${task.id} (rows ${task.startRow}-${task.endRow}, Instance ${task.assignmentId.substring(0,6)})`;
    elements.taskStatus.className = 'status info';
    updateComputationStatusDisplay();

    try {
        let resultData;
        if (state.webgpuSupported && state.device) {
            resultData = await multiplyMatricesGPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        } else {
            resultData = await multiplyMatricesCPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        }
        state.completedTasks++;
        state.statistics.processingTime += resultData.processingTime;
        elements.myTasks.textContent = state.completedTasks;
        elements.processingTime.textContent = `${resultData.processingTime.toFixed(0)}ms`;
        elements.taskStatus.textContent = `Matrix task ${task.id} (Instance ${task.assignmentId.substring(0,6)}) completed`;
        elements.taskStatus.className = 'status success';
        return { result: resultData.result, processingTime: resultData.processingTime }; // Only result and time needed for emission
    } catch (error) {
        elements.taskStatus.textContent = `Matrix task error: ${error.message}`;
        elements.taskStatus.className = 'status error';
        socket.emit('task:error', { assignmentId: task.assignmentId, taskId: task.id, message: error.message, type: 'matrixMultiply' });
        throw error; // Rethrow to be caught by caller
    } finally {
        state.isComputingMatrix = false;
        state.currentTask = null;
        updateComputationStatusDisplay();
    }
}

function joinComputation() {
    if(elements.joinComputation) elements.joinComputation.disabled = true;
    if(elements.leaveComputation) elements.leaveComputation.disabled = false;
    const mode = (state.webgpuSupported && state.device) ? 'WebGPU' : 'CPU';
    logTaskActivity(`Joining computation network (${mode})...`);
    socket.emit('client:join', { gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback', device: 'CPU Computation', description: 'No adapter info' } });
    updateComputationStatusDisplay();
}

function leaveComputation() {
    if(elements.joinComputation) elements.joinComputation.disabled = false;
    if(elements.leaveComputation) elements.leaveComputation.disabled = true;
    state.isComputingMatrix = false;
    state.isComputingWgsl = false;
    state.isComputingChunk = false;
    state.currentTask = null;
    logTaskActivity('Left computation network');
    elements.taskStatus.textContent = 'No matrix task assigned';
    elements.taskStatus.className = 'status info';
    updateComputationStatusDisplay();
    socket.emit('client:leave'); // Inform server (optional, server handles disconnects)
}

function requestMatrixTask() {
    if (state.connected && elements.joinComputation && !elements.joinComputation.disabled && // Joined computation pool
        !state.isComputingMatrix && !state.isComputingWgsl && !state.isComputingChunk && !state.currentTask) {
        logTaskActivity("Requesting a matrix task from server...", "debug");
        socket.emit('task:request');
    }
}

function startMatrixComputation() { // Admin action
    const matrixSize = parseInt(elements.matrixSize.value);
    const chunkSize = parseInt(elements.chunkSize.value);
    socket.emit('admin:start', { matrixSize, chunkSize });
    logAdminActivity(`Starting new matrix computation: ${matrixSize}x${matrixSize} with chunk size ${chunkSize}`, 'matrix');
}

function logTaskActivity(message, type = 'info') {
    if (IS_HEADLESS) { /* ... unchanged ... */ return; }
    const logItem = document.createElement('div');
    logItem.className = `status ${type}`; // Use status class for consistent styling
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.taskLog.appendChild(logItem);
    elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

function logAdminActivity(message, panelType = 'matrix', type = 'info') {
    if (IS_HEADLESS) { console.log(`[Admin Log]: ${message}`); return; }
    let logContainer;
    switch (panelType) {
        case 'wgsl': logContainer = elements.adminLogWgsl; break;
        case 'system': logContainer = elements.adminLogSystem; break;
        case 'matrix':
        default: logContainer = elements.adminLogMatrix; break;
    }
    if (logContainer) {
        const logItem = document.createElement('div');
        logItem.className = `status ${type}`; // Use status class for consistent styling
        logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logContainer.appendChild(logItem);
        logContainer.scrollTop = logContainer.scrollHeight;
    } else {
        console.warn(`Admin log container for ${panelType} not found.`);
    }
}

function updateClientDisplay(clients) {
    if (IS_HEADLESS || !elements.clientGrid) return;
    elements.clientGrid.innerHTML = '';
    if (!clients || clients.length === 0) { elements.clientGrid.innerHTML = '<div>No clients connected</div>'; return; }
    clients.forEach(client => {
        const isCurrentClient = client.id === state.clientId;
        const timeSinceActive = Date.now() - client.lastActive;
        const isActive = client.connected && timeSinceActive < 120000; // 2 min threshold for inactive display
        const clientEl = document.createElement('div');
        clientEl.className = `client-card ${!isActive ? 'client-inactive' : ''} ${client.usingCpu ? 'client-cpu' : 'client-gpu'} ${client.isPuppeteer ? 'client-puppeteer' : ''}`;
        let clientHTML = `<div>${isCurrentClient ? '<strong>You</strong>' : 'Client'} ${client.isPuppeteer ? '(Puppeteer)' : ''}</div>
                          <div><small>${client.id.substring(0, 8)}...</small></div>`;
        if (client.gpuInfo) {
            let displayVendor = 'GPU';
            if (client.gpuInfo.isCpuComputation) {
                displayVendor = 'CPU';
            } else if (client.gpuInfo.vendor && client.gpuInfo.vendor !== 'N/A' && client.gpuInfo.vendor !== 'Unknown') {
                displayVendor = client.gpuInfo.vendor.split(' ')[0];
            }
            clientHTML += `<div><small>${displayVendor}</small></div>`;
        }
        clientHTML += `<div>Tasks: ${client.completedTasks || 0}</div>`;

        if (client.isBusyWithMatrixTask) {
            clientHTML += `<div><small class="status info">Busy (Matrix)</small></div>`;
        } else if (client.isBusyWithCustomChunk) {
            clientHTML += `<div><small class="status info">Busy (Chunk)</small></div>`;
        } else if (client.isBusyWithNonChunkedWGSL) { // Server needs to send this flag
            clientHTML += `<div><small class="status info">Busy (WGSL)</small></div>`;
        }

        clientHTML += `<div><small>${isActive ? 'Active' : 'Inactive ('+ Math.round(timeSinceActive/1000) + 's ago)'}</small></div>`;
        clientEl.innerHTML = clientHTML;
        elements.clientGrid.appendChild(clientEl);
    });
}

function updateStatsDisplay(stats) {
    if (IS_HEADLESS || !elements.activeClients) return;
    elements.activeClients.textContent = stats.activeClients || 0;
    elements.totalTasks.textContent = stats.totalTasks || 0;
    elements.completedTasks.textContent = stats.completedTasks || 0;
    if (stats.elapsedTime !== undefined) { elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`; }
}

function updateComputationStatusDisplay() {
    if (IS_HEADLESS || !elements.computationStatus) return;
    let statusText = '';
    let statusClass = 'status info';

    if (state.isComputingChunk) {
        statusText = `Processing WGSL Chunk...`;
    } else if (state.isComputingWgsl) {
        statusText = `Processing Custom WGSL...`;
    } else if (state.isComputingMatrix) {
        statusText = `Processing Matrix Task...`;
    } else if (state.connected && elements.joinComputation && elements.joinComputation.disabled) { // Joined but idle
        statusText = 'Idle, awaiting task.';
    } else if (state.connected) {
        statusText = 'Connected, not joined computation.';
        statusClass = 'status warning';
    } else {
        statusText = 'Disconnected.';
        statusClass = 'status error';
    }
    elements.computationStatus.textContent = statusText;
    elements.computationStatus.className = statusClass;
}


// --- Custom WGSL Workload Handling (Non-chunked and Chunked) ---
// These are now more granularly controlled by specific busy flags.

socket.on('workload:new', async meta => { // For NON-CHUNKED workloads assignment
    if (meta.isChunkParent) {
        logTaskActivity(`Received parent workload "${meta.label}" - its chunks will be assigned separately.`, 'info');
        return;
    }
    if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
        logTaskActivity(`Received non-chunked workload "${meta.label}" but client is busy. Server should re-assign if needed.`, 'warning');
        // Client is busy, server's dispatch logic should handle this.
        return;
    }
    // `meta.status` might be 'pending_dispatch' or 'pending' from server.
    // The server has selected this client for this task.

    if (!state.device && meta.wgsl) { // Check if it's a WGSL task requiring GPU
        logTaskActivity(`Received non-chunked workload "${meta.label}" but WebGPU device not ready. Emitting error.`, 'warning');
        socket.emit('workload:error', { id: meta.id, message: 'WebGPU device not available on client.' });
        return;
    }

    logTaskActivity(`Processing non-chunked custom workload "${meta.label}" (ID: ${meta.id.substring(0,6)})...`);
    state.isComputingWgsl = true; // Set non-chunked WGSL busy flag
    state.currentTask = meta; // Store task metadata
    updateComputationStatusDisplay();

    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder;

    try {
        const startTime = performance.now();
        shader = state.device.createShaderModule({ code: meta.wgsl });
        // ... (rest of non-chunked WGSL processing logic from your provided version)
        const compilationInfo = await shader.getCompilationInfo();
        if (compilationInfo.messages.some(m => m.type === 'error')) {
            const errorMessages = compilationInfo.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed: ${errorMessages}`);
        }
        pipeline = state.device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: meta.entry || 'main' } });
        const bindGroupEntries = [];
        let bindingIndex = 0;
        if (meta.input) {
            const inputDataBytes = Uint8Array.from(atob(meta.input), c => c.charCodeAt(0));
            inputBuf = state.device.createBuffer({ size: Math.max(16, inputDataBytes.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
            if (inputDataBytes.byteLength > 0) new Uint8Array(inputBuf.getMappedRange()).set(inputDataBytes);
            inputBuf.unmap();
            bindGroupEntries.push({ binding: bindingIndex++, resource: { buffer: inputBuf } });
        }
        outBuf = state.device.createBuffer({ size: Math.max(16, meta.outputSize), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        bindGroupEntries.push({ binding: bindingIndex++, resource: { buffer: outBuf } });
        if (bindGroupEntries.length === 0 && meta.input ) throw new Error("No buffers for bind group, though input was expected."); // Adjusted error

        const bindGroup = state.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindGroupEntries });
        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...meta.workgroupCount);
        pass.end();
        readBuf = state.device.createBuffer({ size: outBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
        state.device.queue.submit([commandEncoder.finish()]);
        await state.device.queue.onSubmittedWorkDone(); // Recommended for ensuring work is complete before map
        await readBuf.mapAsync(GPUMapMode.READ);
        const resultDataMapped = readBuf.getMappedRange();
        const resultDataBytes = new Uint8Array(resultDataMapped.slice(0)); // Important: slice to copy before unmap
        const resultBase64 = btoa(String.fromCharCode(...resultDataBytes));
        readBuf.unmap();


        const processingTime = performance.now() - startTime;
        logTaskActivity(`Non-chunked workload "${meta.label}" completed in ${processingTime.toFixed(0)}ms. Output: ${resultDataBytes.length} bytes.`);
        socket.emit('workload:done', { id: meta.id, result: resultBase64, processingTime });

    } catch (err) {
        logTaskActivity(`Error processing non-chunked workload "${meta.label}": ${err.message}`, 'error');
        console.error(`Non-chunked workload error for ${meta.id}:`, err);
        socket.emit('workload:error', { id: meta.id, message: `Client-side error: ${err.message}` });
    } finally {
        if (inputBuf) inputBuf.destroy();
        if (outBuf) outBuf.destroy();
        if (readBuf) readBuf.destroy();
        state.isComputingWgsl = false;
        state.currentTask = null;
        updateComputationStatusDisplay();
        requestMatrixTaskIfNeeded();
    }
});

socket.on('workload:chunk_assign', async (chunkTask) => {
    if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
        logTaskActivity(`Received chunk ${chunkTask.chunkId} but client is busy. Emitting error for re-queue.`, 'warning');
        socket.emit('workload:chunk_error', { parentId: chunkTask.parentId, chunkId: chunkTask.chunkId, message: 'Client busy, cannot accept chunk instance.' });
        return;
    }
    if (!state.device) {
        logTaskActivity(`Received chunk ${chunkTask.chunkId} for ${chunkTask.parentId} but WebGPU device not ready. Emitting error.`, 'warning');
        socket.emit('workload:chunk_error', { parentId: chunkTask.parentId, chunkId: chunkTask.chunkId, message: 'WebGPU device not available on client.' });
        return;
    }

    logTaskActivity(`Processing chunk ${chunkTask.chunkId} (Parent: ${chunkTask.parentId.substring(0,6)}). Input bytes: ${chunkTask.chunkUniforms.chunkInputSizeBytes}`);
    state.isComputingChunk = true; // Set chunk busy flag
    state.currentTask = chunkTask; // Store chunk task metadata
    updateComputationStatusDisplay();

    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder, chunkUniformBuf;

    try {
        const startTime = performance.now();
        shader = state.device.createShaderModule({ code: chunkTask.wgsl });
        // ... (rest of chunk processing logic from your provided version)
        const compilationInfo = await shader.getCompilationInfo();
        if (compilationInfo.messages.some(m => m.type === 'error')) {
            const errorMessages = compilationInfo.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed for chunk ${chunkTask.chunkId}: ${errorMessages}`);
        }
        pipeline = state.device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: chunkTask.entry || 'main' }});
        const chunkInputDataBytes = Uint8Array.from(atob(chunkTask.inputData), c => c.charCodeAt(0));
        inputBuf = state.device.createBuffer({ size: Math.max(16, chunkInputDataBytes.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        if (chunkInputDataBytes.byteLength > 0) new Uint8Array(inputBuf.getMappedRange()).set(chunkInputDataBytes);
        inputBuf.unmap();

        const uniformValues = [chunkTask.chunkUniforms.chunkOffsetBytes, chunkTask.chunkUniforms.chunkInputSizeBytes, chunkTask.chunkUniforms.totalOriginalInputSizeBytes];
        if (chunkTask.chunkUniforms.hasOwnProperty('chunkOffsetElements')) {
             uniformValues.push(chunkTask.chunkUniforms.chunkOffsetElements, chunkTask.chunkUniforms.chunkInputSizeElements, chunkTask.chunkUniforms.totalOriginalInputSizeElements);
        }
        const chunkUniformData = new Uint32Array(uniformValues);
        chunkUniformBuf = state.device.createBuffer({ size: Math.max(16, chunkUniformData.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        state.device.queue.writeBuffer(chunkUniformBuf, 0, chunkUniformData);

        // --- Output Buffer Sizing for Chunks ---
        // This remains a tricky part. The server *should ideally* provide an expected output size per chunk,
        // or the shader design must be such that output size can be inferred from input chunk size.
        // For now, using a placeholder or assuming shader outputs a predictable size relative to input.
        // The server doesn't currently send `outputChunkSize`. Let's assume for now it's same as input (often wrong).
        let estimatedOutputChunkSize = chunkTask.chunkUniforms.chunkInputSizeBytes > 0 ? chunkTask.chunkUniforms.chunkInputSizeBytes : 16; // Min 16 bytes
        // A better heuristic might be needed if the server cannot specify outputChunkSize.
        // Example: If parent workload `outputSize` is known and `aggregationMethod` is 'concatenate',
        // and if output is proportional to input, one could try to estimate.
        // For now, this might lead to `outBuf` being too small or too large.

        outBuf = state.device.createBuffer({ size: Math.max(16, estimatedOutputChunkSize), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bindGroupEntries = [ { binding: 0, resource: { buffer: chunkUniformBuf } }, { binding: 1, resource: { buffer: inputBuf } }, { binding: 2, resource: { buffer: outBuf } }];
        const bindGroup = state.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindGroupEntries });
        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);

        // Dispatch calculation (remains heuristic based on input size or fixed workgroup size)
        // This needs to align with how the shader is written to process the chunk.
        let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
        const workgroupSizeX = 64; // Default/Example, should match shader's @workgroup_size(X,...)
        if (chunkTask.chunkUniforms.chunkInputSizeBytes > 0) {
            const elementSizeBytes = chunkTask.chunkUniforms.hasOwnProperty('chunkInputSizeElements') && chunkTask.chunkUniforms.chunkInputSizeElements > 0 ?
                                     (chunkTask.chunkUniforms.chunkInputSizeBytes / chunkTask.chunkUniforms.chunkInputSizeElements) : 4;
            const numElementsInChunk = chunkTask.chunkUniforms.chunkInputSizeBytes / Math.max(1, elementSizeBytes);
            dispatchX = Math.ceil(numElementsInChunk / workgroupSizeX);
        }
        dispatchX = Math.max(1, dispatchX);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ); // Use calculated or fixed dispatch
        pass.end();

        readBuf = state.device.createBuffer({ size: outBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
        state.device.queue.submit([commandEncoder.finish()]);
        await state.device.queue.onSubmittedWorkDone();
        await readBuf.mapAsync(GPUMapMode.READ);
        const resultDataMappedChunk = readBuf.getMappedRange();
        const resultDataBytes = new Uint8Array(resultDataMappedChunk.slice(0)); // Copy before unmap
        const resultBase64 = btoa(String.fromCharCode(...resultDataBytes));
        readBuf.unmap();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`Chunk ${chunkTask.chunkId} completed in ${processingTime.toFixed(0)}ms. Output: ${resultDataBytes.length} bytes (Est. output buf: ${estimatedOutputChunkSize}).`);
        socket.emit('workload:chunk_done', {
            parentId: chunkTask.parentId, chunkId: chunkTask.chunkId,
            chunkOrderIndex: chunkTask.chunkOrderIndex, result: resultBase64, processingTime
        });

    } catch (err) {
        logTaskActivity(`Error processing chunk ${chunkTask.chunkId} (Parent ${chunkTask.parentId.substring(0,6)}): ${err.message}`, 'error');
        console.error(`Chunk error for ${chunkTask.chunkId}:`, err);
        socket.emit('workload:chunk_error', {
            parentId: chunkTask.parentId, chunkId: chunkTask.chunkId,
            message: `Client-side error processing chunk: ${err.message}`
        });
    } finally {
        if (inputBuf) inputBuf.destroy();
        if (outBuf) outBuf.destroy();
        if (readBuf) readBuf.destroy();
        if (chunkUniformBuf) chunkUniformBuf.destroy();
        state.isComputingChunk = false;
        state.currentTask = null;
        updateComputationStatusDisplay();
        requestMatrixTaskIfNeeded();
    }
});


function requestMatrixTaskIfNeeded() {
    if (state.connected && elements.joinComputation && !elements.joinComputation.disabled &&
        !state.isComputingMatrix && !state.isComputingWgsl && !state.isComputingChunk && !state.currentTask) {
        logTaskActivity("Client idle, checking for matrix task.", "debug");
        requestMatrixTask();
    }
}

function updateActiveWgslWorkloadsDisplay(workloadsFromServer) {
    // ... (updateActiveWgslWorkloadsDisplay - unchanged from your provided version)
    if (IS_HEADLESS || !elements.activeWgslWorkloadsGrid) return;
    elements.activeWgslWorkloadsGrid.innerHTML = '';
    if (!workloadsFromServer || workloadsFromServer.length === 0) {
        elements.activeWgslWorkloadsGrid.innerHTML = '<p>No custom WGSL workloads currently.</p>';
        return;
    }
    workloadsFromServer.forEach(wl => {
        const card = document.createElement('div');
        // Status class for card might need more nuance based on new statuses like 'pending_dispatch'
        card.className = `wgsl-card status-${wl.status?.replace('_', '-') || 'unknown'}`;
        card.id = `wgsl-card-${wl.id}`;

        let chunkProgressHTML = '';
        if (wl.isChunkParent && wl.chunkProgress) { // Assuming server adds a 'chunkProgress' field
            chunkProgressHTML = `<p>Chunks: ${wl.chunkProgress.completed}/${wl.chunkProgress.total} verified (${((wl.chunkProgress.completed / wl.chunkProgress.total) * 100 || 0).toFixed(0)}%)</p>`;
        }


        let cardHTML = `
            <h4>${wl.label} <small>(${wl.id.substring(0,6)})${wl.isChunkParent ? ' (Chunked)' : ''}</small></h4>
            <p>Status: <strong>${wl.status || 'N/A'}</strong> ${wl.dispatchesMade !== undefined ? `(D:${wl.dispatchesMade})` : ''}</p>
            ${chunkProgressHTML}
            <p>Dispatch: ${wl.workgroupCount.join('x')} groups ${wl.isChunkParent ? '(per parent)' : ''}</p>
            <p>Output: ${wl.outputSize} bytes ${wl.isChunkParent ? '(final aggregated)' : ''}</p>
            ${wl.wgsl ? `<p>Shader: <pre>${wl.wgsl.substring(0,100)}${wl.wgsl.length > 100 ? '...' : ''}</pre></p>` : ''}
            ${wl.results && !wl.isChunkParent ? `<p>Submissions (non-chunked): ${wl.results.length}</p>` : ''}
            ${wl.finalResult ? `<p>Final Result (first 10 bytes): <pre>${JSON.stringify(wl.finalResult.slice(0,10))}...</pre></p>` : ''}
            ${wl.error ? `<p class="status error">Error: ${wl.error}</p>` : ''}
            ${wl.createdAt ? `<p><small>Created: ${new Date(wl.createdAt).toLocaleString()}</small></p>` : ''}
            ${wl.startedAt && wl.status !== 'queued' ? `<p><small>Started: ${new Date(wl.startedAt).toLocaleString()}</small></p>` : ''}`;

        if (wl.status === 'complete' && wl.completedAt) {
             // ... (completion time calculation unchanged)
        }

        cardHTML += `<div class="wgsl-card-actions">`;
        cardHTML += `<button class="remove-wgsl-button danger" data-workload-id="${wl.id}" title="Remove Workload">Remove (X)</button>`;
        cardHTML += `</div>`;

        card.innerHTML = cardHTML;
        elements.activeWgslWorkloadsGrid.appendChild(card);

        const removeButton = card.querySelector('.remove-wgsl-button');
        if (removeButton) {
            removeButton.addEventListener('click', function() {
                const workloadId = this.getAttribute('data-workload-id');
                if (confirm(`Are you sure you want to remove workload "${wl.label}" (ID: ${workloadId.substring(0,6)})? This action cannot be undone.`)) {
                    logAdminActivity(`Requesting removal of WGSL workload ID: ${workloadId}`, 'wgsl');
                    socket.emit('admin:removeCustomWorkload', { workloadId });
                }
            });
        }
    });
}


// Socket.io event handlers
socket.on('connect', () => {
    state.connected = true;
    elements.clientStatus.textContent = 'Connected to server';
    elements.clientStatus.className = 'status success';
    logTaskActivity('Connected to computation server');
    if (elements.joinComputation) elements.joinComputation.disabled = !(state.webgpuSupported || window.isSecureContext);
    updateComputationStatusDisplay();
});

socket.on('disconnect', () => {
    state.connected = false;
    state.isComputingMatrix = false; state.isComputingWgsl = false; state.isComputingChunk = false;
    state.currentTask = null;
    elements.clientStatus.textContent = 'Disconnected from server';
    elements.clientStatus.className = 'status error';
    logTaskActivity('Disconnected from server', 'error');
    if (elements.joinComputation) elements.joinComputation.disabled = true;
    if (elements.leaveComputation) elements.leaveComputation.disabled = true;
    updateComputationStatusDisplay();
});

socket.on('register', (data) => {
    state.clientId = data.clientId;
    elements.clientStatus.textContent = `Connected as: ${data.clientId.substring(0, 8)}... ${IS_HEADLESS ? '(Headless Worker '+WORKER_ID+')' : ''}`;
});

socket.on('state:update', (data) => { // For matrix computation global state
    if (IS_HEADLESS) return;
    updateStatsDisplay(data.stats);
    updateComputationStatusDisplay(); // General status might change
});

socket.on('clients:update', (data) => { updateClientDisplay(data.clients); });

socket.on('task:assign', async (task) => { // Matrix task assignment
    if (!state.connected || (elements.joinComputation && elements.joinComputation.disabled && !IS_HEADLESS)) return;

    if (state.isComputingMatrix || state.isComputingWgsl || state.isComputingChunk) {
         logTaskActivity(`Received matrix task ${task.id} (Instance ${task.assignmentId.substring(0,6)}) but client is busy. Server should re-assign.`, 'warning');
         // Server will handle timeout or re-assignment if this client doesn't pick it up.
         return;
    }

    if (task.type && task.type !== 'matrixMultiply') {
        logTaskActivity(`Received task of unknown type: ${task.type}. Ignoring.`, 'warning');
        return;
    }

    try {
        const resultPayload = await processMatrixTask(task); // task includes assignmentId and id
        socket.emit('task:complete', {
            assignmentId: task.assignmentId, // Important: send back the specific assignmentId
            taskId: task.id,                 // The logical task ID
            ...resultPayload                 // Contains .result and .processingTime
        });
    } catch (error) {
        logTaskActivity(`Error during matrix task processing (Instance ${task.assignmentId?.substring(0,6)}): ${error.message}`, 'error');
        // processMatrixTask's finally block handles state reset
    } finally {
        requestMatrixTaskIfNeeded();
    }
});

socket.on('task:wait', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    if (!state.isComputingMatrix && !state.isComputingWgsl && !state.isComputingChunk) {
        elements.taskStatus.textContent = 'Waiting for available matrix tasks';
        elements.taskStatus.className = 'status warning';
        logTaskActivity('No matrix tasks available from server, waiting...');
        if (state.connected && elements.joinComputation && !elements.joinComputation.disabled) {
            setTimeout(requestMatrixTaskIfNeeded, 7000 + Math.random() * 3000);
        }
    }
});

socket.on('computation:complete', (data) => { // Matrix computation finished
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Matrix computation completed in ${data.totalTime.toFixed(1)}s`);
    updateComputationStatusDisplay();
});

socket.on('task:error', (data) => { // Server error for a matrix task client was working on
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Server error for matrix task ${data.taskId} (Instance ${data.assignmentId?.substring(0,6)}): ${data.message}`, 'error');
    if(state.currentTask && state.currentTask.assignmentId === data.assignmentId) { // Check specific instance
        state.isComputingMatrix = false;
        state.currentTask = null;
        updateComputationStatusDisplay();
        requestMatrixTaskIfNeeded();
    }
});

socket.on('task:submitted', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Matrix task ${data.taskId} submitted. Awaiting verification.`);
});
socket.on('task:verified', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Your submission for matrix task ${data.taskId} was verified!`, 'success');
});

socket.on('workloads:list_update', (allWorkloads) => {
    if (IS_HEADLESS) return;
    updateActiveWgslWorkloadsDisplay(allWorkloads);
});

socket.on('workload:complete', (data) => {
    logTaskActivity(`Server confirmed custom workload "${data.label || data.id.substring(0,6)}" is complete!`, 'success');
});

socket.on('workload:parent_started', (data) => {
    logTaskActivity(`Parent workload "${data.label || data.id.substring(0,6)}" (ID: ${data.id.substring(0,6)}) started chunk processing (Status: ${data.status}).`, 'info');
});

socket.on('workload:removed', (data) => {
    // ... (UI update for removed workload - unchanged)
});

socket.on('admin:feedback', (data) => {
    if (data && data.message) {
        logAdminActivity(`Server: ${data.message}`, data.panelType || 'wgsl', data.success ? 'success' : 'error');
    }
});

// --- K Parameter Admin UI Handling ---
socket.on('admin:k_update', (newK) => {
    if (elements.adminKValueInput) elements.adminKValueInput.value = newK;
    if (elements.currentKDisplay) elements.currentKDisplay.textContent = newK;
    if (!IS_HEADLESS) logAdminActivity(`Redundancy factor K updated by server to: ${newK}`, 'system', 'info');
});


// Event listeners for UI elements
if (!IS_HEADLESS) {
    elements.joinComputation.addEventListener('click', joinComputation);
    elements.leaveComputation.addEventListener('click', leaveComputation);
    elements.startComputation.addEventListener('click', startMatrixComputation); // Admin
    elements.toggleAdmin.addEventListener('click', () => { // Admin
        elements.adminPanel.style.display = (elements.adminPanel.style.display === 'block') ? 'none' : 'block';
    });

    // K Parameter Button
    if (elements.setKButton && elements.adminKValueInput) {
        elements.setKButton.addEventListener('click', () => {
            const newK = parseInt(elements.adminKValueInput.value);
            if (!isNaN(newK) && newK >= 1) {
                socket.emit('admin:set_k_parameter', newK);
            } else {
                logAdminActivity('Invalid K value entered. Must be integer >= 1.', 'system', 'error');
            }
        });
    }

    elements.pushWgslWorkloadButton.addEventListener('click', async () => { // Admin
        // ... (WGSL push logic - unchanged from your provided version)
        const isChunkable = document.getElementById('wgsl-chunkable') ? document.getElementById('wgsl-chunkable').checked : false;
        const inputChunkProcessingType = document.getElementById('wgsl-chunk-processing-type') ? document.getElementById('wgsl-chunk-processing-type').value : 'elements';
        const inputChunkSize = document.getElementById('wgsl-input-chunk-size') ? parseInt(document.getElementById('wgsl-input-chunk-size').value) : 0;
        const inputElementSizeBytes = document.getElementById('wgsl-input-element-size-bytes') ? parseInt(document.getElementById('wgsl-input-element-size-bytes').value) : 4;
        const outputAggregationMethod = document.getElementById('wgsl-output-aggregation-method') ? document.getElementById('wgsl-output-aggregation-method').value : 'concatenate';

        const payload = {
            label: elements.wgslLabel.value || 'Untitled WGSL Workload',
            wgsl: elements.wgslSrc.value,
            entry: elements.wgslEntryPoint.value || 'main',
            workgroupCount: [ Math.max(1, +elements.wgslGroupsX.value || 1), Math.max(1, +elements.wgslGroupsY.value || 1), Math.max(1, +elements.wgslGroupsZ.value || 1)],
            bindLayout: elements.wgslBindLayout.value,
            outputSize: +elements.wgslOutputSize.value,
            input: elements.wgslInputData.value.trim() || undefined,
            chunkable: isChunkable,
            inputChunkProcessingType: isChunkable ? inputChunkProcessingType : undefined,
            inputChunkSize: isChunkable ? inputChunkSize : undefined,
            inputElementSizeBytes: isChunkable && inputChunkProcessingType === 'elements' ? inputElementSizeBytes : undefined,
            outputAggregationMethod: isChunkable ? outputAggregationMethod : undefined,
        };
        if (!payload.wgsl || !payload.outputSize) { logAdminActivity('WGSL Upload: Missing WGSL source or output size.', 'wgsl', 'error'); return; }
        if (isChunkable) {
            if (!payload.input) { logAdminActivity('WGSL Upload: Input data is required for chunkable workloads.', 'wgsl', 'error'); return; }
            if (!payload.inputChunkSize || payload.inputChunkSize <= 0) { logAdminActivity('WGSL Upload: Invalid inputChunkSize for chunkable workload.', 'wgsl', 'error'); return; }
            if (payload.inputChunkProcessingType === 'elements' && (!payload.inputElementSizeBytes || payload.inputElementSizeBytes <= 0)) { logAdminActivity('WGSL Upload: Invalid inputElementSizeBytes for chunkable workload with element processing type.', 'wgsl', 'error'); return;}
        }
        logAdminActivity(`Pushing WGSL workload: "${payload.label}"...`, 'wgsl');
        try {
            const res = await fetch('/api/workloads', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const json = await res.json();
            if (res.ok && json.ok) { logAdminActivity(`WGSL Workload ID ${json.id.substring(0,6)} (${payload.label}) ${json.message || 'submitted successfully.'}`, 'wgsl', 'success');}
            else { logAdminActivity(`WGSL Upload Error: ${json.error || 'Unknown server error'} (Status: ${res.status})`, 'wgsl', 'error');}
        } catch (err) { logAdminActivity(`WGSL Upload Fetch Error: ${err.message}`, 'wgsl', 'error'); console.error("WGSL Push error:", err); }
    });

    if (elements.startQueuedWgslButton) { // Admin
        elements.startQueuedWgslButton.addEventListener('click', () => {
            logAdminActivity('Requesting server to start all queued WGSL workloads...', 'wgsl');
            socket.emit('admin:startQueuedCustomWorkloads');
        });
    }
}

// Initialize the application
async function init() {
    if (IS_HEADLESS) {
        document.documentElement.style.display = 'none'; // Hide UI for headless
        console.log(`Puppeteer worker #${WORKER_ID} initializing. UI hidden.`);
        document.title = `Headless Worker ${WORKER_ID} - Initializing`;
    }
    await initWebGPU();
    updateComputationStatusDisplay();

    if (IS_HEADLESS) {
        if (elements.joinComputation && !elements.joinComputation.disabled) {
            logTaskActivity("Headless mode: Auto-joining computation...");
            joinComputation();
        } else {
             logTaskActivity("Headless mode: Cannot auto-join. WebGPU init might have failed or join disabled.", 'warning');
        }
    } else {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('admin')) {
            if(elements.adminPanel) elements.adminPanel.style.display = 'block';
        }
        // Ensure K parameter UI fields are defined if admin panel is visible
        if(elements.adminPanel.style.display === 'block') {
            if(!elements.adminKValueInput) elements.adminKValueInput = document.getElementById('admin-k-value');
            if(!elements.setKButton) elements.setKButton = document.getElementById('set-k-button');
            if(!elements.currentKDisplay) elements.currentKDisplay = document.getElementById('current-k-display');
            if(!elements.adminLogSystem) elements.adminLogSystem = document.getElementById('admin-log-system');
        }
    }
}

init();