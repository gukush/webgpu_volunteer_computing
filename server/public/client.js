// client.js

// State management (no changes here)
const state = {
    webgpuSupported: false, device: null, adapter: null, adapterInfo: null,
    connected: false, clientId: null,
    isComputingMatrix: false, // More specific flag for matrix tasks
    isComputingWgsl: false,   // More specific for non-chunked WGSL
    isComputingChunk: false,  // More specific for WGSL chunks
    currentTask: null, // For matrix task
    completedTasks: 0, statistics: { processingTime: 0 }
};


// elements object (no changes needed here)
const elements = {
    webgpuStatus: document.getElementById('webgpu-status'),
    gpuInfo: document.getElementById('gpu-info'),
    computationStatus: document.getElementById('computation-status'),
    clientStatus: document.getElementById('client-status'),
    taskStatus: document.getElementById('task-status'), // Used for matrix tasks
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
    wgslBindLayout: document.getElementById('wgsl-bind-layout'), // Could be used for more complex bind group setup
    wgslOutputSize: document.getElementById('wgsl-output-size'),
    wgslInputData: document.getElementById('wgsl-input-data'),
    // Chunking related UI fields could be added here if client needs to submit them
    // For now, server derives chunking from API params
    pushWgslWorkloadButton: document.getElementById('push-wgsl-workload'),
    activeWgslWorkloadsGrid: document.getElementById('active-wgsl-workloads-grid'),
    startQueuedWgslButton: document.getElementById('startQueuedWgslButton')
};

const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';
const socket = io({ query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {} });

async function initWebGPU() {
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

        const discreteVendors = ['nvidia', 'amd', 'advanced micro devices', 'apple', 'qualcomm', 'arm']; // Added more common vendors
        let foundAdapterEntry = adaptersWithInfo.find(a =>
            discreteVendors.some(v =>
                (a.info.vendor?.toLowerCase().includes(v) ||
                 a.info.description?.toLowerCase().includes(v)) &&
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver') &&
                !a.info.description?.toLowerCase().includes('llvmpipe') // Exclude software rasterizers
            )
        );

        if (!foundAdapterEntry && adaptersWithInfo.length > 0) {
            foundAdapterEntry = adaptersWithInfo.find(a => // Fallback to first non-software
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver') &&
                !a.info.description?.toLowerCase().includes('llvmpipe')
            ) || adaptersWithInfo[0]; // Absolute fallback to first adapter
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
    // ... (multiplyMatricesGPU - unchanged from original)
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
        const resultArray = new Float32Array(readBuffer.getMappedRange()); // getMappedRange is ArrayBuffer
        const resultMappedRange = readBuffer.getMappedRange();
        const result = new Array(rowsToCalculate);
        for (let i = 0; i < rowsToCalculate; i++) { result[i] = new Array(size); for (let j = 0; j < size; j++) result[i][j] = resultArray[i * size + j]; }
        readBuffer.unmap();
        aBuffer.destroy(); bBuffer.destroy(); resultBuffer.destroy(); readBuffer.destroy(); uniformBuffer.destroy();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`GPU: Matrix task completed in ${processingTime.toFixed(0)}ms`);
        return { result, processingTime };
    } catch (error) { logTaskActivity(`GPU: Matrix task ERROR: ${error.message}`, 'error'); throw error; }
}

async function multiplyMatricesCPU(matrixA, matrixB, size, startRow, endRow) {
    // ... (multiplyMatricesCPU - unchanged from original)
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

async function processMatrixTask(task) {
    // ... (processMatrixTask - unchanged from original, ensure it sets state.isComputingMatrix)
    state.currentTask = task;
    state.isComputingMatrix = true; // Set matrix specific busy flag
    elements.taskStatus.textContent = `Matrix task ${task.id} (rows ${task.startRow}-${task.endRow})`;
    elements.taskStatus.className = 'status info';
    elements.computationStatus.textContent = `Processing Matrix Task ${task.id.substring(0,6)}`;
    elements.computationStatus.className = 'status info';

    try {
        let resultData;
        if (state.webgpuSupported && state.device) {
            resultData = await multiplyMatricesGPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        } else {
            resultData = await multiplyMatricesCPU(task.matrixA, task.matrixB, task.size, task.startRow, task.endRow);
        }
        state.completedTasks++; state.statistics.processingTime += resultData.processingTime;
        elements.myTasks.textContent = state.completedTasks;
        elements.processingTime.textContent = `${resultData.processingTime.toFixed(0)}ms`;
        elements.taskStatus.textContent = `Matrix task ${task.id} completed`; elements.taskStatus.className = 'status success';
        return { taskId: task.id, result: resultData.result, processingTime: resultData.processingTime };
    } catch (error) {
        elements.taskStatus.textContent = `Matrix task error: ${error.message}`; elements.taskStatus.className = 'status error';
        socket.emit('task:error', { taskId: task.id, message: error.message, type: 'matrixMultiply' });
        throw error;
    } finally {
        state.isComputingMatrix = false; // Clear matrix specific busy flag
        state.currentTask = null;
        updateComputationStatusDisplay();
    }
}

function joinComputation() {
    // ... (joinComputation - unchanged from original)
    if(elements.joinComputation) elements.joinComputation.disabled = true;
    if(elements.leaveComputation) elements.leaveComputation.disabled = false;
    const mode = (state.webgpuSupported && state.device) ? 'WebGPU' : 'CPU';
    logTaskActivity(`Joining computation network (${mode})...`);
    socket.emit('client:join', { gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback', device: 'CPU Computation', description: 'No adapter info' } });
    // state.isComputing = true; // General flag, specific flags used per task type
    updateComputationStatusDisplay(); // Update based on current activities
}

function leaveComputation() {
    // ... (leaveComputation - unchanged from original)
    if(elements.joinComputation) elements.joinComputation.disabled = false;
    if(elements.leaveComputation) elements.leaveComputation.disabled = true;
    state.isComputingMatrix = false;
    state.isComputingWgsl = false;
    state.isComputingChunk = false;
    state.currentTask = null;
    logTaskActivity('Left computation network');
    elements.taskStatus.textContent = 'No matrix task assigned'; elements.taskStatus.className = 'status info';
    updateComputationStatusDisplay();
    socket.emit('client:leave');
}

function requestMatrixTask() {
    // ... (requestMatrixTask - unchanged from original, ensure it checks specific busy flags)
    if (state.connected && !state.isComputingMatrix && !state.isComputingWgsl && !state.isComputingChunk && !state.currentTask) {
        socket.emit('task:request');
    }
}

function startMatrixComputation() {
    // ... (startMatrixComputation - unchanged from original)
    const matrixSize = parseInt(elements.matrixSize.value);
    const chunkSize = parseInt(elements.chunkSize.value);
    socket.emit('admin:start', { matrixSize, chunkSize });
    logAdminActivity(`Starting new matrix computation: ${matrixSize}x${matrixSize} with chunk size ${chunkSize}`, 'matrix');
}

function logTaskActivity(message, type = 'info') {
    // ... (logTaskActivity - unchanged from original)
    if (IS_HEADLESS) {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp} Worker ${WORKER_ID} Log]: ${message}`);
        if (type === 'error') console.error(`[${timestamp} Worker ${WORKER_ID} ErrorLog]: ${message}`);
        document.title = `Worker ${WORKER_ID} | ${message.substring(0,50)}`;
        return;
    }
    const logItem = document.createElement('div'); logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.taskLog.appendChild(logItem); elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

function logAdminActivity(message, panelType = 'matrix', type = 'info') {
    // ... (logAdminActivity - unchanged from original)
    if (IS_HEADLESS) { console.log(`[Admin Log]: ${message}`); return; }
    const logContainer = panelType === 'wgsl' ? elements.adminLogWgsl : elements.adminLogMatrix;
    const logItem = document.createElement('div'); logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    if(logContainer) {
        logContainer.appendChild(logItem);
        logContainer.scrollTop = logContainer.scrollHeight;
    } else {
        console.warn(`Admin log container for ${panelType} not found.`);
    }
}

function updateClientDisplay(clients) {
    // ... (updateClientDisplay - ensure it can show `isBusyWithCustomChunk` if desired)
    if (IS_HEADLESS || !elements.clientGrid) return;
    elements.clientGrid.innerHTML = '';
    if (!clients || clients.length === 0) { elements.clientGrid.innerHTML = '<div>No clients connected</div>'; return; }
    clients.forEach(client => {
        const isCurrentClient = client.id === state.clientId;
        const timeSinceActive = Date.now() - client.lastActive;
        const isActive = client.connected && timeSinceActive < 60000; // 60s threshold
        const clientEl = document.createElement('div');
        clientEl.className = `client-card ${!isActive ? 'client-inactive' : ''} ${client.usingCpu ? 'client-cpu' : 'client-gpu'} ${client.isPuppeteer ? 'client-puppeteer' : ''}`;
        let clientHTML = `<div>${isCurrentClient ? '<strong>You</strong>' : 'Client'} ${client.isPuppeteer ? '(Puppeteer)' : ''}</div>
                          <div><small>${client.id.substring(0, 8)}...</small></div>`;
        if (client.gpuInfo) {
            let displayVendor = 'GPU';
            if (client.gpuInfo.isCpuComputation) {
                displayVendor = 'CPU';
            } else if (client.gpuInfo.vendor && client.gpuInfo.vendor !== 'N/A' && client.gpuInfo.vendor !== 'Unknown') {
                displayVendor = client.gpuInfo.vendor.split(' ')[0]; // First word of vendor
            }
            clientHTML += `<div><small>${displayVendor}</small></div>`;
        }
        clientHTML += `<div>Tasks: ${client.completedTasks || 0}</div>`;
        if (client.isBusyWithCustomChunk) { // Display if busy with a custom chunk
            clientHTML += `<div><small class="status info">Busy (Chunk)</small></div>`;
        }
        clientHTML += `<div><small>${isActive ? 'Active' : 'Inactive ('+ Math.round(timeSinceActive/1000) + 's ago)'}</small></div>`;
        clientEl.innerHTML = clientHTML;
        elements.clientGrid.appendChild(clientEl);
    });
}

function updateStatsDisplay(stats) {
    // ... (updateStatsDisplay - unchanged from original)
    if (IS_HEADLESS || !elements.activeClients) return;
    elements.activeClients.textContent = stats.activeClients || 0;
    elements.totalTasks.textContent = stats.totalTasks || 0; // This is for matrix tasks
    elements.completedTasks.textContent = stats.completedTasks || 0; // This is for matrix tasks
    if (stats.elapsedTime) { elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`; }
}

function updateComputationStatusDisplay() {
    if (IS_HEADLESS || !elements.computationStatus) return;

    if (state.isComputingChunk) {
        elements.computationStatus.textContent = `Processing WGSL Chunk...`;
        elements.computationStatus.className = 'status info';
    } else if (state.isComputingWgsl) {
        elements.computationStatus.textContent = `Processing WGSL Workload...`;
        elements.computationStatus.className = 'status info';
    } else if (state.isComputingMatrix) {
        elements.computationStatus.textContent = `Processing Matrix Task...`;
        elements.computationStatus.className = 'status info';
    } else if (state.connected && elements.joinComputation && elements.joinComputation.disabled) { // Joined but idle
        elements.computationStatus.textContent = 'Idle, awaiting task.';
        elements.computationStatus.className = 'status info';
    } else if (state.connected) {
        elements.computationStatus.textContent = 'Connected, not joined computation.';
        elements.computationStatus.className = 'status warning';
    } else {
        elements.computationStatus.textContent = 'Disconnected.';
        elements.computationStatus.className = 'status error';
    }
}


// --- Custom WGSL Workload Handling ---
socket.on('workload:new', async meta => { // For NON-CHUNKED workloads
    if (meta.isChunkParent) {
        logTaskActivity(`Received parent workload "${meta.label}" - chunks will be assigned separately.`, 'info');
        return;
    }
    if (state.isComputingChunk || state.isComputingMatrix || state.isComputingWgsl) {
        logTaskActivity(`Received non-chunked workload "${meta.label}" but client is busy. Server should re-assign if needed.`, 'warning');
        // Optionally, tell server we are busy if this is a direct assignment model (not typical for 'new' broadcast)
        return;
    }

    if (meta.status !== 'pending') {
        logTaskActivity(`Received non-chunked workload "${meta.label}" with status "${meta.status}". Will not process unless 'pending'.`, meta.status === 'queued' ? 'info' : 'warning');
        return;
    }
    if (!state.device) {
        logTaskActivity(`Received non-chunked workload "${meta.label}" but WebGPU device not ready. Skipping.`, 'warning');
        socket.emit('workload:error', { id: meta.id, message: 'WebGPU device not available on client.' });
        return;
    }

    logTaskActivity(`Processing non-chunked custom workload "${meta.label}" (ID: ${meta.id.substring(0,6)})...`);
    state.isComputingWgsl = true;
    updateComputationStatusDisplay();

    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder;

    try {
        const startTime = performance.now();
        shader = state.device.createShaderModule({ code: meta.wgsl });
        const compilationInfo = await shader.getCompilationInfo();
        if (compilationInfo.messages.some(m => m.type === 'error')) {
            const errorMessages = compilationInfo.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed: ${errorMessages}`);
        }

        // Simplified bind group for non-chunked: input at 0 (if any), output at next available.
        // This requires shader to be written accordingly.
        pipeline = state.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shader, entryPoint: meta.entry || 'main' }
        });

        const bindGroupEntries = [];
        let bindingIndex = 0;

        if (meta.input) {
            const inputDataBytes = Uint8Array.from(atob(meta.input), c => c.charCodeAt(0));
            inputBuf = state.device.createBuffer({
                size: Math.max(16, inputDataBytes.byteLength), // Ensure min buffer size
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            if (inputDataBytes.byteLength > 0) new Uint8Array(inputBuf.getMappedRange()).set(inputDataBytes);
            inputBuf.unmap();
            bindGroupEntries.push({ binding: bindingIndex++, resource: { buffer: inputBuf } });
        }

        outBuf = state.device.createBuffer({
            size: Math.max(16, meta.outputSize), // Ensure min buffer size
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        bindGroupEntries.push({ binding: bindingIndex++, resource: { buffer: outBuf } });

        if (bindGroupEntries.length === 0) throw new Error("No buffers for bind group.");

        const bindGroup = state.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bindGroupEntries
        });

        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...meta.workgroupCount); // Uses parent's full workgroup count
        pass.end();

        readBuf = state.device.createBuffer({
            size: outBuf.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
        state.device.queue.submit([commandEncoder.finish()]);

        await state.device.queue.onSubmittedWorkDone();
        await readBuf.mapAsync(GPUMapMode.READ);

        const resultDataBytes = new Uint8Array(readBuf.getMappedRange().slice(0));
        const resultBase64 = btoa(String.fromCharCode(...resultDataBytes));
        readBuf.unmap();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`Non-chunked workload "${meta.label}" completed in ${processingTime.toFixed(0)}ms. Output: ${resultDataBytes.length} bytes.`);
        socket.emit('workload:done', { id: meta.id, result: resultBase64, processingTime });

    } catch (err) {
        logTaskActivity(`Error processing non-chunked workload "${meta.label}": ${err.message}`, 'error');
        console.error(`Non-chunked workload error for ${meta.id}:`, err);
        socket.emit('workload:error', { id: meta.id, message: err.message });
    } finally {
        if (inputBuf) inputBuf.destroy();
        if (outBuf) outBuf.destroy();
        if (readBuf) readBuf.destroy();
        state.isComputingWgsl = false;
        updateComputationStatusDisplay();
        requestMatrixTaskIfNeeded(); // Check if we can take a matrix task
    }
});

// NEW handler for assigned chunks
socket.on('workload:chunk_assign', async (chunkTask) => {
    if (state.isComputingChunk || state.isComputingMatrix || state.isComputingWgsl) {
        logTaskActivity(`Received chunk ${chunkTask.chunkId} but client is busy. Emitting error to server for re-queue.`, 'warning');
        socket.emit('workload:chunk_error', { parentId: chunkTask.parentId, chunkId: chunkTask.chunkId, message: 'Client busy, cannot accept chunk.' });
        return;
    }
    if (!state.device) {
        logTaskActivity(`Received chunk ${chunkTask.chunkId} for ${chunkTask.parentId} but WebGPU device not ready. Emitting error.`, 'warning');
        socket.emit('workload:chunk_error', { parentId: chunkTask.parentId, chunkId: chunkTask.chunkId, message: 'WebGPU device not available on client.' });
        return;
    }

    logTaskActivity(`Processing chunk ${chunkTask.chunkId} (Parent: ${chunkTask.parentId}). Input bytes: ${chunkTask.chunkUniforms.chunkInputSizeBytes}`);
    state.isComputingChunk = true;
    updateComputationStatusDisplay();

    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder, chunkUniformBuf;

    try {
        const startTime = performance.now();
        shader = state.device.createShaderModule({ code: chunkTask.wgsl });
        const compilationInfo = await shader.getCompilationInfo();
        if (compilationInfo.messages.some(m => m.type === 'error')) {
            const errorMessages = compilationInfo.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed for chunk ${chunkTask.chunkId}: ${errorMessages}`);
        }

        pipeline = state.device.createComputePipeline({
            layout: 'auto', // Assumes WGSL uses @group(0) @binding(0..2) as conventional
            compute: { module: shader, entryPoint: chunkTask.entry || 'main' }
        });

        const chunkInputDataBytes = Uint8Array.from(atob(chunkTask.inputData), c => c.charCodeAt(0));
        inputBuf = state.device.createBuffer({
            size: Math.max(16, chunkInputDataBytes.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        if (chunkInputDataBytes.byteLength > 0) new Uint8Array(inputBuf.getMappedRange()).set(chunkInputDataBytes);
        inputBuf.unmap();

        const uniformValues = [
            chunkTask.chunkUniforms.chunkOffsetBytes,
            chunkTask.chunkUniforms.chunkInputSizeBytes,
            chunkTask.chunkUniforms.totalOriginalInputSizeBytes
        ];
        if (chunkTask.chunkUniforms.hasOwnProperty('chunkOffsetElements')) { // Check if element-based uniforms are provided
             uniformValues.push(chunkTask.chunkUniforms.chunkOffsetElements);
             uniformValues.push(chunkTask.chunkUniforms.chunkInputSizeElements);
             uniformValues.push(chunkTask.chunkUniforms.totalOriginalInputSizeElements);
        }
        // Ensure enough space for uniform struct (e.g. if shader expects 6 u32s, but only 3 sent)
        // This relies on server sending all fields the shader expects, or shader being robust.
        // For now, assume server sends what's needed based on `inputChunkProcessingType`.
        const chunkUniformData = new Uint32Array(uniformValues);

        chunkUniformBuf = state.device.createBuffer({
            size: Math.max(16, chunkUniformData.byteLength),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        state.device.queue.writeBuffer(chunkUniformBuf, 0, chunkUniformData);

        // Output buffer estimation - CRITICAL POINT for generic solution
        // Using placeholder from sketch: output size = input size (often incorrect)
        let estimatedOutputChunkSize = chunkTask.chunkUniforms.chunkInputSizeBytes;
        if (chunkTask.chunkUniforms.chunkInputSizeBytes === 0 && chunkTask.chunkUniforms.totalOriginalInputSizeBytes > 0) {
             logTaskActivity(`Chunk ${chunkTask.chunkId} has 0 input size (likely padding or final small chunk of elements). Estimating minimal output size (16 bytes). Shader must handle this.`, "warning");
             estimatedOutputChunkSize = 16; // Minimum valid size
        } else if (chunkTask.chunkUniforms.chunkInputSizeBytes === 0 && chunkTask.chunkUniforms.totalOriginalInputSizeBytes === 0) {
            logTaskActivity(`Chunk ${chunkTask.chunkId} has 0 input size (original input was empty). Estimating minimal output (16 bytes).`, "warning");
            estimatedOutputChunkSize = 16;
        }
        // TODO: A robust solution needs server to hint outputChunkSize or shader to declare it based on input.
        // Example: if shader processes f32 (4 bytes) and outputs one f32 per input f32
        // estimatedOutputChunkSize = chunkTask.chunkUniforms.chunkInputSizeBytes;
        // If it outputs one u8 per f32 input:
        // estimatedOutputChunkSize = chunkTask.chunkUniforms.chunkInputSizeBytes / 4;

        outBuf = state.device.createBuffer({
            size: Math.max(16, estimatedOutputChunkSize),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const bindGroupEntries = [
            { binding: 0, resource: { buffer: chunkUniformBuf } },
            { binding: 1, resource: { buffer: inputBuf } },
            { binding: 2, resource: { buffer: outBuf } },
        ];
        const bindGroup = state.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bindGroupEntries
        });

        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);

        // Dispatch calculation
        let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
        const workgroupSizeX = 64; // MUST MATCH SHADER'S @workgroup_size(X, Y, Z) X component
        const workgroupSizeY = 1;  // MUST MATCH SHADER'S Y component
        const workgroupSizeZ = 1;  // MUST MATCH SHADER'S Z component
                                   // These should ideally be part of chunkTask or a convention.

        if (chunkTask.chunkUniforms.chunkInputSizeBytes > 0) {
            const elementSizeBytes = chunkTask.chunkUniforms.hasOwnProperty('chunkInputSizeElements') && chunkTask.chunkUniforms.chunkInputSizeElements > 0 ?
                                     (chunkTask.chunkUniforms.chunkInputSizeBytes / chunkTask.chunkUniforms.chunkInputSizeElements) : 4; // Default f32
            const numElementsInChunk = chunkTask.chunkUniforms.chunkInputSizeBytes / Math.max(1, elementSizeBytes);
            dispatchX = Math.ceil(numElementsInChunk / workgroupSizeX);
            // Assuming 1D data processing for dispatch calculation based on X.
            // If shader uses 2D/3D workgroups for 1D data, this needs adjustment.
            // The parent's workgroupCount (chunkTask.workgroupCount) might be [origX, origY, origZ]
            // If shader truly processes a 2D/3D slice, this dispatch logic needs to be more complex.
            // For now, following simple 1D dispatch logic from sketch.
        }
        dispatchX = Math.max(1, dispatchX); // Ensure at least one workgroup.
        // logTaskActivity(`Chunk ${chunkTask.chunkId} dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]. Assumed WG size: [${workgroupSizeX},${workgroupSizeY},${workgroupSizeZ}]`, 'debug');

        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();

        readBuf = state.device.createBuffer({
            size: outBuf.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
        state.device.queue.submit([commandEncoder.finish()]);

        await state.device.queue.onSubmittedWorkDone();
        await readBuf.mapAsync(GPUMapMode.READ);

        const resultDataBytes = new Uint8Array(readBuf.getMappedRange().slice(0));
        const resultBase64 = btoa(String.fromCharCode(...resultDataBytes));
        readBuf.unmap();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`Chunk ${chunkTask.chunkId} completed in ${processingTime.toFixed(0)}ms. Output: ${resultDataBytes.length} bytes (estimated: ${estimatedOutputChunkSize}).`);
        socket.emit('workload:chunk_done', {
            parentId: chunkTask.parentId, chunkId: chunkTask.chunkId,
            chunkOrderIndex: chunkTask.chunkOrderIndex, result: resultBase64, processingTime
        });

    } catch (err) {
        logTaskActivity(`Error processing chunk ${chunkTask.chunkId} (Parent ${chunkTask.parentId}): ${err.message}`, 'error');
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
        updateComputationStatusDisplay();
        requestMatrixTaskIfNeeded();
    }
});


// Helper function to request matrix task if client is idle
function requestMatrixTaskIfNeeded() {
    if (state.connected && elements.joinComputation && !elements.joinComputation.disabled && // Client has joined general computation pool
        !state.isComputingMatrix && !state.isComputingWgsl && !state.isComputingChunk && !state.currentTask) {
        logTaskActivity("Client idle, requesting matrix task.", "debug");
        requestMatrixTask();
    }
}

function updateActiveWgslWorkloadsDisplay(workloadsFromServer) {
    // ... (updateActiveWgslWorkloadsDisplay - unchanged, but will now show parent workloads and their statuses like 'pending_chunks', 'aggregating')
    if (IS_HEADLESS || !elements.activeWgslWorkloadsGrid) return;
    elements.activeWgslWorkloadsGrid.innerHTML = '';
    if (!workloadsFromServer || workloadsFromServer.length === 0) {
        elements.activeWgslWorkloadsGrid.innerHTML = '<p>No custom WGSL workloads currently.</p>';
        return;
    }
    workloadsFromServer.forEach(wl => {
        const card = document.createElement('div');
        card.className = `wgsl-card status-${wl.status || 'unknown'}`;
        card.id = `wgsl-card-${wl.id}`;

        let chunkProgress = '';
        if (wl.isChunkParent && wl.status !== 'complete' && wl.status !== 'error' && wl.status !== 'queued') {
            // Try to get chunk progress info if server sends it (not currently implemented in detail for this UI card)
            // For example, wl.completedChunks / wl.totalChunks if server added that to parent workload meta for UI
            // Placeholder:
            // chunkProgress = `<p>Chunked Task: Awaiting detailed progress.</p>`;
        }


        let cardHTML = `
            <h4>${wl.label} <small>(${wl.id.substring(0,6)})${wl.isChunkParent ? ' (Chunked)' : ''}</small></h4>
            <p>Status: <strong>${wl.status || 'N/A'}</strong></p>
            ${chunkProgress}
            <p>Dispatch: ${wl.workgroupCount.join('x')} groups ${wl.isChunkParent ? '(per parent, client recalculates for chunk)' : ''}</p>
            <p>Output: ${wl.outputSize} bytes ${wl.isChunkParent ? '(final aggregated)' : ''}</p>
            ${wl.wgsl ? `<p>Shader: <pre>${wl.wgsl.substring(0,100)}${wl.wgsl.length > 100 ? '...' : ''}</pre></p>` : ''}
            ${wl.results && !wl.isChunkParent ? `<p>Submissions (non-chunked): ${wl.results.length}</p>` : ''}
            ${wl.finalResult ? `<p>Final Result (first 10 bytes): <pre>${JSON.stringify(wl.finalResult.slice(0,10))}...</pre></p>` : ''}
            ${wl.error ? `<p class="error">Error: ${wl.error}</p>` : ''}
            ${wl.createdAt ? `<p><small>Created: ${new Date(wl.createdAt).toLocaleString()}</small></p>` : ''}
            ${wl.startedAt && wl.status !== 'queued' ? `<p><small>Started: ${new Date(wl.startedAt).toLocaleString()}</small></p>` : ''}`;

        if (wl.status === 'complete' && wl.completedAt) {
            const startTimeForCalc = wl.startedAt || wl.createdAt;
            if (startTimeForCalc) {
                const durationMs = wl.completedAt - startTimeForCalc;
                const durationSeconds = durationMs / 1000;
                let durationFormatted;
                if (durationSeconds < 0) { durationFormatted = "Invalid timestamps"; }
                else if (durationSeconds < 60) { durationFormatted = `${durationSeconds.toFixed(1)}s`; }
                else if (durationSeconds < 3600) { const minutes = Math.floor(durationSeconds / 60); const seconds = (durationSeconds % 60).toFixed(1); durationFormatted = `${minutes}m ${seconds}s`; }
                else { const hours = Math.floor(durationSeconds / 3600); const minutes = Math.floor((durationSeconds % 3600) / 60); const seconds = (durationSeconds % 60).toFixed(1); durationFormatted = `${hours}h ${minutes}m ${seconds}s`;}
                cardHTML += `<p>Completion Time: <strong>${durationFormatted}</strong></p>`;
                 if (wl.completedAt) { cardHTML += `<p><small>Completed: ${new Date(wl.completedAt).toLocaleString()}</small></p>`; }
            }
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
    elements.clientStatus.textContent = 'Connected to server'; elements.clientStatus.className = 'status success';
    logTaskActivity('Connected to computation server');
    if (elements.joinComputation) elements.joinComputation.disabled = !(state.webgpuSupported || window.isSecureContext); // Enable if WebGPU ready or CPU fallback possible
    updateComputationStatusDisplay();
});

socket.on('disconnect', () => {
    state.connected = false;
    state.isComputingMatrix = false; state.isComputingWgsl = false; state.isComputingChunk = false;
    elements.clientStatus.textContent = 'Disconnected from server'; elements.clientStatus.className = 'status error';
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
    // This event is primarily for matrix computation state. WGSL workload status is handled by workloads:list_update.
    // We can update the general computation status text if needed.
    // if (data.isRunning) {
    //    // Handled by task-specific logic now for computationStatus element
    // } else {
    //    // if (!state.isComputingWgsl && !state.isComputingChunk) {
    //    //    elements.computationStatus.textContent = 'No matrix computation in progress'; elements.computationStatus.className = 'status warning';
    //    // }
    // }
    updateStatsDisplay(data.stats); // Update matrix-specific stats
});

socket.on('clients:update', (data) => { updateClientDisplay(data.clients); });

socket.on('task:assign', async (task) => { // Matrix task assignment
    if (!state.connected || (elements.joinComputation && elements.joinComputation.disabled && !IS_HEADLESS)) return; // Not joined computation pool

    if (state.isComputingChunk || state.isComputingWgsl || state.isComputingMatrix) {
         logTaskActivity(`Received matrix task ${task.id} but client is busy. Server should re-assign.`, 'warning');
         // Server timeout will handle re-assignment if client doesn't complete quickly.
         // Or client could explicitly reject: socket.emit('task:reject', {taskId: task.id, reason: 'busy'}); (not implemented)
         return;
    }

    if (task.type && task.type !== 'matrixMultiply') {
        logTaskActivity(`Received task of unknown type: ${task.type}. Ignoring.`, 'warning');
        return;
    }
    try {
        const result = await processMatrixTask(task);
        socket.emit('task:complete', result);
    } catch (error) {
        logTaskActivity(`Error processing matrix task: ${error.message}`, 'error');
        // state.currentTask and busy flags are reset in processMatrixTask's finally block
    } finally {
        requestMatrixTaskIfNeeded();
    }
});

socket.on('task:wait', (data) => { // For matrix tasks
    if (data && data.type && data.type !== 'matrixMultiply') return;
    if (!state.isComputingChunk && !state.isComputingWgsl && !state.isComputingMatrix) { // Only log/act if truly idle
        elements.taskStatus.textContent = 'Waiting for available matrix tasks'; elements.taskStatus.className = 'status warning';
        logTaskActivity('No matrix tasks available, waiting...');
        // state.currentTask should be null here from previous task completion
        if (state.connected && elements.joinComputation && !elements.joinComputation.disabled) { // If joined computation
            setTimeout(requestMatrixTaskIfNeeded, 5000 + Math.random() * 5000); // Retry request if idle
        }
    }
});

socket.on('computation:complete', (data) => { // Matrix computation finished
    if (data && data.type && data.type !== 'matrixMultiply') return;
    // elements.computationStatus.textContent = `Matrix computation completed in ${data.totalTime.toFixed(1)}s`;
    // elements.computationStatus.className = 'status success';
    logTaskActivity(`Matrix computation completed in ${data.totalTime.toFixed(1)}s`);
    // state.currentTask should be null
    updateComputationStatusDisplay(); // Reflect general idleness if no other tasks
});

socket.on('task:error', (data) => { // Server error for a matrix task client was working on
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Server error for matrix task ${data.taskId}: ${data.message}`, 'error');
    if(state.currentTask && state.currentTask.id === data.taskId) {
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
    // logAdminActivity('Received updated list of all custom WGSL workloads.', 'wgsl', 'info'); // Can be noisy
    updateActiveWgslWorkloadsDisplay(allWorkloads);
});

socket.on('workload:complete', (data) => { // For any custom workload (chunked parent or non-chunked)
    logTaskActivity(`Server confirmed custom workload "${data.label || data.id.substring(0,6)}" is complete!`, 'success');
    // UI updates via workloads:list_update
});

socket.on('workload:parent_started', (data) => { // Chunked parent started on server
    logTaskActivity(`Parent workload "${data.label || data.id.substring(0,6)}" started chunk processing (Status: ${data.status}).`, 'info');
    // UI updates via workloads:list_update
});


socket.on('workload:removed', (data) => {
    if (IS_HEADLESS) return;
    logAdminActivity(`Workload "${data.label || data.id.substring(0,6)}" was removed by server.`, 'wgsl', 'info');
    const cardToRemove = document.getElementById(`wgsl-card-${data.id}`);
    if (cardToRemove) {
        cardToRemove.remove();
    }
    if (elements.activeWgslWorkloadsGrid && elements.activeWgslWorkloadsGrid.children.length === 0) {
        elements.activeWgslWorkloadsGrid.innerHTML = '<p>No custom WGSL workloads currently.</p>';
    }
});

socket.on('admin:feedback', (data) => {
    if (data && data.message) {
        logAdminActivity(`Server: ${data.message}`, data.panelType || 'wgsl', data.success ? 'success' : 'error');
    }
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
        // Get chunkable parameters from UI if they exist, or default them
        const isChunkable = document.getElementById('wgsl-chunkable') ? document.getElementById('wgsl-chunkable').checked : false;
        const inputChunkProcessingType = document.getElementById('wgsl-chunk-processing-type') ? document.getElementById('wgsl-chunk-processing-type').value : 'elements';
        const inputChunkSize = document.getElementById('wgsl-input-chunk-size') ? parseInt(document.getElementById('wgsl-input-chunk-size').value) : 0;
        const inputElementSizeBytes = document.getElementById('wgsl-input-element-size-bytes') ? parseInt(document.getElementById('wgsl-input-element-size-bytes').value) : 4;
        const outputAggregationMethod = document.getElementById('wgsl-output-aggregation-method') ? document.getElementById('wgsl-output-aggregation-method').value : 'concatenate';


        const payload = {
            label: elements.wgslLabel.value || 'Untitled WGSL Workload',
            wgsl: elements.wgslSrc.value,
            entry: elements.wgslEntryPoint.value || 'main',
            workgroupCount: [
                Math.max(1, +elements.wgslGroupsX.value || 1), // Ensure positive, default to 1
                Math.max(1, +elements.wgslGroupsY.value || 1),
                Math.max(1, +elements.wgslGroupsZ.value || 1)
            ],
            bindLayout: elements.wgslBindLayout.value, // Potentially for future complex bind group setup
            outputSize: +elements.wgslOutputSize.value, // For non-chunked, this is direct output. For chunked, this is FINAL aggregated size.
            input: elements.wgslInputData.value.trim() || undefined, // Base64 encoded string

            // Chunking parameters
            chunkable: isChunkable,
            inputChunkProcessingType: isChunkable ? inputChunkProcessingType : undefined,
            inputChunkSize: isChunkable ? inputChunkSize : undefined,
            inputElementSizeBytes: isChunkable && inputChunkProcessingType === 'elements' ? inputElementSizeBytes : undefined,
            outputAggregationMethod: isChunkable ? outputAggregationMethod : undefined,
        };

        if (!payload.wgsl || !payload.outputSize) { // workgroupCount now defaults to 1,1,1
            logAdminActivity('WGSL Upload: Missing WGSL source or output size.', 'wgsl', 'error'); return;
        }
        if (isChunkable) {
            if (!payload.input) {
                 logAdminActivity('WGSL Upload: Input data is required for chunkable workloads.', 'wgsl', 'error'); return;
            }
            if (!payload.inputChunkSize || payload.inputChunkSize <= 0) {
                 logAdminActivity('WGSL Upload: Invalid inputChunkSize for chunkable workload.', 'wgsl', 'error'); return;
            }
            if (payload.inputChunkProcessingType === 'elements' && (!payload.inputElementSizeBytes || payload.inputElementSizeBytes <= 0)) {
                 logAdminActivity('WGSL Upload: Invalid inputElementSizeBytes for chunkable workload with element processing type.', 'wgsl', 'error'); return;
            }
        }


        logAdminActivity(`Pushing WGSL workload: "${payload.label}"...`, 'wgsl');
        try {
            const res = await fetch('/api/workloads', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
            });
            const json = await res.json();
            if (res.ok && json.ok) {
                logAdminActivity(`WGSL Workload ID ${json.id.substring(0,6)} (${payload.label}) ${json.message || 'submitted successfully.'}`, 'wgsl', 'success');
            } else {
                logAdminActivity(`WGSL Upload Error: ${json.error || 'Unknown server error'} (Status: ${res.status})`, 'wgsl', 'error');
            }
        } catch (err) {
            logAdminActivity(`WGSL Upload Fetch Error: ${err.message}`, 'wgsl', 'error'); console.error("WGSL Push error:", err);
        }
    });

    if (elements.startQueuedWgslButton) {
        elements.startQueuedWgslButton.addEventListener('click', () => {
            logAdminActivity('Requesting server to start all queued WGSL workloads...', 'wgsl');
            socket.emit('admin:startQueuedCustomWorkloads');
        });
    } else {
        console.warn("Admin button 'startQueuedWgslButton' not found.");
    }
}

// Initialize the application
async function init() {
    if (IS_HEADLESS) {
        document.documentElement.style.display = 'none';
        console.log(`Puppeteer worker #${WORKER_ID} initializing. UI hidden.`);
        document.title = `Headless Worker ${WORKER_ID} - Initializing`;
    }

    await initWebGPU();
    updateComputationStatusDisplay(); // Initial status

    if (IS_HEADLESS) {
        if (elements.joinComputation && !elements.joinComputation.disabled) { // Check if join is possible
            logTaskActivity("Headless mode: Auto-joining computation...");
            joinComputation(); // This will also attempt to get tasks
        } else {
             logTaskActivity("Headless mode: Cannot auto-join. WebGPU init might have failed, or join button is disabled.", 'warning');
        }
    } else {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('admin')) {
            if(elements.adminPanel) elements.adminPanel.style.display = 'block';
        }
    }
}

init();