// client.js

// State management (no changes here)
// ... state object ...
const state = {
    webgpuSupported: false, device: null, adapter: null, adapterInfo: null,
    connected: false, clientId: null, isComputing: false, currentTask: null,
    completedTasks: 0, statistics: { processingTime: 0 }
};


// elements object (no changes needed here)
// ... elements object ...
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
    startQueuedWgslButton: document.getElementById('startQueuedWgslButton')
};

// Headless mode detection & socket connection (no changes)
// ... PARAMS, IS_HEADLESS, WORKER_ID, socket ...
const PARAMS = new URLSearchParams(location.search);
const IS_HEADLESS = PARAMS.get('mode') === 'headless';
const WORKER_ID = PARAMS.get('workerId') || 'N/A';
const socket = io({ query: IS_HEADLESS ? { mode: 'headless', workerId: WORKER_ID } : {} });

// initWebGPU, multiplyMatricesGPU, multiplyMatricesCPU, processMatrixTask,
// joinComputation, leaveComputation, requestMatrixTask, startMatrixComputation,
// logTaskActivity, logAdminActivity, updateClientDisplay, updateStatsDisplay
// (These functions remain unchanged from your previous version)
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

        const discreteVendors = ['nvidia', 'amd', 'advanced micro devices', 'apple'];
        let foundAdapterEntry = adaptersWithInfo.find(a =>
            discreteVendors.some(v =>
                (a.info.vendor?.toLowerCase().includes(v) ||
                 a.info.description?.toLowerCase().includes(v)) &&
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver')
            )
        );

        if (!foundAdapterEntry && adaptersWithInfo.length > 0) {
            foundAdapterEntry = adaptersWithInfo.find(a =>
                !a.info.description?.toLowerCase().includes('swiftshader') &&
                !a.info.description?.toLowerCase().includes('microsoft basic render driver')
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
        const resultArray = new Float32Array(readBuffer.getMappedRange());
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
        socket.emit('task:error', { taskId: task.id, message: error.message, type: 'matrixMultiply' });
        throw error;
    }
}

function joinComputation() {
    if(elements.joinComputation) elements.joinComputation.disabled = true;
    if(elements.leaveComputation) elements.leaveComputation.disabled = false;
    const mode = (state.webgpuSupported && state.device) ? 'WebGPU' : 'CPU';
    logTaskActivity(`Joining computation network (${mode})...`);
    socket.emit('client:join', { gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback', device: 'CPU Computation', description: 'No adapter info' } });
    state.isComputing = true;
    elements.computationStatus.textContent = `Joined computation network (${mode}), waiting for tasks`;
    elements.computationStatus.className = 'status info';
}

function leaveComputation() {
    if(elements.joinComputation) elements.joinComputation.disabled = false;
    if(elements.leaveComputation) elements.leaveComputation.disabled = true;
    state.isComputing = false; state.currentTask = null;
    logTaskActivity('Left computation network');
    elements.taskStatus.textContent = 'No matrix task assigned'; elements.taskStatus.className = 'status info';
    elements.computationStatus.textContent = 'Not participating in computation'; elements.computationStatus.className = 'status warning';
    socket.emit('client:leave');
}

function requestMatrixTask() {
    if (state.isComputing && !state.currentTask) { socket.emit('task:request'); }
}

function startMatrixComputation() {
    const matrixSize = parseInt(elements.matrixSize.value);
    const chunkSize = parseInt(elements.chunkSize.value);
    socket.emit('admin:start', { matrixSize, chunkSize });
    logAdminActivity(`Starting new matrix computation: ${matrixSize}x${matrixSize} with chunk size ${chunkSize}`, 'matrix');
}

function logTaskActivity(message, type = 'info') {
    if (IS_HEADLESS) {
        console.log(`[Worker ${WORKER_ID} Log]: ${message}`);
        if (type === 'error') console.error(`[Worker ${WORKER_ID} ErrorLog]: ${message}`);
        document.title = `Worker ${WORKER_ID} | ${message.substring(0,50)}`;
        return;
    }
    const logItem = document.createElement('div'); logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.taskLog.appendChild(logItem); elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

function logAdminActivity(message, panelType = 'matrix', type = 'info') {
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
    if (IS_HEADLESS || !elements.clientGrid) return;
    elements.clientGrid.innerHTML = '';
    if (!clients || clients.length === 0) { elements.clientGrid.innerHTML = '<div>No clients connected</div>'; return; }
    clients.forEach(client => {
        const isCurrentClient = client.id === state.clientId;
        const timeSinceActive = Date.now() - client.lastActive;
        const isActive = client.connected && timeSinceActive < 60000;
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
    if (stats.elapsedTime) { elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`; }
}


// --- Custom WGSL Workload Handling ---
socket.on('workload:new', async meta => {
    if (meta.status !== 'pending') {
        logTaskActivity(`Received custom workload "${meta.label}" with status "${meta.status}". Will not process unless 'pending'.`, meta.status === 'queued' ? 'info' : 'warning');
        return;
    }

    if (!state.device) {
        logTaskActivity(`Received custom workload "${meta.label}" (status: pending) but WebGPU device not ready. Skipping.`, 'warning');
        socket.emit('workload:error', { id: meta.id, message: 'WebGPU device not available on client.' });
        return;
    }

    logTaskActivity(`Processing custom workload "${meta.label}" (ID: ${meta.id.substring(0,6)}, Status: ${meta.status})...`);
    let shader, pipeline, inputBuf, outBuf, readBuf, commandEncoder;

    try {
        const startTime = performance.now();
        shader = state.device.createShaderModule({ code: meta.wgsl });
        const compilationInfo = await shader.getCompilationInfo();
        if (compilationInfo.messages.some(m => m.type === 'error')) {
            const errorMessages = compilationInfo.messages.filter(m => m.type === 'error').map(m => m.message).join('\n');
            throw new Error(`Shader compilation failed: ${errorMessages}`);
        }

        if (meta.bindLayout !== "storage-in-storage-out") {
            throw new Error(`Unsupported bindLayout: ${meta.bindLayout}. Only 'storage-in-storage-out' is currently implemented.`);
        }
        pipeline = state.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shader, entryPoint: meta.entry || 'main' }
        });

        if (meta.input) {
            const inputDataBytes = Uint8Array.from(atob(meta.input), c => c.charCodeAt(0));
            inputBuf = state.device.createBuffer({
                size: inputDataBytes.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Uint8Array(inputBuf.getMappedRange()).set(inputDataBytes);
            inputBuf.unmap();
        }

        outBuf = state.device.createBuffer({
            size: meta.outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const entries = [];
        if (inputBuf) {
            entries.push({ binding: 0, resource: { buffer: inputBuf } });
            entries.push({ binding: 1, resource: { buffer: outBuf } });
        } else {
            const placeholderInputSize = 16;
            inputBuf = state.device.createBuffer({ size: placeholderInputSize, usage: GPUBufferUsage.STORAGE });
            entries.push({ binding: 0, resource: { buffer: inputBuf } });
            entries.push({ binding: 1, resource: { buffer: outBuf } });
        }

        const bindGroup = state.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: entries
        });

        commandEncoder = state.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...meta.workgroupCount);
        pass.end();

        readBuf = state.device.createBuffer({
            size: meta.outputSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, meta.outputSize);
        state.device.queue.submit([commandEncoder.finish()]);

        await state.device.queue.onSubmittedWorkDone();
        await readBuf.mapAsync(GPUMapMode.READ);

        const resultData = new Uint8Array(readBuf.getMappedRange().slice(0));
        const result = Array.from(resultData);

        readBuf.unmap();

        const processingTime = performance.now() - startTime;
        logTaskActivity(`Custom workload "${meta.label}" completed in ${processingTime.toFixed(0)}ms. Result size: ${result.length} bytes.`);
        socket.emit('workload:done', { id: meta.id, result, processingTime });

    } catch (err) {
        logTaskActivity(`Error processing custom workload "${meta.label}": ${err.message}`, 'error');
        console.error(`Custom workload error for ${meta.id}:`, err);
        socket.emit('workload:error', { id: meta.id, message: err.message });
    } finally {
        if (inputBuf) inputBuf.destroy();
        if (outBuf) outBuf.destroy();
        if (readBuf) readBuf.destroy();
    }
});

// MODIFIED to add Completion Time
function updateActiveWgslWorkloadsDisplay(workloadsFromServer) {
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

        let cardHTML = `
            <h4>${wl.label} <small>(${wl.id.substring(0,6)})</small></h4>
            <p>Status: <strong>${wl.status || 'N/A'}</strong></p>
            <p>Dispatch: ${wl.workgroupCount.join('x')} groups</p>
            <p>Output: ${wl.outputSize} bytes</p>
            ${wl.wgsl ? `<p>Shader: <pre>${wl.wgsl.substring(0,100)}${wl.wgsl.length > 100 ? '...' : ''}</pre></p>` : ''}
            ${wl.results ? `<p>Submissions: ${wl.results.length}</p>` : ''}
            ${wl.finalResult ? `<p>Final Result (first 10 bytes): <pre>${JSON.stringify(wl.finalResult.slice(0,10))}...</pre></p>` : ''}
            ${wl.createdAt ? `<p><small>Created: ${new Date(wl.createdAt).toLocaleString()}</small></p>` : ''}
            ${wl.startedAt && wl.status !== 'queued' ? `<p><small>Started: ${new Date(wl.startedAt).toLocaleString()}</small></p>` : ''}`;

        // --- ADDED Completion Time Display ---
        if (wl.status === 'complete' && wl.completedAt) {
            const startTimeForCalc = wl.startedAt || wl.createdAt; // Use startedAt if available, else createdAt
            if (startTimeForCalc) {
                const durationMs = wl.completedAt - startTimeForCalc;
                const durationSeconds = durationMs / 1000;
                let durationFormatted;
                if (durationSeconds < 0) { // Should not happen with correct timestamps
                    durationFormatted = "Invalid timestamps";
                } else if (durationSeconds < 60) {
                    durationFormatted = `${durationSeconds.toFixed(1)}s`;
                } else if (durationSeconds < 3600) {
                    const minutes = Math.floor(durationSeconds / 60);
                    const seconds = (durationSeconds % 60).toFixed(1);
                    durationFormatted = `${minutes}m ${seconds}s`;
                } else {
                    const hours = Math.floor(durationSeconds / 3600);
                    const minutes = Math.floor((durationSeconds % 3600) / 60);
                    const seconds = (durationSeconds % 60).toFixed(1);
                    durationFormatted = `${hours}h ${minutes}m ${seconds}s`;
                }
                cardHTML += `<p>Completion Time: <strong>${durationFormatted}</strong></p>`;
                 if (wl.completedAt) {
                    cardHTML += `<p><small>Completed: ${new Date(wl.completedAt).toLocaleString()}</small></p>`;
                }
            }
        }
        // --- END Completion Time Display ---

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
    if (elements.joinComputation) elements.joinComputation.disabled = !(state.webgpuSupported || window.isSecureContext);
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
});

socket.on('state:update', (data) => {
    if (IS_HEADLESS) return;
    if (data.isRunning) {
        elements.computationStatus.textContent = 'Matrix computation in progress'; elements.computationStatus.className = 'status info';
    } else {
        elements.computationStatus.textContent = 'No matrix computation in progress'; elements.computationStatus.className = 'status warning';
    }
    updateStatsDisplay(data.stats);
});

socket.on('clients:update', (data) => { updateClientDisplay(data.clients); });

socket.on('task:assign', async (task) => {
    if (!state.isComputing && !IS_HEADLESS) return;
    if (task.type && task.type !== 'matrixMultiply') {
        logTaskActivity(`Received task of unknown type: ${task.type}. Ignoring.`, 'warning');
        return;
    }
    try {
        const result = await processMatrixTask(task);
        socket.emit('task:complete', result); state.currentTask = null; requestMatrixTask();
    } catch (error) {
        logTaskActivity(`Error processing matrix task: ${error.message}`, 'error');
        state.currentTask = null; setTimeout(requestMatrixTask, 2000 + Math.random() * 3000);
    }
});

socket.on('task:wait', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    elements.taskStatus.textContent = 'Waiting for available matrix tasks'; elements.taskStatus.className = 'status warning';
    logTaskActivity('No matrix tasks available, waiting...'); state.currentTask = null;
    if (state.isComputing) {
        setTimeout(requestMatrixTask, 5000 + Math.random() * 5000);
    }
});

socket.on('computation:complete', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    elements.computationStatus.textContent = `Matrix computation completed in ${data.totalTime.toFixed(1)}s`;
    elements.computationStatus.className = 'status success';
    logTaskActivity(`Matrix computation completed in ${data.totalTime.toFixed(1)}s`); state.currentTask = null;
});

socket.on('task:error', (data) => {
    if (data && data.type && data.type !== 'matrixMultiply') return;
    logTaskActivity(`Server error for matrix task ${data.taskId}: ${data.message}`, 'error');
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
    logAdminActivity('Received updated list of all custom WGSL workloads.', 'wgsl', 'info');
    updateActiveWgslWorkloadsDisplay(allWorkloads);
});

socket.on('workload:complete', (data) => {
    logTaskActivity(`Server confirmed custom workload "${data.label || data.id.substring(0,6)}" is complete!`, 'success');
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
        logAdminActivity(`Server: ${data.message}`, 'wgsl', data.success ? 'success' : 'error');
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
        const payload = {
            label: elements.wgslLabel.value || 'Untitled WGSL Workload',
            wgsl: elements.wgslSrc.value,
            entry: elements.wgslEntryPoint.value || 'main',
            workgroupCount: [
                +elements.wgslGroupsX.value, +elements.wgslGroupsY.value, +elements.wgslGroupsZ.value
            ],
            bindLayout: elements.wgslBindLayout.value,
            outputSize: +elements.wgslOutputSize.value,
            input: elements.wgslInputData.value.trim() || undefined
        };

        if (!payload.wgsl || !payload.workgroupCount.every(c => c > 0) || !payload.outputSize) {
            logAdminActivity('WGSL Upload: Missing WGSL source, valid workgroup counts, or output size.', 'wgsl', 'error'); return;
        }

        logAdminActivity(`Pushing WGSL workload: "${payload.label}"...`, 'wgsl');
        try {
            const res = await fetch('/api/workloads', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
            });
            const json = await res.json();
            if (res.ok && json.ok) {
                logAdminActivity(`WGSL Workload ${json.id} (${payload.label}) ${json.message || 'queued successfully.'}`, 'wgsl', 'success');
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

    if (IS_HEADLESS) {
        if (elements.joinComputation && !elements.joinComputation.disabled) {
            logTaskActivity("Headless mode: Auto-joining computation...");
            joinComputation();
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