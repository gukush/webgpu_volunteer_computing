// State management
const state = {
    webgpuSupported: false,
    device: null,
    adapter: null,
    adapterInfo: null,
    connected: false,
    clientId: null,
    isComputing: false,
    currentTask: null,
    completedTasks: 0,
    statistics: {
        processingTime: 0
    }
};

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
    adminLog: document.getElementById('admin-log'),
    clientGrid: document.getElementById('client-grid'),

    // Stats
    activeClients: document.getElementById('active-clients'),
    totalTasks: document.getElementById('total-tasks'),
    completedTasks: document.getElementById('completed-tasks'),
    elapsedTime: document.getElementById('elapsed-time'),
    myTasks: document.getElementById('my-tasks'),
    processingTime: document.getElementById('processing-time')
};

// Connect to Socket.io server
const socket = io();

async function initWebGPU() {
    try {
        // Check if we're in a secure context
        if (!window.isSecureContext) {
            elements.webgpuStatus.innerHTML = `
                <div>WebGPU requires a secure context (HTTPS or localhost).</div>
                <div>Current context is not secure. Please use HTTPS to enable WebGPU.</div>
            `;
            elements.webgpuStatus.className = 'status error';
            console.error("Not in a secure context. WebGPU requires HTTPS or localhost.");

            // Enable join button anyway - will use CPU fallback
            elements.joinComputation.disabled = false;
            return false;
        }

        // Now check for navigator.gpu
        if (!navigator.gpu) {
            elements.webgpuStatus.textContent = 'WebGPU is not supported in this browser - will use CPU computation';
            elements.webgpuStatus.className = 'status warning';

            // Enable join button anyway - will use CPU fallback
            elements.joinComputation.disabled = false;
            return false;
        }

        console.log("Starting WebGPU initialization with advanced GPU selection...");

        // Try to get all available adapters (requires Chrome 113+)
        let selectedAdapter = null;
        let adapters = [];

        // First try high-performance adapter
        try {
            const highPerfAdapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            if (highPerfAdapter) {
                adapters.push({
                    adapter: highPerfAdapter,
                    type: 'high-performance'
                });
            }
        } catch (e) {
            console.warn("Error requesting high-performance adapter:", e);
        }

        // Then try low-power adapter
        try {
            const lowPowerAdapter = await navigator.gpu.requestAdapter({
                powerPreference: 'low-power'
            });
            if (lowPowerAdapter && !adapters.some(a => a.adapter === lowPowerAdapter)) {
                adapters.push({
                    adapter: lowPowerAdapter,
                    type: 'low-power'
                });
            }
        } catch (e) {
            console.warn("Error requesting low-power adapter:", e);
        }

        // Finally try default adapter
        try {
            const defaultAdapter = await navigator.gpu.requestAdapter();
            if (defaultAdapter && !adapters.some(a => a.adapter === defaultAdapter)) {
                adapters.push({
                    adapter: defaultAdapter,
                    type: 'default'
                });
            }
        } catch (e) {
            console.warn("Error requesting default adapter:", e);
        }

        console.log(`Found ${adapters.length} potential adapters`);

        // Get info for all adapters
        for (let i = 0; i < adapters.length; i++) {
            try {
                adapters[i].info = await adapters[i].adapter.requestAdapterInfo();
                console.log(`Adapter ${i} (${adapters[i].type}):`,
                    adapters[i].info.vendor,
                    adapters[i].info.architecture,
                    adapters[i].info.device);
            } catch (e) {
                console.warn(`Couldn't get info for adapter ${i}:`, e);
                adapters[i].info = {};
            }
        }

        // Look for NVIDIA GPU specifically
        for (const adapterInfo of adapters) {
            const info = adapterInfo.info;
            const isNvidia = info.vendor?.toLowerCase().includes('nvidia') ||
                           info.device?.toLowerCase().includes('nvidia') ||
                           info.description?.toLowerCase().includes('nvidia');

            if (isNvidia) {
                console.log("Found NVIDIA GPU! Using it preferentially.");
                selectedAdapter = adapterInfo.adapter;
                state.adapterInfo = info;
                break;
            }
        }

        // If no NVIDIA, fall back to first available adapter
        if (!selectedAdapter && adapters.length > 0) {
            console.log("No NVIDIA GPU found, using first available adapter.");
            selectedAdapter = adapters[0].adapter;
            state.adapterInfo = adapters[0].info;
        }

        if (!selectedAdapter) {
            elements.webgpuStatus.textContent = 'Couldn\'t find a suitable WebGPU adapter';
            elements.webgpuStatus.className = 'status error';

            // Enable join button anyway - will use CPU fallback
            elements.joinComputation.disabled = false;
            return false;
        }

        // Store the selected adapter
        state.adapter = selectedAdapter;

        // Request the device
        console.log("Requesting device from selected adapter...");
        state.device = await state.adapter.requestDevice();
        state.webgpuSupported = true;

        // Update UI with info about the selected GPU
        elements.webgpuStatus.textContent = `WebGPU is ready with ${state.adapterInfo.vendor || ''} ${state.adapterInfo.device || ''}`;
        elements.webgpuStatus.className = 'status success';

        // Display GPU information in the UI
        let gpuInfoHTML = '<div>';

        if (state.adapterInfo.vendor) {
            gpuInfoHTML += `<strong>Vendor:</strong> ${state.adapterInfo.vendor} `;
        }

        if (state.adapterInfo.architecture) {
            gpuInfoHTML += `<strong>Architecture:</strong> ${state.adapterInfo.architecture} `;
        }

        if (state.adapterInfo.device && state.adapterInfo.device !== "") {
            gpuInfoHTML += `<strong>Device:</strong> ${state.adapterInfo.device} `;
        }

        if (state.adapterInfo.description && state.adapterInfo.description !== "") {
            gpuInfoHTML += `<strong>Description:</strong> ${state.adapterInfo.description}`;
        }

        gpuInfoHTML += '</div>';
        elements.gpuInfo.innerHTML = gpuInfoHTML;

        // Always enable join button
        elements.joinComputation.disabled = false;

        return true;
    } catch (error) {
        console.error('Error initializing WebGPU:', error);
        elements.webgpuStatus.textContent = `WebGPU error: ${error.message} - will use CPU computation`;
        elements.webgpuStatus.className = 'status warning';

        // Enable join button anyway - will use CPU fallback
        elements.joinComputation.disabled = false;
        return false;
    }
}

// Matrix multiplication using WebGPU
async function multiplyMatricesGPU(matrixA, matrixB, size, startRow, endRow) {
    // Log start of computation
    logTaskActivity(`Starting GPU computation for rows ${startRow}-${endRow}`);

    const startTime = performance.now();

    try {
        // Create flat typed arrays from the matrices
        const flatMatrixA = new Float32Array(size * size);
        const flatMatrixB = new Float32Array(size * size);

        // Convert 2D arrays to flat arrays
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                flatMatrixA[i * size + j] = matrixA[i][j];
                flatMatrixB[i * size + j] = matrixB[i][j];
            }
        }

        // Create a shader that will compute the matrix multiplication
        const shaderModule = state.device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
                @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
                @group(0) @binding(2) var<storage, write> resultMatrix: array<f32>;

                struct Uniforms {
                    size: u32,
                    startRow: u32,
                    endRow: u32,
                }
                @group(0) @binding(3) var<uniform> uniforms: Uniforms;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let row = global_id.x + uniforms.startRow;
                    let col = global_id.y;

                    if (row >= uniforms.endRow || col >= uniforms.size) {
                        return;
                    }

                    var sum: f32 = 0.0;
                    for (var i: u32 = 0; i < uniforms.size; i = i + 1) {
                        sum = sum + matrixA[row * uniforms.size + i] * matrixB[i * uniforms.size + col];
                    }

                    resultMatrix[(row - uniforms.startRow) * uniforms.size + col] = sum;
                }
            `
        });

        // Create buffers
        const aBuffer = state.device.createBuffer({
            size: flatMatrixA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bBuffer = state.device.createBuffer({
            size: flatMatrixB.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // The result buffer only needs to store the rows we're calculating
        const rowsToCalculate = endRow - startRow;
        const resultBufferSize = rowsToCalculate * size * Float32Array.BYTES_PER_ELEMENT;

        const resultBuffer = state.device.createBuffer({
            size: resultBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const uniformBuffer = state.device.createBuffer({
            size: 3 * Uint32Array.BYTES_PER_ELEMENT, // size, startRow, endRow
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const uniformData = new Uint32Array([size, startRow, endRow]);

        // Copy data to GPU
        state.device.queue.writeBuffer(aBuffer, 0, flatMatrixA);
        state.device.queue.writeBuffer(bBuffer, 0, flatMatrixB);
        state.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Create a pipeline
        const computePipeline = state.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        // Set up bindings
        const bindGroup = state.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuffer } },
                { binding: 1, resource: { buffer: bBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: uniformBuffer } },
            ],
        });

        // Create a command encoder
        const commandEncoder = state.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // Dispatch workgroups
        const rowWorkgroups = Math.ceil((endRow - startRow) / 8);
        const colWorkgroups = Math.ceil(size / 8);
        passEncoder.dispatchWorkgroups(rowWorkgroups, colWorkgroups);
        passEncoder.end();

        // Create a buffer to read the results
        const readBuffer = state.device.createBuffer({
            size: resultBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultBufferSize);
        state.device.queue.submit([commandEncoder.finish()]);

        // Read back the result
        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Float32Array(readBuffer.getMappedRange());

        // Create a result matrix (we're returning just the computed rows)
        const result = new Array(rowsToCalculate);
        for (let i = 0; i < rowsToCalculate; i++) {
            result[i] = new Array(size);
            for (let j = 0; j < size; j++) {
                result[i][j] = resultArray[i * size + j];
            }
        }

        readBuffer.unmap();

        const endTime = performance.now();
        const processingTime = endTime - startTime;

        logTaskActivity(`Completed GPU computation in ${processingTime.toFixed(2)}ms`);

        return {
            result,
            processingTime
        };
    } catch (error) {
        logTaskActivity(`Error in GPU computation: ${error.message}`, 'error');
        throw error;
    }
}

// CPU fallback for matrix multiplication
async function multiplyMatricesCPU(matrixA, matrixB, size, startRow, endRow) {
    logTaskActivity(`Starting CPU computation for rows ${startRow}-${endRow}`);

    const startTime = performance.now();

    try {
        // Create result matrix for the rows we're calculating
        const rowsToCalculate = endRow - startRow;
        const result = new Array(rowsToCalculate);

        for (let i = 0; i < rowsToCalculate; i++) {
            result[i] = new Array(size);
            const rowIndex = startRow + i;

            for (let j = 0; j < size; j++) {
                let sum = 0;
                for (let k = 0; k < size; k++) {
                    sum += matrixA[rowIndex][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }

        const endTime = performance.now();
        const processingTime = endTime - startTime;

        logTaskActivity(`Completed CPU computation in ${processingTime.toFixed(2)}ms`);

        return {
            result,
            processingTime
        };
    } catch (error) {
        logTaskActivity(`Error in CPU computation: ${error.message}`, 'error');
        throw error;
    }
}

// Process a task
async function processTask(task) {
    state.currentTask = task;

    elements.taskStatus.textContent = `Processing task ${task.id} (rows ${task.startRow}-${task.endRow})`;
    elements.taskStatus.className = 'status info';

    try {
        let result;

        // Use GPU if available, otherwise fall back to CPU
        if (state.webgpuSupported && state.device) {
            logTaskActivity("Using WebGPU for computation");
            result = await multiplyMatricesGPU(
                task.matrixA,
                task.matrixB,
                task.size,
                task.startRow,
                task.endRow
            );
        } else {
            logTaskActivity("Using CPU fallback for computation");
            result = await multiplyMatricesCPU(
                task.matrixA,
                task.matrixB,
                task.size,
                task.startRow,
                task.endRow
            );
        }

        state.completedTasks++;
        state.statistics.processingTime += result.processingTime;

        elements.myTasks.textContent = state.completedTasks;
        elements.processingTime.textContent = `${result.processingTime.toFixed(0)}ms`;

        elements.taskStatus.textContent = `Completed task ${task.id}`;
        elements.taskStatus.className = 'status success';

        return {
            taskId: task.id,
            result: result.result,
            processingTime: result.processingTime
        };
    } catch (error) {
        elements.taskStatus.textContent = `Error processing task: ${error.message}`;
        elements.taskStatus.className = 'status error';
        throw error;
    }
}

// Join the computation
function joinComputation() {
    // Allow joining regardless of WebGPU support
    elements.joinComputation.disabled = true;
    elements.leaveComputation.disabled = false;

    if (state.webgpuSupported && state.device) {
        logTaskActivity('Joining computation network with WebGPU acceleration...');
    } else {
        logTaskActivity('Joining computation network with CPU computation...');
    }

    socket.emit('client:join', {
        gpuInfo: state.adapterInfo || { vendor: 'CPU Fallback', device: 'CPU Computation' }
    });

    state.isComputing = true;

    elements.computationStatus.textContent = 'Joined computation network, waiting for tasks';
    elements.computationStatus.className = 'status info';
}

// Leave the computation
function leaveComputation() {
    elements.joinComputation.disabled = false;
    elements.leaveComputation.disabled = true;

    state.isComputing = false;
    state.currentTask = null;

    logTaskActivity('Left computation network');

    elements.taskStatus.textContent = 'No task assigned';
    elements.taskStatus.className = 'status info';

    elements.computationStatus.textContent = 'Not participating in computation';
    elements.computationStatus.className = 'status warning';
}

// Request a task from the server
function requestTask() {
    if (state.isComputing && !state.currentTask) {
        socket.emit('task:request');
    }
}

// Start a new computation (admin)
function startComputation() {
    const matrixSize = parseInt(elements.matrixSize.value);
    const chunkSize = parseInt(elements.chunkSize.value);

    socket.emit('admin:start', {
        matrixSize,
        chunkSize
    });

    logAdminActivity(`Starting new computation: ${matrixSize}×${matrixSize} with chunk size ${chunkSize}`);
}

// Add a log entry to the task log
function logTaskActivity(message, type = 'info') {
    const logItem = document.createElement('div');
    logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.taskLog.appendChild(logItem);
    elements.taskLog.scrollTop = elements.taskLog.scrollHeight;
}

// Add a log entry to the admin log
function logAdminActivity(message, type = 'info') {
    const logItem = document.createElement('div');
    logItem.className = type;
    logItem.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.adminLog.appendChild(logItem);
    elements.adminLog.scrollTop = elements.adminLog.scrollHeight;
}

// Update client display
function updateClientDisplay(clients) {
    elements.clientGrid.innerHTML = '';

    if (clients.length === 0) {
        const emptyMsg = document.createElement('div');
        emptyMsg.textContent = 'No clients connected';
        elements.clientGrid.appendChild(emptyMsg);
        return;
    }

    clients.forEach(client => {
        const isCurrentClient = client.id === state.clientId;
        const timeSinceActive = Date.now() - client.lastActive;
        const isActive = timeSinceActive < 30000; // 30 seconds
        const isCpuClient = client.usingCpu ||
                           (client.gpuInfo && client.gpuInfo.vendor === 'CPU Fallback') ||
                           (client.gpuInfo && client.gpuInfo.device === 'CPU Computation');

        const clientEl = document.createElement('div');
        clientEl.className = `client-card ${!isActive ? 'client-inactive' : ''} ${isCpuClient ? 'client-cpu' : 'client-gpu'}`;

        let clientHTML = `
            <div>${isCurrentClient ? 'You' : 'Client'}</div>
            <div><small>${client.id.substring(0, 8)}...</small></div>
        `;

        if (client.gpuInfo) {
            if (isCpuClient) {
                clientHTML += `<div><small>CPU Computation</small></div>`;
            } else {
                clientHTML += `<div><small>${client.gpuInfo.vendor || ''} ${client.gpuInfo.architecture || ''}</small></div>`;
            }
        }

        clientHTML += `<div>Tasks: ${client.completedTasks}</div>`;

        clientEl.innerHTML = clientHTML;
        elements.clientGrid.appendChild(clientEl);
    });
}

// Update stats display
function updateStatsDisplay(stats) {
    elements.activeClients.textContent = stats.activeClients || 0;
    elements.totalTasks.textContent = stats.totalTasks || 0;
    elements.completedTasks.textContent = stats.completedTasks || 0;

    if (stats.elapsedTime) {
        elements.elapsedTime.textContent = `${stats.elapsedTime.toFixed(1)}s`;
    }
}

// Socket.io event handlers
socket.on('connect', () => {
    state.connected = true;
    elements.clientStatus.textContent = 'Connected to server';
    elements.clientStatus.className = 'status success';

    logTaskActivity('Connected to computation server');

    // Make sure join button is enabled
    elements.joinComputation.disabled = false;
});

socket.on('disconnect', () => {
    state.connected = false;
    elements.clientStatus.textContent = 'Disconnected from server';
    elements.clientStatus.className = 'status error';

    logTaskActivity('Disconnected from server', 'error');
});

socket.on('register', (data) => {
    state.clientId = data.clientId;
    elements.clientStatus.textContent = `Connected as client: ${data.clientId.substring(0, 8)}...`;

    // Enable join button regardless of WebGPU support
    elements.joinComputation.disabled = false;
});

socket.on('state:update', (data) => {
    // Update computation status
    if (data.isRunning) {
        elements.computationStatus.textContent = 'Computation in progress';
        elements.computationStatus.className = 'status info';
    } else {
        elements.computationStatus.textContent = 'No computation in progress';
        elements.computationStatus.className = 'status warning';
    }

    // Update stats
    updateStatsDisplay(data.stats);
});

socket.on('clients:update', (data) => {
    updateClientDisplay(data.clients);
});

socket.on('task:assign', async (task) => {
    if (!state.isComputing) return;

    try {
        // Process the task
        const result = await processTask(task);

        // Send the result back
        socket.emit('task:complete', result);

        // Clear current task
        state.currentTask = null;

        // Request another task
        requestTask();
    } catch (error) {
        console.error('Error processing task:', error);
        logTaskActivity(`Error processing task: ${error.message}`, 'error');

        // Clear current task
        state.currentTask = null;

        // Request another task after a short delay
        setTimeout(requestTask, 1000);
    }
});

socket.on('task:wait', () => {
    elements.taskStatus.textContent = 'Waiting for available tasks';
    elements.taskStatus.className = 'status warning';
    logTaskActivity('No tasks available, waiting...');

    state.currentTask = null;

    // Check again after a short delay
    setTimeout(requestTask, 2000);
});

socket.on('computation:complete', (data) => {
    elements.computationStatus.textContent = `Computation completed in ${data.totalTime.toFixed(2)} seconds`;
    elements.computationStatus.className = 'status success';

    logTaskActivity(`Computation completed in ${data.totalTime.toFixed(2)} seconds`);

    state.currentTask = null;
});

socket.on('error', (data) => {
    logTaskActivity(`Server error: ${data.message}`, 'error');
});

// Event listeners
elements.joinComputation.addEventListener('click', joinComputation);
elements.leaveComputation.addEventListener('click', leaveComputation);
elements.startComputation.addEventListener('click', startComputation);
elements.toggleAdmin.addEventListener('click', () => {
    if (elements.adminPanel.style.display === 'block') {
        elements.adminPanel.style.display = 'none';
    } else {
        elements.adminPanel.style.display = 'block';
    }
});

// Initialize the application
async function init() {
    // Initialize WebGPU
    await initWebGPU();

    // Check if admin panel should be shown (URL param)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('admin')) {
        elements.adminPanel.style.display = 'block';
    }

    // Force enable join button after a short delay
    setTimeout(() => {
        elements.joinComputation.disabled = false;
    }, 2000);
}

init();