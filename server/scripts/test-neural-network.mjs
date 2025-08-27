#!/usr/bin/env node
// test-neural-network.mjs - Test script for simple neural network volunteer computing system

import fs from 'fs/promises';
import path from 'path';
import net from 'net';
import { fileURLToPath } from 'url';
import minimist from 'minimist';
import crypto from 'crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = minimist(process.argv.slice(2));

const inputSize = parseInt(args['input-size'], 10);
const hiddenSize = parseInt(args['hidden-size'], 10);
const outputSize = parseInt(args['output-size'], 10);
const batchSize = parseInt(args['batch-size'], 10);
const chunkSize = parseInt(args['chunk-size'], 10) || Math.min(batchSize, 100); // Default chunk size
const inputPath = args.input ? path.resolve(args.input) : null;
const framework = args.framework || 'webgpu';
const apiBase = args['api-base'] || process.env.API_BASE || 'https://localhost:3000';
const insecureFlag = args.insecure || process.env.TEST_INSECURE;
const pollInterval = Number(args.interval || process.env.TEST_POLL_INTERVAL || 2000);
const streamingMode = args.streaming !== undefined ? args.streaming : (process.env.TEST_STREAMING !== undefined ? process.env.TEST_STREAMING : true);

const SUPPORTED_FRAMEWORKS = ['webgpu', 'webgl', 'cuda', 'opencl', 'vulkan', 'javascript'];

if (!Number.isInteger(inputSize) || !Number.isInteger(hiddenSize) ||
    !Number.isInteger(outputSize) || !Number.isInteger(batchSize)) {
  console.error('Usage: --input-size N --hidden-size M --output-size P --batch-size Q [options]');
  console.error('');
  console.error('Required:');
  console.error('  --input-size     Input layer size (e.g., 784 for MNIST)');
  console.error('  --hidden-size    Hidden layer size (e.g., 1024)');
  console.error('  --output-size    Output layer size (e.g., 10 for classification)');
  console.error('  --batch-size     Number of samples per batch');
  console.error('');
  console.error('Options:');
  console.error('  --chunk-size     Samples per chunk [default: min(batchSize, 100)]');
  console.error('  --input          Path to pre-generated neural network data file');
  console.error('  --framework      GPU framework (webgpu, webgl, cuda, opencl, vulkan, javascript) [default: webgpu]');
  console.error('  --streaming      Enable streaming mode (default: true, use --no-streaming to disable)');
  console.error('  --api-base       API server URL [default: https://localhost:3000]');
  console.error('  --insecure       Accept self-signed certificates');
  console.error('  --interval       Status polling interval in ms [default: 2000]');
  console.error('');
  console.error('Examples:');
  console.error('  # Test with generated data file');
  console.error('  node test-neural-network.mjs --input-size 784 --hidden-size 1024 --output-size 10 --batch-size 100 --input neural_network_784x1024x10_batch100.bin');
  console.error('');
  console.error('  # Test with WebGL framework');
  console.error('  node test-neural-network.mjs --input-size 512 --hidden-size 256 --output-size 5 --batch-size 50 --framework webgl --input my_network.bin');
  console.error('');
  console.error('  # Test with streaming mode (default)');
  console.error('  node test-neural-network.mjs --input-size 784 --hidden-size 1024 --output-size 10 --batch-size 1000 --chunk-size 100 --input large_network.bin');
  console.error('');
  console.error('  # Test with batch mode (disable streaming)');
  console.error('  node test-neural-network.mjs --input-size 784 --hidden-size 1024 --output-size 10 --batch-size 100 --no-streaming --input neural_network.bin');
  process.exit(1);
}

if (batchSize % chunkSize !== 0) {
  console.error(`--batch-size ${batchSize} must be divisible by --chunk-size ${chunkSize}`);
  process.exit(1);
}

// If the user requested insecure, set TLS reject env early so node fetch accepts self-signed certs.
if (insecureFlag) {
  try { process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'; } catch (e) { /* no-op */ }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function parseHostPort(urlString) {
  try {
    const u = new URL(urlString);
    return { host: u.hostname, port: Number(u.port) || (u.protocol === 'https:' ? 443 : 80) };
  } catch (e) {
    const s = urlString.replace(/^https?:\/\//, '');
    const [host, port] = s.split(':');
    return { host, port: Number(port || 80) };
  }
}

function tcpConnectCheck(host, port, timeout = 2000) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host, port, timeout }, () => {
      socket.end();
      resolve(true);
    });
    socket.on('error', (err) => reject(err));
    socket.on('timeout', () => { socket.destroy(); reject(new Error('timeout')); });
  });
}

async function httpGetOk(url, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(id);
    return res.ok;
  } catch (e) {
    clearTimeout(id);
    throw e;
  }
}

async function ensureServerReachable(baseUrl, attempts = 3) {
  const { host, port } = parseHostPort(baseUrl);
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    try {
      process.stdout.write(`Checking TCP ${host}:${port} ... `);
      await tcpConnectCheck(host, port, 2000);
      console.log('ok');
      try {
        process.stdout.write(`Checking HTTP ${baseUrl} ... `);
        const ok = await httpGetOk(baseUrl, 2000).catch(()=>false);
        if (ok) { console.log('ok'); return; }
        const alt = baseUrl.replace(/\/$/, '') + '/api/status';
        process.stdout.write(`Checking HTTP ${alt} ... `);
        const ok2 = await httpGetOk(alt, 2000).catch(()=>false);
        if (ok2) { console.log('ok'); return; }
        console.log('server reachable but returned non-OK to GET (server may accept POST only).');
        return;
      } catch (e) {
        console.log('http-check failed:', e.message);
        return;
      }
    } catch (err) {
      lastErr = err;
      console.log('failed:', err.code || err.message);
      const backoff = 500 * Math.pow(2, i);
      await sleep(backoff);
    }
  }
  throw lastErr || new Error('server unreachable');
}

async function postJSON(url, obj) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });
  if (!res.ok) {
    const txt = await res.text().catch(()=>'<no-body>');
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const txt = await res.text().catch(()=>'<no-body>');
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

async function postMultipart(url, files) {
  const form = new FormData();
  for (const f of files) form.append(f.name, new Blob([f.buffer]), f.filename);
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) {
    const txt = await res.text().catch(()=>'<no-out>');
    throw new Error(`Upload failed: HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

// Check framework availability on server
async function checkFrameworkSupport(framework, apiBase) {
  try {
    const frameworks = await getJSON(`${apiBase}/api/frameworks`);

    if (!frameworks.frameworks || !frameworks.frameworks[framework]) {
      return {
        supported: false,
        reason: `Framework ${framework} not recognized by server`
      };
    }

    const stats = frameworks.stats[framework];
    if (stats.availableClients === 0) {
      return {
        supported: false,
        reason: `No clients support ${framework} framework`,
        stats
      };
    }

    return {
      supported: true,
      stats
    };
  } catch (error) {
    console.warn(`⚠️ Could not check framework support: ${error.message}`);
    return { supported: true }; // Assume supported if we can't check
  }
}

// Get framework-specific chunking strategy
function getChunkingStrategyForFramework(framework) {
  const strategyMap = {
    'webgpu': 'simple_neural_network',
    'webgl': 'simple_neural_network',
    'cuda': 'simple_neural_network',
    'opencl': 'simple_neural_network',
    'vulkan': 'simple_neural_network',
    'javascript': 'simple_neural_network'
  };

  return strategyMap[framework] || 'simple_neural_network';
}

// Get framework-specific assembly strategy
function getAssemblyStrategyForFramework(framework) {
  const strategyMap = {
    'webgpu': 'simple_neural_network_assembly',
    'webgl': 'simple_neural_network_assembly',
    'cuda': 'simple_neural_network_assembly',
    'opencl': 'simple_neural_network_assembly',
    'vulkan': 'simple_neural_network_assembly',
    'javascript': 'simple_neural_network_assembly'
  };

  return strategyMap[framework] || 'simple_neural_network_assembly';
}

// Streaming status polling with assembly progress
async function pollStatus(workloadId, intervalMs = 2000) {
  let lastProgress = -1;
  let lastAssemblyChunks = -1;
  let lastDispatchedChunks = -1;

  while (true) {
    try {
      const status = await getJSON(`${apiBase}/api/workloads/${workloadId}/status`);

      console.log(`⏳ Status: ${status.status}`);

      if (status.chunks) {
        const progress = status.chunks.progress.toFixed(1);
        if (progress !== lastProgress) {
          console.log(`   Chunk Progress: ${status.chunks.completed}/${status.chunks.total} chunks (${progress}%)`);
          lastProgress = progress;
        }

        // Show active chunks if available
        if (status.chunks.active > 0) {
          console.log(`   Active: ${status.chunks.active} chunks being processed`);
        }
      }

      // Show streaming-specific progress
      if (status.streaming) {
        const dispatchedChunks = status.streaming.dispatchedChunks || 0;
        if (dispatchedChunks !== lastDispatchedChunks) {
          console.log(`   Dispatched: ${dispatchedChunks} chunks sent to clients`);
          lastDispatchedChunks = dispatchedChunks;
        }
      }

      // Show assembly progress if available
      if (status.assembly) {
        const assemblyProgress = status.assembly.completedChunks || 0;
        if (assemblyProgress !== lastAssemblyChunks) {
          console.log(`   Assembly: ${assemblyProgress}/${status.assembly.totalChunks} chunks completed`);
          lastAssemblyChunks = assemblyProgress;
        }
      }

      if (status.status === 'complete') {
        return status;
      }

      if (status.status === 'error') {
        throw new Error(`Workload failed: ${status.error || 'Unknown error'}`);
      }

      await sleep(intervalMs);
    } catch (error) {
      if (error.message.includes('Workload failed:')) {
        throw error; // Re-throw actual failures
      }
      console.warn(`⚠️ Status check failed: ${error.message}, retrying...`);
      await sleep(intervalMs);
    }
  }
}

// Validate neural network data file
async function validateNeuralNetworkFile(filePath, expectedInputSize, expectedHiddenSize, expectedOutputSize, expectedBatchSize) {
  try {
    const stats = await fs.stat(filePath);
    const fileSize = stats.size;

    // Calculate expected file size
    const headerSize = 4;
    const inputDataSize = expectedBatchSize * expectedInputSize * 4;
    const weights1Size = expectedHiddenSize * expectedInputSize * 4;
    const weights2Size = expectedOutputSize * expectedHiddenSize * 4;
    const biases1Size = expectedHiddenSize * 4;
    const biases2Size = expectedOutputSize * 4;
    const expectedSize = headerSize + inputDataSize + weights1Size + weights2Size + biases1Size + biases2Size;

    if (fileSize !== expectedSize) {
      return {
        valid: false,
        error: `File size mismatch: expected ${expectedSize} bytes, got ${fileSize} bytes`
      };
    }

    // Read and validate header
    const fileHandle = await fs.open(filePath, 'r');
    const headerBuffer = Buffer.alloc(4);
    await fileHandle.read(headerBuffer, 0, 4, 0);
    const fileInputSize = headerBuffer.readUInt32LE(0);
    await fileHandle.close();

    if (fileInputSize !== expectedInputSize) {
      return {
        valid: false,
        error: `Input size mismatch: file says ${fileInputSize}, expected ${expectedInputSize}`
      };
    }

    return { valid: true, fileSize };
  } catch (error) {
    return {
      valid: false,
      error: `File validation failed: ${error.message}`
    };
  }
}

(async () => {
  console.log(`🧠 Simple Neural Network Test`);
  console.log(`   Framework: ${framework.toUpperCase()}`);
  console.log(`   Network Architecture: ${inputSize} → ${hiddenSize} → ${outputSize}`);
  console.log(`   Batch Size: ${batchSize} samples`);
  console.log(`   Chunk Size: ${chunkSize} samples per chunk`);
  console.log(`   Total Chunks: ${Math.ceil(batchSize / chunkSize)}`);
  console.log(`   API Base: ${apiBase}`);
      console.log(`   Streaming Mode: ${streamingMode ? 'ENABLED (preferred)' : 'DISABLED (batch mode)'}`);

  // Calculate memory requirements
  const inputBytes = batchSize * inputSize * 4;
  const hiddenBytes = batchSize * hiddenSize * 4;
  const outputBytes = batchSize * outputSize * 4;
  const weights1Bytes = hiddenSize * inputSize * 4;
  const weights2Bytes = outputSize * hiddenSize * 4;
  const biases1Bytes = hiddenSize * 4;
  const biases2Bytes = outputSize * 4;
  const totalBytes = inputBytes + hiddenBytes + outputBytes + weights1Bytes + weights2Bytes + biases1Bytes + biases2Bytes;

  console.log(`\n💾 Memory Analysis:`);
  console.log(`   Input Data: ${Math.round(inputBytes/1024/1024)}MB`);
  console.log(`   Hidden Layer: ${Math.round(hiddenBytes/1024/1024)}MB`);
  console.log(`   Output Layer: ${Math.round(outputBytes/1024/1024)}MB`);
  console.log(`   Weights Layer 1: ${Math.round(weights1Bytes/1024/1024)}MB`);
  console.log(`   Weights Layer 2: ${Math.round(weights2Bytes/1024/1024)}MB`);
  console.log(`   Biases: ${Math.round((biases1Bytes + biases2Bytes)/1024)}KB`);
  console.log(`   Total: ${Math.round(totalBytes/1024/1024)}MB`);

  if (insecureFlag) {
    console.log('\n⚠️ Insecure mode enabled: NODE_TLS_REJECT_UNAUTHORIZED=0 (accepting self-signed certs)');
  }

  try {
    await ensureServerReachable(apiBase, 4);
  } catch (err) {
    console.error('\n❌ Server preflight failed:', err.message || err);
    console.error('Tip: if your server uses a self-signed cert, run with:');
    console.error('  NODE_TLS_REJECT_UNAUTHORIZED=0 node test-neural-network.mjs --insecure ...');
    process.exit(2);
  }

  try {
    // Check framework support
    console.log(`\n🔍 Checking ${framework.toUpperCase()} framework support...`);
    const frameworkCheck = await checkFrameworkSupport(framework, apiBase);

    if (!frameworkCheck.supported) {
      console.error(`❌ Framework ${framework.toUpperCase()} not supported: ${frameworkCheck.reason}`);
      if (frameworkCheck.stats) {
        console.error(`   Available clients: ${frameworkCheck.stats.availableClients}`);
        console.error(`   Active workloads: ${frameworkCheck.stats.activeWorkloads}`);
      }
      console.error(`\nTips:`);
      console.error(`   - Make sure clients supporting ${framework.toUpperCase()} are connected`);
      console.error(`   - For WebGL, ensure clients have WebGL2 support`);
      console.error(`   - Try different framework with --framework <name>`);
      process.exit(3);
    } else {
      console.log(`✅ Framework ${framework.toUpperCase()} supported`);
      if (frameworkCheck.stats) {
        console.log(`   Available clients: ${frameworkCheck.stats.availableClients}`);
        console.log(`   Active workloads: ${frameworkCheck.stats.activeWorkloads}`);
        console.log(`   Completed workloads: ${frameworkCheck.stats.completedWorkloads}`);
      }
    }

    // Validate input file if provided
    if (inputPath) {
      console.log(`\n📁 Validating input file: ${inputPath}`);
      const validation = await validateNeuralNetworkFile(inputPath, inputSize, hiddenSize, outputSize, batchSize);

      if (!validation.valid) {
        console.error(`❌ File validation failed: ${validation.error}`);
        process.exit(4);
      }

      console.log(`✅ File validation passed: ${Math.round(validation.fileSize/1024/1024)}MB`);
    } else {
      console.log(`\n⚠️ No input file provided. You can generate one with:`);
      console.log(`   node generate-neural-network-data.mjs --input-size ${inputSize} --hidden-size ${hiddenSize} --output-size ${outputSize} --batch-size ${batchSize}`);
    }

    // Create workload definition
    console.log('\n📋 Creating neural network workload definition...');
    const payload = {
      label: `${framework.toUpperCase()} ${streamingMode ? 'Streaming' : 'Batch'} Neural Network ${inputSize}→${hiddenSize}→${outputSize} test (batch: ${batchSize}, chunk: ${chunkSize})`,
      framework: framework,
      chunkingStrategy: getChunkingStrategyForFramework(framework),
      assemblyStrategy: getAssemblyStrategyForFramework(framework),
      metadata: {
        inputSize,
        hiddenSize,
        outputSize,
        batchSize,
        chunkSize,
        framework: framework,
        testType: 'simple_neural_network',
        streamingMode: streamingMode
      },
      outputSizes: [batchSize * outputSize * 4]
    };

    const workloadInfo = await postJSON(`${apiBase}/api/workloads/advanced`, payload);
    const workloadId = workloadInfo.id;
    if (!workloadId) throw new Error('Server did not return workload ID');

    console.log(`✅ Workload created: ${workloadId}`);
    console.log(`   Status: ${workloadInfo.status}`);
    console.log(`   Framework: ${framework.toUpperCase()}`);
    console.log(`   Streaming: ${streamingMode ? 'YES' : 'NO'}`);
    console.log(`   Requires file upload: ${workloadInfo.requiresFileUpload}`);

    if (workloadInfo.requiresFileUpload) {
      console.log(`   Message: ${workloadInfo.message}`);
    }

    // Upload input file if required
    if (workloadInfo.status === 'awaiting_input' || workloadInfo.requiresFileUpload) {
      if (!inputPath) {
        throw new Error('Input file required but not provided. Use --input <file> or generate data first.');
      }

      console.log(`\n📤 Uploading neural network data file: ${inputPath}`);
      const fileBuffer = await fs.readFile(inputPath);
      console.log(`   File size: ${fileBuffer.length} bytes`);

      console.log('📤 Uploading input data...');
      const uploadResp = await postMultipart(`${apiBase}/api/workloads/${workloadId}/inputs`, [
        { name: 'neural_network_data', buffer: fileBuffer, filename: path.basename(inputPath) }
      ]);

      console.log('✅ Upload successful:');
      console.log(`   Files: ${uploadResp.files.length}`);
      console.log(`   Total bytes: ${uploadResp.totalBytes}`);
      console.log(`   Status: ${uploadResp.status}`);
      console.log(`   Message: ${uploadResp.message}`);
    } else {
      console.log('ℹ️ No file upload required (using inline data)');
    }

    // Start computation
    console.log(`\n🚀 Starting ${streamingMode ? 'streaming' : 'batch'} ${framework.toUpperCase()} neural network computation...`);
    const startPayload = streamingMode ? { streamingMode: true } : {};
    const startResp = await postJSON(`${apiBase}/api/workloads/${workloadId}/compute-start`, startPayload);
    console.log(`✅ Computation started successfully`);
    console.log(`   Status: ${startResp.status}`);
    console.log(`   Framework: ${framework.toUpperCase()}`);
    console.log(`   Total chunks: ${startResp.totalChunks}`);
    console.log(`   Mode: ${streamingMode ? 'STREAMING' : 'BATCH'}`);
    console.log(`   Message: ${startResp.message}`);

    // Poll status until completion
    console.log(`\n⏳ Waiting for ${streamingMode ? 'streaming' : 'batch'} computation to complete...`);
    const startTime = Date.now();
    const finalInfo = await pollStatus(workloadId, pollInterval);
    const totalTime = (Date.now() - startTime) / 1000;

    console.log(`\n🎉 Neural network workload completed successfully in ${totalTime.toFixed(2)}s!`);
    console.log('📊 Final status:', JSON.stringify(finalInfo, null, 2));

    // Try to download result
    try {
      console.log('\n📥 Attempting to download neural network output...');
      const res = await fetch(`${apiBase}/api/workloads/${workloadId}/download/final`);
      if (res.ok) {
        const resultBuffer = await res.arrayBuffer();
        const resultFloats = new Float32Array(resultBuffer);
        console.log(`📊 Result downloaded: ${resultBuffer.byteLength} bytes`);
        console.log(`   Elements: ${resultFloats.length} float32 values`);
        console.log(`   Expected: ${batchSize * outputSize} elements for ${batchSize}×${outputSize} output`);
        console.log(`🔒 SHA256: ${crypto.createHash('sha256').update(Buffer.from(resultBuffer)).digest('hex')}`);

        // Basic validation
        if (resultFloats.length === batchSize * outputSize) {
          console.log('✅ Result size validation passed');

          // Show some sample outputs
          console.log('\n📊 Sample Outputs (first 5 samples, first 3 outputs each):');
          for (let sample = 0; sample < Math.min(5, batchSize); sample++) {
            const sampleStart = sample * outputSize;
            const outputs = resultFloats.slice(sampleStart, sampleStart + Math.min(3, outputSize));
            console.log(`   Sample ${sample}: [${outputs.map(x => x.toFixed(4)).join(', ')}${outputSize > 3 ? '...' : ''}]`);
          }
        } else {
          console.warn(`⚠️ Result size mismatch: got ${resultFloats.length}, expected ${batchSize * outputSize}`);
        }

      } else if (res.status === 404) {
        console.log('📋 Result stored in memory (no file download available)');

        // Try to get result via API
        const statusResp = await getJSON(`${apiBase}/api/status`);
        const workload = statusResp.workloads.find(w => w.id === workloadId);
        if (workload) {
          console.log('📊 Workload info from status API:', {
            id: workload.id,
            status: workload.status,
            completedAt: workload.completedAt
          });
        }
      } else {
        console.warn(`⚠️ Download failed: HTTP ${res.status}`);
      }
    } catch (e) {
      console.warn('⚠️ Could not download result:', e.message);
    }

    console.log('\n🎉 Neural network test completed successfully!');
    console.log(`\n📈 Performance Summary:`);
    console.log(`   Framework: ${framework.toUpperCase()}`);
    console.log(`   Mode: ${streamingMode ? 'STREAMING' : 'BATCH'}`);
    console.log(`   Network: ${inputSize} → ${hiddenSize} → ${outputSize}`);
    console.log(`   Batch size: ${batchSize} samples`);
    console.log(`   Chunk size: ${chunkSize} samples`);
    console.log(`   Total chunks: ${Math.ceil(batchSize / chunkSize)}`);
    console.log(`   Memory usage: ${Math.round(totalBytes/1024/1024)}MB`);
    console.log(`   Total time: ${totalTime.toFixed(2)}s`);
    console.log(`   Throughput: ${(batchSize / totalTime).toFixed(2)} samples/sec`);

  } catch (err) {
    console.error('\n❌ Neural network test failed:', err.message || err);

    // Additional debugging info
    if (err.message.includes('HTTP 400')) {
      console.error('\n🔍 Debugging tips:');
      console.error('   - Check that the server is running with the updated code');
      console.error(`   - Verify the ${framework} framework is properly supported`);
      console.error('   - Ensure the simple_neural_network strategy is properly registered');
      console.error('   - Verify the neural network assembly strategy is available');
      console.error('   - Check that the input file format matches the expected structure');
    }

    if (err.message.includes('Framework') && err.message.includes('not supported')) {
      console.error('\n💡 Framework troubleshooting:');
      console.error(`   - For WebGL: ensure clients have WebGL2 and proper extensions`);
      console.error(`   - For CUDA: ensure CUDA runtime and clients are available`);
      console.error(`   - For OpenCL: ensure OpenCL runtime and clients are available`);
      console.error(`   - Try different framework with --framework <name>`);
    }

    if (err.message.includes('File validation failed')) {
      console.error('\n📁 File validation troubleshooting:');
      console.error('   - Use generate-neural-network-data.mjs to create properly formatted files');
      console.error('   - Ensure file dimensions match the specified network architecture');
      console.error('   - Check that the file is not corrupted');
    }

    if (streamingMode && err.message.includes('streaming')) {
      console.error('\n🌊 Streaming mode troubleshooting:');
      console.error('   - Ensure server supports streaming chunking API');
      console.error('   - Check if strategy supports createChunkDescriptorsStreaming');
      console.error('   - Verify streaming assembly is properly initialized');
      console.error('   - Try without --streaming flag for batch mode');
    }

    if (err.cause && err.cause.socket) {
      const s = err.cause.socket;
      console.error('Socket info:', {
        localAddress: s.localAddress, localPort: s.localPort,
        remoteAddress: s.remoteAddress, remotePort: s.remotePort
      });
    }
    process.exit(1);
  }
})();
