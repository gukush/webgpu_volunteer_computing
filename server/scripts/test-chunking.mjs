#!/usr/bin/env node
// ENHANCED: app/scripts/test-chunking.mjs - Complete version with streaming support and smart memory management

import fs from 'fs/promises';
import path from 'path';
import net from 'net';
import { fileURLToPath } from 'url';
import minimist from 'minimist';
import crypto from 'crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = minimist(process.argv.slice(2));

const size = parseInt(args.size, 10);
const blockSize = parseInt(args['block-size'], 10);
const framework = args.framework || 'webgpu';
const inputPath = args.input ? path.resolve(args.input) : null;
const apiBase = args['api-base'] || process.env.API_BASE || 'https://localhost:3000';
const insecureFlag = args.insecure || process.env.TEST_INSECURE;
const pollInterval = Number(args.interval || process.env.TEST_POLL_INTERVAL || 2000);

// ENHANCED: New options for streaming and memory management
const streamingMode = args.streaming || process.env.TEST_STREAMING || false;
const memoryThreshold = parseInt(args['memory-threshold'], 10) || 512 * 1024 * 1024; // 512MB default
const batchSize = parseInt(args['batch-size'], 10) || 10; // Chunk creation batch size

const SUPPORTED_FRAMEWORKS = ['webgpu', 'webgl', 'cuda', 'opencl', 'vulkan'];

// If the user requested insecure, set TLS reject env early so node fetch accepts self-signed certs.
if (insecureFlag) {
  try { process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'; } catch (e) { /* no-op */ }
}

if (!Number.isInteger(size) || !Number.isInteger(blockSize)) {
  console.error('Usage: --size N --block-size M [options]');
  console.error('');
  console.error('Required:');
  console.error('  --size         Matrix size (must be divisible by block-size)');
  console.error('  --block-size   Block size for matrix subdivision');
  console.error('');
  console.error('Options:');
  console.error('  --framework      GPU framework (webgpu, webgl, cuda, opencl, vulkan) [default: webgpu]');
  console.error('  --input          Path to pre-generated matrix file');
  console.error('  --streaming      Enable streaming chunk creation and assembly');
  console.error('  --memory-threshold  Memory threshold in bytes [default: 512MB]');
  console.error('  --batch-size     Chunk creation batch size [default: 10]');
  console.error('  --api-base       API server URL [default: https://localhost:3000]');
  console.error('  --insecure       Accept self-signed certificates');
  console.error('  --interval       Status polling interval in ms [default: 2000]');
  console.error('');
  console.error('Examples:');
  console.error('  # Small matrix, batch mode');
  console.error('  node test-chunking.mjs --size 512 --block-size 32');
  console.error('');
  console.error('  # Large matrix, streaming mode with 1GB threshold');
  console.error('  node test-chunking.mjs --size 2048 --block-size 128 --streaming --memory-threshold 1073741824');
  console.error('');
  console.error('  # WebGL with streaming');
  console.error('  node test-chunking.mjs --size 1024 --block-size 64 --framework webgl --streaming');
  process.exit(1);
}

if (size % blockSize !== 0) {
  console.error(`--size ${size} must be divisible by --block-size ${blockSize}`);
  process.exit(1);
}

// NEW: Validate framework
if (!SUPPORTED_FRAMEWORKS.includes(framework)) {
  console.error(`Unsupported framework: ${framework}`);
  console.error(`Supported frameworks: ${SUPPORTED_FRAMEWORKS.join(', ')}`);
  process.exit(1);
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
    const txt = await res.text().catch(()=>'<no-body>');
    throw new Error(`Upload failed: HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

function packCombinedMatrices(size, A, B) {
  const buf = Buffer.alloc(4 + size * size * 2 * 4);
  buf.writeUInt32LE(size, 0);
  let off = 4;
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) { buf.writeFloatLE(A[i][j], off); off += 4; }
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) { buf.writeFloatLE(B[i][j], off); off += 4; }
  return buf;
}

// NEW: Check framework availability on server
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

// ENHANCED: Memory strategy estimation
function estimateMemoryStrategy(matrixSize, memoryThreshold) {
  const matrixBytes = matrixSize * matrixSize * 4; // Float32
  const totalInputBytes = matrixBytes * 2; // A and B matrices
  const outputBytes = matrixBytes; // Result matrix C
  const totalBytes = totalInputBytes + outputBytes;

  const strategy = {
    matrixBytes,
    totalInputBytes,
    outputBytes,
    totalBytes,
    canFitAll: totalBytes <= memoryThreshold,
    canFitOutput: outputBytes <= memoryThreshold,
    canFitOneInput: matrixBytes <= memoryThreshold,
    recommendedStrategy: 'unknown'
  };

  if (strategy.canFitAll) {
    strategy.recommendedStrategy = 'all_memory';
  } else if (strategy.canFitOutput) {
    strategy.recommendedStrategy = 'output_memory_input_disk';
  } else {
    strategy.recommendedStrategy = 'all_disk';
  }

  return strategy;
}

// NEW: Get framework-specific chunking strategy
function getChunkingStrategyForFramework(framework) {
  const strategyMap = {
    'webgpu': 'block_matrix',
    'webgl': 'block_matrix', // Same strategy, different shader
    'cuda': 'block_matrix',
    'opencl': 'block_matrix',
    'vulkan': 'block_matrix'
  };

  return strategyMap[framework] || 'block_matrix';
}

// NEW: Get framework-specific assembly strategy
function getAssemblyStrategyForFramework(framework) {
  const strategyMap = {
    'webgpu': 'block_matrix_assembly',
    'webgl': 'block_matrix_assembly',
    'cuda': 'block_matrix_assembly',
    'opencl': 'block_matrix_assembly',
    'vulkan': 'block_matrix_assembly'
  };

  return strategyMap[framework] || 'block_matrix_assembly';
}

// ENHANCED: Streaming status polling with assembly progress
async function pollStatus(workloadId, intervalMs = 2000) {
  let lastProgress = -1;
  let lastAssemblyBlocks = -1;
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

      // ENHANCED: Show streaming-specific progress
      if (status.streaming) {
        const dispatchedChunks = status.streaming.dispatchedChunks || 0;
        if (dispatchedChunks !== lastDispatchedChunks) {
          console.log(`   Dispatched: ${dispatchedChunks} chunks sent to clients`);
          lastDispatchedChunks = dispatchedChunks;
        }
      }

      // ENHANCED: Show assembly progress if available
      if (status.assembly) {
        const assemblyProgress = status.assembly.completedBlocks || 0;
        if (assemblyProgress !== lastAssemblyBlocks) {
          console.log(`   Assembly: ${assemblyProgress}/${status.assembly.totalBlocks} blocks completed`);
          lastAssemblyBlocks = assemblyProgress;
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

(async () => {
  console.log(`🚀 Enhanced Block Matrix Test`);
  console.log(`   Framework: ${framework.toUpperCase()}`);
  console.log(`   Matrix Size: ${size}×${size}`);
  console.log(`   Block Size: ${blockSize}×${blockSize}`);
  console.log(`   API Base: ${apiBase}`);
  console.log(`   Streaming Mode: ${streamingMode ? 'ENABLED' : 'DISABLED'}`);

  // ENHANCED: Memory strategy analysis
  const memoryStrategy = estimateMemoryStrategy(size, memoryThreshold);
  console.log(`\n💾 Memory Analysis:`);
  console.log(`   Matrix Size: ${Math.round(memoryStrategy.matrixBytes/1024/1024)}MB each`);
  console.log(`   Total Input: ${Math.round(memoryStrategy.totalInputBytes/1024/1024)}MB`);
  console.log(`   Output: ${Math.round(memoryStrategy.outputBytes/1024/1024)}MB`);
  console.log(`   Total: ${Math.round(memoryStrategy.totalBytes/1024/1024)}MB`);
  console.log(`   Threshold: ${Math.round(memoryThreshold/1024/1024)}MB`);
  console.log(`   Strategy: ${memoryStrategy.recommendedStrategy}`);

  if (insecureFlag) {
    console.log('\n⚠️ Insecure mode enabled: NODE_TLS_REJECT_UNAUTHORIZED=0 (accepting self-signed certs)');
  }

  try {
    await ensureServerReachable(apiBase, 4);
  } catch (err) {
    console.error('\n❌ Server preflight failed:', err.message || err);
    console.error('Tip: if your server uses a self-signed cert, run with:');
    console.error('  NODE_TLS_REJECT_UNAUTHORIZED=0 node app/scripts/test-chunking.mjs --insecure ...');
    process.exit(2);
  }

  try {
    // NEW: Check framework support
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
      console.error(`   - Try --framework webgpu as fallback`);
      process.exit(3);
    } else {
      console.log(`✅ Framework ${framework.toUpperCase()} supported`);
      if (frameworkCheck.stats) {
        console.log(`   Available clients: ${frameworkCheck.stats.availableClients}`);
        console.log(`   Active workloads: ${frameworkCheck.stats.activeWorkloads}`);
        console.log(`   Completed workloads: ${frameworkCheck.stats.completedWorkloads}`);
      }
    }

    // STEP 1: Create workload (metadata only) - NEW TWO-STEP FLOW with framework selection
    console.log('\n📋 Creating workload definition...');
    const payload = {
      label: `${framework.toUpperCase()} ${streamingMode ? 'Streaming' : 'Batch'} Block Matrix ${size}×${size} test (${blockSize}×${blockSize} blocks)`,
      framework: framework, // NEW: Use selected framework
      chunkingStrategy: getChunkingStrategyForFramework(framework), // NEW: Framework-specific strategy
      assemblyStrategy: getAssemblyStrategyForFramework(framework), // NEW: Framework-specific assembly
      metadata: {
        matrixSize: size,
        blockSize,
        framework: framework, // Additional metadata
        testType: 'block_matrix_multiplication',
        memoryThreshold: memoryThreshold,
        streamingMode: streamingMode,
        batchSize: batchSize
      },
      outputSizes: [ size * size * 4 ]
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

    // STEP 2: Upload input file (only if required)
    if (workloadInfo.status === 'awaiting_input' || workloadInfo.requiresFileUpload) {
      let buf;
      if (inputPath) {
        console.log(`\n📤 Uploading provided file: ${inputPath}`);
        buf = await fs.readFile(inputPath);
        console.log(`   File size: ${buf.length} bytes`);

        // Validate file format
        if (buf.length < 4) {
          throw new Error('Input file too small (missing size header)');
        }
        const fileSizeHeader = buf.readUInt32LE(0);
        const expectedSize = 4 + size * size * 2 * 4;
        if (buf.length !== expectedSize) {
          throw new Error(`File size mismatch: expected ${expectedSize} bytes, got ${buf.length} bytes`);
        }
        if (fileSizeHeader !== size) {
          throw new Error(`Size header mismatch: file says ${fileSizeHeader}, expected ${size}`);
        }
        console.log(`   ✅ File validation passed: ${size}×${size} matrices`);
      } else {
        console.log(`\n⚙️ Generating random matrices for ${framework.toUpperCase()} computation...`);
        const A = Array.from({ length: size }, () => Array.from({ length: size }, () => Math.random()));
        const B = Array.from({ length: size }, () => Array.from({ length: size }, () => Math.random()));
        buf = packCombinedMatrices(size, A, B);
        console.log(`   Generated: ${buf.length} bytes`);
      }

      console.log('📤 Uploading input data...');
      const uploadResp = await postMultipart(`${apiBase}/api/workloads/${workloadId}/inputs`, [
        { name: 'combined_matrix', buffer: buf, filename: `matrix_${size}x${size}_${framework}.bin` }
      ]);

      console.log('✅ Upload successful:');
      console.log(`   Files: ${uploadResp.files.length}`);
      console.log(`   Total bytes: ${uploadResp.totalBytes}`);
      console.log(`   Status: ${uploadResp.status}`);
      console.log(`   Message: ${uploadResp.message}`);
    } else {
      console.log('ℹ️ No file upload required (using inline data)');
    }

    // STEP 3: Start computation
    console.log(`\n🚀 Starting ${streamingMode ? 'streaming' : 'batch'} ${framework.toUpperCase()} computation...`);
    const startPayload = streamingMode ? { streamingMode: true, batchSize } : {};
    const startResp = await postJSON(`${apiBase}/api/workloads/${workloadId}/compute-start`, startPayload);
    console.log(`✅ Computation started successfully`);
    console.log(`   Status: ${startResp.status}`);
    console.log(`   Framework: ${framework.toUpperCase()}`);
    console.log(`   Total chunks: ${startResp.totalChunks}`);
    console.log(`   Mode: ${streamingMode ? 'STREAMING' : 'BATCH'}`);
    console.log(`   Message: ${startResp.message}`);

    // STEP 4: Poll status until completion
    console.log(`\n⏳ Waiting for ${streamingMode ? 'streaming' : 'batch'} computation to complete...`);
    const startTime = Date.now();
    const finalInfo = await pollStatus(workloadId, pollInterval);
    const totalTime = (Date.now() - startTime) / 1000;

    console.log(`\n🎉 Workload completed successfully in ${totalTime.toFixed(2)}s!`);
    console.log('📊 Final status:', JSON.stringify(finalInfo, null, 2));

    // STEP 5: Try to download result
    try {
      console.log('\n📥 Attempting to download result...');
      const res = await fetch(`${apiBase}/api/workloads/${workloadId}/download/final`);
      if (res.ok) {
        const resultBuffer = await res.arrayBuffer();
        const resultFloats = new Float32Array(resultBuffer);
        console.log(`📊 Result downloaded: ${resultBuffer.byteLength} bytes`);
        console.log(`   Elements: ${resultFloats.length} float32 values`);
        console.log(`   Expected: ${size * size} elements for ${size}×${size} result matrix`);
        console.log(`🔒 SHA256: ${crypto.createHash('sha256').update(Buffer.from(resultBuffer)).digest('hex')}`);

        // Basic validation
        if (resultFloats.length === size * size) {
          console.log('✅ Result size validation passed');
        } else {
          console.warn(`⚠️ Result size mismatch: got ${resultFloats.length}, expected ${size * size}`);
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

    console.log('\n🎉 Test completed successfully!');
    console.log(`\n📈 Performance Summary:`);
    console.log(`   Framework: ${framework.toUpperCase()}`);
    console.log(`   Mode: ${streamingMode ? 'STREAMING' : 'BATCH'}`);
    console.log(`   Matrix size: ${size}×${size} (${Math.round(memoryStrategy.matrixBytes/1024/1024)}MB each)`);
    console.log(`   Block size: ${blockSize}×${blockSize}`);
    console.log(`   Memory strategy: ${memoryStrategy.recommendedStrategy}`);
    console.log(`   Expected chunks: ${Math.pow(Math.floor(size / blockSize), 3)}`);
    console.log(`   Actual chunks: ${startResp.totalChunks}`);
    console.log(`   Total time: ${totalTime.toFixed(2)}s`);
    console.log(`   Throughput: ${(startResp.totalChunks / totalTime).toFixed(2)} chunks/sec`);

  } catch (err) {
    console.error('\n❌ Test failed:', err.message || err);

    // Additional debugging info
    if (err.message.includes('HTTP 400')) {
      console.error('\n🔍 Debugging tips:');
      console.error('   - Check that the server is running with the updated code');
      console.error(`   - Verify the ${framework} framework is properly supported`);
      console.error('   - Ensure the block_matrix strategy is properly registered');
      console.error('   - Verify the two-step flow is implemented correctly');
      console.error('   - Check enhanced chunking system is enabled');
    }

    if (err.message.includes('Framework') && err.message.includes('not supported')) {
      console.error('\n💡 Framework troubleshooting:');
      console.error(`   - For WebGL: ensure clients have WebGL2 and proper extensions`);
      console.error(`   - For CUDA: ensure CUDA runtime and clients are available`);
      console.error(`   - For OpenCL: ensure OpenCL runtime and clients are available`);
      console.error(`   - Try different framework with --framework <name>`);
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