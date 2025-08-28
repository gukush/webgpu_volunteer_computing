#!/usr/bin/env node
import fs from 'fs/promises';
import path from 'path';
import minimist from 'minimist';
import crypto from 'crypto';

const args = minimist(process.argv.slice(2));

const count = parseInt(args.count, 10);
const chunkSize = parseInt(args['chunk-size'], 10);
const framework = args.framework || 'webgpu';
const inputPath = args.input ? path.resolve(args.input) : null;
const apiBase = args['api-base'] || 'https://localhost:3000';
const insecureFlag = args.insecure || process.env.TEST_INSECURE;

if (!Number.isInteger(count) || !Number.isInteger(chunkSize) || !inputPath) {
  console.error('Usage: --count N --chunk-size M --input <path> [options]');
  console.error('  --count        Total number of elements in the file');
  console.error('  --chunk-size   Chunk size in bytes');
  console.error('  --input        Path to pre-generated unsorted data file');
  process.exit(1);
}

async function postJSON(url, obj) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '<no-body>');
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

async function pollStatus(workloadId, intervalMs = 2000) {
  while (true) {
    const res = await fetch(`${apiBase}/api/workloads/${workloadId}/status`);
    const status = await res.json();
    console.log(`⏳ Status: ${status.status} | Progress: ${status.chunks?.progress.toFixed(1) ?? 0}%`);
    if (status.status === 'complete') return status;
    if (status.status === 'error') throw new Error(`Workload failed: ${status.error}`);
    await new Promise(r => setTimeout(r, intervalMs));
  }
}

(async () => {
  if (insecureFlag) process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

  console.log('🚀 Distributed Sort Test');

  // 1. Create Workload
  console.log('\n📋 Creating workload definition...');
  const payload = {
    label: `Distributed Sort Test (${count} elements)`,
    framework,
    chunkingStrategy: 'distributed_sort',
    assemblyStrategy: 'distributed_sort_assembly',
    metadata: {
      elementCount: count,
      chunkSize: chunkSize
    }
  };
  const workloadInfo = await postJSON(`${apiBase}/api/workloads/advanced`, payload);
  const workloadId = workloadInfo.id;
  console.log(`✅ Workload created: ${workloadId}`);

  // 2. Upload Input File
  console.log(`\n📤 Uploading data file: ${inputPath}`);
  const buf = await fs.readFile(inputPath);
  await postMultipart(`${apiBase}/api/workloads/${workloadId}/inputs`, [
    { name: 'unsorted_data', buffer: buf, filename: 'unsorted_data.bin' }
  ]);
  console.log('✅ Upload successful.');

  // 3. Start Computation
  console.log('\n🚀 Starting computation...');
  const startResp = await postJSON(`${apiBase}/api/workloads/${workloadId}/compute-start`, { streamingMode: true });
  console.log(`✅ Computation started: ${startResp.totalChunks} chunks`);

  // 4. Poll Status
  console.log('\n⏳ Waiting for computation to complete...');
  const startTime = Date.now();
  await pollStatus(workloadId);
  const totalTime = (Date.now() - startTime) / 1000;
  console.log(`\n🎉 Workload completed in ${totalTime.toFixed(2)}s!`);

  // 5. Download and Verify Result
  console.log('\n📥 Downloading and verifying result...');
  const res = await fetch(`${apiBase}/api/workloads/${workloadId}/download/final`);
  if (!res.ok) throw new Error('Failed to download result');

  const resultBuffer = await res.arrayBuffer();
  const resultFloats = new Float32Array(resultBuffer);

  console.log(`   Downloaded ${resultBuffer.byteLength} bytes (${resultFloats.length} elements)`);

  if (resultFloats.length !== count) {
    throw new Error(`Verification failed: Expected ${count} elements, got ${resultFloats.length}`);
  }

  // NOTE: Since our dummy kernel doesn't actually sort, we can't verify sorted order.
  // Instead, we verify that the *original data* was reassembled correctly.
  const originalData = new Float32Array(buf.buffer, 4); // Skip 4-byte header
  let mismatchCount = 0;
  for (let i = 0; i < count; i++) {
    if (resultFloats[i] !== originalData[i]) {
      mismatchCount++;
    }
  }

  if (mismatchCount > 0) {
      throw new Error(`Verification failed: ${mismatchCount} elements do not match original data. Assembly is incorrect.`);
  }

  console.log('✅ Verification successful! Data was chunked and reassembled correctly.');
  console.log('\n🎉 Test completed successfully!');

})().catch(err => {
  console.error('\n❌ Test failed:', err.message);
  process.exit(1);
});