#!/usr/bin/env node
// Robust Transformer block test with proper intermediate result handling

import net from 'net';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------- CLI ----------
const args = Object.fromEntries(
  process.argv.slice(2).map(s => {
    const m = s.match(/^--([^=]+)=(.*)$/);
    return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
  })
);

const API = args.api || process.env.API_BASE || 'https://localhost:3000';
const FRAMEWORK = (args.framework || 'webgpu').toLowerCase();
const SEQ = parseInt(args.seq || '128', 10);
const DMODEL = parseInt(args.dmodel || '512', 10);
const HEADS = parseInt(args.heads || '8', 10);
const DFF = parseInt(args.dff || '2048', 10);
const STREAMING = String(args.streaming || 'false').toLowerCase() === 'true';
const TIMEOUT_MS = parseInt(args.timeout || '600000', 10);
const POLL_INTERVAL = parseInt(args.interval || '1500', 10);
const INSECURE = args.insecure || process.env.TEST_INSECURE;

// NEW: Intermediate result management
const SAVE_INTERMEDIATES = args['save-intermediates'] !== false; // Default true
const RESUME_FROM = args['resume-from'] || null; // Stage to resume from
const INTERMEDIATES_DIR = args['intermediates-dir'] || path.join(__dirname, 'transformer-intermediates');

if (INSECURE) { try { process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'; } catch {} }

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- Intermediate result management ----------
async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
  }
}

async function saveIntermediateResult(stageName, workloadId, resultB64) {
  if (!SAVE_INTERMEDIATES) return;

  await ensureDir(INTERMEDIATES_DIR);

  const metadata = {
    stageName,
    workloadId,
    timestamp: Date.now(),
    seq: SEQ,
    dModel: DMODEL,
    heads: HEADS,
    dff: DFF,
    framework: FRAMEWORK,
    streaming: STREAMING,
    sha256: crypto.createHash('sha256').update(Buffer.from(resultB64, 'base64')).digest('hex')
  };

  const fileName = `${stageName.toLowerCase().replace(/\s+/g, '_')}.json`;
  const filePath = path.join(INTERMEDIATES_DIR, fileName);

  await fs.writeFile(filePath, JSON.stringify({
    metadata,
    result: resultB64
  }, null, 2));

  console.log(`Saved intermediate result: ${filePath}`);
  console.log(`  Size: ${Buffer.from(resultB64, 'base64').length} bytes`);
  console.log(`  SHA256: ${metadata.sha256.slice(0, 16)}...`);
}

async function loadIntermediateResult(stageName) {
  if (!SAVE_INTERMEDIATES) return null;

  const fileName = `${stageName.toLowerCase().replace(/\s+/g, '_')}.json`;
  const filePath = path.join(INTERMEDIATES_DIR, fileName);

  try {
    const data = await fs.readFile(filePath, 'utf8');
    const { metadata, result } = JSON.parse(data);

    // Validate metadata matches current run
    if (metadata.seq !== SEQ || metadata.dModel !== DMODEL ||
        metadata.heads !== HEADS || metadata.dff !== DFF) {
      console.warn(`Parameter mismatch in ${fileName}, ignoring cached result`);
      return null;
    }

    // Verify result integrity
    const currentHash = crypto.createHash('sha256').update(Buffer.from(result, 'base64')).digest('hex');
    if (currentHash !== metadata.sha256) {
      console.warn(`Integrity check failed for ${fileName}, ignoring cached result`);
      return null;
    }

    console.log(`Loaded intermediate result from: ${filePath}`);
    console.log(`  Original workload: ${metadata.workloadId}`);
    console.log(`  Timestamp: ${new Date(metadata.timestamp).toISOString()}`);

    return result;
  } catch (err) {
    if (err.code !== 'ENOENT') {
      console.warn(`Failed to load intermediate result ${fileName}: ${err.message}`);
    }
    return null;
  }
}

async function verifyResultAvailable(workloadId) {
  try {
    const status = await getJSON(`${API}/api/workloads/${workloadId}/result/status`);
    return status.hasResult;
  } catch (err) {
    console.warn(`Could not verify result for ${workloadId}: ${err.message}`);
    return false;
  }
}

// ---------- Utils (same as before) ----------
function toB64Float32(arr) {
  if (!(arr instanceof Float32Array)) arr = Float32Array.from(arr);
  return Buffer.from(arr.buffer).toString('base64');
}

function fromB64Float32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function randn(n, scale = 0.02) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.random() + 1e-12, v = Math.random() + 1e-12;
    const r = Math.sqrt(-2 * Math.log(u)), t = 2 * Math.PI * v;
    out[i] = scale * r * Math.cos(t);
    if (i + 1 < n) out[i + 1] = scale * r * Math.sin(t);
  }
  return out;
}

function zeros(n) { return new Float32Array(n); }
function ones(n)  { return new Float32Array(n).fill(1); }

// ---------- Network utilities (same as before) ----------
function parseHostPort(urlString) {
  try {
    const u = new URL(urlString);
    return { host: u.hostname, port: Number(u.port) || (u.protocol === 'https:' ? 443 : 80) };
  } catch {
    const s = urlString.replace(/^https?:\/\//, '');
    const [host, port] = s.split(':');
    return { host, port: Number(port || 80) };
  }
}

function tcpConnectCheck(host, port, timeout = 2500) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host, port, timeout }, () => {
      socket.end();
      resolve(true);
    });
    socket.on('error', (err) => reject(err));
    socket.on('timeout', () => { socket.destroy(); reject(new Error('timeout')); });
  });
}

async function httpGetOk(url, timeoutMs = 2500) {
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
      await tcpConnectCheck(host, port, 2500);
      console.log('ok');
      try {
        process.stdout.write(`Checking HTTP ${baseUrl} ... `);
        const ok = await httpGetOk(baseUrl, 2500).catch(()=>false);
        if (ok) { console.log('ok'); return; }
        const alt = baseUrl.replace(/\/$/, '') + '/api/status';
        process.stdout.write(`Checking HTTP ${alt} ... `);
        const ok2 = await httpGetOk(alt, 2500).catch(()=>false);
        if (ok2) { console.log('ok'); return; }
        console.log('reachable (HTTP non-OK is fine).'); return;
      } catch (e) {
        console.log('http-check failed:', e.message); return;
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
    throw new Error(`POST ${url} -> HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const txt = await res.text().catch(()=>'<no-body>');
    throw new Error(`GET ${url} -> HTTP ${res.status} ${res.statusText}: ${txt}`);
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

// ---------- Math helpers ----------
function matmulRowMajor(A, B, M, K, N) {
  const out = new Float32Array(M*N);
  for (let i=0;i<M;i++) {
    for (let j=0;j<N;j++) {
      let s=0;
      for (let k=0;k<K;k++) s += A[i*K+k]*B[k*N+j];
      out[i*N+j]=s;
    }
  }
  return out;
}

// ---------- Strategy discovery ----------
async function discoverStrategies() {
  const data = await getJSON(`${API}/api/strategies`);
  const extractNames = (arr) => arr.map(s => typeof s === 'string' ? s : (s?.name || '')).filter(Boolean);
  const chunking = extractNames(data.available?.chunking || data.chunking || []);
  const assembly = extractNames(data.available?.assembly || data.assembly || []);

  const pick = (list, mustInclude) => {
    const terms = mustInclude.map(s => s.toLowerCase());
    return list.find(name => terms.every(t => name.toLowerCase().includes(t))) || null;
  };

  const lnChunk  = pick(chunking, ['layer','norm']) || pick(chunking, ['layernorm']) || pick(chunking, ['ln']);
  const lnAsm    = pick(assembly, ['layer','norm']) || pick(assembly, ['layernorm']) || pick(assembly, ['ln']);
  const attChunk = pick(chunking, ['attention']) && (pick(chunking, ['multi','attention']) || pick(chunking, ['attention']));
  const attAsm   = pick(assembly, ['attention']);
  const ffnChunk = pick(chunking, ['ffn']) || pick(chunking, ['feed','forward']) || pick(chunking, ['mlp']);
  const ffnAsm   = pick(assembly, ['ffn']) || pick(assembly, ['feed','forward']) || pick(assembly, ['mlp']);

  const missing = [];
  if (!lnChunk || !lnAsm) missing.push('LayerNorm');
  if (!attChunk || !attAsm) missing.push('Attention');
  if (!ffnChunk || !ffnAsm) missing.push('FFN');
  if (missing.length) {
    throw new Error(
      `could not find required strategies (${missing.join(', ')}).\n` +
      `available chunking: ${chunking.join(', ')}\n` +
      `available assembly: ${assembly.join(', ')}`
    );
  }
  return { ln: { chunk: lnChunk, asm: lnAsm }, att: { chunk: attChunk, asm: attAsm }, ffn: { chunk: ffnChunk, asm: ffnAsm } };
}

async function checkFrameworkSupport(framework) {
  try {
    const frameworks = await getJSON(`${API}/api/frameworks`);
    if (!frameworks.frameworks || !frameworks.frameworks[framework]) {
      return { supported: false, reason: `Framework ${framework} not recognized by server` };
    }
    const stats = frameworks.stats?.[framework];
    if (!stats || stats.availableClients === 0) {
      return { supported: false, reason: `No clients support ${framework}`, stats };
    }
    return { supported: true, stats };
  } catch (e) {
    console.warn(`Could not query /api/frameworks: ${e.message}`);
    return { supported: true };
  }
}

// ---------- Status polling ----------
async function pollUntilComplete(workloadId, pollMs = POLL_INTERVAL, timeoutMs = TIMEOUT_MS) {
  const t0 = Date.now();
  for (;;) {
    const s = await getJSON(`${API}/api/workloads/${workloadId}/status`);
    const dur = ((Date.now() - t0)/1000).toFixed(1);
    const prog = s?.chunks?.progress != null ? ` ${s.chunks.completed}/${s.chunks.total} (${s.chunks.progress.toFixed?.(1) ?? s.chunks.progress}%)` : '';
    const streaming = s?.streaming ? ` dispatched:${s.streaming.dispatchedChunks}` : '';
    process.stdout.write(`\r${workloadId.slice(0,8)} status=${s.status}${prog}${streaming} t=${dur}s   `);
    if (s.status === 'complete') { console.log(); return s; }
    if (s.status === 'error') throw new Error(`workload ${workloadId} failed: ${s.error || 'unknown error'}`);
    if (Date.now() - t0 > timeoutMs) { console.log(); throw new Error(`timeout waiting for workload ${workloadId}`); }
    await sleep(pollMs);
  }
}

// ---------- Enhanced workload creation with result verification ----------
async function createWorkloadWithInputs({ label, chunkingStrategy, assemblyStrategy, framework, inputData, metadata, streamingMode }) {
  const outputBytes = (metadata?.seqLength || 0) * (metadata?.dModel || 0) * 4;

  const payload = {
    label,
    chunkingStrategy,
    assemblyStrategy,
    framework,
    metadata: { ...metadata, outputSizes: [outputBytes] },
    streamingMode: !!streamingMode,
    outputSizes: [outputBytes]
  };

  console.log(`   creating workload...`);
  const workloadInfo = await postJSON(`${API}/api/workloads/advanced`, payload);
  const workloadId = workloadInfo.id;

  if (workloadInfo.status === 'awaiting_input' || workloadInfo.requiresFileUpload) {
    console.log(`   uploading inputs...`);

    const files = [];
    for (const [name, base64Data] of Object.entries(inputData)) {
      const buffer = Buffer.from(base64Data, 'base64');
      files.push({ name: name, buffer: buffer, filename: `${name}.bin` });
    }

    await postMultipart(`${API}/api/workloads/${workloadId}/inputs`, files);
    console.log(`   uploaded ${files.length} input files`);
  }

  console.log(`   starting computation...`);
  const startResult = await postJSON(`${API}/api/workloads/${workloadId}/compute-start`, {
    streamingMode: !!streamingMode
  });

  console.log(`   workload: ${workloadId}`);
  console.log(`   status: ${startResult.status}`);
  if (startResult.totalChunks) console.log(`   chunks: ${startResult.totalChunks}`);

  return workloadId;
}

async function downloadFinalWithRetry(workloadId, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // First verify the result is available
      const available = await verifyResultAvailable(workloadId);
      if (!available) {
        throw new Error(`Result not available for workload ${workloadId}`);
      }

      const res = await fetch(`${API}/api/workloads/${workloadId}/download/final`);
      if (res.ok) {
        const arr = await res.arrayBuffer();
        return Buffer.from(arr).toString('base64');
      }

      if (res.status === 404) {
        const s = await getJSON(`${API}/api/workloads/${workloadId}/status`);
        if (s.finalResultBase64) return s.finalResultBase64;
      }

      throw new Error(`result not available (HTTP ${res.status})`);
    } catch (err) {
      if (attempt === maxRetries) throw err;
      console.warn(`   download attempt ${attempt} failed: ${err.message}, retrying...`);
      await sleep(2000 * attempt);
    }
  }
}

// ---------- Tensor utilities ----------
function makeInitialInput() {
  const x = new Float32Array(SEQ * DMODEL);
  for (let i=0;i<x.length;i++) x[i] = Math.sin(i*0.01);
  return x;
}

const lnParams = () => ({ gamma: ones(DMODEL), beta: zeros(DMODEL), epsilon: 1e-5 });

function attentionWeights() {
  const mat = () => randn(DMODEL * DMODEL, 0.02);
  return { Wq: mat(), Wk: mat(), Wv: mat(), Wo: mat() };
}

function ffnWeights() {
  const W1 = randn(DMODEL * DFF, 0.02), b1 = zeros(DFF), W2 = randn(DFF * DMODEL, 0.02), b2 = zeros(DMODEL);
  return { W1, b1, W2, b2 };
}

function preprojectQKV(xRowMajor, { Wq, Wk, Wv }) {
  const Q = matmulRowMajor(xRowMajor, Wq, SEQ, DMODEL, DMODEL);
  const K = matmulRowMajor(xRowMajor, Wk, SEQ, DMODEL, DMODEL);
  const V = matmulRowMajor(xRowMajor, Wv, SEQ, DMODEL, DMODEL);
  return { Q, K, V };
}

// ---------- Main execution ----------
(async () => {
  console.log('Transformer Block (Robust HTTP polling)');
  console.log(`   API: ${API}`);
  console.log(`   Framework: ${FRAMEWORK}`);
  console.log(`   Shape: seq=${SEQ}, d_model=${DMODEL}, heads=${HEADS}, d_ff=${DFF}`);
  console.log(`   Streaming: ${STREAMING ? 'ON' : 'OFF'}`);
  console.log(`   Save intermediates: ${SAVE_INTERMEDIATES ? 'ON' : 'OFF'}`);
  if (SAVE_INTERMEDIATES) console.log(`   Intermediates dir: ${INTERMEDIATES_DIR}`);
  if (RESUME_FROM) console.log(`   Resume from: ${RESUME_FROM}`);
  if (INSECURE) console.log('   Insecure mode: NODE_TLS_REJECT_UNAUTHORIZED=0');

  try { await ensureServerReachable(API, 4); }
  catch (e) { console.error('\nServer preflight failed:', e.message || e); process.exit(2); }

  console.log(`\nChecking ${FRAMEWORK.toUpperCase()} framework support...`);
  const fw = await checkFrameworkSupport(FRAMEWORK);
  if (!fw.supported) { console.error(`Framework not available: ${fw.reason}`); process.exit(3); }
  console.log('Framework ok');

  console.log('\nDiscovering strategies...');
  const STRAT = await discoverStrategies();
  console.log('   LayerNorm:', STRAT.ln.chunk, '/', STRAT.ln.asm);
  console.log('   Attention:', STRAT.att.chunk, '/', STRAT.att.asm);
  console.log('   FFN      :', STRAT.ffn.chunk, '/', STRAT.ffn.asm);

  const timeline = [];
  const stageRun = async (name, { chunk, asm, inputData, metadata }, skipExecution = false) => {
    console.log(`\n${name}`);

    // Check for cached result first
    if (!skipExecution) {
      const cached = await loadIntermediateResult(name);
      if (cached) {
        console.log('   using cached result');
        timeline.push({ stage: name, seconds: 0, cached: true });
        return cached;
      }
    } else {
      console.log('   skipping execution (resume mode)');
      const cached = await loadIntermediateResult(name);
      if (cached) {
        timeline.push({ stage: name, seconds: 0, skipped: true });
        return cached;
      } else {
        throw new Error(`No cached result found for ${name} (required for resume)`);
      }
    }

    const t0 = Date.now();

    const wid = await createWorkloadWithInputs({
      label: `${name} (${FRAMEWORK})`,
      chunkingStrategy: chunk,
      assemblyStrategy: asm,
      framework: FRAMEWORK,
      inputData,
      metadata,
      streamingMode: STREAMING
    });

    await pollUntilComplete(wid, POLL_INTERVAL, TIMEOUT_MS);
    const dt = (Date.now() - t0) / 1000;
    console.log(`${name} complete in ${dt.toFixed(2)}s`);
    timeline.push({ stage: name, seconds: Number(dt.toFixed(2)), workloadId: wid });

    const b64 = await downloadFinalWithRetry(wid);

    // Save intermediate result
    await saveIntermediateResult(name, wid, b64);

    return b64;
  };

  // Define all stages
  const stages = ['LAYER NORM 1', 'ATTENTION', 'LAYER NORM 2', 'FFN'];
  const resumeIndex = RESUME_FROM ? stages.indexOf(RESUME_FROM) : -1;

  if (RESUME_FROM && resumeIndex === -1) {
    console.error(`Invalid resume stage: ${RESUME_FROM}. Valid stages: ${stages.join(', ')}`);
    process.exit(1);
  }

  // STAGE 0: input
  const x0 = makeInitialInput();
  let curB64 = toB64Float32(x0);

  // STAGE 1: LN1
  const ln1 = lnParams();
  const residualZeroB64 = toB64Float32(zeros(SEQ * DMODEL));

  if (resumeIndex >= 0) {
    curB64 = await stageRun('LAYER NORM 1', {}, true); // Skip execution, load cached
  } else {
    curB64 = await stageRun('LAYER NORM 1', {
      chunk: STRAT.ln.chunk,
      asm: STRAT.ln.asm,
      inputData: {
        input: curB64,
        gamma: toB64Float32(ln1.gamma),
        beta: toB64Float32(ln1.beta),
        residual: residualZeroB64
      },
      metadata: { seqLength: SEQ, dModel: DMODEL, epsilon: ln1.epsilon }
    });
  }

  // STAGE 2: Attention
  const att = attentionWeights();
  const x1 = fromB64Float32(curB64);
  const qkv = preprojectQKV(x1, att);

  if (resumeIndex >= 1) {
    curB64 = await stageRun('ATTENTION', {}, true);
  } else {
    curB64 = await stageRun('ATTENTION', {
      chunk: STRAT.att.chunk,
      asm: STRAT.att.asm,
      inputData: {
        input: curB64,
        Wq: toB64Float32(att.Wq),
        Wk: toB64Float32(att.Wk),
        Wv: toB64Float32(att.Wv),
        Wo: toB64Float32(att.Wo),
        Q: toB64Float32(qkv.Q),
        K: toB64Float32(qkv.K),
        V: toB64Float32(qkv.V)
      },
      metadata: { seqLength: SEQ, dModel: DMODEL, numHeads: HEADS }
    });
  }

  // STAGE 3: LN2
  const ln2 = lnParams();

  if (resumeIndex >= 2) {
    curB64 = await stageRun('LAYER NORM 2', {}, true);
  } else {
    curB64 = await stageRun('LAYER NORM 2', {
      chunk: STRAT.ln.chunk,
      asm: STRAT.ln.asm,
      inputData: {
        input: curB64,
        gamma: toB64Float32(ln2.gamma),
        beta: toB64Float32(ln2.beta),
        residual: residualZeroB64
      },
      metadata: { seqLength: SEQ, dModel: DMODEL, epsilon: ln2.epsilon }
    });
  }

  // STAGE 4: FFN
  const ffn = ffnWeights();

  if (resumeIndex >= 3) {
    curB64 = await stageRun('FFN', {}, true);
  } else {
    curB64 = await stageRun('FFN', {
      chunk: STRAT.ffn.chunk,
      asm: STRAT.ffn.asm,
      inputData: {
        input: curB64,
        X: curB64,
        W1: toB64Float32(ffn.W1),
        b1: toB64Float32(ffn.b1),
        W2: toB64Float32(ffn.W2),
        b2: toB64Float32(ffn.b2)
      },
      metadata: { seqLength: SEQ, dModel: DMODEL, dFF: DFF, activation: 'gelu' }
    });
  }

  // Final stats
  console.log('\nTransformer block finished');
  const outBuf = Buffer.from(curB64, 'base64');
  const elems = outBuf.byteLength / 4;
  const mean = (() => {
    const v = new Float32Array(outBuf.buffer, outBuf.byteOffset, elems);
    let s = 0; for (let i = 0; i < v.length; i++) s += v[i];
    return s / v.length;
  })();
  const hash = crypto.createHash('sha256').update(outBuf).digest('hex').slice(0, 16);

  console.log('Output:', {
    bytes: outBuf.byteLength,
    elements: elems,
    expected: SEQ * DMODEL,
    mean: Number(mean.toFixed(6)),
    sha256_16: hash
  });
  console.log('Timeline:', timeline);

  if (SAVE_INTERMEDIATES) {
    console.log(`\nIntermediate results saved to: ${INTERMEDIATES_DIR}`);
    console.log('Resume from any stage using: --resume-from "STAGE NAME"');
  }

  process.exit(0);
})().catch(err => {
  console.error('\nOrchestrator error:', err.message);
  if (err.stack) console.error(err.stack.split('\n').slice(0,4).join('\n'));
  process.exit(1);
});