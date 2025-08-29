#!/usr/bin/env node
// scripts/test-ecm-stage1.mjs
// Smoke-test ECM Stage 1 strategy via the HTTP API.
// Requires a running server (docker compose up) and at least one WebGPU client connected (headless ok).
//
// Example:
//   node scripts/test-ecm-stage1.mjs --server https://localhost:3000 \
//        --N 0xb4c9f5dd3a1 *ignored* \
//        --B1 50000 --curves 512 --chunk-size 64 --framework webgpu --streaming
//
// If --N is not provided, a demo 128-bit semiprime is used.

import fs from 'fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import minimist from 'minimist';
import https from 'https';

const args = minimist(process.argv.slice(2), {
  string: ['server','N','framework'],
  boolean: ['streaming'],
  default: {
    server: 'https://localhost:3000',
    framework: 'webgpu',
    streaming: true
  }
});

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiBase = args.server.replace(/\/$/, '');

function agentFor(urlStr) {
  const u = new URL(urlStr);
  if (u.protocol === 'https:') {
    return new https.Agent({ rejectUnauthorized: false });
  }
  return undefined;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    agent: agentFor(url)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${url} => ${res.status}: ${text}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url, { agent: agentFor(url) });
  if (!res.ok) throw new Error(`GET ${url} => ${res.status}`);
  return res.json();
}

function parseN() {
  if (args.N) {
    let s = String(args.N);
    if (/^0x/i.test(s)) return { N_hex: s.replace(/^0x/i,'') };
    if (/^[0-9A-Fa-f]+$/.test(s)) return { N_hex: s };
    if (/^[0-9]+$/.test(s)) return { N_dec: s };
  }
  // Default demo semiprime (128-bit): p=0xF12F... , q=0xE9AD..., product shown as hex
  const N_hex = 'c8f3f8af8e8b1f5bbce4817b8b9f9cfd'; // 128-bit composite (placeholder)
  return { N_hex };
}

(async () => {
  const Nobj = parseN();
  const B1 = parseInt(args.B1 || '20000', 10);
  const total_curves = parseInt(args.curves || '256', 10);
  const chunk_size = parseInt(args['chunk-size'] || '64', 10);
  const framework = args.framework || 'webgpu';
  const streamingMode = !!args.streaming;

  console.log(`Creating ECM Stage 1 workload:
  N: ${Nobj.N_hex ? '0x'+Nobj.N_hex : Nobj.N_dec}
  B1: ${B1}
  curves: ${total_curves} (chunk_size=${chunk_size})
  framework: ${framework}
  streaming: ${streamingMode}`);

  const payload = {
    label: `ECM Stage 1 (B1=${B1}, curves=${total_curves})`,
    framework,
    chunkingStrategy: 'ecm_stage1',
    assemblyStrategy: 'ecm_stage1_assembly',
    metadata: {
      ...Nobj,
      B1, total_curves, chunk_size, framework, compute_gcd: true,
      streamingMode
    },
    outputSizes: [1]
  };

  const wl = await postJSON(`${apiBase}/api/workloads/advanced`, payload);
  const workloadId = wl.id;
  console.log(`Workload created: ${workloadId}`);
  console.log(` - totalChunks (planned): ${wl.plan?.totalChunks || 'n/a'}`);

  // Start
  const startResp = await postJSON(`${apiBase}/api/workloads/${workloadId}/compute-start`, { streamingMode });
  console.log(`Started: ${startResp.status}, totalChunks=${startResp.totalChunks}`);

  // Poll until done
  while (true) {
    const st = await getJSON(`${apiBase}/api/workloads/${workloadId}/status`);
    process.stdout.write(`\rStatus: ${st.status}  chunks ${st.chunks?.completed || 0}/${st.chunks?.total || 0}  active=${st.chunks?.active || 0}   `);
    if (st.status === 'complete' || st.status === 'error') {
      console.log('');
      console.log('Final:', JSON.stringify(st, null, 2));
      break;
    }
    await new Promise(r => setTimeout(r, 1500));
  }

  // Try to fetch final result
  try {
    const info = await getJSON(`${apiBase}/api/workloads/${workloadId}/info`);
    if (info.finalResultBase64) {
      const json = Buffer.from(info.finalResultBase64, 'base64').toString('utf8');
      console.log('Final Result (JSON):', json);
    } else {
      console.log('No finalResultBase64 in info.');
    }
  } catch (e) {
    console.warn('Could not fetch final info:', e.message);
  }
})();
