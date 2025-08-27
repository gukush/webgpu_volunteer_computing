// server/scripts/submit-ecm.mjs
import fetch from 'node-fetch';

const HOST = process.env.HOST || 'https://localhost:8443';

// Example 64-bit semiprime (replace with your own)
const n = BigInt(process.argv[2] || '18446744065119617077'); // ~2^64 prime-ish test; replace with semiprime for demo

const body = {
  label: `ECM Stage-1 for ${n}`,
  chunkingStrategy: 'ecm_stage1',
  assemblyStrategy: 'ecm_stage1_assembly',
  framework: 'webgpu',
  streamingMode: true,
  metadata: {
    n: n.toString(),
    B1: 50000,           // tweak as desired (10k..100k good to start)
    curvesTotal: 4096,   // total curves to try
    curvesPerChunk: 256  // curves per dispatched chunk
  }
};

const main = async () => {
  const res = await fetch(`${HOST}/api/workloads/advanced`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await res.json();
  console.log(JSON.stringify(j, null, 2));
};
main().catch(e => { console.error(e); process.exit(1); });
