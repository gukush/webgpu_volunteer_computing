#!/usr/bin/env bash
# test_frameworks.sh - submit one tiny job per framework via submit-task.mjs
set -euo pipefail

SUBMIT_TASK=${SUBMIT_TASK:-"./server/scripts/submit-task.mjs"}
SERVER_URL=${SERVER_URL:-"https://localhost:3000"}
LABEL_PREFIX=${LABEL_PREFIX:-"smoke"}
INCLUDE_CUDA=${INCLUDE_CUDA:-"0"}

WG_COUNTS="1,1,1"
OUT_BYTES=$((64 * 4))

run_compute() {
  local fw="$1"; shift
  local kfile="$1"; shift
  local label="${LABEL_PREFIX}-${fw}"
  echo "==> Submitting ${fw} job (${kfile})"
  node "$SUBMIT_TASK" compute \
    --server "$SERVER_URL" \
    --framework "$fw" \
    --kernel "$kfile" \
    --workgroups "$WG_COUNTS" \
    --output-size "$OUT_BYTES" \
    --label "$label" "$@"
}

AVAILABLE=$(node "$SUBMIT_TASK" frameworks 2>/dev/null || echo '["webgpu","webgl","opencl","vulkan"]')
has() { echo "$AVAILABLE" | grep -qi "$1"; }

if has webgpu; then run_compute webgpu "$KDIR/webgpu_smoke.wgsl"; fi
if has webgl; then
  node "$SUBMIT_TASK" compute-advanced \
    --server "$SERVER_URL" \
    --framework webgl \
    --kernel "$KDIR/webgl_smoke.glsl" \
    --label "${LABEL_PREFIX}-webgl" \
    --metadata '{"width":8,"height":8}'
fi
if has opencl; then run_compute opencl "$KDIR/opencl_smoke.cl"; fi
if has vulkan; then run_compute vulkan "$KDIR/vulkan_smoke.comp"; fi
if [[ "$INCLUDE_CUDA" == "1" ]] && has cuda; then run_compute cuda "$KDIR/cuda_smoke.cu"; fi

echo "All submissions attempted."
