#!/usr/bin/env bash
# Run only after the primary suite: each arm owns both GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVSHMEM_SYMMETRIC_SIZE=512M
NVSHMEM_HOME="$(python -c 'import os,site;print(next(os.path.join(p,"nvidia","nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p,"nvidia","nvshmem","lib","libnvshmem_host.so.3"))))')"
export NVSHMEM_HOME LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export E2E_M=1024 E2E_OUTPUT=32 E2E_REQUESTS=6 E2E_WARMUPS=2 E2E_SEED=20261110
export E2E_MODEL=/root/eb-public/huggingface-models/Qwen/Qwen3-14B E2E_ARCH=qwen3 E2E_GPU_UTIL=.65
root=/root/data/stowaway-nvidia/results/e2e_20261110
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local candidate=$1 repeat=$2
  case "$candidate" in
    stock_sp) export E2E_ARM=stock_sp VS_DX_COMPONENTS=row VS_ROW_EXACT=1 ;;
    overlap) export E2E_ARM=dx VS_DX_COMPONENTS=col,row VS_ROW_EXACT=0 ;;
    col_only) export E2E_ARM=dx VS_DX_COMPONENTS=col VS_ROW_EXACT=1 ;;
    row_overlap) export E2E_ARM=dx VS_DX_COMPONENTS=row VS_ROW_EXACT=0 ;;
    *) return 1 ;;
  esac
  export E2E_OUT="$root/qwen_${candidate}_r${repeat}.json"
  echo "RUN qwen $candidate $repeat $(date -Is)" >&2
  timeout 1200 python -m torch.distributed.run --standalone --nproc-per-node=2 experiments/vllm_e2e_bench.py > "$raw/e2e_qwen_${candidate}_r${repeat}_20261110.log" 2>&1
}
run_arm stock_sp 0
run_arm overlap 0
run_arm overlap 1
run_arm stock_sp 1
run_arm col_only 0
run_arm row_overlap 0
run_arm row_overlap 1
run_arm col_only 1
