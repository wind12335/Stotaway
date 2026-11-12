#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVSHMEM_SYMMETRIC_SIZE=512M
NVSHMEM_HOME="$(python -c 'import os,site;print(next(os.path.join(p,"nvidia","nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p,"nvidia","nvshmem","lib","libnvshmem_host.so.3"))))')"
export NVSHMEM_HOME LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export E2E_M=1024 E2E_OUTPUT=32 E2E_BATCH=4 E2E_BATCH_REPEATS=4 E2E_WARMUPS=2
unset VS_DX_COMPONENTS VS_ROW_EXACT
root=/root/data/stowaway-nvidia/results/e2e_20261110
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local model=$1 arm=$2 repeat=$3
  if [[ "$model" == qwen ]]; then
    export E2E_MODEL=/root/eb-public/huggingface-models/Qwen/Qwen3-14B E2E_ARCH=qwen3 E2E_GPU_UTIL=.65 NVSHMEM_SYMMETRIC_SIZE=512M
  else
    export E2E_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B E2E_ARCH=llama E2E_GPU_UTIL=.97 NVSHMEM_SYMMETRIC_SIZE=256M
  fi
  export E2E_ARM="$arm" E2E_OUT="$root/${model}_batch_${arm}_r${repeat}.json"
  echo "RUN batch $model $arm $repeat $(date -Is)" >&2
  timeout 1200 python -m torch.distributed.run --standalone --nproc-per-node=2 experiments/vllm_batch_e2e.py > "$raw/e2e_${model}_batch_${arm}_r${repeat}_20261110.log" 2>&1
}
run_arm qwen stock 0
run_arm qwen dx 0
run_arm qwen dx 1
run_arm qwen stock 1
run_arm llama dx 0
run_arm llama stock 0
run_arm llama stock 1
run_arm llama dx 1
