#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
raw="$res/raw"
mkdir -p "$raw"

export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export PYTHONPATH="$PWD/third_party/qwen36_fla:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_SYMMETRIC_SIZE=2G STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_DX_WORLD=8
export STOWAWAY_QWEN36=1 STOWAWAY_QWEN36_LOCAL_SPEC=1 STOWAWAY_LAYERS=64 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=24
export STOWAWAY_RECOMPUTE=1 STOWAWAY_SEQ=4096 STOWAWAY_MBS=2 STOWAWAY_GBS=2 STOWAWAY_ITERS=12 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 TEST_TIMEOUT=1500 STOWAWAY_OVERLAP_GRAD_REDUCE=1
export STOWAWAY_DX=0 STOWAWAY_ASYNC_TP=1 STOWAWAY_ASYNC_DX_SITES='' STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 STOWAWAY_COLUMN_MAX_N=20000
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
unset STOWAWAY_TP2_LOOKUP STOWAWAY_DX_SHAPE_PLAN STOWAWAY_TP2_ONLINE_BUNDLE STOWAWAY_BRS STOWAWAY_DX8_BACKWARD
unset STOWAWAY_MULTI_PLAN8 VS_MULTI_PLAN8 VS_AUTO_PLAN8
export STOWAWAY_AUTO_PLAN8="$PWD/experiments/tp8_auto_qwen36_20261113.json"
export OUT="$raw/tp8_qwen36_auto_plan_mbs2_fla_retry_20261113.log"
if [[ ! -e "$OUT" ]]; then
  echo "START Qwen automatic plan $(date -u +%T)"
  timeout -k 10 1650 bash experiments/run_megatron_async_smoke.sh > "$raw/tp8_qwen36_auto_plan_driver_20261113.log" 2>&1
  echo "END Qwen automatic plan rc=$? $(date -u +%T)"
fi
if ! grep -q 'iteration *12 */ *12' "$OUT" 2>/dev/null; then
  echo 'Qwen automatic plan incomplete; no result was recorded.'
  python - "$OUT" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
fi
python bench/report_tp8_e2e_async_20261113.py

unset STOWAWAY_AUTO_PLAN8 STOWAWAY_ASYNC_TP STOWAWAY_ASYNC_DX_SITES
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH PYTHONPATH="$PWD" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVSHMEM_HOME=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B V8_M=16384 V8_ROPE_FACTOR=8 V8_REPS=3
export V8_ARM=auto_plan V8_AUTO_PLAN="$PWD/experiments/tp8_auto_llama70_20261113.json"
export V8_OUT="$res/tp8_20261111/vllm_bwmatch_auto_plan_retry_20261113.json"
if [[ ! -e "$V8_OUT" ]]; then
  echo "START Llama automatic plan $(date -u +%T)"
  timeout -k 10 1800 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_async_bwmatch.py > "$raw/tp8_llama70_auto_plan_retry_20261113.log" 2>&1
  echo "END Llama automatic plan rc=$? $(date -u +%T)"
fi
python bench/report_tp8_e2e_async_20261113.py
