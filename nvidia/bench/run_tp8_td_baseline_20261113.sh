#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
if [[ ! -f "$res/tp8_external_dispatch_triton_dist_20261113.json" ]]; then
  echo 'Triton-distributed dispatcher has not passed TP8 validation; skip model runs.'
  exit 0
fi
tdpath=/root/Triton-distributed/3rdparty/triton/python:/root/Triton-distributed/python
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G

export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export PYTHONPATH="$tdpath:$PWD/third_party/qwen36_fla:$PWD"
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if timeout 60 python -c 'import triton_dist; import triton._C.libtriton.distributed' >/dev/null 2>&1; then
 export STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_DX_WORLD=8
 export STOWAWAY_QWEN36=1 STOWAWAY_QWEN36_LOCAL_SPEC=1 STOWAWAY_LAYERS=64 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=24
 export STOWAWAY_RECOMPUTE=1 STOWAWAY_SEQ=4096 STOWAWAY_MBS=2 STOWAWAY_GBS=2 STOWAWAY_ITERS=12 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 TEST_TIMEOUT=1500 STOWAWAY_OVERLAP_GRAD_REDUCE=1
 export STOWAWAY_DX=0 STOWAWAY_ASYNC_TP=0 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 STOWAWAY_COLUMN_MAX_N=20000
 unset STOWAWAY_AUTO_PLAN8 STOWAWAY_ASYNC_DX_SITES STOWAWAY_DX_SHAPE_PLAN STOWAWAY_TP2_LOOKUP STOWAWAY_TP2_ONLINE_BUNDLE
 export STOWAWAY_MULTI_PLAN8="$PWD/experiments/tp8_baseline_td_qwen36_20261113.json"
 export OUT="$res/raw/tp8_qwen36_td_baseline_mbs2_fla_retry_20261113.log"
 if [[ ! -e "$OUT" ]]; then
  echo "START Qwen pure Triton-distributed baseline $(date -u +%T)"
  timeout -k 10 1650 bash experiments/run_megatron_async_smoke.sh > "$res/raw/tp8_qwen36_td_baseline_driver_20261113.log" 2>&1
  echo "END Qwen rc=$? $(date -u +%T)"
 fi
else
 echo 'Megatron environment cannot import Triton-distributed; Qwen model run unavailable.'
fi
if [[ -n "${OUT:-}" && -f "$OUT" ]] && ! grep -q 'iteration *12 */ *12' "$OUT"; then
 python - "$OUT" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
fi
unset STOWAWAY_MULTI_PLAN8

export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH PYTHONPATH="$tdpath:$PWD" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if timeout 240 python -c 'import triton_dist; import vllm; import triton._C.libtriton.distributed' >/dev/null 2>&1; then
 export V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B V8_M=16384 V8_ROPE_FACTOR=8 V8_REPS=3
 export V8_ARM=multi_plan V8_AUTO_PLAN="$PWD/experiments/tp8_baseline_td_llama70_20261113.json"
 export V8_OUT="$res/tp8_20261111/vllm_bwmatch_td_baseline_retry_20261113.json"
 if [[ ! -e "$V8_OUT" ]]; then
  echo "START Llama pure Triton-distributed baseline $(date -u +%T)"
  timeout -k 10 1800 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_async_bwmatch.py > "$res/raw/tp8_llama70_td_baseline_retry_20261113.log" 2>&1
  echo "END Llama rc=$? $(date -u +%T)"
 fi
else
 echo 'vLLM Triton-distributed import precheck did not complete; inspect the environment before retrying.'
fi
python bench/report_tp8_e2e_async_20261113.py
