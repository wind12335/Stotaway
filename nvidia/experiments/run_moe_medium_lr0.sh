#!/usr/bin/env bash
# Isolate step-2 gradient differences from optimizer-induced parameter drift.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_TP=1 STOWAWAY_MOE=1 STOWAWAY_MOE_EXPERTS=8 STOWAWAY_MOE_TOPK=2 STOWAWAY_MOE_FFN=512
export STOWAWAY_LAYERS=2 STOWAWAY_HIDDEN=1024 STOWAWAY_FFN=4096 STOWAWAY_HEADS=16 STOWAWAY_SEQ=512
export STOWAWAY_MBS=1 STOWAWAY_GBS=2 STOWAWAY_ITERS=2 STOWAWAY_RECOMPUTE=0 STOWAWAY_DISTRIBUTED_OPTIMIZER=0
export STOWAWAY_DX=0 STOWAWAY_BRS=0 NVSHMEM_SYMMETRIC_SIZE=1G TEST_TIMEOUT=600
export STOWAWAY_LR=0 STOWAWAY_MIN_LR=0
unset STOWAWAY_TP2_LOOKUP STOWAWAY_MOE_GG STOWAWAY_CAPTURE_DIR
root=/root/data/stowaway-nvidia/results/captures/moe_medium_lr0_20261110
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local arm=$1
  if [[ "$arm" == ep ]]; then export STOWAWAY_MOE_EP=1; else export STOWAWAY_MOE_EP=0; fi
  export STOWAWAY_GRAD_CAPTURE_DIR="$root/$arm" OUT="$raw/e2e_moe_medium_lr0_${arm}_20261110.log"
  if [[ -e "$STOWAWAY_GRAD_CAPTURE_DIR" ]]; then echo "capture exists" >&2; exit 1; fi
  echo "RUN lr0 $arm $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_moe_medium_lr0_${arm}_driver_20261110.log" 2>&1
}
run_arm stock
run_arm ep
OMP_NUM_THREADS=4 python experiments/analyze_capture_differences.py "$root/stock" "$root/ep" --output /root/data/stowaway-nvidia/results/e2e_20261110/moe_medium_lr0_grad_differences.json > "$raw/e2e_moe_medium_lr0_grad_analysis_20261110.log" 2>&1
