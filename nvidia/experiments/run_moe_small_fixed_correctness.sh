#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_TP=1 STOWAWAY_MOE=1 STOWAWAY_MOE_EXPERTS=4 STOWAWAY_MOE_TOPK=2 STOWAWAY_MOE_FFN=128
export STOWAWAY_LAYERS=2 STOWAWAY_HIDDEN=128 STOWAWAY_FFN=512 STOWAWAY_HEADS=4 STOWAWAY_SEQ=128
export STOWAWAY_MBS=1 STOWAWAY_GBS=2 STOWAWAY_ITERS=2 STOWAWAY_RECOMPUTE=0 STOWAWAY_DISTRIBUTED_OPTIMIZER=0
export STOWAWAY_DX=0 STOWAWAY_BRS=0 NVSHMEM_SYMMETRIC_SIZE=256M TEST_TIMEOUT=600
unset STOWAWAY_MOE_PRE_FC2_SCALE STOWAWAY_MOE_GG STOWAWAY_TP2_LOOKUP STOWAWAY_MOE_REFERENCE_GEMM
root=/root/data/stowaway-nvidia/results/captures/moe_small_fixed_20261110
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local arm=$1
  if [[ "$arm" == ep ]]; then export STOWAWAY_MOE_EP=1; else export STOWAWAY_MOE_EP=0; fi
  export STOWAWAY_CAPTURE_DIR="$root/${arm}_params" STOWAWAY_GRAD_CAPTURE_DIR="$root/${arm}_grads"
  export OUT="$raw/e2e_moe_small_fixed_correctness_${arm}_20261110.log"
  if [[ -e "$STOWAWAY_CAPTURE_DIR" || -e "$STOWAWAY_GRAD_CAPTURE_DIR" ]]; then exit 1; fi
  echo "RUN small correctness $arm $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_moe_small_fixed_correctness_${arm}_driver_20261110.log" 2>&1
}
run_arm stock
run_arm ep
python experiments/compare_captures.py "$root/stock_params" "$root/ep_params" | tee "$raw/e2e_moe_small_fixed_params_compare_20261110.log"
python experiments/compare_captures.py "$root/stock_grads" "$root/ep_grads" | tee "$raw/e2e_moe_small_fixed_grads_compare_20261110.log"
