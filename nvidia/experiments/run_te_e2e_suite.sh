#!/usr/bin/env bash
# Diagnostic TE baseline against TE tp_comm_overlap; not the local-layer baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_TRANSFORMER_IMPL=transformer_engine STOWAWAY_DIAGNOSTIC_TORCH_NORM=0 UB_SKIPMC=1
export STOWAWAY_RECOMPUTE=1 STOWAWAY_VEHICLE_27B=1 STOWAWAY_LAYERS=4
export STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=40 STOWAWAY_SEQ=4096
export STOWAWAY_ITERS=15 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 NVSHMEM_SYMMETRIC_SIZE=2G TEST_TIMEOUT=600
export STOWAWAY_DX=0 STOWAWAY_BRS=0 STOWAWAY_TE_INSPECT=1
unset STOWAWAY_CAPTURE_DIR STOWAWAY_GRAD_CAPTURE_DIR STOWAWAY_TP2_LOOKUP STOWAWAY_MOE_EP
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local arm=$1 repeat=$2
  if [[ "$arm" == overlap ]]; then export STOWAWAY_TE_OVERLAP=1; else export STOWAWAY_TE_OVERLAP=0; fi
  export OUT="$raw/e2e_te_${arm}_r${repeat}_20261110.log"
  echo "RUN TE $arm $repeat $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_te_${arm}_r${repeat}_driver_20261110.log" 2>&1
}
run_arm stock 0
run_arm overlap 0
run_arm overlap 1
run_arm stock 1
run_arm stock 2
run_arm overlap 2
