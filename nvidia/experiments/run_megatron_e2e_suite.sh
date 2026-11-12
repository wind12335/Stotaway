#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_RECOMPUTE=1 STOWAWAY_VEHICLE_27B=1 STOWAWAY_LAYERS=4
export STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=40 STOWAWAY_SEQ=4096
export STOWAWAY_ITERS=15 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 NVSHMEM_SYMMETRIC_SIZE=2G TEST_TIMEOUT=600
export STOWAWAY_COLUMN_MAX_N=20000
unset STOWAWAY_CAPTURE_DIR STOWAWAY_GRAD_CAPTURE_DIR STOWAWAY_TP2_INTERPOLATE
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local arm=$1 repeat=$2
  unset STOWAWAY_TP2_LOOKUP STOWAWAY_BRS
  case "$arm" in
    stock) export STOWAWAY_DX=0 STOWAWAY_BRS=0 ;;
    full_dx) export STOWAWAY_DX=1 STOWAWAY_BRS=0 ;;
    planner) export STOWAWAY_DX=1 STOWAWAY_BRS=0 STOWAWAY_TP2_LOOKUP="$PWD/planner/data_a800/tp2_training_lookup_20261110.json" ;;
  esac
  export OUT="$raw/e2e_megatron_${arm}_r${repeat}_20261110.log"
  echo "RUN $arm $repeat $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_megatron_${arm}_r${repeat}_driver_20261110.log" 2>&1
}
# Seed 20261110 selects the first order; rotate across three blocks so each
# treatment appears once in every position. Each arm starts a fresh process.
run_arm planner 0
run_arm stock 0
run_arm full_dx 0
run_arm stock 1
run_arm full_dx 1
run_arm planner 1
run_arm full_dx 2
run_arm planner 2
run_arm stock 2
