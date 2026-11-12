#!/usr/bin/env bash
# Out-of-calibration M values; compare stock to guarded interpolation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_RECOMPUTE=1 STOWAWAY_VEHICLE_27B=1 STOWAWAY_LAYERS=4
export STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=40
export STOWAWAY_ITERS=15 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 NVSHMEM_SYMMETRIC_SIZE=2G TEST_TIMEOUT=900
export STOWAWAY_COLUMN_MAX_N=20000 STOWAWAY_BRS=0
unset STOWAWAY_CAPTURE_DIR STOWAWAY_GRAD_CAPTURE_DIR
raw=/root/data/stowaway-nvidia/results/raw
run_arm() {
  local m=$1 arm=$2 repeat=$3
  export STOWAWAY_SEQ="$m"
  unset STOWAWAY_TP2_LOOKUP STOWAWAY_TP2_INTERPOLATE
  if [[ "$arm" == planner ]]; then
    export STOWAWAY_DX=1 STOWAWAY_TP2_INTERPOLATE=1 STOWAWAY_TP2_LOOKUP="$PWD/planner/data_a800/tp2_training_lookup_20261110.json"
  else
    export STOWAWAY_DX=0
  fi
  export OUT="$raw/e2e_m${m}_${arm}_r${repeat}_20261110.log"
  echo "RUN M=$m $arm $repeat $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_m${m}_${arm}_r${repeat}_driver_20261110.log" 2>&1
}
for m in 5120 6144; do
  run_arm "$m" stock 0
  run_arm "$m" planner 0
  run_arm "$m" planner 1
  run_arm "$m" stock 1
  run_arm "$m" stock 2
  run_arm "$m" planner 2
done
