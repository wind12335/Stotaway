#!/usr/bin/env bash
# Separate one-step full-vehicle parameter comparison; never use for timing.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_RECOMPUTE=1 STOWAWAY_VEHICLE_27B=1 STOWAWAY_LAYERS=4
export STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=40 STOWAWAY_SEQ=4096
export STOWAWAY_ITERS=1 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 NVSHMEM_SYMMETRIC_SIZE=2G TEST_TIMEOUT=900
export STOWAWAY_COLUMN_MAX_N=20000 STOWAWAY_BRS=0
unset STOWAWAY_GRAD_CAPTURE_DIR STOWAWAY_TP2_INTERPOLATE
raw=/root/data/stowaway-nvidia/results/raw
root=/root/data/stowaway-nvidia/results/captures/e2e_fourlayer_20261110
available_kib=$(df -Pk "$raw" | awk 'NR==2 {print $4}')
if (( available_kib < 45 * 1024 * 1024 )); then
  echo "at least 45 GiB free required for the three parameter captures" >&2
  exit 1
fi
run_arm() {
  local arm=$1
  unset STOWAWAY_TP2_LOOKUP
  case "$arm" in
    stock) export STOWAWAY_DX=0 ;;
    full_dx) export STOWAWAY_DX=1 ;;
    planner) export STOWAWAY_DX=1 STOWAWAY_TP2_LOOKUP="$PWD/planner/data_a800/tp2_training_lookup_20261110.json" ;;
  esac
  export STOWAWAY_CAPTURE_DIR="$root/$arm" OUT="$raw/e2e_correctness_${arm}_20261110.log"
  if [[ -e "$STOWAWAY_CAPTURE_DIR" ]]; then echo "capture directory already exists: $STOWAWAY_CAPTURE_DIR" >&2; exit 1; fi
  echo "RUN correctness $arm $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_correctness_${arm}_driver_20261110.log" 2>&1
}
run_arm stock
run_arm planner
run_arm full_dx
python experiments/compare_captures.py "$root/stock" "$root/planner" | tee "$raw/e2e_correctness_planner_compare_20261110.log"
python experiments/compare_captures.py "$root/stock" "$root/full_dx" | tee "$raw/e2e_correctness_full_dx_compare_20261110.log"
