#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export STOWAWAY_TP=1 STOWAWAY_MOE=1 STOWAWAY_MOE_EXPERTS=8 STOWAWAY_MOE_TOPK=2 STOWAWAY_MOE_FFN=512
export STOWAWAY_LAYERS=2 STOWAWAY_HIDDEN=1024 STOWAWAY_FFN=4096 STOWAWAY_HEADS=16 STOWAWAY_SEQ=512
export STOWAWAY_MBS=1 STOWAWAY_GBS=2 STOWAWAY_ITERS=15 STOWAWAY_RECOMPUTE=0 STOWAWAY_DISTRIBUTED_OPTIMIZER=0
export STOWAWAY_DX=0 STOWAWAY_BRS=0 NVSHMEM_SYMMETRIC_SIZE=1G TEST_TIMEOUT=600
export STOWAWAY_MOE_PRE_FC2_SCALE="${STOWAWAY_MOE_PRE_FC2_SCALE:-1}"
unset STOWAWAY_CAPTURE_DIR STOWAWAY_GRAD_CAPTURE_DIR STOWAWAY_TP2_LOOKUP STOWAWAY_MOE_GG
raw=/root/data/stowaway-nvidia/results/raw
tag=${STOWAWAY_MOE_BENCH_TAG:-medium}
run_arm() {
  local arm=$1 repeat=$2
  if [[ "$arm" == ep ]]; then export STOWAWAY_MOE_EP=1; else export STOWAWAY_MOE_EP=0; fi
  export OUT="$raw/e2e_moe_${tag}_${arm}_r${repeat}_20261110.log"
  echo "RUN $arm $repeat $(date -Is)" >&2
  bash experiments/run_megatron_smoke.sh > "$raw/e2e_moe_${tag}_${arm}_r${repeat}_driver_20261110.log" 2>&1
}
run_arm stock 0
run_arm ep 0
run_arm ep 1
run_arm stock 1
run_arm stock 2
run_arm ep 2
