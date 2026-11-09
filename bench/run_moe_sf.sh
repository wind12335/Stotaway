#!/bin/bash
# Operator-level MoE layer (Qwen3-30B-A3B shape, EP=8, real routing traces, fwd+bwd):
# host-sync-free dispatch -> grouped experts -> combine vs the Megatron-style RCCL path.
# Correctness on every round (ours vs RCCL vs fp32 ref). ~10 min on 8 cards.
set -u
source "$(dirname "$(readlink -f "$0")")/../env/env.sh"
OUT=${SW_OUT:-$SW_ROOT/results/raw/moe}; mkdir -p "$OUT"
for T in ${SG_TS:-1024 4096}; do
  SG_T=$T timeout 1500 $SW_MPIRUN -x SG_T -x SG_ITERS -x SG_ROUNDS -x SG_B \
    python3 $SW_ROOT/bench/moe_sf_layer.py > $OUT/moe_sf_layer_T$T.log 2>&1
  echo "rc=$?" >> $OUT/moe_sf_layer_T$T.log
  grep -a "^\[moesf\]" $OUT/moe_sf_layer_T$T.log
done
