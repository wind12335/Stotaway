#!/bin/bash
# Natural-text output gate for the vLLM arms (8 cards, ~3.5 min / arm, ~2 card-hours for 4 arms).
# Prompts = exactly M tokens of real text (model dir README / LICENSE / USE_POLICY, repeated, cut
# mid-sentence) -> confident continuations; every arm's 4 x 48 greedy tokens are compared with stock.
# (The random-token gate in vllm_sp.py is too flat to judge correctness: near-ties flip under bf16
#  rounding within a few tokens even for the rccl arm.)
set -u
source "$(dirname "$(readlink -f "$0")")/../../env/env.sh"
export DUSHMEM_SYMMETRIC_SIZE=${VS_HEAP:-4294967296}
export VS_MNBT=${VS_MNBT:-16384} VS_MODEL=${VS_MODEL:-$SW_VLLM_MODEL}
export VS_OUT=${SW_OUT:-$SW_ROOT/results/raw/vllm/gate_text}; mkdir -p $VS_OUT
PORT=29950
for ARM in ${VS_ARMS:-stock rccl dx plan}; do
  PORT=$((PORT + 3)); echo "[gate] $ARM start $(date +%T)"
  VS_ARM=$ARM MASTER_PORT=$PORT timeout 1500 $SW_MPIRUN -x VS_ARM -x VS_MNBT -x VS_OUT -x VS_MODEL -x MASTER_PORT \
    python3 $SW_ROOT/e2e/vllm/vllm_gate.py > $VS_OUT/${ARM}.log 2>&1
  echo "[gate] $ARM rc=$? $(date +%T)"; sleep 15
done
python3 -I $SW_ROOT/tools/vllm_gate_report.py $VS_OUT $VS_MNBT | tee $VS_OUT/report.md
