#!/bin/bash
# vLLM LLaMA3-70B TP8 prefill: original vs sequence-parallel form + overlap method + predictor (8 cards).
# ~60-80 min for the default 8 arms (= 8-11 card-hours). Smoke first: VS_ARMS="stock dx" VS_REPS=1.
# Arms (ABAB order, one process each):
#   stock  original vLLM (TP AllReduce)
#   rccl   SP form (AR -> RS + AG), all four sites original RCCL: isolates the rewrite itself
#   dx     SP form, all four sites overlap method
#   plan   SP form, each site as the predictor's stage 1 picks (inference: no stage-2 term)
# Cells: gate (random-token consistency check, weak), ttft (1 x M tokens), ttft2 (2 x M), thru (16 x M).
# Correctness: run e2e/vllm/run_gate.sh afterwards (natural-text gate, compares every arm with stock).
# Env: VS_MNBT = prefill chunk M (default 16384), VS_REPS, SW_VLLM_MODEL (HF weights dir).
set -u
source "$(dirname "$(readlink -f "$0")")/../../env/env.sh"
export DUSHMEM_SYMMETRIC_SIZE=${VS_HEAP:-4294967296}
[ -n "${SW_PYSITE:-}" ] && export PYTHONPATH=$PYTHONPATH:$SW_PYSITE
R=${SW_OUT:-$SW_ROOT/results/raw/vllm}; mkdir -p $R; L=$R/chain.log
export VS_MNBT=${VS_MNBT:-16384} VS_REPS=${VS_REPS:-3} VS_OUT=$R VS_MODEL=${VS_MODEL:-$SW_VLLM_MODEL}
[ -d "$VS_MODEL" ] || { echo "model dir not found: $VS_MODEL (set SW_VLLM_MODEL)"; exit 1; }
echo "[vsp] start M=$VS_MNBT $(date '+%F %T')" | tee -a $L
PORT=29880
for ARM in ${VS_ARMS:-stock rccl dx plan stock rccl dx plan}; do
  PORT=$((PORT + 3))
  echo "[vsp] arm=$ARM start $(date +%T)" | tee -a $L
  VS_ARM=$ARM MASTER_PORT=$PORT timeout 1800 $SW_MPIRUN -x VS_ARM -x VS_MNBT -x VS_REPS -x VS_OUT -x VS_MODEL -x MASTER_PORT \
    python3 $SW_ROOT/e2e/vllm/vllm_sp.py > $R/${ARM}_M${VS_MNBT}_$(date +%H%M%S).log 2>&1
  echo "[vsp] arm=$ARM rc=$? $(date +%T)" | tee -a $L
  sleep 20
done
python3 -I $SW_ROOT/tools/vllm_report.py $R $VS_MNBT | tee -a $L
