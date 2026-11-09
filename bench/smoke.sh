#!/bin/bash
# 8-card smoke test (~10 min): one small cell of each operator + the MoE layer at T=1024.
# Every line must say fails=0 / PASS; speedups are printed for orientation only (small M).
set -u
source "$(dirname "$(readlink -f "$0")")/../env/env.sh"
OUT=${SW_OUT:-/tmp/stowaway_smoke}; mkdir -p "$OUT"; echo "logs -> $OUT"
AGF_M=8192 AGF_K=4096 AGF_NCS=1536,3072 AGF_OPS=8 timeout 600 \
  $SW_MPIRUN -x AGF_M -x AGF_K -x AGF_NCS -x AGF_OPS python3 $SW_ROOT/bench/ag_gemm.py > $OUT/ag_gemm.log 2>&1
echo "ag_gemm rc=$?"; grep -a -E "^\[agf\] (Nc=|correctness)" $OUT/ag_gemm.log
FG_NOBWD=1 FG_M=8192 FG_N=4096 FG_KS=1536 FG_OPS=8 timeout 600 \
  $SW_MPIRUN -x FG_NOBWD -x FG_M -x FG_N -x FG_KS -x FG_OPS python3 $SW_ROOT/bench/gemm_rs.py > $OUT/gemm_rs.log 2>&1
echo "gemm_rs rc=$?"; grep -a -E "^\[fg2?\] (K=|correctness)" $OUT/gemm_rs.log
FG_M=8192 FG_N=4096 FG_KS=1536 FG_OPS=8 timeout 600 \
  $SW_MPIRUN -x FG_M -x FG_N -x FG_KS -x FG_OPS python3 $SW_ROOT/bench/ag_dgrad.py > $OUT/ag_dgrad.log 2>&1
echo "ag_dgrad rc=$?"; grep -a -E "^\[bg\] (K=|correctness)" $OUT/ag_dgrad.log
SG_T=1024 SG_ITERS=5 SG_ROUNDS=2 timeout 900 $SW_MPIRUN -x SG_T -x SG_ITERS -x SG_ROUNDS \
  python3 $SW_ROOT/bench/moe_sf_layer.py > $OUT/moe_sf_layer.log 2>&1
echo "moe_sf_layer rc=$?"; grep -a "^\[moesf\]" $OUT/moe_sf_layer.log
