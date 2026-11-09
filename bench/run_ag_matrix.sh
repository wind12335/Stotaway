#!/bin/bash
# Operator-level AG+GEMM matrix (paper table "AG+GEMM"): 11 published dense models x M 8192/16384/32768,
# each cell its own 8-rank process, widths up (ffn/8) and gate_up (2 ffn/8). ~1.5 card-hours on 8 cards.
# Usage: bench/run_ag_matrix.sh [model ...]   (default: all models in planner/shapes.py)
#        SW_MS="8192 16384" bench/run_ag_matrix.sh qwen3-8b
set -u
export DUSHMEM_SYMMETRIC_SIZE=${DUSHMEM_SYMMETRIC_SIZE:-12884901888}   # 12 GB: largest cells (405B, M32768)
source "$(dirname "$(readlink -f "$0")")/../env/env.sh"
OUT=${SW_OUT:-$SW_ROOT/results/raw/ops_8card}; mkdir -p "$OUT/agprod_full"
MODELS=${*:-$(python3 -I -c "import sys; sys.path.insert(0,'$SW_ROOT/planner'); from shapes import SHAPES; print(' '.join(k for k,h,f,kind,_ in SHAPES if kind=='dense'))")}
for m in $MODELS; do
  read H F <<<$(python3 -I -c "import sys; sys.path.insert(0,'$SW_ROOT/planner'); from shapes import SHAPES; d={k:(h,f) for k,h,f,_,_ in SHAPES}; print(*d['$m'])")
  for M in ${SW_MS:-8192 16384 32768}; do
    L=$OUT/agprod_full/${m}_M$M.log
    echo "=== $m M=$M K=$H Nc=$((F/8)),$((2*F/8))" >> $OUT/agprod.log
    AGF_M=$M AGF_K=$H AGF_NCS=$((F/8)),$((2*F/8)) AGF_OPS=${AGF_OPS:-32} timeout 600 \
      $SW_MPIRUN -x AGF_M -x AGF_K -x AGF_NCS -x AGF_OPS python3 $SW_ROOT/bench/ag_gemm.py > $L 2>&1
    echo "rc=$?" >> $L
    grep -a -E "^\[agf\] (Nc=|correctness)" $L >> $OUT/agprod.log
    tail -n 3 $OUT/agprod.log
  done
done
