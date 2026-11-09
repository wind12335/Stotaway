#!/bin/bash
# Operator-level GEMM+RS matrix (paper table "GEMM+RS"): 11 published dense models x M 8192/16384/32768,
# K = ffn/8 (fc2 / down). ~1.5 card-hours on 8 cards. Same usage as run_ag_matrix.sh.
set -u
export DUSHMEM_SYMMETRIC_SIZE=${DUSHMEM_SYMMETRIC_SIZE:-12884901888}   # 12 GB: largest cells (405B, M32768)
source "$(dirname "$(readlink -f "$0")")/../env/env.sh"
OUT=${SW_OUT:-$SW_ROOT/results/raw/ops_8card}; mkdir -p "$OUT/rsb2b_full"
MODELS=${*:-$(python3 -I -c "import sys; sys.path.insert(0,'$SW_ROOT/planner'); from shapes import SHAPES; print(' '.join(k for k,h,f,kind,_ in SHAPES if kind=='dense'))")}
export FG_NOBWD=${FG_NOBWD:-1}
for m in $MODELS; do
  read H F <<<$(python3 -I -c "import sys; sys.path.insert(0,'$SW_ROOT/planner'); from shapes import SHAPES; d={k:(h,f) for k,h,f,_,_ in SHAPES}; print(*d['$m'])")
  for M in ${SW_MS:-8192 16384 32768}; do
    L=$OUT/rsb2b_full/${m}_M$M.log
    echo "=== $m M=$M N=$H K=$((F/8))" >> $OUT/rsb2b.log
    FG_M=$M FG_N=$H FG_KS=$((F/8)) FG_OPS=${FG_OPS:-32} timeout 600 \
      $SW_MPIRUN -x FG_NOBWD -x FG_M -x FG_N -x FG_KS -x FG_OPS python3 $SW_ROOT/bench/gemm_rs.py > $L 2>&1
    echo "rc=$?" >> $L
    grep -a -E "^\[fg2?\] (K=|correctness)" $L >> $OUT/rsb2b.log
    tail -n 2 $OUT/rsb2b.log
  done
done
