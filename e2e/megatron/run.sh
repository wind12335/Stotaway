#!/bin/bash
# End-to-end Megatron A/B on one 8-card node (TP=8 + SP). Each arm = its own mpirun process.
#
#   SW_MODEL=27b|70b  MBS=8  ARMS="b1:base n1:fnull p1:plan b2:base n2:fnull p2:plan"  e2e/megatron/run.sh
#
# Arm = name:kind. Kinds (what each of the 4 TP call sites does):
#   base   original Megatron (vendor GEMM + RCCL, serial)                  QKV  o_proj  fc1  fc2
#   fnull  null control: same env / heap / hooks, but nothing taken over   orig orig   orig orig
#   fdk    only the standalone RSs (no adjacent GEMM) -> engine RS         orig (RS)   orig (RS)
#   fallt  + forward fused GEMM+RS, backward stock                         orig ovl    orig ovl
#   fbag   RS sites overlapped fwd+bwd (fused GEMM+RS, fused AG+dgrad)     orig ovl    orig ovl
#   fbc    all four sites overlapped (fbag + fused AG+GEMM on QKV, fc1)    ovl  ovl    ovl  ovl
#   fcq    fbag + QKV overlapped           fcl  fbag + fc1 overlapped
#   fcth   fbc + host timing of the fused AG (diagnostic: F50 host-block law)
#   plan   the PREDICTOR picks one of base / fbag / fbc / fcq / fcl (planner/planner.py); if a base arm
#          already ran in this ROOT its median step time is fed to the planner (--step-ms), else the
#          planner uses its GEMM probe files. Decision + report -> $ROOT/plan_<name>.md
# Output: $ROOT/driver.log (one line per arm: median step ms from iteration 3 on, fallbacks, val loss)
#         $ROOT/<name>/out_r*.log (rank logs; iteration lines are printed by the last rank, r7)
# Cost per arm (8 cards, ITERS=16): 27B MBS8 ~12.5 min, 70B MBS8 ~6.5 min; less at smaller MBS.
set -u
source "$(dirname "$(readlink -f "$0")")/../../env/env.sh"
ROOT=${SW_OUT:-$SW_ROOT/results/raw/e2e_8card/${SW_MODEL:-27b}_mbs${MBS:-8}_$(date +%m%d_%H%M)}
mkdir -p "$ROOT"
export MBS=${MBS:-8}
case ${SW_MODEL:-27b} in
  27b) export VH_HIDDEN=5120 VH_FFN=17408 VH_HEADS=40 VH_KVG=8 NL=${NL:-32} ;;
  70b) export VH_HIDDEN=8192 VH_FFN=28672 VH_HEADS=64 VH_KVG=8 NL=${NL:-8} ;;
  *)   : "${VH_HIDDEN:?custom model: set VH_HIDDEN VH_FFN VH_HEADS VH_KVG NL}" ;;
esac
PORT=${E2E_PORT:-35000}
SW_VARS="MGD_FUSED MGD_DRY MGD_AG MGD_AG_DRY MGD_AG_VERIFY MGD_AG_TIMING MGD_AG_MIN_N MGD_AG_CHUNK \
 MGD_AG_LANES MGD_AG_FORM MGD_AG_VSAMPLE MGD_TP_TIMING MGD_RS_RELAY MGD_BRS MGD_BRS_MIN_MB MGD_BRS_FORM \
 MGD_BRS_DEFER MGD_BRS_SLOTS MGD_BRS_SLOTS_DDO MGD_BRS_CELLS MGD_BRS_SUBS MGD_BRS_DUAL MGD_BRS_DRY \
 MGD_RS_MIN_K MGD_MIN_MB MGD_RR_WRAP_MIN OPUS_DX OPUS_PRELUDE OPUS_DX_MOD OPUS_DX_K OPUS_HEAP \
 OPUS_DX_BWD OPUS_DX_PHASE OPUS_AUTOTUNE OPUS_CP OPUS_CP_MAX_NC OPUS_CP_TIMING OPUS_AT_PROBE OPUS_AT_WARM \
 OPUS_AT_MARGIN OPUS_AG_STAGE OPUS_ISSUE OPUS_AT_WARM_COL OPUS_AT_FIXED OPUS_CP_MIN_NC OPUS_AT_RELEASE \
 OPUS_AG_PLANES OPUS_AG_GATE OPUS_AG_FORK"

median_ms () {   # $1 = rank-7 log -> median step time from iteration 3 on
    grep -a "elapsed time per iteration" "$1" 2>/dev/null | sed 's/.*per iteration (ms): \([0-9.]*\).*/\1/' \
        | tail -n +3 | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}'
}

plan_kind () {   # $1 arm name -> prints the kind the predictor picks
    local step="" a args
    for a in $ARMS; do
        if [ "${a##*:}" = base ] && [ -f "$ROOT/${a%%:*}/out_r7.log" ]; then step=$(median_ms "$ROOT/${a%%:*}/out_r7.log"); break; fi
    done
    if [ "${SW_MODEL:-27b}" = custom ]; then
        args="--hidden $VH_HIDDEN --ffn $VH_FFN --heads $VH_HEADS --kv $VH_KVG --layers $NL"
    else
        args="--model ${SW_MODEL:-27b} --layers $NL"
    fi
    python3 -I "$SW_ROOT/planner/planner.py" $args --mbs "$MBS" ${step:+--step-ms $step} > "$ROOT/plan_$1.md" 2>&1
    sed -n 's/^\*\*可部署配置：\([a-z]*\).*/\1/p' "$ROOT/plan_$1.md" | head -1
}

run_arm () {   # $1 NAME  $2 KIND  $3 ITERS
    local NAME=$1 KIND=$2 ITERS=$3 CHOSEN=""
    PORT=$((PORT + 7))
    export TRAIN_ITERS=$ITERS LOGDIR=$ROOT/$NAME DIST_PORT=$PORT
    rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
    unset $SW_VARS
    if [ "$KIND" = plan ]; then
        CHOSEN=$(plan_kind "$NAME"); KIND=${CHOSEN:-base}
        echo "[e2e] $NAME predictor -> $KIND (report: plan_$NAME.md)" >> $ROOT/driver.log
    fi
    export OPUS_HEAP=${VEH_HEAP:-6442450944}
    BRS="MGD_BRS=1 MGD_BRS_DEFER=1 MGD_BRS_SLOTS=4 MGD_BRS_MIN_MB=128 MGD_BRS_FORM=dx"
    ROW="MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32"
    case $KIND in
      base)  unset OPUS_HEAP ;;
      fnull) export MGD_BRS=1 MGD_BRS_DEFER=1 MGD_BRS_SLOTS=4 MGD_BRS_MIN_MB=1000000 MGD_BRS_FORM=dx \
                    MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=1000000 OPUS_DX_BWD=ag ;;
      fdk)   export $BRS ;;
      frow0) export MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=0 OPUS_DX_BWD=ag ;;   # fused GEMM+RS fwd + AG+dgrad bwd at o_proj+fc2, payload gate off, no standalone RS (11-09 decomposition)
      frowk) export MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=0 MGD_RS_MIN_K=2048 OPUS_DX_BWD=ag ;;   # same, fc2 only (K_r >= 2048 excludes o_proj)
      fallt) export $BRS $ROW OPUS_DX_BWD=t ;;
      fbag)  export $BRS $ROW OPUS_DX_BWD=ag ;;
      fbc)   export $BRS $ROW OPUS_DX_BWD=ag OPUS_CP=1 ;;
      fcq)   export $BRS $ROW OPUS_DX_BWD=ag OPUS_CP=1 OPUS_CP_MAX_NC=2048 ;;
      fcl)   export $BRS $ROW OPUS_DX_BWD=ag OPUS_CP=1 OPUS_CP_MIN_NC=2048 ;;
      fcth)  export $BRS $ROW OPUS_DX_BWD=ag OPUS_CP=1 OPUS_CP_TIMING=h ;;
      *) echo "[e2e] unknown kind $KIND" >> $ROOT/driver.log; return ;;
    esac
    echo "[e2e] $NAME kind=$KIND iters=$ITERS model=${SW_MODEL:-27b} MBS=$MBS NL=$NL start $(date +%T)" >> $ROOT/driver.log
    timeout ${E2E_TIMEOUT:-1500} mpirun --allow-run-as-root -np 8 -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
        bash $SW_ROOT/e2e/megatron/arm_inner.sh > "$ROOT/mpirun_$NAME.log" 2>&1
    local rc=$? n med fb val
    n=$(grep -ac "elapsed time per iteration" $LOGDIR/out_r7.log 2>/dev/null)
    med=$(median_ms $LOGDIR/out_r7.log)
    fb=$(grep -ac "FALLBACK" $LOGDIR/out_r0.log 2>/dev/null)
    val=$(grep -a 'validation loss at iteration.*on validation set' $LOGDIR/out_r7.log 2>/dev/null | tail -1 | sed 's/.*on validation set | lm loss value: \([0-9.E+-]*\).*/\1/')
    echo "[e2e] $NAME kind=$KIND rc=$rc iters=$n med(it3+)=$med fallbacks_r0=$fb val=$val $(date +%T)" >> $ROOT/driver.log
}

echo "[e2e] ROOT=$ROOT" | tee -a $ROOT/driver.log
for arm in ${ARMS:?set ARMS=\"name:kind ...\"}; do
    run_arm ${arm%%:*} ${arm##*:} ${ITERS:-16}
done
echo "E2E DONE" >> $ROOT/driver.log
python3 -I $SW_ROOT/tools/e2e_summary.py $ROOT | tee $ROOT/summary.md
