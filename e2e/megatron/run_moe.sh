#!/bin/bash
# End-to-end MoE vehicle A/B: Qwen3-30B-A3B shape (128 experts top-8, hidden 2048, expert ffn 768),
# TP=1, EP=8 on 8 cards (or SW_NP=1: one card, 16 experts = the per-rank expert work of EP=8).
#
#   ARMS="b1:base s1:sfl b2:base s2:sfl" e2e/megatron/run_moe.sh          # 8 cards, ~6 min / arm
#   SW_NP=1 ARMS="b1:base g1:gg b2:base g2:gg" e2e/megatron/run_moe.sh    # 1 card
# Kinds:
#   base    original Megatron MoE (alltoall dispatcher: count a2a + .tolist() + varlen a2a + per-expert loop)
#   gg      expert MLP only -> host-sync-free dropless grouped GEMM (device-side counts)
#   sfl     whole MoE layer host-sync-free: SM-store dispatch + grouped experts + combine (8 cards only)
#   timing / ggt / sflt   the same with per-phase dispatcher timing (diagnostic)
set -u
source "$(dirname "$(readlink -f "$0")")/../../env/env.sh"
NP=${SW_NP:-8}
ROOT=${SW_OUT:-$SW_ROOT/results/raw/e2e_moe/np${NP}_$(date +%m%d_%H%M)}
mkdir -p "$ROOT"
export MBS=${MBS:-1} NL=${NL:-12}
if [ "$NP" = 1 ]; then export MOE_E=${MOE_E:-16} MOE_EP=1; else export MOE_E=${MOE_E:-128} MOE_EP=$NP; fi
export OPUS_MOE_TSKIP=$((6 * NL)) OPUS_MOE_TFLUSH=$((4 * NL))
P=$SW_ROOT/e2e/megatron/moe_prelude.py
PORT=${MOE_PORT:-33000}
for arm in ${ARMS:?set ARMS=\"name:kind ...\"}; do
    NAME=${arm%%:*}; KIND=${arm##*:}; PORT=$((PORT + 7))
    export TRAIN_ITERS=${ITERS:-12} LOGDIR=$ROOT/$NAME DIST_PORT=$PORT
    rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
    unset OPUS_HEAP OPUS_MOE_SF OPUS_MOE_TIMING OPUS_PRELUDE OPUS_MOE_GG OPUS_MOE_SFL OPUS_MOE_GG_MAXROWS
    case $KIND in
      base)   ;;
      timing) export OPUS_MOE_TIMING=1 OPUS_PRELUDE=$P ;;
      gg)     export OPUS_MOE_GG=1 OPUS_MOE_GG_MAXROWS=${GG_MAXROWS:-98304} OPUS_PRELUDE=$P ;;
      ggt)    export OPUS_MOE_GG=1 OPUS_MOE_GG_MAXROWS=${GG_MAXROWS:-98304} OPUS_MOE_TIMING=1 OPUS_PRELUDE=$P ;;
      sfl)    export OPUS_HEAP=${VEH_HEAP:-8589934592} OPUS_MOE_SFL=1 OPUS_PRELUDE=$P ;;
      sflt)   export OPUS_HEAP=${VEH_HEAP:-8589934592} OPUS_MOE_SFL=1 OPUS_MOE_TIMING=1 OPUS_PRELUDE=$P ;;
      *) echo "[moe] unknown kind $KIND" >> $ROOT/driver.log; continue ;;
    esac
    echo "[moe] $NAME kind=$KIND np=$NP MBS=$MBS NL=$NL start $(date +%T)" >> $ROOT/driver.log
    timeout ${MOE_TIMEOUT:-1500} mpirun --allow-run-as-root -np $NP -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
        bash $SW_ROOT/e2e/megatron/moe_inner.sh > "$ROOT/mpirun_$NAME.log" 2>&1
    rc=$?
    L=$LOGDIR/out_r$((NP - 1)).log
    n=$(grep -ac "elapsed time per iteration" $L 2>/dev/null)
    med=$(grep -a "elapsed time per iteration" $L 2>/dev/null | sed 's/.*per iteration (ms): \([0-9.]*\).*/\1/' | tail -n +3 | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}')
    loss=$(grep -a "elapsed time per iteration" $L 2>/dev/null | tail -1 | sed 's/.*lm loss: \([0-9.E+-]*\).*/\1/')
    echo "[moe] $NAME kind=$KIND rc=$rc iters=$n med(it3+)=$med last_loss=$loss $(date +%T)" >> $ROOT/driver.log
done
echo "MOE DONE" >> $ROOT/driver.log
python3 -I $SW_ROOT/tools/e2e_summary.py $ROOT | tee $ROOT/summary.md
