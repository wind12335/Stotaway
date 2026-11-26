#!/bin/bash
# End-to-end Megatron TRAINING A/B with the OPRS runtime (src/stowaway/oprs): the dense shape vehicle of
# e2e/megatron (TP + SP, bf16, mock data), one arm = one mpirun process = independent processes.
#
#   NP=8 SW_MODEL=27b PLANS=results/<date>/plans/bw27b_async_tp \
#   ARMS="s1:serial n1:null d1:always_deo a1:always_async_tp o1:oprs s2:serial o2:oprs" e2e/megatron/run_oprs.sh
#
# Arm = name:kind. What each of the four tensor-parallel call sites (qkv / proj / fc1 / fc2) does:
#   serial          original Megatron: no OPRS hook, no symmetric heap (the reference of every ratio)
#   null            plan null.json: same hook and heap, nothing replaced (their cost)
#   always_deo      plan always_deo.json: DEO everywhere it can run
#   always_<B>      plan always_<B>.json: the external backend B (async_tp | triton_dist) everywhere it can run
#   oprs            plan oprs_<B>.json: per call site serial / DEO / B, whichever the predictor says
#   oprs_deo_only   plan oprs_deo_only.json: two-way plan (serial / DEO)
#   name:<path>     an explicit plan file
# Plans come from tools/oprs_plans.py (one directory per deployment = per external backend B; the
# manifest records the B). A call site whose shape is not in the plan takes the original path and the
# fallback is counted: an arm whose plan is not `serial` and that reports no hit is BROKEN, not slow.
# Shape: SW_MODEL=27b|70b|custom (+ VH_HIDDEN VH_FFN VH_HEADS VH_KVG NL; then set OPRS_WORKLOAD=custom_train).
#        MBS (8), SEQ (4096), ACC (8 micro-batches per iteration), ITERS (16), E2E_PORT (35000),
#        OPRS_HEAP (default tools/oprs_heap.py over PLANS), OPRS_DEO_BWD (ag).
# Output: $ROOT/driver.log (one line per arm run), $ROOT/e2e_training.csv + summary.md (tools/oprs_tables.py),
#         $ROOT/<name>/out_r*.log, step_ms.txt (every step of the last rank).
# Cost per arm on 8 cards, ITERS=16: 27B MBS8 ~12.5 min, 70B MBS8 ~6.5 min (less at smaller MBS).
set -u
HERE=$(dirname "$(readlink -f "$0")")
export SW_NP=${SW_NP:-${NP:-8}}                 # before env/env.sh
source "$HERE/../../env/env.sh"
export TP=$SW_NP                            # exported: mpirun -x TP carries it to the inner script
OPRS_RESULTS=${OPRS_RESULTS:-$SW_ROOT/results/bw1000_oprs_$(date +%Y%m%d)}
ROOT=${SW_OUT:-$OPRS_RESULTS/raw/e2e_training/${SW_MODEL:-27b}_tp${TP}_mbs${MBS:-8}_$(date +%m%d_%H%M)}
[ -e "$ROOT/driver.log" ] && { echo "refusing to reuse $ROOT (history is kept)"; exit 2; }
mkdir -p "$ROOT"
export MBS=${MBS:-8}
DEFAULT_WORKLOAD=custom_train
case ${SW_MODEL:-27b} in
  27b) export VH_HIDDEN=5120 VH_FFN=17408 VH_HEADS=40 VH_KVG=8 NL=${NL:-32}; DEFAULT_WORKLOAD=bw27b_train ;;
  70b) export VH_HIDDEN=8192 VH_FFN=28672 VH_HEADS=64 VH_KVG=8 NL=${NL:-8}; DEFAULT_WORKLOAD=bw70b_train ;;
  custom) : "${VH_HIDDEN:?custom model: set VH_HIDDEN VH_FFN VH_HEADS VH_KVG NL and OPRS_WORKLOAD=custom_train}" ;;
  *) echo "unknown SW_MODEL ${SW_MODEL:-27b} (27b | 70b | custom)"; exit 2 ;;
esac
export OPRS_WORKLOAD=${OPRS_WORKLOAD:-$DEFAULT_WORKLOAD}
export SEQ=${SEQ:-4096} ACC=${ACC:-8} VEH_RECOMP=${VEH_RECOMP:-full} OPRS_QUIET=${OPRS_QUIET:-}
export OPRS_TOKENIZER=${OPRS_TOKENIZER:-} VOCAB=${VOCAB:-248320}
PORT=${E2E_PORT:-35000}
# one heap size for every arm that allocates (the serial arm has no heap at all)
export OPRS_HEAP=${OPRS_HEAP:-}
if [ -z "$OPRS_HEAP" ] && [ -n "${PLANS:-}" ]; then
    HEAP=$(python3 -I "$HERE/../../tools/oprs_heap.py" "$PLANS" ${OPRS_DEO_BWD:+--deo-bwd "$OPRS_DEO_BWD"}) || exit 2
    export OPRS_HEAP=$HEAP
    echo "[e2e] heap from $PLANS: $OPRS_HEAP bytes" | tee -a "$ROOT/driver.log"
fi
BACKEND=""
if [ -n "${PLANS:-}" ]; then    # one deployment revolves around one external backend B: oprs_<B>.json
    BACKEND=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['backend'])" "$PLANS/manifest.json") || exit 2
fi

run_arm () {   # $1 NAME  $2 KIND
    local NAME=$1 KIND=$2 PLAN="" start rc last med hits fb
    PORT=$((PORT + 7))
    export TRAIN_ITERS=${ITERS:-16} LOGDIR=$ROOT/$NAME DIST_PORT=$PORT
    rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
    export OPRS_PLAN= OPRS_METHODS=${OPRS_METHODS:-} OPRS_DEO_BWD=${OPRS_DEO_BWD:-ag}
    case $KIND in
      serial|*:*) ;;                 # the reference, or an explicit plan file (name:<path>)
      *) : "${PLANS:?arm kind $KIND needs PLANS = a plan-set directory made by tools/oprs_plans.py}" ;;
    esac
    case $KIND in
      serial) ;;
      oprs) PLAN=$PLANS/oprs_${BACKEND:?}.json ;;             # the predicted plan of this deployment
      null|always_deo|oprs_deo_only|always_*) PLAN=$PLANS/$KIND.json ;;
      *:*)  PLAN=${KIND#*:}; KIND=plan ;;
      *) echo "[e2e] unknown kind $KIND" | tee -a "$ROOT/driver.log"; return ;;
    esac
    if [ -n "$PLAN" ]; then
        [ -f "$PLAN" ] || { echo "[e2e] $NAME kind=$KIND MISSING PLAN $PLAN" | tee -a "$ROOT/driver.log"; return; }
        export OPRS_PLAN=$PLAN
    fi
    start=$(date +%s)
    echo "[e2e] $NAME kind=$KIND plan=${PLAN:-none} model=${SW_MODEL:-27b} tp=$TP MBS=$MBS NL=$NL start $(date +%T)" | tee -a "$ROOT/driver.log"
    timeout ${E2E_TIMEOUT:-1500} $SW_MPIRUN -x TP -x OPRS_PLAN -x OPRS_WORKLOAD -x OPRS_METHODS -x OPRS_DEO_BWD \
        -x OPRS_TOKENIZER -x VOCAB -x VH_HIDDEN -x VH_FFN -x VH_HEADS -x VH_KVG -x NL -x MBS -x SEQ \
        -x ACC -x VEH_RECOMP -x OPRS_QUIET -x LOGDIR -x DIST_PORT -x TRAIN_ITERS \
        bash $SW_ROOT/e2e/megatron/oprs_inner.sh > "$ROOT/mpirun_$NAME.log" 2>&1
    rc=$?
    last=$LOGDIR/out_r$((TP - 1)).log
    # mpirun's own status is 1 for every arm that brought up the engine world (the rank leaves without
    # MPI_Finalize: an "abnormal termination" to prterun). The arm's real exit code is the one its python
    # printed after returning; none = it was killed before it could.
    py_rc=$(grep -a "^EXIT " "$last" 2>/dev/null | tail -1 | awk '{print $2}')
    grep -a "elapsed time per iteration" "$last" 2>/dev/null | sed 's/.*per iteration (ms): \([0-9.]*\).*/\1/' > "$LOGDIR/step_ms.txt"
    med=$(tail -n +3 "$LOGDIR/step_ms.txt" | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}')
    hits=$(grep -ac "OPRS_HIT " "$LOGDIR/out_r0.log" 2>/dev/null)
    fb=$(grep -ac "OPRS_FALLBACK " "$LOGDIR/out_r0.log" 2>/dev/null)
    echo "[e2e] $NAME kind=$KIND rc=$rc py_rc=${py_rc:-none} iters=$(wc -l < "$LOGDIR/step_ms.txt") med(it3+)=${med:-none} hit_lines_r0=${hits:-0} fallback_lines_r0=${fb:-0} end $(date +%T) wall=$(( $(date +%s) - start ))s" | tee -a "$ROOT/driver.log"
    sleep ${GAP:-5}
}

echo "[e2e] ROOT=$ROOT plans=${PLANS:-<none>} heap=${OPRS_HEAP:-<none>}" | tee -a "$ROOT/driver.log"
for arm in ${ARMS:?set ARMS=\"name:kind ...\"}; do
    run_arm "${arm%%:*}" "${arm#*:}"
done
echo "E2E DONE" | tee -a "$ROOT/driver.log"
python3 -I "$HERE/../../tools/oprs_tables.py" training "$ROOT" | tee "$ROOT/summary.md"
