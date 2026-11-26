#!/bin/bash
# vLLM prefill end-to-end A/B with the OPRS runtime: e2e/vllm/vllm_oprs.py, one arm = one mpirun process.
#
#   NP=8 PLANS=results/<date>/plans/llama70_async_tp e2e/vllm/run_oprs.sh            timing cells, arms ABAB
#   NP=8 PLANS=... VS_GATE=1 e2e/vllm/run_oprs.sh                                    natural-text gate (+ prompt logprobs)
#   NP=1 VS_TINY=1 VS_MNBT=2048 VS_REPS=2 PLANS=results/<date>/plans/<tp1 plan set> e2e/vllm/run_oprs.sh
#                                                                                 single-card plumbing check
#
# Arm = name:kind. What each of the four tensor-parallel call sites (qkv / proj / fc1 / fc2) does:
#   stock          original vLLM (TP AllReduce): no hook, no heap, no plan
#   sp_rccl        sequence-parallel form, every site on RCCL (AG -> GEMM / GEMM -> RS). This is the Null
#                  plan (null.json): hook, heap and contexts present, nothing replaced
#   all_deo        DEO at every site (always_deo.json)
#   all_<B>        the external backend B (async_tp | triton_dist) at every site (always_<B>.json)
#   oprs           per call site serial / DEO / B as the plan says (oprs_<B>.json)
#   oprs_deo_only  two-way plan, serial / DEO (oprs_deo_only.json)
#   name:<plan>    an explicit plan file
# PLANS = a plan-set directory of tools/oprs_plans.py for this workload (--workload llama70_infer --tp N
#         --rows M); the backend B is read from its manifest. The plan's rows must be VS_MNBT.
# Env: VS_MNBT (16384), VS_REPS (3), VS_MODEL (default SW_VLLM_MODEL), VS_WORKLOAD (llama70_infer),
#      VS_CONFIG (model config of an hf_infer workload), VS_METHODS, VS_ARMS, VS_HEAP (default
#      tools/oprs_heap.py over PLANS), VS_GPU_UTIL (0.85), VS_MAX_LEN, VS_TIMEOUT (1800), VS_OUT, VS_TAG,
#      VS_GATE (+ VS_GATE_NP 4, VS_GATE_OSL 48), VS_TINY=1 (tiny model, dummy weights, hf_infer).
# A non-stock arm that never runs the bucket forward, or whose plan replaces a site and nothing was
# replaced, exits non-zero: BROKEN is never reported as slow.
set -u
HERE=$(dirname "$(readlink -f "$0")")
export SW_NP=${SW_NP:-${NP:-8}}                 # before env/env.sh
source "$HERE/../../env/env.sh"
[ -n "${SW_PYSITE:-}" ] && export PYTHONPATH=$PYTHONPATH:$SW_PYSITE
TP=$SW_NP
export VS_MNBT=${VS_MNBT:-16384} VS_REPS=${VS_REPS:-3}
export VS_WORKLOAD=${VS_WORKLOAD:-llama70_infer} VS_METHODS=${VS_METHODS:-} VS_GPU_UTIL=${VS_GPU_UTIL:-0.85}
export VS_MAX_LEN=${VS_MAX_LEN:-} VS_TIMEOUT=${VS_TIMEOUT:-1800} VS_GATE=${VS_GATE:-}
export VS_GATE_NP=${VS_GATE_NP:-4} VS_GATE_OSL=${VS_GATE_OSL:-48} VS_LOAD_FORMAT=${VS_LOAD_FORMAT:-auto}
export VS_CONFIG=${VS_CONFIG:-}
BACKEND=""
if [ -n "${PLANS:-}" ]; then
    [ -f "$PLANS/manifest.json" ] || { echo "PLANS=$PLANS is not a plan-set directory (no manifest.json)"; exit 2; }
    # one deployment revolves around one external backend B
    BACKEND=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['backend'])" "$PLANS/manifest.json") || exit 2
    export PLANS
    # the heap is reserved before vLLM sizes its memory: one size for every arm of the comparison
    export DUSHMEM_SYMMETRIC_SIZE=${VS_HEAP:-$(python3 -I "$HERE/../../tools/oprs_heap.py" "$PLANS" --deo-bwd stock)}
    [ "$BACKEND" = triton_dist ] && export OPRS_TRITON_DIST_ROOT=${OPRS_TRITON_DIST_ROOT:-/root/private_data/tri-dist-hygon}
fi
if [ -n "${VS_TINY:-}" ]; then                 # single-card plumbing vehicle (tools/oprs_tiny_model.py)
    export VS_MODEL=${VS_MODEL:-/tmp/oprs_tiny_llama}
    [ -f "$VS_MODEL/config.json" ] || python3 -I "$HERE/../../tools/oprs_tiny_model.py" "$VS_MODEL" || exit 2
    export VS_LOAD_FORMAT=dummy VS_WORKLOAD=hf_infer VS_CONFIG=$VS_MODEL/config.json
    export VS_SEEDED_WEIGHTS=${VS_SEEDED_WEIGHTS:-1}     # sane random weights: the dummy ones make the gate vacuous
    [ -n "$VS_MAX_LEN" ] || export VS_MAX_LEN=$((2 * VS_MNBT + 64))
fi
export VS_MODEL=${VS_MODEL:-$SW_VLLM_MODEL}
[ -d "$VS_MODEL" ] || { echo "model dir not found: $VS_MODEL (set SW_VLLM_MODEL or VS_MODEL)"; exit 1; }
if [ -z "${VS_ARMS:-}" ]; then
    VS_ARMS=""
    for k in 1 2; do for a in s:stock r:sp_rccl d:all_deo "b:always_${BACKEND:?PLANS needed for the default arms}" o:oprs; do
        VS_ARMS="$VS_ARMS ${a%%:*}$k:${a#*:}"; done; done
fi
TAG=${VS_TAG:-$(basename "${PLANS:-stock}")}
SUFFIX=""; [ -n "$VS_GATE" ] && SUFFIX="_gate"
export VS_OUT=${SW_OUT:-$SW_ROOT/results/bw1000_oprs_$(date +%Y%m%d)/raw/e2e_inference/${TAG}_tp${TP}_M${VS_MNBT}${SUFFIX}_$(date +%m%d_%H%M)}
[ -e "$VS_OUT/driver.log" ] && { echo "refusing to reuse $VS_OUT (history is kept)"; exit 2; }
mkdir -p "$VS_OUT"
L=$VS_OUT/driver.log
echo "[e2e] ROOT=$VS_OUT model=$VS_MODEL tp=$TP M=$VS_MNBT workload=$VS_WORKLOAD plans=${PLANS:-<none>} backend=${BACKEND:-<none>} heap=${DUSHMEM_SYMMETRIC_SIZE:-<env default>} gate=${VS_GATE:-0} reps=$VS_REPS load=$VS_LOAD_FORMAT" | tee -a "$L"
PORT=${E2E_PORT:-29880}
for ARM in $VS_ARMS; do
    NAME=${ARM%%:*}; KIND=${ARM#*:}; KIND=${KIND:-$NAME}
    PORT=$((PORT + 3)); start=$(date +%s)
    echo "[e2e] $NAME kind=$KIND start $(date +%T)" | tee -a "$L"
    VS_ARM=$ARM MASTER_PORT=$PORT timeout "$VS_TIMEOUT" $SW_MPIRUN -x VS_ARM -x PLANS -x VS_WORKLOAD -x VS_CONFIG \
        -x VS_METHODS -x VS_MNBT -x VS_REPS -x VS_OUT -x VS_MODEL -x VS_LOAD_FORMAT -x VS_GPU_UTIL -x VS_MAX_LEN \
        -x VS_GATE -x VS_GATE_NP -x VS_GATE_OSL -x VS_SEEDED_WEIGHTS -x OPRS_TRITON_DIST_ROOT -x MASTER_PORT \
        python3 $HERE/vllm_oprs.py > "$VS_OUT/${NAME}_$(date +%H%M%S).log" 2>&1
    rc=$?
    echo "[e2e] $NAME kind=$KIND rc=$rc end $(date +%T) wall=$(( $(date +%s) - start ))s" | tee -a "$L"
    sleep ${GAP:-20}
done
echo "E2E DONE" | tee -a "$L"
python3 -I "$HERE/../../tools/oprs_tables.py" inference "$VS_OUT" "$VS_MNBT" | tee "$VS_OUT/summary.md"
