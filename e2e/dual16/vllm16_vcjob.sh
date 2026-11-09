#!/bin/bash
# e2e/dual16/vllm16_vcjob.sh -- 双机 16 卡 vLLM prefill: LLaMA3-70B, TP8 x PP2 (one TP group per node,
# pipeline stage 0 on node 0 / stage 1 on node 1, the stage boundary crosses nodes over RDMA).
# Arms (same as the 8-card e2e/vllm/run.sh; each arm = one per-node mpirun -np 8 on BOTH nodes):
#   stock  original vLLM (TP AllReduce)
#   rccl   SP form (AR -> RS + AG), all four sites original RCCL
#   dx     SP form, all four sites overlap method
#   plan   SP form, each site as the predictor (stage 1) picks
# Cells per arm: gate (32 greedy tokens + logprobs, compared offline against stock), ttft (1 x M),
# ttft2 (1 x 2M), thru (16 x M). Default ABAB: stock rccl dx plan stock rccl dx plan.
#
# Submit the SAME command on both worker pods (shared storage):
#   bash <repo>/e2e/dual16/vllm16_vcjob.sh
# Env (identical on both workers):
#   RDMA_DISABLE=0|1           (default RDMA; 1 = TCP control)
#   VS_ARMS="stock rccl dx plan stock rccl dx plan"   VS_MNBT=16384 (prefill chunk M)  VS_REPS=3
#   VS_MODEL=<LLaMA3-70B HF dir>  (default $SW_VLLM_MODEL)
#   SW_VLLM_TGZ=<tarball of das vllm + shims>  only if the pod image has no vllm (see pack_vllm_env.sh)
# Cost: ~1.5 h for the default 8 arms + ~20 min natural-text gate (VS_GATE=0 skips it) = ~30 card-hours.
# Smoke first: VS_ARMS="stock dx" VS_REPS=1  (~35 min with the gate, ~9 card-hours)
# Results: results/raw/dual16_runs/vllm_<rdma|tcp>_<job>/ : vsp.csv, gate_*.json, report.md (node 0),
#          status_node<N>.log, <arm>_node<N>.log, rdma_verdict_node<N>.txt
if [ -z "${BASH_VERSION:-}" ]; then echo "[v16] re-exec bash"; exec bash "$0" "$@"; fi
set -uo pipefail
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
D16=$SW_ROOT/e2e/dual16
RES=${V16_RES:-$SW_ROOT/results/raw/dual16_runs}
mkdir -p "$RES"
( while sleep 300; do echo "[v16-hb] $(date +%T) alive node=${NODE:-?}"; done ) & HB=$!
trap 'kill $HB 2>/dev/null || true' EXIT
log() {
    local L="[v16][$(date +%T)] $*"
    echo "$L"
    [[ -n "${OUT:-}" && -d "${OUT:-/nonexistent}" ]] && echo "$L" >> "$OUT/status_node${NODE:-x}.log"
}

# ---- role (vcjob: <job>-worker-N) ----
HOST=$(hostname)
if [[ "$HOST" =~ ^(.*)-worker-([0-9]+)$ ]]; then NODE=${BASH_REMATCH[2]}
else
    log "FATAL: not a vcjob worker pod (hostname=$HOST); V16_FORCE_NODE=0|1 overrides for debugging"
    [[ -n "${V16_FORCE_NODE:-}" ]] && NODE=$V16_FORCE_NODE || exit 2
fi
JOB_PREFIX=${HOST%-worker-$NODE}
kill $HB 2>/dev/null; ( while sleep 300; do echo "[v16-hb] $(date +%T) alive node=$NODE"; done ) & HB=$!   # heartbeat with the node id

# ---- python env: das vllm in the pod? else unpack the shared tarball into /tmp and prepend ----
source "$SW_ROOT/env/env.sh"
# 11-02 first submission: the cluster image has torch / numpy but NO triton (the Megatron job gets it from
# the shared pysite_addons copy); the tarball therefore carries triton as well.
if ! python3 -c "import vllm, triton" 2>/dev/null; then
    TGZ=${SW_VLLM_TGZ:-$SW_ROOT/3rdparty/pyenv_vllm16.tar.gz}
    [[ -f "$TGZ" ]] || { log "FATAL: pod has no vllm/triton and no tarball $TGZ (run e2e/dual16/pack_vllm_env.sh on a node that has it)"; exit 5; }
    rm -rf /tmp/sw_vllm_site && mkdir -p /tmp/sw_vllm_site && tar xzf "$TGZ" -C /tmp/sw_vllm_site
    export PYTHONPATH=/tmp/sw_vllm_site:$PYTHONPATH
    log "vllm + triton unpacked from $TGZ -> /tmp/sw_vllm_site"
fi
[[ -n "${SW_PYSITE:-}" ]] && export PYTHONPATH=$PYTHONPATH:$SW_PYSITE
for m in torch numpy triton vllm flash_attn amdsmi transformers; do     # name the missing piece, not just "failed"
    python3 -c "import $m" 2>/dev/null || { log "FATAL: preflight: cannot import $m (PYTHONPATH=$PYTHONPATH)"; python3 -c "import $m" 2>&1 | tail -3; exit 5; }
done
python3 -c "import vllm, torch, triton; print('[v16] preflight vllm', vllm.__version__, 'torch', torch.__version__, 'triton', triton.__version__, triton.__file__)" \
    || { log "FATAL: preflight import failed"; exit 5; }
VS_MODEL=${VS_MODEL:-$SW_VLLM_MODEL}
[[ -f "$VS_MODEL/config.json" ]] || { log "FATAL: model dir $VS_MODEL has no config.json"; exit 5; }

# ---- RDMA (same recipe as the Megatron 16-card job) ----
source $D16/enable_rdma.sh
enable_rdma
IPCMD=$(command -v ip 2>/dev/null || true)
IFACE=""
[[ -n "$IPCMD" ]] && IFACE=$("$IPCMD" route 2>/dev/null | awk '/^default/ {print $5; exit}')
[[ -n "$IFACE" ]] && export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE}"
RDMA_ARM=$([[ "${RDMA_DISABLE:-0}" == "1" ]] && echo tcp || echo rdma)
RUN_TAG=vllm_${RDMA_ARM}_$(echo "$JOB_PREFIX" | awk -F- '{print $(NF-1)"-"$NF}')
OUT=$RES/$RUN_TAG; mkdir -p "$OUT"
log "node=$NODE host=$HOST out=$OUT model=$VS_MODEL"

# ---- master discovery (env -> DNS -> marker); storage is NOT live-shared between pods, so the
#      marker is a last resort and arm-to-arm sync uses the TCPStore port instead ----
MARK=$RES/.vcjob_v16_master
port_open() { timeout 2 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }
if [[ "$NODE" == "0" ]]; then
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -z "$IP" && -n "$IPCMD" ]] && IP=$("$IPCMD" -4 addr show "$IFACE" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
    echo "$IP $RUN_TAG" > "$MARK".tmp && mv "$MARK".tmp "$MARK"
else
    IP="${V16_MASTER_IP:-${MASTER_ADDR:-}}"
    for i in $(seq 1 360); do
        [[ -n "$IP" ]] && break
        D=$(getent hosts "${JOB_PREFIX}-worker-0" 2>/dev/null | awk '{print $1}' || true)
        [[ -n "$D" ]] && IP=$D && break
        [[ -s "$MARK" ]] && IP=$(awk -v t="$RUN_TAG" '$2==t {print $1}' "$MARK" 2>/dev/null || true)
        [[ -n "$IP" ]] && break
        sleep 5
    done
    [[ -n "$IP" ]] || { log "FATAL: master address not found (env/DNS/marker)"; exit 4; }
fi
log "master=$IP"

# ---- arms ----
export SW_NNODES=2 SW_NODE_RANK=$NODE MASTER_ADDR=$IP VS_MODEL
export VS_MNBT=${VS_MNBT:-16384} VS_REPS=${VS_REPS:-3} VS_OUT=$OUT
export DUSHMEM_SYMMETRIC_SIZE=${VS_HEAP:-4294967296}
ARMS=${VS_ARMS:-stock rccl dx plan stock rccl dx plan}
IDX=0
for ARM in $ARMS; do
    IDX=$((IDX+1))
    PORT=$((30500 + 7*IDX))
    if [[ $IDX == 1 ]]; then export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET; else export NCCL_DEBUG=WARN; unset NCCL_DEBUG_SUBSYS; fi
    if pgrep -f e2e/vllm/vllm_sp.py > /dev/null 2>&1; then log "FATAL: leftover vllm_sp.py on this node"; exit 3; fi
    if [[ "$NODE" != "0" ]]; then          # node 1 launches once node 0's TCPStore is listening
        W=0
        until port_open "$IP" $PORT; do sleep 5; W=$((W+5)); (( W >= 1800 )) && { log "[warn] port $PORT closed after 1800s, launching anyway"; break; }; done
    fi
    log "arm=$ARM ($IDX) port=$PORT start"
    VS_ARM=$ARM MASTER_PORT=$PORT timeout 2700 $SW_MPIRUN \
        -x VS_ARM -x VS_MNBT -x VS_REPS -x VS_OUT -x VS_MODEL -x MASTER_ADDR -x MASTER_PORT -x SW_NNODES -x SW_NODE_RANK \
        -x NCCL_DEBUG -x LD_LIBRARY_PATH \
        python3 $SW_ROOT/e2e/vllm/vllm_sp.py > "$OUT/${ARM}_${IDX}_node$NODE.log" 2>&1
    RC=$?
    log "arm=$ARM ($IDX) rc=$RC $(grep -a '\[vsp\] .* ttft r0' "$OUT/${ARM}_${IDX}_node$NODE.log" | tail -1)"
    if [[ $IDX == 1 ]]; then
        if grep -aqE "NET/IBext|NET/IB : Using.*shca" "$OUT/${ARM}_${IDX}_node$NODE.log"; then V=RDMA
        elif grep -aq "NET/Socket" "$OUT/${ARM}_${IDX}_node$NODE.log"; then V=TCP; else V=UNKNOWN; fi
        echo "transport=$V node=$NODE $(date)" >> "$OUT/rdma_verdict_node$NODE.txt"
        log "RDMA verdict node=$NODE: $V"
    fi
    sleep 20
done
# ---- natural-text output gate (every arm vs stock; the random-token gate above is too flat to judge) ----
if [[ "${VS_GATE:-1}" == "1" ]]; then
    for ARM in $(echo $ARMS | tr ' ' '\n' | awk '!s[$0]++'); do
        IDX=$((IDX+1)); PORT=$((30500 + 7*IDX))
        if [[ "$NODE" != "0" ]]; then W=0; until port_open "$IP" $PORT; do sleep 5; W=$((W+5)); (( W >= 1800 )) && break; done; fi
        log "gate arm=$ARM port=$PORT start"
        VS_ARM=$ARM MASTER_PORT=$PORT VS_OUT=$OUT/gate_text timeout 2700 $SW_MPIRUN \
            -x VS_ARM -x VS_MNBT -x VS_OUT -x VS_MODEL -x MASTER_ADDR -x MASTER_PORT -x SW_NNODES -x SW_NODE_RANK \
            -x NCCL_DEBUG -x LD_LIBRARY_PATH \
            python3 $SW_ROOT/e2e/vllm/vllm_gate.py > "$OUT/gate_${ARM}_node$NODE.log" 2>&1
        log "gate arm=$ARM rc=$?"; sleep 20
    done
fi
if [[ "$NODE" == "0" ]]; then
    python3 -I $SW_ROOT/tools/vllm_report.py "$OUT" "$VS_MNBT" > "$OUT/report.md" 2>&1
    python3 -I $SW_ROOT/tools/vllm_gate_report.py "$OUT/gate_text" "$VS_MNBT" >> "$OUT/report.md" 2>&1
    log "report -> $OUT/report.md"
fi
log "V16 DONE node=$NODE -> $OUT"
