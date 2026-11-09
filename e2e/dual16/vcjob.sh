#!/bin/bash
# e2e/dual16/vcjob.sh -- 双机 16 卡端到端 (TP8 x DP2): overlap method + predictor.
# Cloned from t23_dual16_vcjob.sh (proven cross-pod orchestration, RDMA setup,
# per-node mpirun -np 8); only the arm table, inner script and defaults change.
#
# Submit the SAME command on both worker pods (shared /root/private_data):
#   bash <repo>/e2e/dual16/vcjob.sh
#
# Env (identical on both workers):
#   RDMA_DISABLE=0         (default RDMA; 1 = TCP control run)
#   T23_PHASES=mgd         (default here: vehicle only; "comm,mgd" adds the transport check)
#   T23_SHAPES=70b,27b     (default: both models in one job, 70b first; T23_SHAPE=70b = one model)
#   T23_ARMS default (11-01 predictor test): BASE.1,NULL.1,ALT.1,PLAN.1,BASE.2,NULL.2,ALT.2,PLAN.2
#   T23_PLAN_CFG=FBAG|FBC|FCQ|FCL|BASE   override the predictor's choice for PLAN arms (default: run the predictor)
#   MBS=8 TRAIN_ITERS=16
# Arms (TP group = the 8 cards of one node -> symmetric-heap engines usable;
#       the DP=2 cross-node grad reduce stays on RCCL/RDMA):
#   BASE  stock Megatron
#   FDK   DX direct-exchange RS for the SP reduce-scatter sites (mappings,layers only)
#   FBAG  FDK + fused row-parallel GEMM+RS forward + fused DX-AG backward  (innovation 1)
#   FBC   FBAG + fused column-parallel AG+GEMM (QKV, fc1): all four sites use the overlap method
#   FAT   online greedy dispatcher (KEPT ONLY for reference; biased on this platform, F44 -- not in the default arms)
#   NULL  same fork/heap/wrappers as FBAG, every op ineligible (overlap gain = FBAG vs NULL)
#   FCQ / FCL  FBAG + only QKV / only fc1 fused column-parallel AG+GEMM
#   ALT   the configuration the predictor did NOT pick (PLAN = FBC/FCQ/FCL -> FBAG; FBAG -> FBC; BASE -> FBAG)
#   PLAN  predictor (innovation 2): after BASE.1 on this node, planner/planner.py reads the shape,
#         MBS and the BASE.1 step time, and picks one of BASE/FBAG/FBC/FCQ/FCL; the decision and the
#         full planner report go to $OUT/plan_node<N>.md. Both nodes decide independently from their own
#         BASE.1 log (DP-synchronous step => same input); the two decisions are logged and must agree.
# Results: results/raw/dual16_runs/<rdma|tcp>_<job>/<ARM>/node<N>/out_r*.log
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[t23] 非 bash 入口, re-exec"; exec bash "$0" "$@"
fi
set -uo pipefail
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
source "$SW_ROOT/env/env.sh"
# cluster pods lack triton / einops / transformers / flash_attn ...: shared site-packages copy (3rdparty/pysite
# -> pysite_addons). 11-01 runs prepended it unconditionally; keep that as the 16-card default.
[[ -z "${SW_PYSITE:-}" && -d "$SW_ROOT/3rdparty/pysite" ]] && export SW_PYSITE=$(readlink -f "$SW_ROOT/3rdparty/pysite")
D16=$SW_ROOT/e2e/dual16
RES=${T23_RES:-$SW_ROOT/results/raw/dual16_runs}
mkdir -p "$RES"
BOOT_TS=$(date +%s)
# 心跳（平台只收 stdout）
( while sleep 300; do echo "[t23-hb] $(date +%T) alive node=${T23_NODE:-?}"; done ) &
HB=$!
cleanup() { kill $HB 2>/dev/null || true; }
trap 'echo "[t23] ERR line=$LINENO rc=$?"' ERR
trap cleanup EXIT
log() {
    local L="[t23][$(date +%T)] $*"
    echo "$L"
    # 11-01: mirror to the shared file system so progress can be watched from the notebook pod
    [[ -n "${OUT:-}" && -d "${OUT:-/nonexistent}" ]] && echo "$L" >> "$OUT/status_node${NODE:-x}.log"
}

# ---- 角色识别（vcjob: <job>-worker-N; 本 pod 8 卡, 16 卡=两 pod）----
HOST=$(hostname)
if [[ "$HOST" =~ ^(.*)-worker-([0-9]+)$ ]]; then
    NODE=${BASH_REMATCH[2]}
else
    log "FATAL: 非 vcjob worker pod（hostname=$HOST）。本脚本须在 SCNet 双机 16 卡"
    log "  vcjob 里跑（notebook pod 无 /dev/infiniband 且无对端）。调试单机可"
    log "  T23_FORCE_NODE=0 绕过（无跨节点阶段意义）。"
    [[ "${T23_FORCE_NODE:-}" != "" ]] && NODE=$T23_FORCE_NODE || exit 2
fi
log "node=$NODE hostname=$HOST boot=$(date -d @$BOOT_TS +%T)"
JOB_PREFIX=${HOST%-worker-$NODE}          # 两 worker 同前缀 ⇒ 免通信推导 RUN_TAG
# 平台注入 env 探测（Volcano/VC 系常见 MASTER_ADDR 类变量, 有则白拿）
log "env probe: $(env | grep -iE '^(MASTER|VC_|JOB_|WORKER|TASK|DLC_|KUBEFLOW|SCHED)' | tr '\n' ';' || true)"
kill $HB 2>/dev/null                      # 重启心跳带上 node 号
( while sleep 300; do echo "[t23-hb] $(date +%T) alive node=$NODE"; done ) & HB=$!

# ---- RDMA（两 worker 对称执行; enable_rdma 内建回退-不-die 语义）----
source $D16/enable_rdma.sh
enable_rdma
# 跨节点 TCP 面（TCPStore/RCCL 控制通道）绑定默认路由网卡
export HSA_FORCE_FINE_GRAIN_PCIE=1   # comm 阶段 torchrun 继承（GPU python 纪律）
# vcjob 镜像无 ip 命令（rc=127 噪音）——hostname -I 为主, ip 仅在存在时用
IPCMD=$(command -v ip 2>/dev/null || true)
IFACE=""
[[ -n "$IPCMD" ]] && IFACE=$("$IPCMD" route 2>/dev/null | awk '/^default/ {print $5; exit}')
[[ -n "$IFACE" ]] && export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE}"
export RDMA_ARM=$([[ "${RDMA_DISABLE:-0}" == "1" ]] && echo tcp || echo rdma)

PHASES=${T23_PHASES:-mgd}
# 11-01: shapes. T23_SHAPES=70b,27b runs both models back to back in ONE job (default); T23_SHAPE=70b
# alone still works. Every shape runs the same arm list. PLAN arms run the predictor (planner/planner.py).
T23_SHAPES=${T23_SHAPES:-${T23_SHAPE:-70b,27b}}
set_shape() {   # $1 = 27b | 70b ; same model args as the 8-card vehicle (e2e/megatron/arm_inner.sh)
    T23_SHAPE=$1
    if [[ "$1" == "70b" ]]; then
        export VH_HIDDEN=8192 VH_FFN=28672 VH_HEADS=64 VH_KVG=8 NL=${T23_NL_70B:-8}
    else
        unset VH_HIDDEN VH_FFN VH_HEADS VH_KVG; export NL=${T23_NL_27B:-32}
    fi
}
ARMS=${T23_ARMS:-BASE.1,NULL.1,ALT.1,PLAN.1,BASE.2,NULL.2,ALT.2,PLAN.2}
export TRAIN_ITERS=${TRAIN_ITERS:-16}   # 11-01: export (inner default is 25 otherwise)
export MBS=${MBS:-8}
BENCH_PORT=30101        # comm 阶段 torchrun rendezvous
MARK=$RES/.vcjob_t23_master   # 共享存储标记（含 master_ip + RUN_TAG）
MASTER_PORT=30100       # 标记新鲜度兜底探测口

port_any_open() { timeout 2 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }

# ---- 主地址发布/互等（run_dual_node_test.sh 实测配方: 新鲜度 OR 端口活）----
# 定律(17:50 实测): vcjob 两 worker 的 /root/private_data 非实时共享——
# 运行中新建文件对 peer 600s 不可见（作业前快照语义）。汇合/握手一律
# 不依赖运行期存储: env/DNS 优先, marker 仅后备(1800s); 跨节点同步交给
# torch 自带 rendezvous + 端口探测。
RUN_TAG=${RDMA_ARM}_$(echo "$JOB_PREFIX" | awk -F- '{print $(NF-1)"-"$NF}')
if [[ "$NODE" == "0" ]]; then
    OUT=$RES/$RUN_TAG; mkdir -p "$OUT/node0"
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$IP" && -n "$IPCMD" ]]; then
        IP=$("$IPCMD" -4 addr show "$IFACE" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 || true)
    fi
    echo "$IP $RUN_TAG" > "$MARK".tmp && mv "$MARK".tmp "$MARK"
    log "master marker -> $MARK ($IP $RUN_TAG); out=$OUT"
else
    IP="${T23_MASTER_IP:-${MASTER_ADDR:-}}"
    if [[ -n "$IP" ]]; then
        log "master 地址来自 env: $IP"
    else
        log "发现 master: DNS(${JOB_PREFIX}-worker-0) → marker(最长1800s)..."
        for i in $(seq 1 360); do
            if [[ -z "$IP" ]]; then
                D=$(getent hosts "${JOB_PREFIX}-worker-0" 2>/dev/null | awk '{print $1}' || true)
                [[ -n "$D" ]] && IP=$D && log "master via DNS: $IP"
            fi
            if [[ -z "$IP" && -s "$MARK" ]]; then
                C=$(awk -v t="$RUN_TAG" '$2==t {print $1}' "$MARK" 2>/dev/null || true)
                [[ -n "$C" ]] && IP=$C && log "master via marker: $IP"
            fi
            [[ -n "$IP" ]] && break
            sleep 5
        done
    fi
    [[ -n "$IP" ]] || { log "FATAL: master 地址不可得（DNS/marker/env 全失败）"; exit 4; }
    OUT=$RES/$RUN_TAG; mkdir -p "$OUT/node1"
    log "master=$IP run_tag=$RUN_TAG out=$OUT"
fi
export T23_MASTER_IP=$IP

# ================= 阶段 1: comm（跨节点传输判决数据）=================
if [[ ",$PHASES," == *,comm,* ]]; then
    log "=== comm 阶段（NCCL_DEBUG=INFO, 判据 NET/IBext_v8 vs NET/Socket）==="
    export NCCL_DEBUG=INFO
    mkdir -p "$OUT/comm/node$NODE"
    # bench 墙钟 540s < torchrun rendezvous 600s 窗（互等死锁修复的镜像纪律）
    timeout 540 torchrun --nnodes 2 --nproc-per-node 8 --node-rank "$NODE" \
        --master-addr "$IP" --master-port $BENCH_PORT \
        "$D16/comm_bench.py" --out "$OUT/comm.csv" \
        > "$OUT/comm/node$NODE/torchrun.log" 2>&1
    RC=$?
    export NCCL_DEBUG=WARN
    log "comm torchrun rc=$RC（124=超时: TCP 臂慢属预期, 已产出的行仍有效）"
    if [[ "$NODE" == "0" ]]; then
        # 传输判决（18:40 实测: 本版 RCCL 走 NET/IB : Using ... shca 而非
        # IBext_v8; NET/Plugin 两行是已知噪音）。RDMA=IB+shca, TCP=Socket。
        V=$(grep -m1 'NET/IB : Using' "$OUT"/comm/node0/torchrun.log 2>/dev/null \
            | grep -o 'shca' | head -1)
        [[ -n "$V" ]] && T="RDMA(IB/shca)" || T="TCP"
        grep -m1 'NET/Socket' "$OUT"/comm/node0/torchrun.log >/dev/null 2>&1 && T="TCP"
        echo "transport=$T rdma_arm=$RDMA_ARM run=$RUN_TAG $(date)" > "$OUT/comm_verdict.txt"
        log "comm 传输判决: $T -> $OUT/comm_verdict.txt"
    fi
fi

# ---- 预测器（创新点 2）：PLAN 臂的配置由 planner.py 决定 -------------------------
PLAN_CACHE=""
declare -A PLANC
plan_kind() {
    PLAN_CACHE=${PLANC[$T23_SHAPE]:-}
    # sets PLAN_CACHE (call without $(...) so the cache survives across PLAN arms)
    if [[ -n "${T23_PLAN_CFG:-}" ]]; then PLAN_CACHE=$T23_PLAN_CFG; PLANC[$T23_SHAPE]=$PLAN_CACHE; return; fi
    if [[ -n "$PLAN_CACHE" ]]; then return; fi
    # iteration lines are printed only by the LAST global rank = node1/out_r7. Both nodes read that file;
    # node0 sees it through the shared file system with a delay (10-26: up to ~10 min), so it polls up to
    # T23_PLAN_WAIT seconds, then falls back to the GEMM-time estimate (logged as base_step=none).
    local B7="$OUT/$T23_SHAPE/BASE.1/node1/out_r7.log" STEP="" ARGS W=0
    while [[ $(grep -ac "elapsed time per iteration" "$B7" 2>/dev/null || echo 0) -lt 4 && $W -lt ${T23_PLAN_WAIT:-900} ]]; do
        sleep 15; W=$((W+15))
    done
    [[ $W -gt 0 ]] && log "predictor node=$NODE waited ${W}s for $B7"
    if [[ -s "$B7" ]]; then
        STEP=$(grep -a "elapsed time per iteration" "$B7" | sed 's/.*per iteration (ms): \([0-9.]*\).*/\1/' \
               | tail -n +3 | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}')
    fi
    ARGS="--model $T23_SHAPE --mbs $MBS --layers $NL"
    [[ -n "$STEP" ]] && ARGS="$ARGS --step-ms $STEP"
    # planner reads GEMM times from planner/data/gemm_probe*.json (no GPU needed for 27b/70b at M 8192/16384/32768)
    mkdir -p "$OUT/$T23_SHAPE"
    python3 -I $SW_ROOT/planner/planner.py $ARGS \
        > "$OUT/$T23_SHAPE/plan_node$NODE.md" 2>&1 || log "[warn] planner rc=$? (see $T23_SHAPE/plan_node$NODE.md)"
    local CFG
    CFG=$(grep -o '可部署配置：[a-z]*' "$OUT/$T23_SHAPE/plan_node$NODE.md" | sed 's/可部署配置：//')
    case "$CFG" in
        base) PLAN_CACHE=BASE ;; fbag) PLAN_CACHE=FBAG ;; fbc) PLAN_CACHE=FBC ;;
        fcq) PLAN_CACHE=FCQ ;; fcl) PLAN_CACHE=FCL ;;
        *)  # planner could not run in the pod: use the choice it makes on the notebook pod with the
            # 8-card step times (11-01): 70b MBS8 fbc, 70b MBS4 fbag, 27b MBS8 fbag, 27b MBS4 base
            case "$T23_SHAPE:$MBS" in
                70b:8) PLAN_CACHE=FBC ;; 27b:4) PLAN_CACHE=BASE ;; *) PLAN_CACHE=FBAG ;;
            esac
            log "[warn] planner gave '$CFG' (step='$STEP'), using precomputed $PLAN_CACHE" ;;
    esac
    PLANC[$T23_SHAPE]=$PLAN_CACHE
    log "predictor node=$NODE shape=$T23_SHAPE mbs=$MBS layers=$NL base_step=${STEP:-none} -> $CFG ($PLAN_CACHE); report $OUT/$T23_SHAPE/plan_node$NODE.md"
}

# ================= 阶段 2: mgd（车辆臂, per-node mpirun 8）=================
if [[ ",$PHASES," == *,mgd,* ]]; then
    ITEMS=""
    for SH in ${T23_SHAPES//,/ }; do for A in ${ARMS//,/ }; do ITEMS="$ITEMS $SH:$A"; done; done
    log "=== mgd 阶段 shapes=$T23_SHAPES arms=$ARMS iters=$TRAIN_ITERS mbs=$MBS ==="
    ulimit -c 0
    NARMS=$(echo $ITEMS | wc -w)
    IDX=0
    for ITEM in $ITEMS; do
        IDX=$((IDX+1))
        SH=${ITEM%%:*}; ARM=${ITEM#*:}
        set_shape $SH
        # first arm of the job: RCCL INFO for the NET subsystem only = the RDMA verdict
        if [[ $IDX == 1 ]]; then export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET; else export NCCL_DEBUG=WARN; unset NCCL_DEBUG_SUBSYS; fi
        DIST_PORT=$((30200 + 7*IDX))
        # R12i 碰撞定律: 绝不骑在活车辆上发射
        if pgrep -f megatron/prelude.py > /dev/null 2>&1; then
            log "FATAL: 本节点残留 megatron/prelude.py 进程, 拒绝 arm=$ARM"; pgrep -af megatron/prelude.py | head -3; exit 3
        fi
        unset MGD_FUSED MGD_DRY MGD_AG MGD_AG_DRY MGD_AG_VERIFY MGD_AG_TIMING \
              MGD_AG_MIN_N MGD_AG_CHUNK MGD_AG_LANES MGD_AG_FORM MGD_AG_VSAMPLE \
              S17AG_SUB MGD_TP_TIMING MGD_BP MGD_RS_RELAY MGD_WRAP_TIMING MGD_RESYNC \
              MGD_COMM_STREAM MGD_RS_ATTR MGD_BRS MGD_BRS_MIN_MB MGD_BRS_FORM \
              MGD_BRS_DEFER MGD_BRS_MODS MGD_BRS_SLOTS MGD_BRS_SLOTS_DDO \
              MGD_BRS_CELLS MGD_BRS_DRY MGD_BRSPROBE MGD_AG_HEAPVIEW MGD_RS_MIN_K \
              MGD_MIN_MB \
              MGD_QUIET MGD_NOCHECK MGD_RERUNOFF MGD_CGRAPH MGD_MXOFF MGD_HWQ
        unset OPUS_FORK OPUS_HEAP OPUS_DX_BWD OPUS_AUTOTUNE OPUS_CP OPUS_CP_MAX_NC OPUS_AT_PROBE OPUS_AT_WARM OPUS_AT_MARGIN OPUS_AT_WARM_COL OPUS_AT_FIXED OPUS_AT_RELEASE OPUS_AG_STAGE OPUS_ISSUE OPUS_CP_MIN_NC OPUS_CP_TIMING OPUS_AG_PLANES OPUS_AG_GATE OPUS_AG_FORK
        BRSX="MGD_BRS=1 MGD_BRS_DEFER=1 MGD_BRS_SLOTS=4 MGD_BRS_MIN_MB=128 MGD_BRS_FORM=dx MGD_BRS_MODS=mappings,layers"
        KIND=${ARM%%.*}
        if [[ "$KIND" == "PLAN" || "$KIND" == "ALT" ]]; then
            plan_kind
            if [[ "$KIND" == "PLAN" ]]; then KIND=$PLAN_CACHE
            else case $PLAN_CACHE in FBAG) KIND=FBC ;; *) KIND=FBAG ;; esac; fi
            log "shape=$SH arm=$ARM -> $KIND (predictor chose $PLAN_CACHE)"
        fi
        case $KIND in
            BASE)  ;;
            FDK)   export OPUS_HEAP=6442450944 $BRSX ;;
            FBAG)  export OPUS_HEAP=6442450944 $BRSX \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32 OPUS_DX_BWD=ag ;;
            FBC)   export OPUS_HEAP=6442450944 $BRSX \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32 OPUS_DX_BWD=ag OPUS_CP=1 ;;
            FCQ)   export OPUS_HEAP=6442450944 $BRSX \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32 OPUS_DX_BWD=ag OPUS_CP=1 OPUS_CP_MAX_NC=2048 ;;
            FCL)   export OPUS_HEAP=6442450944 $BRSX \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32 OPUS_DX_BWD=ag OPUS_CP=1 OPUS_CP_MIN_NC=2048 ;;
            FAT)   export OPUS_HEAP=6442450944 $BRSX \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=32 OPUS_AUTOTUNE=1 OPUS_AT_WARM_COL=0 ;;
            NULL)  export OPUS_HEAP=6442450944 \
                        MGD_BRS=1 MGD_BRS_DEFER=1 MGD_BRS_SLOTS=4 MGD_BRS_MIN_MB=1000000 MGD_BRS_FORM=dx MGD_BRS_MODS=mappings,layers \
                        MGD_FUSED=1 MGD_RS_RELAY=dx MGD_MIN_MB=1000000 OPUS_DX_BWD=ag ;;
            *) log "unknown arm $ARM"; exit 2 ;;
        esac
        export NODE_RANK=$NODE DIST_PORT LOGDIR="$OUT/$SH/$ARM/node$NODE"
        mkdir -p "$LOGDIR"
        # 跨节点臂间同步去存储化: worker-1 等主节点 TCPStore 端口开 (rank0 建)
        # 才发射, 慢方零等; worker-0 是端口宿主无需等。
        if [[ "$NODE" != "0" ]]; then
            W=0
            until port_any_open "$T23_MASTER_IP" $DIST_PORT; do
                sleep 5; W=$((W+5))
                (( W >= 1800 )) && { log "[warn] 端口 $DIST_PORT 1800s 未开, 强行发射（对端或已失败）"; break; }
            done
        fi
        log "shape=$SH arm=$ARM ($IDX/$NARMS) nl=$NL port=$DIST_PORT node=$NODE $(date +%T)"
        # S75 fh-spy 取样器: 时间线采样(node0 健康期也静默, mtime 停滞检测必误报,
        # 故纯时间线): 每 5min 对全部 pretrain python 打 SIGUSR1(faulthandler 全线程
        # 栈->/tmp/r0_fh_<rank>.log), 同时记 ps stat/wchan(R自旋 vs S同步 vs D内核),
        # 快照拷 LOGDIR/fh_<ts>/。冻结态栈会连续多帧不动 = 定位死锁点。
        FH_PID=""
        if [[ "${MGD_FH_SPY:-0}" == "1" ]]; then
            (
              T0=$SECONDS; N=0
              while :; do
                sleep 300; N=$((N+1))
                pkill -USR1 -f pretrain_gpt.py 2>/dev/null
                sleep 3
                TS=$(date +%H%M%S)_f$N
                mkdir -p "$LOGDIR/fh_$TS"
                cp /tmp/r0_fh_*.log "$LOGDIR/fh_$TS/" 2>/dev/null
                for pd in $(pgrep -f megatron/prelude.py); do
                    ps -o pid,stat,wchan:28,etime --no-headers -p $pd
                done > "$LOGDIR/fh_$TS/ps_states.txt" 2>/dev/null
                log "[fh-spy] snap#$N age=$((SECONDS-T0))s -> $LOGDIR/fh_$TS"
              done
            ) &
            FH_PID=$!
        fi
        mpirun --allow-run-as-root -np 8 -mca coll ^hcoll \
            -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
            bash $D16/arm_inner_16.sh \
            > "$OUT/mpirun_${SH}_${ARM}_node$NODE.log" 2>&1
        MRC=$?
        [[ -n "$FH_PID" ]] && kill $FH_PID 2>/dev/null
        R7="$LOGDIR/out_r7.log"
        NIT=$(grep -ac "elapsed time per iteration" "$R7" 2>/dev/null)
        MED=$(grep -a "elapsed time per iteration" "$R7" 2>/dev/null | sed 's/.*per iteration (ms): \([0-9.]*\).*/\1/' \
              | tail -n +3 | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]}')
        log "shape=$SH arm=$ARM kind=$KIND node=$NODE mpirun rc=$MRC iters=${NIT:-0} med(it3+)=${MED:-none}（rc=1 teardown 竞态良性）$(date +%T)"
        if [[ $IDX == 1 ]]; then
            NETL=$(grep -ahE "NET/IBext|NET/IB : Using|NET/Socket|NET/Plugin.*shca|Using network" "$LOGDIR"/out_r*.log 2>/dev/null | sort | uniq -c | sort -rn | head -4 | tr '\n' ';')
            if grep -aqE "NET/IBext|NET/IB : Using.*shca" "$LOGDIR"/out_r*.log 2>/dev/null; then V=RDMA
            elif grep -aq "NET/Socket" "$LOGDIR"/out_r*.log 2>/dev/null; then V=TCP; else V=UNKNOWN; fi
            log "RDMA verdict node=$NODE: $V | $NETL"
            echo "transport=$V node=$NODE $(date) | $NETL" >> "$OUT/rdma_verdict_node$NODE.txt"
        fi
    done
fi

log "T23 DONE node=$NODE run=$RUN_TAG -> $OUT"
