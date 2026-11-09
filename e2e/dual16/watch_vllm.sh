#!/bin/bash
# Watch a 16-card vLLM job from another pod (shared storage; cross-pod writes can take ~10 min to show).
#   bash e2e/dual16/watch_vllm.sh [RUN_DIR]     default: newest results/raw/dual16_runs/vllm_* created after start
# Prints every new status line of both nodes, errors found in arm logs, and report.md once node 0 writes it.
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
RES=${V16_RES:-$SW_ROOT/results/raw/dual16_runs}
T0=$(date +%s)
D=${1:-}
while [ -z "$D" ]; do
    for c in $(ls -dt $RES/vllm_* 2>/dev/null); do
        [ $(stat -c %Y "$c") -ge $((T0 - 3600)) ] && { D=$c; break; }
    done
    [ -z "$D" ] && sleep 30
done
echo "[watch] run dir $D"
declare -A seen
errs=""
while true; do
    for n in 0 1; do
        f=$D/status_node$n.log
        [ -f "$f" ] || continue
        k=${seen[$n]:-0}
        tot=$(wc -l < "$f")
        [ "$tot" -gt "$k" ] && { tail -n +$((k + 1)) "$f" | sed "s/^/[node$n] /"; seen[$n]=$tot; }
    done
    for l in $D/*_node*.log; do
        [ -f "$l" ] || continue
        case " $errs " in *" $l "*) continue ;; esac
        if grep -aqE "Traceback|FATAL|out of memory|NCCL error|Segmentation" "$l" 2>/dev/null; then
            echo "[watch] ERROR in $(basename $l): $(grep -aE 'Error|FATAL|error' "$l" | grep -v Warning | tail -1 | cut -c1-200)"
            errs="$errs $l"
        fi
    done
    if [ -f $D/report.md ] && grep -q "V16 DONE" $D/status_node0.log 2>/dev/null; then
        echo "[watch] REPORT"; cat $D/report.md
        grep -q "V16 DONE" $D/status_node1.log 2>/dev/null && { echo "[watch] BOTH DONE"; exit 0; }
    fi
    sleep 30
done
