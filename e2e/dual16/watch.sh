#!/bin/bash
# 从 notebook pod 盯 16 卡作业（共享文件系统；跨 pod 写入可能延迟到 ~10 分钟可见）。
# 用法：bash e2e/dual16/watch.sh [RUN_DIR]   不给则取 results/raw/dual16_runs 下最新的 rdma_*/tcp_* 目录
# 每 60 s 输出一次有变化的状态：两个节点的 status 行、各臂 r7 的迭代进度、报错关键字。
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
RES=${T23_RES:-$SW_ROOT/results/raw/dual16_runs}
D=${1:-$(ls -dt $RES/rdma_* $RES/tcp_* 2>/dev/null | head -1)}
[ -z "$D" ] && { echo "no run dir yet under $RES"; exit 1; }
echo "watching $D"
prev=""
while true; do
  cur=$( { for n in 0 1; do tail -n 2 $D/status_node$n.log 2>/dev/null; done
           for a in $D/*/*/node*; do [ -f $a/out_r7.log ] || continue
             it=$(grep -ac "elapsed time per iteration" $a/out_r7.log)
             err=$(grep -alE "Traceback|Error|FAULT|NaN|nan iter" $a/out_r*.log 2>/dev/null | wc -l)
             echo "$(basename $(dirname $(dirname $a)))/$(basename $(dirname $a))/$(basename $a) iters=$it errfiles=$err"; done
           ls $D/*/plan_node*.md 2>/dev/null | while read p; do echo "$(basename $(dirname $p))/$(basename $p): $(grep -o '可部署配置：[a-z]*' $p)"; done; } )
  [ "$cur" != "$prev" ] && { echo "---- $(date +%T)"; echo "$cur"; prev=$cur; }
  grep -q "T23 DONE" $D/status_node0.log 2>/dev/null && grep -q "T23 DONE" $D/status_node1.log 2>/dev/null && { echo "BOTH DONE"; exit 0; }
  sleep 60
done
