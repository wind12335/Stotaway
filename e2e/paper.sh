#!/bin/bash
# Reproduce the paper's tables (card-h = wall hours x 8 cards; 27B MBS8 arm ~12.5 min, 70B ~6.5 min), one experiment group per call. Each group prints what it measures,
# which table it fills and its cost BEFORE it starts (8-card groups need the whole node).
#
#   e2e/paper.sh <group>
#
# group        measures                                                         table                         cost
# ops          AG+GEMM and GEMM+RS, 11 models x M 8192/16384/32768 (99 cells)   results/算子级结果表.md        ~5 card-h
# stage1       predictor stage 1 leave-one-model-out on those cells (CPU)       端到端与预测器结果表 §4.1      0
# main         27B and 70B MBS8: base / fnull / fbag, ABAB                     端到端 §2.1                    ~16 card-h
# ablation     27B MBS8: base / fdk / fallt / fbag, ABAB                       端到端 §2.4                    ~13 card-h
# agchoice     fbag vs fbc at 27B MBS8/4, 70B MBS8/4/2 (stage-2 fit points)    端到端 §3                      ~18 card-h
# mbs          70B base vs fbag at MBS 1/2/4/8                                 端到端 §2.3                    ~8 card-h
# predictor    base / fnull / plan (predictor decides) at the 5 shapes        端到端 §4.3                    ~26 card-h
# moe-op       MoE layer, T 1024 / 4096, 4 routing skews                       MoE结果表 §1                   ~0.3 card-h
# moe-e2e      MoE vehicle 8 cards: base / sfl / gg, ABAB                      MoE结果表 §2                   ~1.5 card-h
# moe-1gpu     MoE vehicle 1 card: base / gg, ABAB                             MoE结果表 §2                   ~0.6 card-h
# dual16       see e2e/dual16/README.md (submitted as a 2-pod job)             端到端 §5                      ~42 card-h
# vllm         LLaMA3-70B TP8 prefill: stock / rccl / dx / plan, ABAB + gate   results/vLLM结果表.md          ~8 card-h
# vllm16       same on 2 nodes, TP8 x PP2: see e2e/dual16/README.md              results/vLLM结果表.md §4       ~17 card-h
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
source "$HERE/../env/env.sh"
say () { echo "[paper] $*"; }
G=${1:-}
case $G in
  ops)
    say "operator-level matrices (~5 card-hours, 40 min on 8 cards): AG+GEMM then GEMM+RS -> results/raw/ops_8card"
    bash $SW_ROOT/bench/run_ag_matrix.sh && bash $SW_ROOT/bench/run_rs_matrix.sh
    python3 -I $SW_ROOT/planner/extract.py && python3 -I $SW_ROOT/planner/gen_table.py ;;
  stage1)
    say "stage-1 leave-one-model-out (CPU) -> planner/data/planner_validation.md"
    python3 -I $SW_ROOT/planner/validate.py && tail -n 20 $SW_ROOT/planner/data/planner_validation.md ;;
  main)
    for m in 70b 27b; do say "main $m MBS8"; SW_MODEL=$m MBS=8 ITERS=18 \
      ARMS="b1:base n1:fnull f1:fbag b2:base n2:fnull f2:fbag" bash $HERE/megatron/run.sh; done ;;
  ablation)
    say "ablation 27B MBS8"; SW_MODEL=27b MBS=8 \
      ARMS="b1:base d1:fdk t1:fallt f1:fbag b2:base d2:fdk t2:fallt f2:fbag" bash $HERE/megatron/run.sh ;;
  agchoice)
    for p in 27b:8 27b:4 70b:8 70b:4 70b:2; do say "AG choice ${p%%:*} MBS${p##*:}"
      SW_MODEL=${p%%:*} MBS=${p##*:} ARMS="f1:fbag c1:fbc f2:fbag c2:fbc" bash $HERE/megatron/run.sh; done ;;
  mbs)
    for b in 1 2 4 8; do say "70B MBS$b"; SW_MODEL=70b MBS=$b ARMS="b1:base f1:fbag b2:base f2:fbag" bash $HERE/megatron/run.sh; done ;;
  predictor)
    for p in 27b:8 27b:4 70b:8 70b:4 70b:2; do say "predictor ${p%%:*} MBS${p##*:}"
      SW_MODEL=${p%%:*} MBS=${p##*:} ARMS="b1:base n1:fnull p1:plan b2:base n2:fnull p2:plan" bash $HERE/megatron/run.sh; done ;;
  moe-op)   bash $SW_ROOT/bench/run_moe_sf.sh ;;
  moe-e2e)  ARMS="b1:base s1:sfl g1:gg b2:base s2:sfl g2:gg" ITERS=10 bash $HERE/megatron/run_moe.sh ;;
  moe-1gpu) SW_NP=1 ARMS="b1:base g1:gg b2:base g2:gg" bash $HERE/megatron/run_moe.sh ;;
  dual16)   sed -n 1,40p $HERE/dual16/README.md ;;
  vllm)     bash $HERE/vllm/run.sh && bash $HERE/vllm/run_gate.sh ;;
  vllm16)   sed -n '/^## vLLM/,$p' $HERE/dual16/README.md ;;
  *) sed -n 2,20p "$0"; exit 1 ;;
esac
