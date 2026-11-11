#!/bin/bash
# vLLM 归因实验：单独测 M16384 70B 形状的四个调用点，和算子级表对比
# M16384 = vLLM prefill 的 max_num_batched_tokens，对应 Megatron MBS4

set -e
cd "$(dirname "$0")/.."

# 70B 形状：h=8192, f=28672, heads=64, kv=8, hd=128
# TP=8，调用点形状：
#   QKV:    AG M=16384, K=8192, N=(64+2*8)*128/8 = 5120/8*80 = 80*64 = 5120 → GEMM
#   o_proj: GEMM M=16384, K=8192/8=1024, N=8192 → RS
#   fc1:    AG M=16384, K=8192, N=2*28672/8 = 7168 → GEMM
#   fc2:    GEMM M=16384, K=28672/8=3584, N=8192 → RS

echo "=========================================="
echo "vLLM 归因实验：M16384 70B 四个调用点"
echo "=========================================="
echo ""
echo "算子级表（cells.json）的数据："
python3 <<'EOF'
import json
cells = json.load(open('planner/data/cells.json'))
for c in cells:
    if c.get('model') == 'llama3-70b' and c.get('M') == 16384:
        op = c['op']
        if 'dx_b2b' in c and 'stock_b2b' in c:
            dx = c['dx_b2b']
            stock = c['stock_b2b']
            speedup = stock / dx
            print(f"  {op:8s}  stock={stock:6.2f}ms  dx={dx:6.2f}ms  speedup={speedup:.3f}x")
EOF
echo ""
echo "现在重测四个调用点（FG_OPS=64 连发，ABAB 两遍取中位）："
echo ""

# QKV: AG(16384, 8192) → GEMM(..., 5120)
echo "1. QKV (ag_up)"
mpirun -np 8 --allow-run-as-root -x AGF_M=16384 -x AGF_K=8192 -x AGF_NCS=5120 -x AGF_OPS=64 -x AGF_BURST=64 \
  python3 bench/ag_gemm.py 2>&1 | grep -E '\[agf\]|stock|dx-ag-gemm' | tail -5
echo ""

# fc1: AG(16384, 8192) → GEMM(..., 7168)
echo "2. fc1 (ag_gu)"
mpirun -np 8 --allow-run-as-root -x AGF_M=16384 -x AGF_K=8192 -x AGF_NCS=7168 -x AGF_OPS=64 -x AGF_BURST=64 \
  python3 bench/ag_gemm.py 2>&1 | grep -E '\[agf\]|stock|dx-ag-gemm' | tail -5
echo ""

# o_proj: GEMM(16384, 1024, 8192) → RS
echo "3. o_proj (rs)"
mpirun -np 8 --allow-run-as-root -x GRF_M=16384 -x GRF_K=1024 -x GRF_N=8192 -x GRF_OPS=64 -x GRF_BURST=64 \
  python3 bench/gemm_rs.py 2>&1 | grep -E '\[grf\]|stock|dx-gemm-rs' | tail -5
echo ""

# fc2: GEMM(16384, 3584, 8192) → RS
echo "4. fc2 (rs)"
mpirun -np 8 --allow-run-as-root -x GRF_M=16384 -x GRF_K=3584 -x GRF_N=8192 -x GRF_OPS=64 -x GRF_BURST=64 \
  python3 bench/gemm_rs.py 2>&1 | grep -E '\[grf\]|stock|dx-gemm-rs' | tail -5
echo ""

echo "=========================================="
echo "对比结论："
echo "如果新测的 speedup 和 cells.json 一致 → 车内环境不同，需修正 planner"
echo "如果新测的 speedup 比 cells.json 高很多 → cells.json 测错了，需重跑矩阵"
echo "=========================================="
