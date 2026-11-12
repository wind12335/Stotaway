"""Produce a compact two-GPU stock/DX table from complete run artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = []
    for path in args.files:
        data = json.loads(path.read_text())
        meta = data["metadata"]
        if meta["tp"] != 2 or len(data["results"]) != 8:
            raise ValueError(f"incomplete TP2 data: {path}")
        for item in data["results"]:
            stock = item["measurements"]["stock"]["median_ms"]
            dx = item["measurements"]["dx"]["median_ms"]
            if min(stock, dx) <= 0:
                raise ValueError("invalid latency")
            entries.append((item["model"], item["M"], item["site"], item["op"], stock, dx, stock / dx))
    lines = ["# 双卡 A800 算子对照（当前原型）", "",
             "BF16，TP2，CUDA_DEVICE_MAX_CONNECTIONS=1；每点 8 次新输入正确性检查，"
             "8 组、每组 16 次连发，取两卡较慢值的组中位数。speedup = stock / DX。",
             "这是一轮算子采样；两个载具仅指定矩阵尺寸，不代表完整模型或端到端结果。", "",
             "| 载具尺寸 | M | 调用点 | 算子 | stock ms | DX ms | speedup |",
             "|---|---:|---|---|---:|---:|---:|"]
    for model, m, site, op, stock, dx, speedup in sorted(entries):
        lines.append(f"| {model} | {m} | {site} | {op.upper()} | {stock:.4f} | {dx:.4f} | {speedup:.3f}× |")
    lines.extend(["", f"共 {len(entries)} 点，{sum(e[-1] > 1 for e in entries)} 点 speedup > 1；"
                  f"{sum(e[-1] <= 1 for e in entries)} 点 ≤ 1。", "",
                  "AG 表示 All-Gather + GEMM；RS 表示 GEMM + Reduce-Scatter。"
                  "时延来自不同执行路径，不代表完整训练或推理加速。", "",
                  "原始文件："])
    lines.extend(f"- `{p}`" for p in args.files)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output} points={len(entries)}")


if __name__ == "__main__":
    main()
