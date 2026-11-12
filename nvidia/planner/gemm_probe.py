"""Single-GPU GEMM calibration for the DCU vehicle shapes at a chosen TP size.

Measurements are raw A800 inputs, not a fitted planner or TP8 performance.
The row count is seq_length * micro_batch_size. No model weights are loaded.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
from pathlib import Path

# This directory contains platform.py; keep it from shadowing Python's stdlib
# module when torch imports platform as part of its initialization.
sys.path = [path for path in sys.path if Path(path or os.getcwd()).resolve() != Path(__file__).resolve().parent]

import torch
import triton


MODELS = {"27b_vehicle": (5120, 17408, 40, 8),
          "70b_vehicle": (8192, 28672, 64, 8)}


def sites(hidden, ffn, heads, kv, rows, tp):
    head_dim = hidden // heads
    if hidden % heads or hidden % tp or ffn % tp or (heads + 2 * kv) * head_dim % tp:
        raise ValueError("model dimensions cannot be sharded over this TP size")
    return (("qkv", "ag_up", rows, hidden, (heads + 2 * kv) * head_dim // tp),
            ("o_proj", "rs", rows, hidden // tp, hidden),
            ("fc1_gate_up", "ag_gu", rows, hidden, 2 * ffn // tp),
            ("fc2", "rs", rows, ffn // tp, hidden))


def measure(a, b, out, warmup, singles, repeats, batch):
    for _ in range(warmup):
        torch.matmul(a, b, out=out)
    torch.cuda.synchronize()
    single = []
    for _ in range(singles):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b, out=out)
        end.record()
        end.synchronize()
        single.append(start.elapsed_time(end))
    batched = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            torch.matmul(a, b, out=out)
        end.record()
        end.synchronize()
        batched.append(start.elapsed_time(end) / batch)
    return single, batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--singles", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.tp, args.warmup, args.singles, args.repeats, args.batch) <= 0:
        parser.error("all counts must be positive")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    report = {"metadata": {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
                           "torch": torch.__version__, "cuda_build": torch.version.cuda,
                           "triton": triton.__version__, "tp": args.tp, "rows": args.rows,
                           "dtype": "bfloat16", "warmup": args.warmup, "singles": args.singles,
                           "repeats": args.repeats, "batch": args.batch,
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "scope": "single-GPU GEMM for vehicle dimensions; not a planner profile"},
              "results": []}
    for model, dimensions in MODELS.items():
        for name, op, m, k, n in sites(*dimensions, args.rows, args.tp):
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
            out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            single, batched = measure(a, b, out, args.warmup, args.singles, args.repeats, args.batch)
            median = statistics.median(batched)
            record = {"model": model, "site": name, "op": op, "M": m, "K": k, "N": n,
                      "payload_MB": (m * n * 2 if op == "rs" else m * k * 2) / 1e6,
                      "single_ms": single, "b2b_ms_per_op": batched,
                      "single_median_ms": statistics.median(single), "b2b_median_ms": median,
                      "tflops": 2 * m * k * n / (median * 1e9)}
            report["results"].append(record)
            print(json.dumps({k: v for k, v in record.items()
                              if k not in ("single_ms", "b2b_ms_per_op")}), flush=True)
            del a, b, out
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
