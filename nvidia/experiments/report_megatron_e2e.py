"""Report paired Megatron iteration logs with explicit warmup exclusion."""
import argparse
import json
import re
import statistics
from pathlib import Path


PATTERN = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+)")


def read(path, warmup):
    values = [(int(i), int(n), float(t)) for i, n, t in PATTERN.findall(path.read_text(errors="replace"))]
    if not values or len(values) != values[0][1] or [x[0] for x in values] != list(range(1, len(values) + 1)):
        raise ValueError(f"missing or repeated iteration timings: {path}")
    if warmup >= len(values):
        raise ValueError("warmup must leave at least one iteration")
    return [x[2] for x in values[warmup:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = read(args.baseline, args.warmup)
    result = {"warmup": args.warmup, "baseline": str(args.baseline), "baseline_ms": base,
              "baseline_median_ms": statistics.median(base), "candidates": []}
    for path in args.candidates:
        values = read(path, args.warmup)
        if len(values) != len(base):
            raise ValueError("candidate has a different number of measured iterations")
        median = statistics.median(values)
        result["candidates"].append({"path": str(path), "samples_ms": values,
                                     "median_ms": median, "speedup": statistics.median(base) / median})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
