"""Summarize independent, blocked Megatron runs; retain every iteration."""
import argparse
import json
import re
import statistics
from pathlib import Path

from report_megatron_e2e import read

HEALTH = re.compile(r"lm loss:\s*([0-9.eE+-]+).*?grad norm:\s*([0-9.eE+-]+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--prefix", default="megatron")
    p.add_argument("--arms", nargs="+", default=["stock", "full_dx", "planner"])
    args = p.parse_args()
    if len(args.arms) < 2 or args.arms[0] != "stock":
        p.error("first arm must be stock, followed by at least one candidate")
    result = {"warmup_iterations": args.warmup, "repeats": args.repeats, "arms": {}, "paired_speedups": {}}
    for arm in args.arms:
        samples, medians, health = [], [], []
        for repeat in range(args.repeats):
            path = args.raw / f"e2e_{args.prefix}_{arm}_r{repeat}_20261110.log"
            measured = read(path, args.warmup)
            matches = HEALTH.findall(path.read_text(errors="replace"))
            if len(matches) != len(measured) + args.warmup or any(int(x[2]) or int(x[3]) for x in matches):
                raise ValueError(f"missing health records or skipped/NaN iterations: {path}")
            medians.append(statistics.median(measured))
            samples.append(measured)
            health.append([{"loss": float(x[0]), "grad_norm": float(x[1]), "skipped": int(x[2]), "nan": int(x[3])} for x in matches])
        result["arms"][arm] = {"iteration_ms": samples, "process_medians_ms": medians,
                               "median_ms": statistics.median(medians), "health": health}
    baseline = result["arms"]["stock"]
    for arm in args.arms[1:]:
        candidate = result["arms"][arm]
        result["paired_speedups"][arm] = {
            "ratio_of_process_medians": baseline["median_ms"] / candidate["median_ms"],
            "by_block": [a / b for a, b in zip(baseline["process_medians_ms"], candidate["process_medians_ms"])],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"arms": {a: {"process_medians_ms": result["arms"][a]["process_medians_ms"],
                                   "median_ms": result["arms"][a]["median_ms"]} for a in args.arms},
                      "speedups": result["paired_speedups"]}, indent=2))


if __name__ == "__main__":
    main()
