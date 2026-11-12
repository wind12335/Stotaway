"""Compare independent batched runs and check all generated token IDs."""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("stock", type=Path, nargs="+")
p.add_argument("--dx", type=Path, nargs="+", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
stock = [json.loads(f.read_text()) for f in a.stock]
dx = [json.loads(f.read_text()) for f in a.dx]
all_runs = stock + dx
keys = ("model", "arch", "M", "batch", "output", "repeats", "warmups", "gpu_util", "dtype", "enforce_eager", "prefix_caching", "prompt_hashes")
if any(r[k] != all_runs[0][k] for r in all_runs for k in keys):
    raise ValueError("incompatible run settings or inputs")
if any(r["arm"] != "stock" for r in stock) or any(r["arm"] != "dx" for r in dx):
    raise ValueError("incorrect arms")
if any((r["components"], r["row_exact"]) != (dx[0]["components"], dx[0]["row_exact"]) for r in dx):
    raise ValueError("candidate implementations differ between runs")
reference = stock[0]["measurements"][0]["ids"]
mismatches = []
for run in all_runs:
    for i, sample in enumerate(run["measurements"]):
        if len(sample["ids"]) != len(reference):
            raise ValueError("different output batch size")
        for j, (expected, actual) in enumerate(zip(reference, sample["ids"])):
            if expected != actual:
                mismatches.append({"arm": run["arm"], "sample": i, "prompt": j,
                                   "different_positions": sum(x != y for x, y in zip(expected, actual)) + abs(len(expected) - len(actual))})
s = [r["median_ms"] for r in stock]
d = [r["median_ms"] for r in dx]
out = {"settings": {k: all_runs[0][k] for k in keys}, "stock_process_medians_ms": s,
       "dx_process_medians_ms": d, "stock_median_ms": statistics.median(s),
       "dx_median_ms": statistics.median(d), "speedup": statistics.median(s) / statistics.median(d),
       "stock_generated_tokens_per_second": all_runs[0]["batch"] * all_runs[0]["output"] * 1000 / statistics.median(s),
       "dx_generated_tokens_per_second": all_runs[0]["batch"] * all_runs[0]["output"] * 1000 / statistics.median(d),
       "token_mismatches": mismatches, "stock_files": [str(f) for f in a.stock], "dx_files": [str(f) for f in a.dx]}
with a.output.open("x") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
print(json.dumps(out, indent=2))
