"""Compare matching offline vLLM runs; report latency even on token mismatch."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", nargs="+", type=Path)
    parser.add_argument("--dx", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate-arm", choices=("dx", "stock_sp"), default="dx")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stocks = [json.loads(p.read_text()) for p in args.stock]
    candidates = [json.loads(p.read_text()) for p in args.dx]
    all_runs = stocks + candidates
    keys = ("model", "arch", "M", "output", "requests", "warmups", "seed", "gpu_util", "dtype", "enforce_eager", "prefix_caching")
    if any(run[k] != all_runs[0][k] for run in all_runs for k in keys):
        raise ValueError("incompatible run settings")
    if any(x["arm"] != "stock" for x in stocks) or any(x["arm"] != args.candidate_arm for x in candidates):
        raise ValueError("incorrect arm")
    if any((r["components"], r["row_exact"]) != (candidates[0]["components"], candidates[0]["row_exact"]) for r in candidates):
        raise ValueError("candidate implementations differ between runs")
    ref = {(x["index"], x["generated"]): x for x in stocks[0]["results"]}
    mismatch = []
    for run in all_runs:
        rows = {(x["index"], x["generated"]): x for x in run["results"]}
        if rows.keys() != ref.keys():
            raise ValueError("different request set")
        for k, value in rows.items():
            if value["prompt_sha256"] != ref[k]["prompt_sha256"]:
                raise ValueError(f"different prompt {k}")
            if value["ids"] != ref[k]["ids"]:
                mismatch.append({"arm": run["arm"], "request": k, "different_positions":
                                 sum(a != b for a, b in zip(value["ids"], ref[k]["ids"])) +
                                 abs(len(value["ids"]) - len(ref[k]["ids"]))})
    summary = {}
    for n in (1, all_runs[0]["output"]):
        a = [r["summary"][str(n)]["median_ms"] for r in stocks]
        b = [r["summary"][str(n)]["median_ms"] for r in candidates]
        summary[str(n)] = {"stock_process_medians_ms": a, "dx_process_medians_ms": b,
                           "stock_median_ms": statistics.median(a), "dx_median_ms": statistics.median(b),
                           "speedup": statistics.median(a) / statistics.median(b)}
    result = {"settings": {k: all_runs[0][k] for k in keys}, "candidate_arm": args.candidate_arm,
              "candidate_components": [r["components"] for r in candidates],
              "candidate_row_exact": [r["row_exact"] for r in candidates], "summary": summary,
              "token_mismatches": mismatch, "stock_files": [str(p) for p in args.stock],
              "dx_files": [str(p) for p in args.dx]}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
