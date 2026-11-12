"""Evaluate frozen exact-shape choices on independent operator runs."""
import argparse
import hashlib
import json
from pathlib import Path

from tp2_lookup import Lookup, SITES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interpolate", action="store_true")
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        md = report["metadata"]
        if md["checks"] < 1:
            raise ValueError("missing correctness checks")
        runtime = {k: md[k] for k in ("tp", "dtype", "gpu", "torch", "triton", "cuda_build", "cuda_device_max_connections")}
        lookup = Lookup(args.profile, runtime, interpolate=args.interpolate)
        if any(s["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
               for s in lookup.profile["sources"]):
            raise ValueError("evaluation input is a calibration source")
        for row in report["results"]:
            site = SITES[row["site"]]
            choice = lookup.choose(site, row["M"], row["K"], row["N"])
            stock = row["measurements"]["stock"]["median_ms"]
            dx = row["measurements"]["dx"]["median_ms"]
            records.append({"model": row["model"], "site": site, "M": row["M"], "K": row["K"],
                            "N": row["N"], "choice": choice, "stock_ms": stock, "dx_ms": dx,
                            "chosen_ms": dx if choice == "dx" else stock,
                            "oracle": "dx" if dx < stock else "stock"})
    summary = {"cells": len(records), "match_oracle": sum(r["choice"] == r["oracle"] for r in records),
               "chosen_slower_than_stock": sum(r["chosen_ms"] > r["stock_ms"] for r in records),
               "stock_sum_ms": sum(r["stock_ms"] for r in records),
               "chosen_sum_ms": sum(r["chosen_ms"] for r in records),
               "oracle_sum_ms": sum(min(r["stock_ms"], r["dx_ms"]) for r in records)}
    result = {"classification": "diagnostic independent operator measurements",
              "profile": str(args.profile), "interpolate": args.interpolate, "inputs": [str(p) for p in args.inputs],
              "summary": summary, "cells": records}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
