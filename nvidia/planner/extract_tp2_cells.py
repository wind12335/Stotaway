"""Convert exploratory TP2 operator records to the original cost-model schema.

Output is explicitly unvalidated.  Never load it in the deployment planner
without independently evaluating the model and obtaining training-step data.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, seen, signature = [], set(), None
    for path in args.inputs:
        report = json.loads(path.read_text())
        metadata = report["metadata"]
        current = tuple(metadata[key] for key in ("tp", "dtype", "gpu", "torch", "triton", "cuda_device_max_connections"))
        if metadata["tp"] != 2 or metadata["checks"] < 1 or metadata["batch"] != 16:
            raise ValueError(f"unsupported or unchecked input {path}")
        if signature is None:
            signature = current
        elif signature != current:
            raise ValueError("mixing device, software or connection settings is not allowed")
        for entry in report["results"]:
            site = entry["site"]
            op = "rs" if entry["op"] == "rs" else {"qkv": "ag_up", "fc1": "ag_gu"}[site]
            key = (entry["model"], site, entry["M"], entry["K"], entry["N"])
            if key in seen:
                raise ValueError(f"duplicate cell {key}")
            seen.add(key)
            measurements = entry["measurements"]
            cell = {"model": entry["model"], "site": site, "op": op, "M": entry["M"],
                    "K": entry["K"], "N": entry["N"], "B": entry["payload_MB"],
                    "G": measurements["gemm"]["median_ms"], "C": measurements["comm"]["median_ms"],
                    "stock_b2b": measurements["stock"]["median_ms"], "dx_b2b": measurements["dx"]["median_ms"],
                    "anomaly": False, "source": str(path)}
            if op == "rs":
                if "dx_comm" not in measurements:
                    raise ValueError(f"missing measured DX RS in {key}")
                cell["Cdx"] = measurements["dx_comm"]["median_ms"]
            if any(cell[name] <= 0 for name in ("B", "G", "C", "stock_b2b", "dx_b2b")):
                raise ValueError(f"non-positive measurement in {key}")
            cells.append(cell)
    if min(sum(c["op"] == op for c in cells) for op in ("rs", "ag_up", "ag_gu")) < 5:
        raise ValueError("need at least five cells of each operation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"classification": "UNVALIDATED exploratory prototype; not for planner deployment",
                   "tp": 2, "connection_mode": signature[-1], "cells": cells}, stream, indent=2)
        stream.write("\n")
    print(f"wrote {len(cells)} exploratory cells to {args.output}")


if __name__ == "__main__":
    main()
