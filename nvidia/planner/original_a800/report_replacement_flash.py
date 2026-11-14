"""Audit and summarize the fixed TP2 FlashOverlap/stock/DX comparison."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def report(manifest_path, directory):
    raw = manifest_path.read_bytes()
    design = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    groups = []
    for index, target in enumerate(design["targets"]):
        blocks = []
        signature = None
        for block, order in enumerate(design["orders"], 1):
            arms = {}
            for arm in order:
                path = directory / f"tp2_replace_flash_t{index}_{arm}_b{block}_20261113.json"
                data = path.read_bytes()
                entry = json.loads(data)
                if (entry["manifest_sha256"] != manifest_hash or entry["target"] != target
                        or entry["arm"] != arm or entry["checks"] < 2):
                    raise ValueError(f"invalid provenance or correctness checks: {path}")
                keys = ("tp", "dtype", "batch", "warmup", "repeats", "gpu", "torch",
                        "triton", "cuda_build", "cuda_device_max_connections")
                sig = tuple(entry[k] for k in keys)
                if signature is not None and sig != signature:
                    raise ValueError(f"mismatched measurement protocol: {path}")
                signature = sig
                for key in ("tp", "dtype", "batch", "warmup", "repeats"):
                    if entry[key] != design[key]:
                        raise ValueError(f"manifest mismatch for {key}: {path}")
                samples = entry["samples_ms"]
                if len(samples) != design["repeats"] or any(
                    not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0 for t in samples
                ):
                    raise ValueError(f"invalid samples: {path}")
                t = entry["median_ms"]
                if not math.isclose(t, statistics.median(samples), rel_tol=1e-9):
                    raise ValueError(f"incorrect median: {path}")
                if arm == "dx" and not entry.get("dx_scratch"):
                    raise ValueError(f"unexpected DX route: {path}")
                if arm == "flashoverlap" and not entry.get("contract_sha256"):
                    raise ValueError(f"missing layout contract provenance: {path}")
                arms[arm] = {"median_ms": t, "sha256": hashlib.sha256(data).hexdigest(),
                             "path": str(path), "output_semantics": entry["output_semantics"]}
            blocks.append({"block": block, "order": order, "arms": arms,
                           "dx_vs_external": arms["flashoverlap"]["median_ms"] / arms["dx"]["median_ms"],
                           "dx_vs_stock": arms["stock"]["median_ms"] / arms["dx"]["median_ms"],
                           "external_vs_stock": arms["stock"]["median_ms"] / arms["flashoverlap"]["median_ms"]})
        geomean = lambda key: math.exp(statistics.mean(math.log(b[key]) for b in blocks))
        groups.append({"target": target, "blocks": blocks,
                       "dx_vs_external_gmean": geomean("dx_vs_external"),
                       "dx_vs_stock_gmean": geomean("dx_vs_stock"),
                       "external_vs_stock_gmean": geomean("external_vs_stock"),
                       "dx_faster_blocks": sum(b["dx_vs_external"] > 1 for b in blocks)})
    return {"classification": "TP2 raw RS operator timings with different output layouts",
            "manifest_sha256": manifest_hash, "groups": groups,
            "warning": "Targets chosen from prior measurements. FlashOverlap uses reordered output; "
                       "DX and stock use contiguous rank shards. Conversion, planner and E2E costs "
                       "are excluded. Raw speedups do not prove interchangeable or net replacement."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = report(args.manifest, args.directory)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    for group in result["groups"]:
        print(group["target"], "DX/Flash", group["dx_vs_external_gmean"],
              "DX/stock", group["dx_vs_stock_gmean"], "wins", group["dx_faster_blocks"])


if __name__ == "__main__":
    main()
