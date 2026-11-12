"""Verify conservative calibration, metadata guards and unseen-shape fallback."""
import json
import tempfile
from pathlib import Path

from planner.tp2_lookup import Lookup, build


def main():
    runtime = {"tp": 2, "dtype": "bfloat16", "gpu": "A800", "torch": "x", "triton": "y",
               "cuda_build": "z", "cuda_device_max_connections": "1"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        def row(site, op, n, stock, dx):
            return {"site": site, "op": op, "M": 64, "K": 128, "N": n,
                    "measurements": {"stock": {"samples_ms": stock}, "dx": {"samples_ms": dx}}}
        report = {"metadata": {**runtime, "checks": 8}, "results": [
            row("o_proj", "rs", 128, [1, 1, 1, 1], [.8, .8, .8, .8]),
            row("fc1", "ag", 256, [1, 1, 1, 1], [.99, .99, .99, .99]),
        ]}
        source = root / "data.json"
        source.write_text(json.dumps(report))
        profile = build([source])
        target = root / "profile.json"
        target.write_text(json.dumps(profile))
        model = Lookup(target, runtime)
        assert model.choose("proj", 64, 128, 128) == "dx"
        assert model.choose("fc1", 64, 128, 256) == "stock"
        assert model.choose("proj", 128, 128, 128) == "stock"
        try:
            Lookup(target, {**runtime, "cuda_device_max_connections": "unset"})
        except ValueError:
            pass
        else:
            raise AssertionError("different runtime must be rejected")
        report["results"][0]["measurements"]["dx"]["samples_ms"] = [0.5] * 3
        source.write_text(json.dumps(report))
        try:
            build([source])
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete measurement samples must be rejected")
        report["results"][0]["measurements"]["dx"]["samples_ms"] = [.8] * 4
        report["metadata"]["classification"] = "diagnostic unfused site forward+backward"
        source.write_text(json.dumps(report))
        training = build([source], objective="training")
        training["entries"]["proj:128:128:128"] = dict(training["entries"]["proj:64:128:128"])
        target.write_text(json.dumps(training))
        assert Lookup(target, runtime).choose("proj", 64, 128, 128) == "dx"
        interpolated = Lookup(target, runtime, interpolate=True)
        assert interpolated.choose("proj", 96, 128, 128) == "dx"
        assert interpolated.choose("proj", 192, 128, 128) == "stock"
        assert interpolated.choose("proj", 96, 64, 128) == "stock"
        training["entries"]["proj:128:128:128"]["choice"] = "stock"
        target.write_text(json.dumps(training))
        assert Lookup(target, runtime, interpolate=True).choose("proj", 96, 128, 128) == "stock"
        try:
            build([source], objective="forward")
        except ValueError:
            pass
        else:
            raise AssertionError("training measurements must not be labeled forward-only")
    print("TP2 lookup guard, exact shape and fallback PASS")


if __name__ == "__main__":
    main()
