"""Conservative, exact-shape TP2 calibration for Megatron diagnostics.

The input measurements cover forward-only operators; they are not a training
step cost model. Missing or incompatible shapes always use stock.
"""
import hashlib
import json
import math
import os
from pathlib import Path


SITES = {"qkv": "qkv", "o_proj": "proj", "fc1": "fc1", "fc2": "fc2"}
SCHEMA = "stowaway-tp2-forward-lookup-v1"
TRAIN_SCHEMA = "stowaway-tp2-training-lookup-v1"


def _quantile(values, fraction):
    xs = sorted(values)
    if not xs or any(not math.isfinite(x) or x <= 0 for x in xs):
        raise ValueError("invalid latency samples")
    pos = fraction * (len(xs) - 1)
    left = int(pos)
    right = min(left + 1, len(xs) - 1)
    return xs[left] + (xs[right] - xs[left]) * (pos - left)


def key(site, m, k, n):
    return f"{site}:{m}:{k}:{n}"


def build(paths, margin=0.03, lower=0.2, upper=0.8, objective="forward"):
    if not 0 <= margin < 1 or not 0 <= lower < upper <= 1:
        raise ValueError("invalid selection guard")
    if objective not in ("forward", "training"):
        raise ValueError("objective must be forward or training")
    signature = None
    samples = {}
    sources = []
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw)
        md = report["metadata"]
        current = {name: md[name] for name in
                   ("tp", "dtype", "gpu", "torch", "triton", "cuda_build", "cuda_device_max_connections")}
        if current["tp"] != 2 or md["checks"] < 1 or not report["results"]:
            raise ValueError("requires checked TP2 measurements")
        if objective == "training" and md.get("classification") != "diagnostic unfused site forward+backward":
            raise ValueError("training profile requires forward+backward measurements")
        if objective == "forward" and md.get("classification") == "diagnostic unfused site forward+backward":
            raise ValueError("training measurements cannot populate forward profile")
        if signature is not None and current != signature:
            raise ValueError("cannot mix runtime signatures")
        signature = current
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for row in report["results"]:
            site = SITES[row["site"]]
            if row["op"] != ("ag" if site in ("qkv", "fc1") else "rs"):
                raise ValueError("site and collective do not match")
            k = key(site, row["M"], row["K"], row["N"])
            item = samples.setdefault(k, {"stock": [], "dx": []})
            for arm in ("stock", "dx"):
                item[arm].extend(row["measurements"][arm]["samples_ms"])
    entries = {}
    for k, v in samples.items():
        if len(v["stock"]) != len(v["dx"]) or len(v["stock"]) < 4:
            raise ValueError(f"insufficient matched arm samples for {k}")
        stock = _quantile(v["stock"], lower)
        dx = _quantile(v["dx"], upper)
        entries[k] = {"choice": "dx" if dx <= stock * (1 - margin) else "stock",
                      "stock_lower_ms": stock, "dx_upper_ms": dx,
                      "samples": len(v["stock"])}
    return {"schema": SCHEMA if objective == "forward" else TRAIN_SCHEMA,
            "classification": "diagnostic-forward-only" if objective == "forward" else "diagnostic-forward-backward",
            "runtime": signature, "guard": {"margin": margin, "stock_quantile": lower,
                                               "dx_quantile": upper},
            "sources": sources, "entries": entries}


class Lookup:
    def __init__(self, path, runtime, *, interpolate=False):
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.profile = json.loads(raw)
        valid = {(SCHEMA, "diagnostic-forward-only"), (TRAIN_SCHEMA, "diagnostic-forward-backward")}
        if (self.profile.get("schema"), self.profile.get("classification")) not in valid:
            raise ValueError("unsupported calibration profile")
        if self.profile.get("runtime") != runtime:
            raise ValueError(f"calibration runtime mismatch: expected {self.profile.get('runtime')}, got {runtime}")
        if not isinstance(self.profile.get("entries"), dict):
            raise ValueError("invalid calibration entries")
        self.interpolate = interpolate

    def choose(self, site, m, k, n):
        entry = self.profile["entries"].get(key(site, m, k, n))
        if entry is None:
            if not self.interpolate or self.profile.get("schema") != TRAIN_SCHEMA:
                return "stock"
            candidates = []
            for name, value in self.profile["entries"].items():
                parts = name.split(":")
                if len(parts) != 4:
                    raise ValueError("invalid shape key")
                if parts[0] == site and int(parts[2]) == k and int(parts[3]) == n:
                    candidates.append((int(parts[1]), value.get("choice")))
            below = [v for v in candidates if v[0] < m]
            above = [v for v in candidates if v[0] > m]
            if not below or not above:
                return "stock"
            left, right = max(below), min(above)
            return "dx" if left[1] == right[1] == "dx" else "stock"
        choice = entry.get("choice")
        if choice not in ("stock", "dx"):
            raise ValueError("invalid candidate in calibration profile")
        return choice


def runtime_signature(torch, triton):
    return {"tp": 2, "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "triton": triton.__version__, "cuda_build": torch.version.cuda,
            "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset")}
