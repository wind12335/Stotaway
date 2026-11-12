"""Report per-parameter tolerance and norm differences for all saved ranks."""
import argparse
import json
import math
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--rtol", type=float, default=.02)
p.add_argument("--atol", type=float, default=.002)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
baseline = sorted(a.baseline.glob("rank*_step*.pt"))
candidate = sorted(a.candidate.glob("rank*_step*.pt"))
if not baseline or [x.name for x in baseline] != [x.name for x in candidate]:
    raise ValueError("different or empty capture sets")
records = []
for left, right in zip(baseline, candidate):
    ref = torch.load(left, map_location="cpu", weights_only=True)
    got = torch.load(right, map_location="cpu", weights_only=True)
    if ref.keys() != got.keys():
        raise ValueError("different parameter names")
    for name in ref:
        x, y = ref[name].float(), got[name].float()
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch {name}")
        delta = (x - y).abs()
        exceed = delta > a.atol + a.rtol * x.abs()
        n = int(exceed.sum().item())
        if n or not torch.isfinite(delta).all().item():
            reference_norm = float(torch.linalg.vector_norm(x).item())
            error_norm = float(torch.linalg.vector_norm(delta).item())
            records.append({"file": left.name, "parameter": name, "elements": x.numel(),
                            "outside": n, "max_abs": float(delta.max().item()),
                            "reference_l2": reference_norm, "error_l2": error_norm,
                            "relative_l2": error_norm / max(reference_norm, 1e-30),
                            "finite": bool(torch.isfinite(delta).all().item())})
    print(f"checked {left.name}", flush=True)
result = {"rtol": a.rtol, "atol": a.atol, "baseline": str(a.baseline),
          "candidate": str(a.candidate), "failed_tensors": len(records), "records": records}
with a.output.open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write("\n")
print(json.dumps(result, indent=2))
