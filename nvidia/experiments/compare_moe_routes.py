"""Compare captured router maps for matching layer/rank invocation order."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
files = sorted(a.baseline.glob("rank*_route*.pt"))
other = sorted(a.candidate.glob("rank*_route*.pt"))
if not files or [x.name for x in files] != [x.name for x in other]:
    raise ValueError("route capture names do not match")
records = []
for file in files:
    ref = torch.load(file, weights_only=True)
    got = torch.load(a.candidate / file.name, weights_only=True)
    if ref.keys() != got.keys() or ref["routing"].shape != got["routing"].shape:
        raise ValueError("route schemas do not match")
    route_diff = ref["routing"] != got["routing"]
    delta = (ref["probabilities"].float() - got["probabilities"].float()).abs()
    records.append({"file": file.name, "changed_entries": int(route_diff.sum()),
                    "changed_rows": int(route_diff.any(-1).sum()),
                    "rows": int(route_diff.shape[0]), "prob_max_abs": float(delta.max())})
result = {"baseline": str(a.baseline), "candidate": str(a.candidate), "records": records}
with a.output.open("x") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(json.dumps(result, indent=2))
