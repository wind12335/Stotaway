import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("dx", type=Path)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    args = parser.parse_args()
    baseline_files = sorted(args.baseline.glob("rank*_step*.pt"))
    candidate_files = sorted(args.dx.glob("rank*_step*.pt"))
    if not baseline_files or [path.name for path in baseline_files] != [path.name for path in candidate_files]:
        raise RuntimeError("baseline and DX rank/step capture files do not match")
    for baseline_path, candidate_path in zip(baseline_files, candidate_files):
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if baseline.keys() != candidate.keys():
            raise RuntimeError(f"parameter keys do not match: {baseline_path.name}")
        worst, worst_name, bad = 0.0, "", []
        if not baseline:
            raise RuntimeError(f"empty capture: {baseline_path.name}")
        for name in baseline:
            reference, result = baseline[name], candidate[name]
            if reference.shape != result.shape or reference.dtype != result.dtype:
                raise RuntimeError(f"parameter shape or dtype mismatch: {name}")
            difference = (reference.float() - result.float()).abs().max().item()
            if difference > worst:
                worst, worst_name = difference, name
            if not torch.allclose(reference, result, rtol=args.rtol, atol=args.atol):
                bad.append(name)
        print(f"{baseline_path.name} parameters={len(baseline)} max_abs={worst:.8g} "
              f"worst={worst_name} outside_tolerance={len(bad)}", flush=True)
        if bad:
            raise AssertionError(f"parameters outside tolerance: {bad[:10]}")


if __name__ == "__main__":
    main()
