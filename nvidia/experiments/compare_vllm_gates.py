"""Compare greedy-token output gates; fail if inputs differ or SP never ran."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    for key in ("model", "M", "np", "osl"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"incompatible gate {key}: {baseline[key]} vs {candidate[key]}")
    if len(baseline["prompts"]) != baseline["np"] or len(candidate["prompts"]) != candidate["np"]:
        raise ValueError("missing prompts")
    differences = []
    for index, (reference, result) in enumerate(zip(baseline["prompts"], candidate["prompts"])):
        ref_hash, result_hash = reference.get("prompt_sha256"), result.get("prompt_sha256")
        if ref_hash is not None and result_hash is not None and ref_hash != result_hash:
            raise ValueError(f"prompt {index} differs between runs")
        if ref_hash is None or result_hash is None:
            print(f"warning: prompt {index} has no hash in an older capture; input identity is not verified")
        if len(reference["ids"]) != baseline["osl"] or len(result["ids"]) != baseline["osl"]:
            raise ValueError(f"prompt {index} has an incomplete generated sequence")
        for step, (expected, actual) in enumerate(zip(reference["ids"], result["ids"])):
            if expected != actual:
                differences.append((index, step, expected, actual))
    print(f"{baseline['arm']} vs {candidate['arm']}: prompts={baseline['np']} "
          f"tokens_per_prompt={baseline['osl']} differences={len(differences)}")
    if differences:
        print(f"first_differences={differences[:10]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
