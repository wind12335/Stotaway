import argparse
import json
from pathlib import Path


def read_profile(path, expected_tp):
    if not path.is_file():
        raise FileNotFoundError(f"A800 calibration profile not found: {path}")
    with path.open(encoding="utf-8") as source:
        profile = json.load(source)
    if profile.get("platform") != "NVIDIA A800" or profile.get("tp") != expected_tp:
        raise ValueError("profile platform or TP does not match the requested A800 run")
    if profile.get("calibrated") is not True:
        raise ValueError("profile is not calibrated; use stock until validation completes")
    return profile


def shapes(hidden, ffn, heads, kv_heads, head_dim, rows, tp):
    if any(value <= 0 for value in (hidden, ffn, heads, kv_heads, head_dim, rows, tp)):
        raise ValueError("all model dimensions and TP must be positive")
    if heads * head_dim != hidden:
        raise ValueError("head count multiplied by head dimension must equal hidden size")
    widths = {"qkv": (heads + 2 * kv_heads) * head_dim, "o_proj": hidden,
              "fc1": 2 * ffn, "fc2": ffn}
    if any(width % tp for width in widths.values()):
        raise ValueError("model widths must be divisible by TP")
    return {"qkv": (rows, hidden, widths["qkv"] // tp),
            "o_proj": (rows, hidden // tp, hidden),
            "fc1": (rows, hidden, widths["fc1"] // tp),
            "fc2": (rows, ffn // tp, hidden)}


def main():
    parser = argparse.ArgumentParser(description="A800 planner safety gate; no uncalibrated DX choices")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--ffn", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    model_shapes = shapes(args.hidden, args.ffn, args.heads, args.kv_heads,
                          args.head_dim, args.rows, args.tp)
    if args.profile:
        read_profile(args.profile, args.tp)
        raise NotImplementedError("A800 cost model and validation are not yet implemented")
    print(json.dumps({"platform": "NVIDIA A800", "tp": args.tp, "calibrated": False,
                      "reason": "No validated A800 cost model: stock fallback",
                      "sites": {name: {"shape_MKN": shape, "choice": "stock"}
                                for name, shape in model_shapes.items()}}, indent=2))


if __name__ == "__main__":
    main()
