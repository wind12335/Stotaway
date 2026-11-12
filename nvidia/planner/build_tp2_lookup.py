"""Build an explicitly diagnostic profile from forward operator measurements."""
import argparse
import json
from pathlib import Path

from tp2_lookup import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--objective", choices=("forward", "training"), default="forward")
    args = parser.parse_args()
    profile = build(args.inputs, margin=args.margin, objective=args.objective)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    print(f"entries={len(profile['entries'])} dx={sum(v['choice'] == 'dx' for v in profile['entries'].values())}")


if __name__ == "__main__":
    main()
