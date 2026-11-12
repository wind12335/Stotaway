"""Offline exploratory leave-one-model-out evaluation of original cost model."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--costmodel", type=Path, default=Path("/root/Stotaway/planner/costmodel.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if raw.get("classification") != "UNVALIDATED exploratory prototype; not for planner deployment" or raw.get("tp") != 2:
        raise ValueError("expected exploratory TP2 cells")
    spec = importlib.util.spec_from_file_location("original_dcu_costmodel", args.costmodel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = raw["cells"]
    models = sorted({c["model"] for c in cells})
    if len(models) < 2:
        raise ValueError("leave-one-model-out requires at least two models")
    results = []
    for heldout in models:
        train = [c for c in cells if c["model"] != heldout]
        test = [c for c in cells if c["model"] == heldout]
        model = module.CostModel(mode="b2b").fit(train)
        for cell in test:
            choice, prediction = model.choose(cell["op"], cell["G"], cell["B"], allowed={"dx"})
            actual = {"stock": cell["stock_b2b"], "dx": cell["dx_b2b"]}
            if any(value <= 0 for value in prediction.values()):
                raise ValueError("non-positive predicted time; model is invalid on this profile")
            results.append({"heldout": heldout, "site": cell["site"], "M": cell["M"],
                            "choice": choice, "oracle": min(actual, key=actual.get),
                            "stock_ms": actual["stock"], "dx_ms": actual["dx"],
                            "chosen_ms": actual[choice], "oracle_ms": min(actual.values()),
                            "predicted_ms": prediction})
    summary = {"cells": len(results), "matching_oracle": sum(r["choice"] == r["oracle"] for r in results),
               "chosen_slower_than_stock": sum(r["chosen_ms"] > r["stock_ms"] for r in results),
               "stock_total_ms": sum(r["stock_ms"] for r in results),
               "chosen_total_ms": sum(r["chosen_ms"] for r in results),
               "oracle_total_ms": sum(r["oracle_ms"] for r in results)}
    result = {"classification": "UNVALIDATED exploratory analysis; no deployment decision",
              "source": str(args.input), "method": "leave-one-model-out; original DCU model structure", 
              "summary": summary, "cells": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
