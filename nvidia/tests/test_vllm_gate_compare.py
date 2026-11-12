"""Reject mismatched prompts even if model settings and output tokens match."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    case = {"arm": "stock", "model": "model", "M": 4, "np": 1, "osl": 1,
            "prompts": [{"ids": [7], "prompt_sha256": "a"}]}
    with tempfile.TemporaryDirectory() as directory:
        first, second = Path(directory) / "a.json", Path(directory) / "b.json"
        first.write_text(json.dumps(case))
        second.write_text(json.dumps(case))
        command = [sys.executable, str(root / "experiments/compare_vllm_gates.py"), str(first), str(second)]
        assert subprocess.run(command, capture_output=True).returncode == 0
        case["prompts"][0]["prompt_sha256"] = "b"
        second.write_text(json.dumps(case))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and "prompt 0 differs" in result.stderr
    print("vLLM gate prompt identity PASS")


if __name__ == "__main__":
    main()
