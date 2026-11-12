import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a800_platform", ROOT / "planner" / "platform.py")
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


def main():
    result = PLANNER.shapes(5120, 17408, 40, 8, 128, 8192, 2)
    assert result["qkv"] == (8192, 5120, 3584)
    assert result["fc2"] == (8192, 8704, 5120)
    try:
        PLANNER.shapes(5120, 17408, 24, 4, 256, 8192, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent Qwen hybrid dimensions must not be silently approximated")
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "profile.json"
        profile.write_text(json.dumps({"platform": "Hygon BW1000", "tp": 2, "calibrated": True}))
        try:
            PLANNER.read_profile(profile, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("foreign calibration was accepted")
    run = subprocess.run([sys.executable, str(ROOT / "planner" / "platform.py"), "--hidden", "5120",
                          "--ffn", "17408", "--heads", "40", "--kv-heads", "8", "--head-dim", "128",
                          "--rows", "8192"], text=True, check=True, capture_output=True)
    output = json.loads(run.stdout)
    assert not output["calibrated"] and all(site["choice"] == "stock" for site in output["sites"].values())
    print("planner: shape validation, foreign profile rejection, stock fallback PASS")


if __name__ == "__main__":
    main()
