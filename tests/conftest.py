"""Tests of the OPRS port (src/stowaway/oprs) and its tools. CPU only: nothing here opens a device.

    cd <repo> && python3 -m pytest tests -q

The GPU smoke of the runtime is bench/run_oprs_smoke.sh (mpirun, one or more cards); the single-card
end-to-end plumbing checks are e2e/megatron/run_oprs.sh and e2e/vllm/run_oprs.sh at NP=1.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, str(path))


def load(name, *parts):
    """A script outside the package (tools/, bench/) as a module; each inserts src/ itself."""
    spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*parts))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def scripts():
    return load
