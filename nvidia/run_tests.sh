#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${NVSHMEM_HOME:-}" ]]; then
  NVSHMEM_HOME="$(python -c 'import os, site; print(next(os.path.join(p, "nvidia", "nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p, "nvidia", "nvshmem", "lib", "libnvshmem_host.so.3"))))')"
fi
export NVSHMEM_HOME
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-256M}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
timeout "${TEST_TIMEOUT:-120}" python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 "${1:-tests/test_dx.py}"
