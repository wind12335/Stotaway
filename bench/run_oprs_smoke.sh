#!/bin/bash
# OPRS runtime plumbing check on BW (bench/oprs_smoke.py): every candidate method x direction through the
# real dispatcher + autograd wrappers against the serial path. Nothing is timed; the 1-card run is a smoke.
#   NP=1 bench/run_oprs_smoke.sh [args of oprs_smoke.py: --methods deo,async_tp,triton_dist --m 4096 ...]
# NP must be exported BEFORE env.sh (env.sh defaults it to 8).
set -u
HERE=$(dirname "$(readlink -f "$0")")
export SW_NP=${SW_NP:-1}
source "$HERE/../env/env.sh"
export OPRS_TRITON_DIST_ROOT=${OPRS_TRITON_DIST_ROOT:-/root/private_data/tri-dist-hygon}
export PYTHONPATH=$PYTHONPATH${OPRS_TRITON_DIST_ROOT:+:$OPRS_TRITON_DIST_ROOT}
export DUSHMEM_SYMMETRIC_SIZE=${OPRS_HEAP:-4294967296}
timeout ${OPRS_TIMEOUT:-900} $SW_MPIRUN -x DUSHMEM_SYMMETRIC_SIZE -x OPRS_TRITON_DIST_ROOT \
    python3 "$HERE/oprs_smoke.py" "$@"
