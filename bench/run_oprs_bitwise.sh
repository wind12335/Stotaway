#!/bin/bash
# Bit-for-bit agreement of the overlap methods with the vendor GEMM (bench/oprs_bitwise.py). Not timed.
#   NP=1 bash bench/run_oprs_bitwise.sh [--shapes tiny_vllm|bw27b|tiny_megatron|site:op:M:N:K,...] [--methods ...]
# NP must be exported BEFORE env.sh (env.sh defaults it to 8).
set -u
HERE=$(dirname "$(readlink -f "$0")")
export SW_NP=${SW_NP:-${NP:-1}}
source "$HERE/../env/env.sh"
export OPRS_TRITON_DIST_ROOT=${OPRS_TRITON_DIST_ROOT:-/root/private_data/tri-dist-hygon}
export PYTHONPATH=$PYTHONPATH${OPRS_TRITON_DIST_ROOT:+:$OPRS_TRITON_DIST_ROOT}
export DUSHMEM_SYMMETRIC_SIZE=${OPRS_HEAP:-4294967296}
ARGS=()
[ -n "${JSON:-}" ] && ARGS+=(--json "$JSON")
timeout ${OPRS_TIMEOUT:-900} $SW_MPIRUN -x DUSHMEM_SYMMETRIC_SIZE -x OPRS_TRITON_DIST_ROOT \
    python3 "$HERE/oprs_bitwise.py" "${ARGS[@]}" "$@"
