#!/bin/bash
# Build the hipBLASLt grouped-GEMM shim used by stowaway.kernels.moe_ggemm (MoE expert GEMMs).
set -e
cd "$(dirname "$(readlink -f "$0")")"
${HIPCC:-hipcc} -O2 -fPIC -shared -std=c++17 moe_ggemm.cpp -o libmoe_ggemm.so -lhipblaslt
echo "built $(pwd)/libmoe_ggemm.so"
