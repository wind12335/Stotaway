#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
work=/root/data/stowaway-nvidia/syncopate_tp2_work_20261113
out="$res/tp8_dx_fp16_matrix_20261114"
mkdir -p "$out"
export PYTHONPATH="$PWD:/root/Triton-distributed/python:$work" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G SYNC_WORK="$work" SYNC_ARM=dx
for manifest in "$res"/tp8_sync_matrix_20261113/*.manifest.json; do
 name=$(basename "$manifest" .manifest.json)
 op=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["target"]["op"])' "$manifest")
 export SYNC_MANIFEST="$manifest" SYNC_OUT="$out/${name}_dx.json"
 [[ -f "$SYNC_OUT" ]] && continue
 echo "START $name $(date -u +%T)"
 timeout -k 5 180 python -m torch.distributed.run --standalone --nproc-per-node=8 "bench/tp8_syncopate_${op}.py" > "$out/${name}_dx.log" 2>&1
 echo "END $name rc=$? $(date -u +%T)"
done
