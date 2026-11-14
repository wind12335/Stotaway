#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
root=/root/data/stowaway-nvidia/flashoverlap_bf16_20261114
if [[ ! -f "$root/build/lib/libst_pybinding.so" ]]; then
 echo 'BF16 FlashOverlap library not built; smoke skipped.'
 exit 1
fi
mkdir -p "$root/configs"
cp /root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113/configs/m8192n4096k1792_a800.json "$root/configs/"
export PYTHONPATH="$PWD:/root/Triton-distributed/python" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export STOWAWAY_FLASH_ROOT="$root" STOWAWAY_FLASH_BF16=1
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_SYMMETRIC_SIZE=4G STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
export TEST_EXTERNAL_BACKEND=flashoverlap TEST_EXTERNAL_PLAN="$PWD/experiments/tp8_external_smoke_flashoverlap_bf16_20261113.json"
export TEST_EXTERNAL_OUT="$res/tp8_external_dispatch_flashoverlap_bf16_20261114.json"
if [[ ! -f "$TEST_EXTERNAL_OUT" ]]; then
 echo "START FlashOverlap BF16 dispatcher $(date -u +%T)"
 timeout -k 10 360 /root/miniconda3/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_external_dispatch.py > "$res/tp8_external_dispatch_flashoverlap_bf16_20261114.log" 2>&1
 rc=$?
 echo "END FlashOverlap BF16 dispatcher rc=$rc $(date -u +%T)"
 exit "$rc"
fi
