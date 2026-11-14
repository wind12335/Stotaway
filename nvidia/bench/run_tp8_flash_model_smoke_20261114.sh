#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
export PYTHONPATH="$PWD:/root/Triton-distributed/python" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export STOWAWAY_FLASH_ROOT=/root/data/stowaway-nvidia/flashoverlap_bf16_20261114 STOWAWAY_FLASH_BF16=1
export TEST_EXTERNAL_BACKEND=flashoverlap
for model in qwen36 llama70; do
 export TEST_EXTERNAL_PLAN="/tmp/stowaway_flash_bf16_${model}_smoke.json" TEST_EXTERNAL_OUT="$res/tp8_flash_bf16_${model}_smoke_20261114.json"
 if [[ -f "$TEST_EXTERNAL_OUT" ]]; then continue; fi
 echo "START $model $(date -u +%T)"
 timeout -k 10 360 /root/miniconda3/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_external_dispatch.py > "$res/tp8_flash_bf16_${model}_smoke_20261114.log" 2>&1
 echo "END $model rc=$? $(date -u +%T)"
done
