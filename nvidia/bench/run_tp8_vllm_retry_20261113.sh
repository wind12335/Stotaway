#!/bin/bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH PYTHONPATH="$PWD" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVSHMEM_SYMMETRIC_SIZE=2G
export NVSHMEM_HOME=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B V8_M=16384 V8_ROPE_FACTOR=8 V8_REPS=3
for arm in stock async_tp async_dx; do
 export V8_ARM="$arm" V8_OUT="$res/tp8_20261111/vllm_bwmatch_${arm}_retry_20261113.json"
 if [[ -e "$V8_OUT" ]]; then continue; fi
 echo "START vLLM $arm $(date -u +%T)"
 timeout -k 10 1800 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_async_bwmatch.py > "$res/raw/tp8_llama70_${arm}_retry_20261113.log" 2>&1
 echo "END vLLM $arm rc=$? $(date -u +%T)"
done
