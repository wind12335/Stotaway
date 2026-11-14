#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
raw="$res/raw"
mkdir -p "$raw"
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export PYTHONPATH="$PWD/third_party/qwen36_fla${PYTHONPATH:+:$PYTHONPATH}"
export NVSHMEM_SYMMETRIC_SIZE=2G STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_DX_WORLD=8
export STOWAWAY_QWEN36=1 STOWAWAY_QWEN36_LOCAL_SPEC=1 STOWAWAY_LAYERS=64 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=24
export STOWAWAY_RECOMPUTE=1 STOWAWAY_SEQ=4096 STOWAWAY_MBS=2 STOWAWAY_GBS=2 STOWAWAY_ITERS=12 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 TEST_TIMEOUT=1500 STOWAWAY_OVERLAP_GRAD_REDUCE=1
export STOWAWAY_DX=0 STOWAWAY_ASYNC_TP=1 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 STOWAWAY_COLUMN_MAX_N=20000
unset STOWAWAY_TP2_LOOKUP STOWAWAY_DX_SHAPE_PLAN STOWAWAY_TP2_ONLINE_BUNDLE STOWAWAY_BRS STOWAWAY_DX8_BACKWARD
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
for arm in stock async_tp async_dx; do
 export STOWAWAY_ASYNC_DX_SITES="" STOWAWAY_ASYNC_TP=1
 if [[ $arm == stock ]]; then export STOWAWAY_ASYNC_TP=0; fi
 if [[ $arm == async_dx ]]; then export STOWAWAY_ASYNC_DX_SITES=proj,fc2; fi
 export OUT="$raw/tp8_qwen36_${arm}_mbs2_20261113.log"
 if [[ -e "$OUT" ]]; then continue; fi
 echo "START Megatron $arm $(date -u +%T)"
 timeout -k 10 1650 bash experiments/run_megatron_async_smoke.sh > "$raw/tp8_qwen36_${arm}_driver_20261113.log" 2>&1
 echo "END Megatron $arm rc=$? $(date -u +%T)"
done
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" NVSHMEM_SYMMETRIC_SIZE=2G
export V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B V8_M=16384 V8_ROPE_FACTOR=4 V8_REPS=3
for arm in stock async_tp async_dx; do
 export V8_ARM="$arm" V8_OUT="$res/tp8_20261111/vllm_bwmatch_${arm}_20261113.json"
 if [[ -e "$V8_OUT" ]]; then continue; fi
 echo "START vLLM $arm $(date -u +%T)"
 timeout -k 10 1800 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_async_bwmatch.py > "$raw/tp8_llama70_${arm}_20261113.log" 2>&1
 echo "END vLLM $arm rc=$? $(date -u +%T)"
done
