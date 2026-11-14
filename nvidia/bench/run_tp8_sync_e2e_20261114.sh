#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
tdpath=/root/Triton-distributed/3rdparty/triton/python:/root/Triton-distributed/python
sync=/root/data/stowaway-nvidia/syncopate_tp2_work_20261113
export STOWAWAY_SYNC_WORK="$sync" STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=5G
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH PYTHONPATH="$tdpath:$sync:$PWD/third_party/qwen36_fla:$PWD"
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}
export STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_DX_WORLD=8
export STOWAWAY_QWEN36=1 STOWAWAY_QWEN36_LOCAL_SPEC=1 STOWAWAY_LAYERS=64 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=24
export STOWAWAY_RECOMPUTE=1 STOWAWAY_SEQ=4096 STOWAWAY_MBS=2 STOWAWAY_GBS=2 STOWAWAY_ITERS=12 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 TEST_TIMEOUT=900 STOWAWAY_OVERLAP_GRAD_REDUCE=1
export STOWAWAY_DX=0 STOWAWAY_ASYNC_TP=0 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 STOWAWAY_COLUMN_MAX_N=20000
unset STOWAWAY_AUTO_PLAN8 STOWAWAY_ASYNC_DX_SITES STOWAWAY_DX_SHAPE_PLAN STOWAWAY_TP2_LOOKUP STOWAWAY_TP2_ONLINE_BUNDLE
for arm in baseline auto; do
 export STOWAWAY_MULTI_PLAN8="$PWD/experiments/tp8_${arm}_sync_qwen36_20261114.json"
 export OUT="$res/raw/tp8_qwen36_sync_${arm}_mbs2_fla_retry_20261114.log"
 if [[ -e "$OUT" ]] && grep -q 'iteration *12 */ *12' "$OUT"; then continue; fi
 echo "START Qwen Syncopate $arm $(date -u +%T)"
 timeout -k 10 960 bash experiments/run_megatron_async_smoke.sh > "$res/raw/tp8_qwen36_sync_${arm}_driver_20261114.log" 2>&1
 rc=$?
 echo "END Qwen Syncopate $arm rc=$rc $(date -u +%T)"
 python bench/report_tp8_e2e_async_20261113.py
done
unset STOWAWAY_MULTI_PLAN8
export PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/bin:$PATH PYTHONPATH="$tdpath:$sync:$PWD" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem LD_LIBRARY_PATH=/root/data/stowaway-nvidia/envs/vllm-cu128/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B V8_M=16384 V8_ROPE_FACTOR=8 V8_REPS=3 V8_ARM=multi_plan
for arm in baseline auto; do
 export V8_AUTO_PLAN="$PWD/experiments/tp8_${arm}_sync_llama70_20261114.json"
 export V8_OUT="$res/tp8_20261111/vllm_bwmatch_sync_${arm}_retry_20261114.json"
 if [[ -e "$V8_OUT" ]]; then continue; fi
 echo "START Llama Syncopate $arm $(date -u +%T)"
 timeout -k 10 1800 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_async_bwmatch.py > "$res/raw/tp8_llama70_sync_${arm}_retry_20261114.log" 2>&1
 rc=$?
 echo "END Llama Syncopate $arm rc=$rc $(date -u +%T)"
 python bench/report_tp8_e2e_async_20261113.py
done
python bench/report_tp8_e2e_async_20261113.py
