#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
raw="$res/raw"
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export PYTHONPATH="$PWD/third_party/qwen36_fla:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_SYMMETRIC_SIZE=2G STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_DX_WORLD=8
export STOWAWAY_QWEN36=1 STOWAWAY_QWEN36_LOCAL_SPEC=1 STOWAWAY_LAYERS=64 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=24
export STOWAWAY_RECOMPUTE=1 STOWAWAY_SEQ=4096 STOWAWAY_MBS=2 STOWAWAY_GBS=2 STOWAWAY_ITERS=12 STOWAWAY_DISTRIBUTED_OPTIMIZER=1 TEST_TIMEOUT=1500 STOWAWAY_OVERLAP_GRAD_REDUCE=1
export STOWAWAY_DX=0 STOWAWAY_ASYNC_TP=1 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 STOWAWAY_COLUMN_MAX_N=20000
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
unset STOWAWAY_TP2_LOOKUP STOWAWAY_DX_SHAPE_PLAN STOWAWAY_TP2_ONLINE_BUNDLE STOWAWAY_BRS STOWAWAY_DX8_BACKWARD
for arm in stock async_tp async_dx; do
 export STOWAWAY_ASYNC_DX_SITES="" STOWAWAY_ASYNC_TP=1
 if [[ $arm == stock ]]; then export STOWAWAY_ASYNC_TP=0; fi
 if [[ $arm == async_dx ]]; then export STOWAWAY_ASYNC_DX_SITES=proj,fc2; fi
 export OUT="$raw/tp8_qwen36_${arm}_mbs2_fla_retry_20261113.log"
 if [[ -e "$OUT" ]]; then continue; fi
 echo "START Megatron $arm $(date -u +%T)"
 timeout -k 10 1650 bash experiments/run_megatron_async_smoke.sh > "$raw/tp8_qwen36_${arm}_mbs2_fla_retry_driver_20261113.log" 2>&1
 echo "END Megatron $arm rc=$? $(date -u +%T)"
done
