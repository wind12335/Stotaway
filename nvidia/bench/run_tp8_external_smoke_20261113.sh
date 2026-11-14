#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
export PYTHONPATH="$PWD:/root/Triton-distributed/python:/root/data/stowaway-nvidia/syncopate_tp2_work_20261113"
export STOWAWAY_FLASH_ROOT=/root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113
export STOWAWAY_SYNC_WORK=/root/data/stowaway-nvidia/syncopate_tp2_work_20261113
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_SYMMETRIC_SIZE=4G STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
if ! python - <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path('/root/data/stowaway-nvidia/results')
p=r/'tp8_flash_b16_20261113.json'
record=json.loads(p.read_text())
config=Path(record['config'])
if hashlib.sha256(config.read_bytes()).hexdigest()!=record['config_sha256']:
 raise SystemExit('FlashOverlap TP8 smoke config changed after its correctness check')
PY
then exit 1; fi
for label in triton_dist syncopate flashoverlap syncopate_bf16; do
 export TEST_EXTERNAL_BACKEND="${label%_bf16}"
 export TEST_EXTERNAL_PLAN="$PWD/experiments/tp8_external_smoke_${label}_20261113.json"
 export TEST_EXTERNAL_OUT="$res/tp8_external_dispatch_${label}_20261113.json"
 if [[ -f "$TEST_EXTERNAL_OUT" ]]; then continue; fi
 echo "START external dispatcher $label $(date -u +%T)"
 for attempt in 1 2; do
  timeout -k 10 300 /root/miniconda3/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_external_dispatch.py > "$res/tp8_external_dispatch_${label}_20261113.log" 2>&1
  rc=$?
  echo "END external dispatcher $label attempt=$attempt rc=$rc $(date -u +%T)"
  if ((rc == 0)); then break; fi
 done
done
python bench/tp8_report_20261113.py
