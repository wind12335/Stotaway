#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results/tp8_dx_bf16_retest_20261114
mkdir -p "$res"
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G CUDA_DEVICE_MAX_CONNECTIONS=1
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1 TP8_CHECKS=4 TP8_REPEATS=8 TP8_BATCH=16
while read -r model op m n k; do
 export TP8_M="$m" TP8_N="$n" TP8_K="$k" TP8_ONLY="$([[ "$op" == rs ]] && echo rs || echo ag)"
 export TP8_OUT="$res/${model}_${op}_${m}_${n}_${k}.json"
 [[ -f "$TP8_OUT" ]] && continue
 echo "START $model $op $m $n $k $(date -u +%T)"
 timeout -k 5 360 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/tp8_operator.py > "$res/${model}_${op}_${m}_${n}_${k}.log" 2>&1
 echo "END $model $op $m $n $k rc=$? $(date -u +%T)"
done < <(python - <<'PY'
import json
from pathlib import Path
for x in sorted(json.loads(Path('planner/original_a800/data/cells.json').read_text()),key=lambda x:(x['M'],x['model'],x['op'])):
 print(x['model'],x['op'],x['M'],x['N'],x['K'])
PY
)
