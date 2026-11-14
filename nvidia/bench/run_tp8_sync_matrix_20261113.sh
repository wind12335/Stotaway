#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results
work=/root/data/stowaway-nvidia/syncopate_tp2_work_20261113
export PYTHONPATH="$PWD:/root/Triton-distributed/python:$work" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G
export SYNC_WORK="$work"
mkdir -p "$res/tp8_sync_matrix_20261113"
python - <<'PY'
import json
from pathlib import Path
r=Path('/root/data/stowaway-nvidia/results/tp8_sync_matrix_20261113')
rows=json.loads(Path('planner/original_a800/data/cells.json').read_text())
for x in rows:
 op='ag' if x['op'].startswith('ag') else 'rs'
 name=f"{x['model']}_{x['op']}_{x['M']}"
 (r/f'{name}.manifest.json').write_text(json.dumps({'tp':8,'dtype':'float16','batch':16,'warmup':4,'repeats':8,'target':{'op':op,'M':x['M'],'N':x['N'],'K':x['K']}})+'\n')
PY
for m in 8192 16384 32768; do
 for manifest in "$res"/tp8_sync_matrix_20261113/*_"$m".manifest.json; do
  export SYNC_MANIFEST="$manifest"
  name=$(basename "$manifest" .manifest.json)
  op=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["target"]["op"])' "$manifest")
  for arm in stock syncopate; do
   export SYNC_ARM="$arm" SYNC_OUT="$res/tp8_sync_matrix_20261113/${name}_${arm}.json"
   if [[ -f "$SYNC_OUT" ]]; then continue; fi
   echo "START $name $arm $(date -u +%T)"
   timeout -k 10 "${SYNC_TIMEOUT:-180}" python -m torch.distributed.run --standalone --nproc-per-node=8 "bench/tp8_syncopate_${op}.py" > "$res/tp8_sync_matrix_20261113/${name}_${arm}.log" 2>&1
   rc=$?; echo "END $name $arm rc=$rc $(date -u +%T)"
  done
  python bench/tp8_report_20261113.py > "$res/tp8_report_driver_20261113.log" 2>&1
done
done
