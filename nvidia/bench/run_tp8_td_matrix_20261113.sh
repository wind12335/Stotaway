#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
export PYTHONPATH="$PWD:/root/Triton-distributed/python" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G
export TP8_INVENTORY="$PWD/planner/original_a800/data/cells.json"
export TP8_MATRIX_OP="${1:-rs}" TP8_MATRIX_OUT="/root/data/stowaway-nvidia/results/tp8_td_${1:-rs}_20261113.jsonl"
remaining=$(mktemp /tmp/tp8_td_remaining.XXXXXX)
python - "$TP8_MATRIX_OP" "$TP8_MATRIX_OUT" > "$remaining" <<'PY'
import json,sys
from pathlib import Path
op,out=sys.argv[1:]
rows=json.loads(Path('/root/Stotaway/nvidia/planner/original_a800/data/cells.json').read_text())
p=Path(out); done={tuple(json.loads(x)['identity']) for x in p.read_text().splitlines()} if p.exists() else set()
for r in sorted(rows,key=lambda x:(x['M'],x['model'],x['op'])):
 if ('ag' if r['op'].startswith('ag') else 'rs')==op and (r['model'],r['op'],r['M'],r['N'],r['K']) not in done:
  print(r['model'],r['op'],r['M'],sep='\t')
PY
failures=0
completed=0
while IFS=$'\t' read -r model op m; do
 export TP8_MATRIX_MODEL="$model" TP8_EXACT_OP="$op" TP8_MATRIX_M="$m"
 echo "START $model $op $m $(date -u +%T)"
 timeout -k 10 75 python -m torch.distributed.run --standalone --nproc-per-node=8 bench/tp8_triton_dist_matrix.py > "/root/data/stowaway-nvidia/results/tp8_td_${model}_${op}_${m}_conn8_20261113.log" 2>&1
 rc=$?
 echo "END $model $op $m rc=$rc $(date -u +%T)"
 if (( rc == 0 )); then completed=$((completed+1)); failures=0; else failures=$((failures+1)); fi
 if (( failures >= 3 )); then echo "STOP: three consecutive points failed; retain logs and continue other experiments"; break; fi
done < "$remaining"
rm -f "$remaining"
