#!/usr/bin/env bash
# Collect missing A800 inputs for the original Stotaway cost model, serially.
# Existing results remain untouched. Resume by rerunning the script.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
export OMP_NUM_THREADS=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NVSHMEM_HOME="$(python -c 'import os,site;print(next(os.path.join(p,"nvidia","nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p,"nvidia","nvshmem","lib","libnvshmem_host.so.3"))))')"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_SYMMETRIC_SIZE=4G CUDA_DEVICE_MAX_CONNECTIONS=1
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1 TP8_CHECKS=4 TP8_REPEATS=8
raw=/root/data/stowaway-nvidia/results/raw
data=/root/data/stowaway-nvidia/results/tp8_20261111
status="$raw/original_tp8_calibration.tsv"
mkdir -p "$raw" "$data"
if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 2; done
fi
record() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >> "$status"; }
run_probe() {
  local model=$1 m=$2 k=$3 n=$4 batch=$5 op=$6 path log rc
  path="$data/bwmatch_probe_${model}_m${m}_k${k}_n${n}_b${batch}.json"
  log="$raw/bwmatch_probe_${model}_m${m}_k${k}_n${n}_b${batch}.log"
  if [[ -f "$path" ]]; then return; fi
  export TP8_M="$m" TP8_K="$k" TP8_N="$n" TP8_BATCH="$batch" TP8_OUT="$path" TP8_ONLY="$op"
  record probe "$model/$m/$k/$n/b$batch" start
  timeout 360 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/tp8_operator.py >> "$log" 2>&1
  rc=$?
  record probe "$model/$m/$k/$n/b$batch" "$rc"
}
run_component() {
  local model=$1 m=$2 k=$3 n=$4 op=$5 path log rc
  path="$data/component_${model}_m${m}_k${k}_n${n}_${op}.json"
  log="$raw/component_${model}_m${m}_k${k}_n${n}_${op}.log"
  if [[ -f "$path" ]]; then return; fi
  export TP8_M="$m" TP8_K="$k" TP8_N="$n" TP8_OP="$op" TP8_OUT="$path"
  record component "$model/$m/$k/$n/$op" start
  timeout 360 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/tp8_calibration_components.py >> "$log" 2>&1
  rc=$?
  record component "$model/$m/$k/$n/$op" "$rc"
}
while read -r model m k n op; do
  run_probe "$model" "$m" "$k" "$n" 1 "$op"
  run_probe "$model" "$m" "$k" "$n" 16 "$op"
  run_component "$model" "$m" "$k" "$n" "$op"
  python planner/original_a800/extract_a800.py >> "$raw/original_tp8_extract.log" 2>&1 || true
  python planner/original_a800/validate.py >> "$raw/original_tp8_validation.log" 2>&1 || true
done < <(python - <<'PY'
import json
from pathlib import Path
rows=json.loads(Path('/root/Stotaway/planner/data/cells.json').read_text())
models=sorted({r['model'] for r in rows})
# Round-robin models, so an interrupted run covers the whole model family.
for m in (8192,16384,32768):
 for model in models:
  for op in ('rs','ag_up','ag_gu'):
   for r in rows:
    if (r['model'],r['M'],r['op'])==(model,m,op):
     print(model,m,r['K'],r['N'],'rs' if op=='rs' else 'ag')
PY
)
record queue all finished
