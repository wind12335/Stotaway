#!/usr/bin/env bash
set -u
root=/root/data/stowaway-nvidia/flashoverlap_bf16_20261114
logs=/root/data/stowaway-nvidia/results/flash_bf16_model_tune_20261114
mkdir -p "$logs"
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8 MPLBACKEND=Agg PYTHONPATH=/root/Stotaway/nvidia
cd "$root/tune" || exit 1
while read -r m n k; do
 tag="m${m}n${n}k${k}"
 cfg="$root/configs/${tag}_a800.json"
 if [[ -f "$cfg" ]] && python - "$cfg" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); assert isinstance(c.get('hint'),list) and isinstance(c.get('cSeg'),list)
PY
 then echo "SKIP $tag"; continue; fi
 echo "START $tag $(date -u +%T)"
 timeout -k 5 240 python profile_config.py --m "$m" --n "$n" --k "$k" > "$logs/${tag}_profile.log" 2>&1
 rc=$?
 if ((rc == 0)); then
  timeout -k 5 360 python search.py --m "$m" --n "$n" --k "$k" --comm_op reduce_scatter --predictive_search True > "$logs/${tag}_search.log" 2>&1
  rc=$?
 fi
 echo "END $tag rc=$rc $(date -u +%T)"
done <<'SHAPES'
8192 5120 768
8192 5120 2176
16384 8192 1024
16384 8192 3584
SHAPES
