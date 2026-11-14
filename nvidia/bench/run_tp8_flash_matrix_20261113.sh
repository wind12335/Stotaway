#!/usr/bin/env bash
set -u
root=/root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113
res=/root/data/stowaway-nvidia/results/tp8_flash_matrix_20261113
mkdir -p "$res"
export PYTHONPATH=/root/Stotaway/nvidia OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8 MPLBACKEND=Agg
cd "$root/tune" || exit 1
python - <<'PY' > /tmp/tp8_flash_shapes_20261113.tsv
import json
from pathlib import Path
for r in sorted(json.loads(Path('/root/Stotaway/nvidia/planner/original_a800/data/cells.json').read_text()),key=lambda x:(x['M'],x['model'])):
 if r['op']=='rs' and not (r['model']=='llama3-8b' and r['M']==8192):print(r['model'],r['M'],r['N'],r['K'],sep='\t')
PY
while IFS=$'\t' read -r model m n k; do
 name="${model}_${m}_${n}_${k}"
 cfg="$root/configs/m${m}n${n}k${k}_a800.json"
 result="$res/${name}.json"
 if [[ -f "$result" ]]; then continue; fi
 echo "START $name $(date -u +%T)"
 timeout -k 5 180 python profile_config.py --m "$m" --n "$n" --k "$k" > "$res/${name}_profile.log" 2>&1
 rc=$?; if ((rc != 0)); then python - "$res/${name}_profile.log" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
 echo "END $name profile=$rc"; continue; fi
 search_script=search.py
 if [[ "${FLASH_RETRY_LARGER_GROUP:-0}" == 1 ]]; then search_script=search_retry_stowaway.py; fi
 timeout -k 5 360 python "$search_script" --m "$m" --n "$n" --k "$k" --comm_op reduce_scatter --predictive_search True > "$res/${name}_search.log" 2>&1
 rc=$?; if ((rc != 0)); then python - "$res/${name}_search.log" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
 echo "END $name search=$rc"; continue; fi
 cd "$root/example" || exit 1
 timeout -k 5 120 python /root/Stotaway/nvidia/bench/tp8_flash_output_contract.py --m "$m" --n "$n" --k "$k" --config "$cfg" --output "$res/${name}_check.json" > "$res/${name}_check.log" 2>&1
 rc=$?
 if ((rc != 0)); then python - "$res/${name}_check.log" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
 fi
 if ((rc == 0)); then
  cd "$root/test" || exit 1
  timeout -k 5 180 python /root/Stotaway/nvidia/bench/flashoverlap_tp8_b16.py --m "$m" --n "$n" --k "$k" --config "$cfg" --output "$result" > "$res/${name}_measure.log" 2>&1
  rc=$?
  if ((rc != 0)); then python - "$res/${name}_measure.log" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
  fi
 fi
 echo "END $name rc=$rc $(date -u +%T)"
 cd "$root/tune" || exit 1
 python /root/Stotaway/nvidia/bench/tp8_report_20261113.py > /root/data/stowaway-nvidia/results/tp8_report_driver_20261113.log 2>&1
done < /tmp/tp8_flash_shapes_20261113.tsv
