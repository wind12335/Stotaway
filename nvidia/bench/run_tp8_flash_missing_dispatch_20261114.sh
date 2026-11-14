#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
res=/root/data/stowaway-nvidia/results/tp8_flash_missing_dispatch_20261114
mkdir -p "$res"
export PYTHONPATH="$PWD:/root/Triton-distributed/python" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export STOWAWAY_FLASH_ROOT=/root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113 TEST_EXTERNAL_BACKEND=flashoverlap
while read -r model m n k; do
 tag="${model}_${m}_${n}_${k}"
 cfg="$STOWAWAY_FLASH_ROOT/configs/m${m}n${n}k${k}_a800.json"
 result="$res/${tag}.json"
 [[ -f "$result" ]] && continue
 if [[ ! -f "$cfg" ]]; then echo "SKIP $tag missing-config"; continue; fi
 export TEST_EXTERNAL_PLAN="$res/${tag}_plan.json" TEST_EXTERNAL_OUT="$result"
 python - "$TEST_EXTERNAL_PLAN" "$m" "$n" "$k" <<'PY'
import json,sys
p,m,n,k=sys.argv[1:];v={'schema':'stowaway-tp8-auto-plan-v1','workload':'operator_smoke','tp':8,'dtype':'float16','external_backend':'flashoverlap','default':'stock','entries':[{'site':'proj','op':'rs','M':int(m),'N':int(n),'K':int(k),'choice':'flashoverlap'}]};open(p,'w').write(json.dumps(v))
PY
 echo "START $tag $(date -u +%T)"
 timeout -k 5 180 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_external_dispatch.py > "$res/${tag}.log" 2>&1
 rc=$?
 echo "END $tag rc=$rc $(date -u +%T)"
done <<'SHAPES'
gemma3-27b 8192 5376 2688
gemma3-27b 16384 5376 2688
gemma3-27b 32768 5376 2688
llama4-mav-d 8192 5120 2048
llama4-mav-d 16384 5120 2048
llama4-mav-d 32768 5120 2048
qwen3-32b 8192 5120 1280
qwen3-32b 16384 5120 1280
qwen3-32b 32768 5120 1280
SHAPES
