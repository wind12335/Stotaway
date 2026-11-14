#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
result=/root/data/stowaway-nvidia/results
flash=/root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113
sync=/root/data/stowaway-nvidia/syncopate_tp2_work_20261113
export PYTHONPATH="$PWD:/root/Triton-distributed/python:$sync" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export NVSHMEM_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0 NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET NVSHMEM_SYMMETRIC_SIZE=4G
export STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
export REPLACEMENT_PLAN="$PWD/experiments/tp8_replacement_coverage_20261113.json" REPLACEMENT_RESULT="$result/tp8_replacement_coverage_20261113.json"
if [[ ! -e "$REPLACEMENT_RESULT" ]]; then
 echo "START TP8 replacement $(date -u +%T)"
 timeout -k 10 180 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_replacement.py > "$result/tp8_replacement_coverage_20261113.log" 2>&1
 echo "END TP8 replacement rc=$? $(date -u +%T)"
fi
export REPLACEMENT_BENCH_OUT="$result/tp8_async_dx_rs_paired_20261113.json"
if [[ ! -e "$REPLACEMENT_BENCH_OUT" ]]; then
 echo "START paired Async-TP/DX RS $(date -u +%T)"
 timeout -k 10 180 python -m torch.distributed.run --standalone --nproc-per-node=8 bench/tp8_replacement_benchmark.py > "$result/tp8_async_dx_rs_paired_20261113.log" 2>&1
 echo "END paired Async-TP/DX RS rc=$? $(date -u +%T)"
fi
for sites in none proj,fc2; do
 export VS_ASYNC_DX_SITES="$sites"
 if [[ "$sites" == none ]]; then export VS_ASYNC_DX_SITES=""; fi
 echo "START vLLM synthetic $sites $(date -u +%T)"
 timeout -k 10 180 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/test_tp8_vllm_async_synthetic.py > "$result/tp8_vllm_synthetic_${sites}_20261113.log" 2>&1
 echo "END vLLM synthetic $sites rc=$? $(date -u +%T)"
done
for op in ag rs; do
 for arm in stock syncopate; do
  export SYNC_MANIFEST="$PWD/experiments/tp8_syncopate_${op}_20261113.json" SYNC_WORK="$sync" SYNC_ARM="$arm" SYNC_OUT="$result/tp8_syncopate_${op}_${arm}_20261113.json"
  if [[ -e "$SYNC_OUT" ]]; then continue; fi
  echo "START Syncopate $op $arm $(date -u +%T)"
  timeout -k 10 150 python -m torch.distributed.run --standalone --nproc-per-node=8 "bench/tp8_syncopate_${op}.py" > "$result/tp8_syncopate_${op}_${arm}_20261113.log" 2>&1
  echo "END Syncopate $op $arm rc=$? $(date -u +%T)"
 done
done
cd "$flash/tune" || exit 1
export MPLBACKEND=Agg
if [[ ! -f ../configs/bandwidth_reduce_scatter_tp8.pt ]]; then
 echo "START FlashOverlap TP8 bandwidth $(date -u +%T)"
 timeout -k 10 900 python -u bandwidth.py --comm_op reduce_scatter > "$result/tp8_flash_bandwidth_20261113.log" 2>&1
 echo "END FlashOverlap TP8 bandwidth rc=$? $(date -u +%T)"
fi
if [[ -f ../configs/bandwidth_reduce_scatter_tp8.pt ]]; then
 echo "START FlashOverlap configuration $(date -u +%T)"
 timeout -k 10 240 python profile_config.py --m 8192 --n 4096 --k 1792 > "$result/tp8_flash_profile_20261113.log" 2>&1
 echo "END FlashOverlap profile rc=$? $(date -u +%T)"
 timeout -k 10 600 python search.py --m 8192 --n 4096 --k 1792 --comm_op reduce_scatter --predictive_search True > "$result/tp8_flash_search_20261113.log" 2>&1
 echo "END FlashOverlap search rc=$? $(date -u +%T)"
fi
config="$flash/configs/m8192n4096k1792_a800.json"
if [[ -f "$config" ]] && python - "$config" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));raise SystemExit(0 if len(x.get('cSeg',[]))>1 and isinstance(x.get('BM'),int) else 1)
PY
then
 cd "$flash/example" || exit 1
 echo "START FlashOverlap correctness $(date -u +%T)"
 timeout -k 10 180 python /root/Stotaway/nvidia/bench/tp8_flash_output_contract.py --m 8192 --n 4096 --k 1792 --config "$config" --output "$result/tp8_flash_correctness_20261113.json" > "$result/tp8_flash_correctness_20261113.log" 2>&1
 rc=$?
 echo "END FlashOverlap correctness rc=$rc $(date -u +%T)"
 if [[ $rc -eq 0 ]]; then
  cd "$flash/test" || exit 1
  echo "START FlashOverlap benchmark $(date -u +%T)"
  timeout -k 10 240 python /root/Stotaway/nvidia/bench/flashoverlap_tp8_b16.py --m 8192 --n 4096 --k 1792 --config "$config" --output "$result/tp8_flash_b16_20261113.json" > "$result/tp8_flash_b16_20261113.log" 2>&1
  echo "END FlashOverlap benchmark rc=$? $(date -u +%T)"
 fi
fi
