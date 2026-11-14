#!/usr/bin/env bash
set -u
cd /root/data/stowaway-nvidia/flashoverlap_tp8_work_20261113/tune || exit 1
export PYTHONPATH="/root/Stotaway/nvidia${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8 MPLBACKEND=Agg
result=/root/data/stowaway-nvidia/results
config=../configs/m8192n4096k1792_a800.json
echo "START FlashOverlap profile $(date -u +%T)"
timeout -k 10 240 python profile_config.py --m 8192 --n 4096 --k 1792 > "$result/tp8_flash_profile_20261113.log" 2>&1
profile_rc=$?
echo "END FlashOverlap profile rc=$profile_rc $(date -u +%T)"
if (( profile_rc != 0 )); then exit "$profile_rc"; fi
echo "START FlashOverlap search $(date -u +%T)"
timeout -k 10 600 python search.py --m 8192 --n 4096 --k 1792 --comm_op reduce_scatter --predictive_search True > "$result/tp8_flash_search_20261113.log" 2>&1
search_rc=$?
echo "END FlashOverlap search rc=$search_rc $(date -u +%T)"
if (( search_rc != 0 )); then exit "$search_rc"; fi
config=$(realpath "$config")
cd ../example || exit 1
echo "START FlashOverlap correctness $(date -u +%T)"
timeout -k 10 180 python /root/Stotaway/nvidia/bench/tp8_flash_output_contract.py --m 8192 --n 4096 --k 1792 --config "$config" --output "$result/tp8_flash_correctness_20261113.json" > "$result/tp8_flash_correctness_20261113.log" 2>&1
rc=$?
echo "END FlashOverlap correctness rc=$rc $(date -u +%T)"
if (( rc != 0 )); then exit "$rc"; fi
cd ../test || exit 1
echo "START FlashOverlap benchmark $(date -u +%T)"
timeout -k 10 240 python /root/Stotaway/nvidia/bench/flashoverlap_tp8_b16.py --m 8192 --n 4096 --k 1792 --config "$config" --output "$result/tp8_flash_b16_20261113.json" > "$result/tp8_flash_b16_20261113.log" 2>&1
rc=$?
echo "END FlashOverlap benchmark rc=$rc $(date -u +%T)"
exit "$rc"
