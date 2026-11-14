#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
while pgrep -f '^bash bench/run_tp8_dx_fp16_matrix_20261114.sh$' >/dev/null; do sleep 10; done
python -m planner.original_a800.evaluate_tp8_fp16_threeway_20261114 > /root/data/stowaway-nvidia/results/tp8_fp16_predictor_lomo_driver_20261114.log 2>&1
bash bench/run_tp8_dx_bf16_retest_20261114.sh > /root/data/stowaway-nvidia/results/tp8_dx_bf16_retest_driver_20261114.log 2>&1
python bench/tp8_report_20261113.py > /root/data/stowaway-nvidia/results/tp8_report_after_dx_retest_20261114.log 2>&1
