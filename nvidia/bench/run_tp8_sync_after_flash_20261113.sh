#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
while pgrep -f '^bash bench/run_tp8_flash_matrix_20261113.sh$' >/dev/null; do
  sleep 15
done
bash bench/run_tp8_sync_matrix_20261113.sh > /root/data/stowaway-nvidia/results/tp8_sync_retry_driver_20261113.log 2>&1
python bench/tp8_report_20261113.py
