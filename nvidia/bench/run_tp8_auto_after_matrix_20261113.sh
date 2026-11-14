#!/usr/bin/env bash
set -u
cd /root/Stotaway/nvidia || exit 1
while pgrep -f '^bash bench/run_tp8_flash_matrix_20261113.sh$' >/dev/null || pgrep -f '^bash (/root/Stotaway/nvidia/)?bench/run_tp8_sync_after_flash_20261113.sh$' >/dev/null || pgrep -f '^bash bench/run_tp8_sync_matrix_20261113.sh$' >/dev/null; do
  sleep 15
done
bash bench/run_tp8_auto_e2e_20261113.sh
bash bench/run_tp8_external_smoke_20261113.sh > /root/data/stowaway-nvidia/results/tp8_external_dispatch_driver_20261113.log 2>&1
bash bench/run_tp8_td_e2e_20261113.sh > /root/data/stowaway-nvidia/results/tp8_td_e2e_driver_20261113.log 2>&1
FLASH_RETRY_LARGER_GROUP=1 bash bench/run_tp8_flash_matrix_20261113.sh > /root/data/stowaway-nvidia/results/tp8_flash_retry_driver_20261113.log 2>&1
