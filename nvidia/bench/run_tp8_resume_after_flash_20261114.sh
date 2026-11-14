#!/usr/bin/env bash
set -u
while pgrep -f '^bash bench/run_tp8_flash_matrix_20261113.sh$' >/dev/null; do sleep 15; done
cd /root/Stotaway/nvidia || exit 1
bash bench/run_tp8_sync_e2e_20261114.sh > /root/data/stowaway-nvidia/results/tp8_sync_e2e_driver_20261114.log 2>&1
bash bench/run_tp8_flash_bf16_smoke_20261114.sh > /root/data/stowaway-nvidia/results/tp8_flash_bf16_smoke_driver_20261114.log 2>&1
