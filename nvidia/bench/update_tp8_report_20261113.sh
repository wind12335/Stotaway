#!/usr/bin/env bash
cd /root/Stotaway/nvidia || exit 1
while pgrep -f '^bash /tmp/run_td_matrix_conn8_20261113.sh$' >/dev/null || pgrep -f '^bash /tmp/tp8_queue_followup_20261113.sh$' >/dev/null; do
 python bench/tp8_report_20261113.py > /root/data/stowaway-nvidia/results/tp8_report_driver_20261113.log 2>&1
 python bench/report_tp8_e2e_async_20261113.py
 sleep 20
done
python bench/tp8_report_20261113.py > /root/data/stowaway-nvidia/results/tp8_report_driver_20261113.log 2>&1
python bench/report_tp8_e2e_async_20261113.py
