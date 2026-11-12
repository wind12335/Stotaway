#!/usr/bin/env bash
# Serial, independent-process A800 reproduction of the single-node vehicle shapes.
# This is a partial forward-DX comparison, not the BW fbag/fbc implementation.
set -u
cd "$(dirname "$0")/.."
export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH OMP_NUM_THREADS=1
export STOWAWAY_WORLD_SIZE=8 STOWAWAY_TP=8 STOWAWAY_VEHICLE_27B=1 STOWAWAY_BWMATCH=1
export STOWAWAY_SEQ=4096 STOWAWAY_MAX_POSITION=40960 STOWAWAY_RECOMPUTE=1
export STOWAWAY_DISTRIBUTED_OPTIMIZER=1 STOWAWAY_LR=0.00003 STOWAWAY_MIN_LR=0.000003 STOWAWAY_LR_STYLE=cosine
export STOWAWAY_DX_WORLD=8 STOWAWAY_DX8_SCHEDULE=shift STOWAWAY_DX8_BATCH_ACK=1
export STOWAWAY_COLUMN_MAX_N=20000 NVSHMEM_SYMMETRIC_SIZE=4G
deadline_file=/root/data/stowaway-nvidia/results/raw/bwmatch_deadline_epoch
deadline=${STOWAWAY_DEADLINE_EPOCH:-0}
if [[ -r "$deadline_file" ]]; then deadline=$(<"$deadline_file"); fi
raw=/root/data/stowaway-nvidia/results/raw
mkdir -p "$raw"
status="$raw/bwmatch_tp8_queue_status.tsv"
printf 'utc\tmodel\tmbs\tarm\trepeat\tstate\trc\n' >> "$status"
if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 5; done
fi
run() {
  local model=$1 mbs=$2 arm=$3 repeat=$4 iters=$5 rc remaining
  if [[ -r "$deadline_file" ]]; then deadline=$(<"$deadline_file"); fi
  if (( deadline > 0 )); then
    remaining=$((deadline - $(date +%s)))
    if (( remaining < 120 )); then
      printf '%s\t%s\t%s\t%s\t%s\tdeadline_skip\t-\n' "$(date -u +%FT%TZ)" "$model" "$mbs" "$arm" "$repeat" >> "$status"
      return
    fi
    export TEST_TIMEOUT=$((remaining < 2400 ? remaining : 2400))
  else
    export TEST_TIMEOUT=2400
  fi
  case "$model" in
    27b) export STOWAWAY_LAYERS=32 STOWAWAY_HIDDEN=5120 STOWAWAY_FFN=17408 STOWAWAY_HEADS=40 ;;
    70b) export STOWAWAY_LAYERS=8 STOWAWAY_HIDDEN=8192 STOWAWAY_FFN=28672 STOWAWAY_HEADS=64 ;;
    *) return 2 ;;
  esac
  export STOWAWAY_MBS="$mbs" STOWAWAY_GBS=$((8*mbs)) STOWAWAY_ITERS="$iters"
  case "$arm" in
    stock) export STOWAWAY_DX=0 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 ;;
    null) export STOWAWAY_DX=1 STOWAWAY_DX_SITES=, ;;
    row) export STOWAWAY_DX=1 STOWAWAY_DX_SITES=proj,fc2 ;;
    all) export STOWAWAY_DX=1 STOWAWAY_DX_SITES=qkv,proj,fc1,fc2 ;;
    row_qkv) export STOWAWAY_DX=1 STOWAWAY_DX_SITES=qkv,proj,fc2 ;;
    *) return 2 ;;
  esac
  export OUT="$raw/bwmatch_${model}_mbs${mbs}_${arm}_${repeat}.log"
  if [[ -e "$OUT" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\texists_skip\t-\n' "$(date -u +%FT%TZ)" "$model" "$mbs" "$arm" "$repeat" >> "$status"
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\tstart\t-\n' "$(date -u +%FT%TZ)" "$model" "$mbs" "$arm" "$repeat" >> "$status"
  bash experiments/run_megatron_smoke.sh > "${OUT%.log}_driver.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\tend\t%s\n' "$(date -u +%FT%TZ)" "$model" "$mbs" "$arm" "$repeat" "$rc" >> "$status"
  /usr/bin/python3 experiments/report_bwmatch_tp8.py >> "$raw/bwmatch_report.log" 2>&1 || true
}
run_vllm() {
  local arm=$1 repeat=$2 rc remaining venv=/dev/shm/stowaway-vllm-tp8 suffix=
  if [[ -r "$deadline_file" ]]; then deadline=$(<"$deadline_file"); fi
  remaining=$((deadline - $(date +%s)))
  if (( deadline > 0 && remaining < 120 )); then return; fi
  export PATH="$venv/bin:$PATH" V8_MODEL=/root/eb-public/huggingface-models/meta-llama/Meta-Llama-3-70B
  export V8_M=16384 V8_REPS=3 V8_ARM="$arm" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export NVSHMEM_HOME="$venv/lib/python3.12/site-packages/nvidia/nvshmem"
  export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export V8_ROPE_FACTOR="${V8_ROPE_FACTOR:-0}"
  if [[ "$V8_ROPE_FACTOR" != 0 ]]; then suffix=_rope4; fi
  export V8_OUT="/root/data/stowaway-nvidia/results/tp8_20261111/vllm_bwmatch_${arm}${suffix}_${repeat}.json"
  local log="$raw/vllm_bwmatch_${arm}${suffix}_${repeat}.log"
  if [[ -e "$log" || -e "$V8_OUT" ]]; then return; fi
  printf '%s\tllama70\t16384\t%s\t%s\tstart\t-\n' "$(date -u +%FT%TZ)" "$arm" "$repeat" >> "$status"
  timeout "$((remaining < 1800 ? remaining : 1800))" python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/vllm_tp8_bwmatch.py > "$log" 2>&1
  rc=$?
  printf '%s\tllama70\t16384\t%s\t%s\tend\t%s\n' "$(date -u +%FT%TZ)" "$arm" "$repeat" "$rc" >> "$status"
  /usr/bin/python3 experiments/report_bwmatch_tp8.py >> "$raw/bwmatch_report.log" 2>&1 || true
  export PATH=/root/data/stowaway-nvidia/envs/megatron-cu126/bin:$PATH
  unset NVSHMEM_HOME
}
run_predictor_probes() {
  local spec model m k n batch mode remaining rc log dest
  local envbin=/root/data/stowaway-nvidia/envs/megatron-cu126/bin
  export PATH="$envbin:$PATH" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export NVSHMEM_HOME="$($envbin/python -c 'import os,site;print(next(os.path.join(p,"nvidia","nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p,"nvidia","nvshmem","lib","libnvshmem_host.so.3"))))')"
  export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NVSHMEM_SYMMETRIC_SIZE=4G TP8_REPEATS=6 TP8_CHECKS=4
  while read -r model m k n; do
    for batch in 1 16; do
      if [[ -r "$deadline_file" ]]; then deadline=$(<"$deadline_file"); fi
      remaining=$((deadline - $(date +%s)))
      if (( deadline > 0 && remaining < 120 )); then unset NVSHMEM_HOME; return; fi
      mode=b${batch}
      log="$raw/bwmatch_probe_${model}_m${m}_k${k}_n${n}_${mode}.log"
      dest="/root/data/stowaway-nvidia/results/tp8_20261111/bwmatch_probe_${model}_m${m}_k${k}_n${n}_${mode}.json"
      if [[ -e "$log" || -e "$dest" ]]; then continue; fi
      export TP8_M="$m" TP8_K="$k" TP8_N="$n" TP8_BATCH="$batch" TP8_OUT="$dest"
      printf '%s\t%s\t%s\tprobe_%s\t%s\tstart\t-\n' "$(date -u +%FT%TZ)" "$model" "$m" "$mode" "$k-$n" >> "$status"
      timeout "$((remaining < 300 ? remaining : 300))" python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/tp8_operator.py > "$log" 2>&1
      rc=$?
      printf '%s\t%s\t%s\tprobe_%s\t%s\tend\t%s\n' "$(date -u +%FT%TZ)" "$model" "$m" "$mode" "$k-$n" "$rc" >> "$status"
      python planner/evaluate_tp8_holdout.py >> "$raw/bwmatch_predictor_report.log" 2>&1 || true
    done
  done < <(python -c 'import json; p="/root/Stotaway/planner/data/cells.json"; x=json.load(open(p)); seen=set(); [(seen.add((v["model"],v["M"],v["K"],v["N"])),print(v["model"],v["M"],v["K"],v["N"])) for v in x if (v["model"],v["M"],v["K"],v["N"]) not in seen]')
  unset NVSHMEM_HOME
}
# Interleave arms within each configuration; avoid concurrent GPU workloads.
# The 27b/8 stock a1 is already running before this queue is started.
run 27b 8 row a1 18
run 70b 8 stock a1 18
run 70b 8 row a1 18
export V8_ROPE_FACTOR=4
run_vllm stock a1
run_vllm stock_sp a1
run_vllm row a1
run_vllm all a1
run_vllm stock a2
run_vllm stock_sp a2
run 70b 8 all a1 18
run 70b 8 stock a2 18
run 70b 8 row a2 18
run 70b 8 all a2 18
run 70b 4 stock a1 14
run 70b 4 row a1 14
run 70b 4 all a1 14
run 70b 4 stock a2 14
run 70b 4 row a2 14
run 70b 4 all a2 14
run 70b 2 stock a1 14
run 70b 2 row a1 14
run 70b 2 all a1 14
run 70b 2 stock a2 14
run 70b 2 row a2 14
run 70b 2 all a2 14
run 70b 1 stock a1 14
run 70b 1 row a1 14
run 70b 1 stock a2 14
run 70b 1 row a2 14
run 27b 4 stock a1 14
run 27b 4 row a1 14
run 27b 4 all a1 14
run 27b 4 stock a2 14
run 27b 4 row a2 14
run 27b 4 all a2 14
run 27b 8 stock a2 18
run_tp8_dgrad_check() {
  local envbin=/root/data/stowaway-nvidia/envs/megatron-cu126/bin rc
  local log="$raw/bwmatch_tp8_dgrad_check.log"
  local dest=/root/data/stowaway-nvidia/results/tp8_20261111/bwmatch_tp8_dgrad_check.json
  if [[ -e "$log" || -e "$dest" ]]; then return; fi
  export PATH="$envbin:$PATH" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export NVSHMEM_HOME="$($envbin/python -c 'import os,site;print(next(os.path.join(p,"nvidia","nvshmem") for p in site.getsitepackages() if os.path.exists(os.path.join(p,"nvidia","nvshmem","lib","libnvshmem_host.so.3"))))')"
  export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NVSHMEM_SYMMETRIC_SIZE=2G CHECK_OUT="$dest"
  printf '%s\toperator\t1024\tdgrad\ta1\tstart\t-\n' "$(date -u +%FT%TZ)" >> "$status"
  timeout 300 python -m torch.distributed.run --standalone --nproc-per-node=8 experiments/tp8_dgrad_check.py > "$log" 2>&1
  rc=$?
  printf '%s\toperator\t1024\tdgrad\ta1\tend\t%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$status"
  unset NVSHMEM_HOME
}
run_tp8_dgrad_check
run 27b 8 row a2 18
# Prioritize the user's inference repeats and TP8 predictor calibration.
export V8_ROPE_FACTOR=4
run_vllm row a2
run_vllm all a2
run_predictor_probes
run 27b 8 all a1 18
run 27b 8 row_qkv a1 18
run 27b 8 all a2 18
run 27b 8 row_qkv a2 18
run 27b 8 null a1 18
run 70b 8 null a1 18
printf '%s\t-\t-\t-\t-\tqueue_end\t0\n' "$(date -u +%FT%TZ)" >> "$status"
