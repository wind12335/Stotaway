#!/bin/bash
# Operator-level sample matrix under the b2b16 protocol (bench/oprs_matrix.py): the serial path and the
# overlap candidates (DEO / Async-TP / Triton-distributed) measured in ONE process per batch.
#   OUT=<dir> bash bench/run_oprs_matrix.sh --first-batch         7 models x 3 ops x M=32768 (21 shapes)
#   OUT=<dir> bash bench/run_oprs_matrix.sh --table               all 99 operator-table points
#   OUT=<dir> bash bench/run_oprs_matrix.sh --models a,b --ms 8192
#   OUT=<dir> bash bench/run_oprs_matrix.sh --sites bw27b_train   call-site shapes of an end-to-end workload
#   OUT=<dir> bash bench/run_oprs_matrix.sh --plan plans/x.json   plan dispatch check (handoff section 6.4)
# Environment: NP      cards (default 8); export it before the call (env.sh defaults to 8)
#              TAG     file name samples_<TAG>.jsonl (default tp<NP>_<DTYPE>)
#              HEAP    symmetric-heap bytes (default 12 GB: largest table points)
#              METHODS default deo,async_tp,triton_dist      DTYPE default bfloat16
#              PORT    torch master port (default 29991)      TIMEOUT seconds per batch (default 7200)
#              RESUME=1  continue a crashed batch in the same directory (finished points are skipped)
#              OPRS_MATRIX_TRACE=1  print the time of every call of every group (diagnostic)
# A batch directory is never overwritten: a new batch gets a new directory.
set -u
HERE=$(dirname "$(readlink -f "$0")")
export SW_NP=${SW_NP:-${NP:-8}}                 # before env/env.sh
source "$HERE/../env/env.sh"
export OPRS_TRITON_DIST_ROOT=${OPRS_TRITON_DIST_ROOT:-/root/private_data/tri-dist-hygon}
OUT=${OUT:?set OUT=<new directory>, e.g. results/bw1000_oprs_<date>/raw/operators/<batch>}
DTYPE=${DTYPE:-bfloat16}
FILE=$OUT/samples_${TAG:-tp${SW_NP}_${DTYPE}}.jsonl
export DUSHMEM_SYMMETRIC_SIZE=${HEAP:-12884901888}
ARGS=(--methods "${METHODS:-deo,async_tp,triton_dist}" --dtype "$DTYPE" --port "${PORT:-29991}")
[ -n "${TOPOLOGY:-}" ] && ARGS+=(--topology "$TOPOLOGY")
if [ "${RESUME:-0}" = "1" ]; then
    [ -d "$OUT" ] || { echo "RESUME=1 but $OUT does not exist"; exit 2; }
    [ -e "$FILE" ] && ARGS+=(--resume)
else
    mkdir -p "$(dirname "$OUT")"
    mkdir "$OUT" 2>/dev/null || { echo "refusing to overwrite $OUT (history is kept: choose a new directory, or RESUME=1)"; exit 2; }
fi
LOG=$OUT/log_$(date +%Y%m%d_%H%M%S).log
echo "[run_oprs_matrix] np=$SW_NP dtype=$DTYPE methods=${METHODS:-deo,async_tp,triton_dist} heap=$DUSHMEM_SYMMETRIC_SIZE file=$FILE log=$LOG"
timeout "${TIMEOUT:-7200}" $SW_MPIRUN -x DUSHMEM_SYMMETRIC_SIZE -x OPRS_TRITON_DIST_ROOT -x OPRS_MATRIX_TRACE \
    python3 "$HERE/oprs_matrix.py" --out "$FILE" "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[run_oprs_matrix] rc=$RC" | tee -a "$LOG"
[ -e "$FILE" ] && python3 -I - "$FILE" <<'EOF' | tee -a "$LOG"
import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
count = collections.Counter((r.get('candidate'), r.get('status')) for r in rows)
print('[run_oprs_matrix] rows:', len(rows), dict(sorted(count.items())))
for r in rows:
    if r.get('status') != 'ok':
        print(f"  {r.get('status')}: {r.get('model')}/{r.get('table_op') or r.get('site')}/M{r['M']} "
              f"[{r.get('candidate')}] {r.get('support', {}).get('evidence') or r.get('error')}")
EOF
exit "$RC"
