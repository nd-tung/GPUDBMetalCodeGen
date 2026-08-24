#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
RUN_ID=${RUN_ID:-tpch-sf10-primitives-run08}
RESULT_DIR="$ROOT/results/$RUN_ID"
RAW_DIR="$ROOT/results/raw"
LOG="$RESULT_DIR/runner.log"
GROUP_CARDINALITY=${GROUP_CARDINALITY:-1048576}

operators=(
  scan-copy
  filter-materialize
  q1-groupby
  q14-join
  orders-topk
  aggregate-stats
  groupby-part-count
  q6-revenue-unfused
)

mkdir -p "$RESULT_DIR" "$RAW_DIR"
if [[ -e "$RESULT_DIR/COMPLETE" ]]; then
  print -u2 -- "$RUN_ID is already complete"
  exit 1
fi

complete=0
finish() {
  local exit_code=$?
  if (( complete )); then
    print -- "$(date -u +%FT%TZ) COMPLETE $RUN_ID" | tee -a "$LOG"
    touch "$RESULT_DIR/COMPLETE"
    rm -f "$RESULT_DIR/FAILED"
  else
    print -- "$(date -u +%FT%TZ) FAILED status=$exit_code $RUN_ID" | tee -a "$LOG"
    touch "$RESULT_DIR/FAILED"
  fi
}
trap finish EXIT

if ! sudo -n true; then
  print -u2 -- "sudo ticket is not cached; run sudo -v before starting"
  exit 1
fi
if pgrep -x joule-measure >/dev/null || pgrep -x powermetrics >/dev/null; then
  print -u2 -- "another measurement process is already active"
  exit 1
fi

print -- "$(date -u +%FT%TZ) START $RUN_ID" | tee -a "$LOG"
print -- "operators=${(j:,:)operators} group_cardinality=$GROUP_CARDINALITY" |
  tee -a "$LOG"

for trial in 1 2 3; do
  if (( trial == 2 )); then
    backends=(cpu gpu)
  else
    backends=(gpu cpu)
  fi
  for operator in "${operators[@]}"; do
    for backend in "${backends[@]}"; do
      phase="trial-${trial}-${operator}-${backend}"
      output="$RESULT_DIR/$phase.json"
      raw="$RAW_DIR/$RUN_ID-$phase.plist"
      if [[ -s "$output" ]]; then
        print -- "$(date -u +%FT%TZ) SKIP $phase" | tee -a "$LOG"
        continue
      fi
      rm -f "$raw"
      print -- "$(date -u +%FT%TZ) START $phase" | tee -a "$LOG"
      caffeinate -dimsu "$ROOT/build/joule-measure" \
        --cooperative \
        --sample-rate-ms 1000 \
        --baseline-ms 10000 \
        --raw "$raw" \
        --output "$output" \
        -- "$ROOT/build/joule-tpch-benchmark" \
             --data "$DATA" \
             --operator "$operator" \
             --backend "$backend" \
             --group-cardinality "$GROUP_CARDINALITY" \
             --duration-ms 600000 \
             --warmup-iterations 5 \
        >>"$LOG" 2>&1
      if [[ ! -s "$output" ]]; then
        print -u2 -- "missing result for $phase"
        exit 1
      fi
      if pgrep -x powermetrics >/dev/null; then
        print -u2 -- "powermetrics remained active after $phase"
        exit 1
      fi
      print -- "$(date -u +%FT%TZ) DONE $phase" | tee -a "$LOG"
    done
  done
done

complete=1
