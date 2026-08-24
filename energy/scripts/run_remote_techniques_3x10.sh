#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
RUN_ID=${RUN_ID:-tpch-sf10-techniques-run09}
RESULT_DIR="$ROOT/results/$RUN_ID"
RAW_DIR="$ROOT/results/raw"
LOG="$RESULT_DIR/runner.log"
DURATION_MS=${DURATION_MS:-600000}
BASELINE_MS=${BASELINE_MS:-10000}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-5}
GROUP_CARDINALITY=${GROUP_CARDINALITY:-256}

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
  print -u2 -- "sudo ticket is not cached; run sudo -v in this terminal first"
  exit 1
fi
if pgrep -x joule-measure >/dev/null || pgrep -x powermetrics >/dev/null; then
  print -u2 -- "another measurement process is already active"
  exit 1
fi

print -- "$(date -u +%FT%TZ) START $RUN_ID" | tee -a "$LOG"
print -- \
  "duration_ms=$DURATION_MS baseline_ms=$BASELINE_MS warmups=$WARMUP_ITERATIONS group_cardinality=$GROUP_CARDINALITY" |
  tee -a "$LOG"
print -- \
  "comparisons=fusion,intra-simd-reduction,bounded-threadgroup-aggregation primitives=filter-project,hash-build,hash-probe-count,hash-probe-materialize" |
  tee -a "$LOG"

run_case() {
  local trial=$1
  local label=$2
  local operator=$3
  local backend=$4
  shift 4

  local phase="trial-${trial}-${label}"
  local output="$RESULT_DIR/$phase.json"
  local raw="$RAW_DIR/$RUN_ID-$phase.plist"

  if [[ -s "$output" ]]; then
    print -- "$(date -u +%FT%TZ) SKIP $phase" | tee -a "$LOG"
    return
  fi

  rm -f "$raw" "$raw.gz"
  print -- "$(date -u +%FT%TZ) START $phase" | tee -a "$LOG"
  caffeinate -dimsu "$ROOT/build/joule-measure" \
    --cooperative \
    --sample-rate-ms 1000 \
    --baseline-ms "$BASELINE_MS" \
    --raw "$raw" \
    --output "$output" \
    -- "$ROOT/build/joule-tpch-benchmark" \
         --data "$DATA" \
         --operator "$operator" \
         --backend "$backend" \
         --group-cardinality "$GROUP_CARDINALITY" \
         --duration-ms "$DURATION_MS" \
         --warmup-iterations "$WARMUP_ITERATIONS" \
         "$@" \
    >>"$LOG" 2>&1

  if [[ ! -s "$output" ]]; then
    print -u2 -- "missing result for $phase"
    return 1
  fi
  if pgrep -x powermetrics >/dev/null; then
    print -u2 -- "powermetrics remained active after $phase"
    return 1
  fi
  if [[ -s "$raw" ]]; then
    gzip -f "$raw"
  fi
  print -- "$(date -u +%FT%TZ) DONE $phase" | tee -a "$LOG"
}

run_primitive_pair() {
  local trial=$1
  local operator=$2
  if (( trial == 2 )); then
    run_case "$trial" "cpu-$operator" "$operator" cpu
    run_case "$trial" "gpu-$operator" "$operator" gpu
  else
    run_case "$trial" "gpu-$operator" "$operator" gpu
    run_case "$trial" "cpu-$operator" "$operator" cpu
  fi
}

run_q6_variant() {
  local trial=$1
  local label=$2
  local operator=$3
  if (( trial == 2 )); then
    run_case "$trial" "cpu-q6-$label" "$operator" cpu
    run_case "$trial" "gpu-q6-$label" "$operator" gpu
  else
    run_case "$trial" "gpu-q6-$label" "$operator" gpu
    run_case "$trial" "cpu-q6-$label" "$operator" cpu
  fi
}

for trial in 1 2 3; do
  print -- "$(date -u +%FT%TZ) TRIAL $trial" | tee -a "$LOG"

  if (( trial == 2 )); then
    primitive_order=(
      hash-probe-materialize
      hash-probe-count
      hash-build
      filter-project
    )
  else
    primitive_order=(
      filter-project
      hash-build
      hash-probe-count
      hash-probe-materialize
    )
  fi
  for operator in "${primitive_order[@]}"; do
    run_primitive_pair "$trial" "$operator"
  done

  if (( trial == 2 )); then
    run_q6_variant "$trial" fused q6-revenue
    run_q6_variant "$trial" unfused q6-revenue-unfused
  else
    run_q6_variant "$trial" unfused q6-revenue-unfused
    run_q6_variant "$trial" fused q6-revenue
  fi

  if (( trial == 2 )); then
    run_case "$trial" gpu-aggregate-simd aggregate-stats gpu \
      --gpu-aggregate-reduction simdgroup
    run_case "$trial" cpu-aggregate aggregate-stats cpu
    run_case "$trial" gpu-aggregate-tree aggregate-stats gpu \
      --gpu-aggregate-reduction threadgroup-tree
  else
    run_case "$trial" gpu-aggregate-tree aggregate-stats gpu \
      --gpu-aggregate-reduction threadgroup-tree
    run_case "$trial" cpu-aggregate aggregate-stats cpu
    run_case "$trial" gpu-aggregate-simd aggregate-stats gpu \
      --gpu-aggregate-reduction simdgroup
  fi

  if (( trial == 2 )); then
    run_case "$trial" gpu-groupby-bounded groupby-part-count gpu \
      --gpu-groupby-strategy bounded-threadgroup
    run_case "$trial" cpu-groupby groupby-part-count cpu
    run_case "$trial" gpu-groupby-global groupby-part-count gpu \
      --gpu-groupby-strategy global-atomic
  else
    run_case "$trial" gpu-groupby-global groupby-part-count gpu \
      --gpu-groupby-strategy global-atomic
    run_case "$trial" cpu-groupby groupby-part-count cpu
    run_case "$trial" gpu-groupby-bounded groupby-part-count gpu \
      --gpu-groupby-strategy bounded-threadgroup
  fi
done

complete=1
