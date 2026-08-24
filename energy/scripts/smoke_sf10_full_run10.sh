#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
GROUP_CARDINALITY=${GROUP_CARDINALITY:-256}
SCAN_ROWS=${SCAN_ROWS:-60000000}
SMOKE_DIR=${SMOKE_DIR:-/tmp/apple-energy-full-run10-smoke-$$}

typeset -A checksums
mkdir -p "$SMOKE_DIR"
trap 'rm -rf "$SMOKE_DIR"' EXIT

read_checksum() {
  local output=$1
  local checksum
  checksum=$(sed -n \
    -e 's/.*"result_checksum": \([0-9][0-9]*\).*/\1/p' \
    -e 's/.*"checksum": \([-0-9][0-9]*\).*/\1/p' \
    "$output")
  if [[ -z "$checksum" ]]; then
    print -u2 -- "missing checksum in $output"
    return 1
  fi
  print -r -- "$checksum"
}

run_tpch_case() {
  local label=$1
  local operator=$2
  local backend=$3
  shift 3

  local output="$SMOKE_DIR/$label.json"
  "$ROOT/build/joule-tpch-benchmark" \
    --data "$DATA" \
    --operator "$operator" \
    --backend "$backend" \
    --group-cardinality "$GROUP_CARDINALITY" \
    --duration-ms 0 \
    --warmup-iterations 0 \
    "$@" \
    >"$output"
  checksums[$label]=$(read_checksum "$output")
  print -- "OK $label checksum=${checksums[$label]}"
}

run_scan_case() {
  local label=$1
  local backend=$2
  shift 2

  local output="$SMOKE_DIR/$label.json"
  "$ROOT/build/joule-benchmark" \
    --backend "$backend" \
    --rows "$SCAN_ROWS" \
    --input-pattern signed \
    --duration-ms 0 \
    --warmup-iterations 0 \
    --batch-size 1 \
    "$@" \
    >"$output"
  checksums[$label]=$(read_checksum "$output")
  print -- "OK $label checksum=${checksums[$label]}"
}

require_equal() {
  local expected_label=$1
  shift
  local label
  for label in "$@"; do
    if [[ "${checksums[$label]}" != "${checksums[$expected_label]}" ]]; then
      print -u2 -- \
        "checksum mismatch: $expected_label=${checksums[$expected_label]} $label=${checksums[$label]}"
      return 1
    fi
  done
}

ordinary_operators=(
  scan-copy
  filter-count
  filter-bitmap
  filter-materialize
  filter-project
  hash-build
  hash-probe-count
  hash-probe-materialize
  q1-groupby
  q14-join
  orders-topk
)

for operator in "${ordinary_operators[@]}"; do
  run_tpch_case "cpu-$operator" "$operator" cpu
  run_tpch_case "gpu-$operator" "$operator" gpu
  require_equal "cpu-$operator" "gpu-$operator"
done

run_tpch_case cpu-q6-fused q6-revenue cpu
run_tpch_case gpu-q6-fused q6-revenue gpu
run_tpch_case cpu-q6-unfused q6-revenue-unfused cpu
run_tpch_case gpu-q6-unfused q6-revenue-unfused gpu
require_equal \
  cpu-q6-fused \
  gpu-q6-fused \
  cpu-q6-unfused \
  gpu-q6-unfused

for operator in aggregate-sum aggregate-minmax aggregate-stats; do
  run_tpch_case "cpu-$operator" "$operator" cpu
  run_tpch_case "gpu-$operator-simd" "$operator" gpu \
    --gpu-aggregate-reduction simdgroup
  run_tpch_case "gpu-$operator-tree" "$operator" gpu \
    --gpu-aggregate-reduction threadgroup-tree
  require_equal \
    "cpu-$operator" \
    "gpu-$operator-simd" \
    "gpu-$operator-tree"
done

run_tpch_case cpu-groupby-global groupby-part-count cpu
run_tpch_case gpu-groupby-global groupby-part-count gpu \
  --gpu-groupby-strategy global-atomic
run_tpch_case gpu-groupby-bounded groupby-part-count gpu \
  --gpu-groupby-strategy bounded-threadgroup
require_equal \
  cpu-groupby-global \
  gpu-groupby-global \
  gpu-groupby-bounded

run_scan_case cpu-scan-scalar cpu --cpu-kernel scalar
run_scan_case cpu-scan-parallel cpu --cpu-kernel parallel
run_scan_case cpu-scan-simd cpu --cpu-kernel simd
run_scan_case gpu-scan-baseline gpu --gpu-kernel baseline
run_scan_case gpu-scan-multi-item gpu --gpu-kernel multi-item
run_scan_case gpu-scan-simdgroup gpu --gpu-kernel simdgroup
require_equal \
  cpu-scan-scalar \
  cpu-scan-parallel \
  cpu-scan-simd \
  gpu-scan-baseline \
  gpu-scan-multi-item \
  gpu-scan-simdgroup

print -- "ALL 44 FULL-RUN SF10/SYNTHETIC CONFIGURATIONS PASSED"
