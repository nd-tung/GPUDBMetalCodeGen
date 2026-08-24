#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
GROUP_CARDINALITY=${GROUP_CARDINALITY:-256}
SMOKE_DIR=${SMOKE_DIR:-/tmp/apple-energy-run09-smoke-$$}

typeset -A checksums
mkdir -p "$SMOKE_DIR"
trap 'rm -rf "$SMOKE_DIR"' EXIT

run_case() {
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

  local checksum
  checksum=$(sed -n \
    's/.*"result_checksum": \([0-9][0-9]*\).*/\1/p' \
    "$output")
  if [[ -z "$checksum" ]]; then
    print -u2 -- "missing checksum in $output"
    return 1
  fi
  checksums[$label]=$checksum
  print -- "OK $label checksum=$checksum"
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

for operator in \
  filter-project \
  hash-build \
  hash-probe-count \
  hash-probe-materialize; do
  run_case "cpu-$operator" "$operator" cpu
  run_case "gpu-$operator" "$operator" gpu
  require_equal "cpu-$operator" "gpu-$operator"
done

run_case cpu-q6-fused q6-revenue cpu
run_case gpu-q6-fused q6-revenue gpu
run_case cpu-q6-unfused q6-revenue-unfused cpu
run_case gpu-q6-unfused q6-revenue-unfused gpu
require_equal \
  cpu-q6-fused \
  gpu-q6-fused \
  cpu-q6-unfused \
  gpu-q6-unfused

run_case cpu-aggregate aggregate-stats cpu
run_case gpu-aggregate-tree aggregate-stats gpu \
  --gpu-aggregate-reduction threadgroup-tree
run_case gpu-aggregate-simd aggregate-stats gpu \
  --gpu-aggregate-reduction simdgroup
require_equal \
  cpu-aggregate \
  gpu-aggregate-tree \
  gpu-aggregate-simd

run_case cpu-groupby groupby-part-count cpu
run_case gpu-groupby-global groupby-part-count gpu \
  --gpu-groupby-strategy global-atomic
run_case gpu-groupby-bounded groupby-part-count gpu \
  --gpu-groupby-strategy bounded-threadgroup
require_equal \
  cpu-groupby \
  gpu-groupby-global \
  gpu-groupby-bounded

print -- "ALL RUN09 SF10 CPU/GPU CHECKS PASSED"
