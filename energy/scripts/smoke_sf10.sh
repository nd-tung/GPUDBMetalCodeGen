#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
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

for operator in "${operators[@]}"; do
  for backend in cpu gpu; do
    output="/tmp/apple-energy-smoke-$operator-$backend.json"
    "$ROOT/build/joule-tpch-benchmark" \
      --data "$DATA" \
      --operator "$operator" \
      --backend "$backend" \
      --group-cardinality "$GROUP_CARDINALITY" \
      --duration-ms 0 \
      --warmup-iterations 0 \
      >"$output"
    checksum=$(
      sed -n 's/.*"result_checksum": \([0-9]*\).*/\1/p' "$output"
    )
    print -- "OK $operator $backend checksum=$checksum"
  done
done
