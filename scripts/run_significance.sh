#!/usr/bin/env bash
# C4 — Statistical significance sweep.
# For each (sf, query, config) we run --repeat 30 and capture every per-trial
# GPU time via TRIAL_CSV (emitted by the binary under --csv && repeat>1).
# The output is a single tidy CSV that downstream R/Python code can feed
# into Mann-Whitney U tests pairwise across configs (e.g. baseline vs B2,
# baseline vs --autotune-tg).
set -u
cd "$(dirname "$0")/.."

BIN=./build/bin/GPUDBCodegen
[[ -x $BIN ]] || { echo "ERROR: $BIN not built"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=build/exp_significance_${TS}
mkdir -p "$OUT"
CSV=$OUT/trials.csv
echo "Output: $CSV"

WARMUP=5      # Per-config warmup BEFORE the 30 timed trials.
REPEAT=30     # N=30 satisfies the Mann-Whitney U asymptotic-normal regime.

# TRIAL_CSV format:  TRIAL_CSV,query,sf,trial_idx,gpu_ms,compile_ms,e2e_ms
echo "config,sf,query,trial,gpu_ms,compile_ms,e2e_ms" > "$CSV"

# Configs to compare. Each config row is "<tag>|<extra-cli-flags>".
# Use whitespace-free flag bundles; tag becomes a CSV cell.
CONFIGS=(
  "baseline|"
  "scalar_atomic|--scalar-atomic"
  "autotune_tg|--autotune-tg"
)

# Default sample. Override via env QUERIES="q1 q6 q14".
QUERIES_DEFAULT=(q1 q3 q6 q9 q14 q17)
read -r -a QUERIES <<< "${QUERIES:-${QUERIES_DEFAULT[*]}}"
SFS=(${SFS:-sf1})

run_one() {  # $1=config_tag  $2=sf  $3=q  $4...=extra flags
  local cfg=$1 sf=$2 q=$3; shift 3
  local raw
  raw=$("$BIN" --csv --warmup $WARMUP --repeat $REPEAT "$@" $sf $q 2>/dev/null \
        | grep '^TRIAL_CSV') || true
  if [[ -z $raw ]]; then
    printf "  %-14s %-4s %-4s FAIL\n" "$cfg" "$sf" "$q"
    return
  fi
  # TRIAL_CSV,query,sf,trial,gpu,compile,e2e -> config,sf,query,trial,gpu,compile,e2e
  echo "$raw" | awk -F, -v cfg="$cfg" '
    { printf "%s,%s,%s,%s,%s,%s,%s\n", cfg, $3, $2, $4, $5, $6, $7 }
  ' >> "$CSV"
  printf "  %-14s %-4s %-4s ok (%d trials)\n" "$cfg" "$sf" "$q" \
    "$(echo "$raw" | wc -l | tr -d ' ')"
}

for sf in "${SFS[@]}"; do
  for q in "${QUERIES[@]}"; do
    for entry in "${CONFIGS[@]}"; do
      cfg=${entry%%|*}
      flags=${entry#*|}
      # shellcheck disable=SC2086
      run_one "$cfg" "$sf" "$q" $flags
    done
  done
done

echo "Done. Rows: $(($(wc -l < "$CSV") - 1))"
echo "Next: python3 scripts/significance.py $CSV"
