#!/usr/bin/env bash
# B1 — Full autotune-tg sweep across all 22 TPC-H queries.
# Produces a per-trial CSV (compatible with scripts/significance.py) for
# {baseline, autotune_tg} on every query, so a single Mann-Whitney run
# can populate a thesis-ready table.
#
# Override SFS / QUERIES via env. Default: SF1, all 22 queries.
set -u
cd "$(dirname "$0")/.."

BIN=./build/bin/GPUDBCodegen
[[ -x $BIN ]] || { echo "ERROR: $BIN not built"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=build/exp_autotune_${TS}
mkdir -p "$OUT"
TRIALS=$OUT/trials.csv
AUTO=$OUT/autotune_picks.csv
echo "Output: $OUT"

WARMUP=5
REPEAT=30

QUERIES_DEFAULT=(q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22)
read -r -a QUERIES <<< "${QUERIES:-${QUERIES_DEFAULT[*]}}"
SFS=(${SFS:-sf1})

echo "config,sf,query,trial,gpu_ms,compile_ms,e2e_ms" > "$TRIALS"
echo "sf,query,tg,p10_ms,p50_ms,p90_ms" > "$AUTO"

run_one() {  # $1=config_tag $2=sf $3=q $4...=extra flags
  local cfg=$1 sf=$2 q=$3; shift 3
  local raw
  raw=$("$BIN" --csv --warmup $WARMUP --repeat $REPEAT "$@" $sf $q 2>/dev/null) || true
  local trials autotune
  trials=$(echo "$raw" | grep '^TRIAL_CSV' || true)
  autotune=$(echo "$raw" | grep '^AUTOTUNE_CSV' || true)
  if [[ -z $trials ]]; then
    printf "  %-12s %-4s %-4s FAIL\n" "$cfg" "$sf" "$q"
    return
  fi
  echo "$trials" | awk -F, -v cfg="$cfg" \
    '{ printf "%s,%s,%s,%s,%s,%s,%s\n", cfg, $3, $2, $4, $5, $6, $7 }' >> "$TRIALS"
  if [[ -n $autotune ]]; then
    echo "$autotune" | awk -F, '{ printf "%s,%s,%s,%s,%s,%s\n", $2, $3, $4, $5, $6, $7 }' >> "$AUTO"
  fi
  printf "  %-12s %-4s %-4s ok\n" "$cfg" "$sf" "$q"
}

for sf in "${SFS[@]}"; do
  for q in "${QUERIES[@]}"; do
    run_one baseline             "$sf" "$q"
    run_one autotune_tg          "$sf" "$q" --autotune-tg
    run_one autotune_tg_perphase "$sf" "$q" --autotune-tg-per-phase
  done
done

echo "Done. Trials: $(($(wc -l < "$TRIALS") - 1))   Autotune picks: $(($(wc -l < "$AUTO") - 1))"
echo "Next: python3 scripts/significance.py $TRIALS"
