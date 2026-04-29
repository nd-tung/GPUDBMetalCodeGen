#!/usr/bin/env bash
# B2 ablation: SIMD+TG reduction (default) vs scalar-atomic (--scalar-atomic).
# Predicts: scalar-atomic should be slower on reduction-heavy queries.
# Targets: Q1 (multi-accumulator group-by), Q6 (single sum), Q14 (sum/cond),
#          MB1-MB7 (microbenches).
set -u
cd "$(dirname "$0")/.."

BIN=./build/bin/GPUDBCodegen
[[ -x $BIN ]] || { echo "ERROR: $BIN not built"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=build/exp_reduction_${TS}
mkdir -p "$OUT"
CSV=$OUT/reduction.csv
echo "strategy,sf,query,gpu_ms,e2e_ms,pso_ms,load_mibps" > "$CSV"

QUERIES=(mb1 mb2 mb3 mb4 mb5 mb6 mb7 q1 q6 q14)
SFS=(sf1 sf10)

run() {  # $1=strategy_label $2=extra_args $3=sf $4=q
    local label=$1 extra=$2 sf=$3 q=$4
    local line
    line=$($BIN $extra --csv --warmup 3 --repeat 5 $sf $q 2>/dev/null \
           | grep '^TIMING_CSV' | tail -1 | sed 's/^TIMING_CSV,//')
    if [[ -z $line ]]; then
        printf "  %-15s %s %-4s FAIL\n" "$label" "$sf" "$q"
        return
    fi
    local gpu pso e2e bw
    gpu=$(awk -F, '{print $10}' <<<"$line")
    pso=$(awk -F, '{print $7}' <<<"$line")
    e2e=$(awk -F, '{print $14}' <<<"$line")
    bw=$(awk -F, '{print $17}' <<<"$line")
    echo "$label,$sf,$q,$gpu,$e2e,$pso,$bw" >> "$CSV"
    printf "  %-15s %s %-4s  gpu=%6.2fms  e2e=%7.2fms  bw=%9.1f MiB/s\n" \
        "$label" "$sf" "$q" "$gpu" "$e2e" "$bw"
}

for sf in "${SFS[@]}"; do
    for q in "${QUERIES[@]}"; do
        echo "--- $sf $q ---"
        run "tgreduce"      ""                  $sf $q
        run "scalar-atomic" "--scalar-atomic"   $sf $q
    done
done

echo "Done. CSV: $CSV"
echo
echo "Summary (gpu_ms, scalar-atomic vs tgreduce):"
python3 - <<EOF
import csv
from collections import defaultdict
by = defaultdict(dict)
for r in csv.DictReader(open("$CSV")):
    by[(r['sf'], r['query'])][r['strategy']] = float(r['gpu_ms'])
print(f"{'sf':<4}  {'query':<5}  {'tgreduce':>9}  {'scalar':>9}  {'ratio':>7}  {'verdict'}")
for k in sorted(by):
    d = by[k]
    a = d.get('tgreduce', 0); b = d.get('scalar-atomic', 0)
    if a and b:
        ratio = b / a
        verdict = "scalar slower" if ratio > 1.05 else ("scalar faster" if ratio < 0.95 else "neutral")
        print(f"{k[0]:<4}  {k[1]:<5}  {a:>9.2f}  {b:>9.2f}  {ratio:>6.2f}x  {verdict}")
EOF
