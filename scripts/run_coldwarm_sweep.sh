#!/usr/bin/env bash
# Cold / Warm / Hot bandwidth regime experiment.
#
# Cold : OS page cache dropped before each measurement
#        (file must be paged in from SSD on every read)
# Warm : OS page cache hot, but the GPU process is fresh
#        (RAM hit, SLC/L2 miss)
# Hot  : in-process warmup amortizes everything (current default)
#
# Requires passwordless `sudo purge`, e.g.:
#   echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/purge" | sudo tee /etc/sudoers.d/purge
# (The script will prompt once otherwise.)
set -u
cd "$(dirname "$0")/.."

BIN=./build/bin/GPUDBCodegen
[[ -x $BIN ]] || { echo "ERROR: $BIN not built"; exit 1; }

# Confirm sudo purge works without prompting
if ! sudo -n /usr/sbin/purge 2>/dev/null; then
    echo "Note: 'sudo purge' will prompt for password (or once at start)."
    sudo /usr/sbin/purge
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT=build/exp_coldwarm_${TS}
mkdir -p "$OUT"
OUTCSV=$OUT/coldwarm.csv
echo "regime,sf,query,trial,gpu_ms,e2e_ms,load_mibps,load_bytes" > "$OUTCSV"
echo "Output: $OUTCSV"

# Extract fields from a TIMING_CSV line:
#   col 10=gpu_ms, col 14=e2e_ms, col 17=load_mibps, col 16=load_bytes
extract() {
    local timing="$1"
    awk -F, -v r="$2" -v sf="$3" -v q="$4" -v t="$5" \
        '{print r","sf","q","t","$10","$14","$17","$16}' <<< "$timing"
}

run_regime() {  # $1=regime $2=sf $3=q $4=trials $5=purge_between(0|1) $6=warmup $7=repeat
    local regime=$1 sf=$2 q=$3 trials=$4 purge=$5 warm=$6 rep=$7
    for ((t=0; t<trials; t++)); do
        if [[ $purge == 1 ]]; then
            sudo -n /usr/sbin/purge 2>/dev/null || sudo /usr/sbin/purge
        fi
        local line
        line=$($BIN --csv --warmup $warm --repeat $rep $sf $q 2>/dev/null \
               | grep '^TIMING_CSV' | tail -1 | sed 's/^TIMING_CSV,//')
        if [[ -n $line ]]; then
            extract "$line" "$regime" "$sf" "$q" "$t" >> "$OUTCSV"
            local gpu e2e bw
            gpu=$(awk -F, '{print $10}' <<< "$line")
            e2e=$(awk -F, '{print $14}' <<< "$line")
            bw=$(awk -F, '{print $17}' <<< "$line")
            printf "  %-5s %s %-4s t%d  gpu=%6.2fms  e2e=%7.2fms  loadbw=%9.1f MiB/s\n" \
                "$regime" "$sf" "$q" "$t" "$gpu" "$e2e" "$bw"
        else
            printf "  %-5s %s %-4s t%d  FAIL\n" "$regime" "$sf" "$q" "$t"
        fi
    done
}

# Sweep across queries that exercise different working-set sizes:
#  Q6      : single-table scan over lineitem (~500 MiB at SF10)
#  Q1      : full lineitem scan + grouped agg
#  MB1     : pure scan of lineitem.l_extendedprice (~96 MiB at SF10)
#  MB6     : 12-column read (~1.1 GiB at SF10)
QUERIES=(mb1 mb6 q1 q6 q14)
SFS=(sf1 sf10)
TRIALS=5

for sf in "${SFS[@]}"; do
    for q in "${QUERIES[@]}"; do
        echo "--- $sf $q ---"
        # Cold: drop cache before every trial, no warmup, single repeat.
        run_regime cold $sf $q $TRIALS 1 0 1
        # Warm: cache hot from cold runs, no warmup, fresh process per trial.
        run_regime warm $sf $q $TRIALS 0 0 1
        # Hot: in-process warmup amortizes everything (current default).
        run_regime hot  $sf $q 1       0 3 5
    done
done

echo "Done. CSV: $OUTCSV"
echo
echo "Quick summary (median by regime):"
python3 - <<EOF
import csv, statistics
from collections import defaultdict
rows = list(csv.DictReader(open("$OUTCSV")))
by = defaultdict(list)
for r in rows:
    by[(r['regime'], r['sf'], r['query'])].append(r)
print(f"{'regime':<5}  {'sf':<4}  {'query':<5}  {'gpu_p50':>8}  {'e2e_p50':>9}  {'bw_GiB/s':>10}  {'bytes':>10}")
for k in sorted(by):
    rs = by[k]
    gpu = statistics.median(float(r['gpu_ms']) for r in rs)
    e2e = statistics.median(float(r['e2e_ms']) for r in rs)
    mibps = statistics.median(float(r['load_mibps']) for r in rs)
    nbytes = int(rs[0]['load_bytes'])
    print(f"{k[0]:<5}  {k[1]:<4}  {k[2]:<5}  {gpu:>8.2f}  {e2e:>9.2f}  {mibps/1024:>10.2f}  {nbytes:>10}")
EOF
