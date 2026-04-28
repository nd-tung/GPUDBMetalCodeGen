#!/usr/bin/env bash
# Full experiment sweep — produces CSV files in build/exp_<timestamp>/.
# Each CSV row is one TIMING_CSV emission (median over --repeat trials).
set -u
cd "$(dirname "$0")/.."

BIN=./build/bin/GPUDBCodegen
[[ -x $BIN ]] || { echo "ERROR: $BIN not built"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=build/exp_${TS}
mkdir -p "$OUT"
echo "Output dir: $OUT"

WARM=3
REP=5

# TIMING_CSV columns produced by the binary (23 fields; last 4 added by C1).
HEADER='sf,query,analyze_ms,plan_ms,codegen_ms,compile_ms,pso_ms,dataload_ms,bufalloc_ms,gpu_ms,cpu_ms,compile_overhead_ms,cpu_total_ms,e2e_ms,load_source,load_bytes,load_mibps,ingest_ms,query_compute_ms,gpu_trials_n,gpu_p10_ms,gpu_p90_ms,gpu_mad_ms'

run_one() {  # $1=tag $2=sf $3=qn $4..=extra flags
  local tag=$1 sf=$2 q=$3; shift 3
  local out=$OUT/${tag}.csv
  if [[ ! -f $out ]]; then
    echo "$HEADER" > "$out"
  fi
  local line
  line=$($BIN --csv --warmup $WARM --repeat $REP "$@" $sf $q 2>/dev/null \
         | grep '^TIMING_CSV' | tail -1 | sed 's/^TIMING_CSV,//')
  if [[ -n $line ]]; then
    echo "$line" >> "$out"
    printf "  %-12s %-4s %-6s ok\n" "$tag" "$sf" "$q"
  else
    printf "  %-12s %-4s %-6s FAIL\n" "$tag" "$sf" "$q"
  fi
}

QUERIES_TPCH=(q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22)
QUERIES_MB=(mb1 mb2 mb3 mb4 mb5 mb6 mb7)
SAMPLE=(q1 q6 q14 q17)

# 1. Baseline TPCH SF1 + SF10
echo "[1/5] Baseline TPCH SF1+SF10"
for sf in sf1 sf10; do
  for q in "${QUERIES_TPCH[@]}"; do run_one baseline $sf $q; done
done

# 2. Microbenchmarks SF1 + SF10
echo "[2/5] Microbenchmarks SF1+SF10"
for sf in sf1 sf10; do
  for q in "${QUERIES_MB[@]}"; do run_one microbench $sf $q; done
done

# 3. Fastmath sensitivity (SF1)
echo "[3/5] Fastmath OFF (SF1)"
for q in "${QUERIES_TPCH[@]}"; do run_one nofastmath sf1 $q --no-fastmath; done

# 4. Threadgroup-size sweep (SF1, sample queries)
echo "[4/5] Threadgroup sweep"
for tg in 32 64 128 256 512 1024; do
  for q in "${SAMPLE[@]}"; do
    out=$OUT/tg_sweep.csv
    if [[ ! -f $out ]]; then
      echo "tg,${HEADER}" > "$out"
    fi
    line=$($BIN --csv --warmup $WARM --repeat $REP --threadgroup-size $tg sf1 $q 2>/dev/null \
           | grep '^TIMING_CSV' | tail -1 | sed 's/^TIMING_CSV,//')
    if [[ -n $line ]]; then
      echo "$tg,$line" >> "$out"
      printf "  tg=%-4d %-6s ok\n" $tg $q
    else
      printf "  tg=%-4d %-6s FAIL\n" $tg $q
    fi
  done
done

# 5. Pipeline-cache cost: emit per-trial (TRIAL_CSV) so we can see warmup curve
echo "[5/5] Pipeline-cache cost (per-trial)"
out=$OUT/no_pipeline_cache_trials.csv
echo "query,sf,trial,gpu_ms,compile_ms,e2e_trial_ms" > "$out"
for q in q1 q6 q14; do
  $BIN --csv --warmup 0 --repeat 10 --no-pipeline-cache sf1 $q 2>/dev/null \
    | grep '^TRIAL_CSV' | sed 's/^TRIAL_CSV,//' >> "$out"
  printf "  no-cache %s ok\n" $q
done

echo "Done. CSVs in $OUT"
ls -la "$OUT"
