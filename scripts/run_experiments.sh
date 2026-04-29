#!/bin/bash
# =============================================================================
# SF100 Chunk Size & Double-Buffer Experiment
# =============================================================================
# Tests different chunk sizes with/without double buffering on representative
# TPC-H queries to find optimal performance settings.
#
# Experiment matrix:
#   Chunk sizes: 5M, 10M, 25M, 50M, 100M, 200M rows
#   Buffering:   double-buffer (2 slots) vs single-buffer (1 slot)
#   Queries:     Q1 (GPU-heavy), Q6 (scan), Q13 (orders), Q17 (complex)
#   Repetitions: 3 per config
# =============================================================================

set -e

BIN="./build/bin/GPUDBCodegen"
LOGDIR="/tmp/sf100_experiments"
RESULT_FILE="/tmp/sf100_experiment_results.csv"
QUERIES="q1 q6 q13 q17"
CHUNK_SIZES="5 10 25 50 100 200"
REPS=3

if [[ ! -x "$BIN" ]]; then
  echo "Building project..."
  make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

if ! "$BIN" --help 2>/dev/null | grep -q -- '--chunk'; then
  echo "ERROR: $BIN does not expose --chunk/--no-db streaming flags." >&2
  echo "This SF100 streaming experiment script is stale until chunked mode is wired into GPUDBCodegen." >&2
  exit 1
fi

mkdir -p "$LOGDIR"

# CSV header
echo "query,chunk_M,double_buffer,rep,chunks,parse_ms,gpu_ms" > "$RESULT_FILE"

echo "=========================================="
echo "SF100 Chunk & Double-Buffer Experiment"
echo "=========================================="
echo "Queries:     $QUERIES"
echo "Chunk sizes: $CHUNK_SIZES (M rows)"
echo "Reps:        $REPS per config"
echo "Results:     $RESULT_FILE"
echo "=========================================="

total_runs=0
for chunk in $CHUNK_SIZES; do
  for db in "yes" "no"; do
    for q in $QUERIES; do
      total_runs=$((total_runs + REPS))
    done
  done
done
echo "Total runs: $total_runs"
echo ""

run_idx=0
for chunk in $CHUNK_SIZES; do
  for db in "yes" "no"; do
    db_flag=""
    db_label="db"
    if [ "$db" = "no" ]; then
      db_flag="--no-db"
      db_label="no-db"
    fi

    for q in $QUERIES; do
      for rep in $(seq 1 $REPS); do
        run_idx=$((run_idx + 1))
        logfile="$LOGDIR/${q}_chunk${chunk}M_${db_label}_rep${rep}.log"

        echo "[$run_idx/$total_runs] $q chunk=${chunk}M $db_label rep=$rep"
        $BIN sf100 --chunk=${chunk}M $db_flag $q > "$logfile" 2>&1

        # Extract metrics from log: "SF100 streaming: N chunks, parse=Xms, GPU=Yms"
        line=$(grep "SF100 streaming:" "$logfile" 2>/dev/null || echo "")
        if [ -n "$line" ]; then
          chunks=$(echo "$line" | sed 's/.*: \([0-9]*\) chunks.*/\1/')
          parse=$(echo "$line" | sed 's/.*parse=\([0-9.]*\)ms.*/\1/')
          gpu=$(echo "$line" | sed 's/.*GPU=\([0-9.]*\)ms.*/\1/')
          echo "  -> chunks=$chunks parse=${parse}ms GPU=${gpu}ms"
          echo "$q,$chunk,$db,$rep,$chunks,$parse,$gpu" >> "$RESULT_FILE"
        else
          echo "  -> ERROR (no streaming output found)"
          echo "$q,$chunk,$db,$rep,ERR,ERR,ERR" >> "$RESULT_FILE"
        fi
      done
    done
  done
done

echo ""
echo "=========================================="
echo "Experiment complete. Results in: $RESULT_FILE"
echo "=========================================="

# Print summary table
echo ""
echo "=== Summary (median GPU ms per config) ==="
echo "query | chunk_M | db    | gpu_ms (runs)"

prev_key=""
for chunk in $CHUNK_SIZES; do
  for db in "yes" "no"; do
    db_label="db"
    [ "$db" = "no" ] && db_label="no-db"
    for q in $QUERIES; do
      # Collect GPU times for this config
      gpu_vals=$(grep "^$q,$chunk,$db," "$RESULT_FILE" | cut -d, -f7 | sort -n | tr '\n' ' ')
      median=$(grep "^$q,$chunk,$db," "$RESULT_FILE" | cut -d, -f7 | sort -n | awk 'NR==2{print}')
      [ -z "$median" ] && median=$(echo "$gpu_vals" | awk '{print $1}')
      printf "%-4s | %6sM | %-5s | %s (all: %s)\n" "$q" "$chunk" "$db_label" "$median" "$gpu_vals"
    done
  done
done

echo ""
echo "Full CSV: $RESULT_FILE"
echo "Per-run logs: $LOGDIR/"
