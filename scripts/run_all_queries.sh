#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all_queries.sh
#
# Runs all 22 TPC-H queries through the codegen pipeline at one or more scale
# factors and writes a CSV report including chip/OS/GPU/memory information.
#
# Usage:
#   scripts/run_all_queries.sh [sf1|sf10|sf100 ...] [-o results.csv] [-q "q1 q2"]
#
# Defaults: SF=sf1, all 22 queries, output = build/results_<timestamp>.csv
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- CLI parsing ------------------------------------------------------------
SCALE_FACTORS=()
OUTPUT=""
QUERIES_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        sf1|sf10|sf50|sf100) SCALE_FACTORS+=("$1"); shift ;;
        -o|--output)    OUTPUT="$2"; shift 2 ;;
        -q|--queries)   QUERIES_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ ${#SCALE_FACTORS[@]} -eq 0 ]] && SCALE_FACTORS=("sf1")

# ---- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BIN="build/bin/GPUDBCodegen"
if [[ ! -x "$BIN" ]]; then
    echo "Building project..."
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)" >/dev/null
fi

if [[ -n "$QUERIES_OVERRIDE" ]]; then
    read -r -a QUERIES <<< "$QUERIES_OVERRIDE"
else
    QUERIES=(q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22)
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p build
OUTPUT="${OUTPUT:-build/results_${TS}.csv}"
LOG_DIR="build/logs_${TS}"
mkdir -p "$LOG_DIR"

# ---- System info (collected once, prepended as comment lines) ---------------
HOST="$(hostname)"
CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
OS="macOS $(sw_vers -productVersion 2>/dev/null || echo ?)"
RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
RAM_GIB="$(awk -v b="$RAM_BYTES" 'BEGIN{printf "%.1f", b/1073741824}')"
KERNEL="$(uname -sr)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo none)"

{
    echo "# host=$HOST"
    echo "# cpu=$CPU"
    echo "# os=$OS"
    echo "# kernel=$KERNEL"
    echo "# ram_bytes=$RAM_BYTES"
    echo "# ram_gib=$RAM_GIB"
    echo "# git_commit=$GIT_COMMIT"
    echo "# timestamp=$TS"
    # NOTE: gpu/gpu_budget extracted from first SYSINFO_CSV line below
    echo "scale_factor,query,status,analyze_ms,plan_ms,codegen_ms,compile_ms,pso_ms,dataload_ms,bufalloc_ms,gpu_ms,post_ms,cpu_codegen_ms,cpu_total_ms,end2end_ms,gpu_name,gpu_budget_bytes"
} > "$OUTPUT"

GPU_NAME=""
GPU_BUDGET=""

run_one() {
    local sf="$1" q="$2"
    local log="$LOG_DIR/${sf}_${q}.log"
    echo "  -> $sf $q"

    if ! "$BIN" "$sf" "$q" > "$log" 2>&1; then
        echo "${sf},${q},FAIL,,,,,,,,,,,,,${GPU_NAME},${GPU_BUDGET}" >> "$OUTPUT"
        echo "     FAILED (see $log)"
        return
    fi

    # Capture GPU info from first SYSINFO_CSV line we see.
    if [[ -z "$GPU_NAME" ]]; then
        local sysline
        sysline="$(grep -m1 '^SYSINFO_CSV,' "$log" || true)"
        if [[ -n "$sysline" ]]; then
            GPU_NAME="$(echo "$sysline"       | awk -F',' '{print $4}')"
            GPU_BUDGET="$(echo "$sysline"     | awk -F',' '{print $6}')"
        fi
    fi

    local timing
    timing="$(grep -m1 '^TIMING_CSV,' "$log" || true)"
    if [[ -z "$timing" ]]; then
        echo "${sf},${q},NO_TIMING,,,,,,,,,,,,,${GPU_NAME},${GPU_BUDGET}" >> "$OUTPUT"
        return
    fi

    # TIMING_CSV,sf,query,analyze,plan,codegen,compile,pso,dataload,bufalloc,gpu,post,cpu_codegen,cpu_total,end2end
    local body="${timing#TIMING_CSV,}"
    # Append status=OK, and gpu_name/gpu_budget to keep each row fully self-describing.
    echo "${body/,/,},OK,${body#*,*,},${GPU_NAME},${GPU_BUDGET}" >/dev/null 2>&1 || true
    # Cleaner: reconstruct with awk.
    awk -v gpu="$GPU_NAME" -v bud="$GPU_BUDGET" -F',' '
    {
        # $1=sf, $2=query, $3..$15 = timings
        printf "%s,%s,OK", $1, $2;
        for (i = 3; i <= 15; i++) printf ",%s", $i;
        printf ",%s,%s\n", gpu, bud;
    }' <<< "$body" >> "$OUTPUT"
}

echo "=============================================="
echo "  Host:   $HOST"
echo "  CPU:    $CPU"
echo "  RAM:    $RAM_GIB GiB"
echo "  Scales: ${SCALE_FACTORS[*]}"
echo "  Output: $OUTPUT"
echo "  Logs:   $LOG_DIR/"
echo "=============================================="

for sf in "${SCALE_FACTORS[@]}"; do
    echo ">>> Scale factor: $sf"
    for q in "${QUERIES[@]}"; do
        run_one "$sf" "$q"
    done
done

echo ""
echo "Done. Wrote: $OUTPUT"
