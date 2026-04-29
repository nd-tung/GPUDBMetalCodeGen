#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all_queries.sh
#
# Runs all 22 TPC-H queries through the codegen pipeline at one or more scale
# factors and writes a CSV report including chip/OS/GPU/memory information.
#
# Usage:
#   scripts/run_all_queries.sh [sf1|sf10|sf100 ...] [-o results.csv] [-q "q1 q2"] [--check golden]
#
# Defaults: SF=sf1, all 22 queries, output = build/results_<timestamp>.csv
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- CLI parsing ------------------------------------------------------------
SCALE_FACTORS=()
OUTPUT=""
QUERIES_OVERRIDE=""
CHECK_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        sf1|sf10|sf50|sf100) SCALE_FACTORS+=("$1"); shift ;;
        -o|--output)    OUTPUT="$2"; shift 2 ;;
        -q|--queries)   QUERIES_OVERRIDE="$2"; shift 2 ;;
        --check)
            [[ $# -ge 2 ]] || { echo "Missing value for --check" >&2; exit 1; }
            CHECK_DIR="$2"; shift 2 ;;
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
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

CHECK_ARGS=()
if [[ -n "$CHECK_DIR" ]]; then
    if [[ ! -d "$CHECK_DIR" ]]; then
        echo "Check directory does not exist: $CHECK_DIR" >&2
        exit 1
    fi
    CHECK_ARGS=(--check "$CHECK_DIR")
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
    echo "# git_dirty_count=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "# binary=$REPO_ROOT/$BIN"
    echo "# check_dir=${CHECK_DIR:-none}"
    echo "# timestamp=$TS"
    # NOTE: gpu/gpu_budget extracted from first SYSINFO_CSV line below
    echo "scale_factor,query,status,analyze_ms,plan_ms,codegen_ms,compile_ms,pso_ms,dataload_ms,bufalloc_ms,gpu_compute_ms,cpu_compute_ms,compile_overhead_ms,cpu_total_ms,end2end_ms,load_source,load_bytes,load_mibps,ingest_ms,query_compute_ms,gpu_trials_n,gpu_p10_ms,gpu_p90_ms,gpu_mad_ms,gpu_name,gpu_budget_bytes"
} > "$OUTPUT"

GPU_NAME=""
GPU_BUDGET=""
FAILURES=0

run_one() {
    local sf="$1" q="$2"
    local log="$LOG_DIR/${sf}_${q}.log"
    echo "  -> $sf $q"

    local rc=0
    "$BIN" "${CHECK_ARGS[@]}" "$sf" "$q" > "$log" 2>&1 || rc=$?

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
        local status="NO_TIMING"
        [[ $rc -ne 0 ]] && status="FAIL"
        echo "${sf},${q},${status}$(printf ',%.0s' {1..21}),${GPU_NAME},${GPU_BUDGET}" >> "$OUTPUT"
        if [[ $rc -ne 0 ]]; then
            FAILURES=$((FAILURES + 1))
            echo "     FAILED (see $log)"
        fi
        return
    fi

    local status="OK"
    if [[ $rc -ne 0 ]]; then
        status="FAIL"
        if grep -Eq '^\[CHECK\].*(FAIL|golden file missing)' "$log"; then
            status="CHECK_FAIL"
        fi
        FAILURES=$((FAILURES + 1))
        echo "     ${status} (see $log)"
    fi

    # TIMING_CSV,sf,query,analyze,plan,codegen,compile,pso,dataload,bufalloc,
    #           gpu_compute,cpu_compute,compile_overhead,cpu_total,end2end,
    #           load_source,load_bytes,load_mibps,ingest_ms,query_compute,
    #           gpu_trials_n,gpu_p10,gpu_p90,gpu_mad
    local body="${timing#TIMING_CSV,}"
    awk -v gpu="$GPU_NAME" -v bud="$GPU_BUDGET" -v status="$status" -F',' '
    {
        # $1=sf, $2=query, $3..$23 are the current TIMING_CSV payload.
        printf "%s,%s,%s", $1, $2, status;
        for (i = 3; i <= 23; i++) printf ",%s", $i;
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
[[ -n "$CHECK_DIR" ]] && echo "  Check:  $CHECK_DIR"
echo "=============================================="

for sf in "${SCALE_FACTORS[@]}"; do
    echo ">>> Scale factor: $sf"
    for q in "${QUERIES[@]}"; do
        run_one "$sf" "$q"
    done
done

echo ""
echo "Done. Wrote: $OUTPUT"
if [[ $FAILURES -ne 0 ]]; then
    echo "Failures: $FAILURES" >&2
    exit 1
fi
