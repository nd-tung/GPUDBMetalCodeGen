#!/usr/bin/env bash
# Run predefined, generic SQL, and DuckDB in-memory TPC-H comparisons.
#
# Default output folder:
#   build/full_route_duckdb_compare_<timestamp>_<chip>/
set -euo pipefail

SCALE_FACTORS=()
QUERIES_OVERRIDE=""
OUTPUT_DIR=""
WARMUP=1
REPEAT=3
DUCKDB_WARMUP=1
DUCKDB_REPEAT=3
MEMORY_LIMIT="90GB"
SKIP_GOLDENS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        sf1|sf10|sf20|sf50|sf100) SCALE_FACTORS+=("$1"); shift ;;
        -o|--output-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            OUTPUT_DIR="$2"; shift 2 ;;
        -q|--queries)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            QUERIES_OVERRIDE="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "Missing value for --warmup" >&2; exit 1; }
            WARMUP="$2"; shift 2 ;;
        --repeat)
            [[ $# -ge 2 ]] || { echo "Missing value for --repeat" >&2; exit 1; }
            REPEAT="$2"; shift 2 ;;
        --duckdb-warmup)
            [[ $# -ge 2 ]] || { echo "Missing value for --duckdb-warmup" >&2; exit 1; }
            DUCKDB_WARMUP="$2"; shift 2 ;;
        --duckdb-repeat)
            [[ $# -ge 2 ]] || { echo "Missing value for --duckdb-repeat" >&2; exit 1; }
            DUCKDB_REPEAT="$2"; shift 2 ;;
        --memory-limit)
            [[ $# -ge 2 ]] || { echo "Missing value for --memory-limit" >&2; exit 1; }
            MEMORY_LIMIT="$2"; shift 2 ;;
        --skip-goldens) SKIP_GOLDENS=1; shift ;;
        -h|--help)
            cat <<'EOF'
Run predefined, generic SQL, and DuckDB in-memory TPC-H comparisons.

Default output folder:
  build/full_route_duckdb_compare_<timestamp>_<chip>/

Usage:
  scripts/run_full_route_duckdb_compare.sh [sf1 sf10 sf100] [-q "q9 q13"] [options]

Options:
  -o, --output-dir DIR     Override default timestamp/chip output folder
  --warmup N               GPU warmup runs, default 1
  --repeat N               GPU measured runs, default 3
  --duckdb-warmup N        DuckDB warmup runs, default 1
  --duckdb-repeat N        DuckDB measured runs, default 3
  --memory-limit VALUE     DuckDB memory_limit, default 90GB
  --skip-goldens           Reuse existing goldens in output dir
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/bench_common.sh"
cd "$ROOT"

[[ ${#SCALE_FACTORS[@]} -eq 0 ]] && SCALE_FACTORS=(sf1 sf10 sf100)

expand_queries() {
    local input="$1"
    local token lo hi q
    for token in $input; do
        token="${token#,}"
        token="${token%,}"
        token="${token#q}"
        [[ -n "$token" ]] || continue
        if [[ "$token" == *-* ]]; then
            lo="${token%-*}"
            hi="${token#*-}"
            for ((q = lo; q <= hi; q++)); do printf '%s\n' "$q"; done
        else
            printf '%s\n' "$token"
        fi
    done | awk '!seen[$0]++'
}

if [[ -n "$QUERIES_OVERRIDE" ]]; then
    QUERY_NUMS=()
    while IFS= read -r q; do
        QUERY_NUMS+=("$q")
    done < <(expand_queries "$QUERIES_OVERRIDE")
else
    QUERY_NUMS=($(seq 1 22))
fi

TS="$(bench_timestamp)"
CHIP_NAME="$(bench_chip_name)"
OUTPUT_DIR="${OUTPUT_DIR:-$(bench_default_output_dir full_route_duckdb_compare "$TS")}"
mkdir -p "$OUTPUT_DIR/logs"
printf '%s\n' "$OUTPUT_DIR" > build/latest_full_route_duckdb_compare.txt

BIN="build/bin/GPUDBCodegen"
if [[ ! -x "$BIN" ]]; then
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

{
    echo "timestamp=$TS"
    echo "chip=$CHIP_NAME"
    echo "output_dir=$OUTPUT_DIR"
    echo "scales=${SCALE_FACTORS[*]}"
    echo "queries=${QUERY_NUMS[*]}"
    echo "gpu_warmup=$WARMUP"
    echo "gpu_repeat=$REPEAT"
    echo "duckdb_warmup=$DUCKDB_WARMUP"
    echo "duckdb_repeat=$DUCKDB_REPEAT"
    echo "memory_limit=$MEMORY_LIMIT"
} > "$OUTPUT_DIR/run_info.txt"

PRE_GOLDENS="$OUTPUT_DIR/duckdb_predefined_goldens"
GEN_GOLDENS="$OUTPUT_DIR/duckdb_generic_goldens"

if [[ "$SKIP_GOLDENS" != "1" ]]; then
    python3 scripts/gen_duckdb_goldens.py \
        --predefined-out "$PRE_GOLDENS" \
        --generic-out "$GEN_GOLDENS" \
        --scales "${SCALE_FACTORS[@]}" \
        --queries "${QUERY_NUMS[@]}" \
        --memory-limit "$MEMORY_LIMIT" \
        --temp-dir "$OUTPUT_DIR/duckdb_golden_tmp" \
        > "$OUTPUT_DIR/gen_goldens.log" 2>&1
fi

GPU_CSV="$OUTPUT_DIR/gpu_results.csv"
printf 'route,sf,query,status,timing_query,timing_route,analyze_ms,plan_ms,codegen_ms,metal_compile_ms,pso_ms,data_load_ms,io_ms,preprocess_ms,buffer_setup_ms,gpu_compute_ms,cpu_compute_ms,query_compute_ms,query_execution_ms,end_to_end_ms,execute_wall_ms,execute_residual_ms,hook_cpu_ms,hook_gpu_ms,result_collect_ms,host_post_ms,validation_ms,gpu_p50_ms,hot_execution_ms,log\n' > "$GPU_CSV"

failures=0
for route in predefined generic; do
    for sf in "${SCALE_FACTORS[@]}"; do
        for qn in "${QUERY_NUMS[@]}"; do
            q="q${qn}"
            log="$OUTPUT_DIR/logs/${route}_${sf}_${q}.log"
            rc=0
            if [[ "$route" == predefined ]]; then
                "$BIN" "$sf" "$q" --check "$PRE_GOLDENS" --csv \
                    --warmup "$WARMUP" --repeat "$REPEAT" > "$log" 2>&1 || rc=$?
            else
                "$BIN" "$sf" --sql-file "sql/${q}.sql" \
                    --check "$GEN_GOLDENS/$q" --csv \
                    --warmup "$WARMUP" --repeat "$REPEAT" > "$log" 2>&1 || rc=$?
            fi

            status="OK"
            if [[ $rc -ne 0 ]]; then
                status="FAIL"
                if grep -Eq '^\[CHECK\].*(FAIL|golden file missing)' "$log"; then
                    status="CHECK_FAIL"
                fi
            fi
            if grep -Eq '^\[CHECK\].*(FAIL|golden file missing)' "$log"; then
                status="CHECK_FAIL"
            fi
            [[ "$status" == OK ]] || failures=$((failures + 1))

            timing="$(grep -m1 '^TIMING_CSV,' "$log" || true)"
            if [[ -n "$timing" ]]; then
                body="${timing#TIMING_CSV,}"
                awk -v route="$route" -v sf="$sf" -v q="$q" \
                    -v status="$status" -v log="$log" -F, '
                    {
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                               route, sf, q, status, $2, $3, $4, $5, $6, $7,
                               $8, $14, $15, $16, $17, $18, $19, $20, $21,
                               $22, $23, $24, $25, $26, $27, $28, $29, $32,
                               $35, log
                    }' <<< "$body" >> "$GPU_CSV"
            else
                printf '%s,%s,%s,%s,,,,,,,,,,,,,,,,,,,,,,,,,,%s\n' \
                    "$route" "$sf" "$q" "$status" "$log" >> "$GPU_CSV"
                failures=$((failures + 1))
            fi
            echo "GPU $route $sf $q $status"
        done
    done
done

python3 scripts/duckdb_inmemory_benchmark.py \
    "${SCALE_FACTORS[@]}" \
    --queries "${QUERY_NUMS[@]}" \
    --output "$OUTPUT_DIR/duckdb_inmemory.csv" \
    --warmup "$DUCKDB_WARMUP" \
    --repeat "$DUCKDB_REPEAT" \
    --memory-limit "$MEMORY_LIMIT" \
    --temp-dir "$OUTPUT_DIR/duckdb_bench_tmp" \
    > "$OUTPUT_DIR/duckdb_benchmark.log" 2>&1

python3 - "$OUTPUT_DIR" <<'PY'
import csv
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
gpu_rows = list(csv.DictReader((out / "gpu_results.csv").open()))
duck_rows = list(csv.DictReader((out / "duckdb_inmemory.csv").open()))
scales = []
queries = []
for r in gpu_rows:
    if r["sf"] not in scales:
        scales.append(r["sf"])
    if r["query"] not in queries:
        queries.append(r["query"])
queries.sort(key=lambda q: int(q[1:]))

by_gpu = {(r["sf"], r["query"], r["route"]): r for r in gpu_rows}
by_duck = {(r["scale_factor"], r["query"]): r for r in duck_rows}

def fnum(row, key):
    try:
        return float(row.get(key, "") or "nan")
    except Exception:
        return float("nan")

def fmt(value):
    return "" if value != value else f"{value:.3f}"

fields = [
    "sf", "query", "pre_status", "gen_status", "duck_status",
    "pre_exec_ms", "gen_exec_ms", "duck_ms", "gen_over_pre",
    "duck_over_pre", "pre_gpu_ms", "gen_gpu_ms",
]
rows = []
for sf in scales:
    for query in queries:
        pre = by_gpu.get((sf, query, "predefined"), {})
        gen = by_gpu.get((sf, query, "generic"), {})
        duck = by_duck.get((sf, query), {})
        pre_exec = fnum(pre, "query_execution_ms")
        gen_exec = fnum(gen, "query_execution_ms")
        duck_exec = fnum(duck, "duckdb_ms_p50")
        rows.append({
            "sf": sf,
            "query": query,
            "pre_status": pre.get("status", ""),
            "gen_status": gen.get("status", ""),
            "duck_status": duck.get("status", ""),
            "pre_exec_ms": fmt(pre_exec),
            "gen_exec_ms": fmt(gen_exec),
            "duck_ms": fmt(duck_exec),
            "gen_over_pre": fmt(gen_exec / pre_exec if pre_exec and pre_exec == pre_exec and gen_exec == gen_exec else float("nan")),
            "duck_over_pre": fmt(duck_exec / pre_exec if pre_exec and pre_exec == pre_exec and duck_exec == duck_exec else float("nan")),
            "pre_gpu_ms": fmt(fnum(pre, "gpu_compute_ms")),
            "gen_gpu_ms": fmt(fnum(gen, "gpu_compute_ms")),
        })

with (out / "comparison_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [f"out={out}"]
for route in ["predefined", "generic"]:
    route_rows = [r for r in gpu_rows if r["route"] == route]
    ok = sum(1 for r in route_rows if r["status"] == "OK")
    lines.append(f"{route}: {ok}/{len(route_rows)} OK")
duck_ok = sum(1 for r in duck_rows if r["status"] == "OK")
lines.append(f"duckdb: {duck_ok}/{len(duck_rows)} OK")
lines.append("sf,pre_total_ms,gen_total_ms,duck_total_ms,gen/pre,duck/pre,pre_wins_vs_generic")
for sf in scales:
    sf_rows = [r for r in rows if r["sf"] == sf]
    pre_total = sum(float(r["pre_exec_ms"]) for r in sf_rows if r["pre_exec_ms"])
    gen_total = sum(float(r["gen_exec_ms"]) for r in sf_rows if r["gen_exec_ms"])
    duck_total = sum(float(r["duck_ms"]) for r in sf_rows if r["duck_ms"])
    wins = sum(
        1 for r in sf_rows
        if r["pre_exec_ms"] and r["gen_exec_ms"]
        and float(r["pre_exec_ms"]) <= float(r["gen_exec_ms"])
    )
    lines.append(
        f"{sf},{pre_total:.3f},{gen_total:.3f},{duck_total:.3f},"
        f"{gen_total / pre_total:.3f},{duck_total / pre_total:.3f},{wins}/{len(sf_rows)}"
    )
lines.append("generic_faster_than_predefined")
for r in rows:
    if r["pre_exec_ms"] and r["gen_exec_ms"] and float(r["gen_exec_ms"]) < float(r["pre_exec_ms"]):
        lines.append(
            f"{r['sf']} {r['query']}: pre={r['pre_exec_ms']} "
            f"gen={r['gen_exec_ms']} ratio={r['gen_over_pre']}"
        )

(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
PY

if [[ "$failures" -ne 0 ]]; then
    echo "GPU failures: $failures" >&2
    exit 1
fi
