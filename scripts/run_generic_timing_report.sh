#!/usr/bin/env bash
# Run TPC-H Q1-Q22 through the generic --sql-file route and build a timing report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/bin/GPUDBCodegen}"
SCALES="${SCALES:-sf1 sf10 sf20}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
OUT_DIR="${OUT_DIR:-build/generic_timing_$(date +%Y%m%d_%H%M%S)}"
DUCKDB="${DUCKDB:-duckdb}"
GENERIC_GOLDEN_DIR="${GENERIC_GOLDEN_DIR:-$OUT_DIR/generic_goldens}"
SKIP_GOLDEN_GEN="${SKIP_GOLDEN_GEN:-0}"

if [[ ! -x "$BIN" ]]; then
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

mkdir -p "$OUT_DIR"/logs "$OUT_DIR"/check "$OUT_DIR"/reports

if [[ "$SKIP_GOLDEN_GEN" != "1" ]]; then
    scripts/gen_generic_sql_goldens.py \
        --duckdb "$DUCKDB" \
        --out "$GENERIC_GOLDEN_DIR" \
        --scales $SCALES \
        --queries 1-22
fi

CSV="$OUT_DIR/generic_timing_raw.csv"
SUMMARY="$OUT_DIR/generic_timing_summary.csv"
REPORT="$OUT_DIR/reports/generic_timing_report.md"
MARKER_SCAN="$OUT_DIR/marker_scan.txt"

{
    echo "# command=$0"
    echo "# route=generic --sql-file"
    echo "# scales=$SCALES"
    echo "# warmup=$WARMUP"
    echo "# repeat=$REPEAT"
    echo "# git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "# git_dirty_count=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "# timestamp=$(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "scale_factor,query,status,analyze_ms,plan_ms,codegen_ms,compile_ms,pso_ms,dataload_ms,bufalloc_ms,gpu_compute_ms,cpu_compute_ms,compile_overhead_ms,cpu_total_ms,end2end_ms,load_source,load_bytes,load_mibps,ingest_ms,query_compute_ms,gpu_trials_n,gpu_p10_ms,gpu_p90_ms,gpu_mad_ms,io_ms,preprocess_ms,query_execution_ms"
} > "$CSV"

for q in $(seq 1 22); do
    qdir="$OUT_DIR/check/q$q"
    mkdir -p "$qdir"
    for sf in $SCALES; do
        sf_upper="$(printf '%s' "$sf" | tr '[:lower:]' '[:upper:]')"
        src="$GENERIC_GOLDEN_DIR/q$q/SQL_${sf_upper}.csv"
        if [[ -f "$src" ]]; then
            cp "$src" "$qdir/SQL_${sf_upper}.csv"
        fi
    done
done

non_ok=0
contract_failures=0
for sf in $SCALES; do
    echo ">>> $sf"
    for q in $(seq 1 22); do
        query="q$q"
        log="$OUT_DIR/logs/${sf}_${query}.log"
        echo "  -> $sf $query"
        rc=0
        "$BIN" "$sf" \
            --sql-file "sql/${query}.sql" \
            --check "$OUT_DIR/check/$query" \
            --csv \
            --warmup "$WARMUP" \
            --repeat "$REPEAT" \
            --print-plan > "$log" 2>&1 || rc=$?

        status="OK"
        if [[ $rc -ne 0 ]]; then
            status="FAIL"
            if grep -Eq '^\[CHECK\].*(FAIL|golden file missing)' "$log"; then
                status="CHECK_FAIL"
            fi
        fi

        timing="$(grep -m1 '^TIMING_CSV,' "$log" || true)"
        if [[ -z "$timing" ]]; then
            status="NO_TIMING"
            non_ok=$((non_ok + 1))
            echo "${sf},${query},NO_TIMING$(printf ',%.0s' {1..24})" >> "$CSV"
            continue
        fi
        body="${timing#TIMING_CSV,}"
        awk -v status="$status" -v query="$query" -F',' '
        {
            printf "%s,%s,%s", $1, query, status;
            for (i = 3; i <= 26; i++) printf ",%s", $i;
            printf "\n";
        }' <<< "$body" >> "$CSV"

        if [[ "$status" != "OK" ]]; then
            non_ok=$((non_ok + 1))
        fi
    done
done

if rg -n "ADHOC_|cpuSort|cpuGroupBy|cpuScalarAgg|buildQ[0-9]|predefined|Predefined" \
      "$OUT_DIR/logs" > "$MARKER_SCAN"; then
    echo "Forbidden marker found in logs; see $MARKER_SCAN" >&2
    contract_failures=1
else
    : > "$MARKER_SCAN"
fi

python3 - "$CSV" "$SUMMARY" "$REPORT" "$OUT_DIR" "$WARMUP" "$REPEAT" <<'PY'
import csv
import pathlib
import re
import statistics
import sys

raw_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
report_path = pathlib.Path(sys.argv[3])
out_dir = pathlib.Path(sys.argv[4])
warmup = sys.argv[5]
repeat = sys.argv[6]

rows = []
with raw_path.open() as f:
    filtered = (line for line in f if not line.startswith("#"))
    for row in csv.DictReader(filtered):
        rows.append(row)

def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0

def fmt(x):
    return f"{x:.3f}"

def qnum(q):
    return int(q[1:])

scales = []
for row in rows:
    scale = row.get("scale_factor", "").upper()
    if scale and scale not in scales:
        scales.append(scale)
if not scales:
    scales = ["SF1", "SF10", "SF20"]
scale_index = {scale: i for i, scale in enumerate(scales)}
by_scale = {s: [r for r in rows if r["scale_factor"].upper() == s] for s in scales}

def reason_for(row):
    scale = row["scale_factor"].lower()
    query = row["query"].lower()
    log_path = out_dir / "logs" / f"{scale}_{query}.log"
    if not log_path.exists():
        return "log missing"
    text = log_path.read_text(errors="replace")
    for pattern in [
        r"^\[CHECK\].*$",
        r"^Codegen error.*$",
        r".*Metal command buffer failed.*$",
        r".*(Unsupported|unsupported).*$",
        r".*(error|Error|exception|Exception).*$",
    ]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(0).strip().replace("|", "\\|")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first.replace("|", "\\|") if first else "no diagnostic line"

marker_scan = out_dir / "marker_scan.txt"
marker_pass = marker_scan.exists() and marker_scan.stat().st_size == 0
non_ok_rows = [r for r in rows if r["status"] != "OK"]

summary_fields = [
    "scale_factor", "queries", "status_ok", "gpu_total_ms", "query_execution_total_ms",
    "end2end_total_ms", "data_load_total_ms", "cpu_compute_total_ms",
    "median_gpu_ms", "max_gpu_query", "max_gpu_ms",
]
with summary_path.open("w", newline="") as fsum:
    w = csv.DictWriter(fsum, fieldnames=summary_fields)
    w.writeheader()
    for s in scales:
        rs = by_scale[s]
        ok = sum(1 for r in rs if r["status"] == "OK")
        gpu_vals = [f(r, "gpu_compute_ms") for r in rs]
        max_row = max(rs, key=lambda r: f(r, "gpu_compute_ms")) if rs else None
        w.writerow({
            "scale_factor": s,
            "queries": len(rs),
            "status_ok": ok,
            "gpu_total_ms": fmt(sum(gpu_vals)),
            "query_execution_total_ms": fmt(sum(f(r, "query_execution_ms") for r in rs)),
            "end2end_total_ms": fmt(sum(f(r, "end2end_ms") for r in rs)),
            "data_load_total_ms": fmt(sum(f(r, "dataload_ms") for r in rs)),
            "cpu_compute_total_ms": fmt(sum(f(r, "cpu_compute_ms") for r in rs)),
            "median_gpu_ms": fmt(statistics.median(gpu_vals)) if gpu_vals else "0.000",
            "max_gpu_query": max_row["query"] if max_row else "",
            "max_gpu_ms": fmt(f(max_row, "gpu_compute_ms")) if max_row else "0.000",
        })

lines = []
lines.append("# Generic SQL Timing Report")
lines.append("")
lines.append(f"- Route: `--sql-file` generic path")
lines.append(f"- Scales: {', '.join(f'`{s.lower()}`' for s in scales)}")
lines.append(f"- Queries: Q1-Q22")
lines.append(f"- Warmup: `{warmup}`")
lines.append(f"- Repeat: `{repeat}`")
lines.append(f"- Output directory: `{out_dir}`")
lines.append(f"- Correctness: `--check` against generic DuckDB SQL-shaped `qN/SQL_SF*.csv` goldens")
lines.append(f"- Forbidden marker scan: `{out_dir / 'marker_scan.txt'}`")
lines.append(f"- Generic-only marker scan: `{'PASS' if marker_pass else 'FAIL'}`")
lines.append("")
lines.append("## Scale Summary")
lines.append("")
lines.append("| Scale | OK/Total | GPU total ms | Query execution total ms | End-to-end total ms | Data load total ms | Median GPU ms | Slowest GPU query |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
for s in scales:
    rs = by_scale[s]
    ok = sum(1 for r in rs if r["status"] == "OK")
    gpu_vals = [f(r, "gpu_compute_ms") for r in rs]
    max_row = max(rs, key=lambda r: f(r, "gpu_compute_ms")) if rs else None
    slowest = f"{max_row['query']} ({fmt(f(max_row, 'gpu_compute_ms'))} ms)" if max_row else ""
    lines.append(
        f"| {s} | {ok}/{len(rs)} | {fmt(sum(gpu_vals))} | "
        f"{fmt(sum(f(r, 'query_execution_ms') for r in rs))} | "
        f"{fmt(sum(f(r, 'end2end_ms') for r in rs))} | "
        f"{fmt(sum(f(r, 'dataload_ms') for r in rs))} | "
        f"{fmt(statistics.median(gpu_vals)) if gpu_vals else '0.000'} | "
        f"{slowest} |"
    )

lines.append("")
lines.append("## Status Summary")
lines.append("")
lines.append("| Scale | OK | CHECK_FAIL | NO_TIMING | FAIL |")
lines.append("|---|---:|---:|---:|---:|")
for s in scales:
    rs = by_scale[s]
    counts = {name: sum(1 for r in rs if r["status"] == name) for name in ["OK", "CHECK_FAIL", "NO_TIMING", "FAIL"]}
    lines.append(f"| {s} | {counts['OK']} | {counts['CHECK_FAIL']} | {counts['NO_TIMING']} | {counts['FAIL']} |")

lines.append("")
lines.append("## Per-Query Timing")
lines.append("")
gpu_headers = [f"{s} GPU" for s in scales]
exec_headers = [f"{s} Exec" for s in scales]
lines.append("| Query | " + " | ".join(gpu_headers + exec_headers + ["Status"]) + " |")
lines.append("|---|" + "|".join("---:" for _ in gpu_headers) + "|" + "|".join("---:" for _ in exec_headers) + "|---|")
for q in range(1, 23):
    vals = {}
    status = []
    for s in scales:
        row = next((r for r in by_scale[s] if r["query"] == f"q{q}"), None)
        vals[s] = row
        status.append(row["status"] if row else "MISSING")
    gpu_cells = [fmt(f(vals[s], "gpu_compute_ms")) if vals[s] else "" for s in scales]
    exec_cells = [fmt(f(vals[s], "query_execution_ms")) if vals[s] else "" for s in scales]
    lines.append("| Q{} | {} |".format(q, " | ".join(gpu_cells + exec_cells + ["/".join(status)])))

if non_ok_rows:
    lines.append("")
    lines.append("## Non-OK Cases")
    lines.append("")
    lines.append("| Scale | Query | Status | Reason |")
    lines.append("|---|---|---|---|")
    for row in sorted(non_ok_rows, key=lambda r: (scale_index.get(r["scale_factor"].upper(), len(scales)), qnum(r["query"]))):
        lines.append(f"| {row['scale_factor'].upper()} | {row['query'].upper()} | {row['status']} | {reason_for(row)} |")

lines.append("")
lines.append("## Files")
lines.append("")
lines.append(f"- Raw CSV: `{raw_path}`")
lines.append(f"- Scale summary CSV: `{summary_path}`")
lines.append(f"- Logs: `{out_dir / 'logs'}`")

report_path.write_text("\n".join(lines) + "\n")
print(report_path)
PY

echo "Raw CSV: $CSV"
echo "Summary CSV: $SUMMARY"
echo "Report: $REPORT"

if [[ $non_ok -ne 0 || $contract_failures -ne 0 ]]; then
    echo "Non-OK query/scale cases: $non_ok" >&2
    echo "Generic contract marker failures: $contract_failures" >&2
    exit 1
fi
