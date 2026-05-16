#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/bin/GPUDBCodegen}"
SF="${SF:-sf1}"
CHECK_DIR="${CHECK_DIR:-tmp/duckdb_generic_sql}"
OUT="${OUT:-tmp/strict_generic_$(date +%Y%m%d_%H%M%S)}"
MARKERS="${MARKERS:-ADHOC_|cpuSort|cpuGroupBy|cpuScalarAgg|buildQ[0-9]|predefined|Predefined}"

if [ ! -x "$BIN" ]; then
    echo "strict-generic: binary not found or not executable: $BIN" >&2
    echo "strict-generic: run 'make' first or set BIN=/path/to/GPUDBCodegen" >&2
    exit 2
fi

mkdir -p "$OUT"
: > "$OUT/summary.tsv"

failed=0
for q in $(seq 1 22); do
    log="$OUT/q${q}.log"
    if "$BIN" "$SF" \
        --sql-file "sql/q${q}.sql" \
        --check "$CHECK_DIR/q${q}" \
        --print-plan > "$log" 2>&1; then
        printf "q%s\tPASS\n" "$q" >> "$OUT/summary.tsv"
    else
        printf "q%s\tFAIL\n" "$q" >> "$OUT/summary.tsv"
        tail -n 120 "$log" > "$OUT/q${q}.tail"
        failed=1
    fi
done

if rg -n "$MARKERS" "$OUT" > "$OUT/marker_scan.txt"; then
    marker_failed=1
else
    marker_failed=0
    : > "$OUT/marker_scan.txt"
fi

if rg -n "metal_generic_adhoc_builder|buildGenericSingleTableAdhocPlan|buildGenericMultiTableAdhocPlan|ADHOC_|cpuSort|cpuGroupBy|cpuScalarAgg" \
    codegen Makefile --glob '!build/**' --glob '!tmp/**' > "$OUT/source_marker_scan.txt"; then
    source_failed=1
else
    source_failed=0
    : > "$OUT/source_marker_scan.txt"
fi

cat "$OUT/summary.tsv"
echo
echo "strict-generic logs: $OUT"

if [ "$failed" -ne 0 ]; then
    echo "strict-generic: one or more queries failed" >&2
    exit 1
fi
if [ "$marker_failed" -ne 0 ]; then
    echo "strict-generic: forbidden runtime markers found" >&2
    cat "$OUT/marker_scan.txt" >&2
    exit 1
fi
if [ "$source_failed" -ne 0 ]; then
    echo "strict-generic: forbidden source markers found" >&2
    cat "$OUT/source_marker_scan.txt" >&2
    exit 1
fi

echo "strict-generic: PASS"
