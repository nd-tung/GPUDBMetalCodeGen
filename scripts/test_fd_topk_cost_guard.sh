#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/bin/GPUDBCodegen}"
SF="${SF:-sf1}"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "$TMP_ROOT/fd_topk_cost_guard.XXXXXX")"

if [ ! -x "$BIN" ]; then
    echo "fd-topk-cost-guard: binary not found or not executable: $BIN" >&2
    echo "fd-topk-cost-guard: run 'make' first or set BIN=/path/to/GPUDBCodegen" >&2
    exit 2
fi

cat > "$WORK_DIR/small_fd_topk.sql" <<'SQL'
SELECT
  n.n_nationkey,
  n.n_name,
  SUM(s.s_acctbal) AS revenue
FROM nation n, supplier s
WHERE s.s_nationkey = n.n_nationkey
GROUP BY n.n_nationkey, n.n_name
ORDER BY revenue DESC
LIMIT 5;
SQL

run_sql() {
    local sql_file="$1"
    local log_file="$2"
    "$BIN" "$SF" --sql-file "$sql_file" --csv --print-plan \
        --warmup 0 --repeat 1 > "$log_file" 2>&1
}

assert_contains() {
    local pattern="$1"
    local file="$2"
    if ! rg -q "$pattern" "$file"; then
        echo "fd-topk-cost-guard: expected pattern not found: $pattern" >&2
        tail -80 "$file" >&2
        exit 1
    fi
}

assert_not_contains() {
    local pattern="$1"
    local file="$2"
    if rg -q "$pattern" "$file"; then
        echo "fd-topk-cost-guard: unexpected pattern found: $pattern" >&2
        tail -80 "$file" >&2
        exit 1
    fi
}

small_log="$WORK_DIR/small_fd_topk.log"
run_sql "$WORK_DIR/small_fd_topk.sql" "$small_log"
assert_contains "GENERIC_ir_multi_table_fd_group_build" "$small_log"
assert_contains "GENERIC_ir_multi_table_fd_group_compact" "$small_log"
assert_contains "GENERIC_gpu_topk_select_ir_multi_fd_group" "$small_log"
assert_not_contains "GENERIC_ir_multi_table_fd_group_topk_gather" "$small_log"
assert_not_contains "GENERIC_ir_multi_table_fd_group_topk_compact" "$small_log"

large_log="$WORK_DIR/q10.log"
run_sql "sql/q10.sql" "$large_log"
assert_contains "GENERIC_ir_multi_table_fd_group_topk_compact" "$large_log"
assert_contains "GENERIC_ir_multi_table_fd_group_topk_gather" "$large_log"

echo "fd-topk-cost-guard: PASS"
echo "logs: $WORK_DIR"
