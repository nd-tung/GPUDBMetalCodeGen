#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all_queries.sh
#
# Runs all 22 TPC-H queries through the codegen pipeline at one or more scale
# factors and writes a CSV report including chip/OS/GPU/memory information.
# Also writes a compact CSV with query execution time and JIT overhead.
# Ensures data exists before running: existing .tbl files are converted to
# .colbin, and missing scale-factor data is generated via DuckDB dbgen or
# tpch-dbgen.
#
# Usage:
#   scripts/run_all_queries.sh [sf1|sf10|sf20|sf50|sf100 ...] [-o results.csv] [-q "q1 q2"]
#                              [--outer N] [--warmup N] [--repeat N]
#
# Defaults: SF=sf1, all 22 queries, outer=1, warmup=3, repeat=3,
#           output = build/<chip>_<timestamp>.csv
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- CLI parsing ------------------------------------------------------------
SCALE_FACTORS=()
OUTPUT=""
QUERIES_OVERRIDE=""
OUTER=1
WARMUP=3
REPEAT=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        sf1|sf10|sf20|sf50|sf100) SCALE_FACTORS+=("$1"); shift ;;
        -o|--output)    OUTPUT="$2"; shift 2 ;;
        -q|--queries)   QUERIES_OVERRIDE="$2"; shift 2 ;;
        --outer)
            [[ $# -ge 2 ]] || { echo "Missing value for --outer" >&2; exit 1; }
            OUTER="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "Missing value for --warmup" >&2; exit 1; }
            WARMUP="$2"; shift 2 ;;
        --repeat)
            [[ $# -ge 2 ]] || { echo "Missing value for --repeat" >&2; exit 1; }
            REPEAT="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Runs all 22 TPC-H queries through the predefined codegen pipeline and writes:
  - a full CSV report
  - a compact <output>_execution.csv with query,execution_time_ms,jit_overhead_ms

Usage:
  scripts/run_all_queries.sh [sf1|sf10|sf20|sf50|sf100 ...] [-o results.csv] [-q "q1 q2"]
                             [--outer N] [--warmup N] [--repeat N]

Defaults:
  scale=sf1, queries=q1..q22, outer=1, warmup=3, repeat=3,
  output=build/<chip>_<timestamp>.csv
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ ${#SCALE_FACTORS[@]} -eq 0 ]] && SCALE_FACTORS=("sf1")

# ---- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ensure_libpg_query() {
    if [[ -f "third_party/libpg_query/pg_query.h" ]]; then
        return
    fi

    echo "third_party/libpg_query is missing; initializing submodule..."
    if [[ ! -f ".gitmodules" ]]; then
        echo "ERROR: .gitmodules missing; cannot initialize third_party/libpg_query" >&2
        exit 1
    fi
    git submodule update --init --recursive third_party/libpg_query

    if [[ ! -f "third_party/libpg_query/pg_query.h" ]]; then
        echo "ERROR: third_party/libpg_query/pg_query.h still missing after submodule init" >&2
        exit 1
    fi
}

ensure_libpg_query

BIN="build/bin/GPUDBCodegen"
if [[ ! -x "$BIN" ]]; then
    echo "Building project..."
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

if [[ -n "$QUERIES_OVERRIDE" ]]; then
    read -r -a QUERIES <<< "$QUERIES_OVERRIDE"
else
    QUERIES=(q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22)
fi

TABLES=(region nation supplier customer part partsupp orders lineitem)
DUCKDB_BIN="${DUCKDB:-duckdb}"
TPCH_DBGEN_REPO="${TPCH_DBGEN_REPO:-https://github.com/electrum/tpch-dbgen.git}"
TPCH_DBGEN_DIR="${TPCH_DBGEN_DIR:-build/tpch-dbgen}"

all_tables_have_ext() {
    local data_dir="$1" ext="$2" table
    for table in "${TABLES[@]}"; do
        [[ -s "$data_dir/${table}.${ext}" ]] || return 1
    done
    return 0
}

remove_tbl_data_for_scale() {
    local data_dir="$1" table removed=0
    for table in "${TABLES[@]}"; do
        if [[ -f "$data_dir/${table}.tbl" ]]; then
            rm -f "$data_dir/${table}.tbl"
            removed=$((removed + 1))
        fi
    done
    if [[ $removed -gt 0 ]]; then
        echo "  Data ${data_dir##*/}: removed ${removed} .tbl files after .colbin conversion"
    fi
}

sql_quote() {
    printf "%s" "$1" | sed "s/'/''/g"
}

emit_dbgen_export_sql() {
    local sf_num="$1"
    local out_dir="$2"
    local tmp_dir="$3"
    local q_out_dir tmp_q
    q_out_dir="$(sql_quote "$out_dir")"
    tmp_q="$(sql_quote "$tmp_dir")"

    cat <<SQL
PRAGMA temp_directory='${tmp_q}';
PRAGMA memory_limit='12GB';
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=${sf_num});

COPY (SELECT concat_ws('|', r_regionkey, r_name, r_comment) || '|' AS line FROM region)
TO '${q_out_dir}/region.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', n_nationkey, n_name, n_regionkey, n_comment) || '|' AS line FROM nation)
TO '${q_out_dir}/nation.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment) || '|' AS line FROM supplier)
TO '${q_out_dir}/supplier.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment) || '|' AS line FROM customer)
TO '${q_out_dir}/customer.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment) || '|' AS line FROM part)
TO '${q_out_dir}/part.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment) || '|' AS line FROM partsupp)
TO '${q_out_dir}/partsupp.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment) || '|' AS line FROM orders)
TO '${q_out_dir}/orders.tbl' (HEADER false, DELIMITER '', QUOTE '');

COPY (SELECT concat_ws('|', l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment) || '|' AS line FROM lineitem)
TO '${q_out_dir}/lineitem.tbl' (HEADER false, DELIMITER '', QUOTE '');
SQL
}

tpch_dbgen_abs_dir() {
    case "$TPCH_DBGEN_DIR" in
        /*) printf "%s" "$TPCH_DBGEN_DIR" ;;
        *)  printf "%s/%s" "$REPO_ROOT" "$TPCH_DBGEN_DIR" ;;
    esac
}

generate_tbl_data_with_duckdb() {
    local sf_num="$1"
    local data_dir="data/SF-${sf_num}"
    local work_dir="build/dbgen_sf${sf_num}_$(bench_timestamp)"
    local tmp_dir="$work_dir/tmp"
    local db_path="$work_dir/dbgen.duckdb"
    local dbgen_sql="$work_dir/dbgen_export.sql"

    echo "  Data sf${sf_num}: generating .tbl via DuckDB dbgen"
    rm -rf "$data_dir" "$work_dir"
    mkdir -p "$data_dir" "$tmp_dir"
    emit_dbgen_export_sql "$sf_num" "$data_dir" "$tmp_dir" > "$dbgen_sql"
    "$DUCKDB_BIN" "$db_path" < "$dbgen_sql"
    rm -rf "$tmp_dir" "$db_path" "$db_path.wal"
}

ensure_tpch_dbgen() {
    local dbgen_dir
    dbgen_dir="$(tpch_dbgen_abs_dir)"
    local dbgen_bin="$dbgen_dir/dbgen"

    if [[ -x "$dbgen_bin" ]]; then
        return
    fi

    if [[ ! -d "$dbgen_dir/.git" ]]; then
        echo "  tpch-dbgen: cloning $TPCH_DBGEN_REPO"
        rm -rf "$dbgen_dir"
        mkdir -p "$(dirname "$dbgen_dir")"
        git clone --depth 1 "$TPCH_DBGEN_REPO" "$dbgen_dir"
    fi

    echo "  tpch-dbgen: building dbgen"
    make -C "$dbgen_dir" dbgen

    if [[ ! -x "$dbgen_bin" ]]; then
        echo "ERROR: tpch-dbgen build did not produce $dbgen_bin" >&2
        exit 1
    fi
}

generate_tbl_data_with_tpch_dbgen() {
    local sf_num="$1"
    local data_dir="data/SF-${sf_num}"
    local dbgen_dir
    dbgen_dir="$(tpch_dbgen_abs_dir)"
    local dbgen_bin="$dbgen_dir/dbgen"
    local dists_file="$dbgen_dir/dists.dss"

    ensure_tpch_dbgen

    echo "  Data sf${sf_num}: generating .tbl via tpch-dbgen"
    rm -rf "$data_dir"
    mkdir -p "$data_dir"
    (
        cd "$data_dir"
        "$dbgen_bin" -q -f -s "$sf_num" -b "$dists_file"
    )
}

generate_tbl_data_for_scale() {
    local sf_num="$1"

    if command -v "$DUCKDB_BIN" >/dev/null 2>&1; then
        generate_tbl_data_with_duckdb "$sf_num"
    else
        echo "  DuckDB not found; falling back to tpch-dbgen"
        generate_tbl_data_with_tpch_dbgen "$sf_num"
    fi
}

ensure_data_for_scale() {
    local sf="$1"
    local sf_num="${sf#sf}"
    local data_dir="data/SF-${sf_num}"

    if all_tables_have_ext "$data_dir" colbin; then
        echo "  Data $sf: found .colbin in $data_dir"
        remove_tbl_data_for_scale "$data_dir"
        return
    fi

    if all_tables_have_ext "$data_dir" tbl; then
        echo "  Data $sf: .tbl found; building .colbin"
    else
        generate_tbl_data_for_scale "$sf_num"
    fi

    make "colbin-${sf}"

    if ! all_tables_have_ext "$data_dir" colbin; then
        echo "ERROR: missing .colbin files for $sf in $data_dir after data setup" >&2
        exit 1
    fi

    remove_tbl_data_for_scale "$data_dir"
}

echo "Checking benchmark data..."
for sf in "${SCALE_FACTORS[@]}"; do
    ensure_data_for_scale "$sf"
done

TS="$(bench_timestamp)"
CHIP_SLUG="$(bench_chip_slug)"
mkdir -p build
OUTPUT="${OUTPUT:-build/${CHIP_SLUG}_${TS}.csv}"
if [[ "$OUTPUT" == *.csv ]]; then
    EXECUTION_OUTPUT="${OUTPUT%.csv}_execution.csv"
else
    EXECUTION_OUTPUT="${OUTPUT}_execution.csv"
fi
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
    echo "# outer=$OUTER"
    echo "# warmup=$WARMUP"
    echo "# repeat=$REPEAT"
    if [[ "${GPUDB_NO_ZEROCOPY:-0}" == "1" ]]; then
        echo "# zerocopy=disabled"
    else
        echo "# zerocopy=enabled"
    fi
    echo "# timestamp=$TS"
    # gpu fields come from the first SYSINFO_CSV line below.
    echo "scale_factor,query,status,timing_query,route,analyze_ms,plan_ms,codegen_ms,metal_compile_ms,pso_ms,compile_overhead_ms,load_source,load_bytes,load_mibps,ingest_ms,data_load_ms,io_ms,preprocess_ms,buffer_setup_ms,gpu_compute_ms,cpu_compute_ms,query_compute_ms,query_execution_ms,end_to_end_ms,execute_wall_ms,execute_residual_ms,hook_cpu_ms,hook_gpu_ms,result_collect_ms,host_post_ms,validation_ms,gpu_trials_n,gpu_p10_ms,gpu_p50_ms,gpu_p90_ms,gpu_mad_ms,hot_execution_ms,gpu_name,gpu_budget_bytes"
} > "$OUTPUT"
printf 'query,execution_time_ms,jit_overhead_ms\n' > "$EXECUTION_OUTPUT"

GPU_NAME=""
GPU_BUDGET=""
FAILURES=0

run_one() {
    local sf="$1" q="$2"
    local log="$LOG_DIR/${sf}_${q}.log"
    echo "  -> $sf $q"

    local outer_i outer_log outer_rc
    for ((outer_i = 1; outer_i <= OUTER; outer_i++)); do
        outer_log="$LOG_DIR/${sf}_${q}_outer${outer_i}.log"
        outer_rc=0
        "$BIN" --warmup 0 --repeat 1 "$sf" "$q" > "$outer_log" 2>&1 || outer_rc=$?
        if [[ $outer_rc -ne 0 ]]; then
            echo "     OUTER ${outer_i} failed (see $outer_log)"
            FAILURES=$((FAILURES + 1))
        fi
    done

    local rc=0
    "$BIN" --warmup "$WARMUP" --repeat "$REPEAT" "$sf" "$q" > "$log" 2>&1 || rc=$?

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
        echo "${sf},${q},${status}$(printf ',%.0s' {1..34}),${GPU_NAME},${GPU_BUDGET}" >> "$OUTPUT"
        printf '%s,,\n' "$q" >> "$EXECUTION_OUTPUT"
        if [[ $rc -ne 0 ]]; then
            FAILURES=$((FAILURES + 1))
            echo "     FAILED (see $log)"
        fi
        return
    fi

    local status="OK"
    if [[ $rc -ne 0 ]]; then
        status="FAIL"
        FAILURES=$((FAILURES + 1))
        echo "     ${status} (see $log)"
    fi

    # TIMING_CSV,sf,timing_query,route,analyze,plan,codegen,compile,pso,
    #           compile_overhead,load_source,load_bytes,load_mibps,ingest,
    #           data_load,io,preprocess,buffer_setup,gpu_compute,cpu_compute,
    #           query_compute,query_execution,end_to_end,execute_wall,
    #           execute_residual,hook_cpu,hook_gpu,result_collect,host_post,
    #           validation,gpu_trials_n,gpu_p10,gpu_p50,gpu_p90,gpu_mad,
    #           hot_execution
    local body="${timing#TIMING_CSV,}"
    awk -v gpu="$GPU_NAME" -v bud="$GPU_BUDGET" -v status="$status" -v runq="$q" -F',' '
    {
        printf "%s,%s,%s,%s,%s", $1, runq, status, $2, $3;
        for (i = 4; i <= 35; i++) printf ",%s", $i;
        printf ",%s,%s\n", gpu, bud;
    }' <<< "$body" >> "$OUTPUT"
    awk -v runq="$q" -F',' '{
        jit = ($6 + 0.0) + ($7 + 0.0) + ($8 + 0.0);
        # TIMING_CSV field 35 is hot_execution_ms: execute wall + host post,
        # excluding load/preprocess/compile for prepared in-memory comparison.
        printf "%s,%s,%.3f\n", runq, $35, jit;
    }' <<< "$body" >> "$EXECUTION_OUTPUT"
}

echo "=============================================="
echo "  Host:   $HOST"
echo "  CPU:    $CPU"
echo "  RAM:    $RAM_GIB GiB"
echo "  Scales: ${SCALE_FACTORS[*]}"
echo "  Outer:  $OUTER"
echo "  Warmup: $WARMUP"
echo "  Repeat: $REPEAT"
echo "  Output: $OUTPUT"
echo "  Exec:   $EXECUTION_OUTPUT"
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
echo "Done. Wrote: $EXECUTION_OUTPUT"
if [[ $FAILURES -ne 0 ]]; then
    echo "Failures: $FAILURES" >&2
    exit 1
fi
