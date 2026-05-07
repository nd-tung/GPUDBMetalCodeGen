#!/usr/bin/env bash
# Recreate TPC-H .tbl data, DuckDB golden CSVs, correctness checks, and .colbin files.
#
# Default scales: 1 10 20
# Usage:
#   scripts/gen_golden.sh          # rebuild SF1/SF10/SF20
#   scripts/gen_golden.sh 1        # rebuild SF1 only
#   DUCKDB=/path/to/duckdb scripts/gen_golden.sh 1 10

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DUCKDB_BIN="${DUCKDB:-duckdb}"
WORK_DIR="${WORK_DIR:-build/duckdb_rebuild_$(date +%Y%m%d_%H%M%S)}"
SCALES=("${@:-1 10 20}")

if [[ $# -eq 0 ]]; then
  SCALES=(1 10 20)
fi

TABLES=(region nation supplier customer part partsupp orders lineitem)
QUERIES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)

sql_quote() {
  printf "%s" "$1" | sed "s/'/''/g"
}

sf_dir() {
  printf "data/SF-%s" "$1"
}

sf_arg() {
  printf "sf%s" "$1"
}

sf_label() {
  printf "SF%s" "$1"
}

emit_dbgen_export_sql() {
  local sf="$1"
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
CALL dbgen(sf=${sf});

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

emit_load_views_sql() {
  local data_dir="$1"
  local q_data_dir
  q_data_dir="$(sql_quote "$data_dir")"

  cat <<SQL
PRAGMA memory_limit='12GB';

CREATE VIEW lineitem AS SELECT * FROM read_csv('${q_data_dir}/lineitem.tbl', delim='|', header=false, columns={
  'l_orderkey':'BIGINT','l_partkey':'BIGINT','l_suppkey':'BIGINT','l_linenumber':'INT',
  'l_quantity':'DOUBLE','l_extendedprice':'DOUBLE','l_discount':'DOUBLE','l_tax':'DOUBLE',
  'l_returnflag':'VARCHAR','l_linestatus':'VARCHAR','l_shipdate':'DATE','l_commitdate':'DATE',
  'l_receiptdate':'DATE','l_shipinstruct':'VARCHAR','l_shipmode':'VARCHAR','l_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW orders AS SELECT * FROM read_csv('${q_data_dir}/orders.tbl', delim='|', header=false, columns={
  'o_orderkey':'BIGINT','o_custkey':'BIGINT','o_orderstatus':'VARCHAR','o_totalprice':'DOUBLE',
  'o_orderdate':'DATE','o_orderpriority':'VARCHAR','o_clerk':'VARCHAR','o_shippriority':'INT',
  'o_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW customer AS SELECT * FROM read_csv('${q_data_dir}/customer.tbl', delim='|', header=false, columns={
  'c_custkey':'BIGINT','c_name':'VARCHAR','c_address':'VARCHAR','c_nationkey':'INT',
  'c_phone':'VARCHAR','c_acctbal':'DOUBLE','c_mktsegment':'VARCHAR','c_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW part AS SELECT * FROM read_csv('${q_data_dir}/part.tbl', delim='|', header=false, columns={
  'p_partkey':'BIGINT','p_name':'VARCHAR','p_mfgr':'VARCHAR','p_brand':'VARCHAR','p_type':'VARCHAR',
  'p_size':'INT','p_container':'VARCHAR','p_retailprice':'DOUBLE','p_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW partsupp AS SELECT * FROM read_csv('${q_data_dir}/partsupp.tbl', delim='|', header=false, columns={
  'ps_partkey':'BIGINT','ps_suppkey':'BIGINT','ps_availqty':'INT','ps_supplycost':'DOUBLE',
  'ps_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW supplier AS SELECT * FROM read_csv('${q_data_dir}/supplier.tbl', delim='|', header=false, columns={
  's_suppkey':'BIGINT','s_name':'VARCHAR','s_address':'VARCHAR','s_nationkey':'INT',
  's_phone':'VARCHAR','s_acctbal':'DOUBLE','s_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW nation AS SELECT * FROM read_csv('${q_data_dir}/nation.tbl', delim='|', header=false, columns={
  'n_nationkey':'INT','n_name':'VARCHAR','n_regionkey':'INT','n_comment':'VARCHAR','_extra':'VARCHAR'});

CREATE VIEW region AS SELECT * FROM read_csv('${q_data_dir}/region.tbl', delim='|', header=false, columns={
  'r_regionkey':'INT','r_name':'VARCHAR','r_comment':'VARCHAR','_extra':'VARCHAR'});
SQL
}

emit_query_sql() {
  local q="$1"
  case "$q" in
    1)
      cat <<'SQL'
SELECT
    CASE WHEN l_returnflag = 'A' THEN 0 WHEN l_returnflag = 'N' THEN 2 ELSE 4 END
      + CASE WHEN l_linestatus = 'F' THEN 0 ELSE 1 END AS bucket,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    SUM(CAST(ROUND(l_discount * 10000) AS BIGINT)) AS sum_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY bucket
ORDER BY bucket
SQL
      ;;
    4)
      cat <<'SQL'
SELECT
    CAST(SUBSTRING(o_orderpriority, 1, 1) AS INTEGER) - 1 AS bucket,
    COUNT(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '1993-07-01'
  AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH
  AND EXISTS (
      SELECT *
      FROM lineitem
      WHERE l_orderkey = o_orderkey
        AND l_commitdate < l_receiptdate
  )
GROUP BY bucket
ORDER BY bucket
SQL
      ;;
    10)
      cat <<'SQL'
SELECT
    c_custkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1993-10-01' + INTERVAL '3' MONTH
  AND l_returnflag = 'R'
  AND c_nationkey = n_nationkey
GROUP BY c_custkey
ORDER BY revenue DESC
LIMIT 20
SQL
      ;;
    15)
      cat <<'SQL'
WITH revenue0 (supplier_no, total_revenue) AS (
    SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount))
    FROM lineitem
    WHERE l_shipdate >= DATE '1996-01-01'
      AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
    GROUP BY l_suppkey
)
SELECT s_suppkey, total_revenue
FROM supplier, revenue0
WHERE s_suppkey = supplier_no
  AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
ORDER BY s_suppkey
SQL
      ;;
    18)
      cat <<'SQL'
SELECT
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    SUM(l_quantity) AS "sum(l_quantity)"
FROM customer, orders, lineitem
WHERE o_orderkey IN (
    SELECT l_orderkey
    FROM lineitem
    GROUP BY l_orderkey
    HAVING SUM(l_quantity) > 300
)
  AND c_custkey = o_custkey
  AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100
SQL
      ;;
    *)
      perl -0pe 's/;\s*\z//' "sql/q${q}.sql"
      ;;
  esac
}

emit_golden_sql() {
  local sf="$1"
  local data_dir="$2"
  local golden_dir="$3"
  local label q out_path
  label="$(sf_label "$sf")"
  emit_load_views_sql "$data_dir"
  for q in "${QUERIES[@]}"; do
    out_path="$(sql_quote "$golden_dir/Q${q}_${label}.csv")"
    printf '\nCOPY (\n'
    emit_query_sql "$q"
    printf "\n) TO '%s' (HEADER, DELIMITER ',');\n" "$out_path"
  done
}

run_check() {
  local sf="$1"
  local mode="$2"
  local log="$WORK_DIR/check_$(sf_label "$sf")_${mode}.log"
  local cmd=(./build/bin/GPUDBCodegen --check golden "$(sf_arg "$sf")" all)

  echo "==> Checking $(sf_label "$sf") (${mode})"
  if [[ "$mode" == "tbl" ]]; then
    if ! GPUDB_NO_BINARY=1 "${cmd[@]}" > "$log" 2>&1; then
      rg "\[CHECK\]|schema mismatch|column count mismatch|mismatch|FAIL|golden file missing" "$log" || tail -80 "$log"
      return 1
    fi
  else
    if ! "${cmd[@]}" > "$log" 2>&1; then
      rg "\[CHECK\]|schema mismatch|column count mismatch|mismatch|FAIL|golden file missing" "$log" || tail -80 "$log"
      return 1
    fi
  fi
  rg "\[CHECK\]" "$log"
}

echo "==> Using DuckDB: $($DUCKDB_BIN -version)"
echo "==> Work dir: $WORK_DIR"
mkdir -p "$WORK_DIR" golden

echo "==> Building binaries"
make all tools

echo "==> Removing old golden CSVs"
rm -f golden/*.csv

echo "==> Removing old data files for scales: ${SCALES[*]}"
for sf in "${SCALES[@]}"; do
  rm -rf "$(sf_dir "$sf")"
  mkdir -p "$(sf_dir "$sf")"
done

for sf in "${SCALES[@]}"; do
  data_dir="$(sf_dir "$sf")"
  label="$(sf_label "$sf")"
  tmp_dir="$WORK_DIR/tmp_${label}"
  db_path="$WORK_DIR/dbgen_${label}.duckdb"
  dbgen_sql="$WORK_DIR/dbgen_export_${label}.sql"
  golden_sql="$WORK_DIR/golden_${label}.sql"

  echo "==> Recreating ${label} .tbl data via DuckDB dbgen"
  rm -rf "$tmp_dir" "$db_path" "$db_path.wal"
  mkdir -p "$tmp_dir"
  emit_dbgen_export_sql "$sf" "$data_dir" "$tmp_dir" > "$dbgen_sql"
  "$DUCKDB_BIN" "$db_path" < "$dbgen_sql"
  rm -rf "$tmp_dir" "$db_path" "$db_path.wal"

  echo "==> Creating ${label} golden CSVs from recreated .tbl"
  emit_golden_sql "$sf" "$data_dir" golden > "$golden_sql"
  "$DUCKDB_BIN" :memory: < "$golden_sql"

  run_check "$sf" tbl

  echo "==> Converting ${label} .tbl to .colbin"
  GPUDB_FORCE_REBUILD=1 ./build/bin/tbl_to_colbin "$data_dir"

  run_check "$sf" colbin
done

echo "==> Done. Golden CSVs and data/SF-* .tbl/.colbin files were rebuilt from DuckDB."