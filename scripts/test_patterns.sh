#!/bin/bash
# Minimal test cases isolating specific broken patterns.
# Usage: bash scripts/test_patterns.sh
set -e
cd /Users/tea/Documents/GPUDBMetalCodeGen
BIN=build/bin/GPUDBCodegen
TMPDIR=/tmp/tpch_pattern_tests
mkdir -p $TMPDIR

declare -A TESTS
TESTS["T0_single_table_group"]="SELECT l_suppkey, COUNT(*) AS cnt FROM lineitem WHERE l_suppkey < 10 GROUP BY l_suppkey ORDER BY l_suppkey"
TESTS["T1_two_table_probe_group"]="SELECT l_orderkey, COUNT(*) AS cnt FROM lineitem, orders WHERE l_orderkey = o_orderkey AND l_orderkey < 10 GROUP BY l_orderkey ORDER BY l_orderkey"
TESTS["T2_two_table_build_int_group"]="SELECT o_custkey, COUNT(*) AS cnt FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_custkey < 10 GROUP BY o_custkey ORDER BY o_custkey"
TESTS["T3_two_table_build_char_filter"]="SELECT o_orderkey, o_orderdate FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_orderpriority = '1-URGENT' LIMIT 5"
TESTS["T4_two_table_build_char_group"]="SELECT o_orderpriority, COUNT(*) AS cnt FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_orderpriority IN ('1-URGENT','2-HIGH') GROUP BY o_orderpriority ORDER BY o_orderpriority"
TESTS["T5_three_table_grandchild_char_group"]="SELECT n_name, COUNT(*) AS cnt FROM lineitem, supplier, nation WHERE l_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_nationkey < 5 GROUP BY n_name ORDER BY n_name"
TESTS["T6_scalar_agg_expression"]="SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, orders WHERE l_orderkey = o_orderkey AND o_orderpriority = '1-URGENT'"
TESTS["T7_single_table_char_in_group"]="SELECT l_shipmode, COUNT(*) AS cnt FROM lineitem WHERE l_shipmode IN ('MAIL','SHIP') GROUP BY l_shipmode ORDER BY l_shipmode"

# Load DuckDB reference DB once
echo "Loading DuckDB SF1 reference..."
duckdb -csv $TMPDIR/ref.db <<'SQLEOF'
.read sql/schema.sql
COPY nation   FROM 'data/SF-1/nation.tbl'   (DELIMITER '|');
COPY region   FROM 'data/SF-1/region.tbl'   (DELIMITER '|');
COPY part     FROM 'data/SF-1/part.tbl'     (DELIMITER '|');
COPY supplier FROM 'data/SF-1/supplier.tbl'  (DELIMITER '|');
COPY partsupp FROM 'data/SF-1/partsupp.tbl'  (DELIMITER '|');
COPY customer FROM 'data/SF-1/customer.tbl'  (DELIMITER '|');
COPY orders   FROM 'data/SF-1/orders.tbl'    (DELIMITER '|');
COPY lineitem FROM 'data/SF-1/lineitem.tbl'  (DELIMITER '|');
SQLEOF

echo ""
printf "%-45s %6s %6s %s\n" "Test" "Gen" "Ref" "Status"
echo "-------------------------------------------------------------"

for name in T0_single_table_group T1_two_table_probe_group T2_two_table_build_int_group T3_two_table_build_char_filter T4_two_table_build_char_group T5_three_table_grandchild_char_group T6_scalar_agg_expression T7_single_table_char_in_group; do
    sql="${TESTS[$name]}"

    # DuckDB reference
    ref_rows=$(duckdb -csv $TMPDIR/ref.db -c "$sql" 2>/dev/null | wc -l | tr -d ' ')
    
    # GPUDBCodegen generic path
    echo "$sql" > $TMPDIR/tmp.sql
    gen_out=$($BIN --sql-file $TMPDIR/tmp.sql 2>&1 || true)
    gen_rows=$(echo "$gen_out" | grep -ac '|[ ]*[0-9]')
    
    # Determine status
    status="DIFF"
    [ "$gen_rows" = "$ref_rows" ] && status="OK"
    [ "$gen_rows" = "0" ] && [ "$ref_rows" != "0" ] && status="ZERO"
    [ "$gen_rows" != "0" ] && [ "$ref_rows" = "0" ] && status="EXTRA"
    [ "$ref_rows" = "0" ] && [ "$gen_rows" = "0" ] && status="EMPTY_OK"
    
    # Check for errors
    if echo "$gen_out" | grep -aq 'Codegen error\|fatal error\|fatal'; then
        status="ERROR"
        gen_rows="err"
    fi
    
    printf "%-45s %6s %6s %s\n" "$name" "$gen_rows" "$ref_rows" "$status"
done
