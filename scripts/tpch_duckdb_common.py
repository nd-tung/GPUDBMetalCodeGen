"""Shared DuckDB/TPC-H helpers for local scripts."""

from __future__ import annotations

import pathlib
import re
import time
from typing import Any


TABLES = {
    "lineitem": [
        ("l_orderkey", "BIGINT"),
        ("l_partkey", "BIGINT"),
        ("l_suppkey", "BIGINT"),
        ("l_linenumber", "INT"),
        ("l_quantity", "DOUBLE"),
        ("l_extendedprice", "DOUBLE"),
        ("l_discount", "DOUBLE"),
        ("l_tax", "DOUBLE"),
        ("l_returnflag", "VARCHAR"),
        ("l_linestatus", "VARCHAR"),
        ("l_shipdate", "DATE"),
        ("l_commitdate", "DATE"),
        ("l_receiptdate", "DATE"),
        ("l_shipinstruct", "VARCHAR"),
        ("l_shipmode", "VARCHAR"),
        ("l_comment", "VARCHAR"),
    ],
    "orders": [
        ("o_orderkey", "BIGINT"),
        ("o_custkey", "BIGINT"),
        ("o_orderstatus", "VARCHAR"),
        ("o_totalprice", "DOUBLE"),
        ("o_orderdate", "DATE"),
        ("o_orderpriority", "VARCHAR"),
        ("o_clerk", "VARCHAR"),
        ("o_shippriority", "INT"),
        ("o_comment", "VARCHAR"),
    ],
    "customer": [
        ("c_custkey", "BIGINT"),
        ("c_name", "VARCHAR"),
        ("c_address", "VARCHAR"),
        ("c_nationkey", "INT"),
        ("c_phone", "VARCHAR"),
        ("c_acctbal", "DOUBLE"),
        ("c_mktsegment", "VARCHAR"),
        ("c_comment", "VARCHAR"),
    ],
    "part": [
        ("p_partkey", "BIGINT"),
        ("p_name", "VARCHAR"),
        ("p_mfgr", "VARCHAR"),
        ("p_brand", "VARCHAR"),
        ("p_type", "VARCHAR"),
        ("p_size", "INT"),
        ("p_container", "VARCHAR"),
        ("p_retailprice", "DOUBLE"),
        ("p_comment", "VARCHAR"),
    ],
    "partsupp": [
        ("ps_partkey", "BIGINT"),
        ("ps_suppkey", "BIGINT"),
        ("ps_availqty", "INT"),
        ("ps_supplycost", "DOUBLE"),
        ("ps_comment", "VARCHAR"),
    ],
    "supplier": [
        ("s_suppkey", "BIGINT"),
        ("s_name", "VARCHAR"),
        ("s_address", "VARCHAR"),
        ("s_nationkey", "INT"),
        ("s_phone", "VARCHAR"),
        ("s_acctbal", "DOUBLE"),
        ("s_comment", "VARCHAR"),
    ],
    "nation": [
        ("n_nationkey", "INT"),
        ("n_name", "VARCHAR"),
        ("n_regionkey", "INT"),
        ("n_comment", "VARCHAR"),
    ],
    "region": [
        ("r_regionkey", "INT"),
        ("r_name", "VARCHAR"),
        ("r_comment", "VARCHAR"),
    ],
}


def sql_quote(value: pathlib.Path | str) -> str:
    return str(value).replace("'", "''")


def scale_number(scale: str) -> str:
    s = scale.lower()
    if s.startswith("sf"):
        s = s[2:]
    if not s.isdigit():
        raise ValueError(f"invalid scale '{scale}'")
    return s


def normalize_scale(scale: str) -> tuple[str, pathlib.Path]:
    s = scale_number(scale)
    return f"SF{s}", pathlib.Path(f"data/SF-{s}")


def parse_query_args(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip().lower().removeprefix("q")
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
    return sorted(dict.fromkeys(out))


def strip_trailing_semicolon(sql: str) -> str:
    return re.sub(r";\s*$", "", sql.strip(), flags=re.S)


def query_sql_for_copy(path: pathlib.Path) -> str:
    text = path.read_text()
    view_match = re.search(
        r"CREATE\s+VIEW\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s+AS\s+"
        r"(.*?);\s*SELECT\s+(.*?);\s*DROP\s+VIEW\s+\1\s*;?\s*$",
        text,
        flags=re.I | re.S,
    )
    if view_match:
        name, columns, view_select, final_select = view_match.groups()
        column_clause = f" {columns.strip()}" if columns else ""
        return (
            f"WITH {name}{column_clause} AS (\n"
            f"{view_select.strip()}\n"
            f")\nSELECT {final_select.strip()}"
        )
    return strip_trailing_semicolon(text)


def generic_query_sql(root: pathlib.Path, q: int) -> str:
    return query_sql_for_copy(root / "sql" / f"q{q}.sql")


def predefined_query_sql(root: pathlib.Path, q: int) -> str:
    if q == 1:
        return """
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
""".strip()
    if q == 4:
        return """
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
""".strip()
    if q == 15:
        return """
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
""".strip()
    return generic_query_sql(root, q)


def read_csv_columns_sql(columns: list[tuple[str, str]]) -> str:
    all_columns = columns + [("_extra", "VARCHAR")]
    body = ",".join(f"'{name}':'{typ}'" for name, typ in all_columns)
    return "{" + body + "}"


def table_views_sql(
    data_dir: pathlib.Path | str,
    tmp_dir: pathlib.Path | str | None = None,
    memory_limit: str | None = None,
) -> str:
    data_path = pathlib.Path(data_dir)
    parts: list[str] = []
    if memory_limit:
        parts.append(f"PRAGMA memory_limit='{sql_quote(memory_limit)}';")
    if tmp_dir is not None:
        parts.append(f"PRAGMA temp_directory='{sql_quote(tmp_dir)}';")
    for table, columns in TABLES.items():
        path = sql_quote(data_path / f"{table}.tbl")
        parts.append(
            f"CREATE VIEW {table} AS SELECT * FROM read_csv("
            f"'{path}', delim='|', header=false, "
            f"columns={read_csv_columns_sql(columns)});"
        )
    return "\n\n".join(parts) + "\n"


def load_table(con: Any, data_dir: pathlib.Path, table: str) -> float:
    columns = TABLES[table]
    names = ", ".join(name for name, _ in columns)
    schema = read_csv_columns_sql(columns)
    path = sql_quote(data_dir / f"{table}.tbl")
    sql = f"""
        CREATE TABLE {table} AS
        SELECT {names}
        FROM read_csv(
            '{path}',
            delim='|',
            header=false,
            auto_detect=false,
            columns={schema}
        )
    """
    start = time.perf_counter()
    con.execute(sql)
    return (time.perf_counter() - start) * 1000.0


def load_scale(
    con: Any,
    data_dir: pathlib.Path,
    *,
    verbose: bool = True,
) -> tuple[float, dict[str, float]]:
    total = 0.0
    per_table: dict[str, float] = {}
    for table in TABLES:
        elapsed = load_table(con, data_dir, table)
        per_table[table] = elapsed
        total += elapsed
        if verbose:
            print(f"    loaded {table}: {elapsed:.3f} ms", flush=True)
    return total, per_table


def copy_query_to_csv(con: Any, query: str, csv_path: pathlib.Path) -> float:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    con.execute(
        f"COPY (\n{query}\n) TO '{sql_quote(csv_path)}' "
        "(HEADER, DELIMITER ',')"
    )
    return (time.perf_counter() - start) * 1000.0
