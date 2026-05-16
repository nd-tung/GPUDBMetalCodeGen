#!/usr/bin/env python3
"""Generate DuckDB goldens for the generic --sql-file route.

The legacy golden/Q*_SF*.csv files intentionally mirror several predefined
TPC-H result shapes.  The generic SQL route must instead compare against the
literal projection in sql/qN.sql, so this helper emits SQL-shaped
qN/SQL_SF*.csv files.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


TABLE_VIEWS = """
PRAGMA memory_limit='12GB';
PRAGMA temp_directory='{tmp_dir}';

CREATE VIEW lineitem AS SELECT * FROM read_csv('{data_dir}/lineitem.tbl', delim='|', header=false, columns={{
  'l_orderkey':'BIGINT','l_partkey':'BIGINT','l_suppkey':'BIGINT','l_linenumber':'INT',
  'l_quantity':'DOUBLE','l_extendedprice':'DOUBLE','l_discount':'DOUBLE','l_tax':'DOUBLE',
  'l_returnflag':'VARCHAR','l_linestatus':'VARCHAR','l_shipdate':'DATE','l_commitdate':'DATE',
  'l_receiptdate':'DATE','l_shipinstruct':'VARCHAR','l_shipmode':'VARCHAR','l_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW orders AS SELECT * FROM read_csv('{data_dir}/orders.tbl', delim='|', header=false, columns={{
  'o_orderkey':'BIGINT','o_custkey':'BIGINT','o_orderstatus':'VARCHAR','o_totalprice':'DOUBLE',
  'o_orderdate':'DATE','o_orderpriority':'VARCHAR','o_clerk':'VARCHAR','o_shippriority':'INT',
  'o_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW customer AS SELECT * FROM read_csv('{data_dir}/customer.tbl', delim='|', header=false, columns={{
  'c_custkey':'BIGINT','c_name':'VARCHAR','c_address':'VARCHAR','c_nationkey':'INT',
  'c_phone':'VARCHAR','c_acctbal':'DOUBLE','c_mktsegment':'VARCHAR','c_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW part AS SELECT * FROM read_csv('{data_dir}/part.tbl', delim='|', header=false, columns={{
  'p_partkey':'BIGINT','p_name':'VARCHAR','p_mfgr':'VARCHAR','p_brand':'VARCHAR','p_type':'VARCHAR',
  'p_size':'INT','p_container':'VARCHAR','p_retailprice':'DOUBLE','p_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW partsupp AS SELECT * FROM read_csv('{data_dir}/partsupp.tbl', delim='|', header=false, columns={{
  'ps_partkey':'BIGINT','ps_suppkey':'BIGINT','ps_availqty':'INT','ps_supplycost':'DOUBLE',
  'ps_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW supplier AS SELECT * FROM read_csv('{data_dir}/supplier.tbl', delim='|', header=false, columns={{
  's_suppkey':'BIGINT','s_name':'VARCHAR','s_address':'VARCHAR','s_nationkey':'INT',
  's_phone':'VARCHAR','s_acctbal':'DOUBLE','s_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW nation AS SELECT * FROM read_csv('{data_dir}/nation.tbl', delim='|', header=false, columns={{
  'n_nationkey':'INT','n_name':'VARCHAR','n_regionkey':'INT','n_comment':'VARCHAR','_extra':'VARCHAR'}});

CREATE VIEW region AS SELECT * FROM read_csv('{data_dir}/region.tbl', delim='|', header=false, columns={{
  'r_regionkey':'INT','r_name':'VARCHAR','r_comment':'VARCHAR','_extra':'VARCHAR'}});
"""


def sql_quote(path: pathlib.Path | str) -> str:
    return str(path).replace("'", "''")


def normalize_scale(scale: str) -> tuple[str, pathlib.Path]:
    s = scale.lower()
    if s.startswith("sf"):
        s = s[2:]
    if not s.isdigit():
        raise ValueError(f"invalid scale '{scale}'")
    return f"SF{s}", pathlib.Path(f"data/SF-{s}")


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


def build_duckdb_sql(root: pathlib.Path,
                     out_dir: pathlib.Path,
                     scale: str,
                     queries: list[int]) -> str:
    label, data_dir = normalize_scale(scale)
    data_path = root / data_dir
    if not data_path.exists():
        raise FileNotFoundError(f"missing data directory: {data_path}")

    tmp_dir = out_dir / "duckdb_tmp" / label
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        TABLE_VIEWS.format(
            data_dir=sql_quote(data_path),
            tmp_dir=sql_quote(tmp_dir),
        )
    ]
    for q in queries:
        q_out = out_dir / f"q{q}"
        q_out.mkdir(parents=True, exist_ok=True)
        csv_path = q_out / f"SQL_{label}.csv"
        query_path = root / "sql" / f"q{q}.sql"
        query = query_sql_for_copy(query_path)
        parts.append(
            f"\nCOPY (\n{query}\n) TO '{sql_quote(csv_path)}' "
            "(HEADER, DELIMITER ',');\n"
        )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/generic_sql_goldens")
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument("--scales", nargs="+", default=["sf1", "sf10", "sf20"])
    parser.add_argument("--queries", nargs="+", default=["1-22"])
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = parse_query_args(args.queries)

    for scale in args.scales:
        label, _ = normalize_scale(scale)
        sql = build_duckdb_sql(root, out_dir, scale, queries)
        sql_path = out_dir / f"duckdb_{label}.sql"
        log_path = out_dir / f"duckdb_{label}.log"
        sql_path.write_text(sql)
        print(f"Generating {label} generic SQL goldens -> {out_dir}", flush=True)
        with log_path.open("w") as log:
            proc = subprocess.run(
                [args.duckdb, ":memory:"],
                input=sql,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(f"DuckDB failed for {label}; see {log_path}", file=sys.stderr)
            return proc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
