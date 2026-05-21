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
import subprocess
import sys

from tpch_duckdb_common import (
    generic_query_sql,
    normalize_scale,
    parse_query_args,
    sql_quote,
    table_views_sql,
)


def build_duckdb_sql(
    root: pathlib.Path,
    out_dir: pathlib.Path,
    scale: str,
    queries: list[int],
) -> str:
    label, data_dir = normalize_scale(scale)
    data_path = root / data_dir
    if not data_path.exists():
        raise FileNotFoundError(f"missing data directory: {data_path}")

    tmp_dir = out_dir / "duckdb_tmp" / label
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parts = [table_views_sql(data_path, tmp_dir, memory_limit="12GB")]
    for q in queries:
        q_out = out_dir / f"q{q}"
        q_out.mkdir(parents=True, exist_ok=True)
        csv_path = q_out / f"SQL_{label}.csv"
        query = generic_query_sql(root, q)
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
