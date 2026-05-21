#!/usr/bin/env python3
"""Generate DuckDB goldens from existing TPC-H .tbl data.

Unlike scripts/gen_golden.sh, this helper never recreates data/SF-*.
For each scale it loads the .tbl files into an in-memory DuckDB database once,
then emits predefined-shaped Q*_SF*.csv files and/or generic SQL-shaped
q*/SQL_SF*.csv files.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb

from tpch_duckdb_common import (
    copy_query_to_csv,
    generic_query_sql,
    load_scale,
    normalize_scale,
    parse_query_args,
    predefined_query_sql,
    sql_quote,
)


def emit_predefined(
    con: duckdb.DuckDBPyConnection,
    root: pathlib.Path,
    out_dir: pathlib.Path,
    label: str,
    queries: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        csv_path = out_dir / f"Q{q}_{label}.csv"
        elapsed = copy_query_to_csv(con, predefined_query_sql(root, q), csv_path)
        print(f"    predefined Q{q}: {elapsed:.3f} ms -> {csv_path}", flush=True)


def emit_generic(
    con: duckdb.DuckDBPyConnection,
    root: pathlib.Path,
    out_dir: pathlib.Path,
    label: str,
    queries: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        csv_path = out_dir / f"q{q}" / f"SQL_{label}.csv"
        elapsed = copy_query_to_csv(con, generic_query_sql(root, q), csv_path)
        print(f"    generic q{q}: {elapsed:.3f} ms -> {csv_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predefined-out", default="")
    parser.add_argument("--generic-out", default="")
    parser.add_argument("--scales", nargs="+", default=["sf1", "sf10", "sf100"])
    parser.add_argument("--queries", nargs="+", default=["1-22"])
    parser.add_argument("--memory-limit", default="90GB")
    parser.add_argument("--temp-dir", default="build/duckdb_tmp")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if not args.predefined_out and not args.generic_out:
        print("At least one of --predefined-out or --generic-out is required", file=sys.stderr)
        return 2

    root = pathlib.Path(__file__).resolve().parents[1]
    predefined_out = (root / args.predefined_out).resolve() if args.predefined_out else None
    generic_out = (root / args.generic_out).resolve() if args.generic_out else None
    tmp_root = (root / args.temp_dir).resolve()
    queries = parse_query_args(args.queries)

    for scale in args.scales:
        label, data_dir_rel = normalize_scale(scale)
        data_dir = root / data_dir_rel
        if not data_dir.exists():
            raise FileNotFoundError(f"missing data directory: {data_dir}")

        tmp_dir = tmp_root / label
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f">>> {label}: loading {data_dir} into DuckDB memory", flush=True)
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
            con.execute(f"PRAGMA temp_directory='{sql_quote(tmp_dir)}'")
            if args.threads > 0:
                con.execute(f"PRAGMA threads={args.threads}")
            load_ms, _ = load_scale(con, data_dir)
            print(f"    total load: {load_ms:.3f} ms", flush=True)

            if predefined_out is not None:
                emit_predefined(con, root, predefined_out, label, queries)
            if generic_out is not None:
                emit_generic(con, root, generic_out, label, queries)
        finally:
            con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
