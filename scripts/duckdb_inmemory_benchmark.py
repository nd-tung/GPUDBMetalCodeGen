#!/usr/bin/env python3
"""Benchmark TPC-H SQL in DuckDB after loading data into an in-memory DB."""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time

import duckdb

from tpch_duckdb_common import (
    generic_query_sql,
    load_scale,
    normalize_scale,
    parse_query_args,
    sql_quote,
)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def write_comparison(
    duck_csv: pathlib.Path,
    gpu_csv: pathlib.Path,
    out_csv: pathlib.Path,
) -> None:
    duck_rows: dict[tuple[str, str], float] = {}
    with duck_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "OK":
                duck_rows[(row["scale_factor"], row["query"])] = float(row["duckdb_ms_p50"])

    gpu_rows: dict[tuple[str, str, str], float] = {}
    with gpu_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["query_execution_ms"]:
                gpu_rows[(row["route"], row["sf"], row["query"])] = float(row["query_execution_ms"])

    with out_csv.open("w", newline="") as f:
        fields = [
            "sf",
            "query",
            "duckdb_ms_p50",
            "predefined_ms",
            "generic_ms",
            "duckdb_over_predefined",
            "duckdb_over_generic",
            "generic_over_predefined",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sf, query in sorted(duck_rows, key=lambda k: (int(k[0][2:]), int(k[1][1:]))):
            duck_ms = duck_rows[(sf, query)]
            pre_ms = gpu_rows.get(("predefined", sf, query), 0.0)
            generic_ms = gpu_rows.get(("generic", sf, query), 0.0)
            writer.writerow({
                "sf": sf,
                "query": query,
                "duckdb_ms_p50": f"{duck_ms:.3f}",
                "predefined_ms": f"{pre_ms:.3f}" if pre_ms else "",
                "generic_ms": f"{generic_ms:.3f}" if generic_ms else "",
                "duckdb_over_predefined": f"{duck_ms / pre_ms:.3f}" if pre_ms else "",
                "duckdb_over_generic": f"{duck_ms / generic_ms:.3f}" if generic_ms else "",
                "generic_over_predefined": f"{generic_ms / pre_ms:.3f}" if pre_ms and generic_ms else "",
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scales", nargs="*", default=["sf1"])
    parser.add_argument("--queries", nargs="+", default=["1-22"])
    parser.add_argument("--output", default="build/duckdb_inmemory.csv")
    parser.add_argument("--compare-gpu-csv", default="")
    parser.add_argument("--comparison-output", default="")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--memory-limit", default="90GB")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--temp-dir", default="build/duckdb_tmp")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    queries = parse_query_args(args.queries)
    out_csv = (root / args.output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = (root / args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "scale_factor",
        "query",
        "status",
        "rows",
        "load_ms",
        "prepare_ms",
        "duckdb_ms_min",
        "duckdb_ms_p50",
        "duckdb_ms_mean",
        "duckdb_ms_max",
        "warmup",
        "repeat",
        "error",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for raw_scale in args.scales:
            label, data_dir_rel = normalize_scale(raw_scale)
            sf = label.lower()
            data_dir = root / data_dir_rel
            if not data_dir.exists():
                raise FileNotFoundError(f"missing data directory: {data_dir}")

            print(f">>> DuckDB in-memory {sf}", flush=True)
            con = duckdb.connect(database=":memory:")
            con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
            con.execute(f"PRAGMA temp_directory='{sql_quote(temp_dir / sf)}'")
            if args.threads > 0:
                con.execute(f"PRAGMA threads={args.threads}")

            load_ms, _ = load_scale(con, data_dir)
            print(f"    total load: {load_ms:.3f} ms", flush=True)

            for q in queries:
                query = f"q{q}"
                stmt = f"duck_{query}"
                try:
                    sql = generic_query_sql(root, q)
                    prepare_start = time.perf_counter()
                    con.execute(f"PREPARE {stmt} AS {sql}")
                    prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
                    rows = 0
                    for _ in range(args.warmup):
                        rows = len(con.execute(f"EXECUTE {stmt}").fetchall())
                    samples: list[float] = []
                    for _ in range(args.repeat):
                        start = time.perf_counter()
                        rows = len(con.execute(f"EXECUTE {stmt}").fetchall())
                        samples.append((time.perf_counter() - start) * 1000.0)
                    writer.writerow({
                        "scale_factor": sf,
                        "query": query,
                        "status": "OK",
                        "rows": rows,
                        "load_ms": f"{load_ms:.3f}",
                        "prepare_ms": f"{prepare_ms:.3f}",
                        "duckdb_ms_min": f"{min(samples):.3f}",
                        "duckdb_ms_p50": f"{median(samples):.3f}",
                        "duckdb_ms_mean": f"{statistics.fmean(samples):.3f}",
                        "duckdb_ms_max": f"{max(samples):.3f}",
                        "warmup": args.warmup,
                        "repeat": args.repeat,
                        "error": "",
                    })
                    f.flush()
                    print(f"  -> {sf} {query}: {median(samples):.3f} ms", flush=True)
                except Exception as exc:  # keep the rest of the matrix running
                    writer.writerow({
                        "scale_factor": sf,
                        "query": query,
                        "status": "FAIL",
                        "rows": "",
                        "load_ms": f"{load_ms:.3f}",
                        "prepare_ms": "",
                        "duckdb_ms_min": "",
                        "duckdb_ms_p50": "",
                        "duckdb_ms_mean": "",
                        "duckdb_ms_max": "",
                        "warmup": args.warmup,
                        "repeat": args.repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    f.flush()
                    print(f"  -> {sf} {query}: FAIL {exc}", flush=True)
            con.close()

    if args.compare_gpu_csv:
        comparison = pathlib.Path(args.comparison_output or out_csv.with_name("duckdb_vs_gpu.csv"))
        if not comparison.is_absolute():
            comparison = root / comparison
        write_comparison(out_csv, root / args.compare_gpu_csv, comparison)
        print(f"Wrote comparison: {comparison}", flush=True)

    print(f"Wrote DuckDB results: {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
