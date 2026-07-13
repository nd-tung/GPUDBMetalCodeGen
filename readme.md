# Full Benchmark

## Prerequisites

- macOS with Metal support.
- Run from the repo root.

## Run

```bash
scripts/run_all_queries.sh sf1 sf10 sf100
```

This runs all 22 predefined TPC-H queries for each scale factor with the default:

```text
warmup = 3
repeat = 3
```

The script automatically:

- initializes `third_party/libpg_query` if needed
- builds `build/bin/GPUDBCodegen` if missing
- checks for `.colbin` data
- generates missing `.tbl` data with DuckDB `dbgen`
- converts `.tbl` to `.colbin`
- removes `.tbl` files after `.colbin` is available

## Output

Default output files:

```text
build/<chip>_<timestamp>.csv
build/<chip>_<timestamp>_execution.csv
```

The compact execution CSV contains:

```csv
query,execution_time_ms,jit_overhead_ms
```

`jit_overhead_ms` is:

```text
Metal Codegen + Metal Compile + PSO Creation
```

## Useful Options

Run one scale:

```bash
scripts/run_all_queries.sh sf1
```

Run selected queries:

```bash
scripts/run_all_queries.sh sf1 -q "q1 q6 q14"
```

Override warmup/repeat:

```bash
scripts/run_all_queries.sh sf1 sf10 sf100 --warmup 5 --repeat 5
```

Choose output file:

```bash
scripts/run_all_queries.sh sf1 sf10 sf100 -o build/my_benchmark.csv
```

## Local Web UI

Run the Python website from the repo root:

```bash
python3 web/app.py --port 8000
```

Open `http://127.0.0.1:8000`, choose a TPC-H query or switch to custom SQL,
then run it to view generated Metal kernel code, query results, and timing
metrics. If `build/bin/GPUDBCodegen` is missing, the web app tries to build it
with `make` on the first run.