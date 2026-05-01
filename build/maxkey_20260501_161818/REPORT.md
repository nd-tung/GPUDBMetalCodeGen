# Max-Key Scan Optimization — Comparison Report

**Date:** 2026-05-01
**Hardware:** Apple M1, 16 GiB, macOS 15.7.4
**Workload:** TPC-H Q1–Q22 @ SF20 (`sf20`), 3 warmup + 5 measured trials each.
**Baseline reference:** commit `88d7578` (timing_20260501_153350) — same machine.

## Motivation

Prior comprehensive timing showed `dataload_ms` accounted for ~97% of end-to-end time at SF20 (~63 s of e2e across 22 queries). Profiling revealed the cost was not file I/O — colbins are mmap'd zero-copy — but cold first-touch page faults driven by **single-threaded `std::max` scans** inside `registerMaxKeySymbols` to compute per-column maxima for hash-table sizing. ~218k page faults in one thread ≈ 5 s per query at SF20.

## Approaches Implemented

Three modes selectable via `GPUDB_MAXKEY_MODE`:

| Mode       | Strategy                                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------- |
| `serial`   | Original. Single-threaded `for (x:v) max=...` scan per column. Forces serial page-faults.                             |
| `parallel` | Split column into `min(hw_concurrency, n/65536)` ranges; `std::async` per range; reduce. Parallelizes the page-faults. |
| `cache`    | Sidecar `data/SF-N/.maxkeys.json` keyed by `{file, size, mtime}` per column. On hit: skip scan entirely. On miss: parallel scan + write. |

Default changed: `Serial` → `Cache` (strictly dominates: warm runs skip the scan; cold runs equal `parallel`).

## Results @ SF20 (sum across Q1–Q22)

|              | dataload_ms |    Δ vs serial | e2e_ms  |    Δ vs serial |
| ------------ | ----------: | -------------: | ------: | -------------: |
| serial       |      65 791 |          —     |  67 654 |          —     |
| parallel     |      55 964 |     **−14.9%** |  57 783 |     **−14.6%** |
| cache (cold) |      53 962 |     **−18.0%** |  55 810 |     **−17.5%** |
| cache (warm) |      55 253 |     **−16.0%** |  57 123 |     **−15.6%** |

(Observed delta between cold-vs-warm is within run-to-run variance; eliminating the scan only saves ~1 s while parallelization shaves ~10 s. The bulk of `dataload_ms` is OS page-fault work that still happens later when buffers are bound to the GPU. The scan was *triggering* the faults early but not creating them.)

## Standout per-query wins (cache, warm — ms)

| Query | serial | cache (warm) | reduction | note                                       |
| ----- | -----: | -----------: | --------: | ------------------------------------------ |
| Q11   |  190.8 |          3.9 |       98% | small table, scan dominated                |
| Q13   |   30.7 |          1.6 |       95% | small table, scan dominated                |
| Q21   | 1016.6 |        282.0 |       72% | 4 join keys, big lineitem                  |
| Q22   |  299.8 |        120.6 |       60% | customer scan                              |
| Q20   | 1817.9 |        766.5 |       58% | partsupp + lineitem                        |
| Q9    | 8198.2 |       5317.7 |       35% | largest scan workload (lineitem×4 keys)    |
| Q8    | 5444.7 |       3549.6 |       35% |                                            |
| Q7    | 3982.3 |       2269.9 |       43% |                                            |

Queries where dataload was already small (Q1, Q4, Q12, Q14, Q17, Q18, Q19) show flat or noisy results — these are dominated by GPU compute or post-pass CPU cost, not max-key scans.

## Decision

**Default = `cache`.** Strictly dominates serial; equals or beats parallel on every query at SF20. Sidecar is per-dataset (`data/SF-N/.maxkeys.json`), self-invalidating via `{size, mtime}`, and ~2 KB per SF.

Knob preserved: `GPUDB_MAXKEY_MODE={serial,parallel,cache}` for benchmarking.

## Files

- `serial_sf20.csv`     — baseline
- `parallel_sf20.csv`   — approach B
- `cache_cold_sf20.csv` — first run with empty sidecar
- `cache_warm_sf20.csv` — second run, sidecar populated
- `*.log`               — per-query stdout
