#!/usr/bin/env python3
"""C4 — Mann-Whitney U analysis of per-trial GPU times.

Reads a tidy CSV produced by scripts/run_significance.sh:
    config, sf, query, trial, gpu_ms, compile_ms, e2e_ms

For every (sf, query) pair, performs a two-sided Mann-Whitney U test of
each non-baseline config vs `baseline` on gpu_ms. Prints a table with
median, MAD, U statistic, p-value, and the rank-biserial effect size.

Pure stdlib (no scipy). N=30 is comfortably in the asymptotic-normal
regime, so we use the standard normal approximation with a tie
correction.
"""
from __future__ import annotations
import csv
import math
import statistics
import sys
from collections import defaultdict


def median_abs_dev(xs):
    if not xs:
        return 0.0
    m = statistics.median(xs)
    return statistics.median(abs(x - m) for x in xs)


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U with normal approximation and tie correction.

    Returns (U_a, p_two_sided, rank_biserial).
    rank_biserial = 1 - 2*U_a / (n_a*n_b)  (positive => a < b, i.e. a faster)
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan"), float("nan")

    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda t: t[0])

    # Mid-rank assignment for ties.
    ranks = [0.0] * len(combined)
    i = 0
    tie_T = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_T += t**3 - t
        i = j + 1

    R_a = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    U_a = R_a - n_a * (n_a + 1) / 2.0
    U_b = n_a * n_b - U_a

    mu = n_a * n_b / 2.0
    N = n_a + n_b
    var = n_a * n_b * (N + 1) / 12.0
    if tie_T:
        var -= n_a * n_b * tie_T / (12.0 * N * (N - 1))
    if var <= 0:
        return U_a, 1.0, 0.0
    # Continuity correction.
    z = (abs(U_a - mu) - 0.5) / math.sqrt(var)
    # Two-sided p-value via complementary error function.
    p = math.erfc(z / math.sqrt(2.0))
    rank_biserial = 1.0 - 2.0 * U_a / (n_a * n_b)
    return U_a, p, rank_biserial


def main(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["gpu_ms"] = float(r["gpu_ms"])
            except (KeyError, ValueError):
                continue
            rows.append(r)

    # Group: (sf, query) -> config -> [gpu_ms]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["sf"], r["query"])][r["config"]].append(r["gpu_ms"])

    print(f"{'sf':<5} {'query':<5} {'config':<14} "
          f"{'n':>3} {'p50':>7} {'mad':>6} "
          f"{'vs_base_p50':>11} {'speedup':>8} "
          f"{'U':>8} {'p':>9} {'effect':>7} {'sig':>4}")
    print("-" * 100)
    for (sf, q), cfgs in sorted(groups.items()):
        base = cfgs.get("baseline")
        base_med = statistics.median(base) if base else float("nan")
        for cfg, vals in sorted(cfgs.items()):
            n = len(vals)
            med = statistics.median(vals)
            mad = median_abs_dev(vals)
            if cfg == "baseline" or not base:
                U = p = eff = float("nan")
                speedup = 1.0 if cfg == "baseline" else float("nan")
            else:
                U, p, eff = mannwhitney_u(vals, base)
                speedup = base_med / med if med > 0 else float("nan")
            sig = ""
            if not math.isnan(p):
                sig = "***" if p < 1e-3 else "**" if p < 1e-2 else \
                      "*" if p < 5e-2 else "ns"
            delta = med - base_med if not math.isnan(base_med) else float("nan")
            print(f"{sf:<5} {q:<5} {cfg:<14} "
                  f"{n:>3} {med:>7.3f} {mad:>6.3f} "
                  f"{delta:>+11.3f} {speedup:>8.2f} "
                  f"{U:>8.1f} {p:>9.2e} {eff:>+7.3f} {sig:>4}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} trials.csv", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
