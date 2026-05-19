#!/usr/bin/env bash
# Run predefined and generic SQL routes, then summarize generic/predefined gaps.
set -euo pipefail

SCALE_FACTORS=()
OUTPUT_DIR=""
QUERIES_OVERRIDE=""
WARMUP=1
REPEAT=3
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        sf1|sf10|sf20|sf50|sf100) SCALE_FACTORS+=("$1"); shift ;;
        -o|--output-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            OUTPUT_DIR="$2"; shift 2 ;;
        -q|--queries)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            QUERIES_OVERRIDE="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "Missing value for --warmup" >&2; exit 1; }
            WARMUP="$2"; shift 2 ;;
        --repeat)
            [[ $# -ge 2 ]] || { echo "Missing value for --repeat" >&2; exit 1; }
            REPEAT="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ ${#SCALE_FACTORS[@]} -eq 0 ]] && SCALE_FACTORS=(sf1)
if [[ -n "$QUERIES_OVERRIDE" ]]; then
    read -r -a QUERIES <<< "$QUERIES_OVERRIDE"
else
    QUERIES=(q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BIN="build/bin/GPUDBCodegen"
if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    echo "Build before running this script." >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-build/route_gap_${TS}}"
if [[ "$RESUME" -eq 0 ]]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

STATUS_CSV="$OUTPUT_DIR/status.csv"
TIMING_CSV="$OUTPUT_DIR/timing_summary.csv"
GAP_CSV="$OUTPUT_DIR/gap_summary.csv"

if [[ "$RESUME" -eq 0 || ! -f "$STATUS_CSV" ]]; then
    printf 'route,sf,query,status\n' > "$STATUS_CSV"
fi
if [[ "$RESUME" -eq 0 || ! -f "$TIMING_CSV" ]]; then
    printf 'route,sf,query,gpu_ms,e2e_ms,query_ms\n' > "$TIMING_CSV"
fi

already_done() {
    local route="$1"
    local sf="$2"
    local q="$3"
    [[ "$RESUME" -eq 1 ]] || return 1
    awk -F, -v route="$route" -v sf="$sf" -v q="$q" '
        $1 == route && $2 == sf && $3 == q && $4 == "OK" { ok = 1 }
        END { exit ok ? 0 : 1 }
    ' "$STATUS_CSV" &&
    awk -F, -v route="$route" -v sf="$sf" -v q="$q" '
        $1 == route && $2 == sf && $3 == q && $6 != "" { ok = 1 }
        END { exit ok ? 0 : 1 }
    ' "$TIMING_CSV"
}

run_one() {
    local route="$1"
    local sf="$2"
    local q="$3"
    local out="$OUTPUT_DIR/$route/$sf/$q"
    local log="$out/run.log"
    mkdir -p "$out"

    if already_done "$route" "$sf" "$q"; then
        echo "[$route] $sf $q (skip)"
        return
    fi

    echo "[$route] $sf $q"
    local rc=0
    if [[ "$route" == "predefined" ]]; then
        "$BIN" "$sf" "$q" --warmup "$WARMUP" --repeat "$REPEAT" --csv > "$log" 2>&1 || rc=$?
    else
        "$BIN" "$sf" --sql-file "sql/${q}.sql" --warmup "$WARMUP" --repeat "$REPEAT" --csv > "$log" 2>&1 || rc=$?
    fi

    local stat="OK"
    [[ $rc -ne 0 ]] && stat="FAIL_$rc"
    printf '%s,%s,%s,%s\n' "$route" "$sf" "$q" "$stat" >> "$STATUS_CSV"

    local timing
    timing="$(awk -F, '/^TIMING_CSV/ {n=split($0,f,","); gpu=f[11]; e2e=f[15]; query=f[n]} END {if (query != "") printf "%.3f,%.3f,%.3f", gpu, e2e, query}' "$log")"
    if [[ -n "$timing" ]]; then
        printf '%s,%s,%s,%s\n' "$route" "$sf" "$q" "$timing" >> "$TIMING_CSV"
    else
        printf '%s,%s,%s,,,\n' "$route" "$sf" "$q" >> "$TIMING_CSV"
    fi
}

for route in predefined generic; do
    for sf in "${SCALE_FACTORS[@]}"; do
        for q in "${QUERIES[@]}"; do
            run_one "$route" "$sf" "$q"
        done
    done
done

awk -F, '
NR == 1 { next }
{
    key = $2 "," $3
    query_ms[key "," $1] = $6
    gpu_ms[key "," $1] = $4
    e2e_ms[key "," $1] = $5
}
END {
    print "sf,query,pre_query_ms,generic_query_ms,ratio,delta_ms,pre_gpu_ms,generic_gpu_ms,pre_e2e_ms,generic_e2e_ms"
    ns = split("'"${SCALE_FACTORS[*]}"'", sfs, " ")
    nq = split("'"${QUERIES[*]}"'", qs, " ")
    for (si = 1; si <= ns; ++si) {
        sf = sfs[si]
        for (qi = 1; qi <= nq; ++qi) {
            q = qs[qi]
            key = sf "," q
            p = query_ms[key ",predefined"] + 0
            g = query_ms[key ",generic"] + 0
            ratio = p > 0 ? g / p : 0
            delta = g - p
            printf "%s,%s,%.3f,%.3f,%.3f,%+.3f,%.3f,%.3f,%.3f,%.3f\n",
                   sf, q, p, g, ratio, delta,
                   gpu_ms[key ",predefined"] + 0,
                   gpu_ms[key ",generic"] + 0,
                   e2e_ms[key ",predefined"] + 0,
                   e2e_ms[key ",generic"] + 0
        }
    }
}' "$TIMING_CSV" > "$GAP_CSV"

echo ""
echo "Wrote:"
echo "  $STATUS_CSV"
echo "  $TIMING_CSV"
echo "  $GAP_CSV"
echo ""
echo "Top gaps by query execution ratio:"
(head -1 "$GAP_CSV"; tail -n +2 "$GAP_CSV" | sort -t, -k5,5nr | head -20)

if awk -F, 'NR > 1 && $4 ~ /^FAIL/ { found = 1 } END { exit found ? 0 : 1 }' "$STATUS_CSV"; then
    echo "One or more runs failed. See $STATUS_CSV" >&2
    exit 1
fi
