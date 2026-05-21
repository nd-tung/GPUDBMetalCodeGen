#!/usr/bin/env bash

bench_timestamp() {
    date +%Y%m%d_%H%M%S
}

bench_chip_name() {
    local chip=""
    chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    if [[ -z "$chip" ]]; then
        chip="$(system_profiler SPHardwareDataType 2>/dev/null |
            awk -F: '/Chip|Processor Name/ { gsub(/^[ \t]+/, "", $2); print $2; exit }')"
    fi
    if [[ -z "$chip" ]]; then
        chip="$(uname -m 2>/dev/null || echo unknown)"
    fi
    printf '%s\n' "$chip"
}

bench_slugify() {
    tr '[:upper:]' '[:lower:]' |
        sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

bench_chip_slug() {
    bench_chip_name | bench_slugify
}

bench_default_output_dir() {
    local prefix="$1"
    local ts="${2:-$(bench_timestamp)}"
    local chip="${3:-$(bench_chip_slug)}"
    printf 'build/%s_%s_%s\n' "$prefix" "$ts" "$chip"
}
