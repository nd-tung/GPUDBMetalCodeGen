#!/bin/zsh

set -euo pipefail

ROOT=${ROOT:-/Users/nguyen/Documents/apple-energy}
DATA=${DATA:-/Users/nguyen/Documents/GPUDBMetalBenchmark/data/SF-10}
RUN_ID=${RUN_ID:-tpch-sf10-shelly-smoke-run12}
RESULT_DIR="$ROOT/results/$RUN_ID"
RAW_DIR="$ROOT/results/raw"
LOG="$RESULT_DIR/runner.log"
STATUS="$RESULT_DIR/STATUS"
CURRENT="$RESULT_DIR/CURRENT"
LOCK_DIR="$RESULT_DIR/.runner.lock"
CONFIG="$RESULT_DIR/config.txt"
ENVIRONMENT="$RESULT_DIR/environment-start.txt"
ARTIFACT_LOCK="$RESULT_DIR/artifacts.lock"

DURATION_MS=${DURATION_MS:-20000}
BASELINE_MS=${BASELINE_MS:-5000}
SAMPLE_RATE_MS=${SAMPLE_RATE_MS:-200}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-2}
GROUP_CARDINALITY=${GROUP_CARDINALITY:-256}
SCAN_ROWS=${SCAN_ROWS:-60000000}
THREADGROUP_WIDTH=${THREADGROUP_WIDTH:-256}
SHELLY_HOST=${SHELLY_HOST:-192.168.33.1}
SHELLY_PORT=${SHELLY_PORT:-80}
SHELLY_INTERFACE=${SHELLY_INTERFACE:-en1}
SHELLY_SAMPLE_RATE_MS=${SHELLY_SAMPLE_RATE_MS:-1000}
SHELLY_TIMEOUT_MS=${SHELLY_TIMEOUT_MS:-2000}
SHELLY_ATTEMPTS=${SHELLY_ATTEMPTS:-3}
SHELLY_DEVICE_ID=${SHELLY_DEVICE_ID:-shellyplugmg3-08927259b5ec}
COOPERATIVE_TIMEOUT_MS=${COOPERATIVE_TIMEOUT_MS:-0}
COOLDOWN_MS=${COOLDOWN_MS:-5000}
CPU_PARTITIONING=dynamic-atomic-chunk-claiming-approximately-16-aligned-chunks-per-worker
MEASUREMENT_EXECUTION_USER=root-via-sudo
BENCHMARK_EXECUTION_USER=root-child-of-joule-measure

if [[ ! -x /usr/sbin/sysctl ]]; then
  print -u2 -- "required executable is unavailable: /usr/sbin/sysctl"
  exit 1
fi
DETECTED_LOGICAL_CPU_THREADS=$(/usr/sbin/sysctl -n hw.logicalcpu)
if [[ ${CPU_THREADS+x} == x ]]; then
  CPU_THREADS_SOURCE=environment
else
  CPU_THREADS=$DETECTED_LOGICAL_CPU_THREADS
  CPU_THREADS_SOURCE=sysctl-hw.logicalcpu
fi
if [[ "$DETECTED_LOGICAL_CPU_THREADS" != <-> ||
      "$CPU_THREADS" != <-> ||
      "$DETECTED_LOGICAL_CPU_THREADS" == 0 ||
      "$CPU_THREADS" == 0 ]]; then
  print -u2 -- "hw.logicalcpu and CPU_THREADS must be positive integers"
  exit 1
fi
if (( CPU_THREADS == DETECTED_LOGICAL_CPU_THREADS )); then
  CPU_THREADS_POLICY=scheduler-managed-all-logical-workers
else
  CPU_THREADS_POLICY=scheduler-managed-explicit-workers
fi
EXPECTED_CASES=8

mkdir -p "$RESULT_DIR" "$RAW_DIR"

if [[ -e "$RESULT_DIR/COMPLETE" ]]; then
  print -- "$RUN_ID is already complete"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  lock_pid=""
  if [[ -r "$LOCK_DIR/pid" ]]; then
    lock_pid=$(<"$LOCK_DIR/pid")
  fi
  if [[ "$lock_pid" == <-> ]] && kill -0 "$lock_pid" 2>/dev/null; then
    print -u2 -- "$RUN_ID is already running as pid $lock_pid"
    exit 1
  fi
  stale_lock="$LOCK_DIR.stale.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mv "$LOCK_DIR" "$stale_lock"
  mkdir "$LOCK_DIR"
fi
print -- "$$" >"$LOCK_DIR/pid"
print -- "$(date -u +%FT%TZ)" >"$LOCK_DIR/started_utc"

complete=0
completed_cases=0
executed_cases=0
skipped_cases=0
finishing=0
sudo_keepalive_pid=0

write_status() {
  local state=$1
  local phase=${2:-none}
  local temporary="$STATUS.tmp.$$"
  {
    print -- "run_id=$RUN_ID"
    print -- "state=$state"
    print -- "phase=$phase"
    print -- "completed_cases=$completed_cases"
    print -- "expected_cases=$EXPECTED_CASES"
    print -- "executed_cases=$executed_cases"
    print -- "skipped_cases=$skipped_cases"
    print -- "cpu_threads=$CPU_THREADS"
    print -- "cpu_threads_policy=$CPU_THREADS_POLICY"
    print -- "updated_utc=$(date -u +%FT%TZ)"
  } >"$temporary"
  mv "$temporary" "$STATUS"
}

finish() {
  local exit_code=$?
  if (( finishing )); then
    return "$exit_code"
  fi
  finishing=1
  set +e
  trap - EXIT ZERR
  if (( sudo_keepalive_pid > 0 )); then
    kill "$sudo_keepalive_pid" 2>/dev/null
    wait "$sudo_keepalive_pid" 2>/dev/null
  fi
  rm -rf "$LOCK_DIR"
  rm -f "$RESULT_DIR/RUNNING"
  if (( complete )); then
    rm -f "$CURRENT" "$RESULT_DIR/FAILED"
    touch "$RESULT_DIR/COMPLETE"
    write_status complete none
    print -- \
      "$(date -u +%FT%TZ) COMPLETE $RUN_ID cases=$completed_cases" |
      tee -a "$LOG"
  else
    touch "$RESULT_DIR/FAILED"
    write_status failed "${current_phase:-none}"
    print -- \
      "$(date -u +%FT%TZ) FAILED status=$exit_code $RUN_ID completed=$completed_cases/$EXPECTED_CASES phase=${current_phase:-none}" |
      tee -a "$LOG"
  fi
  exit "$exit_code"
}
trap finish EXIT ZERR
trap 'exit 130' INT
trap 'exit 143' TERM

touch "$RESULT_DIR/RUNNING"
rm -f "$RESULT_DIR/FAILED"

for numeric_value in \
  "$DURATION_MS" \
  "$BASELINE_MS" \
  "$SAMPLE_RATE_MS" \
  "$WARMUP_ITERATIONS" \
  "$GROUP_CARDINALITY" \
  "$SCAN_ROWS" \
  "$THREADGROUP_WIDTH" \
  "$SHELLY_PORT" \
  "$SHELLY_SAMPLE_RATE_MS" \
  "$SHELLY_TIMEOUT_MS" \
  "$SHELLY_ATTEMPTS" \
  "$COOPERATIVE_TIMEOUT_MS" \
  "$COOLDOWN_MS" \
  "$CPU_THREADS"; do
  if [[ "$numeric_value" != <-> ]]; then
    print -u2 -- "measurement configuration values must be unsigned integers"
    exit 1
  fi
done
if (( COOPERATIVE_TIMEOUT_MS == 0 )); then
  COOPERATIVE_TIMEOUT_MS=$((DURATION_MS + 300000))
fi
if (( DURATION_MS == 0 ||
      BASELINE_MS == 0 ||
      SAMPLE_RATE_MS == 0 ||
      SCAN_ROWS == 0 ||
      SHELLY_SAMPLE_RATE_MS == 0 ||
      SHELLY_TIMEOUT_MS == 0 ||
      SHELLY_ATTEMPTS == 0 )); then
  print -u2 -- \
    "duration, baseline, sample rates, timeouts, attempts, and scan rows must be nonzero"
  exit 1
fi
if (( SHELLY_PORT == 0 || SHELLY_PORT > 65535 )); then
  print -u2 -- "SHELLY_PORT must be between 1 and 65535"
  exit 1
fi
if [[ -z "$SHELLY_HOST" || -z "$SHELLY_INTERFACE" || -z "$SHELLY_DEVICE_ID" ]]; then
  print -u2 -- "SHELLY_HOST, SHELLY_INTERFACE, and SHELLY_DEVICE_ID must be nonempty"
  exit 1
fi
if (( GROUP_CARDINALITY == 0 ||
      GROUP_CARDINALITY > 4096 ||
      (GROUP_CARDINALITY & (GROUP_CARDINALITY - 1)) != 0 )); then
  print -u2 -- \
    "GROUP_CARDINALITY must be a power of two no greater than 4096"
  exit 1
fi
if (( THREADGROUP_WIDTH < 32 ||
      THREADGROUP_WIDTH > 512 ||
      (THREADGROUP_WIDTH & (THREADGROUP_WIDTH - 1)) != 0 )); then
  print -u2 -- \
    "THREADGROUP_WIDTH must be a power of two between 32 and 512"
  exit 1
fi
MAX_MEASUREMENT_WINDOW_ERROR_MS=$((SAMPLE_RATE_MS * 2))
(( MAX_MEASUREMENT_WINDOW_ERROR_MS >= 250 )) ||
  MAX_MEASUREMENT_WINDOW_ERROR_MS=250
NOMINAL_TOTAL_MS=$((
  EXPECTED_CASES * (DURATION_MS + BASELINE_MS + COOLDOWN_MS)
))

expected_config=$(
  print -- "schema=5"
  print -- "run_id=$RUN_ID"
  print -- "data=$DATA"
  print -- "duration_ms=$DURATION_MS"
  print -- "baseline_ms=$BASELINE_MS"
  print -- "sample_rate_ms=$SAMPLE_RATE_MS"
  print -- "warmup_iterations=$WARMUP_ITERATIONS"
  print -- "cpu_threads=$CPU_THREADS"
  print -- "cpu_threads_policy=$CPU_THREADS_POLICY"
  print -- "cpu_threads_source=$CPU_THREADS_SOURCE"
  print -- "detected_hw_logicalcpu=$DETECTED_LOGICAL_CPU_THREADS"
  print -- "cpu_affinity=scheduler-managed-no-hard-pinning"
  print -- "cpu_partitioning=$CPU_PARTITIONING"
  print -- "measurement_execution_user=$MEASUREMENT_EXECUTION_USER"
  print -- "benchmark_execution_user=$BENCHMARK_EXECUTION_USER"
  print -- "cooldown_ms=$COOLDOWN_MS"
  print -- "nominal_total_ms=$NOMINAL_TOTAL_MS"
  print -- "group_cardinality=$GROUP_CARDINALITY"
  print -- "scan_rows=$SCAN_ROWS"
  print -- "threadgroup_width=$THREADGROUP_WIDTH"
  print -- "scan_input_pattern=signed"
  print -- "scan_batch_size=1"
  print -- "shelly_host=$SHELLY_HOST"
  print -- "shelly_port=$SHELLY_PORT"
  print -- "shelly_interface=$SHELLY_INTERFACE"
  print -- "shelly_sample_rate_ms=$SHELLY_SAMPLE_RATE_MS"
  print -- "shelly_timeout_ms=$SHELLY_TIMEOUT_MS"
  print -- "shelly_attempts=$SHELLY_ATTEMPTS"
  print -- "shelly_device_id=$SHELLY_DEVICE_ID"
  print -- "wall_energy_source=aenergy.total"
  print -- "max_measurement_window_error_ms=$MAX_MEASUREMENT_WINDOW_ERROR_MS"
  print -- "cooperative_timeout_ms=$COOPERATIVE_TIMEOUT_MS"
  print -- "cases=$EXPECTED_CASES"
  print -- "case_order=hash-build-cpu,hash-build-gpu,aggregate-sum-cpu,aggregate-sum-gpu-simdgroup,filter-count-cpu,filter-count-gpu,scan-simd-cpu,scan-simdgroup-gpu"
)
if [[ -e "$CONFIG" ]]; then
  if [[ "$(<"$CONFIG")" != "$expected_config" ]]; then
    print -u2 -- \
      "configuration differs from the existing run; use a new RUN_ID"
    exit 1
  fi
else
  print -r -- "$expected_config" >"$CONFIG"
fi

for command in sudo pgrep caffeinate gzip curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    print -u2 -- "required command is unavailable: $command"
    exit 1
  fi
done
if [[ ! -x /usr/bin/plutil ]]; then
  print -u2 -- "required executable is unavailable: /usr/bin/plutil"
  exit 1
fi
if [[ ! -x /usr/bin/id || ! -x /usr/sbin/chown ]]; then
  print -u2 -- "required id or chown executable is unavailable"
  exit 1
fi
for executable in \
  "$ROOT/build/joule-measure" \
  "$ROOT/build/joule-tpch-benchmark" \
  "$ROOT/build/joule-benchmark"; do
  if [[ ! -x "$executable" ]]; then
    print -u2 -- "missing executable: $executable"
    exit 1
  fi
done
if [[ ! -s "$ROOT/build/metal/joule.metallib" ]]; then
  print -u2 -- "missing Metal library: $ROOT/build/metal/joule.metallib"
  exit 1
fi
for executable in /usr/bin/shasum /usr/bin/stat; do
  if [[ ! -x "$executable" ]]; then
    print -u2 -- "required executable is unavailable: $executable"
    exit 1
  fi
done

if [[ -d "$DATA" ]]; then
  data_directory="$DATA"
else
  if [[ ! -f "$DATA" ]]; then
    print -u2 -- "dataset does not exist: $DATA"
    exit 1
  fi
  data_directory="${DATA:h}"
fi
for colbin in lineitem.colbin part.colbin; do
  if [[ ! -s "$data_directory/$colbin" ]]; then
    print -u2 -- "missing dataset file: $data_directory/$colbin"
    exit 1
  fi
done

artifact_identity() {
  local artifact hash dataset_file stat_identity
  print -- "schema=1"
  for artifact in \
    "$ROOT/build/joule-measure" \
    "$ROOT/build/joule-benchmark" \
    "$ROOT/build/joule-tpch-benchmark" \
    "$ROOT/build/metal/joule.metallib"; do
    hash=$(/usr/bin/shasum -a 256 "$artifact" | /usr/bin/awk '{print $1}') ||
      return 1
    [[ ${#hash} == 64 && "$hash" != *[^0-9a-f]* ]] || return 1
    print -- "sha256.${artifact:t}=$hash"
  done
  for dataset_file in \
    "$data_directory/lineitem.colbin" \
    "$data_directory/part.colbin"; do
    stat_identity=$(/usr/bin/stat -f 'size=%z,mtime_epoch=%m' "$dataset_file") ||
      return 1
    print -- "dataset.${dataset_file:t}=$stat_identity"
  done
}

assert_artifacts_unchanged() {
  local actual
  actual=$(artifact_identity) || {
    print -u2 -- "could not compute benchmark artifact identity"
    return 1
  }
  if [[ ! -r "$ARTIFACT_LOCK" || "$(<"$ARTIFACT_LOCK")" != "$actual" ]]; then
    print -u2 -- \
      "benchmark binaries, Metal library, or dataset changed; use a new RUN_ID"
    return 1
  fi
}

current_artifact_identity=$(artifact_identity) || {
  print -u2 -- "could not compute benchmark artifact identity"
  exit 1
}
if [[ -e "$ARTIFACT_LOCK" ]]; then
  if [[ ! -r "$ARTIFACT_LOCK" ||
        "$(<"$ARTIFACT_LOCK")" != "$current_artifact_identity" ]]; then
    print -u2 -- \
      "artifact lock differs from the current binaries or dataset; use a new RUN_ID"
    exit 1
  fi
else
  artifact_lock_temporary="$ARTIFACT_LOCK.tmp.$$"
  print -r -- "$current_artifact_identity" >"$artifact_lock_temporary"
  mv "$artifact_lock_temporary" "$ARTIFACT_LOCK"
  chmod 0444 "$ARTIFACT_LOCK"
fi

if ! sudo -n true; then
  print -u2 -- \
    "sudo ticket is not cached; start this runner in the same PTY after sudo -v"
  exit 1
fi

RUNNER_UID=$(/usr/bin/id -u)
RUNNER_GID=$(/usr/bin/id -g)
(
  while true; do
    sleep 60
    sudo -n -v || exit 1
  done
) &
sudo_keepalive_pid=$!

fetch_shelly_rpc() {
  local url=$1
  local attempt response
  for attempt in 1 2 3 4 5; do
    if response=$(curl \
        --silent --show-error --fail \
        --interface "$SHELLY_INTERFACE" \
        --connect-timeout 3 --max-time 5 \
        "$url"); then
      print -r -- "$response"
      return 0
    fi
    (( attempt == 5 )) || sleep 1
  done
  return 1
}

fetch_shelly_health() {
  if ! shelly_device_info_json=$(fetch_shelly_rpc \
      "http://$SHELLY_HOST:$SHELLY_PORT/rpc/Shelly.GetDeviceInfo"); then
    print -u2 -- \
      "could not read Shelly device info from $SHELLY_HOST:$SHELLY_PORT"
    return 1
  fi
  if ! shelly_switch_status_json=$(fetch_shelly_rpc \
      "http://$SHELLY_HOST:$SHELLY_PORT/rpc/Switch.GetStatus?id=0"); then
    print -u2 -- \
      "could not read Shelly switch status from $SHELLY_HOST:$SHELLY_PORT"
    return 1
  fi

  shelly_reported_device_id=$(print -r -- "$shelly_device_info_json" | \
    /usr/bin/plutil -extract id raw -o - - 2>/dev/null) || {
    print -u2 -- "Shelly device-info response has no usable id"
    return 1
  }
  if [[ "$shelly_reported_device_id" != "$SHELLY_DEVICE_ID" ]]; then
    print -u2 -- \
      "Shelly device mismatch: expected=$SHELLY_DEVICE_ID actual=$shelly_reported_device_id"
    return 1
  fi
  shelly_apower_w=$(print -r -- "$shelly_switch_status_json" | \
    /usr/bin/plutil -extract apower raw -o - - 2>/dev/null) || {
    print -u2 -- "Shelly switch-status response has no numeric apower"
    return 1
  }
  shelly_aenergy_total_wh=$(print -r -- "$shelly_switch_status_json" | \
    /usr/bin/plutil -extract aenergy.total raw -o - - 2>/dev/null) || {
    print -u2 -- "Shelly switch-status response has no numeric aenergy.total"
    return 1
  }
}

fetch_shelly_health

if [[ ! -e "$ENVIRONMENT" ]]; then
  {
    print -- "captured_utc=$(date -u +%FT%TZ)"
    print -- "cpu_threads=$CPU_THREADS"
    print -- "cpu_threads_policy=$CPU_THREADS_POLICY"
    print -- "cpu_threads_source=$CPU_THREADS_SOURCE"
    print -- "detected_hw_logicalcpu=$DETECTED_LOGICAL_CPU_THREADS"
    print -- "cpu_affinity=scheduler-managed-no-hard-pinning"
    print -- "cpu_partitioning=$CPU_PARTITIONING"
    print -- "measurement_execution_user=$MEASUREMENT_EXECUTION_USER"
    print -- "benchmark_execution_user=$BENCHMARK_EXECUTION_USER"
    print -- "cooldown_ms=$COOLDOWN_MS"
    print -- "nominal_total_ms=$NOMINAL_TOTAL_MS"
    print -- "threadgroup_width=$THREADGROUP_WIDTH"
    print -- "artifact_lock=$ARTIFACT_LOCK"
    print -- "sysctl_cpu_topology:"
    /usr/sbin/sysctl \
      hw.logicalcpu hw.logicalcpu_max hw.physicalcpu hw.physicalcpu_max \
      hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu 2>/dev/null || true
    print -- "benchmark_sha256:"
    /usr/bin/shasum -a 256 \
      "$ROOT/build/joule-measure" \
      "$ROOT/build/joule-benchmark" \
      "$ROOT/build/joule-tpch-benchmark" \
      "$ROOT/build/metal/joule.metallib" 2>/dev/null || true
    print -- "shelly_endpoint=$SHELLY_HOST:$SHELLY_PORT"
    print -- "shelly_interface=$SHELLY_INTERFACE"
    print -- "shelly_expected_device_id=$SHELLY_DEVICE_ID"
    print -- "shelly_reported_device_id=$shelly_reported_device_id"
    print -- "shelly_initial_apower_w=$shelly_apower_w"
    print -- "shelly_initial_aenergy_total_wh=$shelly_aenergy_total_wh"
    print -- "shelly_device_info:"
    print -r -- "$shelly_device_info_json"
    print -- "shelly_switch_status:"
    print -r -- "$shelly_switch_status_json"
    print -- "uname:"
    /usr/bin/uname -a || true
    print -- "macos:"
    /usr/bin/sw_vers || true
    print -- "hardware:"
    /usr/sbin/system_profiler SPHardwareDataType 2>/dev/null || true
    print -- "thermal:"
    /usr/bin/pmset -g therm 2>/dev/null || true
  } >"$ENVIRONMENT"
fi

assert_measurement_idle() {
  local process
  for process in \
    joule-measure \
    powermetrics \
    joule-benchmark \
    joule-tpch-benchmark; do
    if pgrep -x "$process" >/dev/null; then
      print -u2 -- "measurement process is unexpectedly active: $process"
      return 1
    fi
  done
}

wait_for_measurement_idle() {
  local attempt
  for attempt in {1..15}; do
    if ! pgrep -x powermetrics >/dev/null &&
       ! pgrep -x joule-measure >/dev/null &&
       ! pgrep -x joule-benchmark >/dev/null &&
       ! pgrep -x joule-tpch-benchmark >/dev/null; then
      return 0
    fi
    sleep 1
  done
  print -u2 -- "measurement processes did not exit within 15 seconds"
  return 1
}

result_is_valid() {
  local output=$1
  local schema_version exit_code baseline_samples workload_samples
  local wall_baseline_samples wall_workload_samples
  local wall_energy_source wall_device_id wall_expected_device_id wall_interface
  local wall_attempts
  local wall_device_id_match wall_baseline_counter_monotonic
  local wall_workload_counter_monotonic measurement_window_error_ms
  local minimum_workload_samples=$((DURATION_MS / SAMPLE_RATE_MS * 3 / 4))
  local minimum_wall_samples=$((DURATION_MS / SHELLY_SAMPLE_RATE_MS * 3 / 4))
  (( minimum_workload_samples > 0 )) || minimum_workload_samples=1
  (( minimum_wall_samples > 0 )) || minimum_wall_samples=1
  [[ -s "$output" ]] || return 1
  schema_version=$(
    /usr/bin/plutil -extract schema_version raw -o - "$output" 2>/dev/null
  ) || return 1
  exit_code=$(
    /usr/bin/plutil -extract command_exit_code raw -o - "$output" 2>/dev/null
  ) || return 1
  baseline_samples=$(
    /usr/bin/plutil -extract baseline.sample_count raw -o - "$output" 2>/dev/null
  ) || return 1
  workload_samples=$(
    /usr/bin/plutil -extract workload.sample_count raw -o - "$output" 2>/dev/null
  ) || return 1
  wall_baseline_samples=$(
    /usr/bin/plutil -extract wall_power.baseline.sample_count raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_workload_samples=$(
    /usr/bin/plutil -extract wall_power.workload.sample_count raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_energy_source=$(
    /usr/bin/plutil -extract wall_power.energy_source raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_attempts=$(
    /usr/bin/plutil -extract wall_power.attempts raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_interface=$(
    /usr/bin/plutil -extract wall_power.interface raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_device_id=$(
    /usr/bin/plutil -extract wall_power.device_id raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_expected_device_id=$(
    /usr/bin/plutil -extract wall_power.expected_device_id raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_device_id_match=$(
    /usr/bin/plutil -extract wall_power.device_id_match raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_baseline_counter_monotonic=$(
    /usr/bin/plutil -extract wall_power.baseline.counter_monotonic raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  wall_workload_counter_monotonic=$(
    /usr/bin/plutil -extract wall_power.workload.counter_monotonic raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  measurement_window_error_ms=$(
    /usr/bin/plutil -extract measurement_window_error_ms raw -o - \
      "$output" 2>/dev/null
  ) || return 1

  [[ "$schema_version" == "2" && "$exit_code" == "0" ]] || return 1
  [[ "$baseline_samples" == <-> && "$workload_samples" == <-> &&
     "$wall_baseline_samples" == <-> && "$wall_workload_samples" == <-> ]] ||
    return 1
  [[ "$wall_energy_source" == "aenergy.total" &&
     "$wall_attempts" == "$SHELLY_ATTEMPTS" &&
     "$wall_interface" == "$SHELLY_INTERFACE" &&
     "$wall_device_id" == "$SHELLY_DEVICE_ID" &&
     "$wall_expected_device_id" == "$SHELLY_DEVICE_ID" &&
     "$wall_device_id_match" == "true" &&
     "$wall_baseline_counter_monotonic" == "true" &&
     "$wall_workload_counter_monotonic" == "true" ]] || return 1
  local numeric_field numeric_type
  for numeric_field in \
    measurement_window_error_ms \
    wall_power.workload.energy_j \
    wall_power.workload.counter_delta_wh \
    wall_power.dynamic_energy_j \
    workload.energy_j.cpu \
    workload.energy_j.gpu \
    workload.energy_j.total \
    dynamic_energy_j.cpu \
    dynamic_energy_j.gpu \
    dynamic_energy_j.total; do
    numeric_type=$(
      /usr/bin/plutil -type "$numeric_field" "$output" 2>/dev/null
    ) || return 1
    [[ "$numeric_type" == "integer" || "$numeric_type" == "float" ]] ||
      return 1
  done
  /usr/bin/awk \
    -v value="$measurement_window_error_ms" \
    -v limit="$MAX_MEASUREMENT_WINDOW_ERROR_MS" \
    'BEGIN { if (value < 0) value = -value; exit(value <= limit ? 0 : 1) }' ||
    return 1
  local scope rail rail_count expected_rail_count
  for scope in baseline workload; do
    if [[ "$scope" == baseline ]]; then
      expected_rail_count=$baseline_samples
    else
      expected_rail_count=$workload_samples
    fi
    for rail in cpu gpu total; do
      rail_count=$(
        /usr/bin/plutil -extract "$scope.rail_sample_count.$rail" raw -o - \
          "$output" 2>/dev/null
      ) || return 1
      [[ "$rail_count" == <-> && "$rail_count" == "$expected_rail_count" ]] ||
        return 1
    done
  done
  (( baseline_samples > 0 &&
     workload_samples >= minimum_workload_samples &&
     wall_baseline_samples > 0 &&
     wall_workload_samples >= minimum_wall_samples ))
}

benchmark_result_is_valid() {
  local benchmark_result=$1
  local expected_operator=$2
  local expected_backend=$3
  local expected_variant=${4:-}
  local expected_kernel=${5:-}
  local schema_version operator backend iterations wall_time_type
  local cpu_threads_requested cpu_threads_actual warmup_iterations
  local threadgroup_width
  [[ -s "$benchmark_result" ]] || return 1
  schema_version=$(
    /usr/bin/plutil -extract schema_version raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  operator=$(
    /usr/bin/plutil -extract operator raw -o - "$benchmark_result" 2>/dev/null
  ) || return 1
  backend=$(
    /usr/bin/plutil -extract backend raw -o - "$benchmark_result" 2>/dev/null
  ) || return 1
  iterations=$(
    /usr/bin/plutil -extract iterations raw -o - "$benchmark_result" 2>/dev/null
  ) || return 1
  cpu_threads_requested=$(
    /usr/bin/plutil -extract cpu_threads_requested raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  cpu_threads_actual=$(
    /usr/bin/plutil -extract cpu_threads raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  wall_time_type=$(
    /usr/bin/plutil -type wall_time_ms "$benchmark_result" 2>/dev/null
  ) || return 1
  warmup_iterations=$(
    /usr/bin/plutil -extract warmup_iterations raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  threadgroup_width=$(
    /usr/bin/plutil -extract threadgroup_width raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  [[ "$schema_version" == 1 &&
     "$operator" == "$expected_operator" &&
     "$backend" == "$expected_backend" &&
     "$iterations" == <-> && "$iterations" != 0 &&
     "$cpu_threads_requested" == <-> &&
     "$cpu_threads_actual" == <-> &&
     "$warmup_iterations" == "$WARMUP_ITERATIONS" &&
     "$threadgroup_width" == "$THREADGROUP_WIDTH" &&
     ("$wall_time_type" == integer || "$wall_time_type" == float) ]] ||
    return 1
  if [[ "$expected_backend" == cpu ]]; then
    local expected_cpu_threads=$CPU_THREADS
    if [[ "$expected_operator" == scan_sum && "$expected_kernel" == scalar ]]; then
      expected_cpu_threads=1
    fi
    [[ "$cpu_threads_requested" == "$CPU_THREADS" &&
       "$cpu_threads_actual" == "$expected_cpu_threads" ]] || return 1
  else
    [[ "$cpu_threads_requested" == 0 && "$cpu_threads_actual" == 0 ]] ||
      return 1
  fi

  if [[ "$expected_operator" == scan_sum ]]; then
    local rows input_pattern batch_size cpu_kernel gpu_kernel
    rows=$(
      /usr/bin/plutil -extract rows raw -o - "$benchmark_result" 2>/dev/null
    ) || return 1
    input_pattern=$(
      /usr/bin/plutil -extract input_pattern raw -o - \
        "$benchmark_result" 2>/dev/null
    ) || return 1
    batch_size=$(
      /usr/bin/plutil -extract batch_size raw -o - \
        "$benchmark_result" 2>/dev/null
    ) || return 1
    cpu_kernel=$(
      /usr/bin/plutil -extract cpu_kernel raw -o - \
        "$benchmark_result" 2>/dev/null
    ) || return 1
    gpu_kernel=$(
      /usr/bin/plutil -extract gpu_kernel raw -o - \
        "$benchmark_result" 2>/dev/null
    ) || return 1
    [[ "$rows" == "$SCAN_ROWS" &&
       "$input_pattern" == signed &&
       "$batch_size" == 1 ]] || return 1
    if [[ "$expected_backend" == cpu ]]; then
      [[ "$cpu_kernel" == "$expected_kernel" ]]
    else
      [[ "$gpu_kernel" == "$expected_kernel" ]]
    fi
    return
  fi

  local implementation_variant group_cardinality
  implementation_variant=$(
    /usr/bin/plutil -extract implementation_variant raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  group_cardinality=$(
    /usr/bin/plutil -extract group_cardinality raw -o - \
      "$benchmark_result" 2>/dev/null
  ) || return 1
  [[ "$implementation_variant" == "$expected_variant" &&
     "$group_cardinality" == "$GROUP_CARDINALITY" ]]
}

set_case_artifact_paths() {
  local output=$1
  local compressed_raw=$2
  local compressed_shelly_raw=$3
  local benchmark_result=$4
  local recorded_path recorded_shelly_path recorded_benchmark_result
  /usr/bin/plutil -replace raw_trace -string "$compressed_raw" "$output"
  /usr/bin/plutil -replace wall_power.raw_trace \
    -string "$compressed_shelly_raw" "$output"
  if /usr/bin/plutil -extract benchmark_result raw -o - \
      "$output" >/dev/null 2>&1; then
    /usr/bin/plutil -replace benchmark_result \
      -string "$benchmark_result" "$output"
  else
    /usr/bin/plutil -insert benchmark_result \
      -string "$benchmark_result" "$output"
  fi
  recorded_path=$(
    /usr/bin/plutil -extract raw_trace raw -o - "$output" 2>/dev/null
  ) || return 1
  recorded_shelly_path=$(
    /usr/bin/plutil -extract wall_power.raw_trace raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  recorded_benchmark_result=$(
    /usr/bin/plutil -extract benchmark_result raw -o - \
      "$output" 2>/dev/null
  ) || return 1
  [[ "$recorded_path" == "$compressed_raw" &&
     "$recorded_shelly_path" == "$compressed_shelly_raw" &&
     "$recorded_benchmark_result" == "$benchmark_result" ]]
}

shelly_trace_is_valid() {
  local trace=$1
  [[ -s "$trace" ]] || return 1
  /usr/bin/grep -E \
    '"schema_version"[[:space:]]*:[[:space:]]*1' "$trace" >/dev/null ||
    return 1
  /usr/bin/grep -E \
    '"attempt_count"[[:space:]]*:[[:space:]]*[1-9][0-9]*' "$trace" >/dev/null ||
    return 1
  /usr/bin/grep -E \
    '"success"[[:space:]]*:[[:space:]]*true' "$trace" >/dev/null
}

compressed_shelly_trace_is_valid() {
  local trace=$1
  [[ -s "$trace" ]] || return 1
  gzip -t "$trace" >/dev/null 2>&1 || return 1
  gzip -cd "$trace" | /usr/bin/grep -E \
    '"schema_version"[[:space:]]*:[[:space:]]*1' >/dev/null || return 1
  gzip -cd "$trace" | /usr/bin/grep -E \
    '"attempt_count"[[:space:]]*:[[:space:]]*[1-9][0-9]*' >/dev/null || return 1
  gzip -cd "$trace" | /usr/bin/grep -E \
    '"success"[[:space:]]*:[[:space:]]*true' >/dev/null
}

ensure_compressed_trace() {
  local raw=$1
  if [[ -s "$raw.gz" ]]; then
    gzip -t "$raw.gz" >/dev/null 2>&1
    return
  fi
  [[ -s "$raw" ]] || return 1
  gzip -f "$raw" || return 1
  gzip -t "$raw.gz" >/dev/null 2>&1
}

ensure_compressed_shelly_trace() {
  local raw=$1
  if [[ -s "$raw.gz" ]]; then
    compressed_shelly_trace_is_valid "$raw.gz"
    return
  fi
  shelly_trace_is_valid "$raw" || return 1
  gzip -f "$raw" || return 1
  compressed_shelly_trace_is_valid "$raw.gz"
}

archive_invalid_case() {
  local output=$1
  local raw=$2
  local shelly_raw=$3
  local benchmark_result=$4
  local suffix="invalid.$(date -u +%Y%m%dT%H%M%SZ).$$"
  [[ ! -e "$output" ]] || mv "$output" "$output.$suffix"
  [[ ! -e "$raw.gz" ]] || mv "$raw.gz" "$raw.gz.$suffix"
  [[ ! -e "$shelly_raw.gz" ]] ||
    mv "$shelly_raw.gz" "$shelly_raw.gz.$suffix"
  [[ ! -e "$benchmark_result" ]] ||
    mv "$benchmark_result" "$benchmark_result.$suffix"
  rm -f "$raw" "$shelly_raw"
}

case_is_resumable() {
  local output=$1
  local raw=$2
  local shelly_raw=$3
  local benchmark_result=$4
  local expected_operator=$5
  local expected_backend=$6
  local expected_variant=${7:-}
  local expected_kernel=${8:-}
  result_is_valid "$output" || return 1
  benchmark_result_is_valid \
    "$benchmark_result" "$expected_operator" "$expected_backend" \
    "$expected_variant" "$expected_kernel" || return 1
  ensure_compressed_trace "$raw" || return 1
  ensure_compressed_shelly_trace "$shelly_raw" || return 1
  set_case_artifact_paths \
    "$output" "$raw.gz" "$shelly_raw.gz" "$benchmark_result"
}

run_measured_case() {
  local phase=$1
  local description=$2
  local expected_operator=$3
  local expected_backend=$4
  local expected_variant=${5:-}
  local expected_kernel=${6:-}
  shift 6

  local output="$RESULT_DIR/$phase.json"
  local benchmark_result="$RESULT_DIR/$phase-benchmark.json"
  local raw="$RAW_DIR/$RUN_ID-$phase.plist"
  local shelly_raw="$RAW_DIR/$RUN_ID-$phase-shelly.jsonl"
  local command_status=0

  current_phase="$phase"
  print -- \
    "phase=$phase\nstarted_utc=$(date -u +%FT%TZ)\nnext_case=$((completed_cases + 1))\nexpected_cases=$EXPECTED_CASES\ncpu_threads=$CPU_THREADS\nbenchmark_result=$benchmark_result" \
    >"$CURRENT"
  write_status running "$phase"

  assert_artifacts_unchanged

  if case_is_resumable \
      "$output" "$raw" "$shelly_raw" "$benchmark_result" \
      "$expected_operator" "$expected_backend" \
      "$expected_variant" "$expected_kernel"; then
    (( skipped_cases += 1 ))
    (( completed_cases += 1 ))
    print -- \
      "$(date -u +%FT%TZ) SKIP $phase progress=$completed_cases/$EXPECTED_CASES" |
      tee -a "$LOG"
    rm -f "$CURRENT"
    current_phase=none
    write_status running none
    return
  fi

  if [[ -e "$output" || -e "$raw" || -e "$raw.gz" ||
        -e "$shelly_raw" || -e "$shelly_raw.gz" ||
        -e "$benchmark_result" ]]; then
    archive_invalid_case \
      "$output" "$raw" "$shelly_raw" "$benchmark_result"
  fi
  assert_measurement_idle
  rm -f \
    "$raw" "$raw.gz" "$shelly_raw" "$shelly_raw.gz" "$benchmark_result"

  if (( COOLDOWN_MS > 0 )); then
    local cooldown_seconds
    cooldown_seconds=$(printf '%d.%03d' \
      $((COOLDOWN_MS / 1000)) $((COOLDOWN_MS % 1000)))
    print -- \
      "$(date -u +%FT%TZ) COOLDOWN $phase duration_ms=$COOLDOWN_MS" |
      tee -a "$LOG"
    caffeinate -dimsu /bin/sleep "$cooldown_seconds"
    assert_measurement_idle
  fi

  # The dedicated Wi-Fi link can enter a transient unreachable state while the
  # machine cools down. Wake and validate it immediately before powermetrics;
  # this request is outside both the baseline and workload windows.
  if ! fetch_shelly_rpc \
      "http://$SHELLY_HOST:$SHELLY_PORT/rpc/Switch.GetStatus?id=0" \
      >/dev/null; then
    print -u2 -- "Shelly endpoint is unavailable before $phase"
    return 1
  fi

  print -- \
    "$(date -u +%FT%TZ) START $phase $description progress=$((completed_cases + 1))/$EXPECTED_CASES" |
    tee -a "$LOG"

  caffeinate -dimsu sudo -n -- "$ROOT/build/joule-measure" \
    --no-sudo \
    --cooperative \
    --cooperative-timeout-ms "$COOPERATIVE_TIMEOUT_MS" \
    --sample-rate-ms "$SAMPLE_RATE_MS" \
    --baseline-ms "$BASELINE_MS" \
    --raw "$raw" \
    --shelly-host "$SHELLY_HOST" \
    --shelly-port "$SHELLY_PORT" \
    --shelly-interface "$SHELLY_INTERFACE" \
    --shelly-sample-rate-ms "$SHELLY_SAMPLE_RATE_MS" \
    --shelly-timeout-ms "$SHELLY_TIMEOUT_MS" \
    --shelly-attempts "$SHELLY_ATTEMPTS" \
    --shelly-raw "$shelly_raw" \
    --shelly-device-id "$SHELLY_DEVICE_ID" \
    --output "$output" \
    -- "$@" --result-json "$benchmark_result" \
    >>"$LOG" 2>&1 || command_status=$?

  local -a generated_artifacts
  local generated_artifact
  generated_artifacts=()
  for generated_artifact in \
    "$output" "$raw" "$shelly_raw" "$benchmark_result"; do
    [[ ! -e "$generated_artifact" ]] || generated_artifacts+=("$generated_artifact")
  done
  if (( ${#generated_artifacts[@]} > 0 )); then
    if ! sudo -n /usr/sbin/chown \
        "$RUNNER_UID:$RUNNER_GID" "${generated_artifacts[@]}"; then
      print -u2 -- "could not restore result ownership after $phase"
      return 1
    fi
  fi

  wait_for_measurement_idle
  assert_artifacts_unchanged
  if (( command_status != 0 )); then
    print -u2 -- "$phase exited with status $command_status"
    return "$command_status"
  fi
  if ! result_is_valid "$output"; then
    print -u2 -- "invalid or unsuccessful result for $phase"
    return 1
  fi
  if ! benchmark_result_is_valid \
      "$benchmark_result" "$expected_operator" "$expected_backend" \
      "$expected_variant" "$expected_kernel"; then
    print -u2 -- "missing or invalid benchmark result for $phase"
    return 1
  fi
  if [[ ! -s "$raw" ]]; then
    print -u2 -- "missing raw powermetrics trace for $phase"
    return 1
  fi
  if ! shelly_trace_is_valid "$shelly_raw"; then
    print -u2 -- "missing or invalid raw Shelly trace for $phase"
    return 1
  fi

  gzip -f "$raw" "$shelly_raw"
  if ! gzip -t "$raw.gz" >/dev/null 2>&1; then
    print -u2 -- "compressed powermetrics trace failed validation for $phase"
    return 1
  fi
  if ! compressed_shelly_trace_is_valid "$shelly_raw.gz"; then
    print -u2 -- "compressed Shelly trace failed validation for $phase"
    return 1
  fi
  if ! set_case_artifact_paths \
      "$output" "$raw.gz" "$shelly_raw.gz" "$benchmark_result"; then
    print -u2 -- "could not record case artifact paths for $phase"
    return 1
  fi

  (( executed_cases += 1 ))
  (( completed_cases += 1 ))
  print -- \
    "$(date -u +%FT%TZ) DONE $phase progress=$completed_cases/$EXPECTED_CASES" |
    tee -a "$LOG"
  rm -f "$CURRENT"
  current_phase=none
  write_status running none
}

expected_tpch_variant() {
  local operator=$1
  local backend=$2
  shift 2
  local reduction=simdgroup
  local groupby_strategy=global-atomic
  while (( $# > 0 )); do
    case "$1" in
      --gpu-aggregate-reduction)
        reduction=$2
        shift 2
        ;;
      --gpu-groupby-strategy)
        groupby_strategy=$2
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  case "$operator:$backend" in
    filter-project:cpu) print -- cpu-two-pass-stable ;;
    filter-project:gpu) print -- gpu-bitmap-prefix-scatter ;;
    hash-build:cpu) print -- cpu-open-addressed-atomic-build ;;
    hash-build:gpu) print -- gpu-open-addressed-atomic-cas-build ;;
    hash-probe-count:cpu) print -- cpu-prebuilt-hash-probe ;;
    hash-probe-count:gpu) print -- gpu-prebuilt-hash-probe ;;
    hash-probe-materialize:cpu)
      print -- cpu-two-pass-stable-hash-materialize
      ;;
    hash-probe-materialize:gpu)
      print -- gpu-block-prefix-stable-hash-materialize
      ;;
    aggregate-*:cpu) print -- cpu-parallel ;;
    aggregate-*:gpu) print -- "gpu-$reduction" ;;
    groupby-part-count:cpu) print -- cpu-parallel ;;
    groupby-part-count:gpu) print -- "gpu-$groupby_strategy" ;;
    q6-revenue:cpu) print -- cpu-fused ;;
    q6-revenue:gpu) print -- gpu-fused ;;
    q6-revenue-unfused:cpu) print -- cpu-unfused ;;
    q6-revenue-unfused:gpu) print -- gpu-unfused ;;
    *) print -- default ;;
  esac
}

run_tpch_case() {
  local label=$1
  local operator=$2
  local backend=$3
  local description="operator=$operator backend=$backend"
  local -a cpu_arguments
  local expected_variant
  shift 3
  expected_variant=$(expected_tpch_variant "$operator" "$backend" "$@")
  if [[ "$backend" == cpu ]]; then
    cpu_arguments=(--cpu-threads "$CPU_THREADS")
    description+=" cpu_threads_requested=$CPU_THREADS cpu_policy=$CPU_THREADS_POLICY"
  else
    cpu_arguments=()
  fi
  run_measured_case \
    "$label" \
    "$description" \
    "$operator" \
    "$backend" \
    "$expected_variant" \
    "" \
    "$ROOT/build/joule-tpch-benchmark" \
      --data "$DATA" \
      --operator "$operator" \
      --backend "$backend" \
      --group-cardinality "$GROUP_CARDINALITY" \
      --threadgroup-width "$THREADGROUP_WIDTH" \
      --metallib "$ROOT/build/metal/joule.metallib" \
      --duration-ms "$DURATION_MS" \
      --warmup-iterations "$WARMUP_ITERATIONS" \
      "${cpu_arguments[@]}" \
      "$@"
}

run_scan_case() {
  local label=$1
  local backend=$2
  local kernel=$3
  local description="operator=scan-sum backend=$backend kernel=$kernel"
  local -a kernel_arguments
  if [[ "$backend" == cpu ]]; then
    kernel_arguments=(
      --cpu-kernel "$kernel"
      --cpu-threads "$CPU_THREADS"
    )
    description+=" cpu_threads_requested=$CPU_THREADS cpu_policy=$CPU_THREADS_POLICY"
  else
    kernel_arguments=(--gpu-kernel "$kernel")
  fi
  run_measured_case \
    "$label" \
    "$description" \
    scan_sum \
    "$backend" \
    "" \
    "$kernel" \
    "$ROOT/build/joule-benchmark" \
      --backend "$backend" \
      --rows "$SCAN_ROWS" \
      --input-pattern signed \
      --threadgroup-width "$THREADGROUP_WIDTH" \
      --metallib "$ROOT/build/metal/joule.metallib" \
      --duration-ms "$DURATION_MS" \
      --warmup-iterations "$WARMUP_ITERATIONS" \
      --batch-size 1 \
      "${kernel_arguments[@]}"
}

assert_measurement_idle
write_status running none
print -- "$(date -u +%FT%TZ) START $RUN_ID" | tee -a "$LOG"
print -- \
  "cases=$EXPECTED_CASES duration_ms=$DURATION_MS baseline_ms=$BASELINE_MS cooldown_ms=$COOLDOWN_MS nominal_total_ms=$NOMINAL_TOTAL_MS powermetrics_sample_rate_ms=$SAMPLE_RATE_MS shelly_sample_rate_ms=$SHELLY_SAMPLE_RATE_MS shelly_timeout_ms=$SHELLY_TIMEOUT_MS shelly_attempts=$SHELLY_ATTEMPTS cooperative_timeout_ms=$COOPERATIVE_TIMEOUT_MS warmups=$WARMUP_ITERATIONS scan_rows=$SCAN_ROWS threadgroup_width=$THREADGROUP_WIDTH artifact_lock=$ARTIFACT_LOCK" |
  tee -a "$LOG"
print -- \
  "cpu_threads=$CPU_THREADS detected_hw_logicalcpu=$DETECTED_LOGICAL_CPU_THREADS cpu_threads_source=$CPU_THREADS_SOURCE cpu_policy=$CPU_THREADS_POLICY affinity=scheduler-managed-no-hard-pinning partitioning=$CPU_PARTITIONING" |
  tee -a "$LOG"
print -- \
  "measurement_execution_user=$MEASUREMENT_EXECUTION_USER benchmark_execution_user=$BENCHMARK_EXECUTION_USER local_network_privacy=macos-root-exemption-for-detached-runner" |
  tee -a "$LOG"
print -- \
  "shelly_endpoint=$SHELLY_HOST:$SHELLY_PORT shelly_interface=$SHELLY_INTERFACE shelly_device_id=$shelly_reported_device_id initial_apower_w=$shelly_apower_w initial_aenergy_total_wh=$shelly_aenergy_total_wh wall_energy_source=aenergy.total" |
  tee -a "$LOG"

run_tpch_case hash-build-cpu hash-build cpu
run_tpch_case hash-build-gpu hash-build gpu
run_tpch_case aggregate-sum-cpu aggregate-sum cpu
run_tpch_case aggregate-sum-gpu-simdgroup aggregate-sum gpu \
  --gpu-aggregate-reduction simdgroup
run_tpch_case filter-count-cpu filter-count cpu
run_tpch_case filter-count-gpu filter-count gpu
run_scan_case scan-simd-cpu cpu simd
run_scan_case scan-simdgroup-gpu gpu simdgroup

if (( completed_cases != EXPECTED_CASES )); then
  print -u2 -- \
    "internal case-count mismatch: completed=$completed_cases expected=$EXPECTED_CASES"
  exit 1
fi

current_phase=none
complete=1
