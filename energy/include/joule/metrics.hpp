#pragma once

#include "joule/shelly.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace joule::metrics {

struct PowerSample {
    std::uint64_t elapsed_ns{};
    std::optional<double> cpu_power_mw;
    std::optional<double> gpu_power_mw;
    std::optional<double> ane_power_mw;
    std::optional<double> combined_power_mw;
};

struct PowerSummary {
    std::size_t sample_count{};
    std::size_t cpu_sample_count{};
    std::size_t gpu_sample_count{};
    std::size_t ane_sample_count{};
    std::size_t total_sample_count{};
    double sampled_time_s{};
    std::optional<double> cpu_energy_j;
    std::optional<double> gpu_energy_j;
    std::optional<double> ane_energy_j;
    std::optional<double> soc_energy_j;
    std::optional<double> total_energy_j;
    std::optional<double> average_cpu_power_w;
    std::optional<double> average_gpu_power_w;
    std::optional<double> average_ane_power_w;
    std::optional<double> average_soc_power_w;
    std::optional<double> average_total_power_w;
};

struct MeasurementConfig {
    std::uint32_t sample_rate_ms{100};
    std::uint32_t baseline_ms{1'000};
    std::filesystem::path raw_trace_path;
    bool use_sudo{true};
    bool cooperative_boundary{false};
    std::uint32_t cooperative_timeout_ms{};
    std::optional<ShellyConfig> shelly;
};

struct MeasurementResult {
    int command_exit_code{};
    double command_wall_time_ms{};
    PowerSummary baseline;
    PowerSummary workload;
    std::optional<double> dynamic_cpu_energy_j;
    std::optional<double> dynamic_gpu_energy_j;
    std::optional<double> dynamic_ane_energy_j;
    std::optional<double> dynamic_soc_energy_j;
    std::optional<double> dynamic_total_energy_j;
    std::optional<ShellyMeasurementResult> wall_power;
    std::filesystem::path raw_trace_path;
};

[[nodiscard]] std::vector<PowerSample> parse_powermetrics_plist_trace(
    std::string_view trace);

[[nodiscard]] std::vector<PowerSample> read_powermetrics_plist_trace(
    const std::filesystem::path& path);

[[nodiscard]] PowerSummary summarize_power_samples(
    const std::vector<PowerSample>& samples,
    std::size_t begin,
    std::size_t end);

[[nodiscard]] MeasurementResult measure_command(
    const MeasurementConfig& config,
    const std::vector<std::string>& command);

}  // namespace joule::metrics
