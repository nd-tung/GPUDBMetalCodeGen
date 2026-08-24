#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace joule::metrics {

struct ShellyConfig {
    std::string host;
    std::string interface;
    std::uint16_t port{80};
    std::uint32_t sample_rate_ms{1'000};
    std::uint32_t timeout_ms{2'000};
    std::uint32_t attempts{3};
    std::filesystem::path raw_trace_path;
    std::optional<std::string> expected_device_id;
};

struct ShellySwitchStatus {
    std::optional<double> apower_w;
    std::optional<double> voltage_v;
    std::optional<double> current_a;
    std::optional<double> aenergy_total_wh;
};

struct ShellySample {
    std::uint64_t sequence{};
    std::chrono::steady_clock::time_point scheduled_at;
    std::chrono::steady_clock::time_point request_started_at;
    std::chrono::steady_clock::time_point response_received_at;
    std::uint32_t attempt_count{};
    bool success{};
    ShellySwitchStatus status;
    std::string error;

    [[nodiscard]] std::chrono::steady_clock::time_point measurement_time() const;
};

struct ShellyWindowSummary {
    std::size_t request_count{};
    std::size_t sample_count{};
    std::size_t failure_count{};
    double sampled_time_s{};
    bool counter_monotonic{true};
    std::optional<double> counter_start_wh;
    std::optional<double> counter_end_wh;
    std::optional<double> counter_delta_wh;
    std::optional<double> energy_j;
    std::optional<double> average_power_w;
    std::optional<double> sampled_apower_energy_j;
    std::optional<double> sampled_average_power_w;
};

struct ShellyCollection {
    std::string device_id;
    std::vector<ShellySample> samples;
    std::filesystem::path raw_trace_path;
};

struct ShellyMeasurementResult {
    std::string host;
    std::string interface;
    std::uint16_t port{};
    std::uint32_t sample_rate_ms{};
    std::uint32_t timeout_ms{};
    std::uint32_t attempts{};
    std::string device_id;
    std::optional<std::string> expected_device_id;
    std::optional<bool> device_id_match;
    std::filesystem::path raw_trace_path;
    ShellyWindowSummary baseline;
    ShellyWindowSummary workload;
    std::optional<double> dynamic_energy_j;
    std::optional<double> dynamic_sampled_apower_energy_j;
};

[[nodiscard]] std::optional<ShellySwitchStatus> parse_shelly_switch_status(
    std::string_view json);

[[nodiscard]] std::optional<std::string> parse_shelly_device_id(
    std::string_view json);

[[nodiscard]] ShellyWindowSummary summarize_shelly_samples(
    const std::vector<ShellySample>& samples,
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end);

class ShellySampler {
public:
    explicit ShellySampler(ShellyConfig config);
    ~ShellySampler();

    ShellySampler(const ShellySampler&) = delete;
    ShellySampler& operator=(const ShellySampler&) = delete;
    ShellySampler(ShellySampler&&) = delete;
    ShellySampler& operator=(ShellySampler&&) = delete;

    void start();
    [[nodiscard]] bool wait_for_first_success(std::chrono::milliseconds timeout);
    [[nodiscard]] bool wait_for_success_after(
        std::chrono::steady_clock::time_point boundary,
        std::chrono::milliseconds timeout);
    [[nodiscard]] ShellyCollection stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::metrics
