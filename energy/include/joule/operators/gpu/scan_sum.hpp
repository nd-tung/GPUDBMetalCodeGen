#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace joule::operators::gpu {

enum class ScanSumKernel {
    baseline,
    multi_item,
    simdgroup,
    optimized = simdgroup,
};

struct ScanSumConfig {
    ScanSumKernel kernel{ScanSumKernel::simdgroup};
    std::uint32_t threadgroup_width{256};
};

struct ScanSumRun {
    std::int64_t sum{};
    double host_time_ms{};
    double gpu_time_ms{};
    std::uint32_t repetitions{1};
};

class ScanSum {
public:
    ScanSum(
        const std::filesystem::path& metal_library,
        std::span<const std::int32_t> input,
        ScanSumConfig config = {});
    ~ScanSum();

    ScanSum(ScanSum&&) noexcept;
    ScanSum& operator=(ScanSum&&) noexcept;
    ScanSum(const ScanSum&) = delete;
    ScanSum& operator=(const ScanSum&) = delete;

    [[nodiscard]] ScanSumRun execute();
    [[nodiscard]] ScanSumRun execute_batch(std::uint32_t repetitions);
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;
    [[nodiscard]] std::uint32_t max_threads_per_threadgroup() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
