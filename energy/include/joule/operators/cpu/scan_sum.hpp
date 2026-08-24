#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace joule::operators::cpu {

enum class ScanSumKernel {
    scalar,
    parallel,
    parallel_simd,
};

struct ScanSumConfig {
    ScanSumKernel kernel{ScanSumKernel::parallel_simd};
    // Zero selects the performance-core count on Apple Silicon, falling back
    // to all logical CPU cores when performance levels are unavailable.
    std::uint32_t thread_count{0};
};

struct ScanSumRun {
    std::int64_t sum{};
    double host_time_ms{};
    double compute_time_ms{};
};

// Scalar reference implementation. The benchmark uses this outside its timed
// region to establish the result expected from every CPU and GPU variant.
[[nodiscard]] std::int64_t scan_sum_i32(std::span<const std::int32_t> input);

class ScanSum {
public:
    explicit ScanSum(
        std::span<const std::int32_t> input,
        ScanSumConfig config = {});
    ~ScanSum();

    ScanSum(ScanSum&&) noexcept;
    ScanSum& operator=(ScanSum&&) noexcept;
    ScanSum(const ScanSum&) = delete;
    ScanSum& operator=(const ScanSum&) = delete;

    [[nodiscard]] ScanSumRun execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
