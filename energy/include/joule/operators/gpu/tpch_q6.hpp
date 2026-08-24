#pragma once

#include "joule/operators/cpu/tpch_q6.hpp"
#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace joule::operators::gpu {

struct TpchQ6Config {
    cpu::TpchQ6Mode mode{cpu::TpchQ6Mode::revenue};
    std::uint32_t threadgroup_width{256};
};

struct TpchQ6Result {
    std::uint64_t match_count{};
    std::int64_t revenue_1e4_usd{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class TpchQ6 {
public:
    TpchQ6(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        TpchQ6Config config = {});
    ~TpchQ6();

    TpchQ6(TpchQ6&&) noexcept;
    TpchQ6& operator=(TpchQ6&&) noexcept;
    TpchQ6(const TpchQ6&) = delete;
    TpchQ6& operator=(const TpchQ6&) = delete;

    [[nodiscard]] TpchQ6Result execute();
    [[nodiscard]] std::span<const std::uint32_t> bitmap() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;
    [[nodiscard]] std::uint32_t max_threads_per_threadgroup() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
