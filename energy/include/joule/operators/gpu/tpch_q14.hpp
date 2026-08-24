#pragma once

#include "joule/operators/cpu/tpch_q14.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace joule::operators::gpu {

struct TpchQ14Result {
    std::int64_t promo_revenue_1e4_usd{};
    std::int64_t total_revenue_1e4_usd{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class TpchQ14HashJoin {
public:
    TpchQ14HashJoin(
        const std::filesystem::path& metal_library,
        tpch::LineitemView lineitem,
        tpch::PartView part,
        std::uint32_t threadgroup_width = 256);
    ~TpchQ14HashJoin();
    TpchQ14HashJoin(TpchQ14HashJoin&&) noexcept;
    TpchQ14HashJoin& operator=(TpchQ14HashJoin&&) noexcept;
    TpchQ14HashJoin(const TpchQ14HashJoin&) = delete;
    TpchQ14HashJoin& operator=(const TpchQ14HashJoin&) = delete;
    [[nodiscard]] TpchQ14Result execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
