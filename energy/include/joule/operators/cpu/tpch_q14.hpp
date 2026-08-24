#pragma once

#include "joule/tpch/lineitem.hpp"
#include "joule/tpch/part.hpp"

#include <cstdint>
#include <memory>

namespace joule::operators::cpu {

struct TpchQ14Result {
    std::int64_t promo_revenue_1e4_usd{};
    std::int64_t total_revenue_1e4_usd{};
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] TpchQ14Result tpch_q14_reference(
    tpch::LineitemView lineitem,
    tpch::PartView part);

class TpchQ14HashJoin {
public:
    TpchQ14HashJoin(
        tpch::LineitemView lineitem,
        tpch::PartView part,
        std::uint32_t thread_count = 0);
    ~TpchQ14HashJoin();
    TpchQ14HashJoin(TpchQ14HashJoin&&) noexcept;
    TpchQ14HashJoin& operator=(TpchQ14HashJoin&&) noexcept;
    TpchQ14HashJoin(const TpchQ14HashJoin&) = delete;
    TpchQ14HashJoin& operator=(const TpchQ14HashJoin&) = delete;
    [[nodiscard]] TpchQ14Result execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
