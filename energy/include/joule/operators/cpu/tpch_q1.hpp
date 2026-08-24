#pragma once

#include "joule/tpch/lineitem.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace joule::operators::cpu {

struct TpchQ1Group {
    std::int64_t count{};
    std::int64_t sum_quantity_1e2{};
    std::int64_t sum_base_price_1e2{};
    std::int64_t sum_discount_price_1e4_usd{};
    std::int64_t sum_charge_1e6_usd{};
    std::int64_t sum_discount_1e2{};

    auto operator<=>(const TpchQ1Group&) const = default;
};

using TpchQ1Groups = std::array<TpchQ1Group, 6>;

struct TpchQ1Run {
    TpchQ1Groups groups{};
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] TpchQ1Groups tpch_q1_reference(tpch::LineitemView input);

class TpchQ1GroupBy {
public:
    TpchQ1GroupBy(tpch::LineitemView input, std::uint32_t thread_count = 0);
    ~TpchQ1GroupBy();
    TpchQ1GroupBy(TpchQ1GroupBy&&) noexcept;
    TpchQ1GroupBy& operator=(TpchQ1GroupBy&&) noexcept;
    TpchQ1GroupBy(const TpchQ1GroupBy&) = delete;
    TpchQ1GroupBy& operator=(const TpchQ1GroupBy&) = delete;
    [[nodiscard]] TpchQ1Run execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
