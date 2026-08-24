#pragma once

#include "joule/tpch/orders.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace joule::operators::cpu {

struct alignas(16) TopKEntry {
    std::int64_t total_price_cents{};
    std::int32_t order_key{};
    [[nodiscard]] bool operator==(const TopKEntry&) const = default;
};

using Top10 = std::array<TopKEntry, 10>;

struct TopKRun {
    Top10 rows{};
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] Top10 top10_reference(tpch::OrdersView input);

class OrdersTopK {
public:
    explicit OrdersTopK(tpch::OrdersView input, std::uint32_t thread_count = 0);
    ~OrdersTopK();
    OrdersTopK(OrdersTopK&&) noexcept;
    OrdersTopK& operator=(OrdersTopK&&) noexcept;
    OrdersTopK(const OrdersTopK&) = delete;
    OrdersTopK& operator=(const OrdersTopK&) = delete;
    [[nodiscard]] TopKRun execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
