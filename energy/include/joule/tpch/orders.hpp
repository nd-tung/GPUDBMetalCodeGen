#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace joule::tpch {

struct OrdersView {
    std::span<const std::int32_t> order_key;
    std::span<const float> total_price;
    [[nodiscard]] std::size_t row_count() const noexcept { return order_key.size(); }
};

class OrdersStore {
public:
    explicit OrdersStore(const std::filesystem::path& colbin_path);
    ~OrdersStore();
    OrdersStore(OrdersStore&&) noexcept;
    OrdersStore& operator=(OrdersStore&&) noexcept;
    OrdersStore(const OrdersStore&) = delete;
    OrdersStore& operator=(const OrdersStore&) = delete;
    [[nodiscard]] OrdersView view() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::tpch
