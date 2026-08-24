#pragma once

#include "joule/operators/cpu/topk.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace joule::operators::gpu {

struct TopKRun {
    cpu::Top10 rows{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class OrdersTopK {
public:
    OrdersTopK(
        const std::filesystem::path& metal_library,
        tpch::OrdersView input,
        std::uint32_t threadgroup_width = 256);
    ~OrdersTopK();
    OrdersTopK(OrdersTopK&&) noexcept;
    OrdersTopK& operator=(OrdersTopK&&) noexcept;
    OrdersTopK(const OrdersTopK&) = delete;
    OrdersTopK& operator=(const OrdersTopK&) = delete;
    [[nodiscard]] TopKRun execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
