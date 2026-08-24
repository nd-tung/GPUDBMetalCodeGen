#pragma once

#include "joule/operators/cpu/tpch_q6.hpp"
#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace joule::operators::cpu {

struct TpchQ6UnfusedConfig {
    // Zero selects all logical CPU cores, including efficiency cores.
    std::uint32_t thread_count{0};
};

// Executes Q6 as two materialized stages:
//   A. predicate scan -> one exact bit per input row
//   B. bitmap scan -> exact match count and fixed-point revenue
class TpchQ6Unfused {
public:
    explicit TpchQ6Unfused(
        tpch::LineitemView input,
        TpchQ6UnfusedConfig config = {});
    ~TpchQ6Unfused();

    TpchQ6Unfused(TpchQ6Unfused&&) noexcept;
    TpchQ6Unfused& operator=(TpchQ6Unfused&&) noexcept;
    TpchQ6Unfused(const TpchQ6Unfused&) = delete;
    TpchQ6Unfused& operator=(const TpchQ6Unfused&) = delete;

    [[nodiscard]] TpchQ6Result execute();
    [[nodiscard]] std::span<const std::uint32_t> bitmap() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
