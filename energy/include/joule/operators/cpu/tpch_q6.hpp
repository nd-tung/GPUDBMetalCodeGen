#pragma once

#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace joule::operators::cpu {

enum class TpchQ6Mode {
    filter_count,
    filter_bitmap,
    revenue,
};

struct TpchQ6Config {
    TpchQ6Mode mode{TpchQ6Mode::revenue};
    // Zero selects the performance-core count on Apple Silicon.
    std::uint32_t thread_count{0};
};

struct TpchQ6Result {
    std::uint64_t match_count{};
    // Exact fixed-point unit: extended-price cents * discount hundredths,
    // i.e. one unit is 0.0001 USD.
    std::int64_t revenue_1e4_usd{};
    double host_time_ms{};
    double compute_time_ms{};
};

struct TpchQ6Reference {
    std::uint64_t match_count{};
    std::int64_t revenue_1e4_usd{};
    std::vector<std::uint32_t> bitmap;
};

[[nodiscard]] TpchQ6Reference tpch_q6_reference(
    tpch::LineitemView input,
    bool materialize_bitmap);

class TpchQ6 {
public:
    explicit TpchQ6(tpch::LineitemView input, TpchQ6Config config = {});
    ~TpchQ6();

    TpchQ6(TpchQ6&&) noexcept;
    TpchQ6& operator=(TpchQ6&&) noexcept;
    TpchQ6(const TpchQ6&) = delete;
    TpchQ6& operator=(const TpchQ6&) = delete;

    [[nodiscard]] TpchQ6Result execute();
    [[nodiscard]] std::span<const std::uint32_t> bitmap() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
