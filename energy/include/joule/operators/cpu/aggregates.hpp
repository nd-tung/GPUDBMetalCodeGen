#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace joule::operators::cpu {

enum class PriceAggregateMode {
    sum,
    minmax,
    stats,
};

struct PriceAggregateResult {
    std::uint64_t count{};
    std::int64_t sum_price_cents{};
    std::int64_t min_price_cents{};
    std::int64_t max_price_cents{};
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] PriceAggregateResult price_aggregate_reference(
    std::span<const float> input,
    PriceAggregateMode mode);

class PriceAggregate {
public:
    PriceAggregate(
        std::span<const float> input,
        PriceAggregateMode mode,
        std::uint32_t thread_count = 0);
    ~PriceAggregate();
    PriceAggregate(PriceAggregate&&) noexcept;
    PriceAggregate& operator=(PriceAggregate&&) noexcept;
    PriceAggregate(const PriceAggregate&) = delete;
    PriceAggregate& operator=(const PriceAggregate&) = delete;
    [[nodiscard]] PriceAggregateResult execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct GroupByCountRun {
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] std::vector<std::uint32_t> part_key_group_count_reference(
    std::span<const std::int32_t> keys,
    std::uint32_t group_count);

class PartKeyGroupCount {
public:
    PartKeyGroupCount(
        std::span<const std::int32_t> keys,
        std::uint32_t group_count,
        std::uint32_t thread_count = 0);
    ~PartKeyGroupCount();
    PartKeyGroupCount(PartKeyGroupCount&&) noexcept;
    PartKeyGroupCount& operator=(PartKeyGroupCount&&) noexcept;
    PartKeyGroupCount(const PartKeyGroupCount&) = delete;
    PartKeyGroupCount& operator=(const PartKeyGroupCount&) = delete;
    [[nodiscard]] GroupByCountRun execute();
    [[nodiscard]] std::span<const std::uint32_t> output() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;
    [[nodiscard]] std::uint32_t group_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
