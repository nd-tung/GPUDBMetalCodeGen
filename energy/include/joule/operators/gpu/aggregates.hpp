#pragma once

#include "joule/operators/cpu/aggregates.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace joule::operators::gpu {

enum class PriceAggregateReduction {
    threadgroup_tree,
    simdgroup,
};

struct PriceAggregateResult {
    std::uint64_t count{};
    std::int64_t sum_price_cents{};
    std::int64_t min_price_cents{};
    std::int64_t max_price_cents{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class PriceAggregate {
public:
    PriceAggregate(
        const std::filesystem::path& metal_library,
        std::span<const float> input,
        cpu::PriceAggregateMode mode,
        std::uint32_t threadgroup_width = 256,
        PriceAggregateReduction reduction =
            PriceAggregateReduction::simdgroup);
    ~PriceAggregate();
    PriceAggregate(PriceAggregate&&) noexcept;
    PriceAggregate& operator=(PriceAggregate&&) noexcept;
    PriceAggregate(const PriceAggregate&) = delete;
    PriceAggregate& operator=(const PriceAggregate&) = delete;
    [[nodiscard]] PriceAggregateResult execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct GroupByCountRun {
    double host_time_ms{};
    double gpu_time_ms{};
};

enum class GroupByCountStrategy {
    global_atomic,
    bounded_threadgroup,
};

class PartKeyGroupCount {
public:
    PartKeyGroupCount(
        const std::filesystem::path& metal_library,
        std::span<const std::int32_t> keys,
        std::uint32_t group_count,
        std::uint32_t threadgroup_width = 256,
        GroupByCountStrategy strategy =
            GroupByCountStrategy::global_atomic);
    ~PartKeyGroupCount();
    PartKeyGroupCount(PartKeyGroupCount&&) noexcept;
    PartKeyGroupCount& operator=(PartKeyGroupCount&&) noexcept;
    PartKeyGroupCount(const PartKeyGroupCount&) = delete;
    PartKeyGroupCount& operator=(const PartKeyGroupCount&) = delete;
    [[nodiscard]] GroupByCountRun execute();
    [[nodiscard]] std::span<const std::uint32_t> output() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;
    [[nodiscard]] std::uint32_t group_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
