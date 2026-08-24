#pragma once

#include <compare>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "joule/tpch/lineitem.hpp"

namespace joule::operators::cpu {

struct alignas(8) FilterProjectRecord {
    std::uint32_t row_id{};
    std::int32_t part_key{};
    std::int64_t revenue_1e4_usd{};

    auto operator<=>(const FilterProjectRecord&) const = default;
};

static_assert(sizeof(FilterProjectRecord) == 16);

struct FilterProjectRun {
    std::uint64_t output_count{};
    double host_time_ms{};
    double compute_time_ms{};
};

[[nodiscard]] std::vector<FilterProjectRecord> q6_filter_project_reference(
    tpch::LineitemView input);

class Q6FilterProject {
   public:
    Q6FilterProject(tpch::LineitemView input, std::uint32_t thread_count = 0);
    ~Q6FilterProject();
    Q6FilterProject(Q6FilterProject&&) noexcept;
    Q6FilterProject& operator=(Q6FilterProject&&) noexcept;
    Q6FilterProject(const Q6FilterProject&) = delete;
    Q6FilterProject& operator=(const Q6FilterProject&) = delete;

    [[nodiscard]] FilterProjectRun execute();
    [[nodiscard]] std::span<const FilterProjectRecord> output() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
