#pragma once

#include <compare>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "joule/tpch/lineitem.hpp"

namespace joule::operators::gpu {

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
    double gpu_time_ms{};
};

class Q6FilterProject {
   public:
    Q6FilterProject(const std::filesystem::path& metal_library, tpch::LineitemView input,
                    std::uint32_t threadgroup_width = 256);
    ~Q6FilterProject();
    Q6FilterProject(Q6FilterProject&&) noexcept;
    Q6FilterProject& operator=(Q6FilterProject&&) noexcept;
    Q6FilterProject(const Q6FilterProject&) = delete;
    Q6FilterProject& operator=(const Q6FilterProject&) = delete;

    [[nodiscard]] FilterProjectRun execute();
    [[nodiscard]] std::span<const FilterProjectRecord> output() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
