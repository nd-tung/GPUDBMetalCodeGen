#pragma once

#include "joule/operators/cpu/tpch_q1.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace joule::operators::gpu {

struct TpchQ1Run {
    cpu::TpchQ1Groups groups{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class TpchQ1GroupBy {
public:
    TpchQ1GroupBy(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        std::uint32_t threadgroup_width = 256);
    ~TpchQ1GroupBy();
    TpchQ1GroupBy(TpchQ1GroupBy&&) noexcept;
    TpchQ1GroupBy& operator=(TpchQ1GroupBy&&) noexcept;
    TpchQ1GroupBy(const TpchQ1GroupBy&) = delete;
    TpchQ1GroupBy& operator=(const TpchQ1GroupBy&) = delete;
    [[nodiscard]] TpchQ1Run execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
