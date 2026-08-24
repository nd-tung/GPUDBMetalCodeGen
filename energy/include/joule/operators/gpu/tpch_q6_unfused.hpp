#pragma once

#include "joule/operators/gpu/tpch_q6.hpp"
#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace joule::operators::gpu {

struct TpchQ6UnfusedConfig {
    std::uint32_t threadgroup_width{256};
};

// Executes Q6 as two GPU stages separated by a fully materialized private
// bitmap. Every reduction pass remains on the GPU; only the final pair is
// host-visible.
class TpchQ6Unfused {
public:
    TpchQ6Unfused(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        TpchQ6UnfusedConfig config = {});
    ~TpchQ6Unfused();

    TpchQ6Unfused(TpchQ6Unfused&&) noexcept;
    TpchQ6Unfused& operator=(TpchQ6Unfused&&) noexcept;
    TpchQ6Unfused(const TpchQ6Unfused&) = delete;
    TpchQ6Unfused& operator=(const TpchQ6Unfused&) = delete;

    [[nodiscard]] TpchQ6Result execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;
    [[nodiscard]] std::uint32_t max_threads_per_threadgroup() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
