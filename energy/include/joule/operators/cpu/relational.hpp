#pragma once

#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace joule::operators::cpu {

struct RelationalRun {
    std::uint64_t output_count{};
    double host_time_ms{};
    double compute_time_ms{};
};

class ScanCopyF32 {
public:
    ScanCopyF32(std::span<const float> input, std::uint32_t thread_count = 0);
    ~ScanCopyF32();
    ScanCopyF32(ScanCopyF32&&) noexcept;
    ScanCopyF32& operator=(ScanCopyF32&&) noexcept;
    ScanCopyF32(const ScanCopyF32&) = delete;
    ScanCopyF32& operator=(const ScanCopyF32&) = delete;

    [[nodiscard]] RelationalRun execute();
    [[nodiscard]] std::span<const float> output() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::vector<std::uint32_t> q6_materialize_reference(
    tpch::LineitemView input);

class Q6FilterMaterialize {
public:
    Q6FilterMaterialize(tpch::LineitemView input, std::uint32_t thread_count = 0);
    ~Q6FilterMaterialize();
    Q6FilterMaterialize(Q6FilterMaterialize&&) noexcept;
    Q6FilterMaterialize& operator=(Q6FilterMaterialize&&) noexcept;
    Q6FilterMaterialize(const Q6FilterMaterialize&) = delete;
    Q6FilterMaterialize& operator=(const Q6FilterMaterialize&) = delete;

    [[nodiscard]] RelationalRun execute();
    [[nodiscard]] std::span<const std::uint32_t> output() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
