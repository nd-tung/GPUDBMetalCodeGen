#pragma once

#include "joule/tpch/lineitem.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace joule::operators::gpu {

struct RelationalRun {
    std::uint64_t output_count{};
    double host_time_ms{};
    double gpu_time_ms{};
};

class ScanCopyF32 {
public:
    explicit ScanCopyF32(std::span<const float> input);
    ~ScanCopyF32();
    ScanCopyF32(ScanCopyF32&&) noexcept;
    ScanCopyF32& operator=(ScanCopyF32&&) noexcept;
    ScanCopyF32(const ScanCopyF32&) = delete;
    ScanCopyF32& operator=(const ScanCopyF32&) = delete;
    [[nodiscard]] RelationalRun execute();
    [[nodiscard]] std::span<const float> output() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Q6FilterMaterialize {
public:
    Q6FilterMaterialize(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        std::uint32_t threadgroup_width = 256);
    ~Q6FilterMaterialize();
    Q6FilterMaterialize(Q6FilterMaterialize&&) noexcept;
    Q6FilterMaterialize& operator=(Q6FilterMaterialize&&) noexcept;
    Q6FilterMaterialize(const Q6FilterMaterialize&) = delete;
    Q6FilterMaterialize& operator=(const Q6FilterMaterialize&) = delete;
    [[nodiscard]] RelationalRun execute();
    [[nodiscard]] std::span<const std::uint32_t> output() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
