#pragma once

#include "joule/operators/cpu/hash_join.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace joule::operators::gpu {

struct HashBuildRun {
    std::uint64_t entry_count{};
    std::uint64_t promo_entry_count{};
    double host_time_ms{};
    double gpu_time_ms{};
};

struct HashProbeCountRun {
    std::uint64_t match_count{};
    std::uint64_t promo_match_count{};
    double host_time_ms{};
    double gpu_time_ms{};
};

struct HashProbeMaterializeRun {
    std::uint64_t output_count{};
    double host_time_ms{};
    double gpu_time_ms{};
};

using HashBuildVerification = cpu::HashBuildVerification;
using HashMatchRecord = cpu::HashMatchRecord;

// execute() performs the GPU clear, atomic-CAS hash build, and a bounded
// threadgroup/GPU reduction of build statistics. Resource allocation and
// pipeline setup happen in the constructor.
class PartHashBuild {
public:
    PartHashBuild(
        const std::filesystem::path& metal_library,
        tpch::PartView build,
        std::uint32_t threadgroup_width = 256);
    ~PartHashBuild();
    PartHashBuild(PartHashBuild&&) noexcept;
    PartHashBuild& operator=(PartHashBuild&&) noexcept;
    PartHashBuild(const PartHashBuild&) = delete;
    PartHashBuild& operator=(const PartHashBuild&) = delete;

    [[nodiscard]] HashBuildRun execute();
    // Reads the completed shared table after timing and checks every build row.
    [[nodiscard]] HashBuildVerification verify() const;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The immutable table is prepared and uploaded in the constructor.
class PartHashProbeCount {
public:
    PartHashProbeCount(
        const std::filesystem::path& metal_library,
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t threadgroup_width = 256);
    ~PartHashProbeCount();
    PartHashProbeCount(PartHashProbeCount&&) noexcept;
    PartHashProbeCount& operator=(PartHashProbeCount&&) noexcept;
    PartHashProbeCount(const PartHashProbeCount&) = delete;
    PartHashProbeCount& operator=(const PartHashProbeCount&) = delete;

    [[nodiscard]] HashProbeCountRun execute();
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Stable compaction uses per-block counts, a GPU prefix scan, then a
// deterministic scatter. The output buffer is allocated once in the
// constructor and reused by execute().
class PartHashProbeMaterialize {
public:
    PartHashProbeMaterialize(
        const std::filesystem::path& metal_library,
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t threadgroup_width = 256);
    ~PartHashProbeMaterialize();
    PartHashProbeMaterialize(PartHashProbeMaterialize&&) noexcept;
    PartHashProbeMaterialize& operator=(PartHashProbeMaterialize&&) noexcept;
    PartHashProbeMaterialize(const PartHashProbeMaterialize&) = delete;
    PartHashProbeMaterialize& operator=(const PartHashProbeMaterialize&) = delete;

    [[nodiscard]] HashProbeMaterializeRun execute();
    [[nodiscard]] std::span<const HashMatchRecord> output() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] std::uint32_t execution_width() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::gpu
