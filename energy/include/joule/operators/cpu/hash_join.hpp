#pragma once

#include "joule/tpch/lineitem.hpp"
#include "joule/tpch/part.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace joule::operators::cpu {

struct HashBuildRun {
    std::uint64_t entry_count{};
    std::uint64_t promo_entry_count{};
    double host_time_ms{};
    double compute_time_ms{};
};

struct HashBuildVerification {
    bool valid{};
    std::uint64_t entry_count{};
    std::uint64_t promo_entry_count{};
    std::uint64_t key_sum{};

    auto operator<=>(const HashBuildVerification&) const = default;
};

struct HashProbeCountRun {
    std::uint64_t match_count{};
    std::uint64_t promo_match_count{};
    double host_time_ms{};
    double compute_time_ms{};
};

struct HashMatchRecord {
    std::uint32_t probe_row{};
    std::uint32_t promo{};

    auto operator<=>(const HashMatchRecord&) const = default;
};

struct HashProbeMaterializeRun {
    std::uint64_t output_count{};
    double host_time_ms{};
    double compute_time_ms{};
};

// Logical reference results used to validate both CPU and Metal primitives.
[[nodiscard]] HashBuildVerification part_hash_build_reference(tpch::PartView part);
[[nodiscard]] HashProbeCountRun part_hash_probe_count_reference(
    tpch::LineitemView probe,
    tpch::PartView build);
[[nodiscard]] std::vector<HashMatchRecord> part_hash_probe_materialize_reference(
    tpch::LineitemView probe,
    tpch::PartView build);

// Clears and builds an open-addressed, unique-key part hash table on every
// execute(). Both phases are inside the reported timing interval.
class PartHashBuild {
public:
    PartHashBuild(tpch::PartView build, std::uint32_t thread_count = 0);
    ~PartHashBuild();
    PartHashBuild(PartHashBuild&&) noexcept;
    PartHashBuild& operator=(PartHashBuild&&) noexcept;
    PartHashBuild(const PartHashBuild&) = delete;
    PartHashBuild& operator=(const PartHashBuild&) = delete;

    [[nodiscard]] HashBuildRun execute();
    // Verification is intentionally outside execute() and therefore outside
    // the benchmark timing interval.
    [[nodiscard]] HashBuildVerification verify() const;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The immutable hash table is prepared in the constructor, outside execute().
class PartHashProbeCount {
public:
    PartHashProbeCount(
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t thread_count = 0);
    ~PartHashProbeCount();
    PartHashProbeCount(PartHashProbeCount&&) noexcept;
    PartHashProbeCount& operator=(PartHashProbeCount&&) noexcept;
    PartHashProbeCount(const PartHashProbeCount&) = delete;
    PartHashProbeCount& operator=(const PartHashProbeCount&) = delete;

    [[nodiscard]] HashProbeCountRun execute();
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Uses a two-pass deterministic compaction. Records are stable in probe input
// order even when execute() uses multiple worker threads.
class PartHashProbeMaterialize {
public:
    PartHashProbeMaterialize(
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t thread_count = 0);
    ~PartHashProbeMaterialize();
    PartHashProbeMaterialize(PartHashProbeMaterialize&&) noexcept;
    PartHashProbeMaterialize& operator=(PartHashProbeMaterialize&&) noexcept;
    PartHashProbeMaterialize(const PartHashProbeMaterialize&) = delete;
    PartHashProbeMaterialize& operator=(const PartHashProbeMaterialize&) = delete;

    [[nodiscard]] HashProbeMaterializeRun execute();
    [[nodiscard]] std::span<const HashMatchRecord> output() const noexcept;
    [[nodiscard]] std::uint32_t thread_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::operators::cpu
