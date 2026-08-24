#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace joule::operators::cpu::detail {

struct WorkChunk {
    std::size_t begin{};
    std::size_t end{};

    [[nodiscard]] explicit operator bool() const noexcept { return begin < end; }
};

// A small, reusable cursor for heterogeneous CPU worker pools.
//
// Construct it once after the unit and worker counts are known. Before publishing
// each worker-pool generation, call reset(); every worker then repeatedly calls
// claim() and accumulates its own partial result until an empty chunk is returned.
// Ranges are disjoint, and all non-final boundaries are multiples of alignment.
// The default targets about 16 claims per worker: enough to let faster P-cores
// consume more work without putting one atomic operation in the inner row loop.
class DynamicChunkCursor {
public:
    static constexpr std::size_t default_chunks_per_worker = 16;

    DynamicChunkCursor(
        std::size_t unit_count,
        std::uint32_t worker_count,
        std::size_t alignment = 1,
        std::size_t chunks_per_worker = default_chunks_per_worker)
        : unit_count_(unit_count),
          chunk_size_(choose_chunk_size(
              unit_count, worker_count, alignment, chunks_per_worker)) {}

    DynamicChunkCursor(const DynamicChunkCursor&) = delete;
    DynamicChunkCursor& operator=(const DynamicChunkCursor&) = delete;

    // reset() is only valid while workers are idle. The pool's generation mutex
    // supplies publication ordering, so the cursor itself only needs relaxed RMWs.
    void reset() noexcept { next_.store(0, std::memory_order_relaxed); }

    [[nodiscard]] WorkChunk claim() noexcept {
        const auto begin = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
        if (begin >= unit_count_) return {};
        const auto remaining = unit_count_ - begin;
        return {begin, begin + (remaining < chunk_size_ ? remaining : chunk_size_)};
    }

    [[nodiscard]] std::size_t chunk_size() const noexcept { return chunk_size_; }

private:
    [[nodiscard]] static std::size_t choose_chunk_size(
        std::size_t unit_count,
        std::uint32_t worker_count,
        std::size_t alignment,
        std::size_t chunks_per_worker) {
        if (worker_count == 0 || alignment == 0 || chunks_per_worker == 0) {
            throw std::invalid_argument(
                "dynamic chunk worker count, alignment, and chunk target must be non-zero");
        }

        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto workers = static_cast<std::size_t>(worker_count);
        const auto target_chunks = workers > maximum / chunks_per_worker
            ? maximum
            : workers * chunks_per_worker;
        const auto unaligned = unit_count == 0
            ? alignment
            : unit_count / target_chunks + (unit_count % target_chunks != 0);
        const auto aligned_units = unaligned / alignment + (unaligned % alignment != 0);
        if (aligned_units > maximum / alignment) {
            throw std::overflow_error("dynamic chunk size overflow");
        }
        return aligned_units * alignment;
    }

    std::size_t unit_count_{};
    std::size_t chunk_size_{1};
    alignas(64) std::atomic<std::size_t> next_{0};
};

}  // namespace joule::operators::cpu::detail
