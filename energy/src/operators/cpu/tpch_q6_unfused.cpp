#include "joule/operators/cpu/tpch_q6_unfused.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#endif

namespace joule::operators::cpu {
namespace {

constexpr std::int32_t date_begin = 19'940'101;
constexpr std::int32_t date_end = 19'950'101;
constexpr float discount_begin = 0.05F;
constexpr float discount_end = 0.07F;
constexpr float quantity_end = 24.0F;

[[nodiscard]] bool qualifies(tpch::LineitemView input, std::size_t row) noexcept {
    return input.ship_date_yyyymmdd[row] >= date_begin &&
           input.ship_date_yyyymmdd[row] < date_end &&
           input.discount[row] >= discount_begin &&
           input.discount[row] <= discount_end &&
           input.quantity[row] < quantity_end;
}

[[nodiscard]] std::int64_t scaled_revenue(
    tpch::LineitemView input,
    std::size_t row) noexcept {
    const auto price_cents = static_cast<std::int64_t>(
        std::round(input.extended_price[row] * 100.0F));
    const auto discount_hundredths = static_cast<std::int64_t>(
        std::round(input.discount[row] * 100.0F));
    return price_cents * discount_hundredths;
}

void materialize_bitmap_range(
    tpch::LineitemView input,
    std::span<std::uint32_t> bitmap,
    std::size_t word_begin,
    std::size_t word_end) {
    for (auto word = word_begin; word < word_end; ++word) {
        const auto row_begin = word * 32;
        const auto row_end = std::min(row_begin + 32, input.row_count());
        std::uint32_t bits = 0;
        for (auto row = row_begin; row < row_end; ++row) {
            bits |= static_cast<std::uint32_t>(qualifies(input, row))
                    << (row - row_begin);
        }
        bitmap[word] = bits;
    }
}

struct PartialResult {
    std::uint64_t count{};
    std::int64_t revenue{};
};

[[nodiscard]] PartialResult aggregate_bitmap_range(
    tpch::LineitemView input,
    std::span<const std::uint32_t> bitmap,
    std::size_t word_begin,
    std::size_t word_end) {
    PartialResult result;
    for (auto word = word_begin; word < word_end; ++word) {
        auto bits = bitmap[word];
        result.count += std::popcount(bits);
        while (bits != 0) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(bits));
            const auto row = word * 32 + bit;
            result.revenue += scaled_revenue(input, row);
            bits &= bits - 1;
        }
    }
    return result;
}

[[nodiscard]] std::uint32_t default_worker_count() {
#if defined(__APPLE__)
    std::uint32_t logical_threads = 0;
    std::size_t size = sizeof(logical_threads);
    if (::sysctlbyname(
            "hw.logicalcpu",
            &logical_threads,
            &size,
            nullptr,
            0) == 0 &&
        logical_threads > 0) {
        return logical_threads;
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

void validate_input(tpch::LineitemView input) {
    const auto rows = input.row_count();
    if (rows == 0 || input.extended_price.size() != rows ||
        input.discount.size() != rows || input.ship_date_yyyymmdd.size() != rows) {
        throw std::invalid_argument(
            "unfused TPC-H Q6 columns must be non-empty and equally sized");
    }
    if (rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "unfused TPC-H Q6 supports at most 2^32-1 rows");
    }
}

struct alignas(64) WorkerResult {
    PartialResult value;
    std::chrono::steady_clock::time_point start;
};

}  // namespace

struct TpchQ6Unfused::Impl {
    tpch::LineitemView input;
    std::uint32_t thread_count_value{};
    std::vector<std::uint32_t> bitmap_storage;
    std::unique_ptr<detail::DynamicChunkCursor> materialize_chunks;
    std::unique_ptr<detail::DynamicChunkCursor> aggregate_chunks;
    std::vector<WorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable stage_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t stage_a_completed{};
    std::uint32_t completed_workers{};
    bool stopping{};

    Impl(tpch::LineitemView requested_input, TpchQ6UnfusedConfig config)
        : input(requested_input) {
        validate_input(input);
        bitmap_storage.resize((input.row_count() + 31) / 32);
        const auto requested_threads =
            config.thread_count == 0 ? default_worker_count() : config.thread_count;
        thread_count_value = static_cast<std::uint32_t>(
            std::min<std::size_t>(requested_threads, bitmap_storage.size()));
        if (thread_count_value == 0) {
            throw std::invalid_argument("CPU thread count must be greater than zero");
        }
        materialize_chunks = std::make_unique<detail::DynamicChunkCursor>(
            bitmap_storage.size(), thread_count_value, 64);
        aggregate_chunks = std::make_unique<detail::DynamicChunkCursor>(
            bitmap_storage.size(), thread_count_value, 64);

        results.resize(thread_count_value);
        workers.reserve(thread_count_value);
        try {
            for (std::uint32_t worker = 0; worker < thread_count_value; ++worker) {
                workers.emplace_back([this, worker] { worker_loop(worker); });
            }
        } catch (...) {
            {
                std::lock_guard lock(state_mutex);
                stopping = true;
            }
            start_condition.notify_all();
            stage_condition.notify_all();
            workers.clear();
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(state_mutex);
            stopping = true;
        }
        start_condition.notify_all();
        stage_condition.notify_all();
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
#if defined(__APPLE__)
        static_cast<void>(::pthread_set_qos_class_self_np(
            QOS_CLASS_USER_INITIATED, 0));
#endif
        std::uint64_t observed_generation = 0;
        while (true) {
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] {
                    return stopping || generation != observed_generation;
                });
                if (stopping) {
                    return;
                }
                observed_generation = generation;
            }

            auto& output = results[worker];
            output.start = std::chrono::steady_clock::now();

            while (const auto chunk = materialize_chunks->claim()) {
                materialize_bitmap_range(
                    input, bitmap_storage, chunk.begin, chunk.end);
            }

            {
                std::unique_lock lock(state_mutex);
                ++stage_a_completed;
                if (stage_a_completed == thread_count_value) {
                    stage_condition.notify_all();
                } else {
                    stage_condition.wait(lock, [&] {
                        return stopping ||
                               stage_a_completed == thread_count_value;
                    });
                    if (stopping) {
                        return;
                    }
                }
            }

            PartialResult local;
            while (const auto chunk = aggregate_chunks->claim()) {
                const auto partial = aggregate_bitmap_range(
                    input, bitmap_storage, chunk.begin, chunk.end);
                local.count += partial.count;
                local.revenue += partial.revenue;
            }
            output.value = local;
            {
                std::lock_guard lock(state_mutex);
                ++completed_workers;
            }
            completion_condition.notify_one();
        }
    }

    [[nodiscard]] TpchQ6Result execute() {
        std::lock_guard execution_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(state_mutex);
            stage_a_completed = 0;
            completed_workers = 0;
            materialize_chunks->reset();
            aggregate_chunks->reset();
            ++generation;
        }
        start_condition.notify_all();
        {
            std::unique_lock lock(state_mutex);
            completion_condition.wait(lock, [&] {
                return completed_workers == thread_count_value;
            });
        }

        auto compute_start = results.front().start;
        TpchQ6Result combined;
        for (const auto& result : results) {
            compute_start = std::min(compute_start, result.start);
            combined.match_count += result.value.count;
            combined.revenue_1e4_usd += result.value.revenue;
        }
        const auto host_end = std::chrono::steady_clock::now();
        combined.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start)
                .count();
        combined.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start)
                .count();
        return combined;
    }
};

TpchQ6Unfused::TpchQ6Unfused(
    tpch::LineitemView input,
    TpchQ6UnfusedConfig config)
    : impl_(std::make_unique<Impl>(input, config)) {}

TpchQ6Unfused::~TpchQ6Unfused() = default;
TpchQ6Unfused::TpchQ6Unfused(TpchQ6Unfused&&) noexcept = default;
TpchQ6Unfused& TpchQ6Unfused::operator=(TpchQ6Unfused&&) noexcept = default;

TpchQ6Result TpchQ6Unfused::execute() {
    return impl_->execute();
}

std::span<const std::uint32_t> TpchQ6Unfused::bitmap() const noexcept {
    return impl_->bitmap_storage;
}

std::uint32_t TpchQ6Unfused::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
