#include "joule/operators/cpu/aggregates.hpp"

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

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#endif

namespace joule::operators::cpu {
namespace {

[[nodiscard]] std::uint32_t default_worker_count() {
#if defined(__APPLE__)
    std::uint32_t count = 0;
    std::size_t size = sizeof(count);
    if (::sysctlbyname("hw.logicalcpu", &count, &size, nullptr, 0) == 0 &&
        count > 0) {
        return count;
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

void set_worker_qos() {
#if defined(__APPLE__)
    static_cast<void>(::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0));
#endif
}

[[nodiscard]] std::int64_t price_cents(float value) noexcept {
    return static_cast<std::int64_t>(std::round(value * 100.0F));
}

[[nodiscard]] PriceAggregateResult scan_prices(
    std::span<const float> input,
    PriceAggregateMode mode,
    std::size_t begin,
    std::size_t end) {
    PriceAggregateResult result;
    result.count = end - begin;
    if (mode == PriceAggregateMode::sum) {
        for (auto row = begin; row < end; ++row) {
            result.sum_price_cents += price_cents(input[row]);
        }
    } else if (mode == PriceAggregateMode::minmax) {
        result.min_price_cents = std::numeric_limits<std::int64_t>::max();
        result.max_price_cents = std::numeric_limits<std::int64_t>::min();
        for (auto row = begin; row < end; ++row) {
            const auto value = price_cents(input[row]);
            result.min_price_cents = std::min(result.min_price_cents, value);
            result.max_price_cents = std::max(result.max_price_cents, value);
        }
    } else {
        result.min_price_cents = std::numeric_limits<std::int64_t>::max();
        result.max_price_cents = std::numeric_limits<std::int64_t>::min();
        for (auto row = begin; row < end; ++row) {
            const auto value = price_cents(input[row]);
            result.sum_price_cents += value;
            result.min_price_cents = std::min(result.min_price_cents, value);
            result.max_price_cents = std::max(result.max_price_cents, value);
        }
    }
    return result;
}

struct alignas(64) PriceWorkerResult {
    PriceAggregateResult value;
    std::chrono::steady_clock::time_point start;
};

struct alignas(64) GroupWorkerResult {
    std::chrono::steady_clock::time_point start;
};

}  // namespace

PriceAggregateResult price_aggregate_reference(
    std::span<const float> input,
    PriceAggregateMode mode) {
    if (input.empty()) throw std::invalid_argument("price aggregate input must not be empty");
    return scan_prices(input, mode, 0, input.size());
}

struct PriceAggregate::Impl {
    std::span<const float> input;
    PriceAggregateMode mode;
    std::uint32_t thread_count_value{};
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<PriceWorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    bool stopping{};

    Impl(
        std::span<const float> requested_input,
        PriceAggregateMode requested_mode,
        std::uint32_t requested_threads)
        : input(requested_input), mode(requested_mode) {
        if (input.empty()) throw std::invalid_argument("price aggregate input must not be empty");
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_worker_count() : requested_threads,
            input.size()));
        chunks = std::make_unique<detail::DynamicChunkCursor>(
            input.size(), thread_count_value, 64);
        results.resize(thread_count_value);
        workers.reserve(thread_count_value);
        for (std::uint32_t worker = 0; worker < thread_count_value; ++worker) {
            workers.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(state_mutex);
            stopping = true;
        }
        start_condition.notify_all();
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
        set_worker_qos();
        std::uint64_t observed = 0;
        while (true) {
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] { return stopping || observed != generation; });
                if (stopping) return;
                observed = generation;
            }
            results[worker].start = std::chrono::steady_clock::now();
            PriceAggregateResult local;
            if (mode != PriceAggregateMode::sum) {
                local.min_price_cents = std::numeric_limits<std::int64_t>::max();
                local.max_price_cents = std::numeric_limits<std::int64_t>::min();
            }
            while (const auto chunk = chunks->claim()) {
                const auto partial = scan_prices(input, mode, chunk.begin, chunk.end);
                local.count += partial.count;
                local.sum_price_cents += partial.sum_price_cents;
                if (mode != PriceAggregateMode::sum) {
                    local.min_price_cents =
                        std::min(local.min_price_cents, partial.min_price_cents);
                    local.max_price_cents =
                        std::max(local.max_price_cents, partial.max_price_cents);
                }
            }
            results[worker].value = local;
            {
                std::lock_guard lock(state_mutex);
                ++completed;
            }
            completion_condition.notify_one();
        }
    }

    [[nodiscard]] PriceAggregateResult execute() {
        std::lock_guard execution_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(state_mutex);
            completed = 0;
            chunks->reset();
            ++generation;
        }
        start_condition.notify_all();
        {
            std::unique_lock lock(state_mutex);
            completion_condition.wait(lock, [&] { return completed == thread_count_value; });
        }

        PriceAggregateResult result;
        if (mode != PriceAggregateMode::sum) {
            result.min_price_cents = std::numeric_limits<std::int64_t>::max();
            result.max_price_cents = std::numeric_limits<std::int64_t>::min();
        }
        auto compute_start = results.front().start;
        for (const auto& worker : results) {
            compute_start = std::min(compute_start, worker.start);
            result.count += worker.value.count;
            result.sum_price_cents += worker.value.sum_price_cents;
            if (mode != PriceAggregateMode::sum) {
                result.min_price_cents =
                    std::min(result.min_price_cents, worker.value.min_price_cents);
                result.max_price_cents =
                    std::max(result.max_price_cents, worker.value.max_price_cents);
            }
        }
        const auto host_end = std::chrono::steady_clock::now();
        result.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        result.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start).count();
        return result;
    }
};

PriceAggregate::PriceAggregate(
    std::span<const float> input,
    PriceAggregateMode mode,
    std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(input, mode, thread_count)) {}
PriceAggregate::~PriceAggregate() = default;
PriceAggregate::PriceAggregate(PriceAggregate&&) noexcept = default;
PriceAggregate& PriceAggregate::operator=(PriceAggregate&&) noexcept = default;
PriceAggregateResult PriceAggregate::execute() { return impl_->execute(); }
std::uint32_t PriceAggregate::thread_count() const noexcept {
    return impl_->thread_count_value;
}

std::vector<std::uint32_t> part_key_group_count_reference(
    std::span<const std::int32_t> keys,
    std::uint32_t group_count) {
    if (keys.empty() || !std::has_single_bit(group_count)) {
        throw std::invalid_argument(
            "group-by keys must not be empty and group count must be a power of two");
    }
    std::vector<std::uint32_t> result(group_count);
    for (const auto key : keys) {
        if (key <= 0) throw std::invalid_argument("part keys must be positive");
        ++result[(static_cast<std::uint32_t>(key) - 1U) & (group_count - 1U)];
    }
    return result;
}

struct PartKeyGroupCount::Impl {
    enum class Phase { accumulate, merge };

    std::span<const std::int32_t> keys;
    std::uint32_t group_count_value{};
    std::uint32_t thread_count_value{};
    std::vector<std::uint32_t> partials;
    std::vector<std::uint32_t> output_storage;
    std::unique_ptr<detail::DynamicChunkCursor> accumulate_chunks;
    std::unique_ptr<detail::DynamicChunkCursor> merge_chunks;
    std::vector<GroupWorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    Phase phase{Phase::accumulate};
    bool stopping{};

    Impl(
        std::span<const std::int32_t> requested_keys,
        std::uint32_t requested_groups,
        std::uint32_t requested_threads)
        : keys(requested_keys),
          group_count_value(requested_groups),
          output_storage(requested_groups) {
        if (keys.empty() || !std::has_single_bit(group_count_value)) {
            throw std::invalid_argument(
                "group-by keys must not be empty and group count must be a power of two");
        }
        for (const auto key : keys) {
            if (key <= 0) throw std::invalid_argument("part keys must be positive");
        }
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_worker_count() : requested_threads,
            keys.size()));
        if (group_count_value >
            std::numeric_limits<std::size_t>::max() / thread_count_value) {
            throw std::invalid_argument("group-by partial table is too large");
        }
        partials.resize(
            static_cast<std::size_t>(thread_count_value) * group_count_value);
        accumulate_chunks = std::make_unique<detail::DynamicChunkCursor>(
            keys.size(), thread_count_value, 64);
        merge_chunks = std::make_unique<detail::DynamicChunkCursor>(
            group_count_value, thread_count_value, 64);
        results.resize(thread_count_value);
        workers.reserve(thread_count_value);
        for (std::uint32_t worker = 0; worker < thread_count_value; ++worker) {
            workers.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(state_mutex);
            stopping = true;
        }
        start_condition.notify_all();
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
        set_worker_qos();
        std::uint64_t observed = 0;
        while (true) {
            Phase active_phase;
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] { return stopping || observed != generation; });
                if (stopping) return;
                observed = generation;
                active_phase = phase;
            }
            if (active_phase == Phase::accumulate) {
                results[worker].start = std::chrono::steady_clock::now();
                auto* local = partials.data() +
                    static_cast<std::size_t>(worker) * group_count_value;
                std::fill(local, local + group_count_value, 0U);
                while (const auto chunk = accumulate_chunks->claim()) {
                    for (auto row = chunk.begin; row < chunk.end; ++row) {
                        ++local[
                            (static_cast<std::uint32_t>(keys[row]) - 1U) &
                            (group_count_value - 1U)];
                    }
                }
            } else {
                while (const auto chunk = merge_chunks->claim()) {
                    for (auto group = chunk.begin; group < chunk.end; ++group) {
                        std::uint32_t count = 0;
                        for (std::uint32_t source = 0;
                             source < thread_count_value; ++source) {
                            count += partials[
                                static_cast<std::size_t>(source) * group_count_value +
                                group];
                        }
                        output_storage[group] = count;
                    }
                }
            }
            {
                std::lock_guard lock(state_mutex);
                ++completed;
            }
            completion_condition.notify_one();
        }
    }

    void run_phase(Phase requested) {
        {
            std::lock_guard lock(state_mutex);
            phase = requested;
            completed = 0;
            if (phase == Phase::accumulate) {
                accumulate_chunks->reset();
            } else {
                merge_chunks->reset();
            }
            ++generation;
        }
        start_condition.notify_all();
        std::unique_lock lock(state_mutex);
        completion_condition.wait(lock, [&] { return completed == thread_count_value; });
    }

    [[nodiscard]] GroupByCountRun execute() {
        std::lock_guard execution_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        run_phase(Phase::accumulate);
        run_phase(Phase::merge);
        const auto host_end = std::chrono::steady_clock::now();
        auto compute_start = results.front().start;
        for (const auto& worker : results) {
            compute_start = std::min(compute_start, worker.start);
        }
        return GroupByCountRun{
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

PartKeyGroupCount::PartKeyGroupCount(
    std::span<const std::int32_t> keys,
    std::uint32_t group_count,
    std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(keys, group_count, thread_count)) {}
PartKeyGroupCount::~PartKeyGroupCount() = default;
PartKeyGroupCount::PartKeyGroupCount(PartKeyGroupCount&&) noexcept = default;
PartKeyGroupCount& PartKeyGroupCount::operator=(PartKeyGroupCount&&) noexcept = default;
GroupByCountRun PartKeyGroupCount::execute() { return impl_->execute(); }
std::span<const std::uint32_t> PartKeyGroupCount::output() const noexcept {
    return impl_->output_storage;
}
std::uint32_t PartKeyGroupCount::thread_count() const noexcept {
    return impl_->thread_count_value;
}
std::uint32_t PartKeyGroupCount::group_count() const noexcept {
    return impl_->group_count_value;
}

}  // namespace joule::operators::cpu
