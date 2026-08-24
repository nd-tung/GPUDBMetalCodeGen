#include "joule/operators/cpu/topk.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
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

[[nodiscard]] TopKEntry sentinel() noexcept {
    return {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int32_t>::max()};
}

[[nodiscard]] bool better(TopKEntry left, TopKEntry right) noexcept {
    return left.total_price_cents > right.total_price_cents ||
           (left.total_price_cents == right.total_price_cents &&
            left.order_key < right.order_key);
}

void insert(Top10& result, TopKEntry candidate) noexcept {
    if (!better(candidate, result.back())) return;
    auto position = result.size() - 1;
    while (position > 0 && better(candidate, result[position - 1])) {
        result[position] = result[position - 1];
        --position;
    }
    result[position] = candidate;
}

[[nodiscard]] Top10 empty_top10() noexcept {
    Top10 result;
    result.fill(sentinel());
    return result;
}

void validate(tpch::OrdersView input) {
    if (input.row_count() == 0 || input.total_price.size() != input.row_count()) {
        throw std::invalid_argument("TPC-H orders columns must be non-empty and equally sized");
    }
}

[[nodiscard]] Top10 scan_range(tpch::OrdersView input, std::size_t begin, std::size_t end) {
    auto result = empty_top10();
    for (auto row = begin; row < end; ++row) {
        insert(result, {
            static_cast<std::int64_t>(std::round(input.total_price[row] * 100.0F)),
            input.order_key[row]});
    }
    return result;
}

[[nodiscard]] std::uint32_t default_workers() {
#if defined(__APPLE__)
    std::uint32_t count = 0;
    std::size_t size = sizeof(count);
    if (::sysctlbyname("hw.logicalcpu", &count, &size, nullptr, 0) == 0 && count > 0) {
        return count;
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

struct alignas(64) WorkerResult {
    Top10 rows{};
    std::chrono::steady_clock::time_point start;
};

}  // namespace

Top10 top10_reference(tpch::OrdersView input) {
    validate(input);
    return scan_range(input, 0, input.row_count());
}

struct OrdersTopK::Impl {
    tpch::OrdersView input;
    std::uint32_t thread_count_value{};
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<WorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable complete_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    bool stopping{};

    Impl(tpch::OrdersView requested, std::uint32_t requested_threads) : input(requested) {
        validate(input);
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_workers() : requested_threads,
            input.row_count()));
        chunks = std::make_unique<detail::DynamicChunkCursor>(
            input.row_count(), thread_count_value, 64);
        results.resize(thread_count_value);
        workers.reserve(thread_count_value);
        for (std::uint32_t worker = 0; worker < thread_count_value; ++worker) {
            workers.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~Impl() {
        { std::lock_guard lock(state_mutex); stopping = true; }
        start_condition.notify_all();
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
#if defined(__APPLE__)
        static_cast<void>(::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0));
#endif
        std::uint64_t observed = 0;
        while (true) {
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] { return stopping || observed != generation; });
                if (stopping) return;
                observed = generation;
            }
            results[worker].start = std::chrono::steady_clock::now();
            results[worker].rows = empty_top10();
            while (const auto chunk = chunks->claim()) {
                const auto partial = scan_range(input, chunk.begin, chunk.end);
                for (const auto entry : partial) insert(results[worker].rows, entry);
            }
            { std::lock_guard lock(state_mutex); ++completed; }
            complete_condition.notify_one();
        }
    }

    [[nodiscard]] TopKRun execute() {
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
            complete_condition.wait(lock, [&] { return completed == thread_count_value; });
        }
        TopKRun run;
        run.rows = empty_top10();
        auto compute_start = results.front().start;
        for (const auto& result : results) {
            compute_start = std::min(compute_start, result.start);
            for (const auto entry : result.rows) insert(run.rows, entry);
        }
        const auto host_end = std::chrono::steady_clock::now();
        run.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        run.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start).count();
        return run;
    }
};

OrdersTopK::OrdersTopK(tpch::OrdersView input, std::uint32_t threads)
    : impl_(std::make_unique<Impl>(input, threads)) {}
OrdersTopK::~OrdersTopK() = default;
OrdersTopK::OrdersTopK(OrdersTopK&&) noexcept = default;
OrdersTopK& OrdersTopK::operator=(OrdersTopK&&) noexcept = default;
TopKRun OrdersTopK::execute() { return impl_->execute(); }
std::uint32_t OrdersTopK::thread_count() const noexcept { return impl_->thread_count_value; }

}  // namespace joule::operators::cpu
