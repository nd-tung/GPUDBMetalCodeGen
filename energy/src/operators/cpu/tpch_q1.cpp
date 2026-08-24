#include "joule/operators/cpu/tpch_q1.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
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

[[nodiscard]] std::uint32_t default_worker_count() {
#if defined(__APPLE__)
    std::uint32_t count = 0;
    std::size_t size = sizeof(count);
    if (::sysctlbyname("hw.logicalcpu", &count, &size, nullptr, 0) == 0 &&
        count > 0) return count;
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

[[nodiscard]] int group_index(char return_flag, char line_status) noexcept {
    const int first = return_flag == 'A' ? 0 : return_flag == 'N' ? 1 :
                      return_flag == 'R' ? 2 : -1;
    const int second = line_status == 'F' ? 0 : line_status == 'O' ? 1 : -1;
    return first < 0 || second < 0 ? -1 : first * 2 + second;
}

[[nodiscard]] std::int64_t scaled(float value) noexcept {
    return static_cast<std::int64_t>(std::round(value * 100.0F));
}

void add_row(TpchQ1Group& group, tpch::LineitemView input, std::size_t row) {
    const auto quantity = scaled(input.quantity[row]);
    const auto price = scaled(input.extended_price[row]);
    const auto discount = scaled(input.discount[row]);
    const auto tax = scaled(input.tax[row]);
    ++group.count;
    group.sum_quantity_1e2 += quantity;
    group.sum_base_price_1e2 += price;
    group.sum_discount_price_1e4_usd += price * (100 - discount);
    group.sum_charge_1e6_usd += price * (100 - discount) * (100 + tax);
    group.sum_discount_1e2 += discount;
}

void merge(TpchQ1Group& output, const TpchQ1Group& input) {
    output.count += input.count;
    output.sum_quantity_1e2 += input.sum_quantity_1e2;
    output.sum_base_price_1e2 += input.sum_base_price_1e2;
    output.sum_discount_price_1e4_usd += input.sum_discount_price_1e4_usd;
    output.sum_charge_1e6_usd += input.sum_charge_1e6_usd;
    output.sum_discount_1e2 += input.sum_discount_1e2;
}

void validate(tpch::LineitemView input) {
    const auto rows = input.row_count();
    if (rows == 0 || input.extended_price.size() != rows || input.discount.size() != rows ||
        input.tax.size() != rows || input.return_flag.size() != rows ||
        input.line_status.size() != rows || input.ship_date_yyyymmdd.size() != rows) {
        throw std::invalid_argument("TPC-H Q1 columns must be non-empty and equally sized");
    }
}

[[nodiscard]] TpchQ1Groups scan_range(
    tpch::LineitemView input,
    std::size_t begin,
    std::size_t end) {
    TpchQ1Groups groups{};
    for (auto row = begin; row < end; ++row) {
        if (input.ship_date_yyyymmdd[row] > 19'980'902) continue;
        const auto group = group_index(input.return_flag[row], input.line_status[row]);
        if (group >= 0) add_row(groups[static_cast<std::size_t>(group)], input, row);
    }
    return groups;
}

struct alignas(64) WorkerResult {
    TpchQ1Groups groups{};
    std::chrono::steady_clock::time_point start;
};

}  // namespace

TpchQ1Groups tpch_q1_reference(tpch::LineitemView input) {
    validate(input);
    return scan_range(input, 0, input.row_count());
}

struct TpchQ1GroupBy::Impl {
    tpch::LineitemView input;
    std::uint32_t thread_count_value{};
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<WorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    bool stopping{};

    Impl(tpch::LineitemView requested_input, std::uint32_t requested_threads)
        : input(requested_input) {
        validate(input);
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_worker_count() : requested_threads,
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
                start_condition.wait(lock, [&] { return stopping || generation != observed; });
                if (stopping) return;
                observed = generation;
            }
            results[worker].start = std::chrono::steady_clock::now();
            results[worker].groups = {};
            while (const auto chunk = chunks->claim()) {
                const auto partial = scan_range(input, chunk.begin, chunk.end);
                for (std::size_t group = 0;
                     group < results[worker].groups.size(); ++group) {
                    merge(results[worker].groups[group], partial[group]);
                }
            }
            { std::lock_guard lock(state_mutex); ++completed; }
            completion_condition.notify_one();
        }
    }

    [[nodiscard]] TpchQ1Run execute() {
        std::lock_guard execute_lock(execute_mutex);
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
        TpchQ1Run run;
        auto compute_start = results.front().start;
        for (const auto& result : results) {
            compute_start = std::min(compute_start, result.start);
            for (std::size_t group = 0; group < run.groups.size(); ++group) {
                merge(run.groups[group], result.groups[group]);
            }
        }
        const auto host_end = std::chrono::steady_clock::now();
        run.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        run.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start).count();
        return run;
    }
};

TpchQ1GroupBy::TpchQ1GroupBy(tpch::LineitemView input, std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(input, thread_count)) {}
TpchQ1GroupBy::~TpchQ1GroupBy() = default;
TpchQ1GroupBy::TpchQ1GroupBy(TpchQ1GroupBy&&) noexcept = default;
TpchQ1GroupBy& TpchQ1GroupBy::operator=(TpchQ1GroupBy&&) noexcept = default;
TpchQ1Run TpchQ1GroupBy::execute() { return impl_->execute(); }
std::uint32_t TpchQ1GroupBy::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
