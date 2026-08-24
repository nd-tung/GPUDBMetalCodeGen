#include "joule/operators/cpu/filter_project.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
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
    std::uint32_t logical_threads = 0;
    std::size_t size = sizeof(logical_threads);
    if (::sysctlbyname("hw.logicalcpu", &logical_threads, &size, nullptr, 0) == 0 &&
        logical_threads > 0) {
        return logical_threads;
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

void set_worker_qos() {
#if defined(__APPLE__)
    static_cast<void>(::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0));
#endif
}

void validate_input(tpch::LineitemView input) {
    const auto rows = input.row_count();
    if (rows == 0 || input.part_key.size() != rows || input.extended_price.size() != rows ||
        input.discount.size() != rows || input.ship_date_yyyymmdd.size() != rows) {
        throw std::invalid_argument("filter-project columns must be non-empty and equally sized");
    }
    if (rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("filter-project supports at most 2^32-1 rows");
    }
}

[[nodiscard]] bool qualifies(tpch::LineitemView input, std::size_t row) noexcept {
    return input.ship_date_yyyymmdd[row] >= 19'940'101 &&
           input.ship_date_yyyymmdd[row] < 19'950'101 && input.discount[row] >= 0.05F &&
           input.discount[row] <= 0.07F && input.quantity[row] < 24.0F;
}

[[nodiscard]] std::int64_t scaled_revenue(tpch::LineitemView input, std::size_t row) noexcept {
    const auto price_cents =
        static_cast<std::int64_t>(std::round(input.extended_price[row] * 100.0F));
    const auto discount_hundredths =
        static_cast<std::int64_t>(std::round(input.discount[row] * 100.0F));
    return price_cents * (100 - discount_hundredths);
}

[[nodiscard]] FilterProjectRecord make_record(tpch::LineitemView input, std::size_t row) noexcept {
    return FilterProjectRecord{static_cast<std::uint32_t>(row), input.part_key[row],
                               scaled_revenue(input, row)};
}

struct alignas(64) WorkerTiming {
    std::chrono::steady_clock::time_point start;
};

}  // namespace

std::vector<FilterProjectRecord> q6_filter_project_reference(tpch::LineitemView input) {
    validate_input(input);
    std::vector<FilterProjectRecord> result;
    for (std::size_t row = 0; row < input.row_count(); ++row) {
        if (qualifies(input, row)) {
            result.push_back(make_record(input, row));
        }
    }
    return result;
}

struct Q6FilterProject::Impl {
    enum class Phase { count, write };

    tpch::LineitemView input;
    std::vector<FilterProjectRecord> output_storage;
    std::uint64_t output_count{};
    std::uint32_t thread_count_value{};
    std::vector<std::uint64_t> counts;
    std::vector<std::uint64_t> offsets;
    std::vector<WorkerTiming> timings;
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    Phase phase{Phase::count};
    bool stopping{};

    Impl(tpch::LineitemView requested_input, std::uint32_t requested_threads)
        : input(requested_input), output_storage(input.row_count()) {
        validate_input(input);
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_worker_count() : requested_threads,
            input.row_count()));
        chunks = std::make_unique<detail::DynamicChunkCursor>(
            input.row_count(), thread_count_value, 64);
        const auto chunk_count =
            input.row_count() / chunks->chunk_size() +
            (input.row_count() % chunks->chunk_size() != 0);
        counts.resize(chunk_count);
        offsets.resize(chunk_count);
        timings.resize(thread_count_value);
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
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
        set_worker_qos();
        std::uint64_t observed_generation = 0;
        while (true) {
            Phase active_phase;
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock,
                                     [&] { return stopping || generation != observed_generation; });
                if (stopping) return;
                observed_generation = generation;
                active_phase = phase;
            }

            if (active_phase == Phase::count) {
                timings[worker].start = std::chrono::steady_clock::now();
                while (const auto chunk = chunks->claim()) {
                    std::uint64_t count = 0;
                    for (auto row = chunk.begin; row < chunk.end; ++row) {
                        count += qualifies(input, row);
                    }
                    counts[chunk.begin / chunks->chunk_size()] = count;
                }
            } else {
                while (const auto chunk = chunks->claim()) {
                    auto write = offsets[chunk.begin / chunks->chunk_size()];
                    for (auto row = chunk.begin; row < chunk.end; ++row) {
                        if (qualifies(input, row)) {
                            output_storage[write++] = make_record(input, row);
                        }
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

    void run_phase(Phase requested_phase) {
        {
            std::lock_guard lock(state_mutex);
            chunks->reset();
            phase = requested_phase;
            completed = 0;
            ++generation;
        }
        start_condition.notify_all();
        std::unique_lock lock(state_mutex);
        completion_condition.wait(lock, [&] { return completed == thread_count_value; });
    }

    [[nodiscard]] FilterProjectRun execute() {
        std::lock_guard execute_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        run_phase(Phase::count);
        output_count = 0;
        for (std::size_t chunk = 0; chunk < counts.size(); ++chunk) {
            offsets[chunk] = output_count;
            output_count += counts[chunk];
        }
        run_phase(Phase::write);
        const auto host_end = std::chrono::steady_clock::now();

        auto compute_start = timings.front().start;
        for (const auto& timing : timings) {
            compute_start = std::min(compute_start, timing.start);
        }
        return FilterProjectRun{
            output_count, std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

Q6FilterProject::Q6FilterProject(tpch::LineitemView input, std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(input, thread_count)) {}
Q6FilterProject::~Q6FilterProject() = default;
Q6FilterProject::Q6FilterProject(Q6FilterProject&&) noexcept = default;
Q6FilterProject& Q6FilterProject::operator=(Q6FilterProject&&) noexcept = default;
FilterProjectRun Q6FilterProject::execute() { return impl_->execute(); }
std::span<const FilterProjectRecord> Q6FilterProject::output() const noexcept {
    return {impl_->output_storage.data(), static_cast<std::size_t>(impl_->output_count)};
}
std::uint32_t Q6FilterProject::thread_count() const noexcept { return impl_->thread_count_value; }

}  // namespace joule::operators::cpu
