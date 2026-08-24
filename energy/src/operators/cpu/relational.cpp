#include "joule/operators/cpu/relational.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
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
    if (::sysctlbyname(
            "hw.logicalcpu", &logical_threads, &size, nullptr, 0) == 0 &&
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

[[nodiscard]] bool q6_qualifies(tpch::LineitemView input, std::size_t row) noexcept {
    return input.ship_date_yyyymmdd[row] >= 19'940'101 &&
           input.ship_date_yyyymmdd[row] < 19'950'101 &&
           input.discount[row] >= 0.05F && input.discount[row] <= 0.07F &&
           input.quantity[row] < 24.0F;
}

void validate_q6(tpch::LineitemView input) {
    if (input.row_count() == 0 || input.discount.size() != input.row_count() ||
        input.ship_date_yyyymmdd.size() != input.row_count()) {
        throw std::invalid_argument("Q6 columns must be non-empty and equally sized");
    }
}

struct alignas(64) WorkerTiming {
    std::chrono::steady_clock::time_point start;
};

}  // namespace

struct ScanCopyF32::Impl {
    std::span<const float> input;
    std::vector<float> output;
    std::uint32_t thread_count_value{};
    std::vector<WorkerTiming> timings;
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed{};
    bool stopping{};

    Impl(std::span<const float> requested_input, std::uint32_t requested_threads)
        : input(requested_input), output(input.size()) {
        if (input.empty()) {
            throw std::invalid_argument("scan-copy input must not be empty");
        }
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested_threads == 0 ? default_worker_count() : requested_threads,
            input.size()));
        chunks = std::make_unique<detail::DynamicChunkCursor>(
            input.size(), thread_count_value, 16);
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
        std::uint64_t observed = 0;
        while (true) {
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] { return stopping || generation != observed; });
                if (stopping) return;
                observed = generation;
            }
            timings[worker].start = std::chrono::steady_clock::now();
            while (const auto chunk = chunks->claim()) {
                std::memcpy(
                    output.data() + chunk.begin, input.data() + chunk.begin,
                    (chunk.end - chunk.begin) * sizeof(float));
            }
            {
                std::lock_guard lock(state_mutex);
                ++completed;
            }
            completion_condition.notify_one();
        }
    }

    [[nodiscard]] RelationalRun execute() {
        std::lock_guard execute_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(state_mutex);
            chunks->reset();
            completed = 0;
            ++generation;
        }
        start_condition.notify_all();
        {
            std::unique_lock lock(state_mutex);
            completion_condition.wait(lock, [&] { return completed == thread_count_value; });
        }
        auto compute_start = timings.front().start;
        for (const auto& timing : timings) {
            compute_start = std::min(compute_start, timing.start);
        }
        const auto host_end = std::chrono::steady_clock::now();
        return RelationalRun{
            input.size(),
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

ScanCopyF32::ScanCopyF32(std::span<const float> input, std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(input, thread_count)) {}
ScanCopyF32::~ScanCopyF32() = default;
ScanCopyF32::ScanCopyF32(ScanCopyF32&&) noexcept = default;
ScanCopyF32& ScanCopyF32::operator=(ScanCopyF32&&) noexcept = default;
RelationalRun ScanCopyF32::execute() { return impl_->execute(); }
std::span<const float> ScanCopyF32::output() const noexcept { return impl_->output; }
std::uint32_t ScanCopyF32::thread_count() const noexcept {
    return impl_->thread_count_value;
}

std::vector<std::uint32_t> q6_materialize_reference(tpch::LineitemView input) {
    validate_q6(input);
    std::vector<std::uint32_t> rows;
    for (std::size_t row = 0; row < input.row_count(); ++row) {
        if (q6_qualifies(input, row)) {
            rows.push_back(static_cast<std::uint32_t>(row));
        }
    }
    return rows;
}

struct Q6FilterMaterialize::Impl {
    enum class Phase { count, write };
    tpch::LineitemView input;
    std::vector<std::uint32_t> output_storage;
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
        validate_q6(input);
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
        std::uint64_t observed = 0;
        while (true) {
            Phase active_phase;
            {
                std::unique_lock lock(state_mutex);
                start_condition.wait(lock, [&] { return stopping || generation != observed; });
                if (stopping) return;
                observed = generation;
                active_phase = phase;
            }
            if (active_phase == Phase::count) {
                timings[worker].start = std::chrono::steady_clock::now();
                while (const auto chunk = chunks->claim()) {
                    std::uint64_t count = 0;
                    for (auto row = chunk.begin; row < chunk.end; ++row) {
                        count += q6_qualifies(input, row);
                    }
                    counts[chunk.begin / chunks->chunk_size()] = count;
                }
            } else {
                while (const auto chunk = chunks->claim()) {
                    auto write = offsets[chunk.begin / chunks->chunk_size()];
                    for (auto row = chunk.begin; row < chunk.end; ++row) {
                        if (q6_qualifies(input, row)) {
                            output_storage[write++] = static_cast<std::uint32_t>(row);
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

    [[nodiscard]] RelationalRun execute() {
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
        return RelationalRun{
            output_count,
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

Q6FilterMaterialize::Q6FilterMaterialize(
    tpch::LineitemView input,
    std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(input, thread_count)) {}
Q6FilterMaterialize::~Q6FilterMaterialize() = default;
Q6FilterMaterialize::Q6FilterMaterialize(Q6FilterMaterialize&&) noexcept = default;
Q6FilterMaterialize& Q6FilterMaterialize::operator=(Q6FilterMaterialize&&) noexcept = default;
RelationalRun Q6FilterMaterialize::execute() { return impl_->execute(); }
std::span<const std::uint32_t> Q6FilterMaterialize::output() const noexcept {
    return {impl_->output_storage.data(), static_cast<std::size_t>(impl_->output_count)};
}
std::uint32_t Q6FilterMaterialize::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
