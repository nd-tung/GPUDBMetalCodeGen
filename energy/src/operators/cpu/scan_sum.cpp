#include "joule/operators/cpu/scan_sum.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <chrono>
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

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace joule::operators::cpu {
namespace {

[[nodiscard]] std::int64_t scalar_sum(std::span<const std::int32_t> input) {
    std::int64_t sum = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
    for (const auto value : input) {
        sum += value;
    }
    return sum;
}

[[nodiscard]] std::int64_t simd_sum(std::span<const std::int32_t> input) {
#if defined(__aarch64__)
    const auto* values = input.data();
    std::size_t index = 0;
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int64x2_t sum2 = vdupq_n_s64(0);
    int64x2_t sum3 = vdupq_n_s64(0);
    int64x2_t sum4 = vdupq_n_s64(0);
    int64x2_t sum5 = vdupq_n_s64(0);
    int64x2_t sum6 = vdupq_n_s64(0);
    int64x2_t sum7 = vdupq_n_s64(0);

    for (; index + 16 <= input.size(); index += 16) {
        const int32x4_t values0 = vld1q_s32(values + index);
        const int32x4_t values1 = vld1q_s32(values + index + 4);
        const int32x4_t values2 = vld1q_s32(values + index + 8);
        const int32x4_t values3 = vld1q_s32(values + index + 12);
        sum0 = vaddw_s32(sum0, vget_low_s32(values0));
        sum1 = vaddw_high_s32(sum1, values0);
        sum2 = vaddw_s32(sum2, vget_low_s32(values1));
        sum3 = vaddw_high_s32(sum3, values1);
        sum4 = vaddw_s32(sum4, vget_low_s32(values2));
        sum5 = vaddw_high_s32(sum5, values2);
        sum6 = vaddw_s32(sum6, vget_low_s32(values3));
        sum7 = vaddw_high_s32(sum7, values3);
    }

    const int64x2_t first_half = vaddq_s64(
        vaddq_s64(sum0, sum1), vaddq_s64(sum2, sum3));
    const int64x2_t second_half = vaddq_s64(
        vaddq_s64(sum4, sum5), vaddq_s64(sum6, sum7));
    std::int64_t sum = vaddvq_s64(vaddq_s64(first_half, second_half));
    for (; index < input.size(); ++index) {
        sum += values[index];
    }
    return sum;
#else
    return scalar_sum(input);
#endif
}

struct alignas(64) WorkerResult {
    std::int64_t sum{};
    std::chrono::steady_clock::time_point start;
};

[[nodiscard]] std::uint32_t default_worker_count() {
#if defined(__APPLE__)
    std::uint32_t logical_threads = 0;
    std::size_t size = sizeof(logical_threads);
    if (sysctlbyname(
            "hw.logicalcpu", &logical_threads, &size, nullptr, 0) == 0 &&
        logical_threads > 0) {
        return logical_threads;
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

}  // namespace

std::int64_t scan_sum_i32(std::span<const std::int32_t> input) {
    if (input.empty()) {
        throw std::invalid_argument("CPU scan input must not be empty");
    }
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CPU scan supports at most 2^32-1 rows");
    }
    return scalar_sum(input);
}

struct ScanSum::Impl {
    std::span<const std::int32_t> input;
    ScanSumConfig config;
    std::uint32_t thread_count_value{1};
    std::unique_ptr<detail::DynamicChunkCursor> chunks;
    std::vector<WorkerResult> results;
    std::vector<std::jthread> workers;
    std::mutex state_mutex;
    std::mutex execute_mutex;
    std::condition_variable start_condition;
    std::condition_variable completion_condition;
    std::uint64_t generation{};
    std::uint32_t completed_workers{};
    bool stopping{};

    explicit Impl(std::span<const std::int32_t> requested_input, ScanSumConfig requested_config)
        : input(requested_input), config(requested_config) {
        if (input.empty()) {
            throw std::invalid_argument("CPU scan input must not be empty");
        }
        if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("CPU scan supports at most 2^32-1 rows");
        }

        if (config.kernel == ScanSumKernel::scalar) {
            thread_count_value = 1;
            return;
        }

        const auto requested_threads = config.thread_count == 0
            ? default_worker_count()
            : config.thread_count;
        thread_count_value = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            requested_threads, input.size()));
        if (thread_count_value == 0) {
            throw std::invalid_argument("CPU thread count must be greater than zero");
        }

        chunks = std::make_unique<detail::DynamicChunkCursor>(
            input.size(), thread_count_value, 16);
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
        // Join while the condition variables and mutexes are still alive.
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
#if defined(__APPLE__)
        // Active analytical work should be scheduled as user-initiated work;
        // otherwise macOS may place an N=P-core-count pool on efficiency cores.
        static_cast<void>(pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0));
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

            auto& result = results[worker];
            result.start = std::chrono::steady_clock::now();
            result.sum = 0;
            while (const auto chunk = chunks->claim()) {
                const auto values = input.subspan(
                    chunk.begin, chunk.end - chunk.begin);
                result.sum += config.kernel == ScanSumKernel::parallel_simd
                    ? simd_sum(values)
                    : scalar_sum(values);
            }
            {
                std::lock_guard lock(state_mutex);
                ++completed_workers;
            }
            completion_condition.notify_one();
        }
    }

    [[nodiscard]] ScanSumRun execute() {
        std::lock_guard execution_lock(execute_mutex);
        if (config.kernel == ScanSumKernel::scalar) {
            const auto start = std::chrono::steady_clock::now();
            const auto sum = scalar_sum(input);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            return ScanSumRun{sum, elapsed, elapsed};
        }

        const auto host_start = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(state_mutex);
            completed_workers = 0;
            chunks->reset();
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
        std::int64_t sum = 0;
        for (const auto& result : results) {
            compute_start = std::min(compute_start, result.start);
            sum += result.sum;
        }
        const auto host_end = std::chrono::steady_clock::now();
        return ScanSumRun{
            sum,
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

ScanSum::ScanSum(std::span<const std::int32_t> input, ScanSumConfig config)
    : impl_(std::make_unique<Impl>(input, config)) {}

ScanSum::~ScanSum() = default;
ScanSum::ScanSum(ScanSum&&) noexcept = default;
ScanSum& ScanSum::operator=(ScanSum&&) noexcept = default;

ScanSumRun ScanSum::execute() {
    return impl_->execute();
}

std::uint32_t ScanSum::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
