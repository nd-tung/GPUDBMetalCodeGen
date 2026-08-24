#include "joule/operators/cpu/tpch_q6.hpp"

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

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace joule::operators::cpu {
namespace {

constexpr std::int32_t date_begin = 19'940'101;
constexpr std::int32_t date_end = 19'950'101;
constexpr float discount_begin = 0.05F;
constexpr float discount_end = 0.07F;
constexpr float quantity_end = 24.0F;

[[nodiscard]] bool qualifies(tpch::LineitemView input, std::size_t index) noexcept {
    return input.ship_date_yyyymmdd[index] >= date_begin &&
           input.ship_date_yyyymmdd[index] < date_end &&
           input.discount[index] >= discount_begin &&
           input.discount[index] <= discount_end && input.quantity[index] < quantity_end;
}

[[nodiscard]] std::int64_t scaled_revenue(
    tpch::LineitemView input,
    std::size_t index) noexcept {
    const auto price_cents = static_cast<std::int64_t>(
        std::round(input.extended_price[index] * 100.0F));
    const auto discount_hundredths = static_cast<std::int64_t>(
        std::round(input.discount[index] * 100.0F));
    return price_cents * discount_hundredths;
}

struct PartialResult {
    std::uint64_t count{};
    std::int64_t revenue{};
};

[[nodiscard]] PartialResult scan_range(
    tpch::LineitemView input,
    std::size_t begin,
    std::size_t end,
    bool calculate_revenue) {
    PartialResult result;
    std::size_t index = begin;
#if defined(__aarch64__)
    const auto date_min = vdupq_n_s32(date_begin);
    const auto date_max = vdupq_n_s32(date_end);
    const auto discount_min = vdupq_n_f32(discount_begin);
    const auto discount_max = vdupq_n_f32(discount_end);
    const auto quantity_max = vdupq_n_f32(quantity_end);
    for (; index + 4 <= end; index += 4) {
        uint32x4_t mask = vcgeq_s32(
            vld1q_s32(input.ship_date_yyyymmdd.data() + index), date_min);
        mask = vandq_u32(mask, vcltq_s32(
            vld1q_s32(input.ship_date_yyyymmdd.data() + index), date_max));
        const auto discounts = vld1q_f32(input.discount.data() + index);
        mask = vandq_u32(mask, vcgeq_f32(discounts, discount_min));
        mask = vandq_u32(mask, vcleq_f32(discounts, discount_max));
        mask = vandq_u32(mask, vcltq_f32(
            vld1q_f32(input.quantity.data() + index), quantity_max));
        result.count += vaddvq_u32(vshrq_n_u32(mask, 31));
        if (calculate_revenue) {
            std::uint32_t lanes[4];
            vst1q_u32(lanes, mask);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                if (lanes[lane] != 0) {
                    result.revenue += scaled_revenue(input, index + lane);
                }
            }
        }
    }
#endif
    for (; index < end; ++index) {
        if (qualifies(input, index)) {
            ++result.count;
            if (calculate_revenue) {
                result.revenue += scaled_revenue(input, index);
            }
        }
    }
    return result;
}

[[nodiscard]] PartialResult bitmap_range(
    tpch::LineitemView input,
    std::span<std::uint32_t> bitmap,
    std::size_t word_begin,
    std::size_t word_end) {
    PartialResult result;
    for (std::size_t word = word_begin; word < word_end; ++word) {
        std::uint32_t bits = 0;
        const auto row_begin = word * 32;
        const auto row_end = std::min(row_begin + 32, input.row_count());
        for (std::size_t row = row_begin; row < row_end; ++row) {
            bits |= static_cast<std::uint32_t>(qualifies(input, row)) << (row - row_begin);
        }
        bitmap[word] = bits;
    }
    return result;
}

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

struct alignas(64) WorkerResult {
    PartialResult value;
    std::chrono::steady_clock::time_point start;
};

void validate_input(tpch::LineitemView input) {
    if (input.row_count() == 0 || input.extended_price.size() != input.row_count() ||
        input.discount.size() != input.row_count() ||
        input.ship_date_yyyymmdd.size() != input.row_count()) {
        throw std::invalid_argument("TPC-H Q6 columns must be non-empty and equally sized");
    }
    if (input.row_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("TPC-H Q6 supports at most 2^32-1 rows");
    }
}

}  // namespace

TpchQ6Reference tpch_q6_reference(tpch::LineitemView input, bool materialize_bitmap) {
    validate_input(input);
    TpchQ6Reference reference;
    if (materialize_bitmap) {
        reference.bitmap.resize((input.row_count() + 31) / 32);
    }
    for (std::size_t index = 0; index < input.row_count(); ++index) {
        if (qualifies(input, index)) {
            ++reference.match_count;
            reference.revenue_1e4_usd += scaled_revenue(input, index);
            if (materialize_bitmap) {
                reference.bitmap[index / 32] |= 1U << (index % 32);
            }
        }
    }
    return reference;
}

struct TpchQ6::Impl {
    tpch::LineitemView input;
    TpchQ6Config config;
    std::uint32_t thread_count_value{};
    std::vector<std::uint32_t> bitmap_storage;
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

    Impl(tpch::LineitemView requested_input, TpchQ6Config requested_config)
        : input(requested_input), config(requested_config) {
        validate_input(input);
        if (config.mode == TpchQ6Mode::filter_bitmap) {
            bitmap_storage.resize((input.row_count() + 31) / 32);
        }
        const auto units = config.mode == TpchQ6Mode::filter_bitmap
            ? bitmap_storage.size()
            : input.row_count();
        const auto requested_threads = config.thread_count == 0
            ? default_worker_count()
            : config.thread_count;
        thread_count_value = static_cast<std::uint32_t>(
            std::min<std::size_t>(requested_threads, units));
        if (thread_count_value == 0) {
            throw std::invalid_argument("CPU thread count must be greater than zero");
        }
        chunks = std::make_unique<detail::DynamicChunkCursor>(
            units, thread_count_value,
            config.mode == TpchQ6Mode::filter_bitmap ? 1 : 16);
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
        workers.clear();
    }

    void worker_loop(std::uint32_t worker) {
#if defined(__APPLE__)
        static_cast<void>(::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0));
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
            output.value = {};
            while (const auto chunk = chunks->claim()) {
                if (config.mode == TpchQ6Mode::filter_bitmap) {
                    static_cast<void>(bitmap_range(
                        input, bitmap_storage, chunk.begin, chunk.end));
                } else {
                    const auto partial = scan_range(
                        input, chunk.begin, chunk.end,
                        config.mode == TpchQ6Mode::revenue);
                    output.value.count += partial.count;
                    output.value.revenue += partial.revenue;
                }
            }
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
        TpchQ6Result combined;
        for (const auto& result : results) {
            compute_start = std::min(compute_start, result.start);
            combined.match_count += result.value.count;
            combined.revenue_1e4_usd += result.value.revenue;
        }
        const auto host_end = std::chrono::steady_clock::now();
        combined.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        combined.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start).count();
        return combined;
    }
};

TpchQ6::TpchQ6(tpch::LineitemView input, TpchQ6Config config)
    : impl_(std::make_unique<Impl>(input, config)) {}

TpchQ6::~TpchQ6() = default;
TpchQ6::TpchQ6(TpchQ6&&) noexcept = default;
TpchQ6& TpchQ6::operator=(TpchQ6&&) noexcept = default;

TpchQ6Result TpchQ6::execute() {
    return impl_->execute();
}

std::span<const std::uint32_t> TpchQ6::bitmap() const noexcept {
    return impl_->bitmap_storage;
}

std::uint32_t TpchQ6::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
