#include "joule/operators/cpu/tpch_q14.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <bit>
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

struct HashTable {
    std::vector<std::int32_t> keys;
    std::vector<std::uint8_t> promo;
    std::size_t mask{};

    explicit HashTable(tpch::PartView part) {
        if (part.row_count() == 0 || part.type.size() != part.row_count() * 25) {
            throw std::invalid_argument("TPC-H part columns must be non-empty and equally sized");
        }
        const auto capacity = std::bit_ceil(std::max<std::size_t>(2, part.row_count() * 2));
        keys.assign(capacity, 0);
        promo.assign(capacity, 0);
        mask = capacity - 1;
        for (std::size_t row = 0; row < part.row_count(); ++row) {
            const auto key = part.part_key[row];
            if (key <= 0) throw std::invalid_argument("TPC-H part keys must be positive");
            auto slot = static_cast<std::size_t>(static_cast<std::uint32_t>(key) * 2'654'435'761U) & mask;
            while (keys[slot] != 0 && keys[slot] != key) slot = (slot + 1) & mask;
            keys[slot] = key;
            const char* type = part.type.data() + row * 25;
            promo[slot] = type[0] == 'P' && type[1] == 'R' && type[2] == 'O' &&
                          type[3] == 'M' && type[4] == 'O';
        }
    }

    [[nodiscard]] bool lookup(std::int32_t key, bool& is_promo) const noexcept {
        auto slot = static_cast<std::size_t>(static_cast<std::uint32_t>(key) * 2'654'435'761U) & mask;
        while (keys[slot] != 0) {
            if (keys[slot] == key) {
                is_promo = promo[slot] != 0;
                return true;
            }
            slot = (slot + 1) & mask;
        }
        return false;
    }
};

void validate_lineitem(tpch::LineitemView input) {
    const auto rows = input.row_count();
    if (rows == 0 || input.part_key.size() != rows ||
        input.extended_price.size() != rows || input.discount.size() != rows ||
        input.ship_date_yyyymmdd.size() != rows) {
        throw std::invalid_argument("TPC-H Q14 lineitem columns must be non-empty and equally sized");
    }
}

[[nodiscard]] TpchQ14Result scan_range(
    tpch::LineitemView input,
    const HashTable& hash,
    std::size_t begin,
    std::size_t end) {
    TpchQ14Result result;
    for (auto row = begin; row < end; ++row) {
        const auto date = input.ship_date_yyyymmdd[row];
        if (date < 19'950'901 || date >= 19'951'001) continue;
        bool promo = false;
        if (!hash.lookup(input.part_key[row], promo)) continue;
        const auto price = static_cast<std::int64_t>(
            std::round(input.extended_price[row] * 100.0F));
        const auto discount = static_cast<std::int64_t>(
            std::round(input.discount[row] * 100.0F));
        const auto revenue = price * (100 - discount);
        result.total_revenue_1e4_usd += revenue;
        if (promo) result.promo_revenue_1e4_usd += revenue;
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
    TpchQ14Result result;
    std::chrono::steady_clock::time_point start;
};

}  // namespace

TpchQ14Result tpch_q14_reference(tpch::LineitemView input, tpch::PartView part) {
    validate_lineitem(input);
    return scan_range(input, HashTable(part), 0, input.row_count());
}

struct TpchQ14HashJoin::Impl {
    tpch::LineitemView input;
    HashTable hash;
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

    Impl(tpch::LineitemView lineitem, tpch::PartView part, std::uint32_t requested)
        : input(lineitem), hash(part) {
        validate_lineitem(input);
        thread_count_value = static_cast<std::uint32_t>(std::min<std::size_t>(
            requested == 0 ? default_workers() : requested, input.row_count()));
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
            results[worker].result = {};
            while (const auto chunk = chunks->claim()) {
                const auto partial = scan_range(input, hash, chunk.begin, chunk.end);
                results[worker].result.promo_revenue_1e4_usd +=
                    partial.promo_revenue_1e4_usd;
                results[worker].result.total_revenue_1e4_usd +=
                    partial.total_revenue_1e4_usd;
            }
            { std::lock_guard lock(state_mutex); ++completed; }
            complete_condition.notify_one();
        }
    }

    [[nodiscard]] TpchQ14Result execute() {
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
        TpchQ14Result result;
        auto compute_start = results.front().start;
        for (const auto& worker : results) {
            compute_start = std::min(compute_start, worker.start);
            result.promo_revenue_1e4_usd += worker.result.promo_revenue_1e4_usd;
            result.total_revenue_1e4_usd += worker.result.total_revenue_1e4_usd;
        }
        const auto host_end = std::chrono::steady_clock::now();
        result.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        result.compute_time_ms =
            std::chrono::duration<double, std::milli>(host_end - compute_start).count();
        return result;
    }
};

TpchQ14HashJoin::TpchQ14HashJoin(
    tpch::LineitemView lineitem, tpch::PartView part, std::uint32_t threads)
    : impl_(std::make_unique<Impl>(lineitem, part, threads)) {}
TpchQ14HashJoin::~TpchQ14HashJoin() = default;
TpchQ14HashJoin::TpchQ14HashJoin(TpchQ14HashJoin&&) noexcept = default;
TpchQ14HashJoin& TpchQ14HashJoin::operator=(TpchQ14HashJoin&&) noexcept = default;
TpchQ14Result TpchQ14HashJoin::execute() { return impl_->execute(); }
std::uint32_t TpchQ14HashJoin::thread_count() const noexcept {
    return impl_->thread_count_value;
}

}  // namespace joule::operators::cpu
