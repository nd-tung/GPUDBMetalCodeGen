#include "joule/operators/cpu/hash_join.hpp"

#include "dynamic_chunks.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <functional>
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

[[nodiscard]] bool is_promo_type(const char* type) noexcept {
    return type[0] == 'P' && type[1] == 'R' && type[2] == 'O' &&
           type[3] == 'M' && type[4] == 'O';
}

void validate_part(tpch::PartView part) {
    if (part.row_count() == 0 || part.type.size() != part.row_count() * 25) {
        throw std::invalid_argument(
            "part hash build columns must be non-empty and equally sized");
    }
    if (std::any_of(part.part_key.begin(), part.part_key.end(), [](std::int32_t key) {
            return key <= 0;
        })) {
        throw std::invalid_argument("part hash build keys must be positive");
    }
}

[[nodiscard]] std::size_t hash_capacity(std::size_t rows) {
    if (rows > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("part hash table is too large");
    }
    const auto requested = std::max<std::size_t>(2, rows * 2);
    const auto capacity = std::bit_ceil(requested);
    if (capacity < requested) {
        throw std::invalid_argument("part hash table capacity overflow");
    }
    return capacity;
}

[[nodiscard]] std::size_t initial_slot(std::int32_t key, std::size_t mask) noexcept {
    return static_cast<std::size_t>(
               static_cast<std::uint32_t>(key) * 2'654'435'761U) &
           mask;
}

struct HashStorage {
    std::vector<std::int32_t> keys;
    std::vector<std::uint32_t> promo;
    std::size_t mask{};

    explicit HashStorage(std::size_t build_rows)
        : keys(hash_capacity(build_rows), 0),
          promo(keys.size(), 0),
          mask(keys.size() - 1) {}

    [[nodiscard]] bool insert_serial(std::int32_t key, bool is_promo) {
        auto slot = initial_slot(key, mask);
        while (keys[slot] != 0) {
            if (keys[slot] == key) return false;
            slot = (slot + 1) & mask;
        }
        keys[slot] = key;
        promo[slot] = static_cast<std::uint32_t>(is_promo);
        return true;
    }

    [[nodiscard]] bool insert_atomic(std::int32_t key, bool is_promo) noexcept {
        auto slot = initial_slot(key, mask);
        while (true) {
            std::atomic_ref<std::int32_t> target(keys[slot]);
            auto observed = target.load(std::memory_order_relaxed);
            if (observed == key) {
                // Duplicate keys are outside this unique-key primitive's
                // contract. Publishing the same payload is still harmless.
                promo[slot] = static_cast<std::uint32_t>(is_promo);
                return false;
            }
            if (observed == 0) {
                if (target.compare_exchange_strong(
                        observed, key, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    promo[slot] = static_cast<std::uint32_t>(is_promo);
                    return true;
                }
                if (observed == key) {
                    promo[slot] = static_cast<std::uint32_t>(is_promo);
                    return false;
                }
            }
            slot = (slot + 1) & mask;
        }
    }

    [[nodiscard]] bool lookup(std::int32_t key, bool& is_promo) const noexcept {
        if (key <= 0) return false;
        auto slot = initial_slot(key, mask);
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

struct PreparedHash {
    HashStorage storage;

    explicit PreparedHash(tpch::PartView part) : storage(part.row_count()) {
        validate_part(part);
        for (std::size_t row = 0; row < part.row_count(); ++row) {
            if (!storage.insert_serial(
                    part.part_key[row], is_promo_type(part.type.data() + row * 25))) {
                throw std::invalid_argument(
                    "part hash build requires unique part keys");
            }
        }
    }
};

[[nodiscard]] HashBuildVerification verify_hash(
    const HashStorage& storage,
    tpch::PartView part) {
    HashBuildVerification result;
    for (std::size_t slot = 0; slot < storage.keys.size(); ++slot) {
        const auto key = storage.keys[slot];
        if (key == 0) continue;
        ++result.entry_count;
        result.promo_entry_count += storage.promo[slot] != 0;
        result.key_sum += static_cast<std::uint32_t>(key);
    }

    result.valid = result.entry_count == part.row_count();
    for (std::size_t row = 0; result.valid && row < part.row_count(); ++row) {
        bool promo = false;
        result.valid =
            storage.lookup(part.part_key[row], promo) &&
            promo == is_promo_type(part.type.data() + row * 25);
    }
    return result;
}

class PersistentWorkerTeam {
public:
    PersistentWorkerTeam(std::uint32_t requested, std::size_t work_items)
        : count_(static_cast<std::uint32_t>(std::min<std::size_t>(
              requested == 0 ? default_worker_count() : requested,
              std::max<std::size_t>(1, work_items)))),
          starts_(count_) {
        workers_.reserve(count_);
        try {
            for (std::uint32_t worker = 0; worker < count_; ++worker) {
                workers_.emplace_back([this, worker] { worker_loop(worker); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            start_condition_.notify_all();
            workers_.clear();
            throw;
        }
    }

    ~PersistentWorkerTeam() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        start_condition_.notify_all();
        workers_.clear();
    }

    PersistentWorkerTeam(const PersistentWorkerTeam&) = delete;
    PersistentWorkerTeam& operator=(const PersistentWorkerTeam&) = delete;

    [[nodiscard]] std::uint32_t size() const noexcept { return count_; }

    void run(std::function<void(std::uint32_t)> work) {
        {
            std::lock_guard lock(mutex_);
            work_ = std::move(work);
            completed_ = 0;
            ++generation_;
        }
        start_condition_.notify_all();
        std::unique_lock lock(mutex_);
        complete_condition_.wait(lock, [&] { return completed_ == count_; });
    }

    [[nodiscard]] std::chrono::steady_clock::time_point earliest_start() const {
        return *std::min_element(starts_.begin(), starts_.end());
    }

private:
    void worker_loop(std::uint32_t worker) {
        set_worker_qos();
        std::uint64_t observed_generation = 0;
        while (true) {
            std::function<void(std::uint32_t)> work;
            {
                std::unique_lock lock(mutex_);
                start_condition_.wait(lock, [&] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                work = work_;
            }
            starts_[worker] = std::chrono::steady_clock::now();
            work(worker);
            {
                std::lock_guard lock(mutex_);
                ++completed_;
            }
            complete_condition_.notify_one();
        }
    }

    std::uint32_t count_{};
    std::vector<std::chrono::steady_clock::time_point> starts_;
    std::vector<std::jthread> workers_;
    std::function<void(std::uint32_t)> work_;
    std::mutex mutex_;
    std::condition_variable start_condition_;
    std::condition_variable complete_condition_;
    std::uint64_t generation_{};
    std::uint32_t completed_{};
    bool stopping_{};
};

struct alignas(64) WorkerCount {
    std::uint64_t matches{};
    std::uint64_t promo{};
};

}  // namespace

HashBuildVerification part_hash_build_reference(tpch::PartView part) {
    const PreparedHash prepared(part);
    return verify_hash(prepared.storage, part);
}

HashProbeCountRun part_hash_probe_count_reference(
    tpch::LineitemView probe,
    tpch::PartView build) {
    if (probe.part_key.empty()) {
        throw std::invalid_argument("part hash probe input must not be empty");
    }
    const PreparedHash prepared(build);
    HashProbeCountRun result;
    for (const auto key : probe.part_key) {
        bool promo = false;
        if (prepared.storage.lookup(key, promo)) {
            ++result.match_count;
            result.promo_match_count += promo;
        }
    }
    return result;
}

std::vector<HashMatchRecord> part_hash_probe_materialize_reference(
    tpch::LineitemView probe,
    tpch::PartView build) {
    if (probe.part_key.empty() ||
        probe.part_key.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "part hash materialize requires 1..2^32-1 probe rows");
    }
    const PreparedHash prepared(build);
    std::vector<HashMatchRecord> result;
    result.reserve(probe.part_key.size());
    for (std::size_t row = 0; row < probe.part_key.size(); ++row) {
        bool promo = false;
        if (prepared.storage.lookup(probe.part_key[row], promo)) {
            result.push_back(HashMatchRecord{
                static_cast<std::uint32_t>(row),
                static_cast<std::uint32_t>(promo)});
        }
    }
    return result;
}

struct PartHashBuild::Impl {
    tpch::PartView build;
    HashStorage hash;
    PersistentWorkerTeam workers;
    detail::DynamicChunkCursor clear_chunks;
    detail::DynamicChunkCursor build_chunks;
    std::vector<WorkerCount> counts;
    std::mutex execute_mutex;

    Impl(tpch::PartView input, std::uint32_t requested_threads)
        : build(input),
          hash(input.row_count()),
          workers(requested_threads, input.row_count()),
          clear_chunks(hash.keys.size(), workers.size(), 64),
          build_chunks(input.row_count(), workers.size(), 64),
          counts(workers.size()) {
        validate_part(build);
    }

    [[nodiscard]] HashBuildRun execute() {
        std::lock_guard execute_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        clear_chunks.reset();
        workers.run([this](std::uint32_t) {
            while (const auto chunk = clear_chunks.claim()) {
                std::fill(
                    hash.keys.begin() + static_cast<std::ptrdiff_t>(chunk.begin),
                    hash.keys.begin() + static_cast<std::ptrdiff_t>(chunk.end), 0);
                std::fill(
                    hash.promo.begin() + static_cast<std::ptrdiff_t>(chunk.begin),
                    hash.promo.begin() + static_cast<std::ptrdiff_t>(chunk.end), 0);
            }
        });
        const auto compute_start = workers.earliest_start();

        build_chunks.reset();
        workers.run([this](std::uint32_t worker) {
            WorkerCount local;
            while (const auto chunk = build_chunks.claim()) {
                for (auto row = chunk.begin; row < chunk.end; ++row) {
                    const auto promo = is_promo_type(build.type.data() + row * 25);
                    if (hash.insert_atomic(build.part_key[row], promo)) {
                        ++local.matches;
                        local.promo += promo;
                    }
                }
            }
            counts[worker] = local;
        });

        HashBuildRun result;
        for (const auto& local : counts) {
            result.entry_count += local.matches;
            result.promo_entry_count += local.promo;
        }
        const auto host_end = std::chrono::steady_clock::now();
        result.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        result.compute_time_ms = std::chrono::duration<double, std::milli>(
                                     host_end - compute_start)
                                     .count();
        return result;
    }
};

PartHashBuild::PartHashBuild(tpch::PartView build, std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(build, thread_count)) {}
PartHashBuild::~PartHashBuild() = default;
PartHashBuild::PartHashBuild(PartHashBuild&&) noexcept = default;
PartHashBuild& PartHashBuild::operator=(PartHashBuild&&) noexcept = default;
HashBuildRun PartHashBuild::execute() { return impl_->execute(); }
HashBuildVerification PartHashBuild::verify() const {
    return verify_hash(impl_->hash, impl_->build);
}
std::uint32_t PartHashBuild::thread_count() const noexcept {
    return impl_->workers.size();
}

struct PartHashProbeCount::Impl {
    std::span<const std::int32_t> probe_keys;
    PreparedHash hash;
    PersistentWorkerTeam workers;
    detail::DynamicChunkCursor chunks;
    std::vector<WorkerCount> counts;
    std::mutex execute_mutex;

    Impl(
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t requested_threads)
        : probe_keys(probe.part_key),
          hash(build),
          workers(requested_threads, probe_keys.size()),
          chunks(probe_keys.size(), workers.size(), 64),
          counts(workers.size()) {
        if (probe_keys.empty()) {
            throw std::invalid_argument("part hash probe input must not be empty");
        }
    }

    [[nodiscard]] HashProbeCountRun execute() {
        std::lock_guard execute_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        chunks.reset();
        workers.run([this](std::uint32_t worker) {
            WorkerCount local;
            while (const auto chunk = chunks.claim()) {
                for (auto row = chunk.begin; row < chunk.end; ++row) {
                    bool promo = false;
                    if (hash.storage.lookup(probe_keys[row], promo)) {
                        ++local.matches;
                        local.promo += promo;
                    }
                }
            }
            counts[worker] = local;
        });
        const auto compute_start = workers.earliest_start();
        HashProbeCountRun result;
        for (const auto& local : counts) {
            result.match_count += local.matches;
            result.promo_match_count += local.promo;
        }
        const auto host_end = std::chrono::steady_clock::now();
        result.host_time_ms =
            std::chrono::duration<double, std::milli>(host_end - host_start).count();
        result.compute_time_ms = std::chrono::duration<double, std::milli>(
                                     host_end - compute_start)
                                     .count();
        return result;
    }
};

PartHashProbeCount::PartHashProbeCount(
    tpch::LineitemView probe,
    tpch::PartView build,
    std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(probe, build, thread_count)) {}
PartHashProbeCount::~PartHashProbeCount() = default;
PartHashProbeCount::PartHashProbeCount(PartHashProbeCount&&) noexcept = default;
PartHashProbeCount& PartHashProbeCount::operator=(PartHashProbeCount&&) noexcept = default;
HashProbeCountRun PartHashProbeCount::execute() { return impl_->execute(); }
std::uint32_t PartHashProbeCount::thread_count() const noexcept {
    return impl_->workers.size();
}

struct PartHashProbeMaterialize::Impl {
    std::span<const std::int32_t> probe_keys;
    PreparedHash hash;
    PersistentWorkerTeam workers;
    detail::DynamicChunkCursor chunks;
    std::vector<std::uint64_t> chunk_counts;
    std::vector<std::uint64_t> chunk_offsets;
    std::vector<HashMatchRecord> records;
    std::uint64_t output_count{};
    std::mutex execute_mutex;

    Impl(
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t requested_threads)
        : probe_keys(probe.part_key),
          hash(build),
          workers(requested_threads, probe_keys.size()),
          chunks(probe_keys.size(), workers.size(), 64) {
        if (probe_keys.empty() ||
            probe_keys.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "part hash materialize requires 1..2^32-1 probe rows");
        }
        const auto chunk_count =
            (probe_keys.size() + chunks.chunk_size() - 1) / chunks.chunk_size();
        chunk_counts.resize(chunk_count);
        chunk_offsets.resize(chunk_count);
        records.resize(probe_keys.size());
    }

    [[nodiscard]] HashProbeMaterializeRun execute() {
        std::lock_guard execute_lock(execute_mutex);
        const auto host_start = std::chrono::steady_clock::now();
        chunks.reset();
        workers.run([this](std::uint32_t) {
            while (const auto chunk = chunks.claim()) {
                std::uint64_t count = 0;
                for (auto row = chunk.begin; row < chunk.end; ++row) {
                    bool promo = false;
                    count += hash.storage.lookup(probe_keys[row], promo);
                }
                chunk_counts[chunk.begin / chunks.chunk_size()] = count;
            }
        });
        const auto compute_start = workers.earliest_start();

        output_count = 0;
        for (std::size_t chunk = 0; chunk < chunk_counts.size(); ++chunk) {
            chunk_offsets[chunk] = output_count;
            output_count += chunk_counts[chunk];
        }

        chunks.reset();
        workers.run([this](std::uint32_t) {
            while (const auto chunk = chunks.claim()) {
                auto output = chunk_offsets[chunk.begin / chunks.chunk_size()];
                for (auto row = chunk.begin; row < chunk.end; ++row) {
                    bool promo = false;
                    if (hash.storage.lookup(probe_keys[row], promo)) {
                        records[output++] = HashMatchRecord{
                            static_cast<std::uint32_t>(row),
                            static_cast<std::uint32_t>(promo)};
                    }
                }
            }
        });
        const auto host_end = std::chrono::steady_clock::now();
        return HashProbeMaterializeRun{
            output_count,
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            std::chrono::duration<double, std::milli>(host_end - compute_start).count()};
    }
};

PartHashProbeMaterialize::PartHashProbeMaterialize(
    tpch::LineitemView probe,
    tpch::PartView build,
    std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(probe, build, thread_count)) {}
PartHashProbeMaterialize::~PartHashProbeMaterialize() = default;
PartHashProbeMaterialize::PartHashProbeMaterialize(
    PartHashProbeMaterialize&&) noexcept = default;
PartHashProbeMaterialize& PartHashProbeMaterialize::operator=(
    PartHashProbeMaterialize&&) noexcept = default;
HashProbeMaterializeRun PartHashProbeMaterialize::execute() {
    return impl_->execute();
}
std::span<const HashMatchRecord> PartHashProbeMaterialize::output() const noexcept {
    return {
        impl_->records.data(),
        static_cast<std::size_t>(impl_->output_count)};
}
std::uint32_t PartHashProbeMaterialize::thread_count() const noexcept {
    return impl_->workers.size();
}

}  // namespace joule::operators::cpu
