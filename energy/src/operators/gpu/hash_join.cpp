#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/hash_join.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

namespace joule::operators::gpu {
namespace {

struct CountPair {
    std::uint32_t matches;
    std::uint32_t promo;
};
static_assert(sizeof(CountPair) == 8);
static_assert(sizeof(HashMatchRecord) == 8);

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "unknown Metal error";
    }
    const auto* text = error->localizedDescription()->utf8String();
    return text != nullptr ? text : "unknown Metal error";
}

[[nodiscard]] std::string metal_device_name(MTL::Device* device) {
    return device != nullptr && device->name() != nullptr &&
            device->name()->utf8String() != nullptr
        ? device->name()->utf8String()
        : "";
}

[[nodiscard]] double command_gpu_ms(MTL::CommandBuffer* command) {
    const auto begin = command->GPUStartTime();
    const auto end = command->GPUEndTime();
    return end >= begin ? (end - begin) * 1'000.0 : 0.0;
}

[[nodiscard]] MTL::ComputePipelineState* make_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const char* name) {
    auto* function = library->newFunction(
        NS::String::string(name, NS::UTF8StringEncoding));
    if (function == nullptr) {
        throw std::runtime_error(std::string("missing Metal function: ") + name);
    }
    NS::Error* error = nullptr;
    auto* pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
        throw std::runtime_error(
            std::string("could not create pipeline '") + name + "': " +
            error_description(error));
    }
    return pipeline;
}

[[nodiscard]] MTL::Library* load_library(
    MTL::Device* device,
    const std::filesystem::path& library_path) {
    NS::Error* error = nullptr;
    const auto path = library_path.string();
    auto* library = device->newLibrary(
        NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
    if (library == nullptr) {
        throw std::runtime_error(
            "could not load metallib: " + error_description(error));
    }
    return library;
}

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(
    MTL::Device* device,
    std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
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
    if (part.row_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Metal part hash supports at most 2^32-1 rows");
    }
    if (std::any_of(part.part_key.begin(), part.part_key.end(), [](std::int32_t key) {
            return key <= 0;
        })) {
        throw std::invalid_argument("part hash build keys must be positive");
    }
}

[[nodiscard]] std::uint32_t checked_capacity(std::size_t rows) {
    if (rows > std::numeric_limits<std::uint32_t>::max() / 2ULL) {
        throw std::invalid_argument("Metal part hash table is too large");
    }
    const auto requested = std::max<std::size_t>(2, rows * 2);
    const auto capacity = std::bit_ceil(requested);
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Metal part hash table capacity overflow");
    }
    return static_cast<std::uint32_t>(capacity);
}

[[nodiscard]] std::uint32_t group_count(
    std::uint32_t values,
    std::uint32_t width,
    std::uint32_t values_per_thread) {
    const auto denominator =
        static_cast<std::uint64_t>(width) * values_per_thread;
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(values) + denominator - 1) / denominator);
}

void validate_width(
    std::uint32_t width,
    std::initializer_list<MTL::ComputePipelineState*> pipelines,
    std::uint32_t& execution_width) {
    if (width < 32 || width > 512 || !std::has_single_bit(width)) {
        throw std::invalid_argument(
            "threadgroup width must be a power of two from 32 to 512");
    }
    auto maximum = std::numeric_limits<NS::UInteger>::max();
    execution_width = 0;
    for (auto* pipeline : pipelines) {
        if (pipeline == nullptr) continue;
        if (execution_width == 0) {
            execution_width =
                static_cast<std::uint32_t>(pipeline->threadExecutionWidth());
        }
        maximum = std::min(maximum, pipeline->maxTotalThreadsPerThreadgroup());
    }
    if (execution_width == 0 || width % execution_width != 0 || width > maximum) {
        throw std::invalid_argument(
            "threadgroup width is incompatible with hash kernels");
    }
}

struct HostHash {
    std::vector<std::int32_t> keys;
    std::vector<std::uint32_t> promo;
    std::uint32_t mask{};

    explicit HostHash(tpch::PartView part) {
        validate_part(part);
        const auto capacity = checked_capacity(part.row_count());
        keys.assign(capacity, 0);
        promo.assign(capacity, 0);
        mask = capacity - 1;
        for (std::size_t row = 0; row < part.row_count(); ++row) {
            const auto key = part.part_key[row];
            auto slot =
                static_cast<std::uint32_t>(key) * 2'654'435'761U & mask;
            while (keys[slot] != 0 && keys[slot] != key) {
                slot = (slot + 1) & mask;
            }
            if (keys[slot] == key) {
                throw std::invalid_argument(
                    "part hash build requires unique part keys");
            }
            keys[slot] = key;
            promo[slot] = static_cast<std::uint32_t>(
                is_promo_type(part.type.data() + row * 25));
        }
    }
};

[[nodiscard]] bool host_lookup(
    std::int32_t key,
    const std::int32_t* keys,
    const std::uint32_t* promo,
    std::uint32_t mask,
    bool& is_promo) noexcept {
    if (key <= 0) return false;
    auto slot = static_cast<std::uint32_t>(key) * 2'654'435'761U & mask;
    while (keys[slot] != 0) {
        if (keys[slot] == key) {
            is_promo = promo[slot] != 0;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

}  // namespace

struct PartHashBuild::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* clear_pipeline{};
    MTL::ComputePipelineState* build_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* part_key{};
    MTL::Buffer* part_type{};
    MTL::Buffer* hash_keys{};
    MTL::Buffer* hash_promo{};
    std::vector<MTL::Buffer*> reductions;
    std::vector<std::uint32_t> level_counts;
    tpch::PartView build;
    std::uint32_t row_count{};
    std::uint32_t capacity{};
    std::uint32_t hash_mask{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    bool has_executed{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        tpch::PartView input,
        std::uint32_t requested_width)
        : build(input), width(requested_width) {
        validate_part(build);
        row_count = static_cast<std::uint32_t>(build.row_count());
        capacity = checked_capacity(build.row_count());
        hash_mask = capacity - 1;

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            if (queue == nullptr) throw std::runtime_error("could not create Metal queue");
            library = load_library(device, library_path);
            clear_pipeline = make_pipeline(device, library, "part_hash_clear");
            build_pipeline = make_pipeline(device, library, "part_hash_build_atomic");
            reduce_pipeline =
                make_pipeline(device, library, "part_hash_probe_count_reduce");
            validate_width(
                width, {clear_pipeline, build_pipeline, reduce_pipeline},
                execution_width_value);

            part_key = wrap_shared(device, build.part_key);
            part_type = wrap_shared(device, build.type);
            hash_keys = device->newBuffer(
                static_cast<NS::UInteger>(capacity) * sizeof(std::int32_t),
                MTL::ResourceStorageModeShared);
            hash_promo = device->newBuffer(
                static_cast<NS::UInteger>(capacity) * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            if (part_key == nullptr || part_type == nullptr || hash_keys == nullptr ||
                hash_promo == nullptr) {
                throw std::runtime_error(
                    "could not allocate Metal hash-build resources");
            }

            auto count = group_count(row_count, width, 1);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = group_count(count, width, 4);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared
                    : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) *
                        sizeof(CountPair),
                    storage);
                if (buffer == nullptr) {
                    throw std::runtime_error(
                        "could not allocate hash-build reduction");
                }
                reductions.push_back(buffer);
            }
            library->release();
            pool->release();
        } catch (...) {
            if (library != nullptr) library->release();
            pool->release();
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        for (auto* buffer : reductions) {
            if (buffer != nullptr) buffer->release();
        }
        reductions.clear();
        for (auto** buffer : {
                 &hash_promo, &hash_keys, &part_type, &part_key}) {
            if (*buffer != nullptr) {
                (*buffer)->release();
                *buffer = nullptr;
            }
        }
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (build_pipeline != nullptr) build_pipeline->release();
        if (clear_pipeline != nullptr) clear_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        reduce_pipeline = nullptr;
        build_pipeline = nullptr;
        clear_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    void dispatch(
        MTL::ComputeCommandEncoder* encoder,
        std::uint32_t items) const {
        const auto groups = (items + width - 1U) / width;
        encoder->dispatchThreadgroups(
            MTL::Size(groups, 1, 1), MTL::Size(width, 1, 1));
    }

    [[nodiscard]] HashBuildRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder =
                command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error(
                    "could not create Metal hash-build command buffer");
            }
            encoder->setComputePipelineState(clear_pipeline);
            encoder->setBuffer(hash_keys, 0, 0);
            encoder->setBuffer(hash_promo, 0, 1);
            encoder->setBytes(&capacity, sizeof(capacity), 2);
            dispatch(encoder, capacity);

            encoder->setComputePipelineState(build_pipeline);
            encoder->setBuffer(part_key, 0, 0);
            encoder->setBuffer(part_type, 0, 1);
            encoder->setBuffer(hash_keys, 0, 2);
            encoder->setBuffer(hash_promo, 0, 3);
            encoder->setBuffer(reductions.front(), 0, 4);
            encoder->setBytes(&row_count, sizeof(row_count), 5);
            encoder->setBytes(&hash_mask, sizeof(hash_mask), 6);
            dispatch(encoder, row_count);

            const MTL::Size threads(width, 1, 1);
            auto input_count = level_counts.front();
            for (std::size_t level = 1; level < reductions.size(); ++level) {
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(reductions[level - 1], 0, 0);
                encoder->setBuffer(reductions[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->dispatchThreadgroups(
                    MTL::Size(level_counts[level], 1, 1), threads);
                input_count = level_counts[level];
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "Metal hash build failed: " +
                    error_description(command->error()));
            }
            const auto* values =
                static_cast<const CountPair*>(reductions.back()->contents());
            has_executed = true;
            const HashBuildRun result{
                values->matches,
                values->promo,
                std::chrono::duration<double, std::milli>(
                    host_end - host_start).count(),
                command_gpu_ms(command)};
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }

    [[nodiscard]] HashBuildVerification verify() const {
        HashBuildVerification result;
        if (!has_executed) return result;
        const auto* keys = static_cast<const std::int32_t*>(hash_keys->contents());
        const auto* promo =
            static_cast<const std::uint32_t*>(hash_promo->contents());
        for (std::uint32_t slot = 0; slot < capacity; ++slot) {
            if (keys[slot] == 0) continue;
            ++result.entry_count;
            result.promo_entry_count += promo[slot] != 0;
            result.key_sum += static_cast<std::uint32_t>(keys[slot]);
        }
        result.valid = result.entry_count == build.row_count();
        for (std::size_t row = 0; result.valid && row < build.row_count(); ++row) {
            bool stored_promo = false;
            result.valid =
                host_lookup(
                    build.part_key[row], keys, promo, hash_mask, stored_promo) &&
                stored_promo ==
                    is_promo_type(build.type.data() + row * 25);
        }
        return result;
    }
};

PartHashBuild::PartHashBuild(
    const std::filesystem::path& metal_library,
    tpch::PartView build,
    std::uint32_t threadgroup_width)
    : impl_(std::make_unique<Impl>(
          metal_library, build, threadgroup_width)) {}
PartHashBuild::~PartHashBuild() = default;
PartHashBuild::PartHashBuild(PartHashBuild&&) noexcept = default;
PartHashBuild& PartHashBuild::operator=(PartHashBuild&&) noexcept = default;
HashBuildRun PartHashBuild::execute() { return impl_->execute(); }
HashBuildVerification PartHashBuild::verify() const { return impl_->verify(); }
const std::string& PartHashBuild::device_name() const noexcept {
    return impl_->name;
}
std::uint32_t PartHashBuild::execution_width() const noexcept {
    return impl_->execution_width_value;
}

struct PartHashProbeCount::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* first_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* probe_keys{};
    MTL::Buffer* hash_keys{};
    MTL::Buffer* hash_promo{};
    std::vector<MTL::Buffer*> reductions;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t hash_mask{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t requested_width)
        : width(requested_width) {
        if (probe.part_key.empty() ||
            probe.part_key.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "Metal hash probe requires 1..2^32-1 rows");
        }
        const HostHash prepared(build);
        row_count = static_cast<std::uint32_t>(probe.part_key.size());
        hash_mask = prepared.mask;

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            if (queue == nullptr) throw std::runtime_error("could not create Metal queue");
            library = load_library(device, library_path);
            first_pipeline =
                make_pipeline(device, library, "part_hash_probe_count_first");
            reduce_pipeline =
                make_pipeline(device, library, "part_hash_probe_count_reduce");
            validate_width(
                width, {first_pipeline, reduce_pipeline},
                execution_width_value);

            probe_keys = wrap_shared(device, probe.part_key);
            hash_keys = device->newBuffer(
                prepared.keys.data(),
                prepared.keys.size() * sizeof(std::int32_t),
                MTL::ResourceStorageModeShared);
            hash_promo = device->newBuffer(
                prepared.promo.data(),
                prepared.promo.size() * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            if (probe_keys == nullptr || hash_keys == nullptr ||
                hash_promo == nullptr) {
                throw std::runtime_error(
                    "could not allocate Metal hash-probe resources");
            }

            auto count = group_count(row_count, width, 16);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = group_count(count, width, 4);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared
                    : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) *
                        sizeof(CountPair),
                    storage);
                if (buffer == nullptr) {
                    throw std::runtime_error(
                        "could not allocate hash-count reduction");
                }
                reductions.push_back(buffer);
            }
            library->release();
            pool->release();
        } catch (...) {
            if (library != nullptr) library->release();
            pool->release();
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        for (auto* buffer : reductions) {
            if (buffer != nullptr) buffer->release();
        }
        reductions.clear();
        for (auto** buffer : {&hash_promo, &hash_keys, &probe_keys}) {
            if (*buffer != nullptr) {
                (*buffer)->release();
                *buffer = nullptr;
            }
        }
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (first_pipeline != nullptr) first_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        reduce_pipeline = nullptr;
        first_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    [[nodiscard]] HashProbeCountRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder =
                command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error(
                    "could not create Metal hash-count command buffer");
            }
            const MTL::Size threads(width, 1, 1);
            encoder->setComputePipelineState(first_pipeline);
            encoder->setBuffer(probe_keys, 0, 0);
            encoder->setBuffer(hash_keys, 0, 1);
            encoder->setBuffer(hash_promo, 0, 2);
            encoder->setBuffer(reductions.front(), 0, 3);
            encoder->setBytes(&row_count, sizeof(row_count), 4);
            encoder->setBytes(&hash_mask, sizeof(hash_mask), 5);
            encoder->dispatchThreadgroups(
                MTL::Size(level_counts.front(), 1, 1), threads);

            auto input_count = level_counts.front();
            for (std::size_t level = 1; level < reductions.size(); ++level) {
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(reductions[level - 1], 0, 0);
                encoder->setBuffer(reductions[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->dispatchThreadgroups(
                    MTL::Size(level_counts[level], 1, 1), threads);
                input_count = level_counts[level];
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "Metal hash count failed: " +
                    error_description(command->error()));
            }
            const auto* result =
                static_cast<const CountPair*>(reductions.back()->contents());
            const HashProbeCountRun run{
                result->matches,
                result->promo,
                std::chrono::duration<double, std::milli>(
                    host_end - host_start).count(),
                command_gpu_ms(command)};
            pool->release();
            return run;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

PartHashProbeCount::PartHashProbeCount(
    const std::filesystem::path& metal_library,
    tpch::LineitemView probe,
    tpch::PartView build,
    std::uint32_t threadgroup_width)
    : impl_(std::make_unique<Impl>(
          metal_library, probe, build, threadgroup_width)) {}
PartHashProbeCount::~PartHashProbeCount() = default;
PartHashProbeCount::PartHashProbeCount(PartHashProbeCount&&) noexcept = default;
PartHashProbeCount& PartHashProbeCount::operator=(
    PartHashProbeCount&&) noexcept = default;
HashProbeCountRun PartHashProbeCount::execute() { return impl_->execute(); }
const std::string& PartHashProbeCount::device_name() const noexcept {
    return impl_->name;
}
std::uint32_t PartHashProbeCount::execution_width() const noexcept {
    return impl_->execution_width_value;
}

struct PartHashProbeMaterialize::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* block_count_pipeline{};
    MTL::ComputePipelineState* scan_pipeline{};
    MTL::ComputePipelineState* add_pipeline{};
    MTL::ComputePipelineState* scatter_pipeline{};
    MTL::ComputePipelineState* output_count_pipeline{};
    MTL::Buffer* probe_keys{};
    MTL::Buffer* hash_keys{};
    MTL::Buffer* hash_promo{};
    MTL::Buffer* output_records{};
    MTL::Buffer* output_count{};
    MTL::Buffer* dummy_sum{};
    std::vector<MTL::Buffer*> level_inputs;
    std::vector<MTL::Buffer*> level_offsets;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t block_count{};
    std::uint32_t hash_mask{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        tpch::LineitemView probe,
        tpch::PartView build,
        std::uint32_t requested_width)
        : width(requested_width) {
        if (probe.part_key.empty() ||
            probe.part_key.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "Metal hash materialize requires 1..2^32-1 probe rows");
        }
        const HostHash prepared(build);
        row_count = static_cast<std::uint32_t>(probe.part_key.size());
        hash_mask = prepared.mask;

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            if (queue == nullptr) throw std::runtime_error("could not create Metal queue");
            library = load_library(device, library_path);
            block_count_pipeline =
                make_pipeline(device, library, "part_hash_probe_block_counts");
            scan_pipeline =
                make_pipeline(device, library, "part_hash_exclusive_scan_u32");
            add_pipeline =
                make_pipeline(device, library, "part_hash_add_scan_offsets");
            scatter_pipeline =
                make_pipeline(device, library, "part_hash_probe_scatter");
            output_count_pipeline =
                make_pipeline(device, library, "part_hash_materialized_count");
            validate_width(
                width,
                {block_count_pipeline, scan_pipeline, add_pipeline,
                 scatter_pipeline},
                execution_width_value);

            probe_keys = wrap_shared(device, probe.part_key);
            hash_keys = device->newBuffer(
                prepared.keys.data(),
                prepared.keys.size() * sizeof(std::int32_t),
                MTL::ResourceStorageModeShared);
            hash_promo = device->newBuffer(
                prepared.promo.data(),
                prepared.promo.size() * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            output_records = device->newBuffer(
                static_cast<NS::UInteger>(row_count) *
                    sizeof(HashMatchRecord),
                MTL::ResourceStorageModeShared);
            output_count = device->newBuffer(
                sizeof(std::uint32_t), MTL::ResourceStorageModeShared);
            dummy_sum = device->newBuffer(
                sizeof(std::uint32_t), MTL::ResourceStorageModePrivate);
            if (probe_keys == nullptr || hash_keys == nullptr ||
                hash_promo == nullptr || output_records == nullptr ||
                output_count == nullptr || dummy_sum == nullptr) {
                throw std::runtime_error(
                    "could not allocate Metal hash-materialize resources");
            }

            block_count = group_count(row_count, width, 8);
            auto count = block_count;
            while (true) {
                level_counts.push_back(count);
                level_inputs.push_back(device->newBuffer(
                    static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                    MTL::ResourceStorageModePrivate));
                level_offsets.push_back(device->newBuffer(
                    static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                    MTL::ResourceStorageModePrivate));
                if (level_inputs.back() == nullptr ||
                    level_offsets.back() == nullptr) {
                    throw std::runtime_error(
                        "could not allocate hash prefix-scan levels");
                }
                if (count <= width) break;
                count = (count + width - 1U) / width;
            }
            library->release();
            pool->release();
        } catch (...) {
            if (library != nullptr) library->release();
            pool->release();
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        for (auto* buffer : level_offsets) {
            if (buffer != nullptr) buffer->release();
        }
        for (auto* buffer : level_inputs) {
            if (buffer != nullptr) buffer->release();
        }
        level_offsets.clear();
        level_inputs.clear();
        for (auto** buffer : {
                 &dummy_sum, &output_count, &output_records, &hash_promo,
                 &hash_keys, &probe_keys}) {
            if (*buffer != nullptr) {
                (*buffer)->release();
                *buffer = nullptr;
            }
        }
        for (auto** pipeline : {
                 &output_count_pipeline, &scatter_pipeline, &add_pipeline,
                 &scan_pipeline, &block_count_pipeline}) {
            if (*pipeline != nullptr) {
                (*pipeline)->release();
                *pipeline = nullptr;
            }
        }
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        queue = nullptr;
        device = nullptr;
    }

    void dispatch_items(
        MTL::ComputeCommandEncoder* encoder,
        std::uint32_t items) const {
        encoder->dispatchThreadgroups(
            MTL::Size((items + width - 1U) / width, 1, 1),
            MTL::Size(width, 1, 1));
    }

    [[nodiscard]] HashProbeMaterializeRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder =
                command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error(
                    "could not create Metal hash-materialize command buffer");
            }
            const MTL::Size threads(width, 1, 1);
            encoder->setComputePipelineState(block_count_pipeline);
            encoder->setBuffer(probe_keys, 0, 0);
            encoder->setBuffer(hash_keys, 0, 1);
            encoder->setBuffer(hash_promo, 0, 2);
            encoder->setBuffer(level_inputs.front(), 0, 3);
            encoder->setBytes(&row_count, sizeof(row_count), 4);
            encoder->setBytes(&hash_mask, sizeof(hash_mask), 5);
            encoder->dispatchThreadgroups(
                MTL::Size(block_count, 1, 1), threads);

            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto count = level_counts[level];
                encoder->setComputePipelineState(scan_pipeline);
                encoder->setBuffer(level_inputs[level], 0, 0);
                encoder->setBuffer(level_offsets[level], 0, 1);
                encoder->setBuffer(
                    level + 1 < level_inputs.size()
                        ? level_inputs[level + 1]
                        : dummy_sum,
                    0, 2);
                encoder->setBytes(&count, sizeof(count), 3);
                dispatch_items(encoder, count);
            }
            for (std::size_t level = level_counts.size(); level-- > 1;) {
                const auto lower = level - 1;
                const auto count = level_counts[lower];
                encoder->setComputePipelineState(add_pipeline);
                encoder->setBuffer(level_offsets[lower], 0, 0);
                encoder->setBuffer(level_offsets[level], 0, 1);
                encoder->setBytes(&count, sizeof(count), 2);
                encoder->setBytes(&width, sizeof(width), 3);
                dispatch_items(encoder, count);
            }

            encoder->setComputePipelineState(scatter_pipeline);
            encoder->setBuffer(probe_keys, 0, 0);
            encoder->setBuffer(hash_keys, 0, 1);
            encoder->setBuffer(hash_promo, 0, 2);
            encoder->setBuffer(level_offsets.front(), 0, 3);
            encoder->setBuffer(output_records, 0, 4);
            encoder->setBytes(&row_count, sizeof(row_count), 5);
            encoder->setBytes(&hash_mask, sizeof(hash_mask), 6);
            encoder->dispatchThreadgroups(
                MTL::Size(block_count, 1, 1), threads);

            encoder->setComputePipelineState(output_count_pipeline);
            encoder->setBuffer(level_inputs.front(), 0, 0);
            encoder->setBuffer(level_offsets.front(), 0, 1);
            encoder->setBuffer(output_count, 0, 2);
            encoder->setBytes(&block_count, sizeof(block_count), 3);
            encoder->dispatchThreadgroups(
                MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "Metal hash materialize failed: " +
                    error_description(command->error()));
            }
            const auto count =
                *static_cast<const std::uint32_t*>(output_count->contents());
            const HashProbeMaterializeRun result{
                count,
                std::chrono::duration<double, std::milli>(
                    host_end - host_start).count(),
                command_gpu_ms(command)};
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

PartHashProbeMaterialize::PartHashProbeMaterialize(
    const std::filesystem::path& metal_library,
    tpch::LineitemView probe,
    tpch::PartView build,
    std::uint32_t threadgroup_width)
    : impl_(std::make_unique<Impl>(
          metal_library, probe, build, threadgroup_width)) {}
PartHashProbeMaterialize::~PartHashProbeMaterialize() = default;
PartHashProbeMaterialize::PartHashProbeMaterialize(
    PartHashProbeMaterialize&&) noexcept = default;
PartHashProbeMaterialize& PartHashProbeMaterialize::operator=(
    PartHashProbeMaterialize&&) noexcept = default;
HashProbeMaterializeRun PartHashProbeMaterialize::execute() {
    return impl_->execute();
}
std::span<const HashMatchRecord> PartHashProbeMaterialize::output() const noexcept {
    const auto count =
        *static_cast<const std::uint32_t*>(impl_->output_count->contents());
    return {
        static_cast<const HashMatchRecord*>(impl_->output_records->contents()),
        count};
}
const std::string& PartHashProbeMaterialize::device_name() const noexcept {
    return impl_->name;
}
std::uint32_t PartHashProbeMaterialize::execution_width() const noexcept {
    return impl_->execution_width_value;
}

}  // namespace joule::operators::gpu
