#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/tpch_q14.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace joule::operators::gpu {
namespace {

struct Q14Pair {
    std::int64_t promo;
    std::int64_t total;
};
static_assert(sizeof(Q14Pair) == 16);

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) return "unknown Metal error";
    const char* text = error->localizedDescription()->utf8String();
    return text != nullptr ? text : "unknown Metal error";
}

[[nodiscard]] std::uint32_t group_count(
    std::uint32_t values, std::uint32_t width, std::uint32_t per_thread) {
    const auto denominator = static_cast<std::uint64_t>(width) * per_thread;
    return static_cast<std::uint32_t>((values + denominator - 1) / denominator);
}

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(MTL::Device* device, std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
}

struct HostHash {
    std::vector<std::int32_t> keys;
    std::vector<std::uint8_t> promo;

    explicit HostHash(tpch::PartView part) {
        if (part.row_count() == 0 || part.type.size() != part.row_count() * 25) {
            throw std::invalid_argument("TPC-H part columns must be non-empty and equally sized");
        }
        const auto capacity = std::bit_ceil(std::max<std::size_t>(2, part.row_count() * 2));
        if (capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU Q14 part hash table is too large");
        }
        keys.assign(capacity, 0);
        promo.assign(capacity, 0);
        const auto mask = capacity - 1;
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
};

}  // namespace

struct TpchQ14HashJoin::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* first_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    std::vector<MTL::Buffer*> inputs;
    MTL::Buffer* hash_keys{};
    MTL::Buffer* hash_promo{};
    std::vector<MTL::Buffer*> reductions;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t hash_mask{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metallib,
        tpch::LineitemView input,
        tpch::PartView part,
        std::uint32_t requested_width)
        : width(requested_width) {
        const auto rows = input.row_count();
        if (rows == 0 || input.part_key.size() != rows ||
            input.extended_price.size() != rows || input.discount.size() != rows ||
            input.ship_date_yyyymmdd.size() != rows) {
            throw std::invalid_argument("TPC-H Q14 lineitem columns must be non-empty and equally sized");
        }
        if (rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU TPC-H Q14 supports at most 2^32-1 rows");
        }
        if (width < 32 || width > 512 || !std::has_single_bit(width)) {
            throw std::invalid_argument("threadgroup width must be a power of two from 32 to 512");
        }
        const HostHash host_hash(part);

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        MTL::Function* first_function = nullptr;
        MTL::Function* reduce_function = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            if (device->name() != nullptr && device->name()->utf8String() != nullptr) {
                device_name_value = device->name()->utf8String();
            }
            queue = device->newCommandQueue();
            if (queue == nullptr) throw std::runtime_error("could not create Metal queue");
            NS::Error* error = nullptr;
            const auto path = metallib.string();
            library = device->newLibrary(
                NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
            if (library == nullptr) {
                throw std::runtime_error("could not load metallib: " + error_description(error));
            }
            first_function = library->newFunction(
                NS::String::string("tpch_q14_first", NS::UTF8StringEncoding));
            reduce_function = library->newFunction(
                NS::String::string("tpch_q14_reduce", NS::UTF8StringEncoding));
            if (first_function == nullptr || reduce_function == nullptr) {
                throw std::runtime_error("TPC-H Q14 Metal functions are missing");
            }
            first_pipeline = device->newComputePipelineState(first_function, &error);
            if (first_pipeline == nullptr) {
                throw std::runtime_error("could not create Q14 pipeline: " + error_description(error));
            }
            error = nullptr;
            reduce_pipeline = device->newComputePipelineState(reduce_function, &error);
            if (reduce_pipeline == nullptr) {
                throw std::runtime_error("could not create Q14 reduction: " + error_description(error));
            }
            execution_width_value = static_cast<std::uint32_t>(first_pipeline->threadExecutionWidth());
            const auto maximum = std::min(
                first_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup());
            if (width % execution_width_value != 0 || width > maximum) {
                throw std::invalid_argument("threadgroup width is incompatible with Q14 kernels");
            }

            row_count = static_cast<std::uint32_t>(rows);
            inputs = {
                wrap_shared(device, input.part_key),
                wrap_shared(device, input.extended_price),
                wrap_shared(device, input.discount),
                wrap_shared(device, input.ship_date_yyyymmdd)};
            hash_keys = device->newBuffer(
                host_hash.keys.data(), host_hash.keys.size() * sizeof(std::int32_t),
                MTL::ResourceStorageModeShared);
            hash_promo = device->newBuffer(
                host_hash.promo.data(), host_hash.promo.size(),
                MTL::ResourceStorageModeShared);
            hash_mask = static_cast<std::uint32_t>(host_hash.keys.size() - 1);
            if (std::any_of(inputs.begin(), inputs.end(), [](auto* buffer) { return buffer == nullptr; }) ||
                hash_keys == nullptr || hash_promo == nullptr) {
                throw std::runtime_error("could not allocate Q14 Metal input buffers");
            }

            auto count = group_count(row_count, width, 16);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = group_count(count, width, 4);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) * sizeof(Q14Pair), storage);
                if (buffer == nullptr) throw std::runtime_error("could not allocate Q14 reduction buffer");
                reductions.push_back(buffer);
            }

            reduce_function->release();
            first_function->release();
            library->release();
            pool->release();
        } catch (...) {
            if (reduce_function != nullptr) reduce_function->release();
            if (first_function != nullptr) first_function->release();
            if (library != nullptr) library->release();
            pool->release();
            release_resources();
            throw;
        }
    }

    ~Impl() { release_resources(); }

    void release_resources() noexcept {
        for (auto* buffer : reductions) if (buffer != nullptr) buffer->release();
        reductions.clear();
        for (auto* buffer : inputs) if (buffer != nullptr) buffer->release();
        inputs.clear();
        if (hash_promo != nullptr) hash_promo->release();
        if (hash_keys != nullptr) hash_keys->release();
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (first_pipeline != nullptr) first_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        hash_promo = nullptr;
        hash_keys = nullptr;
        reduce_pipeline = nullptr;
        first_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    [[nodiscard]] TpchQ14Result execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder = command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error("could not create Q14 Metal command buffer");
            }
            encoder->setComputePipelineState(first_pipeline);
            for (std::uint32_t index = 0; index < inputs.size(); ++index) {
                encoder->setBuffer(inputs[index], 0, index);
            }
            encoder->setBuffer(hash_keys, 0, 4);
            encoder->setBuffer(hash_promo, 0, 5);
            encoder->setBuffer(reductions.front(), 0, 6);
            encoder->setBytes(&row_count, sizeof(row_count), 7);
            encoder->setBytes(&hash_mask, sizeof(hash_mask), 8);
            const MTL::Size threads(width, 1, 1);
            encoder->dispatchThreadgroups(MTL::Size(level_counts.front(), 1, 1), threads);
            auto input_count = level_counts.front();
            for (std::size_t level = 1; level < reductions.size(); ++level) {
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(reductions[level - 1], 0, 0);
                encoder->setBuffer(reductions[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->dispatchThreadgroups(MTL::Size(level_counts[level], 1, 1), threads);
                input_count = level_counts[level];
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error("Q14 Metal command failed: " + error_description(command->error()));
            }
            const auto* pair = static_cast<const Q14Pair*>(reductions.back()->contents());
            TpchQ14Result result;
            result.promo_revenue_1e4_usd = pair->promo;
            result.total_revenue_1e4_usd = pair->total;
            result.host_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            const auto gpu_start = command->GPUStartTime();
            const auto gpu_end = command->GPUEndTime();
            result.gpu_time_ms = gpu_end >= gpu_start ? (gpu_end - gpu_start) * 1'000.0 : 0.0;
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

TpchQ14HashJoin::TpchQ14HashJoin(
    const std::filesystem::path& metallib,
    tpch::LineitemView lineitem,
    tpch::PartView part,
    std::uint32_t width)
    : impl_(std::make_unique<Impl>(metallib, lineitem, part, width)) {}
TpchQ14HashJoin::~TpchQ14HashJoin() = default;
TpchQ14HashJoin::TpchQ14HashJoin(TpchQ14HashJoin&&) noexcept = default;
TpchQ14HashJoin& TpchQ14HashJoin::operator=(TpchQ14HashJoin&&) noexcept = default;
TpchQ14Result TpchQ14HashJoin::execute() { return impl_->execute(); }
const std::string& TpchQ14HashJoin::device_name() const noexcept { return impl_->device_name_value; }
std::uint32_t TpchQ14HashJoin::execution_width() const noexcept {
    return impl_->execution_width_value;
}

}  // namespace joule::operators::gpu
