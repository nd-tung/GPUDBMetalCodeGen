#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/tpch_q6_unfused.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace joule::operators::gpu {
namespace {

constexpr std::uint32_t bitmap_words_per_thread = 4;
constexpr std::uint32_t reduce_values_per_thread = 4;

struct alignas(16) Q6Pair {
    std::int64_t count;
    std::int64_t revenue;
};

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "unknown Metal error";
    }
    const char* description = error->localizedDescription()->utf8String();
    return description != nullptr ? description : "unknown Metal error";
}

[[nodiscard]] std::uint32_t group_count(
    std::uint32_t item_count,
    std::uint32_t threadgroup_width,
    std::uint32_t values_per_thread) {
    const auto values_per_group =
        static_cast<std::uint64_t>(threadgroup_width) * values_per_thread;
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(item_count) + values_per_group - 1) /
        values_per_group);
}

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(
    MTL::Device* device,
    std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()),
        values.size_bytes(),
        MTL::ResourceStorageModeShared,
        nullptr);
}

}  // namespace

struct TpchQ6Unfused::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* command_queue{};
    MTL::ComputePipelineState* bitmap_pipeline{};
    MTL::ComputePipelineState* aggregate_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* quantity_buffer{};
    MTL::Buffer* price_buffer{};
    MTL::Buffer* discount_buffer{};
    MTL::Buffer* date_buffer{};
    MTL::Buffer* bitmap_buffer{};
    std::vector<MTL::Buffer*> reduction_buffers;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t bitmap_word_count{};
    std::uint32_t threadgroup_width{};
    std::uint32_t execution_width_value{};
    std::uint32_t max_threads_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        TpchQ6UnfusedConfig config)
        : threadgroup_width(config.threadgroup_width) {
        const auto rows = input.row_count();
        if (rows == 0 || input.extended_price.size() != rows ||
            input.discount.size() != rows ||
            input.ship_date_yyyymmdd.size() != rows) {
            throw std::invalid_argument(
                "unfused TPC-H Q6 columns must be non-empty and equally sized");
        }
        if (rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "GPU unfused TPC-H Q6 supports at most 2^32-1 rows");
        }
        if (threadgroup_width < 32 || threadgroup_width > 512 ||
            !std::has_single_bit(threadgroup_width)) {
            throw std::invalid_argument(
                "threadgroup width must be a power of two between 32 and 512");
        }

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        MTL::Function* bitmap_function = nullptr;
        MTL::Function* aggregate_function = nullptr;
        MTL::Function* reduce_function = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) {
                throw std::runtime_error("no Metal device is available");
            }
            if (device->name() != nullptr &&
                device->name()->utf8String() != nullptr) {
                device_name_value = device->name()->utf8String();
            }
            command_queue = device->newCommandQueue();
            if (command_queue == nullptr) {
                throw std::runtime_error("could not create a Metal command queue");
            }

            NS::Error* error = nullptr;
            const auto library_path = metal_library.string();
            library = device->newLibrary(
                NS::String::string(
                    library_path.c_str(), NS::UTF8StringEncoding),
                &error);
            if (library == nullptr) {
                throw std::runtime_error(
                    "could not load metallib '" + library_path + "': " +
                    error_description(error));
            }
            bitmap_function = library->newFunction(NS::String::string(
                "tpch_q6_filter_bitmap", NS::UTF8StringEncoding));
            aggregate_function = library->newFunction(NS::String::string(
                "tpch_q6_revenue_from_bitmap", NS::UTF8StringEncoding));
            reduce_function = library->newFunction(NS::String::string(
                "tpch_q6_reduce_pair", NS::UTF8StringEncoding));
            if (bitmap_function == nullptr || aggregate_function == nullptr ||
                reduce_function == nullptr) {
                throw std::runtime_error(
                    "required unfused TPC-H Q6 Metal function is missing");
            }

            bitmap_pipeline =
                device->newComputePipelineState(bitmap_function, &error);
            if (bitmap_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create Q6 bitmap pipeline: " +
                    error_description(error));
            }
            error = nullptr;
            aggregate_pipeline =
                device->newComputePipelineState(aggregate_function, &error);
            if (aggregate_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create Q6 bitmap aggregate pipeline: " +
                    error_description(error));
            }
            error = nullptr;
            reduce_pipeline =
                device->newComputePipelineState(reduce_function, &error);
            if (reduce_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create Q6 reduction pipeline: " +
                    error_description(error));
            }

            execution_width_value = static_cast<std::uint32_t>(
                aggregate_pipeline->threadExecutionWidth());
            max_threads_value = static_cast<std::uint32_t>(std::min({
                bitmap_pipeline->maxTotalThreadsPerThreadgroup(),
                aggregate_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup()}));
            if (threadgroup_width % execution_width_value != 0 ||
                threadgroup_width > max_threads_value) {
                throw std::invalid_argument(
                    "threadgroup width is incompatible with the unfused Q6 kernels");
            }

            row_count = static_cast<std::uint32_t>(rows);
            bitmap_word_count = (row_count + 31U) / 32U;
            quantity_buffer = wrap_shared(device, input.quantity);
            price_buffer = wrap_shared(device, input.extended_price);
            discount_buffer = wrap_shared(device, input.discount);
            date_buffer = wrap_shared(device, input.ship_date_yyyymmdd);
            if (quantity_buffer == nullptr || price_buffer == nullptr ||
                discount_buffer == nullptr || date_buffer == nullptr) {
                throw std::runtime_error(
                    "could not wrap Q6 colbin payloads as Metal buffers");
            }

            bitmap_buffer = device->newBuffer(
                static_cast<NS::UInteger>(bitmap_word_count) *
                    sizeof(std::uint32_t),
                MTL::ResourceStorageModePrivate);
            if (bitmap_buffer == nullptr) {
                throw std::runtime_error(
                    "could not allocate the private Q6 bitmap buffer");
            }

            auto count = group_count(
                bitmap_word_count,
                threadgroup_width,
                bitmap_words_per_thread);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) {
                    break;
                }
                count = group_count(
                    count, threadgroup_width, reduce_values_per_thread);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage_mode = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared
                    : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) *
                        sizeof(Q6Pair),
                    storage_mode);
                if (buffer == nullptr) {
                    throw std::runtime_error(
                        "could not allocate an unfused Q6 reduction buffer");
                }
                reduction_buffers.push_back(buffer);
            }

            reduce_function->release();
            reduce_function = nullptr;
            aggregate_function->release();
            aggregate_function = nullptr;
            bitmap_function->release();
            bitmap_function = nullptr;
            library->release();
            library = nullptr;
            pool->release();
        } catch (...) {
            if (reduce_function != nullptr) {
                reduce_function->release();
            }
            if (aggregate_function != nullptr) {
                aggregate_function->release();
            }
            if (bitmap_function != nullptr) {
                bitmap_function->release();
            }
            if (library != nullptr) {
                library->release();
            }
            pool->release();
            release_resources();
            throw;
        }
    }

    ~Impl() {
        release_resources();
    }

    void release_resources() noexcept {
        for (auto* buffer : reduction_buffers) {
            if (buffer != nullptr) {
                buffer->release();
            }
        }
        reduction_buffers.clear();
        for (auto** buffer : {
                 &bitmap_buffer,
                 &date_buffer,
                 &discount_buffer,
                 &price_buffer,
                 &quantity_buffer}) {
            if (*buffer != nullptr) {
                (*buffer)->release();
                *buffer = nullptr;
            }
        }
        for (auto** pipeline : {
                 &reduce_pipeline, &aggregate_pipeline, &bitmap_pipeline}) {
            if (*pipeline != nullptr) {
                (*pipeline)->release();
                *pipeline = nullptr;
            }
        }
        if (command_queue != nullptr) {
            command_queue->release();
            command_queue = nullptr;
        }
        if (device != nullptr) {
            device->release();
            device = nullptr;
        }
    }

    [[nodiscard]] TpchQ6Result execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command_buffer = command_queue->commandBuffer();
            auto* encoder = command_buffer != nullptr
                ? command_buffer->computeCommandEncoder()
                : nullptr;
            if (command_buffer == nullptr || encoder == nullptr) {
                throw std::runtime_error(
                    "could not create an unfused Q6 Metal command buffer");
            }

            const MTL::Size threads_per_group(threadgroup_width, 1, 1);

            // Stage A: fully materialize the predicate bitmap in private memory.
            encoder->setComputePipelineState(bitmap_pipeline);
            encoder->setBuffer(quantity_buffer, 0, 0);
            encoder->setBuffer(discount_buffer, 0, 1);
            encoder->setBuffer(date_buffer, 0, 2);
            encoder->setBuffer(bitmap_buffer, 0, 3);
            encoder->setBytes(&row_count, sizeof(row_count), 4);
            const auto bitmap_group_count =
                (bitmap_word_count + threadgroup_width - 1U) /
                threadgroup_width;
            encoder->dispatchThreadgroups(
                MTL::Size(bitmap_group_count, 1, 1), threads_per_group);
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);

            // Stage B: consume only the materialized bitmap and measure columns.
            encoder->setComputePipelineState(aggregate_pipeline);
            encoder->setBuffer(bitmap_buffer, 0, 0);
            encoder->setBuffer(price_buffer, 0, 1);
            encoder->setBuffer(discount_buffer, 0, 2);
            encoder->setBuffer(reduction_buffers.front(), 0, 3);
            encoder->setBytes(&bitmap_word_count, sizeof(bitmap_word_count), 4);
            encoder->dispatchThreadgroups(
                MTL::Size(level_counts.front(), 1, 1), threads_per_group);

            auto input_count = level_counts.front();
            for (std::size_t level = 1;
                 level < reduction_buffers.size();
                 ++level) {
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(reduction_buffers[level - 1], 0, 0);
                encoder->setBuffer(reduction_buffers[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->dispatchThreadgroups(
                    MTL::Size(level_counts[level], 1, 1),
                    threads_per_group);
                input_count = level_counts[level];
            }

            encoder->endEncoding();
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();
            if (command_buffer->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "unfused Q6 Metal command failed: " +
                    error_description(command_buffer->error()));
            }

            const auto* pair = static_cast<const Q6Pair*>(
                reduction_buffers.back()->contents());
            TpchQ6Result result;
            result.match_count = static_cast<std::uint64_t>(pair->count);
            result.revenue_1e4_usd = pair->revenue;
            result.host_time_ms =
                std::chrono::duration<double, std::milli>(
                    host_end - host_start)
                    .count();
            const auto gpu_start = command_buffer->GPUStartTime();
            const auto gpu_end = command_buffer->GPUEndTime();
            result.gpu_time_ms = gpu_end >= gpu_start
                ? (gpu_end - gpu_start) * 1'000.0
                : 0.0;
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

TpchQ6Unfused::TpchQ6Unfused(
    const std::filesystem::path& metal_library,
    tpch::LineitemView input,
    TpchQ6UnfusedConfig config)
    : impl_(std::make_unique<Impl>(metal_library, input, config)) {}

TpchQ6Unfused::~TpchQ6Unfused() = default;
TpchQ6Unfused::TpchQ6Unfused(TpchQ6Unfused&&) noexcept = default;
TpchQ6Unfused& TpchQ6Unfused::operator=(TpchQ6Unfused&&) noexcept = default;

TpchQ6Result TpchQ6Unfused::execute() {
    return impl_->execute();
}

const std::string& TpchQ6Unfused::device_name() const noexcept {
    return impl_->device_name_value;
}

std::uint32_t TpchQ6Unfused::execution_width() const noexcept {
    return impl_->execution_width_value;
}

std::uint32_t TpchQ6Unfused::max_threads_per_threadgroup() const noexcept {
    return impl_->max_threads_value;
}

}  // namespace joule::operators::gpu
