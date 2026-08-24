#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/topk.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace joule::operators::gpu {
namespace {

constexpr std::uint32_t first_values_per_thread = 64;
constexpr std::uint32_t merge_lists_per_thread = 4;
static_assert(sizeof(cpu::TopKEntry) == 16);

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) return "unknown Metal error";
    const char* text = error->localizedDescription()->utf8String();
    return text != nullptr ? text : "unknown Metal error";
}

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(MTL::Device* device, std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
}

}  // namespace

struct OrdersTopK::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* first_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* key_buffer{};
    MTL::Buffer* price_buffer{};
    std::vector<MTL::Buffer*> levels;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t first_group_count{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metallib,
        tpch::OrdersView input,
        std::uint32_t requested_width)
        : width(requested_width) {
        if (input.row_count() == 0 || input.total_price.size() != input.row_count()) {
            throw std::invalid_argument("TPC-H orders columns must be non-empty and equally sized");
        }
        if (input.row_count() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU orders top-k supports at most 2^32-1 rows");
        }
        if (width < 32 || width > 512 || !std::has_single_bit(width)) {
            throw std::invalid_argument("threadgroup width must be a power of two from 32 to 512");
        }

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
                NS::String::string("orders_topk_first", NS::UTF8StringEncoding));
            reduce_function = library->newFunction(
                NS::String::string("orders_topk_reduce", NS::UTF8StringEncoding));
            if (first_function == nullptr || reduce_function == nullptr) {
                throw std::runtime_error("orders top-k Metal functions are missing");
            }
            first_pipeline = device->newComputePipelineState(first_function, &error);
            if (first_pipeline == nullptr) {
                throw std::runtime_error("could not create top-k pipeline: " + error_description(error));
            }
            error = nullptr;
            reduce_pipeline = device->newComputePipelineState(reduce_function, &error);
            if (reduce_pipeline == nullptr) {
                throw std::runtime_error("could not create top-k reduction: " + error_description(error));
            }
            execution_width_value = static_cast<std::uint32_t>(first_pipeline->threadExecutionWidth());
            const auto maximum = std::min(
                first_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup());
            if (width % execution_width_value != 0 || width > maximum) {
                throw std::invalid_argument("threadgroup width is incompatible with top-k kernels");
            }

            row_count = static_cast<std::uint32_t>(input.row_count());
            key_buffer = wrap_shared(device, input.order_key);
            price_buffer = wrap_shared(device, input.total_price);
            if (key_buffer == nullptr || price_buffer == nullptr) {
                throw std::runtime_error("could not wrap orders columns as Metal buffers");
            }
            const auto rows_per_group = static_cast<std::uint64_t>(width) * first_values_per_thread;
            first_group_count = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(row_count) + rows_per_group - 1) / rows_per_group);
            auto count = first_group_count * width;
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = (count + merge_lists_per_thread - 1) / merge_lists_per_thread;
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate;
                const auto bytes = static_cast<std::uint64_t>(level_counts[level]) * 10 *
                                   sizeof(cpu::TopKEntry);
                auto* buffer = device->newBuffer(static_cast<NS::UInteger>(bytes), storage);
                if (buffer == nullptr) throw std::runtime_error("could not allocate top-k buffer");
                levels.push_back(buffer);
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
        for (auto* buffer : levels) if (buffer != nullptr) buffer->release();
        levels.clear();
        if (price_buffer != nullptr) price_buffer->release();
        if (key_buffer != nullptr) key_buffer->release();
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (first_pipeline != nullptr) first_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        price_buffer = nullptr;
        key_buffer = nullptr;
        reduce_pipeline = nullptr;
        first_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    [[nodiscard]] TopKRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder = command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error("could not create top-k Metal command buffer");
            }
            const MTL::Size threads(width, 1, 1);
            encoder->setComputePipelineState(first_pipeline);
            encoder->setBuffer(key_buffer, 0, 0);
            encoder->setBuffer(price_buffer, 0, 1);
            encoder->setBuffer(levels.front(), 0, 2);
            encoder->setBytes(&row_count, sizeof(row_count), 3);
            encoder->dispatchThreadgroups(MTL::Size(first_group_count, 1, 1), threads);
            auto input_count = level_counts.front();
            for (std::size_t level = 1; level < levels.size(); ++level) {
                const auto output_count = level_counts[level];
                const auto groups = (output_count + width - 1) / width;
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(levels[level - 1], 0, 0);
                encoder->setBuffer(levels[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->setBytes(&output_count, sizeof(output_count), 3);
                encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), threads);
                input_count = output_count;
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error("top-k Metal command failed: " + error_description(command->error()));
            }
            TopKRun run;
            std::memcpy(run.rows.data(), levels.back()->contents(), sizeof(run.rows));
            run.host_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            const auto gpu_start = command->GPUStartTime();
            const auto gpu_end = command->GPUEndTime();
            run.gpu_time_ms = gpu_end >= gpu_start ? (gpu_end - gpu_start) * 1'000.0 : 0.0;
            pool->release();
            return run;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

OrdersTopK::OrdersTopK(
    const std::filesystem::path& metallib,
    tpch::OrdersView input,
    std::uint32_t width)
    : impl_(std::make_unique<Impl>(metallib, input, width)) {}
OrdersTopK::~OrdersTopK() = default;
OrdersTopK::OrdersTopK(OrdersTopK&&) noexcept = default;
OrdersTopK& OrdersTopK::operator=(OrdersTopK&&) noexcept = default;
TopKRun OrdersTopK::execute() { return impl_->execute(); }
const std::string& OrdersTopK::device_name() const noexcept { return impl_->device_name_value; }
std::uint32_t OrdersTopK::execution_width() const noexcept { return impl_->execution_width_value; }

}  // namespace joule::operators::gpu
