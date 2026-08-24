#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/tpch_q1.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace joule::operators::gpu {
namespace {

constexpr std::uint32_t first_values_per_thread = 16;
constexpr std::uint32_t reduce_values_per_thread = 4;
constexpr std::uint32_t group_total = 6;
static_assert(sizeof(cpu::TpchQ1Group) == 48);

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

}  // namespace

struct TpchQ1GroupBy::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* first_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    std::vector<MTL::Buffer*> inputs;
    std::vector<MTL::Buffer*> reductions;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metallib,
        tpch::LineitemView input,
        std::uint32_t requested_width)
        : width(requested_width) {
        const auto rows = input.row_count();
        if (rows == 0 || input.extended_price.size() != rows ||
            input.discount.size() != rows || input.tax.size() != rows ||
            input.return_flag.size() != rows || input.line_status.size() != rows ||
            input.ship_date_yyyymmdd.size() != rows) {
            throw std::invalid_argument("TPC-H Q1 columns must be non-empty and equally sized");
        }
        if (rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU TPC-H Q1 supports at most 2^32-1 rows");
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
                NS::String::string("tpch_q1_first", NS::UTF8StringEncoding));
            reduce_function = library->newFunction(
                NS::String::string("tpch_q1_reduce", NS::UTF8StringEncoding));
            if (first_function == nullptr || reduce_function == nullptr) {
                throw std::runtime_error("TPC-H Q1 Metal functions are missing");
            }
            first_pipeline = device->newComputePipelineState(first_function, &error);
            if (first_pipeline == nullptr) {
                throw std::runtime_error("could not create Q1 pipeline: " + error_description(error));
            }
            error = nullptr;
            reduce_pipeline = device->newComputePipelineState(reduce_function, &error);
            if (reduce_pipeline == nullptr) {
                throw std::runtime_error("could not create Q1 reduction: " + error_description(error));
            }
            execution_width_value = static_cast<std::uint32_t>(first_pipeline->threadExecutionWidth());
            const auto maximum = std::min(
                first_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup());
            if (width % execution_width_value != 0 || width > maximum) {
                throw std::invalid_argument("threadgroup width is incompatible with Q1 kernels");
            }

            row_count = static_cast<std::uint32_t>(rows);
            inputs = {
                wrap_shared(device, input.quantity),
                wrap_shared(device, input.extended_price),
                wrap_shared(device, input.discount),
                wrap_shared(device, input.tax),
                wrap_shared(device, input.return_flag),
                wrap_shared(device, input.line_status),
                wrap_shared(device, input.ship_date_yyyymmdd)};
            if (std::any_of(inputs.begin(), inputs.end(), [](auto* buffer) { return buffer == nullptr; })) {
                throw std::runtime_error("could not wrap Q1 input columns as Metal buffers");
            }

            auto count = group_count(row_count, width, first_values_per_thread);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = group_count(count, width, reduce_values_per_thread);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) * group_total *
                        sizeof(cpu::TpchQ1Group),
                    storage);
                if (buffer == nullptr) throw std::runtime_error("could not allocate Q1 reduction buffer");
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
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (first_pipeline != nullptr) first_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        reduce_pipeline = nullptr;
        first_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    [[nodiscard]] TpchQ1Run execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder = command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error("could not create Q1 Metal command buffer");
            }
            encoder->setComputePipelineState(first_pipeline);
            for (std::uint32_t index = 0; index < inputs.size(); ++index) {
                encoder->setBuffer(inputs[index], 0, index);
            }
            auto output_count = level_counts.front();
            encoder->setBuffer(reductions.front(), 0, 7);
            encoder->setBytes(&row_count, sizeof(row_count), 8);
            encoder->setBytes(&output_count, sizeof(output_count), 9);
            const MTL::Size threads(width, 1, 1);
            encoder->dispatchThreadgroups(MTL::Size(output_count, 1, 1), threads);
            auto input_count = output_count;
            for (std::size_t level = 1; level < reductions.size(); ++level) {
                output_count = level_counts[level];
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(reductions[level - 1], 0, 0);
                encoder->setBuffer(reductions[level], 0, 1);
                encoder->setBytes(&input_count, sizeof(input_count), 2);
                encoder->setBytes(&output_count, sizeof(output_count), 3);
                encoder->dispatchThreadgroups(MTL::Size(output_count, group_total, 1), threads);
                input_count = output_count;
            }
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error("Q1 Metal command failed: " + error_description(command->error()));
            }
            TpchQ1Run run;
            std::memcpy(
                run.groups.data(), reductions.back()->contents(),
                run.groups.size() * sizeof(cpu::TpchQ1Group));
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

TpchQ1GroupBy::TpchQ1GroupBy(
    const std::filesystem::path& metallib,
    tpch::LineitemView input,
    std::uint32_t width)
    : impl_(std::make_unique<Impl>(metallib, input, width)) {}
TpchQ1GroupBy::~TpchQ1GroupBy() = default;
TpchQ1GroupBy::TpchQ1GroupBy(TpchQ1GroupBy&&) noexcept = default;
TpchQ1GroupBy& TpchQ1GroupBy::operator=(TpchQ1GroupBy&&) noexcept = default;
TpchQ1Run TpchQ1GroupBy::execute() { return impl_->execute(); }
const std::string& TpchQ1GroupBy::device_name() const noexcept { return impl_->device_name_value; }
std::uint32_t TpchQ1GroupBy::execution_width() const noexcept {
    return impl_->execution_width_value;
}

}  // namespace joule::operators::gpu
