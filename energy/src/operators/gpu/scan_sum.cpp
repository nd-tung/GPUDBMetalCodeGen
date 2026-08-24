#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/scan_sum.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace joule::operators::gpu {
namespace {

constexpr std::uint32_t optimized_scan_values_per_thread = 16;
constexpr std::uint32_t optimized_reduce_values_per_thread = 4;

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

}  // namespace

struct ScanSum::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* command_queue{};
    MTL::ComputePipelineState* scan_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* input_buffer{};
    std::vector<MTL::Buffer*> reduction_buffers;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t row_count{};
    ScanSumConfig config;
    std::uint32_t execution_width_value{};
    std::uint32_t max_threads_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metal_library,
        std::span<const std::int32_t> input,
        ScanSumConfig requested_config)
        : config(requested_config) {
        if (input.empty()) {
            throw std::invalid_argument("GPU scan input must not be empty");
        }
        if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU scan currently supports at most 2^32-1 rows");
        }
        if (config.threadgroup_width < 32 || config.threadgroup_width > 512 ||
            !std::has_single_bit(config.threadgroup_width)) {
            throw std::invalid_argument(
                "threadgroup width must be a power of two between 32 and 512");
        }

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        MTL::Function* scan_function = nullptr;
        MTL::Function* reduce_function = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) {
                throw std::runtime_error("no Metal device is available");
            }
            if (device->name() != nullptr && device->name()->utf8String() != nullptr) {
                device_name_value = device->name()->utf8String();
            }

            command_queue = device->newCommandQueue();
            if (command_queue == nullptr) {
                throw std::runtime_error("could not create a Metal command queue");
            }

            NS::Error* error = nullptr;
            const auto path_string = metal_library.string();
            library = device->newLibrary(
                NS::String::string(path_string.c_str(), NS::UTF8StringEncoding), &error);
            if (library == nullptr) {
                throw std::runtime_error(
                    "could not load metallib '" + path_string + "': " +
                    error_description(error));
            }

            const char* scan_name = "scan_sum_i32_baseline";
            const char* reduce_name = "reduce_sum_i64_baseline";
            if (config.kernel == ScanSumKernel::multi_item) {
                scan_name = "scan_sum_i32_multi_item";
                reduce_name = "reduce_sum_i64_multi_item";
            } else if (config.kernel == ScanSumKernel::simdgroup) {
                scan_name = "scan_sum_i32_simdgroup";
                reduce_name = "reduce_sum_i64_simdgroup";
            }
            scan_function = library->newFunction(
                NS::String::string(scan_name, NS::UTF8StringEncoding));
            reduce_function = library->newFunction(
                NS::String::string(reduce_name, NS::UTF8StringEncoding));
            if (scan_function == nullptr || reduce_function == nullptr) {
                throw std::runtime_error("required scan-sum functions are missing from the metallib");
            }

            error = nullptr;
            scan_pipeline = device->newComputePipelineState(scan_function, &error);
            if (scan_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create scan pipeline: " + error_description(error));
            }
            error = nullptr;
            reduce_pipeline = device->newComputePipelineState(reduce_function, &error);
            if (reduce_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create reduction pipeline: " + error_description(error));
            }

            execution_width_value = static_cast<std::uint32_t>(
                scan_pipeline->threadExecutionWidth());
            max_threads_value = static_cast<std::uint32_t>(std::min(
                scan_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup()));
            if (config.threadgroup_width % execution_width_value != 0) {
                throw std::invalid_argument(
                    "threadgroup width must be a multiple of the pipeline execution width");
            }
            if (config.threadgroup_width > max_threads_value) {
                throw std::invalid_argument(
                    "threadgroup width exceeds the compute pipeline limit");
            }

            row_count = static_cast<std::uint32_t>(input.size());
            const auto input_bytes = input.size_bytes();
            input_buffer = device->newBuffer(input_bytes, MTL::ResourceStorageModeShared);
            if (input_buffer == nullptr) {
                throw std::runtime_error("could not allocate the shared Metal input buffer");
            }
            std::memcpy(input_buffer->contents(), input.data(), input_bytes);

            const auto scan_values_per_thread = config.kernel != ScanSumKernel::baseline
                ? optimized_scan_values_per_thread
                : 1U;
            const auto reduce_values_per_thread = config.kernel != ScanSumKernel::baseline
                ? optimized_reduce_values_per_thread
                : 1U;
            auto count = group_count(
                row_count, config.threadgroup_width, scan_values_per_thread);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) {
                    break;
                }
                count = group_count(
                    count, config.threadgroup_width, reduce_values_per_thread);
            }

            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const bool final_level = level + 1 == level_counts.size();
                const auto storage_mode =
                    config.kernel != ScanSumKernel::baseline && !final_level
                    ? MTL::ResourceStorageModePrivate
                    : MTL::ResourceStorageModeShared;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) * sizeof(std::int64_t),
                    storage_mode);
                if (buffer == nullptr) {
                    throw std::runtime_error("could not allocate a Metal reduction buffer");
                }
                reduction_buffers.push_back(buffer);
            }

            reduce_function->release();
            reduce_function = nullptr;
            scan_function->release();
            scan_function = nullptr;
            library->release();
            library = nullptr;
            pool->release();
        } catch (...) {
            if (reduce_function != nullptr) {
                reduce_function->release();
            }
            if (scan_function != nullptr) {
                scan_function->release();
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
        if (input_buffer != nullptr) {
            input_buffer->release();
            input_buffer = nullptr;
        }
        if (reduce_pipeline != nullptr) {
            reduce_pipeline->release();
            reduce_pipeline = nullptr;
        }
        if (scan_pipeline != nullptr) {
            scan_pipeline->release();
            scan_pipeline = nullptr;
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

    [[nodiscard]] ScanSumRun execute_batch(std::uint32_t repetitions) {
        if (repetitions == 0) {
            throw std::invalid_argument("GPU batch must contain at least one scan");
        }
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command_buffer = command_queue->commandBuffer();
            auto* encoder = command_buffer->computeCommandEncoder();
            if (command_buffer == nullptr || encoder == nullptr) {
                throw std::runtime_error("could not create a Metal command buffer");
            }

            const MTL::Size threads_per_group(config.threadgroup_width, 1, 1);
            for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition) {
                encoder->setComputePipelineState(scan_pipeline);
                encoder->setBuffer(input_buffer, 0, 0);
                encoder->setBuffer(reduction_buffers.front(), 0, 1);
                encoder->setBytes(&row_count, sizeof(row_count), 2);
                encoder->dispatchThreadgroups(
                    MTL::Size(level_counts.front(), 1, 1), threads_per_group);

                auto input_count = level_counts.front();
                for (std::size_t level = 1; level < reduction_buffers.size(); ++level) {
                    encoder->setComputePipelineState(reduce_pipeline);
                    encoder->setBuffer(reduction_buffers[level - 1], 0, 0);
                    encoder->setBuffer(reduction_buffers[level], 0, 1);
                    encoder->setBytes(&input_count, sizeof(input_count), 2);
                    encoder->dispatchThreadgroups(
                        MTL::Size(level_counts[level], 1, 1), threads_per_group);
                    input_count = level_counts[level];
                }
            }

            encoder->endEncoding();
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();

            if (command_buffer->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "Metal command failed: " + error_description(command_buffer->error()));
            }

            ScanSumRun result;
            result.sum = *static_cast<const std::int64_t*>(
                reduction_buffers.back()->contents());
            result.host_time_ms =
                std::chrono::duration<double, std::milli>(host_end - host_start).count();
            const auto gpu_start = command_buffer->GPUStartTime();
            const auto gpu_end = command_buffer->GPUEndTime();
            result.gpu_time_ms = gpu_end >= gpu_start ? (gpu_end - gpu_start) * 1'000.0 : 0.0;
            result.repetitions = repetitions;
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

ScanSum::ScanSum(
    const std::filesystem::path& metal_library,
    std::span<const std::int32_t> input,
    ScanSumConfig config)
    : impl_(std::make_unique<Impl>(metal_library, input, config)) {}

ScanSum::~ScanSum() = default;
ScanSum::ScanSum(ScanSum&&) noexcept = default;
ScanSum& ScanSum::operator=(ScanSum&&) noexcept = default;

ScanSumRun ScanSum::execute() {
    return impl_->execute_batch(1);
}

ScanSumRun ScanSum::execute_batch(std::uint32_t repetitions) {
    return impl_->execute_batch(repetitions);
}

const std::string& ScanSum::device_name() const noexcept {
    return impl_->device_name_value;
}

std::uint32_t ScanSum::execution_width() const noexcept {
    return impl_->execution_width_value;
}

std::uint32_t ScanSum::max_threads_per_threadgroup() const noexcept {
    return impl_->max_threads_value;
}

}  // namespace joule::operators::gpu
