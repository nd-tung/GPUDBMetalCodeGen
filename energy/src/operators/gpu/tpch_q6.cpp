#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/tpch_q6.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace joule::operators::gpu {
namespace {

constexpr std::uint32_t scan_values_per_thread = 16;
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
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
}

}  // namespace

struct TpchQ6::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* command_queue{};
    MTL::ComputePipelineState* first_pipeline{};
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
    std::size_t reduction_element_size{};
    TpchQ6Config config;
    std::uint32_t execution_width_value{};
    std::uint32_t max_threads_value{};
    std::string device_name_value;

    Impl(
        const std::filesystem::path& metal_library,
        tpch::LineitemView input,
        TpchQ6Config requested_config)
        : config(requested_config) {
        if (input.row_count() == 0 || input.extended_price.size() != input.row_count() ||
            input.discount.size() != input.row_count() ||
            input.ship_date_yyyymmdd.size() != input.row_count()) {
            throw std::invalid_argument("TPC-H Q6 columns must be non-empty and equally sized");
        }
        if (input.row_count() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU TPC-H Q6 supports at most 2^32-1 rows");
        }
        if (config.threadgroup_width < 32 || config.threadgroup_width > 512 ||
            !std::has_single_bit(config.threadgroup_width)) {
            throw std::invalid_argument(
                "threadgroup width must be a power of two between 32 and 512");
        }

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        MTL::Function* first_function = nullptr;
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
            const auto library_path = metal_library.string();
            library = device->newLibrary(
                NS::String::string(library_path.c_str(), NS::UTF8StringEncoding), &error);
            if (library == nullptr) {
                throw std::runtime_error(
                    "could not load metallib '" + library_path + "': " +
                    error_description(error));
            }
            const char* first_name = "tpch_q6_revenue";
            if (config.mode == cpu::TpchQ6Mode::filter_count) {
                first_name = "tpch_q6_filter_count";
            } else if (config.mode == cpu::TpchQ6Mode::filter_bitmap) {
                first_name = "tpch_q6_filter_bitmap";
            }
            first_function = library->newFunction(
                NS::String::string(first_name, NS::UTF8StringEncoding));
            if (first_function == nullptr) {
                throw std::runtime_error("required TPC-H Metal function is missing");
            }
            error = nullptr;
            first_pipeline = device->newComputePipelineState(first_function, &error);
            if (first_pipeline == nullptr) {
                throw std::runtime_error(
                    "could not create TPC-H pipeline: " + error_description(error));
            }

            if (config.mode != cpu::TpchQ6Mode::filter_bitmap) {
                const char* reduce_name =
                    config.mode == cpu::TpchQ6Mode::filter_count
                    ? "tpch_q6_reduce_count"
                    : "tpch_q6_reduce_pair";
                reduce_function = library->newFunction(
                    NS::String::string(reduce_name, NS::UTF8StringEncoding));
                if (reduce_function == nullptr) {
                    throw std::runtime_error("TPC-H reduction function is missing");
                }
                error = nullptr;
                reduce_pipeline = device->newComputePipelineState(reduce_function, &error);
                if (reduce_pipeline == nullptr) {
                    throw std::runtime_error(
                        "could not create TPC-H reduction pipeline: " +
                        error_description(error));
                }
            }

            execution_width_value =
                static_cast<std::uint32_t>(first_pipeline->threadExecutionWidth());
            max_threads_value = static_cast<std::uint32_t>(
                first_pipeline->maxTotalThreadsPerThreadgroup());
            if (reduce_pipeline != nullptr) {
                max_threads_value = static_cast<std::uint32_t>(std::min(
                    static_cast<NS::UInteger>(max_threads_value),
                    reduce_pipeline->maxTotalThreadsPerThreadgroup()));
            }
            if (config.threadgroup_width % execution_width_value != 0 ||
                config.threadgroup_width > max_threads_value) {
                throw std::invalid_argument(
                    "threadgroup width is incompatible with the TPC-H pipeline");
            }

            row_count = static_cast<std::uint32_t>(input.row_count());
            quantity_buffer = wrap_shared(device, input.quantity);
            discount_buffer = wrap_shared(device, input.discount);
            date_buffer = wrap_shared(device, input.ship_date_yyyymmdd);
            if (config.mode == cpu::TpchQ6Mode::revenue) {
                price_buffer = wrap_shared(device, input.extended_price);
            }
            if (quantity_buffer == nullptr || discount_buffer == nullptr ||
                date_buffer == nullptr ||
                (config.mode == cpu::TpchQ6Mode::revenue && price_buffer == nullptr)) {
                throw std::runtime_error("could not wrap colbin payloads as Metal buffers");
            }

            if (config.mode == cpu::TpchQ6Mode::filter_bitmap) {
                bitmap_word_count = (row_count + 31U) / 32U;
                bitmap_buffer = device->newBuffer(
                    static_cast<NS::UInteger>(bitmap_word_count) * sizeof(std::uint32_t),
                    MTL::ResourceStorageModeShared);
                if (bitmap_buffer == nullptr) {
                    throw std::runtime_error("could not allocate the Metal bitmap buffer");
                }
            } else {
                reduction_element_size =
                    config.mode == cpu::TpchQ6Mode::filter_count
                    ? sizeof(std::int64_t)
                    : sizeof(Q6Pair);
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
                    const auto storage_mode = level + 1 == level_counts.size()
                        ? MTL::ResourceStorageModeShared
                        : MTL::ResourceStorageModePrivate;
                    auto* buffer = device->newBuffer(
                        static_cast<NS::UInteger>(level_counts[level]) *
                            reduction_element_size,
                        storage_mode);
                    if (buffer == nullptr) {
                        throw std::runtime_error(
                            "could not allocate a Metal TPC-H reduction buffer");
                    }
                    reduction_buffers.push_back(buffer);
                }
            }

            if (reduce_function != nullptr) {
                reduce_function->release();
                reduce_function = nullptr;
            }
            first_function->release();
            first_function = nullptr;
            library->release();
            library = nullptr;
            pool->release();
        } catch (...) {
            if (reduce_function != nullptr) {
                reduce_function->release();
            }
            if (first_function != nullptr) {
                first_function->release();
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
        for (auto** buffer : {&bitmap_buffer, &date_buffer, &discount_buffer,
                              &price_buffer, &quantity_buffer}) {
            if (*buffer != nullptr) {
                (*buffer)->release();
                *buffer = nullptr;
            }
        }
        if (reduce_pipeline != nullptr) {
            reduce_pipeline->release();
            reduce_pipeline = nullptr;
        }
        if (first_pipeline != nullptr) {
            first_pipeline->release();
            first_pipeline = nullptr;
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
                throw std::runtime_error("could not create a Metal command buffer");
            }
            const MTL::Size threads_per_group(config.threadgroup_width, 1, 1);
            encoder->setComputePipelineState(first_pipeline);
            if (config.mode == cpu::TpchQ6Mode::revenue) {
                encoder->setBuffer(quantity_buffer, 0, 0);
                encoder->setBuffer(price_buffer, 0, 1);
                encoder->setBuffer(discount_buffer, 0, 2);
                encoder->setBuffer(date_buffer, 0, 3);
                encoder->setBuffer(reduction_buffers.front(), 0, 4);
                encoder->setBytes(&row_count, sizeof(row_count), 5);
            } else if (config.mode == cpu::TpchQ6Mode::filter_count) {
                encoder->setBuffer(quantity_buffer, 0, 0);
                encoder->setBuffer(discount_buffer, 0, 1);
                encoder->setBuffer(date_buffer, 0, 2);
                encoder->setBuffer(reduction_buffers.front(), 0, 3);
                encoder->setBytes(&row_count, sizeof(row_count), 4);
            } else {
                encoder->setBuffer(quantity_buffer, 0, 0);
                encoder->setBuffer(discount_buffer, 0, 1);
                encoder->setBuffer(date_buffer, 0, 2);
                encoder->setBuffer(bitmap_buffer, 0, 3);
                encoder->setBytes(&row_count, sizeof(row_count), 4);
            }

            if (config.mode == cpu::TpchQ6Mode::filter_bitmap) {
                const auto groups = (bitmap_word_count + config.threadgroup_width - 1U) /
                                    config.threadgroup_width;
                encoder->dispatchThreadgroups(
                    MTL::Size(groups, 1, 1), threads_per_group);
            } else {
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

            TpchQ6Result result;
            if (config.mode == cpu::TpchQ6Mode::filter_count) {
                const auto* count = static_cast<const std::int64_t*>(
                    reduction_buffers.back()->contents());
                result.match_count = static_cast<std::uint64_t>(*count);
            } else if (config.mode == cpu::TpchQ6Mode::revenue) {
                const auto* pair = static_cast<const Q6Pair*>(
                    reduction_buffers.back()->contents());
                result.match_count = static_cast<std::uint64_t>(pair->count);
                result.revenue_1e4_usd = pair->revenue;
            }
            result.host_time_ms =
                std::chrono::duration<double, std::milli>(host_end - host_start).count();
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

TpchQ6::TpchQ6(
    const std::filesystem::path& metal_library,
    tpch::LineitemView input,
    TpchQ6Config config)
    : impl_(std::make_unique<Impl>(metal_library, input, config)) {}

TpchQ6::~TpchQ6() = default;
TpchQ6::TpchQ6(TpchQ6&&) noexcept = default;
TpchQ6& TpchQ6::operator=(TpchQ6&&) noexcept = default;

TpchQ6Result TpchQ6::execute() {
    return impl_->execute();
}

std::span<const std::uint32_t> TpchQ6::bitmap() const noexcept {
    if (impl_->bitmap_buffer == nullptr) {
        return {};
    }
    return {
        static_cast<const std::uint32_t*>(impl_->bitmap_buffer->contents()),
        impl_->bitmap_word_count};
}

const std::string& TpchQ6::device_name() const noexcept {
    return impl_->device_name_value;
}

std::uint32_t TpchQ6::execution_width() const noexcept {
    return impl_->execution_width_value;
}

std::uint32_t TpchQ6::max_threads_per_threadgroup() const noexcept {
    return impl_->max_threads_value;
}

}  // namespace joule::operators::gpu
