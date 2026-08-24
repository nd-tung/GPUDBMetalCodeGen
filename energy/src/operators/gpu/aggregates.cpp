#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/aggregates.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

namespace joule::operators::gpu {
namespace {

struct PriceSum {
    std::int64_t sum;
};
static_assert(sizeof(PriceSum) == 8);

struct PriceMinMax {
    std::int32_t minimum;
    std::int32_t maximum;
};
static_assert(sizeof(PriceMinMax) == 8);

struct alignas(8) PriceStats {
    std::int64_t sum;
    std::int32_t minimum;
    std::int32_t maximum;
};
static_assert(sizeof(PriceStats) == 16);

[[nodiscard]] std::string price_aggregate_kernel_name(
    cpu::PriceAggregateMode mode,
    PriceAggregateReduction reduction,
    bool first) {
    const char* operation = nullptr;
    switch (mode) {
        case cpu::PriceAggregateMode::sum:
            operation = "price_sum";
            break;
        case cpu::PriceAggregateMode::minmax:
            operation = "price_minmax";
            break;
        case cpu::PriceAggregateMode::stats:
            operation = "price_stats";
            break;
    }
    std::string name = operation;
    name += first ? "_first" : "_reduce";
    if (reduction == PriceAggregateReduction::threadgroup_tree) {
        name += "_threadgroup";
    }
    return name;
}

[[nodiscard]] std::size_t price_aggregate_element_size(
    cpu::PriceAggregateMode mode) noexcept {
    switch (mode) {
        case cpu::PriceAggregateMode::sum:
            return sizeof(PriceSum);
        case cpu::PriceAggregateMode::minmax:
            return sizeof(PriceMinMax);
        case cpu::PriceAggregateMode::stats:
            return sizeof(PriceStats);
    }
    return 0;
}

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "unknown Metal error";
    }
    const char* text = error->localizedDescription()->utf8String();
    return text != nullptr ? text : "unknown Metal error";
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

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(
    MTL::Device* device,
    std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
}

[[nodiscard]] std::uint32_t reduction_group_count(
    std::uint32_t values,
    std::uint32_t width,
    std::uint32_t values_per_thread) {
    const auto denominator =
        static_cast<std::uint64_t>(width) * values_per_thread;
    return static_cast<std::uint32_t>((values + denominator - 1) / denominator);
}

[[nodiscard]] std::string metal_device_name(MTL::Device* device) {
    return device != nullptr && device->name() != nullptr &&
            device->name()->utf8String() != nullptr
        ? device->name()->utf8String()
        : "";
}

[[nodiscard]] double gpu_time_ms(MTL::CommandBuffer* command) {
    const auto begin = command->GPUStartTime();
    const auto end = command->GPUEndTime();
    return end >= begin ? (end - begin) * 1'000.0 : 0.0;
}

}  // namespace

struct PriceAggregate::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* first_pipeline{};
    MTL::ComputePipelineState* reduce_pipeline{};
    MTL::Buffer* input_buffer{};
    std::vector<MTL::Buffer*> levels;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t rows{};
    cpu::PriceAggregateMode mode{};
    std::uint32_t width{};
    PriceAggregateReduction reduction{};
    std::uint32_t execution_width_value{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        std::span<const float> input,
        cpu::PriceAggregateMode requested_mode,
        std::uint32_t requested_width,
        PriceAggregateReduction requested_reduction)
        : mode(requested_mode),
          width(requested_width),
          reduction(requested_reduction) {
        if (input.empty() || input.size() > UINT32_MAX) {
            throw std::invalid_argument("GPU price aggregate input size is invalid");
        }
        if (width < 32 || width > 512 || !std::has_single_bit(width)) {
            throw std::invalid_argument(
                "threadgroup width must be a power of two from 32 to 512");
        }
        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            NS::Error* error = nullptr;
            const auto path = library_path.string();
            library = device->newLibrary(
                NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
            if (library == nullptr) {
                throw std::runtime_error(
                    "could not load metallib: " + error_description(error));
            }
            const auto first_name =
                price_aggregate_kernel_name(mode, reduction, true);
            const auto reduce_name =
                price_aggregate_kernel_name(mode, reduction, false);
            first_pipeline = make_pipeline(
                device, library, first_name.c_str());
            reduce_pipeline = make_pipeline(
                device, library, reduce_name.c_str());
            execution_width_value =
                static_cast<std::uint32_t>(first_pipeline->threadExecutionWidth());
            const auto maximum = std::min(
                first_pipeline->maxTotalThreadsPerThreadgroup(),
                reduce_pipeline->maxTotalThreadsPerThreadgroup());
            if (width % execution_width_value != 0 || width > maximum) {
                throw std::invalid_argument(
                    "threadgroup width is incompatible with price aggregate kernels");
            }

            rows = static_cast<std::uint32_t>(input.size());
            input_buffer = wrap_shared(device, input);
            if (queue == nullptr || input_buffer == nullptr) {
                throw std::runtime_error("could not allocate price aggregate inputs");
            }
            auto count = reduction_group_count(rows, width, 16);
            while (true) {
                level_counts.push_back(count);
                if (count == 1) break;
                count = reduction_group_count(count, width, 4);
            }
            for (std::size_t level = 0; level < level_counts.size(); ++level) {
                const auto storage = level + 1 == level_counts.size()
                    ? MTL::ResourceStorageModeShared
                    : MTL::ResourceStorageModePrivate;
                auto* buffer = device->newBuffer(
                    static_cast<NS::UInteger>(level_counts[level]) *
                        price_aggregate_element_size(mode),
                    storage);
                if (buffer == nullptr) {
                    throw std::runtime_error(
                        "could not allocate price aggregate reduction buffer");
                }
                levels.push_back(buffer);
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
        for (auto* buffer : levels) {
            if (buffer != nullptr) buffer->release();
        }
        levels.clear();
        if (input_buffer != nullptr) input_buffer->release();
        if (reduce_pipeline != nullptr) reduce_pipeline->release();
        if (first_pipeline != nullptr) first_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        input_buffer = nullptr;
        reduce_pipeline = nullptr;
        first_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    [[nodiscard]] PriceAggregateResult execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder =
                command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error(
                    "could not create price aggregate command buffer");
            }
            const MTL::Size threads(width, 1, 1);
            encoder->setComputePipelineState(first_pipeline);
            encoder->setBuffer(input_buffer, 0, 0);
            encoder->setBuffer(levels.front(), 0, 1);
            encoder->setBytes(&rows, sizeof(rows), 2);
            encoder->dispatchThreadgroups(
                MTL::Size(level_counts.front(), 1, 1), threads);
            auto input_count = level_counts.front();
            for (std::size_t level = 1; level < levels.size(); ++level) {
                encoder->setComputePipelineState(reduce_pipeline);
                encoder->setBuffer(levels[level - 1], 0, 0);
                encoder->setBuffer(levels[level], 0, 1);
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
                    "price aggregate Metal command failed: " +
                    error_description(command->error()));
            }
            PriceAggregateResult result;
            result.count = rows;
            switch (mode) {
                case cpu::PriceAggregateMode::sum: {
                    const auto* sum = static_cast<const PriceSum*>(
                        levels.back()->contents());
                    result.sum_price_cents = sum->sum;
                    break;
                }
                case cpu::PriceAggregateMode::minmax: {
                    const auto* minmax = static_cast<const PriceMinMax*>(
                        levels.back()->contents());
                    result.min_price_cents = minmax->minimum;
                    result.max_price_cents = minmax->maximum;
                    break;
                }
                case cpu::PriceAggregateMode::stats: {
                    const auto* stats = static_cast<const PriceStats*>(
                        levels.back()->contents());
                    result.sum_price_cents = stats->sum;
                    result.min_price_cents = stats->minimum;
                    result.max_price_cents = stats->maximum;
                    break;
                }
            }
            result.host_time_ms =
                std::chrono::duration<double, std::milli>(
                    host_end - host_start).count();
            result.gpu_time_ms = gpu_time_ms(command);
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }
};

PriceAggregate::PriceAggregate(
    const std::filesystem::path& metal_library,
    std::span<const float> input,
    cpu::PriceAggregateMode mode,
    std::uint32_t threadgroup_width,
    PriceAggregateReduction reduction)
    : impl_(std::make_unique<Impl>(
          metal_library, input, mode, threadgroup_width, reduction)) {}
PriceAggregate::~PriceAggregate() = default;
PriceAggregate::PriceAggregate(PriceAggregate&&) noexcept = default;
PriceAggregate& PriceAggregate::operator=(PriceAggregate&&) noexcept = default;
PriceAggregateResult PriceAggregate::execute() { return impl_->execute(); }
const std::string& PriceAggregate::device_name() const noexcept {
    return impl_->name;
}
std::uint32_t PriceAggregate::execution_width() const noexcept {
    return impl_->execution_width_value;
}

struct PartKeyGroupCount::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* clear_pipeline{};
    MTL::ComputePipelineState* count_pipeline{};
    MTL::Buffer* key_buffer{};
    MTL::Buffer* output_buffer{};
    std::uint32_t rows{};
    std::uint32_t groups{};
    std::uint32_t width{};
    GroupByCountStrategy strategy{};
    std::uint32_t execution_width_value{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        std::span<const std::int32_t> keys,
        std::uint32_t group_count,
        std::uint32_t requested_width,
        GroupByCountStrategy requested_strategy)
        : groups(group_count),
          width(requested_width),
          strategy(requested_strategy) {
        if (keys.empty() || keys.size() > UINT32_MAX ||
            !std::has_single_bit(groups)) {
            throw std::invalid_argument("GPU group-by input or group count is invalid");
        }
        for (const auto key : keys) {
            if (key <= 0) throw std::invalid_argument("part keys must be positive");
        }
        if (width < 32 || width > 512 || !std::has_single_bit(width)) {
            throw std::invalid_argument(
                "threadgroup width must be a power of two from 32 to 512");
        }
        if (strategy == GroupByCountStrategy::bounded_threadgroup &&
            groups > 4096) {
            throw std::invalid_argument(
                "bounded threadgroup aggregation supports at most 4096 groups");
        }
        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) throw std::runtime_error("no Metal device is available");
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            NS::Error* error = nullptr;
            const auto path = library_path.string();
            library = device->newLibrary(
                NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
            if (library == nullptr) {
                throw std::runtime_error(
                    "could not load metallib: " + error_description(error));
            }
            clear_pipeline =
                make_pipeline(device, library, "groupby_count_clear");
            count_pipeline = make_pipeline(
                device, library,
                strategy == GroupByCountStrategy::global_atomic
                    ? "groupby_count_i32"
                    : "groupby_count_i32_threadgroup");
            execution_width_value =
                static_cast<std::uint32_t>(count_pipeline->threadExecutionWidth());
            const auto maximum = std::min(
                clear_pipeline->maxTotalThreadsPerThreadgroup(),
                count_pipeline->maxTotalThreadsPerThreadgroup());
            if (width % execution_width_value != 0 || width > maximum) {
                throw std::invalid_argument(
                    "threadgroup width is incompatible with group-by kernels");
            }
            rows = static_cast<std::uint32_t>(keys.size());
            key_buffer = wrap_shared(device, keys);
            output_buffer = device->newBuffer(
                static_cast<NS::UInteger>(groups) * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            if (queue == nullptr || key_buffer == nullptr ||
                output_buffer == nullptr) {
                throw std::runtime_error("could not allocate group-by buffers");
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
        if (output_buffer != nullptr) output_buffer->release();
        if (key_buffer != nullptr) key_buffer->release();
        if (count_pipeline != nullptr) count_pipeline->release();
        if (clear_pipeline != nullptr) clear_pipeline->release();
        if (queue != nullptr) queue->release();
        if (device != nullptr) device->release();
        output_buffer = nullptr;
        key_buffer = nullptr;
        count_pipeline = nullptr;
        clear_pipeline = nullptr;
        queue = nullptr;
        device = nullptr;
    }

    void dispatch(
        MTL::ComputeCommandEncoder* encoder,
        std::uint32_t item_count) const {
        const auto threadgroups = (item_count + width - 1U) / width;
        encoder->dispatchThreadgroups(
            MTL::Size(threadgroups, 1, 1), MTL::Size(width, 1, 1));
    }

    [[nodiscard]] GroupByCountRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        try {
            const auto host_start = std::chrono::steady_clock::now();
            auto* command = queue->commandBuffer();
            auto* encoder =
                command != nullptr ? command->computeCommandEncoder() : nullptr;
            if (command == nullptr || encoder == nullptr) {
                throw std::runtime_error("could not create group-by command buffer");
            }
            encoder->setComputePipelineState(clear_pipeline);
            encoder->setBuffer(output_buffer, 0, 0);
            encoder->setBytes(&groups, sizeof(groups), 1);
            dispatch(encoder, groups);
            encoder->setComputePipelineState(count_pipeline);
            encoder->setBuffer(key_buffer, 0, 0);
            encoder->setBuffer(output_buffer, 0, 1);
            encoder->setBytes(&rows, sizeof(rows), 2);
            encoder->setBytes(&groups, sizeof(groups), 3);
            if (strategy == GroupByCountStrategy::bounded_threadgroup) {
                encoder->setThreadgroupMemoryLength(
                    static_cast<NS::UInteger>(groups) *
                        sizeof(std::uint32_t),
                    0);
            }
            const auto work_items = (rows + 15U) / 16U;
            dispatch(encoder, work_items);
            encoder->endEncoding();
            command->commit();
            command->waitUntilCompleted();
            const auto host_end = std::chrono::steady_clock::now();
            if (command->status() == MTL::CommandBufferStatusError) {
                throw std::runtime_error(
                    "group-by Metal command failed: " +
                    error_description(command->error()));
            }
            const auto result = GroupByCountRun{
                std::chrono::duration<double, std::milli>(
                    host_end - host_start).count(),
                gpu_time_ms(command)};
            pool->release();
            return result;
        } catch (...) {
            pool->release();
            throw;
        }
    }

    [[nodiscard]] std::span<const std::uint32_t> output() const noexcept {
        return {
            static_cast<const std::uint32_t*>(output_buffer->contents()),
            groups};
    }
};

PartKeyGroupCount::PartKeyGroupCount(
    const std::filesystem::path& metal_library,
    std::span<const std::int32_t> keys,
    std::uint32_t group_count,
    std::uint32_t threadgroup_width,
    GroupByCountStrategy strategy)
    : impl_(std::make_unique<Impl>(
          metal_library, keys, group_count, threadgroup_width, strategy)) {}
PartKeyGroupCount::~PartKeyGroupCount() = default;
PartKeyGroupCount::PartKeyGroupCount(PartKeyGroupCount&&) noexcept = default;
PartKeyGroupCount& PartKeyGroupCount::operator=(PartKeyGroupCount&&) noexcept = default;
GroupByCountRun PartKeyGroupCount::execute() { return impl_->execute(); }
std::span<const std::uint32_t> PartKeyGroupCount::output() const noexcept {
    return impl_->output();
}
const std::string& PartKeyGroupCount::device_name() const noexcept {
    return impl_->name;
}
std::uint32_t PartKeyGroupCount::execution_width() const noexcept {
    return impl_->execution_width_value;
}
std::uint32_t PartKeyGroupCount::group_count() const noexcept {
    return impl_->groups;
}

}  // namespace joule::operators::gpu
