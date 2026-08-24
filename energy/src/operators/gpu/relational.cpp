#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "joule/operators/gpu/relational.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

namespace joule::operators::gpu {
namespace {

[[nodiscard]] std::string error_description(NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "unknown Metal error";
    }
    const auto* text = error->localizedDescription()->utf8String();
    return text != nullptr ? text : "unknown Metal error";
}

template <typename T>
[[nodiscard]] MTL::Buffer* wrap_shared(MTL::Device* device, std::span<const T> values) {
    return device->newBuffer(
        const_cast<T*>(values.data()), values.size_bytes(),
        MTL::ResourceStorageModeShared, nullptr);
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

}  // namespace

struct ScanCopyF32::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::Buffer* input{};
    MTL::Buffer* output{};
    std::size_t count{};
    std::string name;

    explicit Impl(std::span<const float> values) : count(values.size()) {
        if (values.empty()) throw std::invalid_argument("scan-copy input must not be empty");
        device = MTL::CreateSystemDefaultDevice();
        if (device == nullptr) throw std::runtime_error("no Metal device is available");
        name = metal_device_name(device);
        queue = device->newCommandQueue();
        input = wrap_shared(device, values);
        output = device->newBuffer(values.size_bytes(), MTL::ResourceStorageModeShared);
        if (queue == nullptr || input == nullptr || output == nullptr) {
            release();
            throw std::runtime_error("could not allocate scan-copy Metal resources");
        }
    }
    ~Impl() { release(); }
    void release() noexcept {
        if (output) output->release();
        if (input) input->release();
        if (queue) queue->release();
        if (device) device->release();
        output = nullptr; input = nullptr; queue = nullptr; device = nullptr;
    }
    [[nodiscard]] RelationalRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        const auto start = std::chrono::steady_clock::now();
        auto* command = queue->commandBuffer();
        auto* encoder = command != nullptr ? command->blitCommandEncoder() : nullptr;
        if (command == nullptr || encoder == nullptr) {
            pool->release();
            throw std::runtime_error("could not create scan-copy command buffer");
        }
        encoder->copyFromBuffer(input, 0, output, 0, count * sizeof(float));
        encoder->endEncoding();
        command->commit();
        command->waitUntilCompleted();
        const auto end = std::chrono::steady_clock::now();
        if (command->status() == MTL::CommandBufferStatusError) {
            const auto message = error_description(command->error());
            pool->release();
            throw std::runtime_error("Metal scan-copy failed: " + message);
        }
        const auto result = RelationalRun{
            count,
            std::chrono::duration<double, std::milli>(end - start).count(),
            command_gpu_ms(command)};
        pool->release();
        return result;
    }
};

ScanCopyF32::ScanCopyF32(std::span<const float> input)
    : impl_(std::make_unique<Impl>(input)) {}
ScanCopyF32::~ScanCopyF32() = default;
ScanCopyF32::ScanCopyF32(ScanCopyF32&&) noexcept = default;
ScanCopyF32& ScanCopyF32::operator=(ScanCopyF32&&) noexcept = default;
RelationalRun ScanCopyF32::execute() { return impl_->execute(); }
std::span<const float> ScanCopyF32::output() const noexcept {
    return {static_cast<const float*>(impl_->output->contents()), impl_->count};
}
const std::string& ScanCopyF32::device_name() const noexcept { return impl_->name; }

struct Q6FilterMaterialize::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* bitmap_pipeline{};
    MTL::ComputePipelineState* popcount_pipeline{};
    MTL::ComputePipelineState* scan_pipeline{};
    MTL::ComputePipelineState* add_pipeline{};
    MTL::ComputePipelineState* materialize_pipeline{};
    MTL::ComputePipelineState* count_pipeline{};
    MTL::Buffer* quantity{};
    MTL::Buffer* discount{};
    MTL::Buffer* date{};
    MTL::Buffer* bitmap{};
    MTL::Buffer* output_rows{};
    MTL::Buffer* output_count{};
    MTL::Buffer* dummy_sum{};
    std::vector<MTL::Buffer*> level_inputs;
    std::vector<MTL::Buffer*> level_offsets;
    std::vector<std::uint32_t> level_counts;
    std::uint32_t rows{};
    std::uint32_t words{};
    std::uint32_t width{};
    std::uint32_t execution_width_value{};
    std::string name;

    Impl(
        const std::filesystem::path& library_path,
        tpch::LineitemView input,
        std::uint32_t requested_width)
        : width(requested_width) {
        if (input.row_count() == 0 || input.discount.size() != input.row_count() ||
            input.ship_date_yyyymmdd.size() != input.row_count()) {
            throw std::invalid_argument("Q6 columns must be non-empty and equally sized");
        }
        if (input.row_count() > UINT32_MAX || width < 32 || width > 512 ||
            !std::has_single_bit(width)) {
            throw std::invalid_argument("invalid materialize size or threadgroup width");
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
                throw std::runtime_error("could not load metallib: " + error_description(error));
            }
            bitmap_pipeline = make_pipeline(device, library, "tpch_q6_filter_bitmap");
            popcount_pipeline = make_pipeline(device, library, "bitmap_popcounts_u32");
            scan_pipeline = make_pipeline(device, library, "exclusive_scan_u32_blocks");
            add_pipeline = make_pipeline(device, library, "add_scan_block_offsets_u32");
            materialize_pipeline = make_pipeline(device, library, "materialize_bitmap_rows");
            count_pipeline = make_pipeline(device, library, "bitmap_materialized_count");
            execution_width_value =
                static_cast<std::uint32_t>(scan_pipeline->threadExecutionWidth());
            if (width % execution_width_value != 0 ||
                width > scan_pipeline->maxTotalThreadsPerThreadgroup()) {
                throw std::invalid_argument("threadgroup width is incompatible with prefix scan");
            }

            rows = static_cast<std::uint32_t>(input.row_count());
            words = (rows + 31U) / 32U;
            quantity = wrap_shared(device, input.quantity);
            discount = wrap_shared(device, input.discount);
            date = wrap_shared(device, input.ship_date_yyyymmdd);
            bitmap = device->newBuffer(
                static_cast<NS::UInteger>(words) * sizeof(std::uint32_t),
                MTL::ResourceStorageModePrivate);
            output_rows = device->newBuffer(
                static_cast<NS::UInteger>(rows) * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            output_count = device->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared);
            dummy_sum = device->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModePrivate);
            if (!queue || !quantity || !discount || !date || !bitmap || !output_rows ||
                !output_count || !dummy_sum) {
                throw std::runtime_error("could not allocate materialize Metal resources");
            }

            auto count = words;
            while (true) {
                level_counts.push_back(count);
                level_inputs.push_back(device->newBuffer(
                    static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                    MTL::ResourceStorageModePrivate));
                level_offsets.push_back(device->newBuffer(
                    static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                    MTL::ResourceStorageModePrivate));
                if (!level_inputs.back() || !level_offsets.back()) {
                    throw std::runtime_error("could not allocate prefix-scan levels");
                }
                if (count <= width) break;
                count = (count + width - 1U) / width;
            }
            library->release();
            pool->release();
        } catch (...) {
            if (library) library->release();
            pool->release();
            release();
            throw;
        }
    }

    ~Impl() { release(); }
    void release() noexcept {
        for (auto* value : level_offsets) if (value) value->release();
        for (auto* value : level_inputs) if (value) value->release();
        level_offsets.clear(); level_inputs.clear();
        for (auto** value : {&dummy_sum, &output_count, &output_rows, &bitmap, &date,
                             &discount, &quantity}) {
            if (*value) { (*value)->release(); *value = nullptr; }
        }
        for (auto** value : {&count_pipeline, &materialize_pipeline, &add_pipeline,
                             &scan_pipeline, &popcount_pipeline, &bitmap_pipeline}) {
            if (*value) { (*value)->release(); *value = nullptr; }
        }
        if (queue) queue->release();
        if (device) device->release();
        queue = nullptr; device = nullptr;
    }

    void dispatch_1d(
        MTL::ComputeCommandEncoder* encoder,
        std::uint32_t item_count) const {
        const auto groups = (item_count + width - 1U) / width;
        encoder->dispatchThreadgroups(
            MTL::Size(groups, 1, 1), MTL::Size(width, 1, 1));
    }

    [[nodiscard]] RelationalRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        const auto host_start = std::chrono::steady_clock::now();
        auto* command = queue->commandBuffer();
        auto* encoder = command != nullptr ? command->computeCommandEncoder() : nullptr;
        if (!command || !encoder) {
            pool->release();
            throw std::runtime_error("could not create materialize command buffer");
        }
        encoder->setComputePipelineState(bitmap_pipeline);
        encoder->setBuffer(quantity, 0, 0);
        encoder->setBuffer(discount, 0, 1);
        encoder->setBuffer(date, 0, 2);
        encoder->setBuffer(bitmap, 0, 3);
        encoder->setBytes(&rows, sizeof(rows), 4);
        dispatch_1d(encoder, words);

        encoder->setComputePipelineState(popcount_pipeline);
        encoder->setBuffer(bitmap, 0, 0);
        encoder->setBuffer(level_inputs.front(), 0, 1);
        encoder->setBytes(&words, sizeof(words), 2);
        dispatch_1d(encoder, words);

        for (std::size_t level = 0; level < level_counts.size(); ++level) {
            const auto count = level_counts[level];
            encoder->setComputePipelineState(scan_pipeline);
            encoder->setBuffer(level_inputs[level], 0, 0);
            encoder->setBuffer(level_offsets[level], 0, 1);
            encoder->setBuffer(
                level + 1 < level_inputs.size() ? level_inputs[level + 1] : dummy_sum,
                0, 2);
            encoder->setBytes(&count, sizeof(count), 3);
            dispatch_1d(encoder, count);
        }
        for (std::size_t level = level_counts.size(); level-- > 1;) {
            const auto lower = level - 1;
            const auto count = level_counts[lower];
            encoder->setComputePipelineState(add_pipeline);
            encoder->setBuffer(level_offsets[lower], 0, 0);
            encoder->setBuffer(level_offsets[level], 0, 1);
            encoder->setBytes(&count, sizeof(count), 2);
            encoder->setBytes(&width, sizeof(width), 3);
            dispatch_1d(encoder, count);
        }

        encoder->setComputePipelineState(materialize_pipeline);
        encoder->setBuffer(bitmap, 0, 0);
        encoder->setBuffer(level_offsets.front(), 0, 1);
        encoder->setBuffer(output_rows, 0, 2);
        encoder->setBytes(&words, sizeof(words), 3);
        encoder->setBytes(&rows, sizeof(rows), 4);
        dispatch_1d(encoder, words);

        encoder->setComputePipelineState(count_pipeline);
        encoder->setBuffer(bitmap, 0, 0);
        encoder->setBuffer(level_offsets.front(), 0, 1);
        encoder->setBuffer(output_count, 0, 2);
        encoder->setBytes(&words, sizeof(words), 3);
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        encoder->endEncoding();
        command->commit();
        command->waitUntilCompleted();
        const auto host_end = std::chrono::steady_clock::now();
        if (command->status() == MTL::CommandBufferStatusError) {
            const auto message = error_description(command->error());
            pool->release();
            throw std::runtime_error("Metal materialize failed: " + message);
        }
        const auto count = *static_cast<const std::uint32_t*>(output_count->contents());
        const auto result = RelationalRun{
            count,
            std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            command_gpu_ms(command)};
        pool->release();
        return result;
    }
};

Q6FilterMaterialize::Q6FilterMaterialize(
    const std::filesystem::path& metal_library,
    tpch::LineitemView input,
    std::uint32_t threadgroup_width)
    : impl_(std::make_unique<Impl>(metal_library, input, threadgroup_width)) {}
Q6FilterMaterialize::~Q6FilterMaterialize() = default;
Q6FilterMaterialize::Q6FilterMaterialize(Q6FilterMaterialize&&) noexcept = default;
Q6FilterMaterialize& Q6FilterMaterialize::operator=(Q6FilterMaterialize&&) noexcept = default;
RelationalRun Q6FilterMaterialize::execute() { return impl_->execute(); }
std::span<const std::uint32_t> Q6FilterMaterialize::output() const noexcept {
    const auto count = *static_cast<const std::uint32_t*>(impl_->output_count->contents());
    return {static_cast<const std::uint32_t*>(impl_->output_rows->contents()), count};
}
const std::string& Q6FilterMaterialize::device_name() const noexcept { return impl_->name; }
std::uint32_t Q6FilterMaterialize::execution_width() const noexcept {
    return impl_->execution_width_value;
}

}  // namespace joule::operators::gpu
