#include "joule/operators/gpu/filter_project.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
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
    return device->newBuffer(const_cast<T*>(values.data()), values.size_bytes(),
                             MTL::ResourceStorageModeShared, nullptr);
}

[[nodiscard]] MTL::ComputePipelineState* make_pipeline(MTL::Device* device, MTL::Library* library,
                                                       const char* name) {
    auto* function = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (function == nullptr) {
        throw std::runtime_error(std::string("missing Metal function: ") + name);
    }
    NS::Error* error = nullptr;
    auto* pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
        throw std::runtime_error(std::string("could not create pipeline '") + name +
                                 "': " + error_description(error));
    }
    return pipeline;
}

[[nodiscard]] std::string metal_device_name(MTL::Device* device) {
    return device != nullptr && device->name() != nullptr && device->name()->utf8String() != nullptr
               ? device->name()->utf8String()
               : "";
}

[[nodiscard]] double command_gpu_ms(MTL::CommandBuffer* command) {
    const auto begin = command->GPUStartTime();
    const auto end = command->GPUEndTime();
    return end >= begin ? (end - begin) * 1'000.0 : 0.0;
}

void validate_input(tpch::LineitemView input) {
    const auto rows = input.row_count();
    if (rows == 0 || input.part_key.size() != rows || input.extended_price.size() != rows ||
        input.discount.size() != rows || input.ship_date_yyyymmdd.size() != rows) {
        throw std::invalid_argument("filter-project columns must be non-empty and equally sized");
    }
    if (rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("filter-project supports at most 2^32-1 rows");
    }
}

}  // namespace

struct Q6FilterProject::Impl {
    MTL::Device* device{};
    MTL::CommandQueue* queue{};
    MTL::ComputePipelineState* bitmap_pipeline{};
    MTL::ComputePipelineState* popcount_pipeline{};
    MTL::ComputePipelineState* scan_pipeline{};
    MTL::ComputePipelineState* add_pipeline{};
    MTL::ComputePipelineState* scatter_pipeline{};
    MTL::ComputePipelineState* count_pipeline{};
    MTL::Buffer* quantity{};
    MTL::Buffer* part_key{};
    MTL::Buffer* extended_price{};
    MTL::Buffer* discount{};
    MTL::Buffer* date{};
    MTL::Buffer* bitmap{};
    MTL::Buffer* output_records{};
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

    Impl(const std::filesystem::path& library_path, tpch::LineitemView input,
         std::uint32_t requested_width)
        : width(requested_width) {
        validate_input(input);
        if (width < 32 || width > 512 || !std::has_single_bit(width)) {
            throw std::invalid_argument("invalid filter-project threadgroup width");
        }

        auto* pool = NS::AutoreleasePool::alloc()->init();
        MTL::Library* library = nullptr;
        try {
            device = MTL::CreateSystemDefaultDevice();
            if (device == nullptr) {
                throw std::runtime_error("no Metal device is available");
            }
            name = metal_device_name(device);
            queue = device->newCommandQueue();
            NS::Error* error = nullptr;
            const auto path = library_path.string();
            library = device->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding),
                                         &error);
            if (library == nullptr) {
                throw std::runtime_error("could not load metallib: " + error_description(error));
            }

            bitmap_pipeline = make_pipeline(device, library, "tpch_q6_filter_bitmap");
            popcount_pipeline = make_pipeline(device, library, "bitmap_popcounts_u32");
            scan_pipeline = make_pipeline(device, library, "exclusive_scan_u32_blocks");
            add_pipeline = make_pipeline(device, library, "add_scan_block_offsets_u32");
            scatter_pipeline = make_pipeline(device, library, "materialize_q6_project_records");
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
            part_key = wrap_shared(device, input.part_key);
            extended_price = wrap_shared(device, input.extended_price);
            discount = wrap_shared(device, input.discount);
            date = wrap_shared(device, input.ship_date_yyyymmdd);
            bitmap = device->newBuffer(static_cast<NS::UInteger>(words) * sizeof(std::uint32_t),
                                       MTL::ResourceStorageModePrivate);
            output_records =
                device->newBuffer(static_cast<NS::UInteger>(rows) * sizeof(FilterProjectRecord),
                                  MTL::ResourceStorageModeShared);
            output_count = device->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared);
            dummy_sum = device->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModePrivate);
            if (!queue || !quantity || !part_key || !extended_price || !discount || !date ||
                !bitmap || !output_records || !output_count || !dummy_sum) {
                throw std::runtime_error("could not allocate filter-project Metal resources");
            }

            auto count = words;
            while (true) {
                level_counts.push_back(count);
                level_inputs.push_back(
                    device->newBuffer(static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                                      MTL::ResourceStorageModePrivate));
                level_offsets.push_back(
                    device->newBuffer(static_cast<NS::UInteger>(count) * sizeof(std::uint32_t),
                                      MTL::ResourceStorageModePrivate));
                if (!level_inputs.back() || !level_offsets.back()) {
                    throw std::runtime_error(
                        "could not allocate filter-project prefix-scan levels");
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
        for (auto* value : level_offsets) {
            if (value) value->release();
        }
        for (auto* value : level_inputs) {
            if (value) value->release();
        }
        level_offsets.clear();
        level_inputs.clear();
        for (auto** value : {&dummy_sum, &output_count, &output_records, &bitmap, &date, &discount,
                             &extended_price, &part_key, &quantity}) {
            if (*value) {
                (*value)->release();
                *value = nullptr;
            }
        }
        for (auto** value : {&count_pipeline, &scatter_pipeline, &add_pipeline, &scan_pipeline,
                             &popcount_pipeline, &bitmap_pipeline}) {
            if (*value) {
                (*value)->release();
                *value = nullptr;
            }
        }
        if (queue) queue->release();
        if (device) device->release();
        queue = nullptr;
        device = nullptr;
    }

    void dispatch_1d(MTL::ComputeCommandEncoder* encoder, std::uint32_t item_count) const {
        const auto groups = (item_count + width - 1U) / width;
        encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(width, 1, 1));
    }

    [[nodiscard]] FilterProjectRun execute() {
        auto* pool = NS::AutoreleasePool::alloc()->init();
        const auto host_start = std::chrono::steady_clock::now();
        auto* command = queue->commandBuffer();
        auto* encoder = command != nullptr ? command->computeCommandEncoder() : nullptr;
        if (command == nullptr || encoder == nullptr) {
            pool->release();
            throw std::runtime_error("could not create filter-project command buffer");
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
                level + 1 < level_inputs.size() ? level_inputs[level + 1] : dummy_sum, 0, 2);
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

        encoder->setComputePipelineState(scatter_pipeline);
        encoder->setBuffer(bitmap, 0, 0);
        encoder->setBuffer(level_offsets.front(), 0, 1);
        encoder->setBuffer(part_key, 0, 2);
        encoder->setBuffer(extended_price, 0, 3);
        encoder->setBuffer(discount, 0, 4);
        encoder->setBuffer(output_records, 0, 5);
        encoder->setBytes(&words, sizeof(words), 6);
        encoder->setBytes(&rows, sizeof(rows), 7);
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
            throw std::runtime_error("Metal filter-project failed: " + message);
        }

        const auto count = *static_cast<const std::uint32_t*>(output_count->contents());
        const auto result = FilterProjectRun{
            count, std::chrono::duration<double, std::milli>(host_end - host_start).count(),
            command_gpu_ms(command)};
        pool->release();
        return result;
    }
};

Q6FilterProject::Q6FilterProject(const std::filesystem::path& metal_library,
                                 tpch::LineitemView input, std::uint32_t threadgroup_width)
    : impl_(std::make_unique<Impl>(metal_library, input, threadgroup_width)) {}
Q6FilterProject::~Q6FilterProject() = default;
Q6FilterProject::Q6FilterProject(Q6FilterProject&&) noexcept = default;
Q6FilterProject& Q6FilterProject::operator=(Q6FilterProject&&) noexcept = default;
FilterProjectRun Q6FilterProject::execute() { return impl_->execute(); }
std::span<const FilterProjectRecord> Q6FilterProject::output() const noexcept {
    const auto count = *static_cast<const std::uint32_t*>(impl_->output_count->contents());
    return {static_cast<const FilterProjectRecord*>(impl_->output_records->contents()), count};
}
const std::string& Q6FilterProject::device_name() const noexcept { return impl_->name; }
std::uint32_t Q6FilterProject::execution_width() const noexcept {
    return impl_->execution_width_value;
}

}  // namespace joule::operators::gpu
