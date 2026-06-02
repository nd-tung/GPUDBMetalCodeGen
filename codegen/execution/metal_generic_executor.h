#pragma once
// Data-driven GPU dispatch for MetalCodegen phases.

#include "metal_codegen_base.h"
#include "metal_result_collector.h"
#include "runtime_compiler.h"
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace codegen {

struct MetalExecutionResult {
    GenericResult result;
    // GPU totals from the measured run.
    float totalKernelTimeMs = 0.0f;
    std::vector<float> phaseTimesMs;
    std::vector<std::string> phaseNames;    // parallel to phaseTimesMs

    // Wall/CPU accounting for work that is not visible in Metal GPU timestamps.
    float executeWallTimeMs = 0.0f;
    float hookCpuTimeMs = 0.0f;
    float hookGpuTimeMs = 0.0f; // Included in totalKernelTimeMs.
    float resultCollectTimeMs = 0.0f;
    std::vector<float> phaseHookCpuTimesMs; // parallel to phaseTimesMs
    std::vector<float> phaseHookGpuTimesMs; // parallel to phaseTimesMs
    std::vector<float> phaseWallTimesMs;    // parallel to phaseTimesMs
    std::vector<float> phaseOverheadTimesMs; // wall - GPU - hook CPU

    // CPU-side sub-phases are filled by the caller.
    float analyzeTimeMs    = 0.0f;  // SQL analysis
    float planTimeMs       = 0.0f;  // IR build plus GenericRelPlan to MetalQueryPlan
    float codegenTimeMs    = 0.0f;  // Plan to Metal source
    float compileTimeMs    = 0.0f;  // Metal source to MTLLibrary
    float psoTimeMs        = 0.0f;  // MTLLibrary to pipeline states
    float dataLoadTimeMs   = 0.0f;  // .tbl parse + host buffer fill + per-query setup
    float bufferAllocTimeMs = 0.0f; // GPU buffer allocation / upload inside execute()
};

class MetalGenericExecutor {
public:
    MetalGenericExecutor(MTL::Device* device, MTL::CommandQueue* cmdQueue);

    // Releases any owned GPU buffers left after early returns.
    ~MetalGenericExecutor();

    MetalGenericExecutor(const MetalGenericExecutor&) = delete;
    MetalGenericExecutor& operator=(const MetalGenericExecutor&) = delete;

    // Table buffers are borrowed; their row counts seed size symbols.
    void registerTableBuffer(const std::string& name, MTL::Buffer* buffer,
                             size_t rowCount);
    void registerTableBuffer(const std::string& tableName,
                             const std::string& columnName,
                             MTL::Buffer* buffer,
                             size_t rowCount);

    // Register row count for n_tableName resolution.
    void registerTableRowCount(const std::string& tableName, size_t rowCount);

    // Allocated buffers are shared by name across phases and result collection.
    void registerAllocatedBuffer(const std::string& name, MTL::Buffer* buffer);

    // Register a symbolic size.
    void registerSymbol(const std::string& name, size_t value);

    // Look up a registered symbolic size; returns false if absent.
    bool tryGetSymbol(const std::string& name, size_t& out) const;

    // Resolve a full size expression against the current symbol table.
    bool tryResolveSizeExpression(const std::string& expr, size_t& out) const;

    // Register scalar constants used by setBytes bindings.
    void registerScalarInt(const std::string& name, int value);
    void registerScalarFloat(const std::string& name, float value);

    MetalExecutionResult execute(
        const RuntimeCompiler::CompiledQuery& compiled,
        const MetalCodegen& codegen,
        int warmupRuns = 2,
        int measuredRuns = 1
    );

    // Execute phase range [firstPhase, lastPhase); -1 means through the end.
    MetalExecutionResult execute(
        const RuntimeCompiler::CompiledQuery& compiled,
        const MetalCodegen& codegen,
        int warmupRuns,
        int measuredRuns,
        int firstPhase,
        int lastPhase
    );

    // Skip zero-fill when chunked execution must preserve partial outputs.
    void setSkipZeroInit(bool skip) { skipZeroInit_ = skip; }

    // When enabled, measured runs execute one command buffer per phase and
    // return per-phase wall/residual timings. Normal execution batches phases
    // between host hooks to avoid profiling overhead in end-to-end timings.
    void setDetailedPhaseTiming(bool enabled) { detailedPhaseTiming_ = enabled; }

    // Allocate phase-owned device/output/scratch buffers in private GPU
    // storage. Result buffers are copied back to shared temporary buffers
    // before MetalResultCollector reads them.
    void setPrivateDeviceBuffers(bool enabled) { privateDeviceBuffers_ = enabled; }

    // Collect current buffers without re-running GPU work.
    GenericResult collectResult(const MetalCodegen& codegen) const;

    // Release owned scratch/output buffers.
    void releaseAllocatedBuffers();

    // Release buffers allocated from phase bindings while preserving
    // registered table and preprocessing buffers.
    void releasePhaseAllocatedBuffers();

    // Access an allocated buffer by name.
    MTL::Buffer* getAllocatedBuffer(const std::string& name) const {
        auto it = allocatedBuffers_.find(name);
        return it != allocatedBuffers_.end() ? it->second : nullptr;
    }

    MTL::CommandQueue* commandQueue() const { return cmdQueue_; }

    MTL::Device* device() const { return device_; }

    // Used by post-dispatch hooks that re-dispatch kernels.
    MTL::ComputePipelineState* getPipelineState(const std::string& name) const {
        auto it = pipelineStates_.find(name);
        return it != pipelineStates_.end() ? it->second : nullptr;
    }

private:
    MTL::Device* device_;
    MTL::CommandQueue* cmdQueue_;
    MetalSizeResolver sizeResolver_;
    bool skipZeroInit_ = false;
    bool detailedPhaseTiming_ = false;
    bool privateDeviceBuffers_ = false;

    struct TableInfo {
        MTL::Buffer* buffer = nullptr;
        size_t rowCount = 0;
        bool ownsBuffer = false;
    };
    std::unordered_map<std::string, TableInfo> tables_;
    std::unordered_map<std::string, std::string> tableBufferShortOwners_;
    std::unordered_set<std::string> ambiguousTableBufferNames_;

    // Scratch/output buffers owned for the lifetime of this executor.
    std::unordered_map<std::string, MTL::Buffer*> allocatedBuffers_;
    std::unordered_set<std::string> phaseAllocatedBuffers_;

    // Populated at execute() start for post-dispatch hooks.
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelineStates_;

    std::unordered_map<std::string, int> scalarInts_;
    std::unordered_map<std::string, float> scalarFloats_;

    BufferMap allocatePhaseBuffers(const MetalCodegen::PhaseInfo& phase);

    void bindPhaseBuffers(MTL::ComputeCommandEncoder* encoder,
                          const MetalCodegen::PhaseInfo& phase,
                          const BufferMap& buffers);

    void zeroInitBuffersForRange(const std::vector<MetalCodegen::PhaseInfo>& phases,
                                 int firstPhase,
                                 int lastPhase,
                                 const BufferMap& buffers);

    GenericResult collectResultFromBuffers(const MetalResultSchema& schema,
                                           const BufferMap& buffers) const;

    MTL::ComputePipelineState* findPSO(const RuntimeCompiler::CompiledQuery& cq,
                                        const std::string& name);
    MTL::ComputePipelineState* findPSO(const std::string& name) const;

    // Shared by warmup and measured loops.
    void encodePhase(MTL::ComputeCommandEncoder* encoder,
                     MTL::ComputePipelineState* pso,
                     const MetalCodegen::PhaseInfo& phase,
                     const BufferMap& buffers);
};

} // namespace codegen
