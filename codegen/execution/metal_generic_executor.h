#pragma once
// ===================================================================
// Metal Generic Executor — data-driven GPU dispatch
// ===================================================================
//
// Reads MetalCodegen's bindings and result schema to generically:
// 1. Allocate GPU buffers based on symbolic size expressions
// 2. Bind buffers at auto-assigned [[buffer(N)]] indices
// 3. Dispatch each phase with correct threadgroup config
// 4. Collect results via MetalResultCollector
// ===================================================================

#include "metal_codegen_base.h"
#include "metal_result_collector.h"
#include "runtime_compiler.h"
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace codegen {

struct MetalExecutionResult {
    GenericResult result;
    // GPU totals (measured run only)
    float totalKernelTimeMs = 0.0f;
    std::vector<float> phaseTimesMs;        // per-phase GPU time on the measured run
    std::vector<std::string> phaseNames;    // parallel to phaseTimesMs

    // CPU-side sub-phases (filled by caller)
    float analyzeTimeMs    = 0.0f;  // SQL → AnalyzedQuery
    float planTimeMs       = 0.0f;  // AnalyzedQuery → MetalQueryPlan
    float codegenTimeMs    = 0.0f;  // Plan → Metal source
    float compileTimeMs    = 0.0f;  // Metal source → MTLLibrary
    float psoTimeMs        = 0.0f;  // MTLLibrary → pipeline states
    float dataLoadTimeMs   = 0.0f;  // .tbl parse + host buffer fill + per-query setup
    float bufferAllocTimeMs = 0.0f; // GPU buffer allocation / upload inside execute()

    // Same value as dataLoadTimeMs; kept under this name because the
    // TIMING_CSV output schema and downstream analysis scripts already
    // reference `parseTimeMs`.
    float parseTimeMs      = 0.0f;
    float postTimeMs       = 0.0f;
};

class MetalGenericExecutor {
public:
    MetalGenericExecutor(MTL::Device* device, MTL::CommandQueue* cmdQueue);

    // RAII safety net: any owned GPU buffers still held when the executor goes
    // out of scope are released. The normal lifecycle still calls
    // releaseAllocatedBuffers() explicitly after results are consumed.
    ~MetalGenericExecutor();

    MetalGenericExecutor(const MetalGenericExecutor&) = delete;
    MetalGenericExecutor& operator=(const MetalGenericExecutor&) = delete;

    // Register a pre-allocated Metal buffer (zero-copy column or pre-built
    // dimension table). The executor does NOT take ownership.
    void registerTableBuffer(const std::string& name, MTL::Buffer* buffer,
                             size_t rowCount);

    // Register row count for a table (so n_tableName can be resolved)
    void registerTableRowCount(const std::string& tableName, size_t rowCount);

    // Register a pre-allocated buffer into the cross-phase buffer map
    // (used for pre-computed data like bitmaps or lookup arrays)
    void registerAllocatedBuffer(const std::string& name, MTL::Buffer* buffer);

    // Register a symbolic size (e.g. "maxCustkey" → 150000)
    void registerSymbol(const std::string& name, size_t value);

    // Register a scalar constant (for constant T& params set via setBytes)
    void registerScalarInt(const std::string& name, int value);
    void registerScalarFloat(const std::string& name, float value);

    // Execute a compiled query through new codegen pipeline
    MetalExecutionResult execute(
        const RuntimeCompiler::CompiledQuery& compiled,
        const MetalCodegen& codegen,
        int warmupRuns = 2,
        int measuredRuns = 1
    );

    // Clean up allocated buffers (call after execute, results consumed)
    void releaseAllocatedBuffers();

    // Debug: access an allocated buffer by name
    MTL::Buffer* getAllocatedBuffer(const std::string& name) const {
        auto it = allocatedBuffers_.find(name);
        return it != allocatedBuffers_.end() ? it->second : nullptr;
    }

private:
    MTL::Device* device_;
    MTL::CommandQueue* cmdQueue_;
    MetalSizeResolver sizeResolver_;

    // Registered tables: name → {buffer, rowCount}
    struct TableInfo {
        MTL::Buffer* buffer = nullptr;
        size_t rowCount = 0;
        bool ownsBuffer = false;  // if we allocated it
    };
    std::unordered_map<std::string, TableInfo> tables_;

    // Buffers allocated for scratch/output (we own these)
    std::unordered_map<std::string, MTL::Buffer*> allocatedBuffers_;

    // Scalar constant values (set via setBytes during binding)
    std::unordered_map<std::string, int> scalarInts_;
    std::unordered_map<std::string, float> scalarFloats_;

    // Allocate all buffers needed by a phase
    BufferMap allocatePhaseBuffers(const MetalCodegen::PhaseInfo& phase);

    // Bind all parameters for a phase
    void bindPhaseBuffers(MTL::ComputeCommandEncoder* encoder,
                          const MetalCodegen::PhaseInfo& phase,
                          const BufferMap& buffers);

    // Zero-init buffers that require it
    void zeroInitBuffers(const MetalCodegen::PhaseInfo& phase,
                         const BufferMap& buffers);

    // Find PSO by kernel name
    MTL::ComputePipelineState* findPSO(const RuntimeCompiler::CompiledQuery& cq,
                                        const std::string& name);
};

} // namespace codegen
