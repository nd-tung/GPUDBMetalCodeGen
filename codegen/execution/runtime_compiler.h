#pragma once
#include <Metal/Metal.hpp>
#include <string>
#include <unordered_map>

namespace codegen {

// Compiles Metal source at runtime and caches pipeline states.
class RuntimeCompiler {
public:
    explicit RuntimeCompiler(MTL::Device* device) : device_(device) {}
    ~RuntimeCompiler();

    // Compile Metal source into a library. Returns nullptr on error (prints diagnostics).
    MTL::Library* compile(const std::string& source);

    // Get or create a pipeline state for a kernel name from a compiled library.
    MTL::ComputePipelineState* getPipeline(MTL::Library* lib, const std::string& kernelName);

    // Clear pipeline cache (releases all cached PSOs)
    void clearCache();

    // Number of cached pipelines
    size_t cacheSize() const { return pipelineCache_.size(); }

    // Toggle Metal -ffast-math for subsequent compile() calls (default true).
    static void setFastMathEnabled(bool on) { sFastMath_ = on; }
    static bool fastMathEnabled() { return sFastMath_; }

    // Compiled query: library + pipeline states for each phase
    struct CompiledQuery {
        MTL::Library* library = nullptr;
        std::vector<MTL::ComputePipelineState*> pipelines;
        std::vector<std::string> kernelNames;
    };

private:
    MTL::Device* device_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelineCache_;
    static bool sFastMath_;
};

} // namespace codegen
