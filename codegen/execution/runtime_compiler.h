#pragma once
#include <Metal/Metal.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace codegen {

// Compiles Metal source at runtime and caches pipeline states.
class RuntimeCompiler {
public:
    explicit RuntimeCompiler(MTL::Device* device) : device_(device) {}
    ~RuntimeCompiler();

    // Compile Metal source into a library; returns nullptr after diagnostics on error.
    MTL::Library* compile(const std::string& source);

    // Get or create a pipeline state for a kernel name from a compiled library.
    MTL::ComputePipelineState* getPipeline(MTL::Library* lib, const std::string& kernelName);

    // Release all cached pipeline states.
    void clearCache();

    // Number of cached pipeline states.
    size_t cacheSize() const { return pipelineCache_.size(); }

    // Toggle Metal -ffast-math for subsequent compile() calls (default false).
    static void setFastMathEnabled(bool on) { sFastMath_ = on; }
    static bool fastMathEnabled() { return sFastMath_; }

    // Toggle the process-level source/PSO cache. Cold-cache experiments use
    // this to force compilation without changing normal execution behavior.
    static void setGlobalCacheEnabled(bool on) { sGlobalCacheEnabled_ = on; }
    static bool globalCacheEnabled() { return sGlobalCacheEnabled_; }

    // Runtime objects needed to execute a compiled query.
    struct CompiledQuery {
        MTL::Library* library = nullptr;
        std::vector<MTL::ComputePipelineState*> pipelines;
        std::vector<std::string> kernelNames;
    };

private:
    MTL::Device* device_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelineCache_;
    std::string currentLibraryCacheKey_;
    static bool sFastMath_;
    static bool sGlobalCacheEnabled_;
    static std::unordered_map<std::string, MTL::Library*> sLibraryCache_;
    static std::unordered_map<std::string, MTL::ComputePipelineState*> sGlobalPipelineCache_;
};

} // namespace codegen
