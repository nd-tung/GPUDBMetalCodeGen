#include "runtime_compiler.h"
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <sstream>
#include <functional>
#include <utility>

namespace codegen {

bool RuntimeCompiler::sFastMath_ = false;
bool RuntimeCompiler::sGlobalCacheEnabled_ = true;
std::unordered_map<std::string, MTL::Library*> RuntimeCompiler::sLibraryCache_;
std::unordered_map<std::string, MTL::ComputePipelineState*>
    RuntimeCompiler::sGlobalPipelineCache_;

namespace {

std::string metalDeviceName(MTL::Device* device) {
    if (!device || !device->name()) return "unknown-device";
    return device->name()->utf8String();
}

std::string metalSourceCacheKey(MTL::Device* device,
                                const std::string& source,
                                bool fastMath) {
    return metalDeviceName(device) + "|" +
           (fastMath ? "fastmath=1" : "fastmath=0") + "|" +
           std::to_string(source.size()) + "|" +
           std::to_string(std::hash<std::string>{}(source));
}

} // namespace

RuntimeCompiler::~RuntimeCompiler() {
    clearCache();
}

void RuntimeCompiler::clearCache() {
    for (auto& [name, pso] : pipelineCache_) {
        if (pso) pso->release();
    }
    pipelineCache_.clear();
    libraries_.clear();
}

MTL::Library* RuntimeCompiler::compile(const std::string& source) {
    currentLibraryCacheKey_.clear();
    if (sGlobalCacheEnabled_) {
        currentLibraryCacheKey_ =
            metalSourceCacheKey(device_, source, sFastMath_);
        auto cached = sLibraryCache_.find(currentLibraryCacheKey_);
        if (cached != sLibraryCache_.end())
            return cached->second;
    }

    NS::Error* error = nullptr;
    auto* sourceStr = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    MetalOwnedObject<MTL::CompileOptions> opts(MTL::CompileOptions::alloc()->init());
    if (!opts) {
        std::cerr << "Metal compilation failed: could not allocate compile options" << std::endl;
        return nullptr;
    }
    opts->setFastMathEnabled(sFastMath_);
    MetalOwnedObject<MTL::Library> ownedLibrary(
        device_->newLibrary(sourceStr, opts.get(), &error));

    if (!ownedLibrary) {
        std::cerr << "Metal compilation failed:" << std::endl;
        if (error)
            std::cerr << error->localizedDescription()->utf8String() << std::endl;
        // Print source with line numbers for debugging
        std::istringstream ss(source);
        std::string line;
        int lineNo = 1;
        while (std::getline(ss, line)) {
            std::cerr << lineNo++ << ": " << line << "\n";
        }
        return nullptr;
    }
    MTL::Library* library = ownedLibrary.get();
    if (sGlobalCacheEnabled_ && !currentLibraryCacheKey_.empty()) {
        sLibraryCache_[currentLibraryCacheKey_] = library;
        ownedLibrary.release();
    } else {
        libraries_.push_back(std::move(ownedLibrary));
    }
    return library;
}

MTL::ComputePipelineState* RuntimeCompiler::getPipeline(MTL::Library* lib, const std::string& kernelName) {
    if (sGlobalCacheEnabled_ && !currentLibraryCacheKey_.empty()) {
        const std::string key = currentLibraryCacheKey_ + "|" + kernelName;
        auto global = sGlobalPipelineCache_.find(key);
        if (global != sGlobalPipelineCache_.end())
            return global->second;

        auto* funcName = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
        MetalOwnedObject<MTL::Function> func(lib->newFunction(funcName));
        if (!func) {
            std::cerr << "Kernel function not found: " << kernelName << std::endl;
            return nullptr;
        }

        NS::Error* error = nullptr;
        MetalOwnedObject<MTL::ComputePipelineState> ownedPso(
            device_->newComputePipelineState(func.get(), &error));

        if (!ownedPso) {
            std::cerr << "Failed to create pipeline for " << kernelName << ": ";
            if (error) std::cerr << error->localizedDescription()->utf8String();
            std::cerr << std::endl;
            return nullptr;
        }

        MTL::ComputePipelineState* pso = ownedPso.get();
        sGlobalPipelineCache_[key] = pso;
        ownedPso.release();
        return pso;
    }

    auto it = pipelineCache_.find(kernelName);
    if (it != pipelineCache_.end()) return it->second;

    auto* funcName = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
    MetalOwnedObject<MTL::Function> func(lib->newFunction(funcName));
    if (!func) {
        std::cerr << "Kernel function not found: " << kernelName << std::endl;
        return nullptr;
    }

    NS::Error* error = nullptr;
    MetalOwnedObject<MTL::ComputePipelineState> ownedPso(
        device_->newComputePipelineState(func.get(), &error));

    if (!ownedPso) {
        std::cerr << "Failed to create pipeline for " << kernelName << ": ";
        if (error) std::cerr << error->localizedDescription()->utf8String();
        std::cerr << std::endl;
        return nullptr;
    }

    MTL::ComputePipelineState* pso = ownedPso.get();
    pipelineCache_[kernelName] = pso;
    ownedPso.release();
    return pso;
}

} // namespace codegen
