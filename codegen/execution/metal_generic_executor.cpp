#include "metal_generic_executor.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <limits>

namespace codegen {

namespace {

std::string bytesToGiB(size_t bytes) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return os.str();
}

void checkCommandBufferStatus(MTL::CommandBuffer* cmdBuf,
                              const std::string& phaseName) {
    auto status = cmdBuf->status();
    if (status == MTL::CommandBufferStatusError) {
        std::string msg = "Metal command buffer failed";
        if (!phaseName.empty()) msg += " in phase '" + phaseName + "'";
        if (auto* err = cmdBuf->error()) {
            if (auto* desc = err->localizedDescription()) {
                msg += ": ";
                msg += desc->utf8String();
            }
        }
        throw std::runtime_error(msg);
    }
}

} // namespace

MetalGenericExecutor::MetalGenericExecutor(MTL::Device* device, MTL::CommandQueue* cmdQueue)
    : device_(device), cmdQueue_(cmdQueue) {}

MetalGenericExecutor::~MetalGenericExecutor() {
    releaseAllocatedBuffers();
}

void MetalGenericExecutor::registerTableRowCount(const std::string& tableName, size_t rowCount) {
    sizeResolver_.registerSymbol(tableSizeName(tableName), rowCount);
    sizeResolver_.registerSymbol("num" + tableName, rowCount);
}

void MetalGenericExecutor::registerTableBuffer(const std::string& name,
                                                MTL::Buffer* buffer,
                                                size_t rowCount) {
    tables_[name] = {buffer, rowCount, false};
    sizeResolver_.registerSymbol(tableSizeName(name), rowCount);
    sizeResolver_.registerSymbol("num" + name, rowCount);
}

void MetalGenericExecutor::registerAllocatedBuffer(const std::string& name, MTL::Buffer* buffer) {
    auto it = allocatedBuffers_.find(name);
    if (it != allocatedBuffers_.end() && it->second && it->second != buffer) {
        it->second->release();
    }
    phaseAllocatedBuffers_.erase(name);
    allocatedBuffers_[name] = buffer;
}

void MetalGenericExecutor::registerSymbol(const std::string& name, size_t value) {
    sizeResolver_.registerSymbol(name, value);
}
bool MetalGenericExecutor::tryGetSymbol(const std::string& name, size_t& out) const {
    if (!sizeResolver_.hasSymbol(name)) return false;
    out = sizeResolver_.getSymbol(name);
    return true;
}

bool MetalGenericExecutor::tryResolveSizeExpression(const std::string& expr,
                                                    size_t& out) const {
    try {
        out = sizeResolver_.resolve(expr);
        return true;
    } catch (...) {
        return false;
    }
}

void MetalGenericExecutor::registerScalarInt(const std::string& name, int value) {
    scalarInts_[name] = value;
}

void MetalGenericExecutor::registerScalarFloat(const std::string& name, float value) {
    scalarFloats_[name] = value;
}
MTL::ComputePipelineState* MetalGenericExecutor::findPSO(
    const RuntimeCompiler::CompiledQuery& cq, const std::string& name) {
    for (size_t i = 0; i < cq.kernelNames.size(); i++)
        if (cq.kernelNames[i] == name) return cq.pipelines[i];
    return nullptr;
}

MTL::ComputePipelineState* MetalGenericExecutor::findPSO(
    const std::string& name) const {
    auto it = pipelineStates_.find(name);
    return it != pipelineStates_.end() ? it->second : nullptr;
}

BufferMap MetalGenericExecutor::allocatePhaseBuffers(
    const MetalCodegen::PhaseInfo& phase) {

    BufferMap buffers;

    for (const auto& b : phase.bindings) {
        switch (b.kind) {
            case MetalParamKind::TableData: {
                // Prefer column binding, then table binding.
                auto tIt = tables_.find(b.name);
                if (tIt == tables_.end()) tIt = tables_.find(b.tableName);
                if (tIt != tables_.end()) {
                    buffers[b.name] = tIt->second.buffer;
                } else {
                    throw std::runtime_error(
                        "MetalGenericExecutor: required table/column '" + b.name +
                        "' (table='" + b.tableName + "') is not registered");
                }
                break;
            }

            case MetalParamKind::TableSize: {
                // Table sizes are passed via setBytes.
                size_t rowCount = 0;
                auto tIt = tables_.find(b.tableName);
                if (tIt != tables_.end()) {
                    rowCount = tIt->second.rowCount;
                } else {
                    std::string symName = tableSizeName(b.tableName);
                    if (sizeResolver_.hasSymbol(symName)) {
                        rowCount = sizeResolver_.getSymbol(symName);
                    }
                }

                if (rowCount > 0) {
                    uint32_t sz = static_cast<uint32_t>(rowCount);
                    std::string key = b.name;
                    if (allocatedBuffers_.count(key)) {
                        memcpy(allocatedBuffers_[key]->contents(), &sz, sizeof(uint32_t));
                        buffers[key] = allocatedBuffers_[key];
                    } else {
                        auto* buf = device_->newBuffer(sizeof(uint32_t),
                                                       MTL::ResourceStorageModeShared);
                        if (!buf) {
                            throw std::runtime_error(
                                "MetalGenericExecutor: failed to allocate scalar buffer '" +
                                key + "' (4 bytes)");
                        }
                        memcpy(buf->contents(), &sz, sizeof(uint32_t));
                        allocatedBuffers_[key] = buf;
                        buffers[key] = buf;
                    }
                }
                break;
            }

            case MetalParamKind::DeviceBuffer: {
                std::string key = b.name;
                if (allocatedBuffers_.count(key)) {
                    // Reuse buffers registered by preprocessing, hooks, or earlier phases.
                    buffers[key] = allocatedBuffers_[key];
                } else if (!b.sizeExpr.empty()) {
                    // sizeExpr resolves to element count; elemSize converts it to bytes.
                    size_t count = sizeResolver_.resolve(b.sizeExpr);
                    size_t elemSize = b.elemSizeBytes();

                    if (elemSize != 0 &&
                        count > std::numeric_limits<size_t>::max() / elemSize) {
                        throw std::runtime_error(
                            "MetalGenericExecutor: buffer size overflow for '" +
                            key + "', count=" + std::to_string(count) +
                            ", elemSize=" + std::to_string(elemSize) +
                            ", sizeExpr='" + b.sizeExpr + "'");
                    }
                    size_t totalBytes = count * elemSize;
                    if (totalBytes == 0) totalBytes = elemSize;
                    auto* buf = device_->newBuffer(totalBytes,
                                                   MTL::ResourceStorageModeShared);
                    if (!buf) {
                        throw std::runtime_error(
                            "MetalGenericExecutor: failed to allocate buffer '" +
                            key + "' for " + std::to_string(count) + " x " +
                            std::to_string(elemSize) + " bytes (" +
                            bytesToGiB(totalBytes) + " GiB), sizeExpr='" +
                            b.sizeExpr + "'");
                    }
                    allocatedBuffers_[key] = buf;
                    phaseAllocatedBuffers_.insert(key);
                    buffers[key] = buf;
                }
                break;
            }

            case MetalParamKind::ConstantScalar:
            case MetalParamKind::ConstantData:
                break;
        }
    }

    return buffers;
}

void MetalGenericExecutor::zeroInitBuffers(const MetalCodegen::PhaseInfo& phase,
                                            const BufferMap& buffers) {
    if (skipZeroInit_) return;
    for (const auto& b : phase.bindings) {
        if (b.zeroInit && b.kind == MetalParamKind::DeviceBuffer) {
            auto it = buffers.find(b.name);
            if (it != buffers.end() && it->second) {
                memset(it->second->contents(), b.fillByte, it->second->length());
                it->second->didModifyRange(NS::Range::Make(0, it->second->length()));
            }
        }
    }
}

GenericResult MetalGenericExecutor::collectResult(const MetalCodegen& codegen) const {
    return MetalResultCollector::collect(codegen.getResultSchema(), allocatedBuffers_);
}

void MetalGenericExecutor::bindPhaseBuffers(MTL::ComputeCommandEncoder* encoder,
                                             const MetalCodegen::PhaseInfo& phase,
                                             const BufferMap& buffers) {
    for (const auto& b : phase.bindings) {
        if (b.bufferIndex < 0) continue;

        switch (b.kind) {
            case MetalParamKind::TableData:
            case MetalParamKind::DeviceBuffer: {
                auto it = buffers.find(b.name);
                if (it != buffers.end() && it->second) {
                    encoder->setBuffer(it->second, 0, b.bufferIndex);
                } else {
                    // Hooks may register buffers after the per-execute snapshot.
                    auto it2 = allocatedBuffers_.find(b.name);
                    if (it2 != allocatedBuffers_.end() && it2->second) {
                        encoder->setBuffer(it2->second, 0, b.bufferIndex);
                    }
                }
                break;
            }

            case MetalParamKind::TableSize: {
                auto it = buffers.find(b.name);
                if (it != buffers.end() && it->second) {
                    encoder->setBuffer(it->second, 0, b.bufferIndex);
                }
                break;
            }

            case MetalParamKind::ConstantScalar: {
                auto ii = scalarInts_.find(b.name);
                if (ii != scalarInts_.end()) {
                    encoder->setBytes(&ii->second, sizeof(int), b.bufferIndex);
                } else {
                    auto fi = scalarFloats_.find(b.name);
                    if (fi != scalarFloats_.end()) {
                        encoder->setBytes(&fi->second, sizeof(float), b.bufferIndex);
                    } else if (!b.sizeExpr.empty()) {
                        // Derived scalar from sizeResolver_.
                        uint32_t v = static_cast<uint32_t>(
                            sizeResolver_.resolve(b.sizeExpr));
                        encoder->setBytes(&v, sizeof(uint32_t), b.bufferIndex);
                    } else if (sizeResolver_.hasSymbol(b.name)) {
                        uint32_t v = static_cast<uint32_t>(
                            sizeResolver_.getSymbol(b.name));
                        encoder->setBytes(&v, sizeof(uint32_t), b.bufferIndex);
                    }
                }
                break;
            }

            case MetalParamKind::ConstantData:
                break;
        }
    }
}

void MetalGenericExecutor::encodePhase(MTL::ComputeCommandEncoder* encoder,
                                       MTL::ComputePipelineState* pso,
                                       const MetalCodegen::PhaseInfo& phase,
                                       const BufferMap& buffers) {
    encoder->setComputePipelineState(pso);
    bindPhaseBuffers(encoder, phase, buffers);

    NS::UInteger tgSize = pso->maxTotalThreadsPerThreadgroup();
    if (tgSize > (NS::UInteger)phase.threadgroupSize)
        tgSize = phase.threadgroupSize;

    if (phase.isSingleThread) {
        encoder->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                      MTL::Size::Make(1, 1, 1));
        return;
    }

    // Keep small scans occupied without exceeding Metal's grid-Y limit.
    constexpr NS::UInteger kMinThreadgroups = 1024;
    constexpr NS::UInteger kMaxThreadgroups = 65535;
    NS::UInteger numTG = kMinThreadgroups;
    if (!phase.scannedTable.empty()) {
        std::string sym = tableSizeName(phase.scannedTable);
        if (sizeResolver_.hasSymbol(sym)) {
            size_t rowCount = sizeResolver_.getSymbol(sym);
            NS::UInteger computed = (rowCount + tgSize - 1) / tgSize;
            if (computed > kMinThreadgroups) numTG = computed;
            if (numTG > kMaxThreadgroups) numTG = kMaxThreadgroups;
        }
    }
    if (phase.maxThreadgroups > 0 &&
        numTG > (NS::UInteger)phase.maxThreadgroups) {
        numTG = phase.maxThreadgroups;
    }
    encoder->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1),
                                  MTL::Size::Make(tgSize, 1, 1));
}

MetalExecutionResult MetalGenericExecutor::execute(
    const RuntimeCompiler::CompiledQuery& compiled,
    const MetalCodegen& codegen,
    int warmupRuns,
    int measuredRuns) {
    return execute(compiled, codegen, warmupRuns, measuredRuns, 0, -1);
}

MetalExecutionResult MetalGenericExecutor::execute(
    const RuntimeCompiler::CompiledQuery& compiled,
    const MetalCodegen& codegen,
    int warmupRuns,
    int measuredRuns,
    int firstPhase,
    int lastPhase) {

    MetalExecutionResult execResult;
    const auto& allPhases = codegen.getPhases();

    if (allPhases.empty()) {
        std::cerr << "MetalGenericExecutor: no phases to execute\n";
        return execResult;
    }
    if (lastPhase < 0) lastPhase = (int)allPhases.size();
    if (firstPhase >= lastPhase) return execResult;

    if (firstPhase == 0 && lastPhase == (int)allPhases.size()) {
        // Full executions refresh phase-owned buffers but keep registered inputs.
        releasePhaseAllocatedBuffers();
    }

    // Populate pipeline lookup for post-dispatch hooks.
    pipelineStates_.clear();
    for (size_t i = 0; i < compiled.kernelNames.size(); ++i)
        pipelineStates_[compiled.kernelNames[i]] = compiled.pipelines[i];

    auto allocStart = std::chrono::high_resolution_clock::now();
    BufferMap allBuffers;
        for (int _pi = firstPhase; _pi < lastPhase; _pi++) {
            const auto& phase = allPhases[_pi];
            auto phaseBuffers = allocatePhaseBuffers(phase);
        for (auto& [k, v] : phaseBuffers) {
            if (!allBuffers.count(k))
                allBuffers[k] = v;
        }
    }
    auto allocEnd = std::chrono::high_resolution_clock::now();
    execResult.bufferAllocTimeMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(allocEnd - allocStart).count());

    int totalRuns = warmupRuns + measuredRuns;

    for (int iter = 0; iter < totalRuns; iter++) {
        // Zero-init output buffers each iteration.
        for (int _pi = firstPhase; _pi < lastPhase; _pi++) {
            const auto& phase = allPhases[_pi];
            zeroInitBuffers(phase, allBuffers);
        }

        const bool isMeasured = (iter == totalRuns - 1);

        // Hooks require host-visible phase boundaries.
        bool hasHook = false;
        for (int _pi = firstPhase; _pi < lastPhase; _pi++) {
            if (allPhases[_pi].postDispatchHook) { hasHook = true; break; }
        }

        if (!isMeasured && !hasHook) {
            // Fast warmup path.
            auto* cmdBuf = cmdQueue_->commandBuffer();
            auto* encoder = cmdBuf->computeCommandEncoder();

            for (size_t pi = (size_t)firstPhase; pi < (size_t)lastPhase; pi++) {
                const auto& phase = allPhases[pi];
                auto* pso = findPSO(compiled, phase.name);
                if (!pso) {
                    std::cerr << "MetalGenericExecutor: PSO not found for '"
                              << phase.name << "'\n";
                    continue;
                }
                encodePhase(encoder, pso, phase, allBuffers);

                if ((int)pi + 1 < lastPhase) {
                    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
                }
            }

            encoder->endEncoding();
            cmdBuf->commit();
            cmdBuf->waitUntilCompleted();
            checkCommandBufferStatus(cmdBuf, "warmup");
            continue;
        }

        // One command buffer per phase for timing and hook ordering.
        double totalGpuSec = 0.0;
        execResult.phaseTimesMs.clear();
        execResult.phaseNames.clear();
        execResult.phaseTimesMs.reserve((size_t)(lastPhase - firstPhase));
        execResult.phaseNames.reserve((size_t)(lastPhase - firstPhase));

        for (size_t pi = (size_t)firstPhase; pi < (size_t)lastPhase; pi++) {
            const auto& phase = allPhases[pi];
            auto* pso = findPSO(compiled, phase.name);
            if (!pso) {
                std::cerr << "MetalGenericExecutor: PSO not found for '"
                          << phase.name << "'\n";
                execResult.phaseTimesMs.push_back(0.0f);
                execResult.phaseNames.push_back(phase.name);
                continue;
            }

            // Hooks may replace or add named buffers between phases.
            for (auto& kv : allBuffers) {
                auto it = allocatedBuffers_.find(kv.first);
                if (it != allocatedBuffers_.end() && it->second) kv.second = it->second;
            }

            auto* cmdBuf = cmdQueue_->commandBuffer();
            auto* encoder = cmdBuf->computeCommandEncoder();

            encodePhase(encoder, pso, phase, allBuffers);

            encoder->endEncoding();
            cmdBuf->commit();
            cmdBuf->waitUntilCompleted();
            checkCommandBufferStatus(cmdBuf, phase.name);

            double phaseSec = cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime();
            double hookGpuMs = 0.0;

            // Fold hook-launched GPU work into the same phase total.
            if (phase.postDispatchHook) {
                hookGpuMs = phase.postDispatchHook(*this);
                if (hookGpuMs < 0.0) hookGpuMs = 0.0;
            }

            totalGpuSec += phaseSec + hookGpuMs / 1000.0;
            execResult.phaseTimesMs.push_back(
                static_cast<float>(phaseSec * 1000.0 + hookGpuMs));
            execResult.phaseNames.push_back(phase.name);
        }

        execResult.totalKernelTimeMs = static_cast<float>(totalGpuSec * 1000.0);
    }

    execResult.result = MetalResultCollector::collect(codegen.getResultSchema(), allBuffers);

    return execResult;
}

void MetalGenericExecutor::releaseAllocatedBuffers() {
    for (auto& [_, buf] : allocatedBuffers_) {
        if (buf) buf->release();
    }
    allocatedBuffers_.clear();
    phaseAllocatedBuffers_.clear();

    for (auto& [_, info] : tables_) {
        if (info.ownsBuffer && info.buffer)
            info.buffer->release();
    }
    tables_.clear();
}

void MetalGenericExecutor::releasePhaseAllocatedBuffers() {
    for (const auto& name : phaseAllocatedBuffers_) {
        auto it = allocatedBuffers_.find(name);
        if (it != allocatedBuffers_.end()) {
            if (it->second) it->second->release();
            allocatedBuffers_.erase(it);
        }
    }
    phaseAllocatedBuffers_.clear();
}

} // namespace codegen
