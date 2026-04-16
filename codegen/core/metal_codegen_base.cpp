#include "metal_codegen_base.h"
#include "metal_common_header.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace codegen {

// ===================================================================
// Common Metal header — now maintained in metal_common_header.h
// ===================================================================

std::string MetalCodegen::commonHeader() {
    return kMetalCommonHeader;
}

// ===================================================================
// Indentation
// ===================================================================

std::string MetalCodegen::indent() const {
    return std::string(indentLevel_ * INDENT_SIZE, ' ');
}

// ===================================================================
// Code emission
// ===================================================================

void MetalCodegen::addLine(const std::string& line) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addLine: no active phase");
    currentPhase_->code += indent() + line + "\n";
}

void MetalCodegen::addBlock(const std::string& header, std::function<void()> body,
                             const std::string& trailing) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addBlock: no active phase");
    currentPhase_->code += indent() + header + " {\n";
    indentLevel_++;
    body();
    indentLevel_--;
    currentPhase_->code += indent() + "}" + trailing + "\n";
}

void MetalCodegen::addIf(const std::string& condition, std::function<void()> body) {
    addBlock("if (" + condition + ")", body);
}

void MetalCodegen::addComment(const std::string& comment) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addComment: no active phase");
    currentPhase_->code += indent() + "// " + comment + "\n";
}

void MetalCodegen::increaseIndent() { indentLevel_++; }
void MetalCodegen::decreaseIndent() { if (indentLevel_ > 0) indentLevel_--; }

void MetalCodegen::addRawCode(const std::string& code) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addRawCode: no active phase");
    currentPhase_->code += code;
}

void MetalCodegen::addHelper(const std::string& code) {
    helperCode_ += code + "\n";
}

// ===================================================================
// Phase management
// ===================================================================

void MetalCodegen::beginPhase(const std::string& phaseName) {
    if (currentPhase_)
        throw std::runtime_error("MetalCodegen::beginPhase: previous phase '" +
                                 currentPhase_->name + "' not ended");
    phases_.push_back({});
    currentPhase_ = &phases_.back();
    currentPhase_->name = phaseName;
    indentLevel_ = 1;
}

void MetalCodegen::endPhase() {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::endPhase: no active phase");
    currentPhase_ = nullptr;
    indentLevel_ = 1;
}

int MetalCodegen::phaseCount() const {
    return static_cast<int>(phases_.size());
}

void MetalCodegen::setPhaseScannedTable(const std::string& tableName) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseScannedTable: no active phase");
    currentPhase_->scannedTable = tableName;
}

void MetalCodegen::setPhaseThreadgroupSize(int size) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseThreadgroupSize: no active phase");
    currentPhase_->threadgroupSize = size;
}

void MetalCodegen::setPhaseSingleThread(bool single) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseSingleThread: no active phase");
    currentPhase_->isSingleThread = single;
}

void MetalCodegen::setPhaseMaxThreadgroups(int max) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseMaxThreadgroups: no active phase");
    currentPhase_->maxThreadgroups = max;
}

// ===================================================================
// Parameter registration
// ===================================================================

void MetalCodegen::addTableParam(const std::string& table, const std::string& metalType) {
    if (!currentPhase_)
        throw std::runtime_error("addTableParam: no active phase");

    // Data pointer
    {
        MetalParamBinding b;
        b.name = "d_" + table;
        b.metalDecl = "device const " + metalType + "*";
        b.kind = MetalParamKind::TableData;
        b.tableName = table;
        b.elementType = metalType;
        b.readOnly = true;
        currentPhase_->bindings.push_back(b);
    }
    // Size
    {
        MetalParamBinding b;
        b.name = "n_" + table;
        b.metalDecl = "constant uint&";
        b.kind = MetalParamKind::TableSize;
        b.tableName = table;
        b.elementType = "uint";
        currentPhase_->bindings.push_back(b);
    }

    registeredTables_.insert(table);
}

void MetalCodegen::addColumnParam(const std::string& paramName, const std::string& metalType,
                                   const std::string& tableName) {
    if (!currentPhase_)
        throw std::runtime_error("addColumnParam: no active phase");
    MetalParamBinding b;
    b.name = paramName;
    b.metalDecl = "device const " + metalType + "*";
    b.kind = MetalParamKind::TableData;
    b.tableName = tableName.empty() ? paramName : tableName;
    b.elementType = metalType;
    b.readOnly = true;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addTableSizeParam(const std::string& table) {
    if (!currentPhase_)
        throw std::runtime_error("addTableSizeParam: no active phase");
    MetalParamBinding b;
    b.name = "n_" + table;
    b.metalDecl = "constant uint&";
    b.kind = MetalParamKind::TableSize;
    b.tableName = table;
    b.elementType = "uint";
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBufferParam(const std::string& name, const std::string& elemType,
                                   const std::string& sizeExpr, bool zeroInit, int fillByte) {
    if (!currentPhase_)
        throw std::runtime_error("addBufferParam: no active phase");
    // Skip if already registered in this phase
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device " + elemType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = elemType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = zeroInit;
    b.fillByte = fillByte;
    currentPhase_->bindings.push_back(b);

    if (!sizeExpr.empty())
        globalBufferSizes_[name] = sizeExpr;
}

void MetalCodegen::addAtomicBufferParam(const std::string& name,
                                         const std::string& atomicType,
                                         const std::string& sizeExpr) {
    if (!currentPhase_)
        throw std::runtime_error("addAtomicBufferParam: no active phase");
    // Skip if already registered in this phase
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device " + atomicType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = atomicType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = true;
    currentPhase_->bindings.push_back(b);

    if (!sizeExpr.empty())
        globalBufferSizes_[name] = sizeExpr;
}

void MetalCodegen::addScalarParam(const std::string& name, const std::string& type) {
    if (!currentPhase_)
        throw std::runtime_error("addScalarParam: no active phase");
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "constant " + type + "&";
    b.kind = MetalParamKind::ConstantScalar;
    b.elementType = type;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addConstantDataParam(const std::string& name, const std::string& type,
                                         size_t bytes) {
    if (!currentPhase_)
        throw std::runtime_error("addConstantDataParam: no active phase");
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "constant " + type + "*";
    b.kind = MetalParamKind::ConstantData;
    b.elementType = type;
    b.hostCopyBytes = bytes;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBitmapReadParam(const std::string& name, const std::string& sizeExpr) {
    if (!currentPhase_)
        throw std::runtime_error("addBitmapReadParam: no active phase");
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device const uint*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = "uint";
    b.sizeExpr = sizeExpr;
    b.readOnly = true;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBitmapWriteParam(const std::string& name, const std::string& sizeExpr) {
    addAtomicBufferParam(name, "atomic_uint", sizeExpr);
}

void MetalCodegen::addHashMapParam(const std::string& mapName,
                                    const std::string& keysName,
                                    const std::string& valuesLoName,
                                    const std::string& valuesHiName,
                                    const std::string& sizeExpr) {
    addAtomicBufferParam(keysName, "atomic_uint", sizeExpr);
    addAtomicBufferParam(valuesLoName, "atomic_uint", sizeExpr);
    addAtomicBufferParam(valuesHiName, "atomic_uint", sizeExpr);
    addScalarParam("n_" + mapName, "uint");
}

void MetalCodegen::addHashMapReadParam(const std::string& mapName,
                                        const std::string& keysName,
                                        const std::string& valuesLoName,
                                        const std::string& valuesHiName) {
    addBufferParam(keysName, "uint", "", false);
    addBufferParam(valuesLoName, "uint", "", false);
    addBufferParam(valuesHiName, "uint", "", false);
    addScalarParam("n_" + mapName, "uint");
}

// ===================================================================
// Global buffer sizes
// ===================================================================

void MetalCodegen::setBufferSize(const std::string& name, const std::string& sizeExpr) {
    globalBufferSizes_[name] = sizeExpr;
}

const std::unordered_map<std::string, std::string>& MetalCodegen::getGlobalBufferSizes() const {
    return globalBufferSizes_;
}

// ===================================================================
// Output schema registration
// ===================================================================

void MetalCodegen::registerScalarAggOutput(const std::string& loBuffer,
                                            const std::string& hiBuffer,
                                            const std::string& type) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    // Store the buffer names for the next scalarAgg entry
    // (will be associated with the column added by registerScalarAggColumn)
    (void)type;
    // We store lo/hi buffers temporarily — they'll be set on the scalarAggs entry
    // by registerScalarAggColumn which is called right after
    scalarAggPendingLo_ = loBuffer;
    scalarAggPendingHi_ = hiBuffer;
}

void MetalCodegen::registerScalarAggColumn(const std::string& displayName, int index,
                                            int scaleDown) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    MetalResultSchema::ScalarAggEntry entry;
    entry.displayName = displayName;
    entry.scaleDown = scaleDown;
    entry.loBuffer = scalarAggPendingLo_;
    entry.hiBuffer = scalarAggPendingHi_;
    entry.isLongPair = !scalarAggPendingHi_.empty();
    entry.elementType = entry.isLongPair ? "uint" : "float";
    (void)index;
    resultSchema_.scalarAggs.push_back(entry);
}

void MetalCodegen::registerMaterializeOutput(const std::string& counterBuffer) {
    resultSchema_.kind = MetalResultSchema::MATERIALIZE;
    resultSchema_.counterBuffer = counterBuffer;
}

void MetalCodegen::registerOutputColumn(const std::string& displayName,
                                         const std::string& bufferName,
                                         const std::string& elementType,
                                         int stringLen) {
    MetalResultSchema::ColumnDesc col;
    col.displayName = displayName;
    col.bufferName = bufferName;
    col.elementType = elementType;
    col.stringLen = stringLen;
    resultSchema_.columns.push_back(col);
}

void MetalCodegen::registerKeyedAggOutput(const std::string& bufferName,
                                           int numBuckets, int valuesPerBucket,
                                           const std::vector<MetalResultSchema::KeyedAggSlot>& slots) {
    resultSchema_.kind = MetalResultSchema::KEYED_AGG;
    resultSchema_.keyedAgg.bufferName = bufferName;
    resultSchema_.keyedAgg.numBuckets = numBuckets;
    resultSchema_.keyedAgg.valuesPerBucket = valuesPerBucket;
    resultSchema_.keyedAgg.slots = slots;
}

const MetalResultSchema& MetalCodegen::getResultSchema() const {
    return resultSchema_;
}

// ===================================================================
// Buffer index assignment
// ===================================================================

void MetalCodegen::assignBufferIndices(PhaseInfo& phase) {
    int nextIndex = 0;
    for (auto& b : phase.bindings) {
        b.bufferIndex = nextIndex++;
    }
}

// ===================================================================
// Kernel signature generation
// ===================================================================

std::string MetalCodegen::generateSignature(const PhaseInfo& phase) const {
    std::ostringstream sig;
    sig << "kernel void " << phase.name << "(\n";

    bool needsThreadPos = !phase.isSingleThread;

    for (size_t i = 0; i < phase.bindings.size(); i++) {
        const auto& b = phase.bindings[i];
        sig << "    " << b.metalDecl << " " << b.name
            << " [[buffer(" << b.bufferIndex << ")]]";
        if (i + 1 < phase.bindings.size() || needsThreadPos)
            sig << ",";
        sig << "\n";
    }

    if (!phase.isSingleThread) {
        sig << "    uint tid [[thread_position_in_grid]],\n";
        sig << "    uint tpg [[threads_per_grid]],\n";
        sig << "    uint lid [[thread_position_in_threadgroup]],\n";
        sig << "    uint tg_size [[threads_per_threadgroup]],\n";
        sig << "    uint gid [[threadgroup_position_in_grid]],\n";
        sig << "    uint simd_lane [[thread_index_in_simdgroup]],\n";
        sig << "    uint simd_id [[simdgroup_index_in_threadgroup]]\n";
    } else {
        // Single-thread kernel still needs thread_position_in_grid
        sig << "    uint tid [[thread_position_in_grid]]\n";
    }

    sig << ")";
    return sig.str();
}

// ===================================================================
// Final output assembly
// ===================================================================

std::string MetalCodegen::print() {
    std::ostringstream out;

    // 1. Common header (SIMD reductions, bitmap, atomics)
    out << commonHeader();

    // 2. Helper / device functions
    if (!helperCode_.empty()) {
        out << "// --- Helper functions ---\n";
        out << helperCode_ << "\n";
    }

    // 3. Assign buffer indices and emit each phase as one kernel function
    for (size_t pi = 0; pi < phases_.size(); pi++) {
        assignBufferIndices(phases_[pi]);

        const auto& phase = phases_[pi];
        out << "\n// === Phase " << pi << ": " << phase.name << " ===\n";
        out << generateSignature(phase) << " {\n";
        out << phase.code;
        out << "}\n";
    }

    return out.str();
}

// ===================================================================
// Phase info accessors
// ===================================================================

const std::vector<MetalCodegen::PhaseInfo>& MetalCodegen::getPhases() const {
    return phases_;
}

std::vector<MetalParamBinding> MetalCodegen::getAllBindings() const {
    std::vector<MetalParamBinding> all;
    std::unordered_set<std::string> seen;
    for (const auto& phase : phases_) {
        for (const auto& b : phase.bindings) {
            if (seen.insert(b.name).second) {
                all.push_back(b);
            }
        }
    }
    return all;
}

} // namespace codegen
