#include "metal_codegen_base.h"
#include "metal_common_header.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace codegen {

std::string MetalCodegen::commonHeader() {
    return kMetalCommonHeader;
}

std::string MetalCodegen::indent() const {
    return std::string(indentLevel_ * INDENT_SIZE, ' ');
}

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

void MetalCodegen::setPhasePostDispatchHook(PostDispatchHook hook) {
    if (!currentPhase_)
        throw std::runtime_error("setPhasePostDispatchHook: no active phase");
    currentPhase_->postDispatchHook = std::move(hook);
}

void MetalCodegen::pushBinding(const char* op, MetalParamBinding b, bool dedup) {
    if (!currentPhase_)
        throw std::runtime_error(std::string(op) + ": no active phase");
    if (dedup) {
        for (const auto& existing : currentPhase_->bindings)
            if (existing.name == b.name) return;
    }
    // Only writable buffers own their size registration.
    if (!b.sizeExpr.empty() && b.kind == MetalParamKind::DeviceBuffer && !b.readOnly)
        globalBufferSizes_[b.name] = b.sizeExpr;
    currentPhase_->bindings.push_back(std::move(b));
}

void MetalCodegen::addTableParam(const std::string& table, const std::string& metalType) {
    {
        MetalParamBinding b;
        b.name = "d_" + table;
        b.metalTypeDecl = "device const " + metalType + "*";
        b.kind = MetalParamKind::TableData;
        b.tableName = table;
        b.elementType = metalType;
        b.readOnly = true;
        pushBinding("addTableParam", std::move(b), /*dedup=*/false);
    }
    {
        MetalParamBinding b;
        b.name = "n_" + table;
        b.metalTypeDecl = "constant uint&";
        b.kind = MetalParamKind::TableSize;
        b.tableName = table;
        b.elementType = "uint";
        pushBinding("addTableParam", std::move(b), /*dedup=*/false);
    }

    registeredTables_.insert(table);
}

void MetalCodegen::addColumnParam(const std::string& paramName, const std::string& metalType,
                                   const std::string& tableName) {
    MetalParamBinding b;
    b.name = paramName;
    b.metalTypeDecl = "device const " + metalType + "*";
    b.kind = MetalParamKind::TableData;
    b.tableName = tableName.empty() ? paramName : tableName;
    b.elementType = metalType;
    b.readOnly = true;
    pushBinding("addColumnParam", std::move(b), /*dedup=*/true);
}

void MetalCodegen::addTableSizeParam(const std::string& table) {
    MetalParamBinding b;
    b.name = "n_" + table;
    b.metalTypeDecl = "constant uint&";
    b.kind = MetalParamKind::TableSize;
    b.tableName = table;
    b.elementType = "uint";
    pushBinding("addTableSizeParam", std::move(b), /*dedup=*/false);
}

void MetalCodegen::addBufferParam(const std::string& name, const std::string& elemType,
                                   const std::string& sizeExpr, bool zeroInit, int fillByte) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "device " + elemType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = elemType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = zeroInit;
    b.fillByte = fillByte;
    pushBinding("addBufferParam", std::move(b), /*dedup=*/true);
}

void MetalCodegen::addAtomicBufferParam(const std::string& name,
                                         const std::string& atomicType,
                                         const std::string& sizeExpr,
                                         int fillByte) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "device " + atomicType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = atomicType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = true;
    b.fillByte = fillByte;
    pushBinding("addAtomicBufferParam", std::move(b), /*dedup=*/true);
}

void MetalCodegen::addScalarParam(const std::string& name, const std::string& type) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "constant " + type + "&";
    b.kind = MetalParamKind::ConstantScalar;
    b.elementType = type;
    pushBinding("addScalarParam", std::move(b), /*dedup=*/true);
}

void MetalCodegen::addResolvedScalarParam(const std::string& name,
                                          const std::string& type,
                                          const std::string& sizeExpr) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "constant " + type + "&";
    b.kind = MetalParamKind::ConstantScalar;
    b.elementType = type;
    b.sizeExpr = sizeExpr;
    pushBinding("addResolvedScalarParam", std::move(b), /*dedup=*/true);
}

void MetalCodegen::addConstantDataParam(const std::string& name, const std::string& type,
                                         size_t bytes) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "constant " + type + "*";
    b.kind = MetalParamKind::ConstantData;
    b.elementType = type;
    b.hostCopyBytes = bytes;
    pushBinding("addConstantDataParam", std::move(b), /*dedup=*/false);
}

void MetalCodegen::addBitmapReadParam(const std::string& name, const std::string& sizeExpr) {
    MetalParamBinding b;
    b.name = name;
    b.metalTypeDecl = "device const atomic_uint*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = "atomic_uint";
    b.sizeExpr = sizeExpr;
    b.readOnly = true;
    pushBinding("addBitmapReadParam", std::move(b), /*dedup=*/false);
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
    // Empty sizes bind build-phase storage instead of allocating.
    addBufferParam(keysName, "uint", "", false);
    addBufferParam(valuesLoName, "uint", "", false);
    addBufferParam(valuesHiName, "uint", "", false);
    addScalarParam("n_" + mapName, "uint");
}

void MetalCodegen::setBufferSize(const std::string& name, const std::string& sizeExpr) {
    globalBufferSizes_[name] = sizeExpr;
}

const std::unordered_map<std::string, std::string>& MetalCodegen::getGlobalBufferSizes() const {
    return globalBufferSizes_;
}

void MetalCodegen::registerScalarAggOutput(const std::string& loBuffer,
                                            const std::string& hiBuffer,
                                            const std::string& type) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    // Paired with the next registerScalarAggColumn call.
    scalarAggPendingLo_ = loBuffer;
    scalarAggPendingHi_ = hiBuffer;
    scalarAggPendingType_ = type;
}

void MetalCodegen::registerScalarAggColumn(const std::string& displayName, int index,
                                            int scaleDown, ExprPtr projectionExpr) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    MetalResultSchema::ScalarAggEntry entry;
    entry.displayName = displayName;
    entry.scaleDown = scaleDown;
    entry.loBuffer = scalarAggPendingLo_;
    entry.hiBuffer = scalarAggPendingHi_;
    entry.isLongPair = !scalarAggPendingHi_.empty();
    entry.elementType = scalarAggPendingType_;
    entry.projectionExpr = std::move(projectionExpr);
    (void)index;
    resultSchema_.scalarAggs.push_back(entry);
}

void MetalCodegen::registerScalarAggAverageColumn(const std::string& displayName,
                                                  const std::string& numeratorLoBuffer,
                                                  const std::string& numeratorHiBuffer,
                                                  const std::string& denominatorLoBuffer,
                                                  const std::string& denominatorHiBuffer,
                                                  const std::string& type,
                                                  int scaleDown,
                                                  ExprPtr projectionExpr) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    MetalResultSchema::ScalarAggEntry entry;
    entry.displayName = displayName;
    entry.scaleDown = scaleDown;
    entry.loBuffer = numeratorLoBuffer;
    entry.hiBuffer = numeratorHiBuffer;
    entry.denomLoBuffer = denominatorLoBuffer;
    entry.denomHiBuffer = denominatorHiBuffer;
    entry.isLongPair = !numeratorHiBuffer.empty();
    entry.denomIsLongPair = !denominatorHiBuffer.empty();
    entry.divideByDenominator = true;
    entry.elementType = type == "long" ? "uint" : "float";
    entry.projectionExpr = std::move(projectionExpr);
    resultSchema_.scalarAggs.push_back(entry);
}

void MetalCodegen::registerMaterializeOutput(const std::string& counterBuffer) {
    if (resultSchema_.kind != MetalResultSchema::MATERIALIZE ||
        resultSchema_.counterBuffer != counterBuffer) {
        // A new materialize counter starts a separate output schema.
        resultSchema_.columns.clear();
    }
    resultSchema_.kind = MetalResultSchema::MATERIALIZE;
    resultSchema_.counterBuffer = counterBuffer;
}

void MetalCodegen::registerOutputColumn(const std::string& displayName,
                                         const std::string& bufferName,
                                         const std::string& elementType,
                                         int stringLen,
                                         int scaleDown,
                                         bool isLongPair) {
    MetalResultSchema::ColumnDesc col;
    col.displayName = displayName;
    col.bufferName = bufferName;
    col.elementType = elementType;
    col.stringLen = stringLen;
    col.scaleDown = scaleDown;
    col.isLongPair = isLongPair;
    resultSchema_.columns.push_back(col);
}

void MetalCodegen::registerKeyedAggOutput(const std::string& bufferName,
                                           int numBuckets, int valuesPerBucket,
                                           const std::vector<MetalResultSchema::KeyedAggSlot>& slots,
                                           const std::string& keyDisplayName,
                                           int keyBase) {
    resultSchema_.kind = MetalResultSchema::KEYED_AGG;
    resultSchema_.keyedAgg.bufferName = bufferName;
    resultSchema_.keyedAgg.numBuckets = numBuckets;
    resultSchema_.keyedAgg.valuesPerBucket = valuesPerBucket;
    resultSchema_.keyedAgg.keyDisplayName = keyDisplayName;
    resultSchema_.keyedAgg.keyBase = keyBase;
    resultSchema_.keyedAgg.slots = slots;
}

void MetalCodegen::setKeyedAggHaving(const PredPtr& havingPredicate) {
    resultSchema_.keyedAgg.havingPredicate = havingPredicate;
}

void MetalCodegen::setKeyedAggHavingEvaluatedOnGPU() {
    resultSchema_.keyedAgg.havingEvaluatedOnGPU = true;
}

const MetalResultSchema& MetalCodegen::getResultSchema() const {
    return resultSchema_;
}

MetalResultSchema& MetalCodegen::getResultSchemaMutable() {
    return resultSchema_;
}

void MetalCodegen::assignBufferIndices(PhaseInfo& phase) {
    // Buffer indices are phase-local and follow binding order.
    int nextIndex = 0;
    for (auto& b : phase.bindings) {
        b.bufferIndex = nextIndex++;
    }
}

std::string MetalCodegen::generateSignature(const PhaseInfo& phase) const {
    std::ostringstream sig;
    sig << "kernel void " << phase.name << "(\n";

    bool hasThreadParams = true;

    for (size_t i = 0; i < phase.bindings.size(); i++) {
        const auto& b = phase.bindings[i];
        sig << "    " << b.metalTypeDecl << " " << b.name
            << " [[buffer(" << b.bufferIndex << ")]]";
        if (i + 1 < phase.bindings.size() || hasThreadParams)
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
        sig << "    uint tid [[thread_position_in_grid]]\n";
    }

    sig << ")";
    return sig.str();
}

std::string MetalCodegen::print() {
    std::ostringstream out;

    out << commonHeader();

    if (!helperCode_.empty()) {
        out << "// --- Helper functions ---\n";
        out << helperCode_ << "\n";
    }

    for (size_t pi = 0; pi < phases_.size(); pi++) {
        assignBufferIndices(phases_[pi]);

        const auto& phase = phases_[pi];
        out << "\n// --- Phase " << pi << ": " << phase.name << " ---\n";
        out << generateSignature(phase) << " {\n";
        out << phase.code;
        out << "}\n";
    }

    return out.str();
}

const std::vector<MetalCodegen::PhaseInfo>& MetalCodegen::getPhases() const {
    return phases_;
}

std::vector<MetalCodegen::PhaseInfo>& MetalCodegen::getPhasesMutable() {
    return phases_;
}

std::vector<MetalParamBinding> MetalCodegen::getAllBindings() const {
    std::vector<MetalParamBinding> all;
    std::unordered_set<std::string> seen;
    // Allocation planning only needs each named binding once.
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
