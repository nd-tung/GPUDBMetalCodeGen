#pragma once
// Composable Metal shader generation.

#include "metal_param_binding.h"
#include "iu.hpp"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace codegen {

using ConsumerFn = std::function<void()>;

class MetalCodegen;
class MetalGenericExecutor;

// Runs after a phase dispatch; returns extra GPU milliseconds launched by hook.
using PostDispatchHook = std::function<double(MetalGenericExecutor&)>;

class MetalCodegen {
public:
    virtual ~MetalCodegen() = default;

    void addLine(const std::string& line);
    void addBlock(const std::string& header, std::function<void()> body,
                  const std::string& trailing = "");
    void addIf(const std::string& condition, std::function<void()> body);
    void addComment(const std::string& comment);
    void increaseIndent();
    void decreaseIndent();
    void addRawCode(const std::string& code);

    // Emitted once before all kernels.
    void addHelper(const std::string& code);

    // Each phase emits one Metal kernel.
    void beginPhase(const std::string& phaseName);
    void endPhase();
    int phaseCount() const;

    void setPhaseScannedTable(const std::string& tableName);
    void setPhaseThreadgroupSize(int size);
    void setPhaseSingleThread(bool single);
    void setPhaseMaxThreadgroups(int max);
    // Hooks run after the command buffer for the phase has completed.
    void setPhasePostDispatchHook(PostDispatchHook hook);

    // Table: device const T* plus constant uint& size.
    void addTableParam(const std::string& table, const std::string& metalType);

    // Column: device const T*.
    void addColumnParam(const std::string& paramName, const std::string& metalType,
                        const std::string& tableName = "");

    // Table size only: constant uint& n_<table>.
    void addTableSizeParam(const std::string& table);

    // Device buffer: device T*.
    void addBufferParam(const std::string& name, const std::string& elemType,
                        const std::string& sizeExpr, bool zeroInit = true, int fillByte = 0);

    // Atomic device buffer; fillByte supports sentinel initialization.
    void addAtomicBufferParam(const std::string& name, const std::string& atomicType,
                              const std::string& sizeExpr, int fillByte = 0);

    // Constant scalar: constant T& passed via setBytes.
    void addScalarParam(const std::string& name, const std::string& type);

    // Scalar derived from sizeExpr at dispatch time.
    void addResolvedScalarParam(const std::string& name, const std::string& type,
                                const std::string& sizeExpr);

    // Constant data: constant T* passed via setBytes.
    void addConstantDataParam(const std::string& name, const std::string& type,
                              size_t bytes);

    // Bitmap shorthand.
    void addBitmapReadParam(const std::string& name, const std::string& sizeExpr);
    void addBitmapWriteParam(const std::string& name, const std::string& sizeExpr);

    // Hash map shorthand: atomic keys, lo/hi values, and size scalar.
    void addHashMapParam(const std::string& mapName,
                         const std::string& keysName,
                         const std::string& valuesLoName,
                         const std::string& valuesHiName,
                         const std::string& sizeExpr);
    // Read-only hash map for lookup phases.
    void addHashMapReadParam(const std::string& mapName,
                             const std::string& keysName,
                             const std::string& valuesLoName,
                             const std::string& valuesHiName);

    void setBufferSize(const std::string& name, const std::string& sizeExpr);
    const std::unordered_map<std::string, std::string>& getGlobalBufferSizes() const;

    // Result schema is consumed by MetalResultCollector after execution.
    void registerScalarAggOutput(const std::string& loBuffer, const std::string& hiBuffer,
                                 const std::string& type);
    void registerScalarAggColumn(const std::string& displayName, int index, int scaleDown = 0,
                                 ExprPtr projectionExpr = nullptr);
    void registerScalarAggAverageColumn(const std::string& displayName,
                                        const std::string& numeratorLoBuffer,
                                        const std::string& numeratorHiBuffer,
                                        const std::string& denominatorLoBuffer,
                                        const std::string& denominatorHiBuffer,
                                        const std::string& type,
                                        int scaleDown = 0,
                                        ExprPtr projectionExpr = nullptr);
    void registerMaterializeOutput(const std::string& counterBuffer);
    void registerOutputColumn(const std::string& displayName, const std::string& bufferName,
                              const std::string& elementType, int stringLen = 0,
                              int scaleDown = 0, bool isLongPair = false);
    void registerKeyedAggOutput(const std::string& bufferName, int numBuckets, int valuesPerBucket,
                                const std::vector<MetalResultSchema::KeyedAggSlot>& slots = {},
                                const std::string& keyDisplayName = "",
                                int keyBase = 0);
    void setKeyedAggHaving(const PredPtr& havingPredicate);
    void setKeyedAggHavingEvaluatedOnGPU();
    const MetalResultSchema& getResultSchema() const;
    MetalResultSchema& getResultSchemaMutable();

    std::string print();

    struct PhaseInfo {
        std::string name;
        // Generated Metal kernel body without the signature.
        std::string code;
        // Empty scannedTable means dispatch size comes from explicit symbols.
        std::string scannedTable;
        int threadgroupSize = 1024;
        int maxThreadgroups = 0;
        bool isSingleThread = false;
        std::vector<MetalParamBinding> bindings;
        PostDispatchHook postDispatchHook;
    };
    const std::vector<PhaseInfo>& getPhases() const;

    // Injected by schema layer for IU auto-projection.
    void setColumnTypeResolver(ColumnTypeResolver r) {
        columnTypeResolver_ = std::move(r);
    }
    std::string resolveColumnType(const std::string& table,
                                  const std::string& col) const {
        return columnTypeResolver_ ? columnTypeResolver_(table, col)
                                   : std::string{};
    }

    // Allows --autotune-tg to change TG size without recompiling.
    std::vector<PhaseInfo>& getPhasesMutable();

    // Flattened binding list used by the executor for allocation planning.
    std::vector<MetalParamBinding> getAllBindings() const;

private:
    std::vector<PhaseInfo> phases_;
    PhaseInfo* currentPhase_ = nullptr;

    std::string helperCode_;

    std::unordered_set<std::string> registeredTables_;

    std::unordered_map<std::string, std::string> globalBufferSizes_;

    MetalResultSchema resultSchema_;
    std::string scalarAggPendingLo_;
    std::string scalarAggPendingHi_;
    std::string scalarAggPendingType_;

    unsigned indentLevel_ = 1;
    static constexpr unsigned INDENT_SIZE = 4;

    std::string indent() const;
    std::string generateSignature(const PhaseInfo& phase) const;
    static std::string commonHeader();

    // Central binding registration with active-phase and duplicate handling.
    void pushBinding(const char* op, MetalParamBinding b, bool dedup);
    void assignBufferIndices(PhaseInfo& phase);

    ColumnTypeResolver columnTypeResolver_;
};

// Restores indentation on scope exit.
class IndentGuard {
public:
    explicit IndentGuard(MetalCodegen& cg) : cg_(cg) { cg_.increaseIndent(); }
    ~IndentGuard() { cg_.decreaseIndent(); }
    IndentGuard(const IndentGuard&) = delete;
    IndentGuard& operator=(const IndentGuard&) = delete;
private:
    MetalCodegen& cg_;
};

} // namespace codegen
