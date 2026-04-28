#pragma once
// ===================================================================
// MetalCodegen — Base class for composable Metal shader generation
// ===================================================================
//
// Manages code emission, indentation, multi-phase kernel generation,
// parameter binding with auto-numbered [[buffer(N)]] attributes,
// and result schema registration.
//
// Operators call methods on this class during produce() to emit
// Metal shader code.
// ===================================================================

#include "metal_param_binding.h"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace codegen {

using ConsumerFn = std::function<void()>;

class MetalCodegen;

class MetalCodegen {
public:
    virtual ~MetalCodegen() = default;

    // ---------------------------------------------------------------
    // Code emission
    // ---------------------------------------------------------------
    void addLine(const std::string& line);
    void addBlock(const std::string& header, std::function<void()> body,
                  const std::string& trailing = "");
    void addIf(const std::string& condition, std::function<void()> body);
    void addComment(const std::string& comment);
    void increaseIndent();
    void decreaseIndent();
    void addRawCode(const std::string& code);  // no indentation added

    // ---------------------------------------------------------------
    // Helper / device functions (emitted once before all kernels)
    // ---------------------------------------------------------------
    void addHelper(const std::string& code);

    // ---------------------------------------------------------------
    // Phase management — each phase = one Metal kernel function
    // ---------------------------------------------------------------
    void beginPhase(const std::string& phaseName);
    void endPhase();
    int phaseCount() const;

    // Phase metadata (for dispatch configuration)
    void setPhaseScannedTable(const std::string& tableName);
    void setPhaseThreadgroupSize(int size);
    void setPhaseSingleThread(bool single);
    void setPhaseMaxThreadgroups(int max);

    // ---------------------------------------------------------------
    // Parameter registration (creates MetalParamBinding entries)
    // ---------------------------------------------------------------
    // Table: device const T* + constant uint& size (AoS struct)
    void addTableParam(const std::string& table, const std::string& metalType);

    // Column: device const T* (columnar layout, one buffer per column)
    void addColumnParam(const std::string& paramName, const std::string& metalType,
                        const std::string& tableName = "");

    // Table size only: constant uint& n_<table>
    void addTableSizeParam(const std::string& table);

    // Device buffer: device T* (read-write)
    void addBufferParam(const std::string& name, const std::string& elemType,
                        const std::string& sizeExpr, bool zeroInit = true, int fillByte = 0);

    // Atomic device buffer: device atomic_uint* (read-write, always zero-init)
    void addAtomicBufferParam(const std::string& name, const std::string& atomicType,
                              const std::string& sizeExpr);

    // Constant scalar: constant T& (passed via setBytes)
    void addScalarParam(const std::string& name, const std::string& type);

    // Constant data: constant T* (host data, passed via setBytes)
    void addConstantDataParam(const std::string& name, const std::string& type,
                              size_t bytes);

    // Bitmap shorthand: device const uint* (read-only) / device atomic_uint* (write)
    void addBitmapReadParam(const std::string& name, const std::string& sizeExpr);
    void addBitmapWriteParam(const std::string& name, const std::string& sizeExpr);

    // Hash map shorthand: atomic keys + lo/hi value buffers + size scalar
    // Registers 3 atomic buffers (keys, valuesLo, valuesHi) + 1 scalar (n_mapName)
    void addHashMapParam(const std::string& mapName,
                         const std::string& keysName,
                         const std::string& valuesLoName,
                         const std::string& valuesHiName,
                         const std::string& sizeExpr);
    // Read-only hash map (for lookup phases, already allocated by build)
    void addHashMapReadParam(const std::string& mapName,
                             const std::string& keysName,
                             const std::string& valuesLoName,
                             const std::string& valuesHiName);

    // ---------------------------------------------------------------
    // Global buffer size registration
    // ---------------------------------------------------------------
    void setBufferSize(const std::string& name, const std::string& sizeExpr);
    const std::unordered_map<std::string, std::string>& getGlobalBufferSizes() const;

    // ---------------------------------------------------------------
    // Output schema registration (called by terminal operators)
    // ---------------------------------------------------------------
    void registerScalarAggOutput(const std::string& loBuffer, const std::string& hiBuffer,
                                 const std::string& type);
    void registerScalarAggColumn(const std::string& displayName, int index, int scaleDown = 0);
    void registerMaterializeOutput(const std::string& counterBuffer);
    void registerOutputColumn(const std::string& displayName, const std::string& bufferName,
                              const std::string& elementType, int stringLen = 0);
    void registerKeyedAggOutput(const std::string& bufferName, int numBuckets, int valuesPerBucket,
                                const std::vector<MetalResultSchema::KeyedAggSlot>& slots = {});
    const MetalResultSchema& getResultSchema() const;

    // ---------------------------------------------------------------
    // Code generation — produce final Metal source
    // ---------------------------------------------------------------
    std::string print();

    // Access phase info (for executor dispatch configuration)
    struct PhaseInfo {
        std::string name;
        std::string code;
        std::string scannedTable;
        int threadgroupSize = 1024;
        int maxThreadgroups = 0;
        bool isSingleThread = false;
        std::vector<MetalParamBinding> bindings;
    };
    const std::vector<PhaseInfo>& getPhases() const;

    // Mutable phase access (used by --autotune-tg to change TG size between
    // dispatches without recompiling the kernels).
    std::vector<PhaseInfo>& getPhasesMutable();

    // Access all bindings across all phases (for buffer allocation)
    std::vector<MetalParamBinding> getAllBindings() const;

private:
    // Phases
    std::vector<PhaseInfo> phases_;
    PhaseInfo* currentPhase_ = nullptr;

    // Helper / device functions
    std::string helperCode_;

    // Tables already registered (avoid double-registration)
    std::unordered_set<std::string> registeredTables_;

    // Global buffer sizes
    std::unordered_map<std::string, std::string> globalBufferSizes_;

    // Result schema
    MetalResultSchema resultSchema_;
    std::string scalarAggPendingLo_;
    std::string scalarAggPendingHi_;

    // Current indentation
    unsigned indentLevel_ = 1;
    static constexpr unsigned INDENT_SIZE = 4;

    // Helpers
    std::string indent() const;
    std::string generateSignature(const PhaseInfo& phase) const;
    static std::string commonHeader();
    void assignBufferIndices(PhaseInfo& phase);
};

// RAII guard for indentation scope — automatically decrements on destruction.
// Ensures indent level is always restored even if exceptions are thrown.
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
