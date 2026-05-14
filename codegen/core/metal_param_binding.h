#pragma once
// ===================================================================
// Metal Param Binding — metadata-driven buffer management
// ===================================================================
//
// Metadata-driven bridge from logical query parameters to Metal buffers.
// Each binding describes one kernel parameter: its Metal type, memory
// management, symbolic size, and auto-assigned [[buffer(N)]] index.
// ===================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "query_analyzer.h"  // for PredPtr

namespace codegen {

// Knuth multiplicative hash constant (golden-ratio reciprocal × 2^32).
// Used by the open-addressing hash tables in query_preprocessing.cpp
// (host-side build) and by the matching Metal probes emitted from
// TPC-H query builders (`q9_ht_probe`, `q20_ht_add`). Host and device
// MUST use the same constant or probes will miss; if you change this
// value, update both sides.
inline constexpr uint32_t kKnuthHashMul = 2654435769u;

// What kind of parameter is it?
enum class MetalParamKind {
    TableData,        // device const T* — read-only table column / row array
    TableSize,        // constant uint&  — row count for a table
    DeviceBuffer,     // device T*       — read-write scratch / output buffer
    ConstantScalar,   // constant T&     — small value passed via setBytes()
    ConstantData,     // constant T*     — host data copied via setBytes()
};

// Naming convention for buffer-name strings produced by codegen and resolved by the executor:
//
//   tableDataName(t)  ==  "d_" + t   — device-side data pointer for table `t`
//   tableSizeName(t)  ==  "n_" + t   — row-count scalar (constant uint&) for table `t`
//
// Use these helpers instead of bare "d_"/"n_" string concatenations.
inline std::string tableDataName(const std::string& table) { return "d_" + table; }
inline std::string tableSizeName(const std::string& table) { return "n_" + table; }

struct MetalParamBinding {
    std::string name;            // e.g. "d_lineitem_shipdate", "data_size"
    std::string metalTypeDecl;   // full Metal parameter declaration without the name,
                                 // e.g. "device const int*", "constant uint&"
    MetalParamKind kind;
    std::string tableName;       // for TableData/TableSize: which table
    std::string elementType;     // e.g. "float", "uint", "atomic_uint"
    std::string sizeExpr;        // symbolic: "numLineItems", "maxCustkey + 1"
    bool zeroInit = false;       // zero-fill on allocation
    int fillByte = 0;            // byte value for memset (0 for zero, 0xFF for -1 sentinel)
    bool readOnly = false;       // hint for storage mode
    size_t hostCopyBytes = 0;    // for ConstantData: how many bytes
    int bufferIndex = -1;        // auto-assigned [[buffer(N)]]

    // Return the byte size of one element based on elementType.
    // Centralises type→size mapping that was previously scattered as string checks.
    size_t elemSizeBytes() const {
        if (elementType == "long"  || elementType == "ulong"  || elementType == "double" ||
            elementType == "atomic_long" || elementType == "atomic_ulong") return 8;
        if (elementType == "char"  || elementType == "uchar")  return 1;
        if (elementType == "short" || elementType == "ushort") return 2;
        return 4; // uint, int, float, atomic_uint, atomic_int, …
    }
};

// ===================================================================
// Size Resolver — symbolic name → actual byte count
// ===================================================================

class MetalSizeResolver {
public:
    void registerSymbol(const std::string& name, size_t value) {
        symbols_[name] = value;
    }

    bool hasSymbol(const std::string& name) const {
        return symbols_.count(name) > 0;
    }

    size_t getSymbol(const std::string& name) const {
        auto it = symbols_.find(name);
        if (it == symbols_.end())
            throw std::runtime_error("MetalSizeResolver: unknown symbol '" + name + "'");
        return it->second;
    }

    // Resolve a simple expression like "maxCustkey + 1" or "numLineItems"
    // Supports: constant, symbol, symbol + constant, symbol * constant
    size_t resolve(const std::string& expr) const;

private:
    std::unordered_map<std::string, size_t> symbols_;
};

// ===================================================================
// Result Schema — describes what a query produces
// ===================================================================

struct MetalResultSchema {
    enum Kind { NONE, MATERIALIZE, SCALAR_AGG, KEYED_AGG };
    Kind kind = NONE;

    struct ColumnDesc {
        std::string displayName;    // "l_returnflag"
        std::string bufferName;     // "d_returnflag"
        std::string elementType;    // "uint", "float", "char"
        int stringLen = 0;          // for fixed-width strings
        bool isLongPair = false;    // needs lo/hi reconstruction
    };
    std::vector<ColumnDesc> columns;

    // MATERIALIZE: output rows scattered to arrays
    std::string counterBuffer;      // "d_resultCount"

    // SCALAR_AGG: single-row output
    struct ScalarAggEntry {
        std::string displayName;
        std::string loBuffer;       // lo part (or main buffer if not long pair)
        std::string hiBuffer;       // hi part (empty if not long pair)
        std::string denomLoBuffer;  // optional denominator for AVG-style output
        std::string denomHiBuffer;
        std::string elementType;    // "uint" for long pair, "float" for direct
        bool isLongPair = false;
        bool denomIsLongPair = false;
        bool divideByDenominator = false;
        int scaleDown = 0;          // optional display divisor (e.g. 100 for cents→dollars)
        ExprPtr projectionExpr;     // optional SELECT expression over this aggregate
    };
    std::vector<ScalarAggEntry> scalarAggs;

    // KEYED_AGG: N-bucket output
    struct KeyedAggSlot {
        std::string name;           // display name
        int offset;                 // slot offset within bucket
        bool isLongPair = false;    // true → lo/hi at offset/offset+1
        int scaleDown = 0;          // divisor for fixed-point
        bool isFloatSum = false;    // true → float atomic buffer, read as float
        bool isMinMax = false;      // true → min/max aggregate (special init logic)
        std::string atomicOp;       // "add", "min", "max" (for result post-processing hint)
        int avgDenomOffset = -1;    // for AVG: offset of the count slot (≥0 → divide by count)
    };
    struct KeyedAggInfo {
        int numBuckets = 0;
        int valuesPerBucket = 0;
        std::string bufferName;
        std::string keyDisplayName;  // single-key display name (backwards compat)
        int keyBase = 0;             // single-key base (backwards compat)
        std::vector<KeyedAggSlot> slots; // describes each logical value
        // Multi-key decode info (when non-empty, overrides keyDisplayName/keyBase)
        struct MultiKeyInfo {
            std::string displayName;
            int numValues = 0;
            int stride = 0;
            std::vector<char> charMap;  // for CHAR1 keys (reverse map: valueIndex→char)
            int keyBase = 0;            // for integer keys (additive constant)
        };
        std::vector<MultiKeyInfo> multiKeys;
        // Optional HAVING predicate (applied during result collection)
        PredPtr havingPredicate;
        // When true, the HAVING predicate was already evaluated on GPU
        // (inline filter after threadgroup reduction). Skip CPU-side evaluation.
        bool havingEvaluatedOnGPU = false;
    };
    KeyedAggInfo keyedAgg;
};

} // namespace codegen
