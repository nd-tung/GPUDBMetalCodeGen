#pragma once
// Metadata bridge from logical query parameters to Metal buffers.

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "query_plan.h"

namespace codegen {

// Host and Metal hash probes must use the same multiplier.
inline constexpr uint32_t kKnuthHashMul = 2654435769u;

enum class MetalParamKind {
    TableData,
    TableSize,
    DeviceBuffer,
    ConstantScalar,
    ConstantData,
};

// Shared names used by codegen and executor.
inline std::string tableDataName(const std::string& table) { return "d_" + table; }
inline std::string tableSizeName(const std::string& table) { return "n_" + table; }
inline std::string tableColumnDataName(const std::string& table,
                                       const std::string& column) {
    return table + "." + column;
}

struct MetalParamBinding {
    std::string name;
    std::string metalTypeDecl;
    MetalParamKind kind;
    std::string tableName;
    std::string elementType;
    // Empty sizeExpr means the buffer is registered elsewhere.
    std::string sizeExpr;
    // fillByte is used when zeroInit clears buffers; 0xFF marks -1 sentinels.
    bool zeroInit = false;
    int fillByte = 0;
    // readOnly buffers are views over shared storage and do not own sizing.
    bool readOnly = false;
    size_t hostCopyBytes = 0;
    int bufferIndex = -1;

    size_t elemSizeBytes() const {
        if (elementType == "long"  || elementType == "ulong"  || elementType == "double" ||
            elementType == "atomic_long" || elementType == "atomic_ulong") return 8;
        if (elementType == "char"  || elementType == "uchar")  return 1;
        if (elementType == "short" || elementType == "ushort") return 2;
        return 4;
    }
};

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

    // Supports literals, symbols, arithmetic, and next_pow2(...).
    size_t resolve(const std::string& expr) const;

private:
    std::unordered_map<std::string, size_t> symbols_;
};

struct MetalResultSchema {
    enum Kind { NONE, MATERIALIZE, SCALAR_AGG, KEYED_AGG };
    Kind kind = NONE;

    // MATERIALIZE columns are read row-wise from the listed output buffers.
    struct ColumnDesc {
        std::string displayName;
        std::string bufferName;
        std::string elementType;
        int stringLen = 0;
        bool isLongPair = false;
        int scaleDown = 0;
    };
    std::vector<ColumnDesc> columns;

    // Atomic counter buffer used by materialized outputs.
    std::string counterBuffer;

    // SCALAR_AGG entries are single-row outputs; lo/hi pairs reconstruct longs.
    struct ScalarAggEntry {
        std::string displayName;
        std::string loBuffer;
        std::string hiBuffer;
        std::string denomLoBuffer;
        std::string denomHiBuffer;
        std::string elementType;
        bool isLongPair = false;
        bool denomIsLongPair = false;
        bool divideByDenominator = false;
        int scaleDown = 0;
        ExprPtr projectionExpr;
    };
    std::vector<ScalarAggEntry> scalarAggs;

    // KEYED_AGG stores bucketed values in a compact fixed-slot layout.
    struct KeyedAggSlot {
        std::string name;
        int offset;
        bool isLongPair = false;
        int scaleDown = 0;
        bool isFloatSum = false;
        bool isMinMax = false;
        std::string atomicOp;
        int avgDenomOffset = -1;
    };
    struct KeyedAggInfo {
        int numBuckets = 0;
        int valuesPerBucket = 0;
        std::string bufferName;
        std::string keyDisplayName;
        int keyBase = 0;
        std::vector<KeyedAggSlot> slots;
        // Multi-key output decodes a packed bucket id into display columns.
        struct MultiKeyInfo {
            std::string displayName;
            int numValues = 0;
            int stride = 0;
            std::vector<char> charMap;
            int keyBase = 0;
        };
        std::vector<MultiKeyInfo> multiKeys;
        PredPtr havingPredicate;
        // True when HAVING was already evaluated on GPU.
        bool havingEvaluatedOnGPU = false;
    };
    KeyedAggInfo keyedAgg;
};

} // namespace codegen
