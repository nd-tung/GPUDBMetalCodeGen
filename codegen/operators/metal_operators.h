#pragma once
// ===================================================================
// Metal Operators — Composable producer-consumer code generation
// ===================================================================
//
// Each operator has a produce() method that emits Metal shader code
// into a MetalCodegen instance. Operators form trees where each
// calls its child's produce() and wraps the consumer callback.
//
// Emits Metal Shading Language with Apple GPU-specific optimizations
// (SIMD group reductions, Metal atomics, [[buffer(N)]] attributes).
// ===================================================================

#include "metal_codegen_base.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace codegen {

// ===================================================================
// BASE CLASSES
// ===================================================================

class MetalOperator {
public:
    virtual ~MetalOperator() = default;
    virtual void produce(MetalCodegen& cg, ConsumerFn consume) = 0;
    virtual std::string describe() const = 0;
};

class MetalUnaryOperator : public MetalOperator {
protected:
    std::unique_ptr<MetalOperator> child_;
public:
    explicit MetalUnaryOperator(std::unique_ptr<MetalOperator> child)
        : child_(std::move(child)) {}
};

// ===================================================================
// LEAF OPERATORS — Table Scans
// ===================================================================

// Grid-stride loop over a table. Supports columnar layout.
// Emits:
//   for (uint {idxVar} = tid; {idxVar} < n_{table}; {idxVar} += tpg) {
//       <consume()>
//   }
class MetalGridStrideScan : public MetalOperator {
public:
    // Column descriptor for columnar scan
    struct ColumnDesc {
        std::string paramName;   // buffer parameter name (e.g. "l_shipdate")
        std::string metalType;   // Metal type (e.g. "int", "float", "char")
    };

    MetalGridStrideScan(const std::string& table,
                        const std::string& rowVar = "row",
                        const std::string& idxVar = "i");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

    // Add a column for columnar scan
    void addColumn(const std::string& paramName, const std::string& metalType);

    const std::string& tableName() const { return tableName_; }
    const std::string& rowVar() const { return rowVar_; }
    const std::string& idxVar() const { return idxVar_; }

private:
    std::string tableName_;
    std::string rowVar_;
    std::string idxVar_;
    std::vector<ColumnDesc> columns_;
};

// Grid-stride loop over a synthetic [0, n_<rangeName>) index range.
// Useful for scanning sparse direct-address arrays keyed by a maxKey
// (e.g. d_order_revenue[orderkey]) without an actual table to drive
// the scan. The caller must register `n_<rangeName>` with the executor
// as BOTH a sizeResolver symbol (for dispatch sizing) and a scalar int
// (for the kernel's loop bound) before this phase runs.
//
// Emits:
//   for (uint {idxVar} = tid; {idxVar} < n_{rangeName}; {idxVar} += tpg) {
//       <consume()>
//   }
class MetalRangeScan : public MetalOperator {
public:
    MetalRangeScan(const std::string& rangeName,
                   const std::string& idxVar = "i");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    const std::string& rangeName() const { return rangeName_; }
    const std::string& idxVar() const { return idxVar_; }
    // Side-load one or more columns of a real GPU table without driving
    // the scan from that table. The columns become read-only kernel
    // params (TableData kind) the consumer can index by any expression.
    void addSideColumn(const std::string& tableName,
                       const std::string& paramName,
                       const std::string& metalType);
private:
    struct SideColumn { std::string table, param, type; };
    std::string rangeName_;
    std::string idxVar_;
    std::vector<SideColumn> sideColumns_;
};

// ===================================================================
// UNARY OPERATORS — Pipeline operators
// ===================================================================

// Selection (WHERE filter): if (predicate) { consume(); }
class MetalSelection : public MetalUnaryOperator {
public:
    MetalSelection(std::unique_ptr<MetalOperator> child,
                   const std::string& predicate);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string predicate_;
};

// Compute expression: type var = expr; consume();
class MetalComputeExpr : public MetalUnaryOperator {
public:
    MetalComputeExpr(std::unique_ptr<MetalOperator> child,
                     const std::string& varName,
                     const std::string& varType,
                     const std::string& expression);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string varName_;
    std::string varType_;
    std::string expression_;
};

// ===================================================================
// BITMAP OPERATORS
// ===================================================================

// Build a bitmap: atomic_fetch_or to set bit for key
class MetalBitmapBuild : public MetalUnaryOperator {
public:
    MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                     const std::string& bitmapName,
                     const std::string& keyExpr,
                     const std::string& sizeExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
    std::string sizeExpr_;
};

// Probe a bitmap: if (bitmap_test(bitmap, key)) { consume(); }
class MetalBitmapProbe : public MetalUnaryOperator {
public:
    MetalBitmapProbe(std::unique_ptr<MetalOperator> child,
                     const std::string& bitmapName,
                     const std::string& keyExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
};

// Anti-probe: if (!bitmap_test(bitmap, key)) { consume(); }
class MetalAntiBitmapProbe : public MetalUnaryOperator {
public:
    MetalAntiBitmapProbe(std::unique_ptr<MetalOperator> child,
                         const std::string& bitmapName,
                         const std::string& keyExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
};

// ===================================================================
// DIRECT-ADDRESS MAP OPERATORS
// ===================================================================

// Store: map[key] = value
class MetalArrayStore : public MetalUnaryOperator {
public:
    MetalArrayStore(std::unique_ptr<MetalOperator> child,
                    const std::string& arrayName,
                    const std::string& keyExpr,
                    const std::string& valueExpr,
                    const std::string& valueType = "int",
                    const std::string& sizeExpr = "",
                    int fillByte = 0xFF);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string valueExpr_;
    std::string valueType_;
    std::string sizeExpr_;
    int fillByte_;
};

// Lookup: type var = map[key]; (with optional guard for sentinel value)
class MetalArrayLookup : public MetalUnaryOperator {
public:
    MetalArrayLookup(std::unique_ptr<MetalOperator> child,
                     const std::string& arrayName,
                     const std::string& keyExpr,
                     const std::string& resultVar,
                     const std::string& resultType = "int",
                     int sentinel = -1);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string resultVar_;
    std::string resultType_;
    int sentinel_;
};

// ===================================================================
// HASH-MAP OPERATORS  (composite-key, linear-probing)
// ===================================================================
//
// Layout (registered via cg.addHashMapParam / addHashMapReadParam):
//   <map>_keys1 (atomic_uint, sentinel = 0xFFFFFFFFu)
//   <map>_keys2 (atomic_uint, sentinel = 0xFFFFFFFFu)
//   <map>_values (atomic_uint -- holds either an int payload or the
//                 bit-pattern of a float, interpreted by the lookup
//                 result type)
//   n_<map>     (uint capacity, MUST be a power of two)
//
// Use HashMapBuild + HashMapLookup for HashJoin (composite-key equi-
// join carrying one value).  Use HashMapAgg + HashMapLookup for
// HashGroupJoin (composite-key build aggregates a value per key).
// ===================================================================

// HashMapBuild: insert (key1, key2) -> value at hashed slot.
// First-writer-wins on duplicate composite keys.
class MetalHashMapBuild : public MetalUnaryOperator {
public:
    MetalHashMapBuild(std::unique_ptr<MetalOperator> child,
                      const std::string& mapName,
                      const std::string& key1Expr,
                      const std::string& key2Expr,
                      const std::string& valueExpr,
                      const std::string& capacityExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string mapName_;
    std::string key1Expr_, key2Expr_;
    std::string valueExpr_;
    std::string capacityExpr_;
};

// HashMapAgg: insert + atomic-add aggregation on `value`.  When
// valueIsFloat is true the value buffer is read as float bits.
class MetalHashMapAgg : public MetalUnaryOperator {
public:
    MetalHashMapAgg(std::unique_ptr<MetalOperator> child,
                    const std::string& mapName,
                    const std::string& key1Expr,
                    const std::string& key2Expr,
                    const std::string& valueExpr,
                    const std::string& capacityExpr,
                    bool valueIsFloat = false);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string mapName_;
    std::string key1Expr_, key2Expr_;
    std::string valueExpr_;
    std::string capacityExpr_;
    bool valueIsFloat_;
};

// HashMapLookup: probe (key1, key2); on hit emits
//   <resultType> resultVar = <values>[slot];
// and gates the consumer chain on the lookup hit.
class MetalHashMapLookup : public MetalUnaryOperator {
public:
    MetalHashMapLookup(std::unique_ptr<MetalOperator> child,
                       const std::string& mapName,
                       const std::string& key1Expr,
                       const std::string& key2Expr,
                       const std::string& capacityExpr,
                       const std::string& resultVar,
                       const std::string& resultType = "uint");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string mapName_;
    std::string key1Expr_, key2Expr_;
    std::string capacityExpr_;
    std::string resultVar_;
    std::string resultType_;
};

// ===================================================================================
// AGGREGATION OPERATORS
// ===================================================================

// Threadgroup reduce using SIMD group reductions.
// Emits per-thread local accumulation + tg_reduce_float/long + atomic to global.
class MetalTGReduce : public MetalUnaryOperator {
public:
    enum class ReduceOp { SUM, MIN, MAX };

    struct Accumulator {
        std::string name;        // local variable name
        std::string loBuffer;    // output buffer name (lo part or direct)
        std::string hiBuffer;    // output buffer name (hi part, empty if float)
        std::string stateBuffer; // min/max initialization guard
        std::string valueExpr;   // expression to accumulate
        std::string type;        // "float" or "long"
        ReduceOp op = ReduceOp::SUM;
        int binIndex = 0;        // for multi-bin, which index in the output
    };

    MetalTGReduce(std::unique_ptr<MetalOperator> child,
                  const std::string& outputPrefix);
    int addAccumulator(const std::string& name, const std::string& valueExpr,
                       const std::string& type = "float",
                       const std::string& loBuffer = "",
                       const std::string& hiBuffer = "",
                       ReduceOp op = ReduceOp::SUM);

    // Register result schema for this reduce's output
    // scaleDown: divisor for fixed-point (e.g. 100 means stored as val*100)
    void setResultAlias(const std::string& displayName, int scaleDown = 0);
    void setAccumulatorResultAlias(const std::string& displayName,
                                   int accumulatorIndex,
                                   int scaleDown = 0);
    void setAverageResultAlias(const std::string& displayName,
                               int numeratorIndex,
                               int denominatorIndex,
                               int scaleDown = 0);

    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string outputPrefix_;
    std::vector<Accumulator> accumulators_;

    // Result schema info
    struct ResultInfo {
        std::string displayName;
        int scaleDown = 0;
        int accumulatorIndex = -1;
        int denominatorIndex = -1;
    };
    std::vector<ResultInfo> resultInfos_;
};

// Keyed aggregation using atomics.
// For "add" with isLongPair=true, uses atomic_add_long_pair for 64-bit correctness.
// offset/offset+1 form the lo/hi pair in the output buffer.
// Descriptor for one group-by key in multi-key encoding.
struct GroupKeyDecode {
    std::string name;   // display name (e.g. "l_returnflag")
    int numValues = 0;  // distinct values this key can take
    int stride = 0;     // multiplier for this key in flat bucket
    // CHAR1 decode map: for each flat value 0..numValues-1, what CHAR1 does it map to.
    // Empty for integer keys (recovered via keyBase + value).
    std::vector<char> charMap;
    int keyBase = 0;    // additive constant for integer keys after extraction
};

class MetalKeyedAgg : public MetalUnaryOperator {
public:
    struct Aggregate {
        std::string name;
        int offset;
        std::string valueExpr;
        std::string atomicOp;   // "add", "min", "max"
        bool isLongPair = false; // true → uses lo/hi atomic_uint pair at offset/offset+1
        // scaleDown is applied during result COLLECTION only (post-process),
        // see MetalResultCollector::collectKeyedAgg. The GPU kernel always
        // accumulates the raw fixed-point value; this divisor is for display.
        int scaleDown = 0;      // result divisor (e.g. 100 for cents→dollars, 0=none)
        bool isFloatSum = false; // true → float value stored via atomic CAS in single uint slot
        bool isMinMax = false;   // true → min/max aggregate using special init/update logic
        // Agg function metadata for HAVING predicate matching.
        // funcName is uppercase ("SUM","COUNT","AVG","MIN","MAX").
        // innerColumn is the referenced column for SUM/AVG/MIN/MAX (empty for COUNT(*) star).
        std::string funcName;    // aggregate function name for HAVING matching
        std::string innerColumn; // column referenced by the aggregate (empty for COUNT(*))
    };

    MetalKeyedAgg(std::unique_ptr<MetalOperator> child,
                  const std::string& outputArrayName,
                  const std::string& bucketExpr,
                  int numBuckets,
                  int valuesPerBucket,
                  const std::string& sizeExpr = "");
    void addAggregate(const std::string& name, int offset,
                      const std::string& valueExpr,
                      const std::string& atomicOp = "add",
                      bool isLongPair = false,
                      int scaleDown = 0);
    // Extended version that also sets isFloatSum / isMinMax flags.
    void addAggregateWithMeta(const std::string& name, int offset,
                               const std::string& valueExpr,
                               const std::string& atomicOp,
                               bool isLongPair,
                               int scaleDown,
                               bool isFloatSum,
                               bool isMinMax,
                               const std::string& funcName = "",
                               const std::string& innerColumn = "");
    // Register a COUNT(DISTINCT col) bitmap: one bit per distinct value.
    // `valueExpr` is the column expression, `maxValueExpr` is the max possible
    // value (e.g., "maxSuppkey"), used to compute bitmapStride.
    void addDistinctBitmap(const std::string& outputName,
                           const std::string& valueExpr,
                           const std::string& maxValueExpr);
    void setKeyResult(const std::string& displayName, int base = 0);
    // Multi-key result info: caller provides list of GroupKeyDecode descriptors
    // (one per group-by key) so the result collector can reconstruct each
    // key column from the flat bucket index.
    void setMultiKeyResult(const std::vector<std::string>& displayNames,
                           const std::vector<GroupKeyDecode>& keys,
                           int totalBuckets);
    void setHaving(const PredPtr& havingPredicate) { havingPredicate_ = havingPredicate; }
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string outputArrayName_;
    std::string bucketExpr_;
    int numBuckets_;
    int valuesPerBucket_;
    std::string sizeExpr_;
    std::string keyDisplayName_;
    int keyBase_ = 0;
    std::vector<Aggregate> aggregates_;
    std::vector<GroupKeyDecode> multiKeyDecode_;
    PredPtr havingPredicate_;

    struct DistinctBitmap {
        std::string outputName;     // buffer name for popcount output
        std::string bitmapName;     // buffer name for atomicOr bitmap
        std::string valueExpr;      // column expression (e.g., "s_suppkey[i]")
        std::string maxValueExpr;   // max value symbol (e.g., "maxSuppkey")
    };
    std::vector<DistinctBitmap> distinctBitmaps_;  // Optional HAVING filter
};

// Simple atomic add to array: atomic_fetch_add(&arr[bucket], value)
class MetalAtomicAgg : public MetalUnaryOperator {
public:
    MetalAtomicAgg(std::unique_ptr<MetalOperator> child,
                   const std::string& arrayName,
                   const std::string& bucketExpr,
                   const std::string& valueExpr,
                   const std::string& sizeExpr = "",
                   const std::string& atomicType = "atomic_uint",
                   const std::string& castType = "uint");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string bucketExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
    std::string atomicType_;
    std::string castType_;
};

// Atomic count: atomic_fetch_add(&arr[bucket], 1)
class MetalAtomicCount : public MetalUnaryOperator {
public:
    MetalAtomicCount(std::unique_ptr<MetalOperator> child,
                     const std::string& arrayName,
                     const std::string& bucketExpr,
                     const std::string& sizeExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string bucketExpr_;
    std::string sizeExpr_;
};

// ===================================================================
// MATERIALIZATION
// ===================================================================

// Materialize rows to output arrays via atomic counter
class MetalMaterialize : public MetalUnaryOperator {
public:
    struct Column {
        std::string arrayName;
        std::string type;
        std::string valueExpr;
        std::string displayName;
        std::string sizeExpr;
        int stringLen = 0;
    };

    MetalMaterialize(std::unique_ptr<MetalOperator> child,
                     const std::string& counterName,
                     const std::string& counterSizeExpr = "1");
    void addColumn(const std::string& arrayName, const std::string& type,
                   const std::string& valueExpr, const std::string& displayName = "",
                   const std::string& sizeExpr = "", int stringLen = 0);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string counterName_;
    std::string counterSizeExpr_;
    std::vector<Column> columns_;
};

// COUNT(DISTINCT) popcount kernel: counts set bits per group in a bitmap.
// Emits a grid-stride loop over numGroups, with popcount(bitmap[g * stride + w]).
class MetalBitmapPopcount : public MetalOperator {
public:
    MetalBitmapPopcount(const std::string& bitmapName,
                        const std::string& outputName,
                        const std::string& numGroupsExpr,
                        const std::string& bitmapStrideExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string outputName_;
    std::string numGroupsExpr_;
    std::string bitmapStrideExpr_;
};

// ===================================================================
// GPU Bitonic Sort operators
// ===================================================================
//
// MetalInitSortKeys: encodes a source column into sort keys (uint64_t)
// and row indices.  Grid-strides over the source data.  Supports
// ascending/descending direction via key encoding.
//
// MetalBitonicSortStep: a single comparison-swap step of the bitonic
// sort network.  The kernel takes (k, j) constants and swaps elements
// if needed.  The plan attaches a PostDispatchHook that re-dispaches
// this kernel in the classic (k, j) nested loop.

class MetalInitSortKeys : public MetalOperator {
public:
    // sourceColumn: name of the source buffer to sort (e.g. "d_adhoc_0_expr")
    // sourceType: "int" or "float" (or "double" for Metal float)
    // sortKeyBuf: name of output uint64_t sort key buffer
    // sortIdxBuf: name of output int row index buffer
    // nResultsExpr: expression for the number of source rows
    // descending: if true, invert sort keys so ascending sort → descending order
    MetalInitSortKeys(const std::string& sourceColumn, const std::string& sourceType,
                      const std::string& sortKeyBuf, const std::string& sortIdxBuf,
                      const std::string& nResultsExpr, bool descending);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

    // Exposed for the sort hook: the padded element count (next power of 2)
    static unsigned int nextPow2(unsigned int n);
    // Build the PostDispatchHook that runs the (k, j) bitonic loop after the
    // sort-step phase.  The caller attaches this to the sort-step phase.
    static PostDispatchHook makeBitonicHook(
        const std::string& sortPhaseName,
        const std::string& sortKeyBufName,
        const std::string& sortIdxBufName,
        const std::string& nResultsExpr);

private:
    std::string sourceColumn_;
    std::string sourceType_;
    std::string sortKeyBuf_;
    std::string sortIdxBuf_;
    std::string nResultsExpr_;
    bool descending_;
};

class MetalBitonicSortStep : public MetalOperator {
public:
    // sortKeyBuf / sortIdxBuf: buffer names matching those in MetalInitSortKeys
    // nResultsExpr: same expression used for init — used for grid sizing
    MetalBitonicSortStep(const std::string& sortKeyBuf, const std::string& sortIdxBuf,
                         const std::string& nResultsExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string sortKeyBuf_;
    std::string sortIdxBuf_;
    std::string nResultsExpr_;
};

} // namespace codegen
