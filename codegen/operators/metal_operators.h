#pragma once
#include "../core/metal_codegen_base.h"
#include "../core/iu.hpp"
#include "../../third_party/nlohmann/json.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

// --- Base Classes ---

class MetalOperator {
public:
    virtual ~MetalOperator() = default;
    virtual void produce(MetalCodegen& cg, ConsumerFn consume) = 0;
    virtual std::string describe() const = 0;

    // Parent chain for upward traversal — enables scan operators to
    // auto-deduce required columns from downstream operators (IU chain).
    void setParent(MetalOperator* p) { parent_ = p; }
    MetalOperator* parent() const { return parent_; }

    // Collect all column references (IUs) consumed by this operator.
    // Operators override this to report which columns they reference.
    virtual void iusUsed(std::vector<IU>& /*out*/) const {}

    // Extract every colName[idxVar] reference from a kernel-side
    // expression string (e.g. "l_shipdate[i]", "o_orderkey[j] + 1").
    // The IUs returned have tableName and metalType empty — the
    // consuming scan fills them in via ColumnTypeResolver.
    static void appendIUsFromExpr(const std::string& expr,
                                  std::vector<IU>& out);

    // Serialize this operator (and children) to JSON for plan visualization.
    // Returns a json object with at minimum "type" (the describe() string).
    virtual nlohmann::json toJSON() const;

protected:
    MetalOperator* parent_ = nullptr;
};

class MetalUnaryOperator : public MetalOperator {
protected:
    std::unique_ptr<MetalOperator> child_;
public:
    explicit MetalUnaryOperator(std::unique_ptr<MetalOperator> child)
        : child_(std::move(child)) {
        if (child_) {
            // Preserve parent-walk behavior through wrapper operators.
            this->parent_ = child_->parent();
            child_->setParent(this);
        }
    }
    const MetalOperator* child() const { return child_.get(); }
    nlohmann::json toJSON() const override;
};

// --- Leaf Operators ---

// Grid-stride table scan with columnar buffer binding.
class MetalGridStrideScan : public MetalOperator {
public:
    // Column descriptor for columnar scan.
    struct ColumnDesc {
        std::string paramName;   // buffer parameter name (e.g. "l_shipdate")
        std::string metalType;   // Metal type (e.g. "int", "float", "char")
    };

    MetalGridStrideScan(const std::string& table,
                        const std::string& rowVar = "row",
                        const std::string& idxVar = "i");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

    // Auto-discover required columns by walking the parent chain and
    // collecting IUs from downstream operators. Returns a vector of
    // ColumnDesc with resolved metalType (caller must have set a
    // ColumnTypeResolver on the codegen).
    std::vector<ColumnDesc> deduceRequiredColumns(MetalCodegen& cg) const;

    // Add an explicit column read to the scan.
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

// Grid-stride scan over synthetic [0, n_<rangeName>) ranges.
// The caller must register n_<rangeName> as both dispatch symbol and scalar.
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

// --- Unary Operators ---

// Selection (WHERE filter): if (predicate) { consume(); }
class MetalSelection : public MetalUnaryOperator {
public:
    MetalSelection(std::unique_ptr<MetalOperator> child,
                   const std::string& predicate);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(predicate_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(expression_, out);
    }

private:
    std::string varName_;
    std::string varType_;
    std::string expression_;
};

// --- Bitmap Operators ---

// Build a bitmap by atomically setting the bit for each key.
class MetalBitmapBuild : public MetalUnaryOperator {
public:
    MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                     const std::string& bitmapName,
                     const std::string& keyExpr,
                     const std::string& sizeExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string bitmapName_;
    std::string keyExpr_;
};

// Left-outer probe: always consumes, but zeroes-out right-side
// variables when the bitmap test fails. Preserves left-table rows
// that have no match.
class MetalLeftOuterProbe : public MetalUnaryOperator {
public:
    struct DefaultVar {
        std::string varName;
        std::string varType;   // "int", "float", etc.
        std::string defaultVal; // "0", "0.0f", etc.
    };
    MetalLeftOuterProbe(std::unique_ptr<MetalOperator> child,
                        const std::string& bitmapName,
                        const std::string& keyExpr,
                        std::vector<DefaultVar> rightVars);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string bitmapName_;
    std::string keyExpr_;
    std::vector<DefaultVar> rightVars_;
};

// --- Direct-Address Maps ---

// Store value at direct-address map[key].
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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string valueExpr_;
    std::string valueType_;
    std::string sizeExpr_;
    int fillByte_;
};

// Lookup map[key] and optionally guard against a sentinel value.
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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string resultVar_;
    std::string resultType_;
    int sentinel_;
};

// Store fixed-width slices at map[key * width + byte].
class MetalArraySliceStore : public MetalUnaryOperator {
public:
    MetalArraySliceStore(std::unique_ptr<MetalOperator> child,
                         const std::string& arrayName,
                         const std::string& keyExpr,
                         const std::string& valuePtrExpr,
                         int sliceLen,
                         const std::string& valueType = "char",
                         const std::string& sizeExpr = "",
                         int fillByte = 0,
                         std::string sourceColumn = {},
                         std::string sourceIdxVar = {});
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valuePtrExpr_, out);
        if (!sourceColumn_.empty() && !sourceIdxVar_.empty())
            out.emplace_back(sourceColumn_, sourceIdxVar_);
    }

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string valuePtrExpr_;
    int sliceLen_ = 0;
    std::string valueType_;
    std::string sizeExpr_;
    int fillByte_;
    std::string sourceColumn_;
    std::string sourceIdxVar_;
};

// Lookup a fixed-width slice pointer at map + key * width.
class MetalArraySliceLookup : public MetalUnaryOperator {
public:
    MetalArraySliceLookup(std::unique_ptr<MetalOperator> child,
                          const std::string& arrayName,
                          const std::string& keyExpr,
                          const std::string& resultVar,
                          int sliceLen,
                          const std::string& resultType = "char");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string resultVar_;
    int sliceLen_ = 0;
    std::string resultType_;
};

// --- Hash Maps ---
// Layout (registered via cg.addHashMapParam / addHashMapReadParam):
//   <map>_keys1 (atomic_uint, sentinel = 0xFFFFFFFFu)
//   <map>_keys2 (atomic_uint, sentinel = 0xFFFFFFFFu)
//   <map>_values (atomic_uint -- holds either an int payload or the
//                 bit-pattern of a float, interpreted by the lookup
//                 result type)
//   n_<map>     (uint capacity, MUST be a power of two)
//
// Build operators write the table; lookup operators bind the same storage read-only.

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key1Expr_, out);
        appendIUsFromExpr(key2Expr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key1Expr_, out);
        appendIUsFromExpr(key2Expr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key1Expr_, out);
        appendIUsFromExpr(key2Expr_, out);
    }

private:
    std::string mapName_;
    std::string key1Expr_, key2Expr_;
    std::string capacityExpr_;
    std::string resultVar_;
    std::string resultType_;
};

// --- Aggregation Operators ---

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

    // Register the result schema for this reduce output.
    // scaleDown is the fixed-point display divisor.
    void setResultAlias(const std::string& displayName, int scaleDown = 0);
    void setAccumulatorResultAlias(const std::string& displayName,
                                   int accumulatorIndex,
                                   int scaleDown = 0,
                                   ExprPtr projectionExpr = nullptr);
    void setAverageResultAlias(const std::string& displayName,
                                int numeratorIndex,
                                int denominatorIndex,
                                int scaleDown = 0,
                                ExprPtr projectionExpr = nullptr);

    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        for (const auto& acc : accumulators_) {
            appendIUsFromExpr(acc.valueExpr, out);
        }
    }

private:
    std::string outputPrefix_;
    std::vector<Accumulator> accumulators_;

    // Result schema info
    struct ResultInfo {
        std::string displayName;
        int scaleDown = 0;
        int accumulatorIndex = -1;
        int denominatorIndex = -1;
        ExprPtr projectionExpr;
    };
    std::vector<ResultInfo> resultInfos_;
};

// Descriptor for one group-by key in multi-key bucket decoding.
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
        // scaleDown is applied by the result collector; GPU buffers store raw values.
        int scaleDown = 0;      // result divisor (e.g. 100 for cents→dollars, 0=none)
        bool isFloatSum = false; // true → float value stored via atomic CAS in single uint slot
        bool isMinMax = false;   // true → min/max aggregate using special init/update logic
        // Aggregate metadata lets HAVING match function calls to slots.
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
    // COUNT(DISTINCT) bitmap stores one bit per distinct value.
    void addDistinctBitmap(const std::string& outputName,
                           const std::string& valueExpr,
                           const std::string& maxValueExpr);
    void setKeyResult(const std::string& displayName, int base = 0);
    // Multi-key decode reconstructs display columns from the flat bucket index.
    void setMultiKeyResult(const std::vector<std::string>& displayNames,
                           const std::vector<GroupKeyDecode>& keys,
                           int totalBuckets);
    void setHaving(const PredPtr& havingPredicate) { havingPredicate_ = havingPredicate; }
    void setHavingTotal(const std::string& bufferName, int aggregateOffset);
    void setActiveBucketTracking(const std::string& flagBuffer,
                                 const std::string& listBuffer,
                                 const std::string& counterBuffer,
                                 const std::string& bucketCountExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(bucketExpr_, out);
        for (const auto& agg : aggregates_) {
            appendIUsFromExpr(agg.valueExpr, out);
        }
        for (const auto& db : distinctBitmaps_) {
            appendIUsFromExpr(db.valueExpr, out);
        }
    }

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

    struct HavingTotal {
        std::string bufferName;
        int aggregateOffset = -1;
    };
    std::optional<HavingTotal> havingTotal_;

    struct ActiveBucketTracking {
        std::string flagBuffer;
        std::string listBuffer;
        std::string counterBuffer;
        std::string bucketCountExpr;
    };
    std::optional<ActiveBucketTracking> activeBucketTracking_;

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(bucketExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

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
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(bucketExpr_, out);
    }

private:
    std::string arrayName_;
    std::string bucketExpr_;
    std::string sizeExpr_;
};

// --- Materialization ---

// Materialize rows to output arrays via an atomic counter.
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
    void iusUsed(std::vector<IU>& out) const override {
        for (const auto& col : columns_) {
            appendIUsFromExpr(col.valueExpr, out);
        }
    }

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

// --- GPU Bitonic Sort ---
// MetalInitSortKeys: encodes a source column into sort keys (uint64_t)
// and row indices.  Grid-strides over the source data.  Supports
// ascending/descending direction via key encoding.
//
// MetalBitonicSortStep: one comparison-swap pass of the bitonic
// sort network.  The kernel takes (k, j) constants and swaps elements
// if needed.  The plan attaches a PostDispatchHook that re-dispaches
// this kernel in the classic (k, j) nested loop.

class MetalInitSortKeys : public MetalOperator {
public:
    // capacityExpr is used when nResultsExpr is a runtime scalar from an earlier phase.
    MetalInitSortKeys(const std::string& sourceColumn, const std::string& sourceType,
                      const std::string& sortKeyBuf, const std::string& sortIdxBuf,
                      const std::string& nResultsExpr, bool descending,
                      const std::string& capacityExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

    // Exposed for the sort hook: the padded element count (next power of 2)
    static unsigned int nextPow2(unsigned int n);
    // Build the PostDispatchHook that runs the (k, j) bitonic loop.
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
    std::string capacityExpr_;
    bool descending_;
};

class MetalBitonicSortStep : public MetalOperator {
public:
    // sortKeyBuf / sortIdxBuf: buffer names matching those in MetalInitSortKeys
    // nResultsExpr: same expression used for init — used for grid sizing
    MetalBitonicSortStep(const std::string& sortKeyBuf, const std::string& sortIdxBuf,
                         const std::string& nResultsExpr,
                         const std::string& capacityExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string sortKeyBuf_;
    std::string sortIdxBuf_;
    std::string nResultsExpr_;
    std::string capacityExpr_;
};

} // namespace codegen
