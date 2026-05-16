#include "metal_operators.h"
#include "metal_plan_common.h"
#include "metal_generic_executor.h"
#include "../core/iu.hpp"
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace codegen {

// ===================================================================
// Static helper: parse colName[idxVar] references from expressions
// ===================================================================

void MetalOperator::appendIUsFromExpr(const std::string& expr,
                                       std::vector<IU>& out) {
    auto isIdent = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    };
    size_t n = expr.size();

    // --- Pass 1: colName[idxVar] patterns ---
    size_t i = 0;
    while (i < n) {
        if (!isIdent(expr[i])) { ++i; continue; }
        size_t s = i;
        while (i < n && isIdent(expr[i])) ++i;
        if (i >= n || expr[i] != '[') continue;
        if (s > 0 && isIdent(expr[s - 1])) continue;
        size_t lb = i;
        size_t ce = lb + 1;
        while (ce < n && expr[ce] != ']') ++ce;
        if (ce >= n) continue;
        size_t is = lb + 1;
        while (is < ce && !isIdent(expr[is])) ++is;
        if (is >= ce) continue;
        size_t ie = is;
        while (ie < ce && isIdent(expr[ie])) ++ie;
        out.emplace_back(expr.substr(s, lb - s),
                         expr.substr(is, ie - is));
        i = ce + 1;
    }

    // --- Pass 2: fixed-string pointer expressions ---
    // Materialization of CHAR_FIXED/CHAR1 values passes a row pointer, e.g.
    // "o_orderpriority + i * 15". Register the bare column with the row index.
    i = 0;
    while (i < n) {
        if (!isIdent(expr[i])) { ++i; continue; }
        size_t cs = i;
        while (i < n && isIdent(expr[i])) ++i;
        std::string colName = expr.substr(cs, i - cs);
        size_t p = i;
        while (p < n && expr[p] == ' ') ++p;
        if (p >= n || expr[p] != '+') continue;
        ++p;
        while (p < n && expr[p] == ' ') ++p;
        if (p >= n || !isIdent(expr[p])) continue;
        size_t is = p;
        while (p < n && isIdent(expr[p])) ++p;
        out.emplace_back(colName, expr.substr(is, p - is));
    }

    // --- Pass 3: bare column refs in fixed-string helpers ---
    // These helpers take the column buffer as first arg (bare identifier
    // without [idx]) and the row index as second arg: (uint)(idxVar).
    // Pattern: fixed_like_one_segment(COL, (uint)(IDX), ...
    //          fixed_like_two_segment(COL, (uint)(IDX), ...
    const char* helpers[] = {
        "fixed_like_one_segment(", "fixed_like_two_segment(",
        "fixed_string_segment_eq(", "fixed_string_padding_ok(",
        "q2_type_ends_brass(", "q13_comment_match(",
        "q16_has_complaint(", "brand_eq(", "container_match("
    };
    for (const char* hdr : helpers) {
        size_t pos = 0;
        while ((pos = expr.find(hdr, pos)) != std::string::npos) {
            pos += strlen(hdr);
            // Skip whitespace
            while (pos < n && expr[pos] == ' ') ++pos;
            if (pos >= n || !isIdent(expr[pos])) { ++pos; continue; }
            size_t cs = pos;
            while (pos < n && isIdent(expr[pos])) ++pos;
            std::string colName = expr.substr(cs, pos - cs);
            // Find idxVar from second arg: (uint)(i), — skip type casts
            std::string idxVar;
            while (pos < n && expr[pos] != ',') ++pos;
            if (pos < n) {
                ++pos; // skip ','
                while (pos < n && expr[pos] == ' ') ++pos;
                // Skip type-cast: "(uint)" style
                if (pos < n && expr[pos] == '(') {
                    while (pos < n && expr[pos] != ')') ++pos;
                    if (pos < n) ++pos; // skip ')'
                    while (pos < n && expr[pos] == ' ') ++pos;
                }
                // The next token may be wrapped in parens: "(i)" — extract it
                if (pos < n && expr[pos] == '(') {
                    ++pos; // skip '('
                    while (pos < n && expr[pos] == ' ') ++pos;
                    if (pos < n && isIdent(expr[pos])) {
                        size_t is = pos;
                        while (pos < n && isIdent(expr[pos])) ++pos;
                        idxVar = expr.substr(is, pos - is);
                    }
                } else if (pos < n && isIdent(expr[pos])) {
                    size_t is = pos;
                    while (pos < n && isIdent(expr[pos])) ++pos;
                    idxVar = expr.substr(is, pos - is);
                }
            }
            out.emplace_back(colName, idxVar);
        }
    }
}

// JSON serialization for operator trees
nlohmann::json MetalOperator::toJSON() const {
    nlohmann::json j;
    j["type"] = describe();
    return j;
}

nlohmann::json MetalUnaryOperator::toJSON() const {
    nlohmann::json j;
    j["type"] = describe();
    if (child_) j["child"] = child_->toJSON();
    return j;
}

// Ablation: when GPUDB_SCALAR_ATOMIC=1, MetalTGReduce skips the
// SIMD+threadgroup reduction and has every thread issue a global atomic.
// This isolates the value of the existing reduction strategy.
static bool scalarAtomicMode() {
    const char* e = std::getenv("GPUDB_SCALAR_ATOMIC");
    return e && e[0] && e[0] != '0';
}

// ===================================================================
// MetalGridStrideScan
// ===================================================================

MetalGridStrideScan::MetalGridStrideScan(const std::string& table,
                                         const std::string& rowVar,
                                         const std::string& idxVar)
    : tableName_(table), rowVar_(rowVar), idxVar_(idxVar) {}

void MetalGridStrideScan::addColumn(const std::string& paramName, const std::string& metalType) {
    columns_.push_back({paramName, metalType});
}

std::vector<MetalGridStrideScan::ColumnDesc>
MetalGridStrideScan::deduceRequiredColumns(MetalCodegen& cg) const {
    std::vector<IU> ius;
    for (MetalOperator* p = parent(); p; p = p->parent()) {
        p->iusUsed(ius);
    }
    std::vector<ColumnDesc> result;
    std::unordered_set<std::string> seen;
    for (const auto& iu : ius) {
        if (iu.idxVar != idxVar_) continue;
        if (seen.count(iu.colName)) continue;
        std::string mt = cg.resolveColumnType(tableName_, iu.colName);
        if (mt.empty()) continue;
        result.push_back({iu.colName, mt});
        seen.insert(iu.colName);
    }
    return result;
}

void MetalGridStrideScan::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable(tableName_);

    // Auto-discover additional columns via IU chain.
    auto discovered = deduceRequiredColumns(cg);
    std::unordered_set<std::string> seen;
    for (const auto& col : columns_) seen.insert(col.paramName);
    for (const auto& col : discovered) {
        if (!seen.count(col.paramName)) {
            columns_.push_back(col);
            seen.insert(col.paramName);
        }
    }

    if (columns_.empty()) {
        throw std::runtime_error(
            "MetalGridStrideScan(" + tableName_ +
            "): no columns registered. Call addColumn() before produce() "
            "or set a ColumnTypeResolver for auto-projection.");
    }
    for (const auto& col : columns_) {
        cg.addColumnParam(col.paramName, col.metalType, tableName_);
    }
    cg.addTableSizeParam(tableName_);

    // Emit grid-stride loop
    cg.addBlock("for (uint " + idxVar_ + " = tid; " + idxVar_ + " < " +
                tableSizeName(tableName_) + "; " + idxVar_ + " += tpg)", [&]() {
        consume();
    });
}

std::string MetalGridStrideScan::describe() const {
    return "GridStrideScan(" + tableName_ + ")";
}

// ===================================================================
// MetalRangeScan
// ===================================================================

MetalRangeScan::MetalRangeScan(const std::string& rangeName, const std::string& idxVar)
    : rangeName_(rangeName), idxVar_(idxVar) {}

void MetalRangeScan::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable(rangeName_);
    cg.addResolvedScalarParam("n_" + rangeName_, "uint", rangeName_);
    for (const auto& sc : sideColumns_) {
        cg.addColumnParam(sc.param, sc.type, sc.table);
    }
    cg.addBlock("for (uint " + idxVar_ + " = tid; " + idxVar_ + " < n_" +
                rangeName_ + "; " + idxVar_ + " += tpg)", [&]() {
        consume();
    });
}

void MetalRangeScan::addSideColumn(const std::string& tableName,
                                    const std::string& paramName,
                                    const std::string& metalType) {
    sideColumns_.push_back({tableName, paramName, metalType});
}

std::string MetalRangeScan::describe() const {
    return "RangeScan(" + rangeName_ + ")";
}

// ===================================================================
// MetalSelection
// ===================================================================

MetalSelection::MetalSelection(std::unique_ptr<MetalOperator> child,
                               const std::string& predicate)
    : MetalUnaryOperator(std::move(child)), predicate_(predicate) {}

void MetalSelection::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addIf(predicate_, [&]() {
            consume();
        });
    });
}

std::string MetalSelection::describe() const {
    return "Selection(" + predicate_ + ")";
}

// ===================================================================
// MetalComputeExpr
// ===================================================================

MetalComputeExpr::MetalComputeExpr(std::unique_ptr<MetalOperator> child,
                                   const std::string& varName,
                                   const std::string& varType,
                                   const std::string& expression)
    : MetalUnaryOperator(std::move(child)),
      varName_(varName), varType_(varType), expression_(expression) {}

void MetalComputeExpr::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addLine(varType_ + " " + varName_ + " = " + expression_ + ";");
        consume();
    });
}

std::string MetalComputeExpr::describe() const {
    return "ComputeExpr(" + varName_ + " = " + expression_ + ")";
}

// ===================================================================
// MetalBitmapBuild
// ===================================================================

MetalBitmapBuild::MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr), sizeExpr_(sizeExpr) {}

void MetalBitmapBuild::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap buffer as atomic for writes
    cg.addBitmapWriteParam(bitmapName_, sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("bitmap_set(" + bitmapName_ + ", " + keyExpr_ + ");");
        consume();  // allow chaining after bitmap set
    });
}

std::string MetalBitmapBuild::describe() const {
    return "BitmapBuild(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalBitmapProbe
// ===================================================================

MetalBitmapProbe::MetalBitmapProbe(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap as read-only in probe phase
    cg.addBitmapReadParam(bitmapName_, "");  // size comes from build phase

    child_->produce(cg, [&]() {
        cg.addIf("bitmap_test_atomic(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalBitmapProbe::describe() const {
    return "BitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalAntiBitmapProbe
// ===================================================================

MetalAntiBitmapProbe::MetalAntiBitmapProbe(std::unique_ptr<MetalOperator> child,
                                           const std::string& bitmapName,
                                           const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalAntiBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBitmapReadParam(bitmapName_, "");

    child_->produce(cg, [&]() {
        cg.addIf("!bitmap_test_atomic(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalAntiBitmapProbe::describe() const {
    return "AntiBitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalLeftOuterProbe
// ===================================================================

MetalLeftOuterProbe::MetalLeftOuterProbe(std::unique_ptr<MetalOperator> child,
                                         const std::string& bitmapName,
                                         const std::string& keyExpr,
                                         std::vector<DefaultVar> rightVars)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr), rightVars_(std::move(rightVars)) {}

void MetalLeftOuterProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBitmapReadParam(bitmapName_, "");

    child_->produce(cg, [&]() {
        cg.addIf("bitmap_test_atomic(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
        cg.addLine("else {");
        for (const auto& dv : rightVars_)
            cg.addLine(dv.varType + " " + dv.varName + " = " + dv.defaultVal + ";");
        consume();
        cg.addLine("}");
    });
}

std::string MetalLeftOuterProbe::describe() const {
    return "LeftOuterProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalArrayStore
// ===================================================================

MetalArrayStore::MetalArrayStore(std::unique_ptr<MetalOperator> child,
                                 const std::string& arrayName,
                                 const std::string& keyExpr,
                                 const std::string& valueExpr,
                                 const std::string& valueType,
                                 const std::string& sizeExpr,
                                 int fillByte)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), valueExpr_(valueExpr),
      valueType_(valueType), sizeExpr_(sizeExpr), fillByte_(fillByte) {}

void MetalArrayStore::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register array as device buffer; fillByte selects sentinel (0xFF = -1
    // for signed-int arrays, 0 for non-negative value arrays).
    cg.addBufferParam(arrayName_, valueType_, sizeExpr_, true, fillByte_);

    child_->produce(cg, [&]() {
        cg.addLine(arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ";");
        consume();
    });
}

std::string MetalArrayStore::describe() const {
    return "ArrayStore(" + arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ")";
}

// ===================================================================
// MetalArrayLookup
// ===================================================================

MetalArrayLookup::MetalArrayLookup(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& keyExpr,
                                   const std::string& resultVar,
                                   const std::string& resultType,
                                   int sentinel)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), resultVar_(resultVar),
      resultType_(resultType), sentinel_(sentinel) {}

void MetalArrayLookup::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Read-only access to array
    cg.addBufferParam(arrayName_, resultType_, "", false);

    child_->produce(cg, [&]() {
        cg.addLine(resultType_ + " " + resultVar_ + " = " + arrayName_ +
                   "[" + keyExpr_ + "];");
        // Guard: skip sentinel values (e.g., -1 means not found)
        cg.addIf(resultVar_ + " != " + std::to_string(sentinel_), [&]() {
            consume();
        });
    });
}

std::string MetalArrayLookup::describe() const {
    return "ArrayLookup(" + resultVar_ + " = " + arrayName_ + "[" + keyExpr_ + "])";
}

// ===================================================================
// MetalArraySliceStore
// ===================================================================

MetalArraySliceStore::MetalArraySliceStore(std::unique_ptr<MetalOperator> child,
                                           const std::string& arrayName,
                                           const std::string& keyExpr,
                                           const std::string& valuePtrExpr,
                                           int sliceLen,
                                           const std::string& valueType,
                                           const std::string& sizeExpr,
                                           int fillByte,
                                           std::string sourceColumn,
                                           std::string sourceIdxVar)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), valuePtrExpr_(valuePtrExpr),
      sliceLen_(sliceLen), valueType_(valueType), sizeExpr_(sizeExpr),
      fillByte_(fillByte), sourceColumn_(std::move(sourceColumn)),
      sourceIdxVar_(std::move(sourceIdxVar)) {}

void MetalArraySliceStore::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBufferParam(arrayName_, valueType_, sizeExpr_, true, fillByte_);

    child_->produce(cg, [&]() {
        cg.addBlock("for (uint _slice_i = 0; _slice_i < " +
                    std::to_string(sliceLen_) + "u; ++_slice_i)", [&]() {
            cg.addLine(arrayName_ + "[(" + keyExpr_ + ") * " +
                       std::to_string(sliceLen_) + "u + _slice_i] = (" +
                       valuePtrExpr_ + ")[_slice_i];");
        });
        consume();
    });
}

std::string MetalArraySliceStore::describe() const {
    return "ArraySliceStore(" + arrayName_ + "[" + keyExpr_ + ", width=" +
           std::to_string(sliceLen_) + "] = " + valuePtrExpr_ + ")";
}

// ===================================================================
// MetalArraySliceLookup
// ===================================================================

MetalArraySliceLookup::MetalArraySliceLookup(std::unique_ptr<MetalOperator> child,
                                             const std::string& arrayName,
                                             const std::string& keyExpr,
                                             const std::string& resultVar,
                                             int sliceLen,
                                             const std::string& resultType)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), resultVar_(resultVar),
      sliceLen_(sliceLen), resultType_(resultType) {}

void MetalArraySliceLookup::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBufferParam(arrayName_, resultType_, "", false);

    child_->produce(cg, [&]() {
        cg.addLine("const device " + resultType_ + "* " + resultVar_ + " = " +
                   arrayName_ + " + (" + keyExpr_ + ") * " +
                   std::to_string(sliceLen_) + "u;");
        consume();
    });
}

std::string MetalArraySliceLookup::describe() const {
    return "ArraySliceLookup(" + resultVar_ + " = " + arrayName_ + "[" +
           keyExpr_ + ", width=" + std::to_string(sliceLen_) + "])";
}

// ===================================================================
// MetalHashMapBuild
// ===================================================================

namespace {
inline std::string hmKeys1(const std::string& m) { return m + "_keys1"; }
inline std::string hmKeys2(const std::string& m) { return m + "_keys2"; }
inline std::string hmValues(const std::string& m) { return m + "_values"; }
} // namespace

MetalHashMapBuild::MetalHashMapBuild(std::unique_ptr<MetalOperator> child,
                                     const std::string& mapName,
                                     const std::string& key1Expr,
                                     const std::string& key2Expr,
                                     const std::string& valueExpr,
                                     const std::string& capacityExpr)
    : MetalUnaryOperator(std::move(child)),
      mapName_(mapName), key1Expr_(key1Expr), key2Expr_(key2Expr),
      valueExpr_(valueExpr), capacityExpr_(capacityExpr) {}

void MetalHashMapBuild::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Keys initialised to 0xFFFFFFFFu sentinel; value slot zero-init.
    cg.addAtomicBufferParam(hmKeys1(mapName_), "atomic_uint", capacityExpr_, 0xFF);
    cg.addAtomicBufferParam(hmKeys2(mapName_), "atomic_uint", capacityExpr_, 0xFF);
    cg.addAtomicBufferParam(hmValues(mapName_), "atomic_uint", capacityExpr_, 0);
    cg.addResolvedScalarParam("n_" + mapName_, "uint", capacityExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("hashmap_insert_kv(" + hmKeys1(mapName_) + ", " +
                   hmKeys2(mapName_) + ", " + hmValues(mapName_) + ", n_" +
                   mapName_ + ", (uint)(" + key1Expr_ + "), (uint)(" +
                   key2Expr_ + "), (uint)(" + valueExpr_ + "));");
        consume();
    });
}

std::string MetalHashMapBuild::describe() const {
    return "HashMapBuild(" + mapName_ + ", k=(" + key1Expr_ + "," + key2Expr_ +
           "), v=" + valueExpr_ + ")";
}

// ===================================================================
// MetalHashMapAgg
// ===================================================================

MetalHashMapAgg::MetalHashMapAgg(std::unique_ptr<MetalOperator> child,
                                 const std::string& mapName,
                                 const std::string& key1Expr,
                                 const std::string& key2Expr,
                                 const std::string& valueExpr,
                                 const std::string& capacityExpr,
                                 bool valueIsFloat)
    : MetalUnaryOperator(std::move(child)),
      mapName_(mapName), key1Expr_(key1Expr), key2Expr_(key2Expr),
      valueExpr_(valueExpr), capacityExpr_(capacityExpr),
      valueIsFloat_(valueIsFloat) {}

void MetalHashMapAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(hmKeys1(mapName_), "atomic_uint", capacityExpr_, 0xFF);
    cg.addAtomicBufferParam(hmKeys2(mapName_), "atomic_uint", capacityExpr_, 0xFF);
    cg.addAtomicBufferParam(hmValues(mapName_), "atomic_uint", capacityExpr_, 0);
    cg.addResolvedScalarParam("n_" + mapName_, "uint", capacityExpr_);

    child_->produce(cg, [&]() {
        if (valueIsFloat_) {
            cg.addLine("hashmap_insert_add_float(" + hmKeys1(mapName_) + ", " +
                       hmKeys2(mapName_) + ", " + hmValues(mapName_) + ", n_" +
                       mapName_ + ", (uint)(" + key1Expr_ + "), (uint)(" +
                       key2Expr_ + "), (float)(" + valueExpr_ + "));");
        } else {
            cg.addLine("hashmap_insert_add(" + hmKeys1(mapName_) + ", " +
                       hmKeys2(mapName_) + ", " + hmValues(mapName_) + ", n_" +
                       mapName_ + ", (uint)(" + key1Expr_ + "), (uint)(" +
                       key2Expr_ + "), (uint)(" + valueExpr_ + "));");
        }
        consume();
    });
}

std::string MetalHashMapAgg::describe() const {
    return "HashMapAgg(" + mapName_ + ", k=(" + key1Expr_ + "," + key2Expr_ +
           "), +=" + valueExpr_ + ")";
}

// ===================================================================
// MetalHashMapLookup
// ===================================================================

MetalHashMapLookup::MetalHashMapLookup(std::unique_ptr<MetalOperator> child,
                                       const std::string& mapName,
                                       const std::string& key1Expr,
                                       const std::string& key2Expr,
                                       const std::string& capacityExpr,
                                       const std::string& resultVar,
                                       const std::string& resultType)
    : MetalUnaryOperator(std::move(child)),
      mapName_(mapName), key1Expr_(key1Expr), key2Expr_(key2Expr),
      capacityExpr_(capacityExpr),
      resultVar_(resultVar), resultType_(resultType) {}

void MetalHashMapLookup::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBufferParam(hmKeys1(mapName_), "uint", "", false);
    cg.addBufferParam(hmKeys2(mapName_), "uint", "", false);
    cg.addBufferParam(hmValues(mapName_), "uint", "", false);
    cg.addResolvedScalarParam("n_" + mapName_, "uint", capacityExpr_);

    child_->produce(cg, [&]() {
        std::string slot = resultVar_ + "_slot";
        cg.addLine("uint " + slot + " = hashmap_lookup(" + hmKeys1(mapName_) +
                   ", " + hmKeys2(mapName_) + ", n_" + mapName_ +
                   ", (uint)(" + key1Expr_ + "), (uint)(" + key2Expr_ + "));");
        cg.addIf(slot + " != 0xFFFFFFFFu", [&]() {
            std::string raw = hmValues(mapName_) + "[" + slot + "]";
            std::string casted;
            if (resultType_ == "float") {
                casted = "as_type<float>(" + raw + ")";
            } else if (resultType_ == "int") {
                casted = "(int)(" + raw + ")";
            } else {
                casted = raw;
            }
            cg.addLine(resultType_ + " " + resultVar_ + " = " + casted + ";");
            consume();
        });
    });
}

std::string MetalHashMapLookup::describe() const {
    return "HashMapLookup(" + mapName_ + ", k=(" + key1Expr_ + "," + key2Expr_ +
           ") -> " + resultVar_ + ")";
}

// ===================================================================
// MetalTGReduce
// ===================================================================

MetalTGReduce::MetalTGReduce(std::unique_ptr<MetalOperator> child,
                             const std::string& outputPrefix)
    : MetalUnaryOperator(std::move(child)), outputPrefix_(outputPrefix) {}

int MetalTGReduce::addAccumulator(const std::string& name,
                                   const std::string& valueExpr,
                                   const std::string& type,
                                   const std::string& loBuffer,
                                   const std::string& hiBuffer,
                                   ReduceOp op) {
    Accumulator acc;
    acc.name = name;
    acc.valueExpr = valueExpr;
    acc.type = type;
    acc.op = op;
    acc.loBuffer = loBuffer.empty() ? (outputPrefix_ + "_" + name + "_lo") : loBuffer;
    acc.hiBuffer = hiBuffer.empty() ? (type == "long" ? (outputPrefix_ + "_" + name + "_hi") : "") : hiBuffer;
    if (op != ReduceOp::SUM) acc.stateBuffer = outputPrefix_ + "_" + name + "_state";
    acc.binIndex = static_cast<int>(accumulators_.size());
    accumulators_.push_back(acc);
    return acc.binIndex;
}

void MetalTGReduce::setResultAlias(const std::string& displayName, int scaleDown) {
    setAccumulatorResultAlias(displayName, static_cast<int>(resultInfos_.size()), scaleDown);
}

void MetalTGReduce::setAccumulatorResultAlias(const std::string& displayName,
                                              int accumulatorIndex,
                                              int scaleDown,
                                              ExprPtr projectionExpr) {
    resultInfos_.push_back({displayName, scaleDown, accumulatorIndex, -1,
                            std::move(projectionExpr)});
}

void MetalTGReduce::setAverageResultAlias(const std::string& displayName,
                                          int numeratorIndex,
                                          int denominatorIndex,
                                          int scaleDown,
                                          ExprPtr projectionExpr) {
    resultInfos_.push_back({displayName, scaleDown, numeratorIndex, denominatorIndex,
                            std::move(projectionExpr)});
}

void MetalTGReduce::produce(MetalCodegen& cg, ConsumerFn consume) {
    const bool scalar = scalarAtomicMode();

    // Register output buffers
    for (const auto& acc : accumulators_) {
        if (acc.type == "float") {
            // Float path: single atomic_uint buffer (reinterpreted as float via CAS)
            cg.addAtomicBufferParam(acc.loBuffer, "atomic_uint", "1");
        } else if (acc.type == "int") {
            cg.addAtomicBufferParam(acc.loBuffer, "atomic_int", "1");
        } else {
            // Long path: lo/hi atomic_uint pair
            cg.addAtomicBufferParam(acc.loBuffer, "atomic_uint", "1");
            cg.addAtomicBufferParam(acc.hiBuffer, "atomic_uint", "1");
        }
        if (acc.op != ReduceOp::SUM) {
            cg.addAtomicBufferParam(acc.stateBuffer, "atomic_uint", "1");
        }
    }

    if (scalar) {
        // ===== SCALAR-ATOMIC ABLATION =====
        // Each thread issues a global atomic per row consumed. No local
        // accumulation, no SIMD/TG reduction, no shared memory.
        cg.addComment("--- Scalar-atomic mode: per-row global atomic ---");
        child_->produce(cg, [&]() {
            for (const auto& acc : accumulators_) {
                if (acc.type == "float") {
                    std::string value = "(float)(" + acc.valueExpr + ")";
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_float_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + value + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_float_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + value + ");");
                    } else {
                        cg.addLine("atomic_add_float(" + acc.loBuffer + ", " + value + ");");
                    }
                } else if (acc.type == "int") {
                    std::string value = "(int)(" + acc.valueExpr + ")";
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_int_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + value + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_int_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + value + ");");
                    } else {
                        cg.addLine("atomic_fetch_add_explicit(" + acc.loBuffer + ", " +
                                   value + ", memory_order_relaxed);");
                    }
                } else {
                    std::string value = "(long)(" + acc.valueExpr + ")";
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_long_pair_seen(" + acc.loBuffer + ", " +
                                   acc.hiBuffer + ", " + acc.stateBuffer + ", " + value + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_long_pair_seen(" + acc.loBuffer + ", " +
                                   acc.hiBuffer + ", " + acc.stateBuffer + ", " + value + ");");
                    } else {
                        cg.addLine("atomic_add_long_pair(" + acc.loBuffer + ", "
                                   + acc.hiBuffer + ", " + value + ");");
                    }
                }
            }
            consume();
        });
    } else {
        // Declare local accumulators
        for (const auto& acc : accumulators_) {
            if (acc.type == "float") {
                std::string init = "0.0f";
                if (acc.op == ReduceOp::MIN) init = "3.402823466e+38f";
                else if (acc.op == ReduceOp::MAX) init = "-3.402823466e+38f";
                cg.addLine("float local_" + acc.name + " = " + init + ";");
            } else if (acc.type == "int") {
                std::string init = "0";
                if (acc.op == ReduceOp::MIN) init = "2147483647";
                else if (acc.op == ReduceOp::MAX) init = "-2147483647";
                cg.addLine("int local_" + acc.name + " = " + init + ";");
            } else {
                std::string init = "0";
                if (acc.op == ReduceOp::MIN) init = "9223372036854775807L";
                else if (acc.op == ReduceOp::MAX) init = "-9223372036854775807L";
                cg.addLine("long local_" + acc.name + " = " + init + ";");
            }
        }

        // Child produces rows; inside the loop we accumulate
        child_->produce(cg, [&]() {
            for (const auto& acc : accumulators_) {
                if (acc.type == "float") {
                    std::string valueVar = "_value_" + acc.name;
                    cg.addLine("float " + valueVar + " = (float)(" + acc.valueExpr + ");");
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " < local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " > local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else {
                        cg.addLine("local_" + acc.name + " += " + valueVar + ";");
                    }
                } else if (acc.type == "int") {
                    std::string valueVar = "_value_" + acc.name;
                    cg.addLine("int " + valueVar + " = (int)(" + acc.valueExpr + ");");
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " < local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " > local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else {
                        cg.addLine("local_" + acc.name + " += " + valueVar + ";");
                    }
                } else {
                    std::string valueVar = "_value_" + acc.name;
                    cg.addLine("long " + valueVar + " = (long)(" + acc.valueExpr + ");");
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " < local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("local_" + acc.name + " = (" + valueVar + " > local_" +
                                   acc.name + ") ? " + valueVar + " : local_" + acc.name + ";");
                    } else {
                        cg.addLine("local_" + acc.name + " += " + valueVar + ";");
                    }
                }
            }
            consume();
        });

        // After the loop: SIMD + threadgroup reduction → atomic write
        cg.addComment("--- Threadgroup reduction ---");
        for (const auto& acc : accumulators_) {
            std::string localVar = "local_" + acc.name;
            std::string tgVar = "tg_" + acc.name;

            if (acc.type == "float") {
                cg.addLine("threadgroup float tg_shared_" + acc.name + "[32];");
                std::string reduceFn = "tg_reduce_float";
                if (acc.op == ReduceOp::MIN) reduceFn = "tg_reduce_min_float";
                else if (acc.op == ReduceOp::MAX) reduceFn = "tg_reduce_max_float";
                cg.addLine("float " + tgVar + " = " + reduceFn + "(" + localVar +
                           ", lid, tg_size, tg_shared_" + acc.name + ");");
                cg.addIf("lid == 0", [&]() {
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_float_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + tgVar + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_float_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + tgVar + ");");
                    } else {
                        cg.addLine("atomic_add_float(" + acc.loBuffer + ", " + tgVar + ");");
                    }
                });
            } else if (acc.type == "int") {
                cg.addLine("threadgroup int tg_shared_" + acc.name + "[32];");
                std::string reduceFn = "tg_reduce_uint";
                if (acc.op == ReduceOp::MIN) reduceFn = "tg_reduce_min_int";
                else if (acc.op == ReduceOp::MAX) reduceFn = "tg_reduce_max_int";
                cg.addLine("int " + tgVar + " = " + reduceFn + "(" + localVar +
                           ", lid, tg_size, tg_shared_" + acc.name + ");");
                cg.addIf("lid == 0", [&]() {
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_int_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + tgVar + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_int_seen(" + acc.loBuffer + ", " +
                                   acc.stateBuffer + ", " + tgVar + ");");
                    } else {
                        cg.addLine("atomic_fetch_add_explicit(" + acc.loBuffer + ", " +
                                   tgVar + ", memory_order_relaxed);");
                    }
                });
            } else {
                cg.addLine("threadgroup long tg_shared_" + acc.name + "[32];");
                std::string reduceFn = "tg_reduce_long";
                if (acc.op == ReduceOp::MIN) reduceFn = "tg_reduce_min_long";
                else if (acc.op == ReduceOp::MAX) reduceFn = "tg_reduce_max_long";
                cg.addLine("long " + tgVar + " = " + reduceFn + "(" + localVar +
                           ", lid, tg_size, tg_shared_" + acc.name + ");");
                cg.addIf("lid == 0", [&]() {
                    if (acc.op == ReduceOp::MIN) {
                        cg.addLine("atomic_min_long_pair_seen(" + acc.loBuffer + ", " +
                                   acc.hiBuffer + ", " + acc.stateBuffer + ", " + tgVar + ");");
                    } else if (acc.op == ReduceOp::MAX) {
                        cg.addLine("atomic_max_long_pair_seen(" + acc.loBuffer + ", " +
                                   acc.hiBuffer + ", " + acc.stateBuffer + ", " + tgVar + ");");
                    } else {
                        cg.addLine("atomic_add_long_pair(" + acc.loBuffer + ", " +
                                   acc.hiBuffer + ", " + tgVar + ");");
                    }
                });
            }
        }
    }

    // Register result schema
    if (!resultInfos_.empty()) {
        for (const auto& info : resultInfos_) {
            if (info.accumulatorIndex < 0 ||
                static_cast<size_t>(info.accumulatorIndex) >= accumulators_.size()) continue;
            const auto& acc = accumulators_[info.accumulatorIndex];
            if (info.denominatorIndex >= 0 &&
                static_cast<size_t>(info.denominatorIndex) < accumulators_.size()) {
                const auto& denom = accumulators_[info.denominatorIndex];
                cg.registerScalarAggAverageColumn(info.displayName,
                                                  acc.loBuffer, acc.hiBuffer,
                                                  denom.loBuffer, denom.hiBuffer,
                                                  acc.type, info.scaleDown,
                                                  info.projectionExpr);
            } else {
                cg.registerScalarAggOutput(acc.loBuffer, acc.hiBuffer, acc.type);
                cg.registerScalarAggColumn(info.displayName, info.accumulatorIndex, info.scaleDown,
                                           info.projectionExpr);
            }
        }
    }
}

std::string MetalTGReduce::describe() const {
    return "TGReduce(" + std::to_string(accumulators_.size()) + " accumulators)";
}

// ===================================================================
// MetalKeyedAgg
// ===================================================================

MetalKeyedAgg::MetalKeyedAgg(std::unique_ptr<MetalOperator> child,
                             const std::string& outputArrayName,
                             const std::string& bucketExpr,
                             int numBuckets,
                             int valuesPerBucket,
                             const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      outputArrayName_(outputArrayName), bucketExpr_(bucketExpr),
      numBuckets_(numBuckets), valuesPerBucket_(valuesPerBucket),
      sizeExpr_(sizeExpr) {}

void MetalKeyedAgg::addAggregate(const std::string& name, int offset,
                                  const std::string& valueExpr,
                                  const std::string& atomicOp,
                                  bool isLongPair,
                                  int scaleDown) {
    aggregates_.push_back({name, offset, valueExpr, atomicOp, isLongPair, scaleDown,
                           false, false, "", ""});
}

void MetalKeyedAgg::addAggregateWithMeta(const std::string& name, int offset,
                                          const std::string& valueExpr,
                                          const std::string& atomicOp,
                                          bool isLongPair,
                                          int scaleDown,
                                          bool isFloatSum,
                                          bool isMinMax,
                                          const std::string& funcName,
                                          const std::string& innerColumn) {
    aggregates_.push_back({name, offset, valueExpr, atomicOp, isLongPair, scaleDown,
                           isFloatSum, isMinMax, funcName, innerColumn});
}

void MetalKeyedAgg::setKeyResult(const std::string& displayName, int base) {
    keyDisplayName_ = displayName;
    keyBase_ = base;
    multiKeyDecode_.clear();
}

void MetalKeyedAgg::setMultiKeyResult(const std::vector<std::string>& displayNames,
                                       const std::vector<GroupKeyDecode>& keys,
                                       int /*totalBuckets*/) {
    keyDisplayName_ = displayNames.empty() ? "bucket" : displayNames[0];
    keyBase_ = 0;
    multiKeyDecode_ = keys;
}

void MetalKeyedAgg::addDistinctBitmap(const std::string& outputName,
                                       const std::string& valueExpr,
                                       const std::string& maxValueExpr) {
    std::string bmpName = "d_distinct_bmp_" + outputName;
    distinctBitmaps_.push_back({outputName, bmpName, valueExpr, maxValueExpr});
}

void MetalKeyedAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register output buffer
    std::string sz = sizeExpr_.empty()
        ? std::to_string(numBuckets_ * valuesPerBucket_)
        : sizeExpr_;
    cg.addAtomicBufferParam(outputArrayName_, "atomic_uint", sz);

    // Register COUNT(DISTINCT) bitmap buffers.
    for (const auto& db : distinctBitmaps_) {
        std::string strideExpr = "(" + db.maxValueExpr + " + 32) / 32";
        std::string bmpSize = std::to_string(numBuckets_) + " * " + strideExpr;
        cg.addAtomicBufferParam(db.bitmapName, "atomic_uint", bmpSize);
    }

    // --- Thread-local accumulation + TG reduction strategy ---
    // Instead of per-row global atomics, accumulate in thread-local arrays,
    // then do threadgroup SIMD reduction and a single atomic per TG per bucket.
    // This reduces atomic operations from O(rows) to O(threadgroups × buckets).

    // Check if all aggregates are "add" (reduction-compatible)
    bool allAdds = true;
    for (const auto& agg : aggregates_) {
        if (agg.atomicOp != "add") { allAdds = false; break; }
    }

    // Only use TG reduction when there are enough aggregates per row to justify
    // the reduction overhead. With few aggs (e.g. 1 count), the barrier cost
    // exceeds the atomic savings, especially for low-selectivity joins.
    //
    // Tuning knobs (empirical):
    //   - kMaxBucketsForTGReduce: per-thread accumulator array length cap.
    //     Above this, register pressure and TG-shared memory dominate.
    //   - kMinAggsForTGReduce: minimum aggs per row to amortise the two-level
    //     reduction barrier cost.
    constexpr int kMaxBucketsForTGReduce = 64;
    constexpr int kMinAggsForTGReduce    = 3;
    if (allAdds && numBuckets_ <= kMaxBucketsForTGReduce &&
        (int)aggregates_.size() >= kMinAggsForTGReduce) {
        // === OPTIMIZED PATH: thread-local + TG reduction ===
        // When HAVING is present, force single-threadgroup dispatch so the
        // threadgroup reduction produces global (not just per-TG) totals.
        if (havingPredicate_)
            cg.setPhaseMaxThreadgroups(1);
        else
            cg.setPhaseMaxThreadgroups(1024);

        // Declare and initialize thread-local accumulator arrays (merged init)
        for (const auto& agg : aggregates_) {
            if (agg.isFloatSum)
                cg.addLine("float _local_" + agg.name + "[" + std::to_string(numBuckets_) + "];");
            else if (agg.isLongPair)
                cg.addLine("long _local_" + agg.name + "[" + std::to_string(numBuckets_) + "];");
            else
                cg.addLine("uint _local_" + agg.name + "[" + std::to_string(numBuckets_) + "];");
        }
        // Single merged init loop for all aggregates
        cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
            for (const auto& agg : aggregates_) {
                if (agg.isFloatSum)
                    cg.addLine("_local_" + agg.name + "[_b] = 0.0f;");
                else
                    cg.addLine("_local_" + agg.name + "[_b] = 0;");
            }
        });

        // Child produces rows; inside the loop we accumulate locally (no atomics)
        child_->produce(cg, [&]() {
            cg.addLine("int _bucket = " + bucketExpr_ + ";");
            for (const auto& agg : aggregates_) {
                if (agg.isFloatSum) {
                    cg.addLine("_local_" + agg.name + "[_bucket] += (float)(" + agg.valueExpr + ");");
                } else if (agg.isLongPair) {
                    cg.addLine("_local_" + agg.name + "[_bucket] += (long)(" + agg.valueExpr + ");");
                } else {
                    cg.addLine("_local_" + agg.name + "[_bucket] += (uint)(" + agg.valueExpr + ");");
                }
            }
            // COUNT(DISTINCT): per-row atomic bit set.
            for (const auto& db : distinctBitmaps_) {
                std::string strideExpr = "(" + db.maxValueExpr + " + 32u) / 32u";
                cg.addLine("atomic_fetch_or_explicit(&" + db.bitmapName +
                           "[_bucket * " + strideExpr + " + ((" + db.valueExpr +
                           ") >> 5)], 1u << ((" + db.valueExpr + ") & 31u), memory_order_relaxed);");
            }
            consume();
        });

        // After the loop: TG reduction per bucket per aggregate, then single atomic.
        // When a HAVING predicate is present, restructure to per-bucket outer loop
        // so the GPU can filter groups before writing to global memory.
        cg.addComment("--- Threadgroup reduction for keyed aggregation ---");

        if (havingPredicate_) {
            // === HAVING-AWARE path: per-bucket loop, reduce all aggs, filter, write ===

            // Declare all TG shared arrays upfront (one per aggregate)
            for (const auto& agg : aggregates_) {
                if (agg.isFloatSum)
                    cg.addLine("threadgroup float _tg_shared_" + agg.name + "[32];");
                else if (agg.isLongPair)
                    cg.addLine("threadgroup long _tg_shared_" + agg.name + "[32];");
                else
                    cg.addLine("threadgroup uint _tg_shared_" + agg.name + "[32];");
            }

            // Build slot info for HAVING expression translation
            std::vector<MetalKeyedAggSlotForHaving> havingSlots;
            for (const auto& agg : aggregates_) {
                havingSlots.push_back({agg.name, agg.isFloatSum, agg.isLongPair,
                                       agg.funcName, agg.innerColumn});
            }
            std::string havingCond = predToMetalForHaving(havingPredicate_, havingSlots);

            cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                // Reduce each aggregate for this bucket
                for (const auto& agg : aggregates_) {
                    if (agg.isFloatSum) {
                        cg.addLine("float _tg_" + agg.name + " = tg_reduce_float(_local_" +
                                   agg.name + "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                    } else if (agg.isLongPair) {
                        cg.addLine("long _tg_" + agg.name + " = tg_reduce_long(_local_" +
                                   agg.name + "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                    } else {
                        cg.addLine("uint _tg_" + agg.name + " = tg_reduce_uint(_local_" +
                                   agg.name + "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                    }
                }

                cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");

                // HAVING check + conditional atomic write of all aggregates
                cg.addIf("lid == 0 && " + havingCond, [&]() {
                    for (const auto& agg : aggregates_) {
                        if (agg.isFloatSum) {
                            std::string idx = "_b * " + std::to_string(valuesPerBucket_) +
                                              " + " + std::to_string(agg.offset);
                            cg.addLine("if (_tg_" + agg.name + " != 0.0f) atomic_add_float(&" +
                                       outputArrayName_ + "[" + idx + "], _tg_" + agg.name + ");");
                        } else if (agg.isLongPair) {
                            std::string loIdx = "_b * " + std::to_string(valuesPerBucket_) +
                                                " + " + std::to_string(agg.offset);
                            std::string hiIdx = "_b * " + std::to_string(valuesPerBucket_) +
                                                " + " + std::to_string(agg.offset + 1);
                            cg.addLine("if (_tg_" + agg.name + " != 0) atomic_add_long_pair(&" +
                                       outputArrayName_ + "[" + loIdx + "], &" +
                                       outputArrayName_ + "[" + hiIdx + "], _tg_" + agg.name + ");");
                        } else {
                            std::string idx = "_b * " + std::to_string(valuesPerBucket_) +
                                              " + " + std::to_string(agg.offset);
                            cg.addLine("if (_tg_" + agg.name + " != 0) atomic_fetch_add_explicit(&" +
                                       outputArrayName_ + "[" + idx + "], _tg_" + agg.name +
                                       ", memory_order_relaxed);");
                        }
                    }
                });

                cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            });

            // Mark HAVING as evaluated on GPU so the CPU-side result collector skips it
            cg.setKeyedAggHavingEvaluatedOnGPU();
        } else {
            // === ORIGINAL path (no HAVING): per-aggregate outer loop ===
            for (const auto& agg : aggregates_) {
                if (agg.isFloatSum) {
                    cg.addLine("threadgroup float _tg_shared_" + agg.name + "[32];");
                    cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                        cg.addLine("float _tg_val = tg_reduce_float(_local_" + agg.name +
                                   "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                        cg.addIf("lid == 0 && _tg_val != 0.0f", [&]() {
                            std::string idx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset);
                            cg.addLine("atomic_add_float(&" + outputArrayName_ + "[" + idx +
                                       "], _tg_val);");
                        });
                    });
                } else if (agg.isLongPair) {
                    cg.addLine("threadgroup long _tg_shared_" + agg.name + "[32];");
                    cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                        cg.addLine("long _tg_val = tg_reduce_long(_local_" + agg.name +
                                   "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                        cg.addIf("lid == 0 && _tg_val != 0", [&]() {
                            std::string loIdx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset);
                            std::string hiIdx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset + 1);
                            cg.addLine("atomic_add_long_pair(&" + outputArrayName_ + "[" + loIdx +
                                       "], &" + outputArrayName_ + "[" + hiIdx +
                                       "], _tg_val);");
                        });
                    });
                } else {
                    cg.addLine("threadgroup uint _tg_shared_" + agg.name + "[32];");
                    cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                        cg.addLine("uint _tg_val = tg_reduce_uint(_local_" + agg.name +
                                   "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                        cg.addIf("lid == 0 && _tg_val != 0", [&]() {
                            std::string idx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset);
                            cg.addLine("atomic_fetch_add_explicit(&" + outputArrayName_ + "[" + idx +
                                       "], _tg_val, memory_order_relaxed);");
                        });
                    });
                }
            }
        }
    } else {
        // === FALLBACK: per-row global atomics (for min/max or high-cardinality) ===
        child_->produce(cg, [&]() {
            cg.addLine("int _bucket = " + bucketExpr_ + ";");
            for (const auto& agg : aggregates_) {
                std::string base = "_bucket * " + std::to_string(valuesPerBucket_);
                if (agg.isFloatSum && agg.atomicOp == "add") {
                    std::string idx = base + " + " + std::to_string(agg.offset);
                    cg.addLine("atomic_add_float(&" + outputArrayName_ + "[" + idx +
                               "], (float)(" + agg.valueExpr + "));");
                } else if (agg.isLongPair && agg.atomicOp == "add") {
                    std::string loIdx = base + " + " + std::to_string(agg.offset);
                    std::string hiIdx = base + " + " + std::to_string(agg.offset + 1);
                    cg.addLine("atomic_add_long_pair(&" + outputArrayName_ + "[" + loIdx +
                               "], &" + outputArrayName_ + "[" + hiIdx +
                               "], (long)(" + agg.valueExpr + "));");
                } else {
                    std::string idx = base + " + " + std::to_string(agg.offset);
                    if (agg.atomicOp == "add") {
                        cg.addLine("atomic_fetch_add_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    } else if (agg.atomicOp == "min") {
                        cg.addLine("atomic_fetch_min_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    } else if (agg.atomicOp == "max") {
                        cg.addLine("atomic_fetch_max_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    }
                }
            }
            consume();
        });
    }

    // Register result schema with slot layout
    std::vector<MetalResultSchema::KeyedAggSlot> slots;
    for (const auto& agg : aggregates_) {
        slots.push_back({agg.name, agg.offset, agg.isLongPair, agg.scaleDown,
                         agg.isFloatSum, agg.isMinMax, agg.atomicOp, -1});
    }
    cg.registerKeyedAggOutput(outputArrayName_, numBuckets_, valuesPerBucket_, slots,
                              keyDisplayName_, keyBase_);
    // Set HAVING predicate if present
    if (havingPredicate_) {
        cg.setKeyedAggHaving(havingPredicate_);
    }
    // If multi-key decode info is present, set it on the schema
    if (!multiKeyDecode_.empty()) {
        for (const auto& mk : multiKeyDecode_) {
            MetalResultSchema::KeyedAggInfo::MultiKeyInfo info;
            info.displayName = mk.name;
            info.numValues = mk.numValues;
            info.stride = mk.stride;
            info.charMap = mk.charMap;
            info.keyBase = mk.keyBase;
            cg.getResultSchemaMutable().keyedAgg.multiKeys.push_back(info);
        }
    }
}

std::string MetalKeyedAgg::describe() const {
    return "KeyedAgg(" + outputArrayName_ + ", " + std::to_string(numBuckets_) +
           " buckets, " + std::to_string(aggregates_.size()) + " aggs)";
}

// ===================================================================
// MetalAtomicAgg
// ===================================================================

MetalAtomicAgg::MetalAtomicAgg(std::unique_ptr<MetalOperator> child,
                               const std::string& arrayName,
                               const std::string& bucketExpr,
                               const std::string& valueExpr,
                               const std::string& sizeExpr,
                               const std::string& atomicType,
                               const std::string& castType)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr),
      valueExpr_(valueExpr), sizeExpr_(sizeExpr),
      atomicType_(atomicType), castType_(castType) {}

void MetalAtomicAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, atomicType_, sizeExpr_);

    child_->produce(cg, [&]() {
        if (atomicType_ == "atomic_uint" && castType_ == "float") {
            cg.addLine("atomic_add_float(&" + arrayName_ + "[" + bucketExpr_ +
                       "], (float)(" + valueExpr_ + "));");
        } else {
            cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                       "], (" + castType_ + ")(" + valueExpr_ + "), memory_order_relaxed);");
        }
        consume();
    });
}

std::string MetalAtomicAgg::describe() const {
    return "AtomicAgg(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalAtomicCount
// ===================================================================

MetalAtomicCount::MetalAtomicCount(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& bucketExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr), sizeExpr_(sizeExpr) {}

void MetalAtomicCount::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, "atomic_uint", sizeExpr_);

    // Parse sizeExpr to determine if we can use threadgroup-local histogram.
    // For small, statically-known bucket counts (≤ 256), use TG-local histogram
    // to drastically reduce global atomic contention.
    int staticSize = 0;
    try { staticSize = std::stoi(sizeExpr_); } catch (...) {}

    if (staticSize > 0 && staticSize <= 256) {
        // === OPTIMIZED: Threadgroup-local histogram ===
        cg.setPhaseMaxThreadgroups(1024);
        std::string szStr = std::to_string(staticSize);

        // Declare threadgroup-local histogram and zero-initialize
        cg.addLine("threadgroup uint _tg_hist_" + arrayName_ + "[" + szStr + "];");
        cg.addBlock("for (uint _h = lid; _h < " + szStr + "u; _h += tg_size)", [&]() {
            cg.addLine("_tg_hist_" + arrayName_ + "[_h] = 0;");
        });
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");

        // Child scan loop — accumulate into threadgroup-local histogram
        child_->produce(cg, [&]() {
            cg.addLine("atomic_fetch_add_explicit((threadgroup atomic_uint*)&_tg_hist_" +
                       arrayName_ + "[" + bucketExpr_ + "], 1u, memory_order_relaxed);");
            consume();
        });

        // Barrier, then flush non-zero bins to global
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        cg.addBlock("for (uint _h = lid; _h < " + szStr + "u; _h += tg_size)", [&]() {
            cg.addIf("_tg_hist_" + arrayName_ + "[_h] > 0", [&]() {
                cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[_h], " +
                           "_tg_hist_" + arrayName_ + "[_h], memory_order_relaxed);");
            });
        });
    } else {
        // === FALLBACK: per-row global atomics ===
        child_->produce(cg, [&]() {
            cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                       "], 1u, memory_order_relaxed);");
            consume();
        });
    }
}

std::string MetalAtomicCount::describe() const {
    return "AtomicCount(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalMaterialize
// ===================================================================

MetalMaterialize::MetalMaterialize(std::unique_ptr<MetalOperator> child,
                                   const std::string& counterName,
                                   const std::string& counterSizeExpr)
    : MetalUnaryOperator(std::move(child)),
      counterName_(counterName), counterSizeExpr_(counterSizeExpr) {}

void MetalMaterialize::addColumn(const std::string& arrayName, const std::string& type,
                                  const std::string& valueExpr,
                                  const std::string& displayName,
                                  const std::string& sizeExpr,
                                  int stringLen) {
    columns_.push_back({arrayName, type, valueExpr,
                        displayName.empty() ? arrayName : displayName, sizeExpr, stringLen});
}

void MetalMaterialize::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register counter
    cg.addAtomicBufferParam(counterName_, "atomic_uint", counterSizeExpr_);

    // Register output column buffers
    for (const auto& col : columns_) {
        cg.addBufferParam(col.arrayName, col.type, col.sizeExpr, false);
    }

    // Register result schema
    cg.registerMaterializeOutput(counterName_);
    for (const auto& col : columns_) {
        cg.registerOutputColumn(col.displayName, col.arrayName, col.type, col.stringLen);
    }

    child_->produce(cg, [&]() {
        // Atomic increment counter to get output position
        cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + counterName_ +
                   "[0], 1u, memory_order_relaxed);");
        // Scatter values to output arrays
        for (const auto& col : columns_) {
            if (col.stringLen > 0) {
                cg.addBlock("for (uint _ci = 0; _ci < " + std::to_string(col.stringLen) + "; _ci++)", [&]() {
                    cg.addLine(col.arrayName + "[_pos * " + std::to_string(col.stringLen) + " + _ci] = " +
                               "(" + col.valueExpr + ")[_ci];");
                });
            } else {
                cg.addLine(col.arrayName + "[_pos] = " + col.valueExpr + ";");
            }
        }
        consume();
    });
}

std::string MetalMaterialize::describe() const {
    return "Materialize(" + std::to_string(columns_.size()) + " columns)";
}

MetalBitmapPopcount::MetalBitmapPopcount(const std::string& bitmapName,
                                          const std::string& outputName,
                                          const std::string& numGroupsExpr,
                                          const std::string& bitmapStrideExpr)
    : bitmapName_(bitmapName), outputName_(outputName),
      numGroupsExpr_(numGroupsExpr), bitmapStrideExpr_(bitmapStrideExpr) {}

void MetalBitmapPopcount::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBufferParam(bitmapName_, "uint", bitmapStrideExpr_ + " * " + numGroupsExpr_, false);
    cg.addAtomicBufferParam(outputName_, "atomic_uint", numGroupsExpr_, 0xFF);
    cg.addScalarParam("n_" + numGroupsExpr_, "uint");
    cg.addResolvedScalarParam("n_bmp_stride", "uint", bitmapStrideExpr_);

    cg.addBlock("for (uint _g = tid; _g < n_" + numGroupsExpr_ + "; _g += tpg)", [&]() {
        cg.addLine("uint _cnt = 0;");
        cg.addBlock("for (uint _w = 0; _w < n_bmp_stride; ++_w)", [&]() {
            cg.addLine("_cnt += popcount(" + bitmapName_ + "[_g * n_bmp_stride + _w]);");
        });
        cg.addLine("atomic_store_explicit(&" + outputName_ + "[_g], _cnt, memory_order_relaxed);");
    });
    consume();
}

std::string MetalBitmapPopcount::describe() const {
    return "BitmapPopcount(" + bitmapName_ + ")";
}

// ===================================================================
// GPU Bitonic Sort — Init Sort Keys
// ===================================================================

MetalInitSortKeys::MetalInitSortKeys(const std::string& sourceColumn, const std::string& sourceType,
                                     const std::string& sortKeyBuf, const std::string& sortIdxBuf,
                                     const std::string& nResultsExpr, bool descending,
                                     const std::string& capacityExpr)
    : sourceColumn_(sourceColumn), sourceType_(sourceType),
      sortKeyBuf_(sortKeyBuf), sortIdxBuf_(sortIdxBuf),
      nResultsExpr_(nResultsExpr), capacityExpr_(capacityExpr),
      descending_(descending) {}

void MetalInitSortKeys::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Size expressions
    std::string capacity = capacityExpr_.empty() ? nResultsExpr_ : capacityExpr_;
    std::string srckeySize = capacity;
    std::string paddedSize = "next_pow2(" + capacity + ")";

    // Allocate padded sort buffers (zero-init; fills padding with 0xFF for keys, 0 for indices)
    cg.addBufferParam(sortKeyBuf_, "uint64_t", paddedSize, true, 0xFF);
    cg.addBufferParam(sortIdxBuf_, "int", paddedSize, true);
    cg.addScalarParam(nResultsExpr_, "uint");

    // Source buffer (read-only, already allocated by a prior phase)
    cg.addBufferParam(sourceColumn_, sourceType_, srckeySize, false);

    // Key encoding: grid-stride over the source column.  Keep the encoding
    // inline because Metal does not allow helper function declarations inside
    // generated kernel bodies.
    std::string idxVar = "i";

    cg.addBlock("for (uint " + idxVar + " = tid; " + idxVar +
                " < " + nResultsExpr_ + "; " + idxVar + " += tpg)", [&]() {
        std::string valExpr = sourceColumn_ + "[" + idxVar + "]";
        if (sourceType_ == "float") {
            cg.addLine("uint _sort_bits = as_type<uint>(" + valExpr + ");");
            cg.addLine("uint _sort_key32 = ((_sort_bits & 0x80000000u) != 0u) ? "
                       "(~_sort_bits) : (_sort_bits ^ 0x80000000u);");
            cg.addLine("uint64_t _sort_key = uint64_t(_sort_key32);");
        } else {
            cg.addLine("uint64_t _sort_key = uint64_t(uint(" + valExpr + ") ^ 0x80000000u);");
        }
        if (descending_)
            cg.addLine("_sort_key = ~_sort_key;");
        cg.addLine(sortKeyBuf_ + "[" + idxVar + "] = _sort_key;");
        cg.addLine(sortIdxBuf_ + "[" + idxVar + "] = " + idxVar + ";");
    });
    consume(); // no downstream operators
}

unsigned int MetalInitSortKeys::nextPow2(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

PostDispatchHook MetalInitSortKeys::makeBitonicHook(
    const std::string& sortPhaseName,
    const std::string& sortKeyBufName,
    const std::string& sortIdxBufName,
    const std::string& nResultsExpr) {
    return [=](MetalGenericExecutor& executor) {
        auto* pso = executor.getPipelineState(sortPhaseName);
        if (!pso) return;

        auto* keyBuf = executor.getAllocatedBuffer(sortKeyBufName);
        auto* idxBuf = executor.getAllocatedBuffer(sortIdxBufName);
        if (!keyBuf || !idxBuf) return;

        // Read n_results from a registered scalar or symbol
        size_t n_results = 0;
        if (!executor.tryGetSymbol(nResultsExpr, n_results) && n_results == 0) return;
        unsigned int n = (unsigned int)n_results;
        unsigned int np2 = nextPow2(n);

        // Pad sort buffer: fill key slots [n .. np2) with UINT64_MAX,
        //                  fill idx slots [n .. np2) with 0
        if (np2 > n) {
            uint64_t* keys = static_cast<uint64_t*>(keyBuf->contents());
            int* idxs = static_cast<int*>(idxBuf->contents());
            memset(keys + n, 0xFF, (np2 - n) * sizeof(uint64_t));
            memset(idxs + n, 0, (np2 - n) * sizeof(int));
        }

        auto* queue = executor.commandQueue();
        if (!queue) return;

        for (unsigned int k = 2; k <= np2; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                auto* cmdBuf = queue->commandBuffer();
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(pso);

                // Buffer bindings match the sorted shader's [[buffer(N)]] indices:
                // 0: sortKeyBuf, 1: sortIdxBuf, 2: sort_k, 3: sort_j, 4: n_sort
                enc->setBuffer(keyBuf, 0, 0);
                enc->setBuffer(idxBuf, 0, 1);
                enc->setBytes(&k, sizeof(uint), 2);
                enc->setBytes(&j, sizeof(uint), 3);
                enc->setBytes(&np2, sizeof(uint), 4);

                // Grid: ceil(n / 256) threadgroups (only elements 0..n-1 have valid data)
                uint tgSize = pso->maxTotalThreadsPerThreadgroup();
                if (tgSize > 256) tgSize = 256;
                uint numTG = (n + tgSize - 1) / tgSize;
                if (numTG < 1) numTG = 1;
                enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1),
                                          MTL::Size::Make(tgSize, 1, 1));
                enc->endEncoding();
                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();
            }
        }
    };
}

std::string MetalInitSortKeys::describe() const {
    return "InitSortKeys(" + sourceColumn_ + " → " + sortKeyBuf_ + ")";
}

// ===================================================================
// GPU Bitonic Sort — Sort Step (comparison-swap)
// ===================================================================

MetalBitonicSortStep::MetalBitonicSortStep(const std::string& sortKeyBuf,
                                           const std::string& sortIdxBuf,
                                           const std::string& nResultsExpr,
                                           const std::string& capacityExpr)
    : sortKeyBuf_(sortKeyBuf), sortIdxBuf_(sortIdxBuf),
      nResultsExpr_(nResultsExpr), capacityExpr_(capacityExpr) {}

void MetalBitonicSortStep::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Allocate sort buffers (already allocated by init phase; addBitmapReadParam
    // tells the executor to look these up in allocatedBuffers_)
    std::string capacity = capacityExpr_.empty() ? nResultsExpr_ : capacityExpr_;
    std::string paddedSize = "next_pow2(" + capacity + ")";
    cg.addBufferParam(sortKeyBuf_, "uint64_t", paddedSize, false);
    cg.addBufferParam(sortIdxBuf_, "int", paddedSize, false);
    cg.addScalarParam("sort_k", "uint");
    cg.addScalarParam("sort_j", "uint");
    cg.addScalarParam("n_sort", "uint");

    // Bitonic comparison-swap step body (always ascending on encoded keys)
    cg.addLine("uint _i = tid;");
    cg.addLine("uint _ixj = _i ^ sort_j;");
    cg.addLine("if (_ixj > _i && _ixj < n_sort) {");
    cg.addLine("    bool _asc = (_i & sort_k) == 0;");
    cg.addLine("    uint64_t _ki = " + sortKeyBuf_ + "[_i];");
    cg.addLine("    uint64_t _kj = " + sortKeyBuf_ + "[_ixj];");
    cg.addLine("    bool _swap = _asc ? (_ki > _kj) : (_ki < _kj);");
    cg.addBlock("    if (_swap)", [&]() {
        cg.addLine(sortKeyBuf_ + "[_i] = _kj;");
        cg.addLine(sortKeyBuf_ + "[_ixj] = _ki;");
        cg.addLine("int _tmpv = " + sortIdxBuf_ + "[_i];");
        cg.addLine(sortIdxBuf_ + "[_i] = " + sortIdxBuf_ + "[_ixj];");
        cg.addLine(sortIdxBuf_ + "[_ixj] = _tmpv;");
    });
    cg.addLine("}");
    consume();
}

std::string MetalBitonicSortStep::describe() const {
    return "BitonicSortStep(" + sortKeyBuf_ + ")";
}

} // namespace codegen
