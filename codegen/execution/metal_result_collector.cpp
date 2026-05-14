#include "metal_result_collector.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_map>

namespace codegen {

// CHAR1 display map: maps a single-char TPC-H CHAR1 value to its full display string.
static const std::unordered_map<std::string, std::unordered_map<char, std::string>> kChar1DisplayMap = {
    {"o_orderpriority", {{'1',"1-URGENT"},{'2',"2-HIGH"},{'3',"3-MEDIUM"},{'4',"4-NOT SPECIFIED"},{'5',"5-LOW"}}},
    {"c_mktsegment", {{'A',"AUTOMOBILE"},{'B',"BUILDING"},{'F',"FURNITURE"},{'M',"MACHINERY"},{'H',"HOUSEHOLD"}}},
    {"l_shipmode", {{'M',"MAIL"},{'S',"SHIP"}}}, // l_shipmode is CHAR_FIXED(2) but stored first char
};

static std::string char1Display(const std::string& colName, const std::string& value) {
    auto it = kChar1DisplayMap.find(colName);
    if (it == kChar1DisplayMap.end()) return value;
    if (value.empty()) return value;
    auto ci = it->second.find(value[0]);
    if (ci == it->second.end()) return value;
    return ci->second;
}

// ===================================================================
// GenericResult::print
// ===================================================================

void GenericResult::print(int limit) const {
    if (columns.empty()) return;

    // Compute column widths
    std::vector<size_t> widths(columns.size());
    for (size_t c = 0; c < columns.size(); c++)
        widths[c] = columns[c].name.size();

    size_t rowCount = (limit > 0 && (size_t)limit < rows.size()) ? (size_t)limit : rows.size();
    for (size_t r = 0; r < rowCount; r++) {
        for (size_t c = 0; c < columns.size() && c < rows[r].size(); c++) {
            size_t w = 0;
            std::visit([&w](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>)
                    w = v.size();
                else
                    w = std::to_string(v).size();
            }, rows[r][c]);
            widths[c] = std::max(widths[c], w + 2);
        }
    }
    // Minimum width
    for (auto& w : widths) w = std::max(w, (size_t)10);

    // Print header separator
    auto printSep = [&]() {
        std::cout << "+";
        for (size_t c = 0; c < columns.size(); c++) {
            std::cout << std::string(widths[c] + 2, '-') << "+";
        }
        std::cout << "\n";
    };

    printSep();
    std::cout << "|";
    for (size_t c = 0; c < columns.size(); c++) {
        std::cout << " " << std::setw((int)widths[c]) << std::right << columns[c].name << " |";
    }
    std::cout << "\n";
    printSep();

    // Print rows
    for (size_t r = 0; r < rowCount; r++) {
        std::cout << "|";
        for (size_t c = 0; c < columns.size() && c < rows[r].size(); c++) {
            std::cout << " ";
            const std::string& colName = columns[c].name;
            bool isDateCol = colName.size() >= 4 &&
                             colName.substr(colName.size() - 4) == "date";
            std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                    if (isDateCol) {
                        int d = static_cast<int>(v);
                        char buf[12];
                        snprintf(buf, sizeof(buf), "%04d-%02d-%02d", d / 10000, (d / 100) % 100, d % 100);
                        std::cout << std::setw((int)widths[c]) << std::right << buf;
                    } else {
                        std::cout << std::setw((int)widths[c]) << std::right << v;
                    }
                }                 else if constexpr (std::is_same_v<T, double>)
                    std::cout << std::setw((int)widths[c]) << std::right << std::fixed << std::setprecision(4) << v;
                else {
                    std::string display = char1Display(colName, v);
                    std::cout << std::setw((int)widths[c]) << std::right << display;
                }
            }, rows[r][c]);
            std::cout << " |";
        }
        std::cout << "\n";
    }
    printSep();

    if (limit > 0 && (size_t)limit < rows.size())
        std::cout << "... (" << rows.size() - limit << " more rows)\n";
}

// ===================================================================
// GenericResult::toCanonical
// ===================================================================

std::string GenericResult::toCanonical() const {
    std::ostringstream os;
    for (size_t c = 0; c < columns.size(); c++) {
        if (c) os << ",";
        os << columns[c].name;
    }
    os << "\n";
    for (const auto& row : rows) {
        for (size_t c = 0; c < row.size(); c++) {
            if (c) os << ",";
            const std::string& colName = columns[c].name;
            // Detect date columns by name suffix (e.g. o_orderdate, l_shipdate)
            bool isDateCol = colName.size() >= 4 &&
                             colName.substr(colName.size() - 4) == "date";
            std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                    if (isDateCol) {
                        int d = static_cast<int>(v);
                        char buf[12];
                        snprintf(buf, sizeof(buf), "%04d-%02d-%02d", d / 10000, (d / 100) % 100, d % 100);
                        os << buf;
                    } else {
                        os << v;
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "%.4f", v);
                    os << buf;
                } else {
                    // Quote string fields that contain commas, quotes, or newlines
                    std::string sv = char1Display(colName, v);
                    bool needsQuote = sv.find_first_of(",\"\n\r") != std::string::npos;
                    if (needsQuote) {
                        os << '"';
                        for (char ch : sv) {
                            if (ch == '"') os << '"'; // escape embedded quote
                            os << ch;
                        }
                        os << '"';
                    } else {
                        os << sv;
                    }
                }
            }, row[c]);
        }
        os << "\n";
    }
    return os.str();
}

// ===================================================================
// Long reconstruction
// ===================================================================

int64_t MetalResultCollector::reconstructLong(uint32_t lo, uint32_t hi) {
    uint64_t uval = ((uint64_t)hi << 32) | (uint64_t)lo;
    return static_cast<int64_t>(uval);
}

// ===================================================================
// collect — dispatch based on schema kind
// ===================================================================

GenericResult MetalResultCollector::collect(const MetalResultSchema& schema,
                                            const BufferMap& buffers) {
    switch (schema.kind) {
        case MetalResultSchema::SCALAR_AGG:
            return collectScalarAgg(schema, buffers);
        case MetalResultSchema::KEYED_AGG:
            return collectKeyedAgg(schema, buffers);
        case MetalResultSchema::MATERIALIZE:
            return collectMaterialize(schema, buffers);
        case MetalResultSchema::NONE:
        default:
            return {};
    }
}

// ===================================================================
// collectScalarAgg
// ===================================================================

static double scalarValueAsDouble(const GenericResult::Value& value) {
    double out = 0.0;
    std::visit([&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, double>) out = v;
        else if constexpr (std::is_same_v<T, int64_t>) out = static_cast<double>(v);
    }, value);
    return out;
}

static bool isAggregateFuncName(std::string name) {
    for (auto& c : name) c = (char)std::tolower((unsigned char)c);
    return name == "sum" || name == "count" || name == "avg" ||
           name == "min" || name == "max";
}

static GenericResult::Value evaluateScalarAggProjection(const ExprPtr& expr,
                                                        double aggregateValue) {
    if (!expr) return aggregateValue;

    if (auto* lit = std::get_if<Literal>(&expr->node)) {
        if (auto* i = std::get_if<int>(&lit->value)) return (int64_t)*i;
        if (auto* f = std::get_if<float>(&lit->value)) return (double)*f;
        if (auto* s = std::get_if<std::string>(&lit->value)) return *s;
        return (int64_t)0;
    }

    if (auto* call = std::get_if<FuncCall>(&expr->node)) {
        if (isAggregateFuncName(call->name)) return aggregateValue;
        return (int64_t)0;
    }

    if (auto* bin = std::get_if<BinaryExpr>(&expr->node)) {
        double l = scalarValueAsDouble(evaluateScalarAggProjection(bin->left, aggregateValue));
        double r = scalarValueAsDouble(evaluateScalarAggProjection(bin->right, aggregateValue));
        switch (bin->op) {
            case ExprOp::ADD: return l + r;
            case ExprOp::SUB: return l - r;
            case ExprOp::MUL: return l * r;
            case ExprOp::DIV: return r != 0.0 ? l / r : 0.0;
        }
    }

    return (int64_t)0;
}

GenericResult MetalResultCollector::collectScalarAgg(const MetalResultSchema& schema,
                                                     const BufferMap& buffers) {
    GenericResult result;
    GenericResult::Row row;

    auto readScalar = [&](const std::string& loBuffer,
                          const std::string& hiBuffer,
                          bool isLongPair) -> double {
        if (isLongPair) {
            auto loIt = buffers.find(loBuffer);
            auto hiIt = buffers.find(hiBuffer);
            if (loIt != buffers.end() && hiIt != buffers.end()) {
                uint32_t lo = *static_cast<uint32_t*>(loIt->second->contents());
                uint32_t hi = *static_cast<uint32_t*>(hiIt->second->contents());
                return static_cast<double>(reconstructLong(lo, hi));
            }
            return 0.0;
        }

        auto it = buffers.find(loBuffer);
        if (it == buffers.end()) return 0.0;
        if (loBuffer.empty()) return 0.0;
        if (!isLongPair) {
            auto schemaIt = std::find_if(schema.scalarAggs.begin(), schema.scalarAggs.end(),
                [&](const auto& candidate) { return candidate.loBuffer == loBuffer; });
            if (schemaIt != schema.scalarAggs.end() && schemaIt->elementType == "int") {
                return static_cast<double>(*static_cast<int32_t*>(it->second->contents()));
            }
        }
        uint32_t raw = *static_cast<uint32_t*>(it->second->contents());
        float fval;
        memcpy(&fval, &raw, sizeof(float));
        return static_cast<double>(fval);
    };

    for (const auto& entry : schema.scalarAggs) {
        result.columns.push_back({entry.displayName, entry.isLongPair ? "long" : entry.elementType});

        double value = readScalar(entry.loBuffer, entry.hiBuffer, entry.isLongPair);
        if (entry.divideByDenominator) {
            double denominator = readScalar(entry.denomLoBuffer, entry.denomHiBuffer,
                                           entry.denomIsLongPair);
            value = denominator != 0.0 ? value / denominator : 0.0;
        }

        // Apply scale-down (divide by the scaleDown factor, e.g. 100 → /100)
        if (entry.scaleDown > 0) {
            value /= static_cast<double>(entry.scaleDown);
        }

        if (entry.projectionExpr) {
            row.push_back(evaluateScalarAggProjection(entry.projectionExpr, value));
        } else {
            row.push_back(value);
        }
    }

    if (!row.empty())
        result.rows.push_back(std::move(row));

    return result;
}

// ===================================================================
// HAVING Predicate Evaluation
// ===================================================================

// Context for evaluating HAVING predicates over aggregated results
struct HavingContext {
    const GenericResult::Row&                row;              // decoded keys + agg values
    const std::vector<MetalResultSchema::KeyedAggSlot>& slots; // agg slot metadata
    const std::vector<MetalResultSchema::KeyedAggInfo::MultiKeyInfo>& multiKeys;
    const std::string& keyDisplayName;
    int numMultiKeys;  // count of GROUP BY keys (at front of row)
    int keyBase;
};

// Forward declare evaluateExpr
static GenericResult::Value evaluateExpr(const ExprPtr& expr, const HavingContext& ctx);

// Evaluate a predicate tree, returning true if the row satisfies it
static bool evaluatePredicate(const PredPtr& pred, const HavingContext& ctx) {
    if (!pred) return true;  // null predicate always true

    auto& node = pred->node;

    // Comparison: left CMP right
    if (auto* cmp = std::get_if<Comparison>(&node)) {
        auto lval = evaluateExpr(cmp->left, ctx);
        auto rval = evaluateExpr(cmp->right, ctx);

        // Extract comparable values
        double l = 0, r = 0;
        bool lIsDouble = false, rIsDouble = false;

        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) { l = v; lIsDouble = true; }
            else if constexpr (std::is_same_v<T, int64_t>) { l = (double)v; lIsDouble = true; }
            else if constexpr (std::is_same_v<T, std::string>) { /* skip */ }
        }, lval);

        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) { r = v; rIsDouble = true; }
            else if constexpr (std::is_same_v<T, int64_t>) { r = (double)v; rIsDouble = true; }
            else if constexpr (std::is_same_v<T, std::string>) { /* skip */ }
        }, rval);

        if (!lIsDouble || !rIsDouble) return true;  // non-numeric comparison: skip

        bool result = false;
        switch (cmp->op) {
            case CmpOp::EQ: result = (l == r); break;
            case CmpOp::NE: result = (l != r); break;
            case CmpOp::LT: result = (l < r); break;
            case CmpOp::LE: result = (l <= r); break;
            case CmpOp::GT: result = (l > r); break;
            case CmpOp::GE: result = (l >= r); break;
        }
        return result;
    }

    // LogicalAnd: all children must be true
    if (auto* land = std::get_if<LogicalAnd>(&node)) {
        for (const auto& child : land->children) {
            if (!evaluatePredicate(child, ctx)) return false;
        }
        return true;
    }

    // LogicalOr: any child being true means true
    if (auto* lor = std::get_if<LogicalOr>(&node)) {
        for (const auto& child : lor->children) {
            if (evaluatePredicate(child, ctx)) return true;
        }
        return false;
    }

    // LogicalNot: negate the child
    if (auto* lnot = std::get_if<LogicalNot>(&node)) {
        return !evaluatePredicate(lnot->child, ctx);
    }

    // Other predicate types (Between, InList, Like, ExistsPred): not supported in HAVING
    // For now, return true (allow the row)
    return true;
}

// Evaluate an expression in the context of aggregated row data
static GenericResult::Value evaluateExpr(const ExprPtr& expr, const HavingContext& ctx) {
    if (!expr) return (int64_t)0;

    auto& node = expr->node;

    // Literal: return directly
    if (auto* lit = std::get_if<Literal>(&node)) {
        if (auto* i = std::get_if<int>(&lit->value)) {
            return (int64_t)*i;
        }
        if (auto* f = std::get_if<float>(&lit->value)) {
            return (double)*f;
        }
        if (auto* s = std::get_if<std::string>(&lit->value)) {
            return *s;
        }
        return (int64_t)0;
    }

    // ColRef: look up from GROUP BY keys (first N positions in row)
    if (auto* col = std::get_if<ColRef>(&node)) {
        if (!ctx.multiKeys.empty()) {
            // Multi-key: look up by matching display name
            for (size_t ki = 0; ki < ctx.multiKeys.size(); ++ki) {
                if (ctx.multiKeys[ki].displayName == col->column) {
                    if (ki < ctx.row.size()) {
                        return ctx.row[ki];
                    }
                }
            }
        } else {
            // Single key: always at position 0
            if (!ctx.row.empty()) {
                return ctx.row[0];
            }
        }
        return (int64_t)0;
    }

    // FuncCall: handle aggregate functions
    if (auto* call = std::get_if<FuncCall>(&node)) {
        std::string funcName = call->name;
        
        // Convert function name to uppercase for comparison
        for (auto& c : funcName) c = std::toupper((unsigned char)c);

        // Special handling for SUM(col), COUNT(*), etc.
        // The function name might be "SUM", "COUNT", "AVG", "MIN", "MAX"
        // In the slots, the name is like "SUM(o_totalprice)" or "COUNT(*)"
        // We need to match by looking for slots that start with the function name
        
        for (size_t si = 0; si < ctx.slots.size(); ++si) {
            const auto& slot = ctx.slots[si];
            std::string slotName = slot.name;
            for (auto& c : slotName) c = std::toupper((unsigned char)c);

            // Check if this slot matches the aggregate function
            if (slotName.find(funcName) == 0) {
                int valueIdx = ctx.numMultiKeys + si;
                if (valueIdx < (int)ctx.row.size()) {
                    return ctx.row[valueIdx];
                }
            }
        }

        // If not found, return 0
        return (int64_t)0;
    }

    // BinaryExpr: evaluate both sides and apply operator
    if (auto* binexpr = std::get_if<BinaryExpr>(&node)) {
        auto lval = evaluateExpr(binexpr->left, ctx);
        auto rval = evaluateExpr(binexpr->right, ctx);

        // Convert to numbers
        double l = 0, r = 0;
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) l = v;
            else if constexpr (std::is_same_v<T, int64_t>) l = (double)v;
        }, lval);
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) r = v;
            else if constexpr (std::is_same_v<T, int64_t>) r = (double)v;
        }, rval);

        double result = 0;
        switch (binexpr->op) {
            case ExprOp::ADD: result = l + r; break;
            case ExprOp::SUB: result = l - r; break;
            case ExprOp::MUL: result = l * r; break;
            case ExprOp::DIV: result = (r != 0) ? l / r : 0; break;
        }
        return result;
    }

    // CaseWhen: not typically in HAVING, return 0
    if (auto* casewhen = std::get_if<CaseWhen>(&node)) {
        for (const auto& branch : casewhen->branches) {
            // For simplicity, just evaluate first branch (not a true CASE evaluation)
            if (branch.result) {
                return evaluateExpr(branch.result, ctx);
            }
        }
        if (casewhen->elseResult) {
            return evaluateExpr(casewhen->elseResult, ctx);
        }
        return (int64_t)0;
    }

    return (int64_t)0;
}

// ===================================================================
// collectKeyedAgg
// ===================================================================

GenericResult MetalResultCollector::collectKeyedAgg(const MetalResultSchema& schema,
                                                    const BufferMap& buffers) {
    GenericResult result;

    auto it = buffers.find(schema.keyedAgg.bufferName);
    if (it == buffers.end()) return result;

    const auto* data = static_cast<const uint32_t*>(it->second->contents());
    const int numBuckets       = schema.keyedAgg.numBuckets;
    const int valuesPerBucket  = schema.keyedAgg.valuesPerBucket;
    const auto& slots = schema.keyedAgg.slots;
    const auto& multiKeys = schema.keyedAgg.multiKeys;

    // --- Build column headers ---
    if (!multiKeys.empty()) {
        // Multi-key: emit one column per key
        for (const auto& mk : multiKeys) {
            std::string type = mk.charMap.empty() ? "int" : "string";
            result.columns.push_back({mk.displayName, type});
        }
    } else {
        std::string keyName = schema.keyedAgg.keyDisplayName.empty()
            ? "bucket"
            : schema.keyedAgg.keyDisplayName;
        result.columns.push_back({keyName, "int"});
    }
    if (!slots.empty()) {
        for (size_t si = 0; si < slots.size(); ++si) {
            const auto& slot = slots[si];
            result.columns.push_back({slot.name, slot.isLongPair ? "long" : "uint"});
            // AVG sum with scaleDown < 0 → skip the following COUNT slot
            if (slot.scaleDown < 0 && si + 1 < slots.size()) si++;
        }
    } else {
        for (int v = 0; v < valuesPerBucket; v++) {
            result.columns.push_back({"val_" + std::to_string(v), "uint"});
        }
    }

    // --- Helper: decode a flat bucket index into multi-key values ---
    auto decodeKeys = [&](int bucket, std::vector<GenericResult::Value>& keyValues) {
        if (multiKeys.empty()) {
            keyValues.push_back((int64_t)(bucket + schema.keyedAgg.keyBase));
            return;
        }
        // Decode in REVERSE order: bucket = k0 + k1*N0 + k2*N0*N1 + ...
        // So last key = bucket / stride[last]; recurse with remainder.
        int remaining = bucket;
        std::vector<int> encodedValues(multiKeys.size());
        for (int ki = (int)multiKeys.size() - 1; ki >= 0; --ki) {
            const auto& mk = multiKeys[ki];
            encodedValues[ki] = remaining / mk.stride;
            remaining = remaining % mk.stride;
        }
        for (size_t ki = 0; ki < multiKeys.size(); ++ki) {
            const auto& mk = multiKeys[ki];
            int encoded = encodedValues[ki];
            if (!mk.charMap.empty()) {
                if (encoded >= 0 && (size_t)encoded < mk.charMap.size()) {
                    keyValues.push_back(std::string(1, mk.charMap[encoded]));
                } else {
                    keyValues.push_back(std::string("?"));
                }
            } else {
                keyValues.push_back((int64_t)(encoded + mk.keyBase));
            }
        }
    };

    // --- Read per-bucket data ---
    for (int bucket = 0; bucket < numBuckets; bucket++) {
        const int rowBase = bucket * valuesPerBucket;
        // Skip empty buckets (all slots zero)
        bool hasData = false;
        for (int v = 0; v < valuesPerBucket; v++) {
            if (data[rowBase + v] != 0) { hasData = true; break; }
        }
        if (!hasData) continue;

        GenericResult::Row row;
        decodeKeys(bucket, row);

        if (!slots.empty()) {
            for (size_t si = 0; si < slots.size(); ++si) {
                const auto& slot = slots[si];
                // AVG decomposition: scaleDown < 0 marks a SUM slot whose COUNT follows.
                if (slot.scaleDown < 0 && si + 1 < slots.size()) {
                    const auto& cntSlot = slots[si + 1];
                    double sumVal = 0, cntVal = 0;
                    if (slot.isFloatSum) {
                        uint32_t raw = data[rowBase + slot.offset];
                        float f; memcpy(&f, &raw, sizeof(float));
                        sumVal = (double)f;
                    } else if (slot.isLongPair) {
                        uint32_t lo = data[rowBase + slot.offset];
                        uint32_t hi = data[rowBase + slot.offset + 1];
                        sumVal = (double)(((int64_t)hi << 32) | (int64_t)lo);
                    } else {
                        sumVal = (double)(int64_t)data[rowBase + slot.offset];
                    }
                    if (slot.scaleDown < -1)
                        sumVal /= static_cast<double>(-slot.scaleDown);
                    if (cntSlot.isFloatSum) {
                        uint32_t raw = data[rowBase + cntSlot.offset];
                        float f; memcpy(&f, &raw, sizeof(float));
                        cntVal = (double)f;
                    } else {
                        cntVal = (double)(int64_t)data[rowBase + cntSlot.offset];
                    }
                    row.push_back(cntVal > 0 ? sumVal / cntVal : 0.0);
                    si++; // skip COUNT slot
                } else if (slot.isMinMax && slot.atomicOp == "min") {
                    // min aggregate: stored as raw value (int or float reinterpreted)
                    uint32_t raw = data[rowBase + slot.offset];
                    if (slot.isFloatSum) {
                        float f;
                        memcpy(&f, &raw, sizeof(float));
                        row.push_back((double)f);
                    } else {
                        row.push_back((int64_t)(int32_t)raw);
                    }
                } else if (slot.isMinMax && slot.atomicOp == "max") {
                    uint32_t raw = data[rowBase + slot.offset];
                    if (slot.isFloatSum) {
                        float f;
                        memcpy(&f, &raw, sizeof(float));
                        row.push_back((double)f);
                    } else {
                        row.push_back((int64_t)(int32_t)raw);
                    }
                } else if (slot.isFloatSum) {
                    // Float sum: stored as float in single uint slot
                    uint32_t raw = data[rowBase + slot.offset];
                    float f;
                    memcpy(&f, &raw, sizeof(float));
                    if (slot.scaleDown > 0)
                        row.push_back((double)f / slot.scaleDown);
                    else
                        row.push_back((double)f);
                } else if (slot.isLongPair) {
                    uint32_t lo = data[rowBase + slot.offset];
                    uint32_t hi = data[rowBase + slot.offset + 1];
                    int64_t val = ((int64_t)hi << 32) | (int64_t)lo;
                    if (slot.scaleDown > 0)
                        row.push_back((double)val / slot.scaleDown);
                    else
                        row.push_back(val);
                } else {
                    int64_t val = (int64_t)data[rowBase + slot.offset];
                    if (slot.scaleDown > 0)
                        row.push_back((double)val / slot.scaleDown);
                    else
                        row.push_back(val);
                }
            }
        } else {
            for (int v = 0; v < valuesPerBucket; v++) {
                row.push_back((int64_t)data[rowBase + v]);
            }
        }

        // Apply HAVING predicate filter if present and not already evaluated on GPU
        if (schema.keyedAgg.havingPredicate && !schema.keyedAgg.havingEvaluatedOnGPU) {
            HavingContext ctx{
                row,
                slots,
                multiKeys,
                schema.keyedAgg.keyDisplayName,
                static_cast<int>(multiKeys.empty() ? 1 : multiKeys.size()),
                schema.keyedAgg.keyBase
            };

            if (!evaluatePredicate(schema.keyedAgg.havingPredicate, ctx)) {
                // Row does not satisfy HAVING: skip it
                continue;
            }
        }

        result.rows.push_back(std::move(row));
    }

    return result;
}

// ===================================================================
// collectMaterialize
// ===================================================================

GenericResult MetalResultCollector::collectMaterialize(const MetalResultSchema& schema,
                                                       const BufferMap& buffers) {
    GenericResult result;

    // Get row count
    auto cntIt = buffers.find(schema.counterBuffer);
    if (cntIt == buffers.end()) return result;
    uint32_t rowCount = *static_cast<uint32_t*>(cntIt->second->contents());

    // Build columns
    for (const auto& col : schema.columns) {
        result.columns.push_back({col.displayName, col.elementType});
    }

    // Read rows
    for (uint32_t r = 0; r < rowCount; r++) {
        GenericResult::Row row;
        for (const auto& col : schema.columns) {
            auto bIt = buffers.find(col.bufferName);
            if (bIt == buffers.end()) {
                row.push_back((int64_t)0);
                continue;
            }

            if (col.isLongPair) {
                const auto* arr = static_cast<const uint32_t*>(bIt->second->contents());
                uint32_t lo = arr[r * 2];
                uint32_t hi = arr[r * 2 + 1];
                int64_t v = ((int64_t)hi << 32) | (int64_t)lo;
                if (col.scaleDown > 0)
                    row.push_back(static_cast<double>(v) / col.scaleDown);
                else
                    row.push_back(v);
            } else if (col.elementType == "float") {
                const auto* arr = static_cast<const float*>(bIt->second->contents());
                row.push_back(static_cast<double>(arr[r]));
            } else if (col.elementType == "int" || col.elementType == "uint") {
                const auto* arr = static_cast<const uint32_t*>(bIt->second->contents());
                int64_t v = static_cast<int64_t>(arr[r]);
                if (col.scaleDown > 0)
                    row.push_back(static_cast<double>(v) / col.scaleDown);
                else
                    row.push_back(v);
            } else if (col.elementType == "long") {
                const auto* arr = static_cast<const int64_t*>(bIt->second->contents());
                if (col.scaleDown > 0)
                    row.push_back(static_cast<double>(arr[r]) / col.scaleDown);
                else
                    row.push_back(static_cast<int64_t>(arr[r]));
            } else if (col.elementType == "char") {
                if (col.stringLen > 0) {
                    const auto* arr = static_cast<const char*>(bIt->second->contents());
                    std::string s(arr + r * col.stringLen, col.stringLen);
                    while (!s.empty() && s.back() == '\0')
                        s.pop_back();
                    row.push_back(std::move(s));
                } else {
                    const auto* arr = static_cast<const char*>(bIt->second->contents());
                    row.push_back(std::string(1, arr[r]));
                }
            } else {
                const auto* arr = static_cast<const uint32_t*>(bIt->second->contents());
                row.push_back(static_cast<int64_t>(arr[r]));
            }
        }
        result.rows.push_back(std::move(row));
    }

    return result;
}

} // namespace codegen
