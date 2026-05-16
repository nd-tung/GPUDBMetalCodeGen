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

// Display labels for compact one-character TPC-H values.
static const std::unordered_map<std::string, std::unordered_map<char, std::string>> kChar1DisplayMap = {
    {"o_orderpriority", {{'1',"1-URGENT"},{'2',"2-HIGH"},{'3',"3-MEDIUM"},{'4',"4-NOT SPECIFIED"},{'5',"5-LOW"}}},
    {"c_mktsegment", {{'A',"AUTOMOBILE"},{'B',"BUILDING"},{'F',"FURNITURE"},{'M',"MACHINERY"},{'H',"HOUSEHOLD"}}},
    {"l_shipmode", {{'M',"MAIL"},{'S',"SHIP"}}},
};

static std::string char1Display(const std::string& colName, const std::string& value) {
    auto it = kChar1DisplayMap.find(colName);
    if (it == kChar1DisplayMap.end()) return value;
    if (value.empty()) return value;
    auto ci = it->second.find(value[0]);
    if (ci == it->second.end()) return value;
    return ci->second;
}

void GenericResult::print(int limit) const {
    if (columns.empty()) return;

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
    for (auto& w : widths) w = std::max(w, (size_t)10);

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
            // Date columns are stored as YYYYMMDD integers.
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
                    std::string sv = char1Display(colName, v);
                    bool needsQuote = sv.find_first_of(",\"\n\r") != std::string::npos;
                    if (needsQuote) {
                        os << '"';
                        for (char ch : sv) {
                            if (ch == '"') os << '"';
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

int64_t MetalResultCollector::reconstructLong(uint32_t lo, uint32_t hi) {
    uint64_t uval = ((uint64_t)hi << 32) | (uint64_t)lo;
    return static_cast<int64_t>(uval);
}

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
    // Scalar projections support simple aggregate arithmetic for SELECT output.
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

    // Float scalar aggregates are stored through atomic uint buffers.
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

struct HavingContext {
    const GenericResult::Row& row;
    const std::vector<MetalResultSchema::KeyedAggSlot>& slots;
    const std::vector<MetalResultSchema::KeyedAggInfo::MultiKeyInfo>& multiKeys;
    const std::string& keyDisplayName;
    int numMultiKeys;
    int keyBase;
};

// HAVING evaluation reads rows after CPU-side keyed aggregation decoding.
static GenericResult::Value evaluateExpr(const ExprPtr& expr, const HavingContext& ctx);

static bool evaluatePredicate(const PredPtr& pred, const HavingContext& ctx) {
    if (!pred) return true;

    auto& node = pred->node;

    if (auto* cmp = std::get_if<Comparison>(&node)) {
        auto lval = evaluateExpr(cmp->left, ctx);
        auto rval = evaluateExpr(cmp->right, ctx);

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

        if (!lIsDouble || !rIsDouble) return true;

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

    if (auto* land = std::get_if<LogicalAnd>(&node)) {
        for (const auto& child : land->children) {
            if (!evaluatePredicate(child, ctx)) return false;
        }
        return true;
    }

    if (auto* lor = std::get_if<LogicalOr>(&node)) {
        for (const auto& child : lor->children) {
            if (evaluatePredicate(child, ctx)) return true;
        }
        return false;
    }

    if (auto* lnot = std::get_if<LogicalNot>(&node)) {
        return !evaluatePredicate(lnot->child, ctx);
    }

    // Unsupported HAVING predicates are pass-through.
    return true;
}

static GenericResult::Value evaluateExpr(const ExprPtr& expr, const HavingContext& ctx) {
    if (!expr) return (int64_t)0;

    auto& node = expr->node;

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

    if (auto* col = std::get_if<ColRef>(&node)) {
        if (!ctx.multiKeys.empty()) {
            for (size_t ki = 0; ki < ctx.multiKeys.size(); ++ki) {
                if (ctx.multiKeys[ki].displayName == col->column) {
                    if (ki < ctx.row.size()) {
                        return ctx.row[ki];
                    }
                }
            }
        } else {
            if (!ctx.row.empty()) {
                return ctx.row[0];
            }
        }
        return (int64_t)0;
    }

    if (auto* call = std::get_if<FuncCall>(&node)) {
        std::string funcName = call->name;
        
        for (auto& c : funcName) c = std::toupper((unsigned char)c);

        for (size_t si = 0; si < ctx.slots.size(); ++si) {
            const auto& slot = ctx.slots[si];
            std::string slotName = slot.name;
            for (auto& c : slotName) c = std::toupper((unsigned char)c);

            if (slotName.find(funcName) == 0) {
                int valueIdx = ctx.numMultiKeys + si;
                if (valueIdx < (int)ctx.row.size()) {
                    return ctx.row[valueIdx];
                }
            }
        }

        return (int64_t)0;
    }

    if (auto* binexpr = std::get_if<BinaryExpr>(&node)) {
        auto lval = evaluateExpr(binexpr->left, ctx);
        auto rval = evaluateExpr(binexpr->right, ctx);

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

    if (auto* casewhen = std::get_if<CaseWhen>(&node)) {
        for (const auto& branch : casewhen->branches) {
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

    if (!multiKeys.empty()) {
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
            // AVG stores SUM followed by COUNT.
            if (slot.scaleDown < 0 && si + 1 < slots.size()) si++;
        }
    } else {
        for (int v = 0; v < valuesPerBucket; v++) {
            result.columns.push_back({"val_" + std::to_string(v), "uint"});
        }
    }

    auto decodeKeys = [&](int bucket, std::vector<GenericResult::Value>& keyValues) {
        if (multiKeys.empty()) {
            keyValues.push_back((int64_t)(bucket + schema.keyedAgg.keyBase));
            return;
        }
        // bucket = k0 + k1*N0 + k2*N0*N1 + ...
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

    for (int bucket = 0; bucket < numBuckets; bucket++) {
        const int rowBase = bucket * valuesPerBucket;
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
                // AVG stores SUM followed by COUNT.
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
                    si++;
                } else if (slot.isMinMax && slot.atomicOp == "min") {
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
                continue;
            }
        }

        result.rows.push_back(std::move(row));
    }

    return result;
}

GenericResult MetalResultCollector::collectMaterialize(const MetalResultSchema& schema,
                                                       const BufferMap& buffers) {
    GenericResult result;

    auto cntIt = buffers.find(schema.counterBuffer);
    if (cntIt == buffers.end()) return result;
    uint32_t rowCount = *static_cast<uint32_t*>(cntIt->second->contents());

    for (const auto& col : schema.columns) {
        result.columns.push_back({col.displayName, col.elementType});
    }

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
