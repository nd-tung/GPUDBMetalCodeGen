#include "metal_generic_adhoc_builder.h"
#include "metal_plan_common.h"
#include "metal_generic_executor.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace codegen {

namespace {

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front()))) {
        name = "c_" + name;
    }
    return name;
}

std::string aggName(AggFunc func) {
    switch (func) {
        case AggFunc::SUM: return "sum";
        case AggFunc::COUNT: return "count";
        case AggFunc::AVG: return "avg";
        case AggFunc::MIN: return "min";
        case AggFunc::MAX: return "max";
        case AggFunc::COUNT_DISTINCT: return "count_distinct";
    }
    return "agg";
}

std::string columnNameForExpr(const ExprPtr& expr) {
    if (!expr) return "expr";
    if (auto* col = std::get_if<ColRef>(&expr->node)) return col->column;
    if (auto* func = std::get_if<FuncCall>(&expr->node)) return func->name;
    return "expr";
}

std::string displayNameForTarget(const SelectTarget& target, size_t targetIndex) {
    if (!target.alias.empty()) return target.alias;
    if (target.isAgg && target.agg) {
        std::string base = target.agg->isStar ? "star" : columnNameForExpr(target.agg->innerExpr);
        return aggName(target.agg->func) + "_" + base;
    }
    if (target.expr && std::holds_alternative<ColRef>(target.expr->node)) {
        return columnNameForExpr(target.expr);
    }
    return "expr_" + std::to_string(targetIndex);
}

// Find a target's display name by matching a ColRef (for GROUP BY columns).
std::string displayNameForTargetByCol(const AnalyzedQuery& aq, const ColRef& ref) {
    for (size_t i = 0; i < aq.targets.size(); ++i) {
        const auto& t = aq.targets[i];
        if (!t.isAgg && t.expr && std::holds_alternative<ColRef>(t.expr->node)) {
            auto* col = std::get_if<ColRef>(&t.expr->node);
            if (col->column == ref.column && col->table == ref.table)
                return displayNameForTarget(t, i);
        }
    }
    // No matching SELECT target — use the column name directly.
    return ref.column;
}

std::string aggFuncName(AggFunc func) {
    switch (func) {
        case AggFunc::SUM: return "SUM";
        case AggFunc::COUNT: return "COUNT";
        case AggFunc::AVG: return "AVG";
        case AggFunc::MIN: return "MIN";
        case AggFunc::MAX: return "MAX";
        case AggFunc::COUNT_DISTINCT: return "COUNT_DISTINCT";
        default: return "SUM";
    }
}

bool exprIsColumn(const ExprPtr& expr, const ColRef& expected) {
    auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr;
    return col && col->table == expected.table && col->column == expected.column;
}

std::optional<GroupDomain> smallIntGroupDomain(const ColRef& col, const SchemaProvider* schema = nullptr) {
    if (col.dataType != DataType::INT && col.dataType != DataType::DATE) return std::nullopt;
    if (schema && col.table.size()) {
        auto d = schema->groupDomain(col.table, col.column);
        if (d) return d;
    }
    return std::nullopt;
}

std::optional<std::string> orderColumnForExpr(const ExprPtr& expr,
                                              const std::vector<SelectTarget>& targets) {
    if (!expr) return std::nullopt;
    if (auto* lit = std::get_if<Literal>(&expr->node)) {
        if (auto* ordinal = std::get_if<int>(&lit->value)) {
            if (*ordinal >= 1 && static_cast<size_t>(*ordinal) <= targets.size()) {
                return displayNameForTarget(targets[*ordinal - 1], static_cast<size_t>(*ordinal - 1));
            }
        }
        return std::nullopt;
    }
    auto* orderCol = std::get_if<ColRef>(&expr->node);
    if (orderCol) {
        for (size_t i = 0; i < targets.size(); ++i) {
            const auto& target = targets[i];
            std::string displayName = displayNameForTarget(target, i);
            if (displayName == orderCol->column) return displayName;
            if (auto* targetCol = target.expr ? std::get_if<ColRef>(&target.expr->node) : nullptr) {
                if (targetCol->column == orderCol->column) return displayName;
            }
        }
    }
    // Non-ColRef ORDER BY (e.g. FuncCall after subquery alias resolution):
    // match by position — the ORDER BY expression is typically the Nth target.
    for (size_t i = 0; i < targets.size(); ++i) {
        std::string dn = displayNameForTarget(targets[i], i);
        if (!orderCol) return dn;  // first non-ColRef match
    }
    return std::nullopt;
}

std::string metalTypeForDataType(DataType type) {
    switch (type) {
        case DataType::INT:
        case DataType::DATE: return "int";
        case DataType::FLOAT: return "float";
        case DataType::CHAR1: return "char";
        case DataType::CHAR_FIXED: return "char";
    }
    return "int";
}

DataType inferExprDataType(const ExprPtr& expr) {
    if (!expr) return DataType::INT;
    return std::visit([&](auto&& node) -> DataType {
        using Node = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<Node, ColRef>) {
            return node.dataType;
        } else if constexpr (std::is_same_v<Node, Literal>) {
            if (std::holds_alternative<float>(node.value)) return DataType::FLOAT;
            if (std::holds_alternative<std::string>(node.value)) return DataType::CHAR_FIXED;
            return DataType::INT;
        } else if constexpr (std::is_same_v<Node, BinaryExpr>) {
            DataType leftType = inferExprDataType(node.left);
            DataType rightType = inferExprDataType(node.right);
            if (node.op == ExprOp::DIV || leftType == DataType::FLOAT || rightType == DataType::FLOAT)
                return DataType::FLOAT;
            return DataType::INT;
        } else if constexpr (std::is_same_v<Node, CaseWhen>) {
            // Return type of first branch result, or elseResult, or INT.
            if (!node.branches.empty()) return inferExprDataType(node.branches[0].result);
            if (node.elseResult) return inferExprDataType(node.elseResult);
            return DataType::INT;
        } else if constexpr (std::is_same_v<Node, FuncCall>) {
            if (node.name == "date_part" || node.name == "extract") return DataType::INT;
            if (node.name == "substring") return DataType::INT;
            return DataType::INT;
        } else {
            return DataType::INT;
        }
    }, expr->node);
}

bool isNumericLike(DataType type) {
    return type == DataType::INT || type == DataType::FLOAT || type == DataType::DATE;
}

bool isDateLiteralString(const std::string& value) {
    return value.size() >= 10 && value[4] == '-' && value[7] == '-';
}

bool literalMatchesType(const ExprPtr& expr, DataType type) {
    if (!expr) return false;
    auto* lit = std::get_if<Literal>(&expr->node);
    if (!lit) return false;
    auto* value = std::get_if<std::string>(&lit->value);
    if (!value) return true; // int or float literal always matches
    if (type == DataType::DATE) return isDateLiteralString(*value);
    if (type == DataType::CHAR1) return value->size() == 1;
    if (type == DataType::INT || type == DataType::FLOAT) {
        // String literals can match numeric types if they represent numbers
        try { (void)std::stoi(*value); return true; } catch (...) {}
        try { (void)std::stof(*value); return true; } catch (...) {}
    }
    return false;
}

bool exprSupported(const ExprPtr& expr, bool allowChar1Literal);

bool comparisonExprsSupported(const ExprPtr& left, const ExprPtr& right) {
    if (exprSupported(left, false) && exprSupported(right, false)) return true;
    if (exprSupported(left, false) && literalMatchesType(right, inferExprDataType(left))) return true;
    if (exprSupported(right, false) && literalMatchesType(left, inferExprDataType(right))) return true;
    return false;
}

bool fixedStringCompareSupported(const ExprPtr& maybeCol, const ExprPtr& maybeLiteral) {
    auto* col = maybeCol ? std::get_if<ColRef>(&maybeCol->node) : nullptr;
    auto* lit = maybeLiteral ? std::get_if<Literal>(&maybeLiteral->node) : nullptr;
    if (!col || col->dataType != DataType::CHAR_FIXED || !lit) return false;
    return std::holds_alternative<std::string>(lit->value);
}

bool fixedStringLikeSupported(const Like& like) {
    auto* col = like.expr ? std::get_if<ColRef>(&like.expr->node) : nullptr;
    if (!col || col->dataType != DataType::CHAR_FIXED) return false;
    if (like.pattern.find('_') != std::string::npos ||
        like.pattern.find('\\') != std::string::npos) return false;

    std::vector<size_t> segmentLens;
    size_t currentLen = 0;
    bool inSegment = false;
    for (char ch : like.pattern) {
        if (ch == '%') {
            if (inSegment) {
                segmentLens.push_back(currentLen);
                inSegment = false;
                currentLen = 0;
            }
        } else {
            inSegment = true;
            currentLen++;
        }
    }
    if (inSegment) {
        segmentLens.push_back(currentLen);
    }
    if (segmentLens.size() > 2) return false;
    if (segmentLens.size() == 1 && like.pattern.back() == '%' && like.pattern.front() != '%') {
        return true;
    }
    return std::all_of(segmentLens.begin(), segmentLens.end(), [](size_t len) { return len <= 16; });
}

bool exprSupported(const ExprPtr& expr, bool allowChar1Literal);
bool predSupported(const PredPtr& pred);
static bool validateHavingPredicate(const PredPtr& having,
                                     const std::vector<ExprPtr>& groupBy,
                                     const std::vector<SelectTarget>& targets,
                                     std::string* error);

bool exprSupported(const ExprPtr& expr, bool allowChar1Literal) {
    if (!expr) return false;
    return std::visit([&](auto&& node) -> bool {
        using Node = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<Node, ColRef>) {
            return node.dataType != DataType::CHAR_FIXED;
        } else if constexpr (std::is_same_v<Node, Literal>) {
            if (std::holds_alternative<std::string>(node.value)) {
                const auto& value = std::get<std::string>(node.value);
                return allowChar1Literal && value.size() == 1;
            }
            return true;
        } else if constexpr (std::is_same_v<Node, BinaryExpr>) {
            return exprSupported(node.left, allowChar1Literal) &&
                   exprSupported(node.right, allowChar1Literal) &&
                   isNumericLike(inferExprDataType(node.left)) &&
                   isNumericLike(inferExprDataType(node.right));
        } else if constexpr (std::is_same_v<Node, CaseWhen>) {
            for (const auto& br : node.branches) {
                if (!predSupported(br.condition)) return false;
                DataType rt = inferExprDataType(br.result);
                if (rt != DataType::INT && rt != DataType::FLOAT && rt != DataType::DATE)
                    return false;
                if (!exprSupported(br.result, allowChar1Literal)) return false;
            }
            if (node.elseResult) {
                DataType et = inferExprDataType(node.elseResult);
                if (et != DataType::INT && et != DataType::FLOAT && et != DataType::DATE)
                    return false;
                if (!exprSupported(node.elseResult, allowChar1Literal)) return false;
            }
            return true;
        } else if constexpr (std::is_same_v<Node, FuncCall>) {
            // Known translatable SQL functions and aggregate wrappers
            if (node.name == "date_part" || node.name == "extract") return true;
            if (node.name == "substring") return true;
            if (node.name == "sum" || node.name == "count" || node.name == "avg" ||
                node.name == "min" || node.name == "max") return true;
            return false;
        } else {
            return false;
        }
    }, expr->node);
}

bool materializeExprSupported(const ExprPtr& expr) {
    if (!expr) return false;
    if (auto* col = std::get_if<ColRef>(&expr->node)) {
        return col->dataType == DataType::INT || col->dataType == DataType::FLOAT ||
               col->dataType == DataType::DATE || col->dataType == DataType::CHAR1 ||
               col->dataType == DataType::CHAR_FIXED;
    }
    return exprSupported(expr, false);
}

int fixedStringLenForExpr(const ExprPtr& expr, const SchemaProvider* schema = nullptr) {
    if (!expr) return 0;
    auto* col = std::get_if<ColRef>(&expr->node);
    if (!col) return 0;
    if (col->dataType == DataType::CHAR1) return 1;
    if (col->dataType != DataType::CHAR_FIXED) return 0;
    return schema ? schema->columnFixedWidth(col->table, col->column) : 0;
}

std::string materializeValueExpr(const ExprPtr& expr, const std::string& idxVar,
                                  const SchemaProvider* schema = nullptr) {
    if (auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr) {
        if (col->dataType == DataType::CHAR1) {
            return col->column + " + " + idxVar;
        }
        if (col->dataType == DataType::CHAR_FIXED) {
            int len = fixedStringLenForExpr(expr, schema);
            return col->column + " + " + idxVar + " * " + std::to_string(len);
        }
    }
    return exprToMetal(expr, idxVar, schema);
}

bool predSupported(const PredPtr& pred) {
    if (!pred) return true;
    return std::visit([&](auto&& node) -> bool {
        using Node = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<Node, Comparison>) {
            if (node.op == CmpOp::EQ || node.op == CmpOp::NE) {
                if (fixedStringCompareSupported(node.left, node.right) ||
                    fixedStringCompareSupported(node.right, node.left)) return true;
                // CHAR1 column compared to string literal (e.g. c_mktsegment = 'BUILDING')
                auto* lc = node.left ? std::get_if<ColRef>(&node.left->node) : nullptr;
                auto* rc = node.right ? std::get_if<ColRef>(&node.right->node) : nullptr;
                auto* ll = node.left ? std::get_if<Literal>(&node.left->node) : nullptr;
                auto* rl = node.right ? std::get_if<Literal>(&node.right->node) : nullptr;
                if ((lc && lc->dataType == DataType::CHAR1 && rl && std::holds_alternative<std::string>(rl->value)) ||
                    (rc && rc->dataType == DataType::CHAR1 && ll && std::holds_alternative<std::string>(ll->value)))
                    return true;
            }
            return comparisonExprsSupported(node.left, node.right);
        } else if constexpr (std::is_same_v<Node, Between>) {
            DataType exprType = inferExprDataType(node.expr);
            return exprSupported(node.expr, false) &&
                   (exprSupported(node.low, false) || literalMatchesType(node.low, exprType)) &&
                   (exprSupported(node.high, false) || literalMatchesType(node.high, exprType));
        } else if constexpr (std::is_same_v<Node, InList>) {
            auto* inCol = node.expr ? std::get_if<ColRef>(&node.expr->node) : nullptr;
            if (inCol && inCol->dataType == DataType::CHAR_FIXED) {
                return std::all_of(node.values.begin(), node.values.end(), [](const ExprPtr& value) {
                    auto* lit = value ? std::get_if<Literal>(&value->node) : nullptr;
                    return lit && std::holds_alternative<std::string>(lit->value);
                });
            }
            if (!exprSupported(node.expr, false)) return false;
            DataType exprType = inferExprDataType(node.expr);
            return std::all_of(node.values.begin(), node.values.end(), [exprType](const ExprPtr& value) {
                return exprSupported(value, false) || literalMatchesType(value, exprType);
            });
        } else if constexpr (std::is_same_v<Node, LogicalAnd>) {
            return std::all_of(node.children.begin(), node.children.end(), predSupported);
        } else if constexpr (std::is_same_v<Node, LogicalOr>) {
            return std::all_of(node.children.begin(), node.children.end(), predSupported);
        } else if constexpr (std::is_same_v<Node, LogicalNot>) {
            return predSupported(node.child);
        } else if constexpr (std::is_same_v<Node, Like>) {
            return fixedStringLikeSupported(node);
        } else if constexpr (std::is_same_v<Node, ExistsPred>) {
            return true;  // EXISTS subqueries handled via inlining; remaining placeholders OK
        } else {
            return false;
        }
    }, pred->node);
}

bool filtersSupported(const std::vector<PredPtr>& filters) {
    return std::all_of(filters.begin(), filters.end(), predSupported);
}

std::unique_ptr<MetalOperator> makeFilteredScan(const AnalyzedQuery& aq,
                                                const std::set<std::string>& columns,
                                                const std::string& idxVar) {
    std::set<std::string> scanColumns = columns;
    if (scanColumns.empty()) {
        if (aq.schema) {
            auto pk = aq.schema->pkInfo(aq.tables[0]);
            if (pk) scanColumns.insert(pk->first);
        }
    }
    auto scan = makeScanForCols(aq.tables[0], idxVar, scanColumns);
    return maybeSelect(std::move(scan), combineFilters(aq.filters, idxVar));
}

std::optional<MetalQueryPlan> buildScalarAggPlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.hasAggregation()) return std::nullopt;
    if (aq.hasGroupBy()) return fail("Scalar aggregation: GROUP BY not supported in scalar-agg path.");
    if (aq.having) return fail("Scalar aggregation: HAVING not supported in scalar-agg path.");
    if (!aq.orderBy.empty()) return fail("Scalar aggregation: ORDER BY not supported in scalar-agg path.");
    if (aq.limit >= 0) return fail("Scalar aggregation: LIMIT not supported in scalar-agg path.");

    std::set<std::string> usedColumns;
    if (!aq.schema) {
        // TPC-H manual path: compute which columns the scan needs.
        for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);
    }

    for (const auto& target : aq.targets) {
        if (!target.isAgg || !target.agg) return fail("Scalar aggregation: non-aggregate SELECT target.");
        const AggFunc func = target.agg->func;
        if (func != AggFunc::SUM && func != AggFunc::COUNT && func != AggFunc::AVG &&
            func != AggFunc::MIN && func != AggFunc::MAX)
            return fail("Scalar aggregation: unsupported aggregate function '" + aggName(func) + "'.");
        if (func != AggFunc::COUNT) {
            if (!target.agg->innerExpr)
                return fail("Scalar aggregation: aggregate '" + aggName(func) + "' requires an inner expression.");
            if (!exprSupported(target.agg->innerExpr, false))
                return fail("Scalar aggregation: inner expression of '" + aggName(func) + "' not supported on GPU.");
            if (!isNumericLike(inferExprDataType(target.agg->innerExpr)))
                return fail("Scalar aggregation: inner expression of '" + aggName(func) + "' must be numeric.");
            if (!aq.schema) collectColumns(target.agg->innerExpr, usedColumns);
        }
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_SCALAR";
    const std::string idxVar = "i";
    auto filtered = makeFilteredScan(aq, usedColumns, idxVar);
    auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_adhoc_scalar");

    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        std::string alias = displayNameForTarget(target, targetIndex);
        std::string accName = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(alias);
        AggFunc func = target.agg->func;
        if (func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(alias, accIndex, 0);
        } else if (func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum",
                                                  exprToMetal(target.agg->innerExpr, idxVar),
                                                  "float");
            int countIndex = reduce->addAccumulator(accName + "_count", "1.0f", "float");
            reduce->setAverageResultAlias(alias, sumIndex, countIndex, 0);
        } else {
            DataType valueType = inferExprDataType(target.agg->innerExpr);
            std::string outputType = (valueType == DataType::FLOAT) ? "float" : "long";
            MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
            if (func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
            else if (func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
            if (op != MetalTGReduce::ReduceOp::SUM && valueType != DataType::FLOAT) {
                outputType = "int";
            }
            int accIndex = reduce->addAccumulator(accName, exprToMetal(target.agg->innerExpr, idxVar),
                                                  outputType, "", "", op);
            reduce->setAccumulatorResultAlias(alias, accIndex, 0);
        }
    }

    appendPhase(plan, "ADHOC_single_table_scalar", std::move(reduce));
    return plan;
}

// Struct to describe how each group-by key is flattened into a linear bucket index.
struct GroupKeyDesc {
    std::string keyExpr;       // expression to index the column value (e.g. "l_returnflag[i]")
    int numValues = 0;         // number of distinct values this key can take
    int stride = 0;            // multiplier for this key's contribution to flat bucket
};

// Build a bucket expression for a single CHAR1 group key domain.
// Returns "(char)col[idx] - 'A'" style expression, or empty if unsupported.
static std::string char1BucketExpr(const ColRef& col, const std::string& idxVar,
                                    int& outNumValues, const SchemaProvider* schema = nullptr) {
    if (schema && col.table.size()) {
        auto chars = schema->charDomain(col.table, col.column);
        if (!chars.empty()) {
            outNumValues = (int)chars.size();
            if (outNumValues == 1) return "0";
            const std::string expr = col.column + "[" + idxVar + "]";
            std::string result;
            for (int i = 0; i < outNumValues - 1; ++i) {
                result += "(" + expr + " == '" + chars[i] + "' ? " + std::to_string(i) + " : ";
            }
            result += std::to_string(outNumValues - 1);
            for (int i = 0; i < outNumValues - 1; ++i) result += ")";
            return result;
        }
    }
    outNumValues = 0;
    return "";
}

std::optional<MetalQueryPlan> buildGroupedAggPlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.hasAggregation()) return fail("Grouped aggregation: query has no aggregation.");
    if (!aq.hasGroupBy()) return fail("Grouped aggregation: query has no GROUP BY.");
    // HAVING is allowed; ORDER BY and LIMIT are handled CPU-side.

    struct PendingAgg {
        std::string displayName;
        std::string name;
        int offset = 0;
        std::string valueExpr;
        bool isLongPair = false;
        int scaleDown = 0;
        bool isFloatSum = false;
        bool isMinMax = false;
        std::string atomicOp = "add";
        std::string funcName;      // aggregate function for HAVING matching
        std::string innerColumn;   // referenced column for HAVING matching
    };

    std::set<std::string> usedColumns;
    if (!aq.schema) {
        for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);
    }
    // GROUP BY columns are added during the key descriptor loop below
    // (derived columns need special handling to not add alias names).

    const std::string idxVar = "i";

    // --- Build group-key descriptors ---
    std::vector<GroupKeyDesc> keyDescriptors;
    int totalBuckets = 1;

    for (size_t ki = 0; ki < aq.groupBy.size(); ++ki) {
        auto* gc = aq.groupBy[ki] ? std::get_if<ColRef>(&aq.groupBy[ki]->node) : nullptr;
        auto* fc = (!gc && aq.groupBy[ki]) ? std::get_if<FuncCall>(&aq.groupBy[ki]->node) : nullptr;
        if (!gc && !fc) return fail("Grouped aggregation: GROUP BY expression #" + std::to_string(ki+1) + " must be a column reference or translatable function call.");

        GroupKeyDesc kd;
        // For derived columns (empty table), use the source expression from
        // subqueryExprMap as the key expression directly.
        if (gc && gc->table.empty() && aq.subqueryExprMap.count(gc->column)) {
            auto srcExpr = aq.subqueryExprMap.at(gc->column);
            kd.keyExpr = exprToMetal(srcExpr, idxVar);
            kd.numValues = 256;  // safe upper bound for derived INT columns
            if (!aq.schema) collectColumns(srcExpr, usedColumns);
        } else if (fc) {
            // FuncCall GROUP BY (e.g. extract(year from ...) resolved via subquery alias)
            kd.keyExpr = exprToMetal(aq.groupBy[ki], idxVar);
            kd.numValues = 256;
            if (!aq.schema) collectColumns(aq.groupBy[ki], usedColumns);
        } else if (gc && gc->dataType == DataType::CHAR1) {
            kd.keyExpr = char1BucketExpr(*gc, idxVar, kd.numValues, aq.schema);
        } else {
            auto domain = smallIntGroupDomain(*gc, aq.schema);
            if (!domain || domain->maxValue < domain->minValue) return fail("Grouped aggregation: column '" + gc->column + "' has no known integer domain.");
            kd.numValues = domain->maxValue - domain->minValue + 1;
            std::string groupValue = gc->column + "[" + idxVar + "]";
            if (domain->minValue != 0) {
                kd.keyExpr = "(" + groupValue + " - " + std::to_string(domain->minValue) + ")";
            } else {
                kd.keyExpr = groupValue;
            }
            kd.keyExpr = "clamp(" + kd.keyExpr + ", 0, " + std::to_string(kd.numValues - 1) + ")";
        }
        // Add column to scan if not derived
        if (!aq.schema && gc && !gc->table.empty()) usedColumns.insert(gc->column);
        kd.stride = totalBuckets;
        totalBuckets *= kd.numValues;
        keyDescriptors.push_back(kd);
    }

    // Cap: refuse plans with > 4096 buckets (excessive GPU buffer waste).
    if (totalBuckets > 4096) return fail("Grouped aggregation: composite bucket count " + std::to_string(totalBuckets) + " exceeds maximum 4096.");
    const int numBuckets = totalBuckets;

    // --- Build the flat bucket expression ---
    // Encode: bucket = k0 + k1*stride1 + k2*stride2 + ...
    // where stride[i] = product of numValues[0..i-1] (computed above).
    std::string bucketExpr = "(" + keyDescriptors[0].keyExpr + ")";
    for (size_t ki = 1; ki < keyDescriptors.size(); ++ki) {
        bucketExpr = "(" + bucketExpr + " + (" + keyDescriptors[ki].keyExpr + ") * " +
                     std::to_string(keyDescriptors[ki].stride) + ")";
    }

    // --- Resolve group-key display names ---
    std::vector<std::string> keyDisplayNames(keyDescriptors.size());
    for (size_t ki = 0; ki < keyDescriptors.size(); ++ki) {
        auto* gc = std::get_if<ColRef>(&aq.groupBy[ki]->node);
        if (gc) keyDisplayNames[ki] = gc->column;
        else keyDisplayNames[ki] = "expr_" + std::to_string(ki);
    }
    // Override with aliases from SELECT targets where present
    for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
        const auto& target = aq.targets[ti];
        if (target.isAgg) continue;
        for (size_t ki = 0; ki < keyDescriptors.size(); ++ki) {
            auto* gc = std::get_if<ColRef>(&aq.groupBy[ki]->node);
            if (gc && exprIsColumn(target.expr, *gc)) {
                keyDisplayNames[ki] = displayNameForTarget(target, ti);
            } else if (!gc && !target.isAgg) {
                // FuncCall GROUP BY: match position
                keyDisplayNames[ki] = displayNameForTarget(target, ti);
            }
        }
    }

    // --- Collect aggregate slots ---
    std::vector<PendingAgg> pending;
    int valuesPerBucket = 0;

    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        if (!target.isAgg) continue; // group keys already accounted for
        if (!target.agg) return fail("Grouped aggregation: malformed aggregate SELECT target #" + std::to_string(targetIndex) + ".");

        const AggFunc func = target.agg->func;

        auto extractInnerColumn = [&](const ExprPtr& inner) -> std::string {
            if (!inner) return "";
            if (auto* cr = std::get_if<ColRef>(&inner->node)) return cr->column;
            return "";
        };

        if (func == AggFunc::COUNT) {
            PendingAgg agg;
            agg.displayName = displayNameForTarget(target, targetIndex);
            agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
            agg.offset = valuesPerBucket++;
            agg.valueExpr = "1u";
            agg.atomicOp = "add";
            agg.funcName = "COUNT";
            agg.innerColumn = "";
            pending.push_back(std::move(agg));
            continue;
        }

        if (func == AggFunc::AVG) {
            // AVG = SUM/COUNT. Allocate two slots: sum (long pair) + count (uint).
            if (!target.agg->innerExpr)
                return fail("Grouped aggregation: AVG requires an inner expression.");
            if (!exprSupported(target.agg->innerExpr, false))
                return fail("Grouped aggregation: AVG inner expression not supported on GPU.");
            DataType vt = inferExprDataType(target.agg->innerExpr);
            bool isFloat = (vt == DataType::FLOAT);
            if (!aq.schema) collectColumns(target.agg->innerExpr, usedColumns);

            std::string dname = displayNameForTarget(target, targetIndex);
            std::string sname = sanitizeIdentifier(dname);

            // SUM slot (long pair at offset/offset+1, or float at offset with isFloatSum)
            {
                PendingAgg agg;
                agg.displayName = dname;
                agg.name = "a" + std::to_string(targetIndex) + "_" + sname + "_sum";
                agg.offset = valuesPerBucket;
                if (isFloat) {
                    agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
                    agg.isFloatSum = true;
                    agg.scaleDown = -1; // marker: this is AVG sum, needs denominator
                    valuesPerBucket += 1;
                } else {
                    agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
                    agg.isLongPair = true;
                    agg.scaleDown = -1; // marker: AVG sum, denominator follows
                    valuesPerBucket += 2;
                }
                agg.atomicOp = "add";
                agg.funcName = "AVG";
                agg.innerColumn = extractInnerColumn(target.agg->innerExpr);
                pending.push_back(std::move(agg));
            }

            // COUNT slot (for AVG denominator)
            {
                PendingAgg agg;
                agg.displayName = dname + "_cnt";
                agg.name = "a" + std::to_string(targetIndex) + "_" + sname + "_cnt";
                agg.offset = valuesPerBucket++;
                agg.valueExpr = "1u";
                agg.atomicOp = "add";
                agg.scaleDown = 0;
                agg.funcName = "AVG";
                agg.innerColumn = "";
                pending.push_back(std::move(agg));
            }
            continue;
        }

        if (func == AggFunc::MIN || func == AggFunc::MAX) {
            if (!target.agg->innerExpr)
                return fail("Grouped aggregation: " + aggName(func) + " requires an inner expression.");
            if (!exprSupported(target.agg->innerExpr, false))
                return fail("Grouped aggregation: " + aggName(func) + " inner expression not supported on GPU.");
            DataType vt = inferExprDataType(target.agg->innerExpr);
            bool isFloat = (vt == DataType::FLOAT);
            if (!aq.schema) collectColumns(target.agg->innerExpr, usedColumns);

            PendingAgg agg;
            agg.displayName = displayNameForTarget(target, targetIndex);
            agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
            agg.offset = valuesPerBucket;
            agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
            agg.atomicOp = (func == AggFunc::MIN) ? "min" : "max";
            agg.isMinMax = true;
            agg.funcName = (func == AggFunc::MIN) ? "MIN" : "MAX";
            agg.innerColumn = extractInnerColumn(target.agg->innerExpr);
            if (isFloat) {
                agg.isFloatSum = true; // re-use float path for single-uint storage
                agg.scaleDown = 0;
            }
            valuesPerBucket += 1;
            pending.push_back(std::move(agg));
            continue;
        }

        // COUNT(DISTINCT col): allocate a popcount result slot.
        // The distinct bitmap is set per-row via atomicOr; after the
        // keyed-agg phase, a bitmap-popcount kernel reads the per-group
        // bitmaps and stores the distinct count into the output slot.
        if (func == AggFunc::COUNT_DISTINCT) {
            if (!target.agg->innerExpr)
                return fail("Grouped aggregation: COUNT(DISTINCT) requires an inner expression.");
            auto* innerCol = target.agg->innerExpr ? std::get_if<ColRef>(&target.agg->innerExpr->node) : nullptr;
            if (!innerCol)
                return fail("Grouped aggregation: COUNT(DISTINCT) inner expression must be a column reference.");
            DataType vt = inferExprDataType(target.agg->innerExpr);
            if (vt != DataType::INT && vt != DataType::DATE)
                return fail("Grouped aggregation: COUNT(DISTINCT) only supports integer/date columns.");

            // Find max value for bitmap sizing.
            std::string maxExpr;
            auto gd = aq.schema->groupDomain(innerCol->table, innerCol->column);
            if (gd && gd->maxValue >= 0)
                maxExpr = std::to_string(gd->maxValue + 1);
            if (maxExpr.empty()) {
                auto ms = aq.schema->maxKeySymbol(innerCol->table);
                if (!ms.empty())
                    maxExpr = ms + " + 1";
            }
            if (maxExpr.empty())
                return fail("Grouped aggregation: COUNT(DISTINCT) on column '" + innerCol->column +
                           "' — no known max value for bitmap sizing.");

            if (!aq.schema) collectColumns(target.agg->innerExpr, usedColumns);
            std::string dname = displayNameForTarget(target, targetIndex);
            std::string valueExpr = innerCol->column + "[" + idxVar + "]";

            // Record a placeholder pending agg for the result offset,
            // storing the maxExpr in a secondary field (reuse isFloatSum
            // since it's not used for COUNT_DISTINCT; the pending loop
            // handles scaleDown == -2 specially).
            PendingAgg agg;
            agg.displayName = dname;
            agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(dname);
            agg.offset = valuesPerBucket++;
            agg.valueExpr = maxExpr;        // stash maxExpr here for the pending loop
            agg.isFloatSum = false;
            agg.isMinMax = false;
            agg.atomicOp = "add";
            agg.funcName = "COUNT_DISTINCT";
            agg.innerColumn = innerCol->column;
            agg.scaleDown = -2;  // sentinel: COUNT_DISTINCT slot
            agg.isLongPair = false;
            pending.push_back(std::move(agg));
            continue;
        }

        if (func != AggFunc::SUM)
            return fail("Grouped aggregation: unsupported aggregate function '" + aggName(func) + "'.");
        if (!target.agg->innerExpr)
            return fail("Grouped aggregation: SUM requires an inner expression.");
        if (!exprSupported(target.agg->innerExpr, false))
            return fail("Grouped aggregation: SUM inner expression not supported on GPU.");
        DataType vt = inferExprDataType(target.agg->innerExpr);
        if (!aq.schema) collectColumns(target.agg->innerExpr, usedColumns);

        PendingAgg agg;
        agg.displayName = displayNameForTarget(target, targetIndex);
        agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
        agg.offset = valuesPerBucket;
        agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
        agg.atomicOp = "add";
        agg.funcName = "SUM";
        agg.innerColumn = extractInnerColumn(target.agg->innerExpr);

        if (vt == DataType::FLOAT) {
            // Float SUM: accumulate via atomic CAS on uint slot (reinterpret as float)
            agg.isFloatSum = true;
            agg.scaleDown = 0;
            valuesPerBucket += 1;
        } else {
            // Integer/date SUM: long-pair for 64-bit correctness
            agg.isLongPair = true;
            agg.scaleDown = 0;
            valuesPerBucket += 2;
        }
        pending.push_back(std::move(agg));
    }

    if (pending.empty()) return fail("Grouped aggregation: no valid aggregate functions found in SELECT targets.");

    // --- Build the scan → selection → keyed-agg tree ---
    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_GROUP";
    auto filtered = makeFilteredScan(aq, usedColumns, idxVar);

    // Add bucket guard if all keys have known domain bounds
    bool needGuard = true;
    for (const auto& kd : keyDescriptors) {
        if (kd.numValues <= 0) { needGuard = false; break; }
    }
    if (needGuard) {
        // Emit guard: bucket >= 0 && bucket < numBuckets
        std::string guard = "(" + bucketExpr + " >= 0 && " + bucketExpr + " < " + std::to_string(numBuckets) + ")";
        filtered = maybeSelect(std::move(filtered), guard);
    }

    auto agg = std::make_unique<MetalKeyedAgg>(
        std::move(filtered), "d_adhoc_group_aggs", bucketExpr,
        numBuckets, valuesPerBucket, std::to_string(numBuckets * valuesPerBucket));

    // Set key result info with multi-key decoding
    // Convert GroupKeyDesc → GroupKeyDecode for the operator
    std::vector<GroupKeyDecode> decodeInfo;
    for (size_t ki = 0; ki < keyDescriptors.size(); ++ki) {
        GroupKeyDecode d;
        d.name = keyDisplayNames[ki];
        d.numValues = keyDescriptors[ki].numValues;
        d.stride = keyDescriptors[ki].stride;
        // CHAR1: populate charMap. Integer: populate keyBase.
        auto* gc = std::get_if<ColRef>(&aq.groupBy[ki]->node);
        if (gc && gc->dataType == DataType::CHAR1) {
            // Build reverse map: flat index → char from schema domain
            d.charMap = aq.schema->charDomain(gc->table, gc->column);
        } else if (gc) {
            // Integer: find the base offset from domain
            auto domain = smallIntGroupDomain(*gc, aq.schema);
            d.keyBase = domain ? domain->minValue : 0;
        }
        // FuncCall GROUP BY: no decode needed (keyExpr is raw computed value)
        decodeInfo.push_back(d);
    }
    agg->setMultiKeyResult(keyDisplayNames, decodeInfo, numBuckets);

    // Track COUNT(DISTINCT) entries for popcount phases.
    struct DistinctEntry {
        std::string displayName;
        std::string valueExpr;
        std::string maxValExpr;
        int offset;
    };
    std::vector<DistinctEntry> distinctEntries;

    for (const auto& pendingAgg : pending) {
        if (pendingAgg.scaleDown == -2) {
            // COUNT(DISTINCT) slot — valueExpr holds the max value expression.
            std::string bmpOutput = "d_adhoc_distinct_" + std::to_string(distinctEntries.size());
            std::string maxExpr = pendingAgg.valueExpr;
            std::string colExpr = pendingAgg.innerColumn + "[" + idxVar + "]";
            agg->addDistinctBitmap(bmpOutput, colExpr, maxExpr);
            distinctEntries.push_back({pendingAgg.displayName, colExpr,
                                       maxExpr, pendingAgg.offset});
            // Add a zero placeholder aggregate slot.
            agg->addAggregateWithMeta(pendingAgg.displayName, pendingAgg.offset, "0u",
                                      "add", false, 0, false, false,
                                      pendingAgg.funcName, pendingAgg.innerColumn);
        } else {
            agg->addAggregateWithMeta(pendingAgg.displayName, pendingAgg.offset, pendingAgg.valueExpr,
                                      pendingAgg.atomicOp, pendingAgg.isLongPair, pendingAgg.scaleDown,
                                      pendingAgg.isFloatSum, pendingAgg.isMinMax,
                                      pendingAgg.funcName, pendingAgg.innerColumn);
        }
    }

    // Set HAVING predicate if present
    if (aq.having) {
        if (!validateHavingPredicate(aq.having, aq.groupBy, aq.targets, error))
            return std::nullopt;
        agg->setHaving(aq.having);
    }

    appendPhase(plan, "ADHOC_single_table_group", std::move(agg));

    // Add bitmap popcount phases for each COUNT(DISTINCT).
    for (size_t di = 0; di < distinctEntries.size(); ++di) {
        const auto& de = distinctEntries[di];
        std::string bmpName = "d_distinct_bmp_d_adhoc_distinct_" + std::to_string(di);
        std::string bmpOutput = "d_adhoc_distinct_" + std::to_string(di);
        std::string strideExpr = "((" + de.maxValExpr + " + 32) / 32)";
        auto popcnt = std::make_unique<MetalBitmapPopcount>(
            bmpName, bmpOutput, std::to_string(numBuckets), strideExpr);
        appendPhase(plan, "ADHOC_single_table_popcount_" + std::to_string(di), std::move(popcnt));
    }

    if (!aq.orderBy.empty() || aq.limit >= 0) {
        MetalQueryPlan::CpuSort cpuSort;
        cpuSort.limit = aq.limit;
        for (const auto& order : aq.orderBy) {
            auto column = orderColumnForExpr(order.expr, aq.targets);
            if (column) cpuSort.keys.push_back({*column, order.descending});
        }
        plan.cpuSort = cpuSort;
    }
    return plan;
}

std::optional<MetalQueryPlan> buildMaterializePlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (aq.hasAggregation()) return fail("Materialization: query has aggregates; use scalar or grouped aggregation instead.");
    if (aq.hasGroupBy()) return fail("Materialization: query has GROUP BY; use grouped aggregation instead.");
    if (aq.having) return fail("Materialization: HAVING requires aggregation.");
    if (aq.targets.empty()) return fail("Materialization: no SELECT targets.");

    MetalQueryPlan::CpuSort cpuSort;
    cpuSort.limit = aq.limit;
    for (const auto& order : aq.orderBy) {
        auto column = orderColumnForExpr(order.expr, aq.targets);
        if (!column) {
            std::string orderStr = order.expr ? "expression" : "?";
            return fail("Materialization: ORDER BY " + orderStr + " could not be resolved to a SELECT target.");
        }
        cpuSort.keys.push_back({*column, order.descending});
    }

    std::set<std::string> usedColumns;
    if (!aq.schema) {
        for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);
    }
    for (const auto& target : aq.targets) {
        if (!target.expr)
            return fail("Materialization: SELECT target has no expression.");
        if (!materializeExprSupported(target.expr))
            return fail("Materialization: SELECT expression not supported on GPU.");
        if (!aq.schema) collectColumns(target.expr, usedColumns);
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_MATERIALIZE";
    const std::string idxVar = "i";
    auto filtered = makeFilteredScan(aq, usedColumns, idxVar);
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(filtered), "d_adhoc_result_count", "1");

    const std::string outputSize = tableSizeName(aq.tables[0]);
    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        DataType type = inferExprDataType(target.expr);
        std::string displayName = displayNameForTarget(target, targetIndex);
        std::string bufferName = "d_adhoc_" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(displayName);
        int stringLen = fixedStringLenForExpr(target.expr, aq.schema);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        materialize->addColumn(bufferName, metalTypeForDataType(type),
                       materializeValueExpr(target.expr, idxVar, aq.schema), displayName,
                       sizeExpr, stringLen);
    }

    appendPhase(plan, "ADHOC_single_table_materialize", std::move(materialize));

    // ── GPU Sort (if ORDER BY is on a single numeric column) ──
    bool useGpuSort = false;
    std::string sortSourceCol;  // materialized output buffer to sort on
    std::string sortType;       // "int" or "float"
    bool sortDesc = false;

    if (!cpuSort.keys.empty() && cpuSort.keys.size() == 1) {
        const auto& sk = cpuSort.keys[0];
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            if (displayNameForTarget(aq.targets[ti], ti) == sk.column) {
                DataType dt = inferExprDataType(aq.targets[ti].expr);
                if (dt == DataType::INT || dt == DataType::DATE || dt == DataType::FLOAT) {
                    sortSourceCol = "d_adhoc_" + std::to_string(ti) + "_" + sanitizeIdentifier(sk.column);
                    sortType = (dt == DataType::FLOAT) ? "float" : "int";
                    sortDesc = sk.descending;
                    useGpuSort = true;
                }
                break;
            }
        }
    }

    if (useGpuSort) {
        // Hook on the materialize phase: read the atomic counter and register
        // it as a scalar for the sort phases that follow.
        const std::string cntName = "d_adhoc_result_count";
        const std::string nSym = "n_sort_results";
        auto& matPhase = plan.phases.back();
        matPhase.postDispatchHook = [cntName, nSym](MetalGenericExecutor& executor) {
            auto* buf = executor.getAllocatedBuffer(cntName);
            if (buf) {
                uint32_t n = *static_cast<const uint32_t*>(buf->contents());
                executor.registerScalarInt(nSym, (int)n);
            }
        };

        // Init sort keys phase: encodes materialized output column → sort keys
        auto initSort = std::make_unique<MetalInitSortKeys>(
            sortSourceCol, sortType, "d_sortKey", "d_sortIdx", nSym, sortDesc);
        appendPhase(plan, "ADHOC_sort_init", std::move(initSort));

        // Sort step phase with PostDispatchHook for (k,j) bitonic loop
        auto sortStep = std::make_unique<MetalBitonicSortStep>("d_sortKey", "d_sortIdx", nSym);
        const std::string sortPhaseName = "ADHOC_sort_step";
        auto& sortPhase = appendPhase(plan, sortPhaseName, std::move(sortStep));
        sortPhase.postDispatchHook = MetalInitSortKeys::makeBitonicHook(
            sortPhaseName, "d_sortKey", "d_sortIdx", nSym);

        // Tell post-processing to use sorted indices
        plan.gpuSort = MetalQueryPlan::GpuSort{"d_sortIdx", nSym, sortDesc};

        // CPU sort only for LIMIT (sort order comes from GPU)
        if (cpuSort.limit >= 0) {
            MetalQueryPlan::CpuSort cpuLimit;
            cpuLimit.limit = cpuSort.limit;
            plan.cpuSort = cpuLimit;
        }
    } else if (!cpuSort.keys.empty() || cpuSort.limit >= 0) {
        plan.cpuSort = cpuSort;
    }

    return plan;
}

// ===================================================================
// MULTI-TABLE GENERIC AD-HOC BUILDER
// ===================================================================
//
// Plans an arbitrary equi-join query whose join graph forms a tree
// rooted at the largest referenced table (the "probe").  Build-side
// tables that contribute no values to the output are realised as
// SemiJoins (Bitmap build/probe).  Build tables that supply values
// to SELECT / GROUP BY / aggregate inputs become IndexJoins
// (ArrayStore on the build side, ArrayLookup on the probe side).
// Values originating in deeper nodes are forwarded one level at a
// time, so a value from `customer` can reach `lineitem` via an
// intermediate ArrayStore at `orders`.
//
// Supported terminal patterns:
//   - Scalar aggregation (no GROUP BY)            → MetalTGReduce
//   - Grouped aggregation by a single small-int
//     domain key (probe-side or carried)         → MetalKeyedAgg
//
// The plan is rejected when:
//   - the join graph is not a connected tree
//   - a non-probe table with carries has no PK descriptor or joins on non-PK
//     (SemiJoin-only edges accept any column since bitmaps are idempotent)
//   - HAVING / ORDER BY / LIMIT / DISTINCT are present (deferred)
// ===================================================================

struct MultiTablePkInfo {
    std::string column;
    std::string sizeSym;
};

std::optional<MultiTablePkInfo> multiTablePkInfo(const std::string& table, const SchemaProvider* schema = nullptr) {
    if (schema) {
        auto pk = schema->pkInfo(table);
        if (pk) return MultiTablePkInfo{pk->first, pk->second};
        return std::nullopt;
    }
    if (table == "customer") return MultiTablePkInfo{"c_custkey",   "maxCustkey"};
    if (table == "orders")   return MultiTablePkInfo{"o_orderkey",  "maxOrderkey"};
    if (table == "supplier") return MultiTablePkInfo{"s_suppkey",   "maxSuppkey"};
    if (table == "part")     return MultiTablePkInfo{"p_partkey",   "maxPartkey"};
    if (table == "partsupp") return MultiTablePkInfo{"ps_suppkey",  "maxSuppkey"};
    if (table == "nation")   return MultiTablePkInfo{"n_nationkey", "25"};
    if (table == "region")   return MultiTablePkInfo{"r_regionkey", "5"};
    return std::nullopt;
}

// Larger value = better probe candidate (largest TPC-H tables first).
int multiTableProbePriority(const std::string& t, const SchemaProvider* schema = nullptr) {
    if (schema) return schema->tableProbePriority(t);
    if (t == "lineitem") return 100;
    if (t == "orders")   return 80;
    if (t == "partsupp") return 70;
    if (t == "customer") return 50;
    if (t == "part")     return 40;
    if (t == "supplier") return 30;
    if (t == "nation")   return 10;
    if (t == "region")   return 5;
    return 0;
}

struct MultiTableTreeNode {
    std::string table;       // alias (unique per table, even self-joins)
    std::string baseTable;   // base table name for schema lookups
    int parent = -1;
    std::string keyOnSelf;    // column on this table participating in edge to parent
    std::string keyOnParent;  // column on parent participating in edge to this
    // Composite-key support: when keyOnSelf2/keyOnParent2 are non-empty,
    // the edge to the parent is a 2-column join (HashJoin).  The single-
    // column path is used only when keyOnSelf2 is empty.
    std::string keyOnSelf2;
    std::string keyOnParent2;
    // True when the single-column edge has to be served by a hash map
    // because keyOnSelf is not the build table's primary key (i.e. not a
    // direct-address key).  Composite-key edges always set this true.
    bool useHashJoin = false;
    bool anti = false;            // NOT EXISTS → anti-semi-join
    bool leftOuter = false;       // LEFT OUTER JOIN
    std::vector<int> children;
    bool composite() const { return !keyOnSelf2.empty(); }
};

// Build a tree rooted at `probeIdx` from `aq.joins`.  Each edge is
// consumed exactly once via BFS; if any join is unused or any table
// is unreachable the function returns false.  Edges between the same
// pair of tables are coalesced into a single composite-key edge.
bool multiTableBuildJoinTree(const AnalyzedQuery& aq, int probeIdx,
                             std::vector<MultiTableTreeNode>& nodes,
                             std::string* error) {
    const size_t n = aq.tables.size();
    nodes.assign(n, MultiTableTreeNode{});
    for (size_t i = 0; i < n; ++i) {
        nodes[i].table = aq.tableAliases[i];  // alias identity (unique per instance)
        nodes[i].baseTable = aq.tables[i];    // schema base name
    }

    // Coalesce JoinClauses by unordered (table, table) pair.  Each
    // coalesced edge carries 1 or 2 column pairs.
    struct Edge {
        std::string a, b;                 // table names
        std::vector<std::pair<std::string, std::string>> cols; // (col_a, col_b)
        bool anti = false;                // NOT EXISTS → anti-semi-join
        bool leftOuter = false;           // LEFT OUTER JOIN
    };
    std::vector<Edge> edges;
    auto findEdge = [&](const std::string& l, const std::string& r) -> int {
        for (size_t i = 0; i < edges.size(); ++i) {
            if ((edges[i].a == l && edges[i].b == r) ||
                (edges[i].a == r && edges[i].b == l)) return (int)i;
        }
        return -1;
    };
    for (const auto& jc : aq.joins) {
        int ei = (jc.leftTable == jc.rightTable) ? -1 : findEdge(jc.leftTable, jc.rightTable);
        // Self-joins (e.g. l1↔l2, l1↔l3) always get separate edges.
        if (ei < 0) {
            Edge e;
            e.a = jc.leftTable; e.b = jc.rightTable;
            e.cols.emplace_back(jc.leftCol, jc.rightCol);
            e.anti = jc.anti;
            e.leftOuter = jc.leftOuter;
            edges.push_back(std::move(e));
        } else {
            // Normalise column pair to edge orientation.
            if (edges[ei].a == jc.leftTable) {
                edges[ei].cols.emplace_back(jc.leftCol, jc.rightCol);
            } else {
                edges[ei].cols.emplace_back(jc.rightCol, jc.leftCol);
            }
            edges[ei].anti = edges[ei].anti || jc.anti;
            edges[ei].leftOuter = edges[ei].leftOuter || jc.leftOuter;
        }
    }
    for (const auto& e : edges) {
        if (e.cols.size() > 2) {
            if (error) *error = "Multi-table planner: more than 2 join columns between '" +
                                e.a + "' and '" + e.b + "' not supported.";
            return false;
        }
    }
    if (edges.size() < n - 1) {
        if (error) *error = "Multi-table planner: not enough join edges to connect all tables.";
        return false;
    }
    std::vector<bool> visited(n, false);

    // findIdx returns the first unvisited node with the given base table name.
    auto findIdx = [&](const std::string& baseTable) -> int {
        for (size_t k = 0; k < n; ++k)
            if (aq.tables[k] == baseTable && !visited[k]) return (int)k;
        return -1;
    };

    std::vector<int> order;
    order.push_back(probeIdx);
    visited[probeIdx] = true;
    std::vector<bool> edgeUsed(edges.size(), false);

    for (size_t qhead = 0; qhead < order.size(); ++qhead) {
        int u = order[qhead];
        const std::string& uTable = aq.tables[u];
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            if (edgeUsed[ei]) continue;
            const auto& e = edges[ei];
            int other = -1;
            // Determine which side is `u` and pick column orientations.
            std::vector<std::pair<std::string, std::string>> oriented; // (col_on_u, col_on_other)
            if (e.a == uTable) {
                other = findIdx(e.b);
                for (const auto& c : e.cols) oriented.emplace_back(c.first, c.second);
            } else if (e.b == uTable) {
                other = findIdx(e.a);
                for (const auto& c : e.cols) oriented.emplace_back(c.second, c.first);
            }
            if (other < 0 || visited[other]) continue;
            edgeUsed[ei] = true;
            visited[other] = true;
            nodes[other].parent = u;
            nodes[other].anti = e.anti;
            nodes[other].leftOuter = e.leftOuter;
            nodes[other].keyOnSelf   = oriented[0].second;
            nodes[other].keyOnParent = oriented[0].first;
            if (oriented.size() == 2) {
                nodes[other].keyOnSelf2   = oriented[1].second;
                nodes[other].keyOnParent2 = oriented[1].first;
            }
            nodes[u].children.push_back(other);
            order.push_back(other);
        }
    }

    if (order.size() != n) {
        if (error) {
            std::string dbg = std::to_string(order.size()) + "/" + std::to_string(n) + " reachable, tables: ";
            for (auto& t : aq.tables) dbg += t + " ";
            dbg += "joins: ";
            for (auto& j : aq.joins) dbg += j.leftTable + "." + j.leftCol + "=" + j.rightTable + "." + j.rightCol + " ";
            *error = "Join graph not connected (" + dbg + ").";
        }
        return false;
    }
    return true;
}

// Identifier for a column to be carried forward toward the probe.
struct CarriedKey {
    std::string table;
    std::string column;
    bool operator<(const CarriedKey& o) const {
        if (table != o.table) return table < o.table;
        return column < o.column;
    }
    bool operator==(const CarriedKey& o) const {
        return table == o.table && column == o.column;
    }
    std::string varName() const { return "_carry_" + table + "_" + column; }
    std::string storageArray(const std::string& storedAtTable) const {
        return "d_carry_" + table + "_" + column + "_at_" + storedAtTable;
    }
};

// Encode an origin-column value for raw int storage.  Raw int storage
// is uniform across all carry types so relays through intermediate
// build nodes can pass values through without per-hop reinterpretation.
//   INT/DATE  -> the value is already int.
//   FLOAT     -> as_type<int>(...) preserves bit pattern.
//   CHAR1     -> char widens to int.
// CHAR_FIXED is intentionally rejected upstream because expression
// rewriting cannot represent multi-byte access through a scalar carry.
std::string encodeCarryValue(DataType type, const std::string& expr) {
    switch (type) {
        case DataType::FLOAT:    return "as_type<int>(" + expr + ")";
        case DataType::CHAR1:    return "(int)(" + expr + ")";
        case DataType::INT:
        case DataType::DATE:
        default:                 return expr;
    }
}

// Mirror of encodeCarryValue: turn a raw int variable back into the
// origin-column value as it would have appeared on the probe.
std::string decodeCarryValue(DataType type, const std::string& var) {
    switch (type) {
        case DataType::FLOAT:    return "as_type<float>(" + var + ")";
        case DataType::CHAR1:    return "(char)(" + var + ")";
        case DataType::INT:
        case DataType::DATE:
        default:                 return var;
    }
}

// Replace every `<column>[<idxVar>]` occurrence in `expr` with the
// carried-variable name for ColRefs whose origin table is not the
// probe.  Column names in the TPC-H schema are unique so a textual
// substitution is unambiguous.  The substitution is type-aware:
// FLOAT/CHAR1 carries are wrapped in `as_type<...>` / `(char)` so the
// rewritten expression has the same observable type as the original.
std::string rewriteForProbe(std::string expr,
                              const std::string& idxVar,
                              const std::map<CarriedKey, std::string>& carryVar,
                              const SchemaProvider* schema = nullptr) {
    for (const auto& [key, var] : carryVar) {
        DataType t = schema->columnType(key.table, key.column);
        std::string sub = decodeCarryValue(t, var);
        // Standard column access pattern: col[idxVar]
        const std::string from = key.column + "[" + idxVar + "]";
        size_t pos = 0;
        while ((pos = expr.find(from, pos)) != std::string::npos) {
            expr.replace(pos, from.size(), sub);
            pos += sub.size();
        }
        // CHAR1 materialize access pattern: col + idxVar (not col[idxVar])
        if (t == DataType::CHAR1) {
            const std::string char1From = key.column + " + " + idxVar;
            pos = 0;
            while ((pos = expr.find(char1From, pos)) != std::string::npos) {
                expr.replace(pos, char1From.size(), sub);
                pos += sub.size();
            }
        }
    }
    return expr;
}

// Returns the table that owns column `c` (looking up the schema).
// If the column appears in multiple tables (column-name collision is
// not present in TPC-H), returns the first match.
std::string ownerTableForColumn(const AnalyzedQuery& aq, const std::string& c) {
    for (const auto& t : aq.tables) {
        if (aq.schema->hasColumn(t, c)) return t;
    }
    // Check subquery alias maps for derived columns.
    if (aq.subqueryColMap.count(c))
        return aq.subqueryColMap.at(c).table;
    return "";
}

bool carriedColumnSupported(DataType type) {
    // INT / DATE store directly; FLOAT and CHAR1 reinterpret-cast into
    // a 32-bit int slot at carry time and decode at probe time.
    // CHAR_FIXED is handled differently: the column buffer is added as a
    // read-only side buffer in the probe phase and indexed by the join key.
    return type == DataType::INT || type == DataType::DATE ||
           type == DataType::FLOAT || type == DataType::CHAR1 ||
           type == DataType::CHAR_FIXED;
}

// Collect aggregate function references from a predicate tree.
// Each element is (funcName, isStar, innerColumn).
namespace {
// Walk an expression tree inside a HAVING predicate and collect FuncCall references.
void collectFuncCalls(const ExprPtr& expr,
                      std::set<std::tuple<std::string, bool, std::string>>& out) {
    if (!expr) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, FuncCall>) {
            bool isStar = (node.name == "count" && node.args.empty());
            std::string innerCol;
            if (!isStar && !node.args.empty()) {
                if (auto* cr = node.args[0] ? std::get_if<ColRef>(&node.args[0]->node) : nullptr) {
                    innerCol = cr->column;
                }
            }
            out.insert({node.name, isStar, innerCol});
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            collectFuncCalls(node.left, out);
            collectFuncCalls(node.right, out);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (auto& b : node.branches) {
                collectFuncCalls(b.result, out);
            }
            if (node.elseResult) collectFuncCalls(node.elseResult, out);
        }
        // ColRef, Literal — not aggregates, skip
    }, expr->node);
}
} // namespace

// Collect FuncCall references from a HAVING predicate (walk Comparison/Between/InList).
static void collectAggFuncCalls(const PredPtr& pred,
                                std::set<std::tuple<std::string, bool, std::string>>& out) {
    if (!pred) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            collectFuncCalls(node.left, out);
            collectFuncCalls(node.right, out);
        } else if constexpr (std::is_same_v<T, Between>) {
            collectFuncCalls(node.expr, out);
            collectFuncCalls(node.low, out);
            collectFuncCalls(node.high, out);
        } else if constexpr (std::is_same_v<T, InList>) {
            collectFuncCalls(node.expr, out);
            for (auto& v : node.values) collectFuncCalls(v, out);
        } else if constexpr (std::is_same_v<T, LogicalAnd>) {
            for (auto& c : node.children) collectAggFuncCalls(c, out);
        } else if constexpr (std::is_same_v<T, LogicalOr>) {
            for (auto& c : node.children) collectAggFuncCalls(c, out);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            collectAggFuncCalls(node.child, out);
        }
        // Like, ExistsPred — no aggregate references
    }, pred->node);
}

// Validate HAVING predicate: must only reference GROUP BY keys and aggregates.
// Returns true if valid; sets *error if invalid.
static bool validateHavingPredicate(const PredPtr& having,
                                     const std::vector<ExprPtr>& groupBy,
                                     const std::vector<SelectTarget>& targets,
                                     std::string* error) {
    if (!having) return true;

    // 1. Collect column references from HAVING
    std::set<std::string> havingCols;
    collectColumns(having, havingCols);

    // 2. Collect aggregate function references from HAVING
    std::set<std::tuple<std::string, bool, std::string>> havingAggs;
    collectAggFuncCalls(having, havingAggs);

    // 3. Build set of GROUP BY column names
    std::set<std::string> groupCols;
    for (const auto& gb : groupBy) {
        if (auto* cr = gb ? std::get_if<ColRef>(&gb->node) : nullptr) {
            groupCols.insert(cr->column);
        }
    }

    // 4. Verify each HAVING column is a GROUP BY column
    for (const auto& col : havingCols) {
        if (!groupCols.count(col)) {
            if (error) *error = "HAVING clause references column '" + col +
                                "' which is not a GROUP BY column.";
            return false;
        }
    }

    // 5. Verify each aggregate reference matches a SELECT target
    for (const auto& [funcName, isStar, innerCol] : havingAggs) {
        bool found = false;
        for (const auto& t : targets) {
            if (!t.isAgg || !t.agg) continue;
            if (t.agg->func == AggFunc::SUM && funcName == "sum") found = true;
            if (t.agg->func == AggFunc::COUNT && funcName == "count") {
                if (isStar && t.agg->isStar) found = true;
                else if (!isStar && !t.agg->isStar) found = true;
            }
            if (t.agg->func == AggFunc::AVG && funcName == "avg") found = true;
            if (t.agg->func == AggFunc::MIN && funcName == "min") found = true;
            if (t.agg->func == AggFunc::MAX && funcName == "max") found = true;
            if (found) {
                // Verify inner expression if non-star
                if (!isStar && t.agg->innerExpr) {
                    if (auto* cr = std::get_if<ColRef>(&t.agg->innerExpr->node)) {
                        if (cr->column != innerCol) found = false;
                    }
                }
                if (found) break;
            }
        }
        if (!found) {
            if (error) {
                std::string desc = funcName;
                if (isStar) desc += "(*)";
                else if (!innerCol.empty()) desc += "(" + innerCol + ")";
                *error = "HAVING clause references aggregate " + desc +
                         " which does not match any SELECT target.";
            }
            return false;
        }
    }

    return true;
}

std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan_impl(
    const AnalyzedQuery& aq, std::string* error) {

    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (aq.tables.size() < 2) return fail("Multi-table planner: query references fewer than 2 tables.");
    for (const auto& t : aq.tables) {
        if (t == "__subquery__")
            return fail("Multi-table planner: subqueries (__subquery__) not supported.");
    }
    if (aq.joins.empty())
        return fail("Multi-table planner: no join conditions found between tables.");

    // HAVING is allowed if GROUP BY is present; validate it references only
    // aggregates and GROUP BY columns (checked when GROUP BY is set up).
    if (aq.having && !aq.hasGroupBy())
        return fail("HAVING requires GROUP BY.");
    
    if (!aq.hasAggregation() && !aq.hasGroupBy()) {
        // No aggregation: build joined materialize path.
        // Fall through to the materialize section below.
    } else {
        // Aggregation path guard checks.
        // ORDER BY and LIMIT are allowed — they'll be handled CPU-side.

        // Aggregations: SUM/COUNT/AVG/MIN/MAX (no DISTINCT).
        for (const auto& t : aq.targets) {
            if (t.isAgg) {
                if (!t.agg) return fail("Malformed aggregate target.");
                // COUNT(DISTINCT) is wired in the single-table grouped planner;
                // the multi-table planner defers to the same aggregate terminal,
                // which will fail with a more specific error if unsupported.
            }
        }
    }

    if (!filtersSupported(aq.filters))
        return fail("WHERE clause contains expressions not supported on GPU.");

    // ---------- Pick probe table ----------
    int probeIdx = 0;
    int bestPrio = -1;
    for (size_t i = 0; i < aq.tables.size(); ++i) {
        int p = multiTableProbePriority(aq.tables[i], aq.schema);
        if (p > bestPrio) { bestPrio = p; probeIdx = (int)i; }
    }
    const std::string& probeTable = aq.tables[probeIdx];

    // ---------- Build join tree ----------
    std::vector<MultiTableTreeNode> nodes;
    if (!multiTableBuildJoinTree(aq, probeIdx, nodes, error)) return std::nullopt;

    // ---------- Collect needed (carried) columns per table ----------
    // Compute BEFORE join validation so we know which edges are SemiJoin-only.
    std::map<std::string, std::set<std::string>> neededByTable;

    auto addNeededFromExpr = [&](const ExprPtr& e) {
        if (!e) return;
        std::set<std::string> cols;
        collectColumns(e, cols);
        for (const auto& c : cols) {
            std::string owner = ownerTableForColumn(aq, c);
            if (!owner.empty()) neededByTable[owner].insert(c);
        }
    };

    for (const auto& t : aq.targets) {
        if (t.isAgg) {
            if (t.agg && !t.agg->isStar) addNeededFromExpr(t.agg->innerExpr);
        } else {
            addNeededFromExpr(t.expr);
        }
    }
    for (const auto& g : aq.groupBy) addNeededFromExpr(g);

    // ---------- Per-table filters ----------
    std::map<std::string, std::vector<PredPtr>> filtersByTable;
    std::vector<PredPtr> crossFilters;  // multi-table filters applied in probe phase
    for (const auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        std::set<std::string> tbls;
        for (const auto& c : cols) {
            std::string owner = ownerTableForColumn(aq, c);
            if (!owner.empty()) tbls.insert(owner);
        }
        if (tbls.size() != 1) {
            crossFilters.push_back(f);
            continue;
        }
        filtersByTable[*tbls.begin()].push_back(f);
    }

    // ---------- Validate non-probe nodes ----------
    // PK uniqueness is required for IndexJoin (value carrying).  SemiJoin-only
    // edges (no carries from this node or its descendants) can use any column
    // since bitmap sets are idempotent.
    for (size_t i = 0; i < nodes.size(); ++i) {
        if ((int)i == probeIdx) continue;
        nodes[i].useHashJoin = nodes[i].composite();
        if (nodes[i].composite()) {
            if (nodes[i].parent != probeIdx) {
                return fail("Multi-table planner: composite-key joins are only "
                            "supported when the build table connects directly to "
                            "the probe table.");
            }
            if (!nodes[i].children.empty()) {
                return fail("Multi-table planner: composite-key build side must be "
                            "a leaf (chained joins on the build side not supported).");
            }
            continue;
        }
        // Check if this node or any descendant has carries; if not, SemiJoin-only.
        bool hasCarries = !neededByTable[nodes[i].table].empty();
        if (!hasCarries) {
            // SemiJoin-only: any column works, skip PK check.
            nodes[i].useHashJoin = false;
            continue;
        }
        auto pk = multiTablePkInfo(nodes[i].table, aq.schema);
        if (!pk)
            return fail("Multi-table planner: table '" + nodes[i].table + "' has no PK descriptor.");
    }

    // ---------- Carried columns: per-non-probe-table local + subtree ----------
    std::map<int, std::vector<CarriedKey>> localCarry;       // owned by this node
    std::map<int, std::vector<CarriedKey>> subtreeCarry;     // local + descendants
    std::function<void(int)> dfs = [&](int u) {
        const std::string& tname = nodes[u].baseTable;
        const std::string& tnameAlias = nodes[u].table;
        // Local carries: columns from this table needed at probe, EXCEPT
        // the join key itself when the value is implicitly carried by
        // probe's lookup key.
        if (u != probeIdx) {
            const auto& need = neededByTable[tname];
            for (const auto& c : need) {
                CarriedKey ck{tname, c};
                DataType ct = aq.schema->columnType(tname, c);
                if (!carriedColumnSupported(ct)) {
                    // Defer: only int/date carried columns.
                    // We'll fail later if such a column is actually required.
                    continue;
                }
                localCarry[u].push_back(ck);
                subtreeCarry[u].push_back(ck);
            }
        }
        for (int c : nodes[u].children) {
            dfs(c);
            auto& subC = subtreeCarry[c];
            subtreeCarry[u].insert(subtreeCarry[u].end(), subC.begin(), subC.end());
        }
    };
    dfs(probeIdx);

        // Validate that every needed non-probe column was supportable.
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType ctype = aq.schema->columnType(tname, c);
                if (!carriedColumnSupported(ctype)) {
                    return fail("Multi-table planner: carried column '" + tname + "." + c +
                                "' has unsupported type '" + std::string(ctype == DataType::CHAR_FIXED ? "CHAR_FIXED" : "?") + "'.");
                }
            }
        }

    // ---------- Assemble plan ----------
    MetalQueryPlan plan;
    plan.name = "ADHOC_MULTI_TABLE";
    const std::string idxVar = "i";

    // BFS order of nodes from probe; build phases are produced in
    // reverse (deepest leaves first) so that every ArrayStore is
    // available when its parent runs.
    std::vector<int> bfsOrder;
    {
        std::vector<bool> seen(nodes.size(), false);
        bfsOrder.push_back(probeIdx); seen[probeIdx] = true;
        for (size_t h = 0; h < bfsOrder.size(); ++h) {
            int u = bfsOrder[h];
            for (int c : nodes[u].children) {
                if (!seen[c]) { seen[c] = true; bfsOrder.push_back(c); }
            }
        }
    }

    // Build phases for each non-probe node in reverse BFS order (leaves first).
    // Scans auto-discover filter/probe columns via the IU chain, but join keys
    // and carried columns must be explicitly loaded (they're referenced in
    // different phases or via extra buffers).
    for (auto it = bfsOrder.rbegin(); it != bfsOrder.rend(); ++it) {
        int u = *it;
        if (u == probeIdx) continue;

        const std::string& tname = nodes[u].baseTable;
        const std::string& tag = nodes[u].table;  // alias for unique naming

        std::set<std::string> scanCols;
        // Join key to parent
        scanCols.insert(nodes[u].keyOnSelf);
        if (nodes[u].composite()) scanCols.insert(nodes[u].keyOnSelf2);
        // Keys for children to probe this table
        for (int c : nodes[u].children) {
            scanCols.insert(nodes[c].keyOnParent);
            if (nodes[c].composite()) scanCols.insert(nodes[c].keyOnParent2);
        }
        // Carried columns (needed for ArrayStore/HashMapBuild value or extra buffers)
        for (const auto& ck : localCarry[u]) scanCols.insert(ck.column);

        auto scan = makeScanForCols(tname, idxVar, scanCols);
        std::unique_ptr<MetalOperator> pipe = std::move(scan);

        // Per-table filters
        std::string filterCond = combineFilters(filtersByTable[tname], idxVar);
        pipe = maybeSelect(std::move(pipe), filterCond);

        // For each child of u, attach probe (BitmapProbe or ArrayLookup
        // for each carried column from that child's subtree).
        for (int c : nodes[u].children) {
            const std::string& probeKey = nodes[c].keyOnParent + "[" + idxVar + "]";
            const auto& subC = subtreeCarry[c];
            // Hash-join children (composite or non-PK) are validated to
            // attach only to the probe, so we don't need a HashMapLookup
            // branch here.
            if (subC.empty()) {
                if (nodes[c].anti) {
                    pipe = std::make_unique<MetalAntiBitmapProbe>(
                        std::move(pipe), "d_bitmap_" + nodes[c].table, probeKey);
                } else {
                    pipe = std::make_unique<MetalBitmapProbe>(
                        std::move(pipe), "d_bitmap_" + nodes[c].table, probeKey);
                }
            } else {
                for (const auto& ck : subC) {
                    pipe = std::make_unique<MetalArrayLookup>(
                        std::move(pipe), ck.storageArray(nodes[c].table),
                        probeKey, ck.varName(), "int", -1);
                }
            }
        }

        // Now emit storage for parent.
        const std::string storeKey = nodes[u].keyOnSelf + "[" + idxVar + "]";

        if (nodes[u].useHashJoin) {
            // Hash-join build (composite key, or non-PK single-column key).
            // The build subtree is restricted to a leaf, so subtreeCarry
            // contains at most this table's local carry list.
            const auto& sub = subtreeCarry[u];
            if (sub.size() > 1) {
                return fail("Multi-table planner: hash-join build can carry "
                            "at most one column to the probe.");
            }
            const std::string mapName = "hm_" + tname;
            const std::string capExpr = "next_pow2((" +
                tableSizeName(tname) + ") * 4 + 16)";
            const std::string k1 = nodes[u].keyOnSelf + "[" + idxVar + "]";
            const std::string k2 = nodes[u].composite()
                ? (nodes[u].keyOnSelf2 + "[" + idxVar + "]")
                : std::string("0u");
            std::string valExpr = "0u";
            if (!sub.empty()) {
                const auto& ck = sub.front();
                DataType origType = aq.schema->columnType(ck.table, ck.column);
                valExpr = encodeCarryValue(origType,
                                           ck.column + "[" + idxVar + "]");
            }
            pipe = std::make_unique<MetalHashMapBuild>(
                std::move(pipe), mapName, k1, k2, valExpr, capExpr);
            appendPhase(plan, "ADHOC_multi_build_" + tag, std::move(pipe));
            continue;
        }

        auto pkU = multiTablePkInfo(tname, aq.schema);
        const std::string sizeSym = pkU->sizeSym;

        const auto& sub = subtreeCarry[u];
        // Always create a bitmap for the SemiJoin filter.
        pipe = std::make_unique<MetalBitmapBuild>(
            std::move(pipe), "d_bitmap_" + tag, storeKey,
            "(" + sizeSym + " + 31) / 32");

        // For non-CHAR_FIXED carries, also create ArrayStores for value propagation.
        for (const auto& ck : sub) {
            DataType ckType = aq.schema->columnType(ck.table, ck.column);
            if (ckType == DataType::CHAR_FIXED) continue;
            std::string valExpr;
            if (ck.table == tname) {
                DataType origType = aq.schema->columnType(tname, ck.column);
                valExpr = encodeCarryValue(origType,
                                           ck.column + "[" + idxVar + "]");
            } else {
                valExpr = ck.varName();
            }
            pipe = std::make_unique<MetalArrayStore>(
                std::move(pipe), ck.storageArray(tname),
                storeKey, valExpr, "int", sizeSym);
        }

        appendPhase(plan, "ADHOC_multi_build_" + tag, std::move(pipe));
    }

    // Build map from build-table to probe-side join key for CHAR_FIXED direct access.
    std::map<std::string, std::string> charFixedJoinKey;  // tableName → keyOnParent column
    for (int c : nodes[probeIdx].children) {
        charFixedJoinKey[aq.tables[c]] = nodes[c].keyOnParent;
    }

    // Collect build-side CHAR_FIXED columns needed by cross-table filters
    // so they can be registered as probe-phase extra buffers.
    std::vector<std::pair<std::string, std::string>> crossExtraCols; // (colName, colType)
    {
        std::set<std::string> crossCols;
        for (auto& cf : crossFilters) collectColumns(cf, crossCols);
        for (const auto& c : crossCols) {
            std::string owner = ownerTableForColumn(aq, c);
            if (!owner.empty() && owner != probeTable) {
                DataType colDt = aq.schema->columnType(owner, c);
                if (colDt == DataType::CHAR_FIXED || colDt == DataType::CHAR1)
                    crossExtraCols.push_back({c, "char"});
                else if (colDt == DataType::INT || colDt == DataType::DATE)
                    crossExtraCols.push_back({c, "int"});
            }
        }
    }

    // ---------- Probe phase ----------
    // Probe scan loads neededByTable columns explicitly and auto-discovers
    // filter/key columns via IU chain. Also add any probe-table CHAR_FIXED
    // columns from cross-filters that the IU chain may miss (OR-branch rewrites
    // change their index expression).
    std::set<std::string> probeScanCols;
    for (const auto& c : neededByTable[probeTable]) probeScanCols.insert(c);
    {
        std::set<std::string> cfCols;
        for (auto& cf : crossFilters) collectColumns(cf, cfCols);
        for (const auto& c : cfCols) {
            std::string owner = ownerTableForColumn(aq, c);
            if (owner == probeTable) probeScanCols.insert(c);
        }
    }
    auto probeScan = makeScanForCols(probeTable, idxVar, probeScanCols);
    std::unique_ptr<MetalOperator> probePipe = std::move(probeScan);

    // Probe's own filters.
    probePipe = maybeSelect(std::move(probePipe),
                            combineFilters(filtersByTable[probeTable], idxVar));

    // Probe each direct child.
    std::map<CarriedKey, std::string> carryVar; // for expression rewrite
    for (int c : nodes[probeIdx].children) {
        const std::string& probeKey = nodes[c].keyOnParent + "[" + idxVar + "]";
        const auto& subC = subtreeCarry[c];

        if (nodes[c].useHashJoin) {
            // HashJoin probe.  Capacity expression must match the
            // build-phase choice exactly so that resolve() yields the
            // same value here.
            const std::string mapName = "hm_" + nodes[c].table;
            const std::string capExpr = "next_pow2((" +
                tableSizeName(nodes[c].baseTable) + ") * 4 + 16)";
            const std::string k2 = nodes[c].composite()
                ? (nodes[c].keyOnParent2 + "[" + idxVar + "]")
                : std::string("0u");
            if (subC.empty()) {
                // Semi-join: lookup with a discardable result variable.
                std::string dummy = "_hjsemi_" + aq.tables[c];
                probePipe = std::make_unique<MetalHashMapLookup>(
                    std::move(probePipe), mapName, probeKey, k2, capExpr,
                    dummy, "uint");
            } else {
                const auto& ck = subC.front();
                probePipe = std::make_unique<MetalHashMapLookup>(
                    std::move(probePipe), mapName, probeKey, k2, capExpr,
                    ck.varName(), "int");
                carryVar[ck] = ck.varName();
            }
            continue;
        }

        // Always probe the bitmap for SemiJoin filtering.
        if (nodes[c].anti) {
            probePipe = std::make_unique<MetalAntiBitmapProbe>(
                std::move(probePipe), "d_bitmap_" + nodes[c].table, probeKey);
            } else {
                probePipe = std::make_unique<MetalBitmapProbe>(
                std::move(probePipe), "d_bitmap_" + nodes[c].table, probeKey);
        }

        // For non-CHAR_FIXED carries, create ArrayLookups for value propagation.
        for (const auto& ck : subC) {
            DataType ckType = aq.schema->columnType(ck.table, ck.column);
            if (ckType == DataType::CHAR_FIXED) continue;
            probePipe = std::make_unique<MetalArrayLookup>(
                std::move(probePipe), ck.storageArray(aq.tables[c]),
                probeKey, ck.varName(), "int", -1);
            carryVar[ck] = ck.varName();
        }
    }

    // Apply cross-table filters (e.g. Q19's OR branches) after all probes.
    // Build-side column references are rewritten to use the join key index.
    if (!crossFilters.empty()) {
        std::string cond = combineFilters(crossFilters, idxVar);
        cond = rewriteForProbe(cond, idxVar, carryVar, aq.schema);
        // Rewrite build-side column indices for CHAR_FIXED and INT columns.
        for (const auto& [tname, jk] : charFixedJoinKey) {
            // CHAR_FIXED: `col[i * width]` → `col[jk[i] * width]`
            std::string fromIdx = idxVar + " *";
            std::string toIdx = jk + "[" + idxVar + "] *";
            size_t pos = 0;
            while ((pos = cond.find(fromIdx, pos)) != std::string::npos) {
                cond.replace(pos, fromIdx.size(), toIdx);
                pos += toIdx.size();
            }
            // INT/DATE: `col[i]` → `col[jk[i]]`
            // Collect INT column names from crossExtraCols
            for (auto& [colName, colType] : crossExtraCols) {
                if (colType != "int") continue;
                std::string from = colName + "[" + idxVar + "]";
                std::string to = colName + "[" + jk + "[" + idxVar + "]]";
                pos = 0;
                while ((pos = cond.find(from, pos)) != std::string::npos) {
                    cond.replace(pos, from.size(), to);
                    pos += to.size();
                }
            }
        }
        probePipe = maybeSelect(std::move(probePipe), cond);
    }

    // ---------- Terminal operators ----------
    // Materialize path: no aggregation, no GROUP BY → emit joined rows directly.
    if (!aq.hasAggregation() && !aq.hasGroupBy()) {
        MetalQueryPlan::CpuSort cpuSort;
        cpuSort.limit = aq.limit;
        for (const auto& order : aq.orderBy) {
            auto column = orderColumnForExpr(order.expr, aq.targets);
            if (!column) return fail("ORDER BY column not found in SELECT targets.");
            cpuSort.keys.push_back({*column, order.descending});
        }

        std::set<std::string> matCols;
        for (const auto& target : aq.targets) {
            if (!target.expr || !materializeExprSupported(target.expr))
                return fail("Materialize expression not supported.");
            collectColumns(target.expr, matCols);
        }

        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(probePipe), "d_adhoc_multi_result_count", "1");
        const std::string outputSize = tableSizeName(probeTable);

        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            DataType type = inferExprDataType(target.expr);
            std::string displayName = displayNameForTarget(target, ti);
            std::string bufferName = "d_adhoc_multi_" + std::to_string(ti) + "_" + sanitizeIdentifier(displayName);
            int stringLen = fixedStringLenForExpr(target.expr, aq.schema);
            std::string sizeExpr = outputSize;
            if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
            std::string expr = materializeValueExpr(target.expr, idxVar, aq.schema);
            // Rewrite non-probe column references to carried variables
            expr = rewriteForProbe(expr, idxVar, carryVar, aq.schema);
            // CHAR1 from non-probe tables: after rewrite, the carry variable is a
            // scalar.  Reset stringLen to 0 so the materialize operator emits a
            // scalar assignment instead of a pointer-indexed [_ci] loop.
            if (auto* col = target.expr ? std::get_if<ColRef>(&target.expr->node) : nullptr) {
                if (col->dataType == DataType::CHAR1) {
                    std::string owner = ownerTableForColumn(aq, col->column);
                    if (!owner.empty() && owner != probeTable) {
                        stringLen = 0;
                    }
                }
                // For CHAR_FIXED columns from build-side tables, rewrite the index from
                // `colName + i * width` to `colName + joinKey[i] * width` so the column
                // buffer is accessed by the join key rather than the probe row index.
                if (col->dataType == DataType::CHAR_FIXED) {
                    std::string owner = ownerTableForColumn(aq, col->column);
                    if (!owner.empty() && owner != probeTable) {
                        auto jkIt = charFixedJoinKey.find(owner);
                        if (jkIt != charFixedJoinKey.end()) {
                            int len = fixedStringLenForExpr(target.expr, aq.schema);
                            std::string from = col->column + " + " + idxVar + " * " + std::to_string(len);
                            std::string to = col->column + " + " + jkIt->second + "[" + idxVar + "] * " + std::to_string(len);
                            size_t pos = expr.find(from);
                            if (pos != std::string::npos)
                                expr.replace(pos, from.size(), to);
                        }
                    }
                }
            }
            materialize->addColumn(bufferName, metalTypeForDataType(type),
                                   expr, displayName, sizeExpr, stringLen);
        }

        if (!cpuSort.keys.empty() || cpuSort.limit >= 0) plan.cpuSort = cpuSort;
        appendPhase(plan, "ADHOC_multi_materialize", std::move(materialize));
        // Register build-side CHAR_FIXED column buffers as read-only extra buffers.
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType cdType = aq.schema->columnType(tname, c);
                if (cdType == DataType::CHAR_FIXED) {
                    plan.phases.back().extraBuffers.push_back({c, "char", true, false});
                }
            }
        }
        // Also add columns needed by cross-table filters.
        for (auto& [c, t] : crossExtraCols) {
            plan.phases.back().extraBuffers.push_back({c, t, true, false});
        }
        return plan;
    }

    if (!aq.hasGroupBy()) {
        // Scalar reduction over probe rows that survived all joins.
        auto reduce = std::make_unique<MetalTGReduce>(std::move(probePipe), "d_adhoc_multi_scalar");
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            if (!target.isAgg || !target.agg)
                return fail("Non-aggregate target without GROUP BY in multi-table query.");
            std::string alias = displayNameForTarget(target, ti);
            std::string accName = "a" + std::to_string(ti) + "_" + sanitizeIdentifier(alias);
            AggFunc func = target.agg->func;

            if (func == AggFunc::COUNT) {
                int idx = reduce->addAccumulator(accName, "1", "long");
                reduce->setAccumulatorResultAlias(alias, idx, 0);
                continue;
            }

            if (!target.agg->innerExpr || !exprSupported(target.agg->innerExpr, false))
                return fail("Aggregate expression not supported on GPU.");
            if (!isNumericLike(inferExprDataType(target.agg->innerExpr)))
                return fail("Aggregate expression must be numeric.");

            std::string raw = exprToMetal(target.agg->innerExpr, idxVar, aq.schema);
            std::string finalExpr = rewriteForProbe(raw, idxVar, carryVar, aq.schema);

            if (func == AggFunc::AVG) {
                int sumIdx = reduce->addAccumulator(accName + "_sum", finalExpr, "float");
                int cntIdx = reduce->addAccumulator(accName + "_count", "1.0f", "float");
                reduce->setAverageResultAlias(alias, sumIdx, cntIdx, 0);
            } else {
                DataType vt = inferExprDataType(target.agg->innerExpr);
                std::string outType = (vt == DataType::FLOAT) ? "float" : "long";
                MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
                if (func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
                else if (func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
                if (op != MetalTGReduce::ReduceOp::SUM && vt != DataType::FLOAT) outType = "int";
                int idx = reduce->addAccumulator(accName, finalExpr, outType, "", "", op);
                reduce->setAccumulatorResultAlias(alias, idx, 0);
            }
        }
        auto& phaseRef = appendPhase(plan, "ADHOC_multi_probe_scalar", std::move(reduce));
        for (auto& [c, t] : crossExtraCols) {
            phaseRef.extraBuffers.push_back({c, "char", true, false});
        }
        if (aq.limit >= 0) {
            plan.cpuSort = MetalQueryPlan::CpuSort{{}, aq.limit};
        }
        return plan;
    }

    // ---------- Grouped aggregation ----------
    // Try GPU KeyedAgg first; if the group keys are all known small-int or
    // CHAR1 domains with total buckets ≤ 4096, use GPU. Otherwise fall
    // back to materialize + host-side GROUP BY.
    struct GpuKeyDesc { std::string keyExpr; int numValues; int stride; std::string colName; DataType colType; };
    std::vector<GpuKeyDesc> gpuKeys;
    int totalBuckets = 1;
    bool canUseGpuKeyedAgg = true;

    for (size_t ki = 0; ki < aq.groupBy.size() && canUseGpuKeyedAgg; ++ki) {
        auto* gc = std::get_if<ColRef>(&aq.groupBy[ki]->node);
        if (!gc) { canUseGpuKeyedAgg = false; break; }

        GpuKeyDesc kd;
        kd.colName = gc->column;
        kd.colType = gc->dataType;

        if (gc->dataType == DataType::CHAR1) {
            std::string idxVar = "i";
            kd.keyExpr = char1BucketExpr(*gc, idxVar, kd.numValues);
            if (kd.numValues == 0) { canUseGpuKeyedAgg = false; break; }
        } else {
            auto d = smallIntGroupDomain(*gc);
            if (!d || d->maxValue < d->minValue) { canUseGpuKeyedAgg = false; break; }
            kd.numValues = d->maxValue - d->minValue + 1;
            std::string groupValue = gc->column + "[i]";
            std::string keySource;
            if (gc->table == probeTable) {
                keySource = groupValue;
            } else {
                CarriedKey ck{gc->table, gc->column};
                auto it = carryVar.find(ck);
                if (it == carryVar.end()) { canUseGpuKeyedAgg = false; break; }
                keySource = it->second;
            }
            if (d->minValue != 0)
                kd.keyExpr = "(" + keySource + " - " + std::to_string(d->minValue) + ")";
            else
                kd.keyExpr = keySource;
            kd.keyExpr = "clamp(" + kd.keyExpr + ", 0, " + std::to_string(kd.numValues - 1) + ")";
        }
        kd.stride = totalBuckets;
        totalBuckets *= kd.numValues;
        gpuKeys.push_back(kd);
    }

    if (totalBuckets > 4096) canUseGpuKeyedAgg = false;

    if (!canUseGpuKeyedAgg) {
        // ---------- MaterializeAgg fallback: emit raw rows on GPU, group on host ----------
        // Build a materialize plan that outputs group-by keys and aggregate inputs.
        // Host-side GROUP BY happens after GPU result collection.
        MetalQueryPlan::CpuSort cpuSort;
        cpuSort.limit = aq.limit;
        for (const auto& order : aq.orderBy) {
            auto column = orderColumnForExpr(order.expr, aq.targets);
            if (!column) return fail("ORDER BY column not found in SELECT targets.");
            cpuSort.keys.push_back({*column, order.descending});
        }

        // Build CpuGroupBy metadata.
        MetalQueryPlan::CpuGroupBy cpuGB;
        for (const auto& g : aq.groupBy) {
            auto* gcRef = std::get_if<ColRef>(&g->node);
            if (gcRef) {
                cpuGB.keyColumns.push_back(displayNameForTargetByCol(aq, *gcRef));
            } else {
                // FuncCall GROUP BY (resolved from subquery alias):
                // match by position against non-aggregate targets.
                for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
                    if (!aq.targets[ti].isAgg) {
                        cpuGB.keyColumns.push_back(displayNameForTarget(aq.targets[ti], ti));
                        break;
                    }
                }
            }
        }
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            if (target.isAgg && target.agg) {
                cpuGB.aggColumns.push_back(displayNameForTarget(target, ti));
                cpuGB.aggFuncs.push_back(aggFuncName(target.agg->func));
            }
        }

        // Build materialize for all needed columns.
        std::set<std::string> matCols;
        for (const auto& target : aq.targets) {
            if (target.expr) collectColumns(target.expr, matCols);
        }
        for (const auto& g : aq.groupBy) collectColumns(g, matCols);

        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(probePipe), "d_adhoc_multi_result_count", "1");
        const std::string outputSize = tableSizeName(probeTable);

        // Emit group-by keys and raw aggregate input values as columns.
        int matColIdx = 0;
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            DataType type = inferExprDataType(target.expr);
            std::string displayName = displayNameForTarget(target, ti);
            std::string bufferName = "d_adhoc_matgb_" + std::to_string(matColIdx) + "_" + sanitizeIdentifier(displayName);
            int stringLen = fixedStringLenForExpr(target.expr, aq.schema);
            std::string sizeExpr = outputSize;
            if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);

            std::string expr;
            if (target.isAgg && target.agg) {
                // For aggregates in materialize mode, emit the raw input value
                // (COUNT → 1.0, others → inner expression) — host will aggregate.
                if (target.agg->func == AggFunc::COUNT || target.agg->isStar) {
                    expr = "1.0f";
                    type = DataType::FLOAT;
                } else if (target.agg->innerExpr) {
                    expr = exprToMetal(target.agg->innerExpr, idxVar);
                    expr = rewriteForProbe(expr, idxVar, carryVar, aq.schema);
                    type = DataType::FLOAT;
                } else {
                    expr = "0";
                }
            } else {
                expr = materializeValueExpr(target.expr, idxVar, aq.schema);
                expr = rewriteForProbe(expr, idxVar, carryVar, aq.schema);
            }

            // Handle CHAR1/CHAR_FIXED from non-probe tables
            if (auto* col = target.expr ? std::get_if<ColRef>(&target.expr->node) : nullptr) {
                if (col->dataType == DataType::CHAR1) {
                    std::string owner = ownerTableForColumn(aq, col->column);
                    if (!owner.empty() && owner != probeTable) {
                        stringLen = 0; // scalar carry
                    }
                }
                if (col->dataType == DataType::CHAR_FIXED) {
                    std::string owner = ownerTableForColumn(aq, col->column);
                    if (!owner.empty() && owner != probeTable) {
                        auto jkIt = charFixedJoinKey.find(owner);
                        if (jkIt != charFixedJoinKey.end()) {
                            int len = fixedStringLenForExpr(target.expr, aq.schema);
                            std::string from = col->column + " + " + idxVar + " * " + std::to_string(len);
                            std::string to = col->column + " + " + jkIt->second + "[" + idxVar + "] * " + std::to_string(len);
                            size_t pos = expr.find(from);
                            if (pos != std::string::npos)
                                expr.replace(pos, from.size(), to);
                        }
                    }
                }
            }

            materialize->addColumn(bufferName, metalTypeForDataType(type),
                                   expr, displayName, sizeExpr, stringLen);
            matColIdx++;
        }

        if (!cpuSort.keys.empty() || cpuSort.limit >= 0) plan.cpuSort = cpuSort;
        plan.cpuGroupBy = cpuGB;
        appendPhase(plan, "ADHOC_multi_matgb", std::move(materialize));

        // Register build-side CHAR_FIXED extra buffers on the newly created phase.
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType cdType = aq.schema->columnType(tname, c);
                if (cdType == DataType::CHAR_FIXED) {
                    plan.phases.back().extraBuffers.push_back({c, "char", true, false});
                }
            }
        }
        return plan;
    }

    // --- GPU KeyedAgg path ---
    // Build flat bucket expression from all group keys.
    std::string bucketExpr = "(" + gpuKeys[0].keyExpr + ")";
    for (size_t ki = 1; ki < gpuKeys.size(); ++ki) {
        bucketExpr = "(" + bucketExpr + " + (" + gpuKeys[ki].keyExpr + ") * " +
                     std::to_string(gpuKeys[ki].stride) + ")";
    }
    int numBuckets = totalBuckets;

    // Build multi-key decode info for result collection.
    std::vector<GroupKeyDecode> decodeInfo;
    std::vector<std::string> keyDisplayNames;
    for (size_t ki = 0; ki < gpuKeys.size(); ++ki) {
        GroupKeyDecode d;
        d.name = gpuKeys[ki].colName;
        d.numValues = gpuKeys[ki].numValues;
        d.stride = gpuKeys[ki].stride;
        auto* gc = std::get_if<ColRef>(&aq.groupBy[ki]->node);
        if (gc && gc->dataType == DataType::CHAR1) {
            if (gc->column == "l_returnflag") d.charMap = {'A', 'N', 'R'};
            else if (gc->column == "l_linestatus") d.charMap = {'F', 'O'};
        } else {
            auto sd = smallIntGroupDomain(*gc);
            d.keyBase = sd ? sd->minValue : 0;
        }
        decodeInfo.push_back(d);
        std::string kn = gpuKeys[ki].colName;
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            if (!aq.targets[ti].isAgg && aq.targets[ti].expr &&
                std::holds_alternative<ColRef>(aq.targets[ti].expr->node)) {
                auto* tcol = std::get_if<ColRef>(&aq.targets[ti].expr->node);
                if (tcol->column == kn) { kn = displayNameForTarget(aq.targets[ti], ti); break; }
            }
        }
        keyDisplayNames.push_back(kn);
    }

    // Layout aggregates and slot count.
    struct PendingAgg {
        std::string display;
        std::string name;
        int offset = 0;
        std::string valueExpr;
        bool isLongPair = false;
        int scaleDown = 0;
        bool isFloatSum = false;
        bool isMinMax = false;
        std::string atomicOp = "add";
        std::string funcName;
        std::string innerColumn;
    };
    std::vector<PendingAgg> pending;
    int valuesPerBucket = 0;

    // COUNT(DISTINCT) tracking for popcount phases.
    struct DistinctEntry {
        std::string displayName;
        std::string colExpr;
        std::string maxValExpr;
        int offset;
    };
    std::vector<DistinctEntry> distinctEntries;

    auto extractInnerCol = [&](const ExprPtr& inner) -> std::string {
        if (!inner) return "";
        if (auto* cr = std::get_if<ColRef>(&inner->node)) return cr->column;
        return "";
    };

    for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
        const auto& target = aq.targets[ti];
        if (!target.isAgg) continue;
        if (!target.agg) return fail("Malformed aggregate target.");
        AggFunc func = target.agg->func;
        std::string display = displayNameForTarget(target, ti);
        std::string sname = sanitizeIdentifier(display);

        if (func == AggFunc::COUNT) {
            PendingAgg p;
            p.display = display;
            p.name = "a" + std::to_string(ti) + "_" + sname;
            p.offset = valuesPerBucket++;
            p.valueExpr = "1u";
            p.atomicOp = "add";
            p.funcName = "COUNT";
            p.innerColumn = "";
            pending.push_back(std::move(p));
            continue;
        }

        if (!target.agg->innerExpr || !exprSupported(target.agg->innerExpr, false))
            return fail("Aggregate expression not supported on GPU.");
        DataType vt = inferExprDataType(target.agg->innerExpr);

        std::string raw = exprToMetal(target.agg->innerExpr, idxVar);
        std::string finalExpr = rewriteForProbe(raw, idxVar, carryVar, aq.schema);

        if (func == AggFunc::AVG) {
            bool isFloat = (vt == DataType::FLOAT);
            PendingAgg sumA;
            sumA.display = display;
            sumA.name = "a" + std::to_string(ti) + "_" + sname + "_sum";
            sumA.offset = valuesPerBucket;
            sumA.valueExpr = finalExpr;
            sumA.scaleDown = -1;
            sumA.atomicOp = "add";
            sumA.funcName = "AVG";
            sumA.innerColumn = extractInnerCol(target.agg->innerExpr);
            if (isFloat) {
                sumA.isFloatSum = true;
                valuesPerBucket += 1;
            } else {
                sumA.isLongPair = true;
                valuesPerBucket += 2;
            }
            pending.push_back(std::move(sumA));

            PendingAgg cntA;
            cntA.display = display + "_cnt";
            cntA.name = "a" + std::to_string(ti) + "_" + sname + "_cnt";
            cntA.offset = valuesPerBucket++;
            cntA.valueExpr = "1u";
            cntA.atomicOp = "add";
            cntA.funcName = "AVG";
            cntA.innerColumn = "";
            pending.push_back(std::move(cntA));
            continue;
        }

        if (func == AggFunc::MIN || func == AggFunc::MAX) {
            PendingAgg p;
            p.display = display;
            p.name = "a" + std::to_string(ti) + "_" + sname;
            p.offset = valuesPerBucket++;
            p.valueExpr = finalExpr;
            p.atomicOp = (func == AggFunc::MIN) ? "min" : "max";
            p.isMinMax = true;
            p.funcName = (func == AggFunc::MIN) ? "MIN" : "MAX";
            p.innerColumn = extractInnerCol(target.agg->innerExpr);
            if (vt == DataType::FLOAT) p.isFloatSum = true;
            pending.push_back(std::move(p));
            continue;
        }

        // COUNT(DISTINCT) — same pattern as single-table grouped builder.
        if (func == AggFunc::COUNT_DISTINCT) {
            auto* innerCol = std::get_if<ColRef>(&target.agg->innerExpr->node);
            if (!innerCol)
                return fail("COUNT(DISTINCT) inner expression must be a column reference.");
            DataType ct = inferExprDataType(target.agg->innerExpr);
            if (ct != DataType::INT && ct != DataType::DATE)
                return fail("COUNT(DISTINCT) only supports integer/date columns.");
            std::string maxExpr;
            auto gd = aq.schema->groupDomain(innerCol->table, innerCol->column);
            if (gd && gd->maxValue >= 0) maxExpr = std::to_string(gd->maxValue + 1);
            if (maxExpr.empty()) {
                auto ms = aq.schema->maxKeySymbol(innerCol->table);
                if (!ms.empty()) maxExpr = ms + " + 1";
            }
            if (maxExpr.empty())
                return fail("COUNT(DISTINCT) on column '" + innerCol->column + "' — no known max value for bitmap sizing.");

            PendingAgg p;
            p.display = display;
            p.name = "a" + std::to_string(ti) + "_" + sname;
            p.offset = valuesPerBucket++;
            // Stash maxExpr in valueExpr for the distinct-entry loop to read.
            p.valueExpr = finalExpr + "\x01" + maxExpr; // sentinel separator
            p.scaleDown = -2;  // COUNT(DISTINCT) sentinel
            p.funcName = "COUNT_DISTINCT";
            p.innerColumn = innerCol->column;
            pending.push_back(std::move(p));
            continue;
        }

        if (func != AggFunc::SUM)
            return fail("Aggregate function not supported by multi-table planner.");

        PendingAgg p;
        p.display = display;
        p.name = "a" + std::to_string(ti) + "_" + sname;
        p.offset = valuesPerBucket;
        p.valueExpr = finalExpr;
        p.atomicOp = "add";
        p.funcName = "SUM";
        p.innerColumn = extractInnerCol(target.agg->innerExpr);
        if (vt == DataType::FLOAT) {
            p.isFloatSum = true;
            valuesPerBucket += 1;
        } else {
            p.isLongPair = true;
            valuesPerBucket += 2;
        }
        pending.push_back(std::move(p));
    }

    if (pending.empty())
        return fail("Multi-table planner: no aggregate output columns.");

    // Build keyed-agg operator.
    auto agg = std::make_unique<MetalKeyedAgg>(
        std::move(probePipe), "d_adhoc_multi_group_aggs", bucketExpr,
        numBuckets, valuesPerBucket,
        std::to_string(numBuckets * valuesPerBucket));

    // Decode info for group keys (multi-key support).
    agg->setMultiKeyResult(keyDisplayNames, decodeInfo, numBuckets);

    for (const auto& p : pending) {
        if (p.scaleDown == -2) {
            // COUNT(DISTINCT): valueExpr contains "colExpr\x01maxExpr"
            auto sep = p.valueExpr.find('\x01');
            std::string colExpr = p.valueExpr.substr(0, sep);
            std::string maxExpr = (sep != std::string::npos) ? p.valueExpr.substr(sep + 1) : "";
            std::string bmpOutput = "d_adhoc_multi_distinct_" + std::to_string(distinctEntries.size());
            agg->addDistinctBitmap(bmpOutput, colExpr, maxExpr);
            distinctEntries.push_back({p.display, colExpr, maxExpr, p.offset});
            // Also add a zero-valued aggregate slot placeholder.
            agg->addAggregateWithMeta(p.display, p.offset, "0u",
                                      "add", false, 0, false, false,
                                      p.funcName, p.innerColumn);
        } else {
            agg->addAggregateWithMeta(p.display, p.offset, p.valueExpr,
                                      p.atomicOp, p.isLongPair, p.scaleDown,
                                      p.isFloatSum, p.isMinMax,
                                      p.funcName, p.innerColumn);
        }
    }

    // Set HAVING predicate if present
    if (aq.having) {
        if (!validateHavingPredicate(aq.having, aq.groupBy, aq.targets, error))
            return std::nullopt;
        agg->setHaving(aq.having);
    }

    appendPhase(plan, "ADHOC_multi_probe_group", std::move(agg));

    // Add bitmap popcount phases for each COUNT(DISTINCT).
    for (size_t di = 0; di < distinctEntries.size(); ++di) {
        const auto& de = distinctEntries[di];
        std::string bmpName = "d_distinct_bmp_d_adhoc_multi_distinct_" + std::to_string(di);
        std::string bmpOutput = "d_adhoc_multi_distinct_" + std::to_string(di);
        std::string strideExpr = "((" + de.maxValExpr + " + 32) / 32)";
        auto popcnt = std::make_unique<MetalBitmapPopcount>(
            bmpName, bmpOutput, std::to_string(numBuckets), strideExpr);
        appendPhase(plan, "ADHOC_multi_popcount_" + std::to_string(di), std::move(popcnt));
    }

    // CPU-side ORDER BY + LIMIT for grouped results.
    if (!aq.orderBy.empty() || aq.limit >= 0) {
        MetalQueryPlan::CpuSort cpuSort;
        cpuSort.limit = aq.limit;
        for (const auto& order : aq.orderBy) {
            auto column = orderColumnForExpr(order.expr, aq.targets);
            if (column) cpuSort.keys.push_back({*column, order.descending});
        }
        plan.cpuSort = cpuSort;
    }
    return plan;
}

} // namespace

std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.isSingleTable()) return fail("Single-table planner: query references multiple tables.");
    if (aq.tables.empty()) return fail("Single-table planner: no tables in FROM clause.");
    if (aq.tables[0] == "__subquery__") return fail("Single-table planner: subqueries not supported.");
    if (!filtersSupported(aq.filters)) return fail("Single-table planner: WHERE clause contains expressions not supported on GPU.");

    // Cascade through sub-builders. Each sub-builder may return nullopt
    // without an error when it simply doesn't match the pattern (e.g.
    // scalar agg when GROUP BY is present). Only the last builder in the
    // chain writes the final error.
    std::string subError;
    if (auto scalar = buildScalarAggPlan(aq, &subError)) return scalar;
    if (auto grouped = buildGroupedAggPlan(aq, &subError)) return grouped;
    return buildMaterializePlan(aq, error);
}

std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan(
    const AnalyzedQuery& aq, std::string* error) {
    return buildGenericMultiTableAdhocPlan_impl(aq, error);
}

} // namespace codegen