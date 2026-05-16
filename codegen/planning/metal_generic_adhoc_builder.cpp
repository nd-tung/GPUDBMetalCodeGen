#include "generic_scalar_lookup_helpers.h"
#include "metal_generic_sql_physical_ops.h"
#include "metal_plan_common.h"
#include "metal_generic_executor.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../../third_party/nlohmann/json.hpp"

namespace codegen {

namespace {

// Walk expression tree to collect aggregate FuncCall nodes and check for division.
bool isAggregateFuncCallName(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return name == "sum" || name == "avg" || name == "count" ||
           name == "min" || name == "max";
}

struct AggCallWalker {
    static void walk(const ExprPtr& e, std::vector<const FuncCall*>& out, bool& div) {
        if (!e) return;
        if (auto* fc = std::get_if<FuncCall>(&e->node)) {
            if (isAggregateFuncCallName(fc->name)) {
                out.push_back(fc); return;
            }
            for (auto& a : fc->args) walk(a, out, div);
            return;
        }
        if (auto* be = std::get_if<BinaryExpr>(&e->node)) {
            walk(be->left, out, div);
            walk(be->right, out, div);
            if (be->op == ExprOp::DIV) div = true;
        } else if (auto* cw = std::get_if<CaseWhen>(&e->node)) {
            for (auto& br : cw->branches) walk(br.result, out, div);
            if (cw->elseResult) walk(cw->elseResult, out, div);
        }
    }
};

bool scalarAggProjectionExprSupported(const ExprPtr& expr) {
    if (!expr) return false;
    if (auto* lit = std::get_if<Literal>(&expr->node)) {
        return std::holds_alternative<int>(lit->value) ||
               std::holds_alternative<float>(lit->value);
    }
    if (auto* fc = std::get_if<FuncCall>(&expr->node))
        return isAggregateFuncCallName(fc->name);
    if (auto* be = std::get_if<BinaryExpr>(&expr->node)) {
        return scalarAggProjectionExprSupported(be->left) &&
               scalarAggProjectionExprSupported(be->right);
    }
    return false;
}

ExprPtr scalarProjectionExprForSingleAgg(const SelectTarget& target) {
    std::vector<const FuncCall*> aggCalls;
    bool hasDivision = false;
    AggCallWalker::walk(target.expr, aggCalls, hasDivision);
    (void)hasDivision;
    return aggCalls.size() == 1 && scalarAggProjectionExprSupported(target.expr)
        ? target.expr
        : nullptr;
}

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

std::string displayNameForExpr(const ExprPtr& expr) {
    if (!expr) return "expr";
    if (auto* col = std::get_if<ColRef>(&expr->node)) return col->column;
    if (auto* lit = std::get_if<Literal>(&expr->node)) {
        if (auto* i = std::get_if<int>(&lit->value)) return std::to_string(*i);
        if (auto* f = std::get_if<float>(&lit->value)) {
            std::ostringstream oss;
            oss << *f;
            return oss.str();
        }
        if (auto* s = std::get_if<std::string>(&lit->value)) return "'" + *s + "'";
    }
    if (auto* func = std::get_if<FuncCall>(&expr->node)) {
        std::string out = func->name + "(";
        for (size_t i = 0; i < func->args.size(); ++i) {
            if (i) out += ",";
            out += displayNameForExpr(func->args[i]);
        }
        out += ")";
        return out;
    }
    return "expr";
}

std::string displayNameForTarget(const SelectTarget& target, size_t targetIndex) {
    if (!target.alias.empty()) return target.alias;
    if (target.isAgg && target.agg) {
        std::string base = target.agg->isStar ? "*" : displayNameForExpr(target.agg->innerExpr);
        return aggName(target.agg->func) + "(" + base + ")";
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
            if (col->column == ref.column && col->table == ref.table) {
                // For multi-instance tables, also match on table alias
                if (!ref.tableAlias.empty() && col->tableAlias != ref.tableAlias) continue;
                return displayNameForTarget(t, i);
            }
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

int numericScaleForExpr(const ExprPtr& expr, const SchemaProvider* schema) {
    if (!schema || !expr) return 0;
    if (auto* col = std::get_if<ColRef>(&expr->node))
        return schema->numericScale(col->table, col->column);
    return 0;
}

std::string distinctDomainSymbolForExpr(const ExprPtr& expr,
                                        const SchemaProvider* schema) {
    if (!schema || !expr) return "";
    auto* col = std::get_if<ColRef>(&expr->node);
    if (!col) return "";
    if (auto gd = schema->groupDomain(col->table, col->column))
        return std::to_string(gd->maxValue + 1);
    return schema->distinctDomainSymbol(col->table, col->column);
}

std::string scaledLongExpr(const std::string& rawExpr, int scale) {
    return "(long)round((" + rawExpr + ") * " + std::to_string(scale) + ".0f)";
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
        std::optional<std::string> unqualifiedMatch;
        const bool orderQualified = !orderCol->table.empty() || !orderCol->tableAlias.empty();
        for (size_t i = 0; i < targets.size(); ++i) {
            const auto& target = targets[i];
            std::string displayName = displayNameForTarget(target, i);
            if (displayName == orderCol->column) return displayName;
            if (auto* targetCol = target.expr ? std::get_if<ColRef>(&target.expr->node) : nullptr) {
                if (targetCol->column == orderCol->column) {
                    bool tableMatches = true;
                    if (!orderCol->table.empty() && targetCol->table != orderCol->table)
                        tableMatches = false;
                    if (!orderCol->tableAlias.empty() && targetCol->tableAlias != orderCol->tableAlias)
                        tableMatches = false;
                    if (orderQualified) {
                        if (tableMatches) return displayName;
                    } else if (!unqualifiedMatch) {
                        unqualifiedMatch = displayName;
                    }
                }
            }
        }
        if (unqualifiedMatch) return unqualifiedMatch;
    }
    // Non-ColRef ORDER BY (e.g. FuncCall after subquery alias resolution):
    // match by position — the Nth non-ColRef ORDER BY item maps to the
    // Nth non-aggregate target.  The caller must pass the position.
    return std::nullopt; // caller must handle via position
}

// Resolve an ORDER BY expression to a SELECT target display name.
// For non-ColRef expressions (FuncCall, BinaryExpr after subquery alias
// resolution), uses positional matching: the Nth non-ColRef ORDER BY item
// maps to the Nth non-aggregate SELECT target.
static std::optional<std::string> resolveOrderColumn(const ExprPtr& expr,
                                                     int orderIdx,
                                                     const std::vector<OrderByItem>& orderBy,
                                                     const std::vector<SelectTarget>& targets) {
    (void)orderBy;
    auto col = orderColumnForExpr(expr, targets);
    if (col) return col;
    if (expr) {
        const std::string orderExprName = displayNameForExpr(expr);
        for (size_t i = 0; i < targets.size(); ++i) {
            if (targets[i].expr && displayNameForExpr(targets[i].expr) == orderExprName)
                return displayNameForTarget(targets[i], i);
        }
    }
    // Positional fallback: the Nth ORDER BY item maps to the Nth SELECT target
    // (ColRef items are matched via the column name above; non-ColRef items
    //  and unresolved ColRefs fall through here).
    if (orderIdx >= 0 && orderIdx < (int)targets.size())
        return displayNameForTarget(targets[orderIdx], orderIdx);
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
            std::string aliasPrefix;
            if (!col->tableAlias.empty()) aliasPrefix = "/*" + col->tableAlias + "*/";
            return aliasPrefix + col->column + " + " + idxVar + " * " + std::to_string(len);
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
            if (inCol && (inCol->dataType == DataType::CHAR_FIXED || inCol->dataType == DataType::CHAR1)) {
                return std::all_of(node.values.begin(), node.values.end(), [](const ExprPtr& value) {
                    auto* lit = value ? std::get_if<Literal>(&value->node) : nullptr;
                    return lit && std::holds_alternative<std::string>(lit->value);
                });
            }
            if (!exprSupported(node.expr, false)) {
                // expr may be CHAR_FIXED which exprSupported rejects in
                // expression context, but InList with string values is
                // handled via fixedStringCol downstream.
                if (!inCol || inCol->dataType != DataType::CHAR_FIXED)
                    return false;
            }
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
    auto scan = makeScanForCols(aq.tables[0], idxVar, scanColumns, aq.schema);
    return maybeSelect(std::move(scan), combineFilters(aq.filters, idxVar, aq.schema));
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

    for (const auto& target : aq.targets) {
        if (!target.isAgg || !target.agg) return fail("Scalar aggregation: non-aggregate SELECT target.");
        const AggFunc func = target.agg->func;
        if (func != AggFunc::SUM && func != AggFunc::COUNT && func != AggFunc::AVG &&
            func != AggFunc::MIN && func != AggFunc::MAX)
            return fail("Scalar aggregation: unsupported aggregate function '" + aggName(func) + "'.");
        std::vector<const FuncCall*> aggCalls;
        bool hasDivision = false;
        AggCallWalker::walk(target.expr, aggCalls, hasDivision);
        (void)hasDivision;
        if (aggCalls.size() > 1)
            return fail("Scalar aggregation: complex aggregate expressions with multiple aggregate calls are not supported in scalar-agg path.");
        if (aggCalls.size() == 1 && !scalarAggProjectionExprSupported(target.expr))
            return fail("Scalar aggregation: unsupported aggregate projection expression.");
        if (func != AggFunc::COUNT) {
            if (!target.agg->innerExpr)
                return fail("Scalar aggregation: aggregate '" + aggName(func) + "' requires an inner expression.");
            if (!exprSupported(target.agg->innerExpr, false))
                return fail("Scalar aggregation: inner expression of '" + aggName(func) + "' not supported on GPU.");
            if (!isNumericLike(inferExprDataType(target.agg->innerExpr)))
                return fail("Scalar aggregation: inner expression of '" + aggName(func) + "' must be numeric.");
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
        ExprPtr projectionExpr = scalarProjectionExprForSingleAgg(target);
        if (func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(alias, accIndex, 0, projectionExpr);
        } else if (func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum",
                                                  exprToMetal(target.agg->innerExpr, idxVar),
                                                  "float");
            int countIndex = reduce->addAccumulator(accName + "_count", "1.0f", "float");
            reduce->setAverageResultAlias(alias, sumIndex, countIndex, 0, projectionExpr);
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
            reduce->setAccumulatorResultAlias(alias, accIndex, 0, projectionExpr);
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
        } else if (fc) {
            // FuncCall GROUP BY (e.g. extract(year from ...) resolved via subquery alias)
            kd.keyExpr = exprToMetal(aq.groupBy[ki], idxVar);
            kd.numValues = 256;
        } else if (gc && gc->dataType == DataType::CHAR1) {
            kd.keyExpr = char1BucketExpr(*gc, idxVar, kd.numValues, aq.schema);
            if (kd.numValues == 0) return fail("Grouped aggregation: CHAR1 column '" + gc->column + "' has no known charDomain; use CPU group-by fallback.");
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

            std::string dname = displayNameForTarget(target, targetIndex);
            std::string sname = sanitizeIdentifier(dname);

            // SUM slot (long pair at offset/offset+1, or float at offset with isFloatSum)
            {
                PendingAgg agg;
                agg.displayName = dname;
                agg.name = "a" + std::to_string(targetIndex) + "_" + sname + "_sum";
                agg.offset = valuesPerBucket;
                const int fixedScale = isFloat ? numericScaleForExpr(target.agg->innerExpr, aq.schema) : 0;
                if (isFloat && fixedScale > 0) {
                    agg.valueExpr = scaledLongExpr(exprToMetal(target.agg->innerExpr, idxVar), fixedScale);
                    agg.isLongPair = true;
                    agg.scaleDown = -fixedScale; // marker: AVG sum, denominator follows
                    valuesPerBucket += 2;
                } else if (isFloat) {
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

            std::string maxExpr = distinctDomainSymbolForExpr(target.agg->innerExpr, aq.schema);
            if (maxExpr.empty())
                return fail("Grouped aggregation: COUNT(DISTINCT) on column '" + innerCol->column +
                           "' — no schema distinct-domain metadata.");

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

        PendingAgg agg;
        agg.displayName = displayNameForTarget(target, targetIndex);
        agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
        agg.offset = valuesPerBucket;
        agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
        agg.atomicOp = "add";
        agg.funcName = "SUM";
        agg.innerColumn = extractInnerColumn(target.agg->innerExpr);

        if (vt == DataType::FLOAT) {
            const int fixedScale = numericScaleForExpr(target.agg->innerExpr, aq.schema);
            if (fixedScale > 0) {
                agg.valueExpr = scaledLongExpr(agg.valueExpr, fixedScale);
                agg.isLongPair = true;
                agg.scaleDown = fixedScale;
                valuesPerBucket += 2;
            } else {
                // Float SUM without schema scale falls back to atomic float.
                agg.isFloatSum = true;
                agg.scaleDown = 0;
                valuesPerBucket += 1;
            }
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
        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < (int)aq.orderBy.size(); ++oi) {
            auto column = resolveOrderColumn(aq.orderBy[oi].expr, oi, aq.orderBy, aq.targets);
            if (column) sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
        }
        std::vector<KeyedCompactKeySpec> compactKeys;
        std::vector<GenericMatColumnDesc> compactCols;
        for (const auto& d : decodeInfo) {
            compactKeys.push_back({d.name, d.numValues, d.stride, d.charMap, d.keyBase});
            const std::string buf = "d_adhoc_keyed_out_" + std::to_string(compactCols.size()) +
                "_" + sanitizeIdentifier(d.name);
            if (!d.charMap.empty())
                compactCols.push_back({d.name, buf, "char", 0, 0});
            else
                compactCols.push_back({d.name, buf, "int", 0, 0});
        }

        std::vector<KeyedCompactAggSpec> compactAggs;
        for (size_t pi = 0; pi < pending.size(); ++pi) {
            const auto& p = pending[pi];
            KeyedCompactAggSpec outAgg;
            outAgg.displayName = p.displayName;
            outAgg.offset = p.offset;
            outAgg.isLongPair = p.isLongPair;
            outAgg.scaleDown = p.scaleDown;
            outAgg.isFloatSum = p.isFloatSum;
            outAgg.isMinMax = p.isMinMax;
            outAgg.atomicOp = p.atomicOp;
            outAgg.avgSumIsLongPair = p.isLongPair;

            std::string metalType = "uint";
            int outScale = 0;
            if (p.scaleDown < 0 && pi + 1 < pending.size()) {
                const auto& cnt = pending[pi + 1];
                outAgg.isAvg = true;
                outAgg.countOffset = cnt.offset;
                outAgg.countIsFloat = cnt.isFloatSum;
                outAgg.isLongPair = false;
                metalType = "float";
                ++pi;
            } else if (p.isLongPair) {
                metalType = "uint";
                outScale = p.scaleDown > 0 ? p.scaleDown : 0;
                outAgg.isLongPair = true;
            } else if (p.isFloatSum || p.isMinMax) {
                metalType = "float";
            }

            const std::string buf = "d_adhoc_keyed_out_" + std::to_string(compactCols.size()) +
                "_" + sanitizeIdentifier(outAgg.displayName);
            compactAggs.push_back(outAgg);
            compactCols.push_back({outAgg.displayName, buf, metalType, 0,
                                   outScale, outAgg.isLongPair});
        }

        const std::string compactCounter = "d_adhoc_keyed_result_count";
        auto& compactPhase = appendPhase(plan, "ADHOC_single_table_group_compact",
            makeKeyedAggCompactOperator(
                "d_adhoc_group_aggs", compactCounter, numBuckets, valuesPerBucket,
                compactKeys, compactAggs, compactCols));
        const std::string sortRowsSym = "n_gpu_sort_single_keyed_rows";
        attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);
        if (!appendGenericGpuSort(plan, "single_keyed", sortRowsSym,
                                  std::to_string(numBuckets), compactCols, sortSpec, error)) {
            return std::nullopt;
        }
    }
    return plan;
}

std::optional<MetalQueryPlan> buildMaterializePlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (aq.having) return fail("Materialization: HAVING requires aggregation.");
    if (aq.targets.empty()) return fail("Materialization: no SELECT targets.");
    if (aq.hasAggregation())
        return fail("Materialization: aggregates use aggregation paths.");
    if (aq.hasGroupBy())
        return fail("Materialization: GROUP BY requires an aggregation path.");

    std::set<std::string> usedColumns;
    for (const auto& target : aq.targets) {
        if (!target.expr)
            return fail("Materialization: SELECT target has no expression.");
        if (!materializeExprSupported(target.expr))
            return fail("Materialization: SELECT expression not supported on GPU.");
        collectColumns(target.expr, usedColumns);
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_MATERIALIZE";
    const std::string idxVar = "i";
    auto filtered = makeFilteredScan(aq, usedColumns, idxVar);
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(filtered), "d_adhoc_result_count", "1");

    const std::string outputSize = tableSizeName(aq.tables[0]);
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        DataType type = inferExprDataType(target.expr);
        std::string displayName = displayNameForTarget(target, targetIndex);
        std::string bufferName = "d_adhoc_" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(displayName);
        int stringLen = fixedStringLenForExpr(target.expr, aq.schema);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);

        std::string expr = materializeValueExpr(target.expr, idxVar, aq.schema);
        materialize->addColumn(bufferName, metalTypeForDataType(type),
                               expr, displayName, sizeExpr, stringLen);
        materializedCols.push_back({displayName, bufferName,
                                    metalTypeForDataType(type), stringLen,
                                    0, false});
    }

    auto& matPhase = appendPhase(plan, "ADHOC_single_table_materialize", std::move(materialize));

    GenericSortSpec sortSpec;
    sortSpec.limit = aq.limit;
    for (int oi = 0; oi < (int)aq.orderBy.size(); ++oi) {
        auto column = resolveOrderColumn(aq.orderBy[oi].expr, oi, aq.orderBy, aq.targets);
        if (!column)
            return fail("Materialization: ORDER BY expression could not be resolved to a SELECT target.");
        sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
    }
    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string sortRowsSym = "n_gpu_sort_single_materialize_legacy_rows";
        attachMaterializedCountHook(matPhase, "d_adhoc_result_count", sortRowsSym);
        if (!appendGenericGpuSort(plan, "single_materialize_legacy", sortRowsSym,
                                  outputSize, materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
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
    }
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
    bool semi = false;            // EXISTS → semi-join (inner table goes to child)
    std::vector<int> children;
    bool composite() const { return !keyOnSelf2.empty(); }
};

struct ExistsDistinctInfo {
    int childIdx = -1;
    size_t filterIndex = 0;
    std::string childValueCol;
    std::string parentValueCol;
    std::string firstBuffer;
    std::string stateBuffer;
    std::string multiBitmap;
    bool anti = false;
};

bool colRefMatchesNode(const ColRef& col, const MultiTableTreeNode& node) {
    if (!col.tableAlias.empty()) return col.tableAlias == node.table;
    return col.table == node.baseTable || col.table == node.table;
}

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
        bool semi = false;                // EXISTS → semi-join
        std::string innerTable;           // for semi joins: EXISTS inner table
    };
    std::vector<Edge> edges;
    auto findEdge = [&](const std::string& l, const std::string& r) -> int {
        for (size_t i = 0; i < edges.size(); ++i) {
            if ((edges[i].a == l && edges[i].b == r) ||
                (edges[i].a == r && edges[i].b == l)) return (int)i;
        }
        return -1;
    };
    std::vector<int> tableToIdx; // baseTable → canonical node index mapping
    for (const auto& jc : aq.joins) {
        // Resolve left/right table to specific node indices.
        // For duplicate base tables (e.g., nation x2), the join clause's
        // column references were qualified with the alias (n1, n2), but
        // the query analyzer only preserves the base name ("nation").
        // We disambiguate by: the column's owner must be the CORRECT
        // instance.  Since the Sql analyzer loses the alias, we scan
        // all matching indices.  For a simple join graph the first
        // unvisited index is the correct one, but for duplicate-tbl
        // graphs we pick greedily during BFS (see findIdxAlts below).
        
        // Don't coalesce edges when either table has multiple instances
        // (e.g. lineitem x2 from IN subquery). Each instance needs its own edge.
        bool multiL = false, multiR = false;
        {
            int cntL = 0, cntR = 0;
            for (const auto& t : aq.tables) {
                if (t == jc.leftTable) cntL++;
                if (t == jc.rightTable) cntR++;
            }
            multiL = cntL > 1;
            multiR = cntR > 1;
        }
        int ei = (jc.leftTable == jc.rightTable) ? -1 : 
                 (multiL || multiR) ? -1 : findEdge(jc.leftTable, jc.rightTable);
        // Self-joins (e.g. l1↔l2, l1↔l3) always get separate edges.
        if (ei < 0) {
            Edge e;
            e.a = jc.leftTable; e.b = jc.rightTable;
            e.cols.emplace_back(jc.leftCol, jc.rightCol);
            e.anti = jc.anti;
            e.leftOuter = jc.leftOuter;
            e.semi = jc.semi;
            e.innerTable = jc.innerTable;
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
            edges[ei].semi = edges[ei].semi || jc.semi;
            if (jc.semi && !jc.innerTable.empty()) edges[ei].innerTable = jc.innerTable;
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

    // findIdx returns the first unvisited node with the given
    // table name or alias (e.g. "n1" → index of n1 node).
    auto findIdx = [&](const std::string& ident) -> int {
        for (size_t k = 0; k < n; ++k)
            if (!visited[k] && (aq.tables[k] == ident || aq.tableAliases[k] == ident))
                return (int)k;
        return -1;
    };

    std::vector<int> order;
    order.push_back(probeIdx);
    visited[probeIdx] = true;
    std::vector<bool> edgeUsed(edges.size(), false);

    for (size_t qhead = 0; qhead < order.size(); ++qhead) {
        int u = order[qhead];
        const std::string& uBase = aq.tables[u];
        const std::string& uAlias = aq.tableAliases[u];
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            if (edgeUsed[ei]) continue;
            const auto& e = edges[ei];
            int other = -1;
            // Determine which side is `u` and pick column orientations.
            std::vector<std::pair<std::string, std::string>> oriented; // (col_on_u, col_on_other)
            if (e.a == uBase || e.a == uAlias) {
                other = findIdx(e.b);
                for (const auto& c : e.cols) oriented.emplace_back(c.first, c.second);
            } else if (e.b == uBase || e.b == uAlias) {
                other = findIdx(e.a);
                for (const auto& c : e.cols) oriented.emplace_back(c.second, c.first);
            }
            if (other < 0 || visited[other]) continue;
            edgeUsed[ei] = true;
            visited[other] = true;
            nodes[other].parent = u;
            nodes[other].anti = e.anti;
            nodes[other].leftOuter = e.leftOuter;
            nodes[other].semi = e.semi;
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
    std::string table;      // alias for unique naming (e.g., "l2")
    std::string column;
    std::string baseTable;  // base table for schema lookups (e.g., "lineitem")
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

// CHAR_FIXED access descriptor: records how a build-side CHAR_FIXED
// column is indexed in probe-phase expressions.
struct CharFixedAccess {
    std::string expr;    // column name or scalar carry variable
    bool isArray = true; // true → expr[idxVar] (direct child), false → scalar carried var
    int indexOffset = 0; // -1 for 1-based PK tables (customer, orders, etc.), 0 for 0-based (nation, region)
    std::string baseTable; // base table name for schema lookups (alias is the map key)
};

std::string rewriteForProbe(std::string expr,
                              const std::string& idxVar,
                              const std::map<CarriedKey, std::string>& carryVar,
                              const std::map<std::string, CharFixedAccess>& charFixedKeys,
                              const SchemaProvider* schema = nullptr) {
    for (const auto& [key, var] : carryVar) {
        std::string schemaTable = key.baseTable.empty() ? key.table : key.baseTable;
        DataType t = schema->columnType(schemaTable, key.column);
        std::string sub = decodeCarryValue(t, var);
        // Standard column access pattern: col[idxVar]
        const std::string from = key.column + "[" + idxVar + "]";
        size_t pos = 0;
        while ((pos = expr.find(from, pos)) != std::string::npos) {
            // Check for alias comment: /*alias*/col[i] 
            bool match = true;
            if (pos >= 6 && expr[pos - 2] == '*' && expr[pos - 1] == '/') {
                // Scan back to find "/*"
                size_t cmtStart = pos - 2;
                while (cmtStart > 0 && !(expr[cmtStart] == '/' && expr[cmtStart + 1] == '*'))
                    cmtStart--;
                if (expr[cmtStart] == '/' && expr[cmtStart + 1] == '*') {
                    std::string aliasInExpr = expr.substr(cmtStart + 2, pos - cmtStart - 4);
                    if (!aliasInExpr.empty() && aliasInExpr != key.table)
                        match = false;
                }
            }
            if (match) {
                expr.replace(pos, from.size(), sub);
                pos += sub.size();
            } else {
                pos += from.size();
            }
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
    // CHAR_FIXED index rewrite: `i * width` → `jk[i] * width` or `jkVar * width`
    // Patterns: `col + i * width` or `col[i * width + offset]`
    std::string fromIdx = idxVar + " *";
    size_t pos = 0;
    while ((pos = expr.find(fromIdx, pos)) != std::string::npos) {
        // Find the column name before the index. It may be `col + ` or `col[`
        size_t colEnd = pos;
        while (colEnd > 0 && (expr[colEnd-1] == ' ' || expr[colEnd-1] == '+'))
            colEnd--;
        if (colEnd > 0 && expr[colEnd-1] == '[') colEnd--; // bracket access: col[i*...]
        while (colEnd > 0 && expr[colEnd-1] == ' ') colEnd--;
        size_t colStart = colEnd;
        while (colStart > 0 && (isalnum(expr[colStart-1]) || expr[colStart-1] == '_'))
            colStart--;
        // Check for alias comment marker: /*alias*/colname
        std::string colAlias;
        if (colStart >= 4 && colStart >= 2 && expr.substr(colStart - 2, 2) == "*/") {
            // Scan back to find "/*"
            size_t commentStart = colStart - 3;
            while (commentStart > 0 && expr.substr(commentStart, 2) != "/*")
                commentStart--;
            if (commentStart < expr.size() && expr.substr(commentStart, 2) == "/*") {
                colAlias = expr.substr(commentStart + 2, colStart - commentStart - 4);
                colStart = commentStart;
            }
        }
        std::string colName = expr.substr(colStart, colEnd - colStart);
        if (colName.empty()) { pos += fromIdx.size(); continue; }
        std::string actualCol = colName;
        if (!colAlias.empty() && colName.size() > colAlias.size() + 4)
            actualCol = colName.substr(colAlias.size() + 4);
        bool rewritten = false;
        for (const auto& [tname, jk] : charFixedKeys) {
            if (!colAlias.empty() && tname != colAlias) continue;
            if (schema && !schema->hasColumn(tname, actualCol) &&
                !schema->hasColumn(jk.baseTable, actualCol)) continue;
            std::string idxExpr = jk.expr + (jk.isArray ? ("[" + idxVar + "]") : "");
            if (jk.indexOffset != 0)
                idxExpr = "(" + idxExpr + " " + (jk.indexOffset < 0 ? "-" : "+") + " " + std::to_string(std::abs(jk.indexOffset)) + ")";
            expr.replace(pos, fromIdx.size(), idxExpr + " *");
            pos += idxExpr.size() + 2; // idxExpr + " *"
            rewritten = true;
            break;
        }
        if (!rewritten) pos += fromIdx.size();
    }
    // Pass 2: function-call index patterns — `fixed_like_one_segment(col, (uint)(i), w, ...)`
    // Replace `(uint)(idxVar)` with `(uint)(jkExpr)` for build-side columns.
    const char* funcPatterns[] = {
        "fixed_like_one_segment(", "fixed_like_two_segment(",
        "fixed_string_segment_eq(", "fixed_string_padding_ok("
    };
    for (const char* fp : funcPatterns) {
        size_t fpPos = 0;
        while ((fpPos = expr.find(fp, fpPos)) != std::string::npos) {
            fpPos += strlen(fp);
            // Extract column name (first arg before comma)
            while (fpPos < expr.size() && expr[fpPos] == ' ') fpPos++;
            size_t colEnd = fpPos;
            while (colEnd < expr.size() && expr[colEnd] != ',' && expr[colEnd] != ' ') colEnd++;
            std::string colName = expr.substr(fpPos, colEnd - fpPos);
            if (colName.empty()) { fpPos++; continue; }
            // Find `(uint)(idxVar)` pattern after the first comma
            std::string idxPattern = "(uint)(" + idxVar + ")";
            size_t idxPos = expr.find(idxPattern, colEnd);
            if (idxPos == std::string::npos) { fpPos++; continue; }
            // Check if this column belongs to a build-side table (has charFixedJoinKey entry)
            for (const auto& [tname, jk] : charFixedKeys) {
                if (schema && !schema->hasColumn(tname, colName) &&
                    !schema->hasColumn(jk.baseTable, colName)) continue;
                std::string idxExpr = jk.expr + (jk.isArray ? ("[" + idxVar + "]") : "");
                if (jk.indexOffset != 0)
                    idxExpr = "(" + idxExpr + " " + (jk.indexOffset < 0 ? "-" : "+") + " " + std::to_string(std::abs(jk.indexOffset)) + ")";
                std::string repl = "(uint)(" + idxExpr + ")";
                expr.replace(idxPos, idxPattern.size(), repl);
                break;
            }
            fpPos++; // advance past this match
        }
    }
    return expr;
}

std::string upperCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    return s;
}

std::optional<std::string> inSubqueryAggregateCarryExpr(
    const AnalyzedQuery& aq,
    const SelectTarget& target,
    const std::map<CarriedKey, std::string>& carryVar) {
    if (!target.agg || !target.agg->innerExpr) return std::nullopt;
    auto* col = std::get_if<ColRef>(&target.agg->innerExpr->node);
    if (!col) return std::nullopt;

    const std::string outerFunc = aggFuncName(target.agg->func);
    for (const auto& sa : aq.inSubAggs) {
        if (upperCopy(sa.aggFunc) != outerFunc) continue;
        if (sa.aggExpr != col->column) continue;
        if (!sa.baseTable.empty() && !col->table.empty() && sa.baseTable != col->table)
            continue;

        CarriedKey ck{sa.alias, sa.aggExpr, sa.baseTable};
        auto it = carryVar.find(ck);
        if (it == carryVar.end()) continue;

        std::string schemaTable = sa.baseTable.empty() ? col->table : sa.baseTable;
        DataType carryType = aq.schema->columnType(schemaTable, sa.aggExpr);
        return decodeCarryValue(carryType, it->second);
    }
    return std::nullopt;
}

class MetalExistsDistinctBuild : public MetalUnaryOperator {
public:
    MetalExistsDistinctBuild(std::unique_ptr<MetalOperator> child,
                             std::string firstBuffer,
                             std::string stateBuffer,
                             std::string multiBitmap,
                             std::string keyExpr,
                             std::string valueExpr,
                             std::string sizeExpr)
        : MetalUnaryOperator(std::move(child)),
          firstBuffer_(std::move(firstBuffer)),
          stateBuffer_(std::move(stateBuffer)),
          multiBitmap_(std::move(multiBitmap)),
          keyExpr_(std::move(keyExpr)),
          valueExpr_(std::move(valueExpr)),
          sizeExpr_(std::move(sizeExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addAtomicBufferParam(firstBuffer_, "atomic_uint", sizeExpr_);
        cg.addAtomicBufferParam(stateBuffer_, "atomic_uint", sizeExpr_);
        cg.addBitmapWriteParam(multiBitmap_, "(" + sizeExpr_ + " + 31) / 32");

        const std::string suffix = sanitizeIdentifier(firstBuffer_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _exists_key_" + suffix + " = (uint)(" + keyExpr_ + ");");
            cg.addLine("uint _exists_val_" + suffix + " = (uint)(" + valueExpr_ + ");");
            cg.addLine("while (true) {");
            cg.addLine("    uint _exists_state_" + suffix + " = atomic_load_explicit(&" +
                       stateBuffer_ + "[_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("    if (_exists_state_" + suffix + " == 0u) {");
            cg.addLine("        uint _exists_expected_" + suffix + " = 0u;");
            cg.addLine("        if (atomic_compare_exchange_weak_explicit(&" + stateBuffer_ +
                       "[_exists_key_" + suffix + "], &_exists_expected_" + suffix +
                       ", 1u, memory_order_relaxed, memory_order_relaxed)) {");
            cg.addLine("            atomic_store_explicit(&" + firstBuffer_ + "[_exists_key_" +
                       suffix + "], _exists_val_" + suffix + ", memory_order_relaxed);");
            cg.addLine("            atomic_store_explicit(&" + stateBuffer_ + "[_exists_key_" +
                       suffix + "], 2u, memory_order_relaxed);");
            cg.addLine("            break;");
            cg.addLine("        }");
            cg.addLine("    } else if (_exists_state_" + suffix + " == 2u) {");
            cg.addLine("        uint _exists_first_" + suffix + " = atomic_load_explicit(&" +
                       firstBuffer_ + "[_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("        if (_exists_first_" + suffix + " != _exists_val_" + suffix +
                       ") bitmap_set(" + multiBitmap_ + ", _exists_key_" + suffix + ");");
            cg.addLine("        break;");
            cg.addLine("    }");
            cg.addLine("}");
            consume();
        });
    }

    std::string describe() const override {
        return "ExistsDistinctBuild(" + firstBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string firstBuffer_;
    std::string stateBuffer_;
    std::string multiBitmap_;
    std::string keyExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
};

class MetalExistsDistinctProbe : public MetalUnaryOperator {
public:
    MetalExistsDistinctProbe(std::unique_ptr<MetalOperator> child,
                             std::string firstBuffer,
                             std::string stateBuffer,
                             std::string multiBitmap,
                             std::string keyExpr,
                             std::string valueExpr,
                             bool anti)
        : MetalUnaryOperator(std::move(child)),
          firstBuffer_(std::move(firstBuffer)),
          stateBuffer_(std::move(stateBuffer)),
          multiBitmap_(std::move(multiBitmap)),
          keyExpr_(std::move(keyExpr)),
          valueExpr_(std::move(valueExpr)),
          anti_(anti) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(firstBuffer_, "const atomic_uint", "", false);
        cg.addBufferParam(stateBuffer_, "const atomic_uint", "", false);
        cg.addBitmapReadParam(multiBitmap_, "");

        const std::string suffix = sanitizeIdentifier(firstBuffer_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _exists_key_" + suffix + " = (uint)(" + keyExpr_ + ");");
            cg.addLine("uint _exists_val_" + suffix + " = (uint)(" + valueExpr_ + ");");
            cg.addLine("uint _exists_state_" + suffix + " = atomic_load_explicit(&" +
                       stateBuffer_ + "[_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("bool _exists_other_" + suffix + " = false;");
            cg.addLine("if (_exists_state_" + suffix + " == 2u) {");
            cg.addLine("    uint _exists_first_" + suffix + " = atomic_load_explicit(&" +
                       firstBuffer_ + "[_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("    _exists_other_" + suffix + " = bitmap_test_atomic(" +
                       multiBitmap_ + ", _exists_key_" + suffix + ") || " +
                       "(_exists_first_" + suffix + " != _exists_val_" + suffix + ");");
            cg.addLine("}");
            cg.addIf(std::string(anti_ ? "!" : "") + "_exists_other_" + suffix, [&]() {
                consume();
            });
        });
    }

    std::string describe() const override {
        return std::string(anti_ ? "NotExistsDistinctProbe(" : "ExistsDistinctProbe(") +
               firstBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string firstBuffer_;
    std::string stateBuffer_;
    std::string multiBitmap_;
    std::string keyExpr_;
    std::string valueExpr_;
    bool anti_;
};

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

using ScalarLookupInfo = GenericScalarLookupInfo;

static std::string scalarFloatLiteral(float v) {
    std::ostringstream oss;
    oss << v;
    std::string s = oss.str();
    if (s.find_first_of(".eE") == std::string::npos)
        s += ".0";
    return s + "f";
}

static std::string scalarNanExpr() {
    return "as_type<float>(0x7fc00000u)";
}

static std::string scalarLookupKeyExpr(const ScalarLookupInfo& info,
                                       size_t keyIndex,
                                       const std::string& idxVar,
                                       const std::string& probeTable,
        const SchemaProvider* schema) {
    std::string inner = keyIndex < info.keyCols.size() ? info.keyCols[keyIndex] :
                        (keyIndex == 0 ? info.keyCol : info.keyCol2);
    std::string outer = keyIndex < info.outerKeyCols.size() ? info.outerKeyCols[keyIndex] : "";
    if (schema && !probeTable.empty()) {
        if (!outer.empty() && schema->hasColumn(probeTable, outer))
            return outer + "[" + idxVar + "]";
        if (!inner.empty() && schema->hasColumn(probeTable, inner))
            return inner + "[" + idxVar + "]";
    }
    return (outer.empty() ? inner : outer) + "[" + idxVar + "]";
}

static std::string scalarHashLookupRaw(const std::string& mapName,
                                       const std::string& key1,
                                       const std::string& key2) {
    return "scalar_hash_lookup_raw64(" + mapName + "_states, " + mapName + "_keys, " +
           mapName + "_values, n_" + mapName + ", " + key1 + ", " + key2 + ")";
}

static std::string scalarLookupReplacement(const ScalarLookupInfo& info,
                                           const std::string& idxVar,
                                           const std::string& probeTable,
                                           const SchemaProvider* schema) {
    const std::string key0 = scalarLookupKeyExpr(info, 0, idxVar, probeTable, schema);
    const std::string key1 = scalarLookupKeyExpr(info, 1, idxVar, probeTable, schema);
    switch (info.kind) {
        case ScalarLookupInfo::SumByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.sumBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.sumBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::AvgByKey:
            return "((" + info.cntVar + " > 0) ? ("
                 + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" + info.sumVar
                 + ") / (float)" + info.cntVar
                 + ") : " + scalarNanExpr() + ")";
        case ScalarLookupInfo::MinByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.minBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.minBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::MaxByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.maxBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.maxBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::CountByKey:
            return "(" + scalarFloatLiteral(info.multiplier) + " * (float)" +
                   info.countBuffer + "[" + key0 + "])";
        case ScalarLookupInfo::SumByCompositeHash: {
            std::string val = "scalar_hash_lookup_float_or_nan64(" + info.hashMap + "_states, " +
                              info.hashMap + "_keys, " + info.hashMap + "_values, n_" +
                              info.hashMap + ", (uint)(" + key0 + "), (uint)(" + key1 + "))";
            return "(" + scalarFloatLiteral(info.multiplier) + " * " + val + ")";
        }
        case ScalarLookupInfo::CountByCompositeHash: {
            std::string raw = scalarHashLookupRaw(info.countHashMap, "(uint)(" + key0 + ")",
                                                  "(uint)(" + key1 + ")");
            return "(" + scalarFloatLiteral(info.multiplier) + " * (float)(" + raw + "))";
        }
        case ScalarLookupInfo::AvgByCompositeHash: {
            std::string sk = "(uint)(" + key0 + ")";
            std::string tk = "(uint)(" + key1 + ")";
            std::string sumRaw = scalarHashLookupRaw(info.hashMap, sk, tk);
            std::string cntRaw = scalarHashLookupRaw(info.countHashMap, sk, tk);
            return "((" + cntRaw + " > 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(" + sumRaw + ") / (float)(" + cntRaw + ")) : " +
                   scalarNanExpr() + ")";
        }
        case ScalarLookupInfo::GlobalSum:
            return "(" + scalarFloatLiteral(info.multiplier) +
                   " * atomic_load_explicit(&" + info.sumBuffer +
                   "[0], memory_order_relaxed))";
        case ScalarLookupInfo::GlobalAvg:
            return "((atomic_load_explicit(&" + info.countBuffer +
                   "[0], memory_order_relaxed) > 0u) ? (" +
                   scalarFloatLiteral(info.multiplier) +
                   " * atomic_load_explicit(&" + info.sumBuffer +
                   "[0], memory_order_relaxed) / (float)atomic_load_explicit(&" +
                   info.countBuffer + "[0], memory_order_relaxed)) : " +
                   scalarNanExpr() + ")";
        case ScalarLookupInfo::GlobalCount:
            return "(" + scalarFloatLiteral(info.multiplier) +
                   " * (float)atomic_load_explicit(&" + info.countBuffer +
                   "[0], memory_order_relaxed))";
        case ScalarLookupInfo::GlobalMin:
            return "((" + info.stateBuffer +
                   "[0] != 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(atomic_load_explicit(&" + info.minBuffer +
                   "[0], memory_order_relaxed))) : " + scalarNanExpr() + ")";
        case ScalarLookupInfo::GlobalMax:
            return "((" + info.stateBuffer +
                   "[0] != 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(atomic_load_explicit(&" + info.maxBuffer +
                   "[0], memory_order_relaxed))) : " + scalarNanExpr() + ")";
        default: return "0";
    }
}

static void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string rewriteScalarSentinels(const std::string& cond, const std::string& idxVar,
                                          const std::vector<ScalarLookupInfo>& lookups,
                                          const std::string& probeTable,
                                          const SchemaProvider* schema) {
    std::string result = cond;
    for (const auto& info : lookups) {
        replaceAll(result, std::to_string(info.sentinel),
                   scalarLookupReplacement(info, idxVar, probeTable, schema));
    }
    return result;
}

static bool referencesScalarSentinel(const std::string& text,
                                     const std::vector<ScalarLookupInfo>& lookups) {
    for (const auto& info : lookups) {
        if (text.find(std::to_string(info.sentinel)) != std::string::npos)
            return true;
    }
    return false;
}

static bool referencesScalarLookupBuffer(const std::string& text,
                                         const std::vector<ScalarLookupInfo>& lookups) {
    auto hasName = [&](const std::string& name) {
        return !name.empty() && text.find(name) != std::string::npos;
    };
    for (const auto& info : lookups) {
        if (hasName(info.sumBuffer) ||
            hasName(info.countBuffer) ||
            hasName(info.minBuffer) ||
            hasName(info.maxBuffer) ||
            hasName(info.stateBuffer) ||
            hasName(info.htFlags) ||
            hasName(info.htKeys) ||
            hasName(info.htVals)) {
            return true;
        }
        if (!info.hashMap.empty()) {
            if (hasName(info.hashMap + "_states") ||
                hasName(info.hashMap + "_keys") ||
                hasName(info.hashMap + "_values") ||
                hasName("n_" + info.hashMap)) {
                return true;
            }
        }
        if (!info.countHashMap.empty()) {
            if (hasName(info.countHashMap + "_states") ||
                hasName(info.countHashMap + "_keys") ||
                hasName(info.countHashMap + "_values") ||
                hasName("n_" + info.countHashMap)) {
                return true;
            }
        }
    }
    return false;
}

static void attachScalarLookupBuffers(MetalQueryPlan::Phase& phase,
                                       const std::vector<ScalarLookupInfo>& lookups) {
    auto addResolvedScalar = [&](const std::string& name,
                                 const std::string& type,
                                 const std::string& sizeExpr) {
        if (name.empty() || sizeExpr.empty()) return;
        for (const auto& existing : phase.resolvedScalarParams) {
            if (existing.name == name) return;
        }
        phase.resolvedScalarParams.push_back({name, type, sizeExpr});
    };

    for (const auto& info : lookups) {
        bool global = info.kind == ScalarLookupInfo::GlobalSum ||
                      info.kind == ScalarLookupInfo::GlobalAvg ||
                      info.kind == ScalarLookupInfo::GlobalMin ||
                      info.kind == ScalarLookupInfo::GlobalMax ||
                      info.kind == ScalarLookupInfo::GlobalCount;
        if (global) {
            if (!info.sumBuffer.empty())
                phase.extraBuffers.push_back({
                    info.sumBuffer,
                    (info.kind == ScalarLookupInfo::GlobalSum ||
                     info.kind == ScalarLookupInfo::GlobalAvg) ? "atomic_float" : "atomic_uint",
                    true,
                    false});
            if (!info.countBuffer.empty())
                phase.extraBuffers.push_back({info.countBuffer, "atomic_uint", true, false});
            if (!info.minBuffer.empty())
                phase.extraBuffers.push_back({info.minBuffer, "atomic_uint", true, false});
            if (!info.maxBuffer.empty())
                phase.extraBuffers.push_back({info.maxBuffer, "atomic_uint", true, false});
            if (!info.stateBuffer.empty())
                phase.extraBuffers.push_back({info.stateBuffer, "uint", true, false});
            continue;
        }
        if (info.kind == ScalarLookupInfo::AvgByKey) continue; // handled by ScalarAtomicLookup
        if (!info.sumBuffer.empty())
            phase.extraBuffers.push_back({info.sumBuffer, "uint", true, false});
        if (!info.countBuffer.empty())
            phase.extraBuffers.push_back({info.countBuffer, "uint", true, false});
        if (!info.minBuffer.empty())
            phase.extraBuffers.push_back({info.minBuffer, "uint", true, false});
        if (!info.maxBuffer.empty())
            phase.extraBuffers.push_back({info.maxBuffer, "uint", true, false});
        if (!info.stateBuffer.empty())
            phase.extraBuffers.push_back({info.stateBuffer, "uint", true, false});
        if (!info.hashMap.empty()) {
            phase.extraBuffers.push_back({info.hashMap + "_states", "uint", true, false});
            phase.extraBuffers.push_back({info.hashMap + "_keys", "ulong", true, false});
            phase.extraBuffers.push_back({info.hashMap + "_values", "uint", true, false});
            addResolvedScalar("n_" + info.hashMap, "uint", info.hashCapacityExpr);
        }
        if (!info.countHashMap.empty()) {
            phase.extraBuffers.push_back({info.countHashMap + "_states", "uint", true, false});
            phase.extraBuffers.push_back({info.countHashMap + "_keys", "ulong", true, false});
            phase.extraBuffers.push_back({info.countHashMap + "_values", "uint", true, false});
            addResolvedScalar("n_" + info.countHashMap, "uint", info.hashCapacityExpr);
        }
        if (!info.htFlags.empty())
            phase.extraBuffers.push_back({info.htFlags, "uint", true, false});
        if (!info.htKeys.empty())
            phase.extraBuffers.push_back({info.htKeys, "uint", true, false});
        if (!info.htVals.empty())
            phase.extraBuffers.push_back({info.htVals, "uint", true, false});
    }
}

struct DecorrCol {
    std::string table;
    std::string column;
    std::string qualifier;
    bool inner = false;
};

struct DecorrJoin {
    DecorrCol left;
    DecorrCol right;
};

struct DecorrCorrelation {
    DecorrCol inner;
    DecorrCol outer;
};

struct DecorrelatedScalarSubquery {
    int sqIdx = 0;
    AggFunc func = AggFunc::SUM;
    bool countStar = false;
    float multiplier = 1.0f;
    std::string valueTable;
    std::string valueCol;
    std::vector<std::string> tables;
    std::map<std::string, std::string> aliases;
    std::vector<DecorrJoin> joins;
    std::vector<DecorrCorrelation> correlations;
    std::map<std::string, std::vector<PredPtr>> filtersByTable;
};

static std::optional<std::string> jsonStringValue(const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") && node["String"].contains("sval"))
        return node["String"]["sval"].get<std::string>();
    return std::nullopt;
}

static std::string jsonAExprOp(const nlohmann::json& ae) {
    if (!ae.contains("name") || !ae["name"].is_array() || ae["name"].empty()) return {};
    if (auto s = jsonStringValue(ae["name"][0])) return *s;
    return {};
}

static std::optional<DecorrCol> jsonRawColumnRef(const nlohmann::json& node) {
    const nlohmann::json* cr = nullptr;
    if (node.is_object() && node.contains("ColumnRef")) cr = &node["ColumnRef"];
    else if (node.is_object() && node.contains("fields")) cr = &node;
    if (!cr || !cr->contains("fields") || !(*cr)["fields"].is_array()) return std::nullopt;
    std::vector<std::string> fields;
    for (const auto& f : (*cr)["fields"]) {
        if (auto s = jsonStringValue(f)) fields.push_back(*s);
    }
    if (fields.empty()) return std::nullopt;
    DecorrCol out;
    out.column = fields.back();
    if (fields.size() >= 2) out.qualifier = fields[fields.size() - 2];
    return out;
}

static std::string jsonFuncName(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() || fc["funcname"].empty())
        return {};
    auto s = jsonStringValue(fc["funcname"].back());
    if (!s) return {};
    std::string name = *s;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return name;
}

static std::optional<double> jsonNumericConst(const nlohmann::json& node) {
    const nlohmann::json* ac = nullptr;
    if (node.is_object() && node.contains("A_Const")) ac = &node["A_Const"];
    else if (node.is_object()) ac = &node;
    if (!ac) return std::nullopt;
    auto readNum = [](const nlohmann::json& v) -> std::optional<double> {
        if (v.is_number()) return v.get<double>();
        if (v.is_string()) return std::stod(v.get<std::string>());
        return std::nullopt;
    };
    try {
        if (ac->contains("fval")) {
            const auto& f = (*ac)["fval"];
            if (f.is_object() && f.contains("fval")) return readNum(f["fval"]);
            return readNum(f);
        }
        if (ac->contains("ival")) {
            const auto& i = (*ac)["ival"];
            if (i.is_object() && i.contains("ival")) return readNum(i["ival"]);
            return readNum(i);
        }
        if (ac->contains("val")) {
            const auto& v = (*ac)["val"];
            if (v.contains("Float")) return readNum(v["Float"].at("fval"));
            if (v.contains("Integer")) return readNum(v["Integer"].at("ival"));
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

static int parseDateLiteralLocal(const std::string& s) {
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        return std::stoi(s.substr(0, 4)) * 10000 +
               std::stoi(s.substr(5, 2)) * 100 +
               std::stoi(s.substr(8, 2));
    }
    return 0;
}

static int dateAddYearsLocal(int yyyymmdd, int years) {
    int y = yyyymmdd / 10000;
    int md = yyyymmdd % 10000;
    return (y + years) * 10000 + md;
}

static DecorrCol resolveDecorrCol(DecorrCol col,
                                  const DecorrelatedScalarSubquery& dsq,
                                  const SchemaProvider* schema) {
    if (!col.qualifier.empty()) {
        auto it = dsq.aliases.find(col.qualifier);
        if (it != dsq.aliases.end()) {
            col.table = it->second;
            col.inner = true;
            return col;
        }
        if (schema && schema->hasColumn(col.qualifier, col.column)) {
            col.table = col.qualifier;
            col.inner = true;
            return col;
        }
        col.table = col.qualifier;
        col.inner = false;
        return col;
    }

    std::string match;
    for (const auto& table : dsq.tables) {
        if (schema && schema->hasColumn(table, col.column)) {
            if (!match.empty()) {
                col.inner = false;
                return col;
            }
            match = table;
        }
    }
    if (!match.empty()) {
        col.table = match;
        col.inner = true;
    }
    return col;
}

static ExprPtr jsonExprToExpr(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema);

static ExprPtr jsonConstToExpr(const nlohmann::json& ac) {
    if (ac.contains("ival")) {
        const auto& iv = ac["ival"];
        if (iv.is_object() && iv.contains("ival")) return Expr::lit(iv["ival"].get<int>());
        return Expr::lit(0);
    }
    if (ac.contains("fval")) {
        const auto& fv = ac["fval"];
        if (fv.is_object() && fv.contains("fval"))
            return Expr::litf(std::stof(fv["fval"].get<std::string>()));
        return Expr::litf(0.0f);
    }
    if (ac.contains("sval")) {
        const auto& sv = ac["sval"];
        if (sv.is_object() && sv.contains("sval"))
            return Expr::lits(sv["sval"].get<std::string>());
    }
    return Expr::lit(0);
}

static ExprPtr jsonTypeCastToExpr(const nlohmann::json& tc,
                                  const DecorrelatedScalarSubquery& dsq,
                                  const SchemaProvider* schema) {
    std::string typ;
    if (tc.contains("typeName") && tc["typeName"].contains("names")) {
        for (const auto& n : tc["typeName"]["names"]) {
            if (auto s = jsonStringValue(n)) typ = *s;
        }
    }
    auto arg = jsonExprToExpr(tc.value("arg", nlohmann::json{}), dsq, schema);
    if (typ == "date") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value))
                return Expr::lit(parseDateLiteralLocal(*sv));
        }
    }
    if (typ == "interval") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) return Expr::lit(std::stoi(*sv));
            if (auto* iv = std::get_if<int>(&lit->value)) return Expr::lit(*iv);
        }
    }
    return arg;
}

static ExprPtr jsonExprToExpr(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema) {
    if (node.contains("ColumnRef")) {
        auto raw = jsonRawColumnRef(node);
        if (!raw) return Expr::lit(0);
        auto col = resolveDecorrCol(*raw, dsq, schema);
        DataType dt = (col.inner && schema) ? schema->columnType(col.table, col.column) : DataType::INT;
        int fw = (dt == DataType::CHAR_FIXED && schema) ? schema->columnFixedWidth(col.table, col.column) : 0;
        return Expr::col(col.inner ? col.table : "", col.column, -1, dt, fw, col.qualifier);
    }
    if (node.contains("A_Const")) return jsonConstToExpr(node["A_Const"]);
    if (node.contains("TypeCast")) return jsonTypeCastToExpr(node["TypeCast"], dsq, schema);
    if (node.contains("FuncCall")) {
        FuncCall fc;
        fc.name = jsonFuncName(node["FuncCall"]);
        if (node["FuncCall"].contains("args")) {
            for (const auto& arg : node["FuncCall"]["args"])
                fc.args.push_back(jsonExprToExpr(arg, dsq, schema));
        }
        auto out = std::make_shared<Expr>();
        out->node = std::move(fc);
        return out;
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string op = jsonAExprOp(ae);
        if (op == "+" || op == "-" || op == "*" || op == "/") {
            auto left = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto right = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
            ExprOp eop = ExprOp::ADD;
            if (op == "-") eop = ExprOp::SUB;
            else if (op == "*") eop = ExprOp::MUL;
            else if (op == "/") eop = ExprOp::DIV;
            if ((eop == ExprOp::ADD || eop == ExprOp::SUB)) {
                auto* l = std::get_if<Literal>(&left->node);
                auto* r = std::get_if<Literal>(&right->node);
                if (l && r) {
                    auto* dateVal = std::get_if<int>(&l->value);
                    auto* intervalVal = std::get_if<int>(&r->value);
                    if (dateVal && intervalVal && *dateVal > 19000101 && *dateVal < 21001231) {
                        int years = *intervalVal;
                        if (eop == ExprOp::SUB) years = -years;
                        return Expr::lit(dateAddYearsLocal(*dateVal, years));
                    }
                }
            }
            return Expr::binary(eop, left, right);
        }
    }
    return Expr::lit(0);
}

static CmpOp cmpOpFromJson(const std::string& op) {
    if (op == "=") return CmpOp::EQ;
    if (op == "<>" || op == "!=") return CmpOp::NE;
    if (op == "<") return CmpOp::LT;
    if (op == "<=") return CmpOp::LE;
    if (op == ">") return CmpOp::GT;
    if (op == ">=") return CmpOp::GE;
    return CmpOp::EQ;
}

static PredPtr jsonPredToPred(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema) {
    if (node.contains("BoolExpr")) {
        const auto& be = node["BoolExpr"];
        std::string op = be.value("boolop", "AND_EXPR");
        std::vector<PredPtr> children;
        if (be.contains("args")) {
            for (const auto& arg : be["args"])
                children.push_back(jsonPredToPred(arg, dsq, schema));
        }
        if (op == "OR_EXPR") return Predicate::logOr(std::move(children));
        if (op == "NOT_EXPR" && !children.empty()) return Predicate::logNot(children.front());
        return Predicate::logAnd(std::move(children));
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string kind = ae.value("kind", "AEXPR_OP");
        std::string op = jsonAExprOp(ae);
        if (kind == "AEXPR_IN") {
            auto expr = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            std::vector<ExprPtr> values;
            if (ae.contains("rexpr") && ae["rexpr"].contains("List") &&
                ae["rexpr"]["List"].contains("items")) {
                for (const auto& item : ae["rexpr"]["List"]["items"])
                    values.push_back(jsonExprToExpr(item, dsq, schema));
            }
            auto pred = Predicate::inList(expr, std::move(values));
            if (op == "<>" || op == "!=") return Predicate::logNot(pred);
            return pred;
        }
        if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
            auto expr = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto patExpr = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
            std::string pat;
            if (auto* lit = std::get_if<Literal>(&patExpr->node))
                if (auto* sv = std::get_if<std::string>(&lit->value)) pat = *sv;
            return Predicate::like(expr, pat, op == "!~~" || op == "!~~*");
        }
        auto left = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
        auto right = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
        return Predicate::cmp(cmpOpFromJson(op), left, right);
    }
    return Predicate::cmp(CmpOp::EQ, Expr::lit(1), Expr::lit(1));
}

static void collectJsonConjuncts(const nlohmann::json& node,
                                 std::vector<nlohmann::json>& out) {
    if (node.contains("BoolExpr") && node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        for (const auto& arg : node["BoolExpr"]["args"]) collectJsonConjuncts(arg, out);
        return;
    }
    out.push_back(node);
}

static void collectPredTables(const PredPtr& pred, std::set<std::string>& tables) {
    std::map<std::string, std::string> colToTable;
    collectColumnTables(pred, colToTable);
    for (const auto& [_, table] : colToTable) {
        if (!table.empty()) tables.insert(table);
    }
}

static std::string maxKeySymbolForColumn(const std::string& table,
                                         const std::string& col,
                                         const SchemaProvider* schema) {
    if (schema) {
        auto keySym = schema->keyDomainSymbol(table, col);
        if (!keySym.empty()) return keySym;
        if (auto gd = schema->groupDomain(table, col))
            return std::to_string(gd->maxValue + 1);
        auto pk = schema->pkInfo(table);
        if (pk && pk->first == col) return pk->second;
        auto tableSym = schema->maxKeySymbol(table);
        if (!tableSym.empty()) return tableSym;
    }
    return "";
}

static bool extractDecorrelatedAggTarget(const nlohmann::json& node,
                                         DecorrelatedScalarSubquery& dsq,
                                         const SchemaProvider* schema,
                                         float multiplier = 1.0f) {
    if (!node.is_object()) return false;
    if (node.contains("TypeCast"))
        return extractDecorrelatedAggTarget(node["TypeCast"].value("arg", nlohmann::json{}),
                                            dsq, schema, multiplier);
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        std::string name = jsonFuncName(fc);
        if (!isAggregateFuncCallName(name)) return false;
        if (name == "sum") dsq.func = AggFunc::SUM;
        else if (name == "avg") dsq.func = AggFunc::AVG;
        else if (name == "min") dsq.func = AggFunc::MIN;
        else if (name == "max") dsq.func = AggFunc::MAX;
        else if (name == "count") dsq.func = AggFunc::COUNT;
        dsq.multiplier = multiplier;
        dsq.countStar = !fc.contains("args") || fc["args"].empty();
        if (!dsq.countStar) {
            auto raw = jsonRawColumnRef(fc["args"][0]);
            if (!raw) return false;
            auto col = resolveDecorrCol(*raw, dsq, schema);
            if (!col.inner) return false;
            dsq.valueTable = col.table;
            dsq.valueCol = col.column;
        }
        return true;
    }
    if (!node.contains("A_Expr")) return false;
    const auto& ae = node["A_Expr"];
    std::string op = jsonAExprOp(ae);
    if (op == "*") {
        if (auto lit = jsonNumericConst(ae.value("lexpr", nlohmann::json{})))
            return extractDecorrelatedAggTarget(ae.value("rexpr", nlohmann::json{}),
                                                dsq, schema, multiplier * (float)*lit);
        if (auto lit = jsonNumericConst(ae.value("rexpr", nlohmann::json{})))
            return extractDecorrelatedAggTarget(ae.value("lexpr", nlohmann::json{}),
                                                dsq, schema, multiplier * (float)*lit);
    }
    if (op == "/") {
        if (auto lit = jsonNumericConst(ae.value("rexpr", nlohmann::json{}))) {
            if (*lit != 0.0)
                return extractDecorrelatedAggTarget(ae.value("lexpr", nlohmann::json{}),
                                                    dsq, schema, multiplier / (float)*lit);
        }
    }
    return false;
}

static std::optional<DecorrelatedScalarSubquery> parseDecorrelatedScalarSubquery(
        const std::string& sqlJson,
        const AnalyzedQuery& aq,
        int sqIdx) {
    nlohmann::json root;
    try { root = nlohmann::json::parse(sqlJson); } catch (...) { return std::nullopt; }
    if (!root.contains("SelectStmt")) return std::nullopt;
    const auto& ss = root["SelectStmt"];

    DecorrelatedScalarSubquery dsq;
    dsq.sqIdx = sqIdx;
    if (!ss.contains("fromClause") || !ss["fromClause"].is_array()) return std::nullopt;
    for (const auto& from : ss["fromClause"]) {
        if (!from.contains("RangeVar")) continue;
        const auto& rv = from["RangeVar"];
        std::string rel = rv.value("relname", "");
        if (rel.empty()) continue;
        dsq.tables.push_back(rel);
        dsq.aliases[rel] = rel;
        if (rv.contains("alias")) {
            if (rv["alias"].contains("Alias"))
                dsq.aliases[rv["alias"]["Alias"].value("aliasname", rel)] = rel;
            else if (rv["alias"].contains("aliasname"))
                dsq.aliases[rv["alias"].value("aliasname", rel)] = rel;
        }
    }
    if (dsq.tables.empty()) return std::nullopt;

    if (!ss.contains("targetList") || !ss["targetList"].is_array() || ss["targetList"].empty())
        return std::nullopt;
    bool foundAgg = false;
    for (const auto& target : ss["targetList"]) {
        if (!target.contains("ResTarget") || !target["ResTarget"].contains("val")) continue;
        if (extractDecorrelatedAggTarget(target["ResTarget"]["val"], dsq, aq.schema)) {
            foundAgg = true;
            break;
        }
    }
    if (!foundAgg) return std::nullopt;

    std::vector<nlohmann::json> conjuncts;
    if (ss.contains("whereClause")) collectJsonConjuncts(ss["whereClause"], conjuncts);
    for (const auto& predJson : conjuncts) {
        bool classified = false;
        if (predJson.contains("A_Expr") && jsonAExprOp(predJson["A_Expr"]) == "=") {
            auto leftRaw = jsonRawColumnRef(predJson["A_Expr"].value("lexpr", nlohmann::json{}));
            auto rightRaw = jsonRawColumnRef(predJson["A_Expr"].value("rexpr", nlohmann::json{}));
            if (leftRaw && rightRaw) {
                auto left = resolveDecorrCol(*leftRaw, dsq, aq.schema);
                auto right = resolveDecorrCol(*rightRaw, dsq, aq.schema);
                if (left.inner && right.inner) {
                    if (left.table != right.table || left.column != right.column)
                        dsq.joins.push_back({left, right});
                    classified = true;
                } else if (left.inner != right.inner) {
                    dsq.correlations.push_back(left.inner
                        ? DecorrCorrelation{left, right}
                        : DecorrCorrelation{right, left});
                    classified = true;
                }
            }
        }
        if (classified) continue;

        auto pred = jsonPredToPred(predJson, dsq, aq.schema);
        std::set<std::string> predTables;
        collectPredTables(pred, predTables);
        if (predTables.size() != 1) return std::nullopt;
        dsq.filtersByTable[*predTables.begin()].push_back(pred);
    }

    if (dsq.correlations.size() > 2) return std::nullopt;
    if (dsq.countStar && dsq.valueTable.empty())
        dsq.valueTable = dsq.correlations.empty() ? dsq.tables.front()
                                                  : dsq.correlations.front().inner.table;
    if (dsq.valueTable.empty()) return std::nullopt;
    return dsq;
}

struct DecorrelatedBitmapState {
    std::string table;
    std::string column;
    std::string bitmap;
    std::string sizeExpr;
    bool externalToTable = false;
};

static std::optional<ScalarLookupInfo> buildDecorrelatedScalarPreAgg(
        const DecorrelatedScalarSubquery& dsq,
        const AnalyzedQuery& aq,
        MetalQueryPlan& plan) {
    const std::string idxVar = "i";
    std::map<std::string, std::set<std::string>> relevantCols;
    for (const auto& j : dsq.joins) {
        relevantCols[j.left.table].insert(j.left.column);
        relevantCols[j.right.table].insert(j.right.column);
    }
    for (const auto& c : dsq.correlations)
        relevantCols[c.inner.table].insert(c.inner.column);

    auto filterCondFor = [&](const std::string& table) {
        auto it = dsq.filtersByTable.find(table);
        if (it == dsq.filtersByTable.end()) return std::string("true");
        return combineFilters(it->second, idxVar, aq.schema);
    };
    auto makeFilteredScan = [&](const std::string& table) -> std::unique_ptr<MetalOperator> {
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(table, idxVar);
        return maybeSelect(std::move(pipe), filterCondFor(table));
    };

    std::map<std::pair<std::string, std::string>, DecorrelatedBitmapState> states;
    auto addState = [&](const std::string& table, const std::string& col,
                        DecorrelatedBitmapState state) {
        states[{table, col}] = std::move(state);
    };
    auto hasState = [&](const std::string& table, const std::string& col) {
        return states.count({table, col}) != 0;
    };
    auto stateName = [&](const std::string& table, const std::string& col,
                         const std::string& suffix) {
        return "d_scalar_" + std::to_string(dsq.sqIdx) + "_" +
               sanitizeIdentifier(table + "_" + col + "_" + suffix) + "_bmp";
    };

    for (const auto& [table, filters] : dsq.filtersByTable) {
        auto colsIt = relevantCols.find(table);
        if (colsIt == relevantCols.end()) continue;
        for (const auto& col : colsIt->second) {
            std::string bitmap = stateName(table, col, "seed");
            auto pipe = makeFilteredScan(table);
            auto build = std::make_unique<MetalBitmapBuild>(
                std::move(pipe), bitmap, col + "[" + idxVar + "]",
                maxKeySymbolForColumn(table, col, aq.schema));
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              "_" + sanitizeIdentifier(table + "_" + col + "_seed"),
                        std::move(build));
            addState(table, col, {table, col, bitmap,
                                  maxKeySymbolForColumn(table, col, aq.schema), false});
        }
    }

    bool changed = true;
    int guard = 0;
    while (changed && guard++ < 64) {
        changed = false;
        std::vector<DecorrelatedBitmapState> snapshot;
        for (const auto& [_, state] : states) snapshot.push_back(state);

        for (const auto& st : snapshot) {
            auto colsIt = relevantCols.find(st.table);
            if (colsIt != relevantCols.end()) {
                for (const auto& outCol : colsIt->second) {
                    if (outCol == st.column || hasState(st.table, outCol)) continue;
                    std::string bitmap = stateName(st.table, outCol, "xfer");
                    std::unique_ptr<MetalOperator> pipe = makeAutoScan(st.table, idxVar);
                    pipe = std::make_unique<MetalBitmapProbe>(
                        std::move(pipe), st.bitmap, st.column + "[" + idxVar + "]");
                    pipe = maybeSelect(std::move(pipe), filterCondFor(st.table));
                    auto build = std::make_unique<MetalBitmapBuild>(
                        std::move(pipe), bitmap, outCol + "[" + idxVar + "]",
                        maxKeySymbolForColumn(st.table, outCol, aq.schema));
                    appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                                      "_" + sanitizeIdentifier(st.table + "_" + outCol + "_xfer"),
                                std::move(build));
                    addState(st.table, outCol, {st.table, outCol, bitmap,
                                                maxKeySymbolForColumn(st.table, outCol, aq.schema),
                                                st.externalToTable});
                    changed = true;
                }
            }

            for (const auto& j : dsq.joins) {
                const DecorrCol* src = nullptr;
                const DecorrCol* dst = nullptr;
                if (j.left.table == st.table && j.left.column == st.column) {
                    src = &j.left; dst = &j.right;
                } else if (j.right.table == st.table && j.right.column == st.column) {
                    src = &j.right; dst = &j.left;
                }
                if (!src || !dst || hasState(dst->table, dst->column)) continue;
                std::string bitmap = stateName(dst->table, dst->column, "join");
                std::unique_ptr<MetalOperator> pipe = makeAutoScan(dst->table, idxVar);
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, dst->column + "[" + idxVar + "]");
                pipe = maybeSelect(std::move(pipe), filterCondFor(dst->table));
                auto build = std::make_unique<MetalBitmapBuild>(
                    std::move(pipe), bitmap, dst->column + "[" + idxVar + "]",
                    maxKeySymbolForColumn(dst->table, dst->column, aq.schema));
                appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                                  "_" + sanitizeIdentifier(dst->table + "_" + dst->column + "_join"),
                            std::move(build));
                addState(dst->table, dst->column, {dst->table, dst->column, bitmap,
                                                   maxKeySymbolForColumn(dst->table, dst->column, aq.schema),
                                                   true});
                changed = true;
            }
        }
    }

    auto makeAggInput = [&]() -> std::unique_ptr<MetalOperator> {
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(dsq.valueTable, idxVar);
        pipe = maybeSelect(std::move(pipe), filterCondFor(dsq.valueTable));
        for (const auto& [_, st] : states) {
            if (st.table == dsq.valueTable && st.externalToTable) {
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, st.column + "[" + idxVar + "]");
            }
        }
        return pipe;
    };

    ScalarLookupInfo info;
    info.sentinel = INT_MIN + dsq.sqIdx;
    info.valueTable = dsq.valueTable;
    info.valueCol = dsq.valueCol;
    info.multiplier = dsq.multiplier;
    for (const auto& c : dsq.correlations) {
        info.keyCols.push_back(c.inner.column);
        info.outerKeyCols.push_back(c.outer.column);
    }
    info.keyCol = info.keyCols.empty() ? "" : info.keyCols[0];
    info.keyCol2 = info.keyCols.size() > 1 ? info.keyCols[1] : "";

    const std::string base = "d_scalar_" + std::to_string(dsq.sqIdx) + "_" +
                             sanitizeIdentifier(dsq.valueTable + "_" + info.keyCol +
                                                (info.keyCol2.empty() ? "" : "_" + info.keyCol2) +
                                                "_" + (dsq.valueCol.empty() ? "star" : dsq.valueCol));
    if (info.keyCols.empty()) {
        const std::string valueExpr = dsq.countStar ? "1.0f" : dsq.valueCol + "[" + idxVar + "]";
        struct ScalarGlobalFloatAgg : MetalUnaryOperator {
            std::string op_, buf_, state_, value_;
            ScalarGlobalFloatAgg(std::unique_ptr<MetalOperator> child,
                                 std::string op,
                                 std::string buf,
                                 std::string state,
                                 std::string value)
                : MetalUnaryOperator(std::move(child)), op_(std::move(op)),
                  buf_(std::move(buf)), state_(std::move(state)),
                  value_(std::move(value)) {}
            void produce(MetalCodegen& cg, ConsumerFn) override {
                cg.addAtomicBufferParam(buf_,
                                        op_ == "sum" ? "atomic_float" : "atomic_uint",
                                        "1");
                if (!state_.empty()) cg.addAtomicBufferParam(state_, "atomic_uint", "1");
                child_->produce(cg, [&]() {
                    std::string v = "(float)(" + value_ + ")";
                    if (op_ == "sum")
                        cg.addLine("atomic_fetch_add_explicit(&" + buf_ + "[0], " +
                                   v + ", memory_order_relaxed);");
                    else if (op_ == "min")
                        cg.addLine("atomic_min_float_seen(&" + buf_ + "[0], &" +
                                   state_ + "[0], " + v + ");");
                    else if (op_ == "max")
                        cg.addLine("atomic_max_float_seen(&" + buf_ + "[0], &" +
                                   state_ + "[0], " + v + ");");
                });
            }
            std::string describe() const override { return "ScalarGlobalFloatAgg(" + op_ + ")"; }
            void iusUsed(std::vector<IU>& out) const override {
                appendIUsFromExpr(value_, out);
            }
        };

        if (dsq.func == AggFunc::COUNT) {
            info.kind = ScalarLookupInfo::GlobalCount;
            info.countBuffer = base + "_cnt";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, "0u", "1");
            appendPhase(plan, "ADHOC_scalar_global_" + std::to_string(dsq.sqIdx) + "_count",
                        std::move(count), 256);
            return info;
        }
        if (dsq.func == AggFunc::AVG) {
            info.kind = ScalarLookupInfo::GlobalAvg;
            info.countBuffer = base + "_cnt";
            info.sumBuffer = base + "_sum";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, "0u", "1");
            appendPhase(plan, "ADHOC_scalar_global_" + std::to_string(dsq.sqIdx) + "_avg_count",
                        std::move(count), 256);
            auto sum = std::make_unique<ScalarGlobalFloatAgg>(
                makeAggInput(), "sum", info.sumBuffer, "", valueExpr);
            appendPhase(plan, "ADHOC_scalar_global_" + std::to_string(dsq.sqIdx) + "_avg_sum",
                        std::move(sum), 256);
            return info;
        }
        if (dsq.func == AggFunc::SUM) {
            info.kind = ScalarLookupInfo::GlobalSum;
            info.sumBuffer = base + "_sum";
            auto sum = std::make_unique<ScalarGlobalFloatAgg>(
                makeAggInput(), "sum", info.sumBuffer, "", valueExpr);
            appendPhase(plan, "ADHOC_scalar_global_" + std::to_string(dsq.sqIdx) + "_sum",
                        std::move(sum), 256);
            return info;
        }
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) {
            bool isMin = dsq.func == AggFunc::MIN;
            info.kind = isMin ? ScalarLookupInfo::GlobalMin : ScalarLookupInfo::GlobalMax;
            if (isMin) info.minBuffer = base + "_min";
            else info.maxBuffer = base + "_max";
            info.stateBuffer = base + "_seen";
            auto minmax = std::make_unique<ScalarGlobalFloatAgg>(
                makeAggInput(), isMin ? "min" : "max",
                isMin ? info.minBuffer : info.maxBuffer,
                info.stateBuffer, valueExpr);
            appendPhase(plan, "ADHOC_scalar_global_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min" : "_max"),
                        std::move(minmax), 256);
            return info;
        }
        return std::nullopt;
    }

    if (info.keyCols.size() == 1) {
        info.sizeSymbol = maxKeySymbolForColumn(dsq.valueTable, info.keyCol, aq.schema);
        const std::string keyExpr = info.keyCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::COUNT) {
            info.kind = ScalarLookupInfo::CountByKey;
            info.countBuffer = base + "_cnt";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, keyExpr, info.sizeSymbol);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_count",
                        std::move(count));
            return info;
        }

        struct ScalarDirectFloatAgg : MetalUnaryOperator {
            std::string op_, buf_, state_, key_, value_, size_;
            ScalarDirectFloatAgg(std::unique_ptr<MetalOperator> child,
                                 std::string op, std::string buf, std::string state,
                                 std::string key, std::string value, std::string size)
                : MetalUnaryOperator(std::move(child)), op_(std::move(op)), buf_(std::move(buf)),
                  state_(std::move(state)), key_(std::move(key)), value_(std::move(value)),
                  size_(std::move(size)) {}
            void produce(MetalCodegen& cg, ConsumerFn) override {
                cg.addAtomicBufferParam(buf_, "atomic_uint", size_);
                if (!state_.empty()) cg.addAtomicBufferParam(state_, "atomic_uint", size_);
                child_->produce(cg, [&]() {
                    std::string k = "(uint)(" + key_ + ")";
                    std::string v = "(float)(" + value_ + ")";
                    if (!state_.empty()) {
                        cg.addLine("atomic_store_explicit(&" + state_ + "[" + k +
                                   "], 1u, memory_order_relaxed);");
                    }
                    if (op_ == "sum")
                        cg.addLine("atomic_add_float(&" + buf_ + "[" + k + "], " + v + ");");
                    else if (op_ == "min")
                        cg.addLine("atomic_min_float(&" + buf_ + "[" + k + "], " + v + ");");
                    else if (op_ == "max")
                        cg.addLine("atomic_max_float(&" + buf_ + "[" + k + "], " + v + ");");
                });
            }
            std::string describe() const override { return "ScalarDirectFloatAgg(" + op_ + ")"; }
            void iusUsed(std::vector<IU>& out) const override {
                appendIUsFromExpr(key_, out);
                appendIUsFromExpr(value_, out);
            }
        };

        const std::string valueExpr = dsq.countStar ? "1.0f" : dsq.valueCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::AVG) {
            info.kind = ScalarLookupInfo::AvgByKey;
            info.countBuffer = base + "_cnt";
            info.sumBuffer = base + "_sum";
            info.cntVar = "_scalar_" + std::to_string(dsq.sqIdx) + "_cnt";
            info.sumVar = "_scalar_" + std::to_string(dsq.sqIdx) + "_sum";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, keyExpr, info.sizeSymbol);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_avg_count",
                        std::move(count));
            auto sum = std::make_unique<ScalarDirectFloatAgg>(
                makeAggInput(), "sum", info.sumBuffer, "", keyExpr, valueExpr, info.sizeSymbol);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_avg_sum",
                        std::move(sum));
            return info;
        }
        if (dsq.func == AggFunc::SUM) {
            info.kind = ScalarLookupInfo::SumByKey;
            info.sumBuffer = base + "_sum";
            info.stateBuffer = base + "_seen";
            auto sum = std::make_unique<ScalarDirectFloatAgg>(
                makeAggInput(), "sum", info.sumBuffer, info.stateBuffer, keyExpr, valueExpr,
                info.sizeSymbol);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_sum",
                        std::move(sum));
            return info;
        }
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) {
            bool isMin = dsq.func == AggFunc::MIN;
            info.kind = isMin ? ScalarLookupInfo::MinByKey : ScalarLookupInfo::MaxByKey;
            if (isMin) info.minBuffer = base + "_min";
            else info.maxBuffer = base + "_max";
            info.stateBuffer = base + "_seen";
            struct ScalarFillFloatBuffer : MetalOperator {
                std::string buf_, size_, fill_;
                ScalarFillFloatBuffer(std::string buf, std::string size, std::string fill)
                    : buf_(std::move(buf)), size_(std::move(size)), fill_(std::move(fill)) {}
                void produce(MetalCodegen& cg, ConsumerFn) override {
                    std::string n = "n_" + sanitizeIdentifier(buf_);
                    cg.addAtomicBufferParam(buf_, "atomic_uint", size_);
                    cg.addResolvedScalarParam(n, "uint", size_);
                    cg.addBlock("for (uint i = tid; i < " + n + "; i += tpg)", [&]() {
                        cg.addLine("atomic_store_explicit(&" + buf_ +
                                   "[i], as_type<uint>(" + fill_ + "), memory_order_relaxed);");
                    });
                }
                std::string describe() const override { return "ScalarFillFloatBuffer"; }
            };
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min_init" : "_max_init"),
                        std::make_unique<ScalarFillFloatBuffer>(
                            isMin ? info.minBuffer : info.maxBuffer,
                            info.sizeSymbol,
                            isMin ? "3.402823466e38f" : "-3.402823466e38f"));
            auto minmax = std::make_unique<ScalarDirectFloatAgg>(
                makeAggInput(), isMin ? "min" : "max", isMin ? info.minBuffer : info.maxBuffer,
                info.stateBuffer, keyExpr, valueExpr, info.sizeSymbol);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min" : "_max"),
                        std::move(minmax));
            return info;
        }
        return std::nullopt;
    }

    if (info.keyCols.size() == 2) {
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) return std::nullopt;
        info.hashCapacityExpr = "next_pow2(" + tableSizeName(dsq.valueTable) + " * 2)";
        std::string k1 = "(uint)(" + info.keyCol + "[" + idxVar + "])";
        std::string k2 = "(uint)(" + info.keyCol2 + "[" + idxVar + "])";
        std::string valueExpr = dsq.countStar ? "1u" : dsq.valueCol + "[" + idxVar + "]";

        const std::string hashLookupHelpers = R"(
	static ulong scalar_hash_pack2(uint k1, uint k2) {
	    return ((ulong)k1 << 32) | (ulong)k2;
	}
	
	static uint scalar_hash_mix64(ulong x) {
	    x ^= x >> 33;
	    x *= 0xff51afd7ed558ccdUL;
	    x ^= x >> 33;
	    x *= 0xc4ceb9fe1a85ec53UL;
	    x ^= x >> 33;
	    return (uint)x ^ (uint)(x >> 32);
	}
	
	static void scalar_hash_insert_add_u32_64(device atomic_uint* states,
	                                          device ulong* keys,
	                                          device atomic_uint* vals,
	                                          uint cap, uint k1, uint k2, uint value) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        if (state == 0u) {
	            uint expected = 0u;
	            if (atomic_compare_exchange_weak_explicit(&states[slot], &expected, 1u,
	                    memory_order_relaxed, memory_order_relaxed)) {
	                keys[slot] = key;
	                atomic_store_explicit(&vals[slot], 0u, memory_order_relaxed);
	                atomic_store_explicit(&states[slot], 2u, memory_order_relaxed);
	                atomic_fetch_add_explicit(&vals[slot], value, memory_order_relaxed);
	                return;
	            }
	            continue;
	        }
	        while (state == 1u) {
	            state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        }
	        if (state == 2u && keys[slot] == key) {
	            atomic_fetch_add_explicit(&vals[slot], value, memory_order_relaxed);
	            return;
	        }
	        slot = (slot + 1u) & mask;
	    }
	}
	
	static void scalar_hash_insert_add_float_64(device atomic_uint* states,
	                                            device ulong* keys,
	                                            device atomic_uint* vals,
	                                            uint cap, uint k1, uint k2, float value) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        if (state == 0u) {
	            uint expected = 0u;
	            if (atomic_compare_exchange_weak_explicit(&states[slot], &expected, 1u,
	                    memory_order_relaxed, memory_order_relaxed)) {
	                keys[slot] = key;
	                atomic_store_explicit(&vals[slot], 0u, memory_order_relaxed);
	                atomic_store_explicit(&states[slot], 2u, memory_order_relaxed);
	                atomic_add_float(&vals[slot], value);
	                return;
	            }
	            continue;
	        }
	        while (state == 1u) {
	            state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        }
	        if (state == 2u && keys[slot] == key) {
	            atomic_add_float(&vals[slot], value);
	            return;
	        }
	        slot = (slot + 1u) & mask;
	    }
	}
	
	static uint scalar_hash_lookup_raw64(const device uint* states,
	                                     const device ulong* keys,
	                                     const device uint* vals,
	                                     uint cap, uint k1, uint k2) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = states[slot];
	        if (state == 0u) return 0u;
	        if (state == 2u && keys[slot] == key) return vals[slot];
	        slot = (slot + 1u) & mask;
	    }
	    return 0u;
	}
	
	static float scalar_hash_lookup_float_or_nan64(const device uint* states,
	                                               const device ulong* keys,
	                                               const device uint* vals,
	                                               uint cap, uint k1, uint k2) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = states[slot];
	        if (state == 0u) return as_type<float>(0x7fc00000u);
	        if (state == 2u && keys[slot] == key) return as_type<float>(vals[slot]);
	        slot = (slot + 1u) & mask;
	    }
	    return as_type<float>(0x7fc00000u);
	}
	)";
        if (std::find(plan.helpers.begin(), plan.helpers.end(), hashLookupHelpers) ==
            plan.helpers.end()) {
            plan.helpers.push_back(hashLookupHelpers);
        }

        struct ScalarCompositeHashAgg : MetalUnaryOperator {
            std::string map_, key1_, key2_, value_, cap_;
            bool valueIsFloat_;
            ScalarCompositeHashAgg(std::unique_ptr<MetalOperator> child,
                                   std::string map,
                                   std::string key1,
                                   std::string key2,
                                   std::string value,
                                   std::string cap,
                                   bool valueIsFloat)
                : MetalUnaryOperator(std::move(child)), map_(std::move(map)),
                  key1_(std::move(key1)), key2_(std::move(key2)),
                  value_(std::move(value)), cap_(std::move(cap)),
                  valueIsFloat_(valueIsFloat) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                cg.addAtomicBufferParam(map_ + "_states", "atomic_uint", cap_, 0);
                cg.addBufferParam(map_ + "_keys", "ulong", cap_, false);
                cg.addAtomicBufferParam(map_ + "_values", "atomic_uint", cap_, 0);
                cg.addResolvedScalarParam("n_" + map_, "uint", cap_);
                child_->produce(cg, [&]() {
                    if (valueIsFloat_) {
                        cg.addLine("scalar_hash_insert_add_float_64(" + map_ + "_states, " +
                                   map_ + "_keys, " + map_ + "_values, n_" + map_ +
                                   ", (uint)(" + key1_ + "), (uint)(" + key2_ +
                                   "), (float)(" + value_ + "));");
                    } else {
                        cg.addLine("scalar_hash_insert_add_u32_64(" + map_ + "_states, " +
                                   map_ + "_keys, " + map_ + "_values, n_" + map_ +
                                   ", (uint)(" + key1_ + "), (uint)(" + key2_ +
                                   "), (uint)(" + value_ + "));");
                    }
                    consume();
                });
            }
            std::string describe() const override { return "ScalarCompositeHashAgg(" + map_ + ")"; }
            void iusUsed(std::vector<IU>& out) const override {
                appendIUsFromExpr(key1_, out);
                appendIUsFromExpr(key2_, out);
                appendIUsFromExpr(value_, out);
            }
        };

        if (dsq.func == AggFunc::COUNT || dsq.func == AggFunc::AVG) {
            info.countHashMap = "hm_scalar_" + std::to_string(dsq.sqIdx) + "_count";
            auto count = std::make_unique<ScalarCompositeHashAgg>(
                makeAggInput(), info.countHashMap, k1, k2, "1u", info.hashCapacityExpr, false);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_hash_count",
                        std::move(count));
        }
        if (dsq.func == AggFunc::SUM || dsq.func == AggFunc::AVG) {
            info.hashMap = "hm_scalar_" + std::to_string(dsq.sqIdx) + "_sum";
            auto sum = std::make_unique<ScalarCompositeHashAgg>(
                makeAggInput(), info.hashMap, k1, k2, valueExpr, info.hashCapacityExpr, true);
            appendPhase(plan, "ADHOC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_hash_sum",
                        std::move(sum));
        }
        if (dsq.func == AggFunc::AVG) info.kind = ScalarLookupInfo::AvgByCompositeHash;
        else if (dsq.func == AggFunc::COUNT) info.kind = ScalarLookupInfo::CountByCompositeHash;
        else info.kind = ScalarLookupInfo::SumByCompositeHash;
        return info;
    }

    return std::nullopt;
}

static std::vector<ScalarLookupInfo> buildCorrelatedScalarPreAggs(const AnalyzedQuery& aq,
                                                                   MetalQueryPlan& plan) {
    std::vector<ScalarLookupInfo> result;
    int sqIdx = 0;
    for (const auto& sq : aq.subqueries) {
        if (sq.type == AnalyzedQuery::Subquery::SCALAR_SUBQUERY) {
            if (auto dsq = parseDecorrelatedScalarSubquery(sq.sql, aq, sqIdx)) {
                if (auto info = buildDecorrelatedScalarPreAgg(*dsq, aq, plan))
                    result.push_back(std::move(*info));
            }
        }
        sqIdx++;
    }
    return result;
}

static std::optional<MetalQueryPlan> buildSingleTableScalarLookupMaterializePlan(
        const AnalyzedQuery& aq,
        std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.isSingleTable() || aq.tables.empty()) return std::nullopt;
    if (aq.hasAggregation() || aq.hasGroupBy() || aq.having) return std::nullopt;
    if (aq.targets.empty()) return std::nullopt;
    if (aq.subqueries.empty()) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_SCALAR_LOOKUP_MATERIALIZE";
    auto scalarLookups = buildCorrelatedScalarPreAggs(aq, plan);
    if (scalarLookups.empty()) return std::nullopt;

    const std::string idxVar = "i";
    const std::string table = aq.tables[0];

    std::set<std::string> scanCols;
    for (const auto& target : aq.targets) {
        if (!target.expr)
            return fail("Materialization: SELECT target has no expression.");
        if (!materializeExprSupported(target.expr))
            return fail("Materialization: SELECT expression not supported on GPU.");
        collectColumns(target.expr, scanCols);
    }
    for (const auto& filter : aq.filters) collectColumns(filter, scanCols);
    for (const auto& info : scalarLookups) {
        for (const auto& key : info.outerKeyCols) {
            if (!key.empty()) scanCols.insert(key);
        }
        if (info.outerKeyCols.empty()) {
            for (const auto& key : info.keyCols) {
                if (!key.empty() && aq.schema && aq.schema->hasColumn(table, key))
                    scanCols.insert(key);
            }
        }
    }

    std::string filterCond = combineFilters(aq.filters, idxVar, aq.schema);
    if (!referencesScalarSentinel(filterCond, scalarLookups))
        return std::nullopt;

    auto scan = makeScanForCols(table, idxVar, scanCols, aq.schema);
    std::unique_ptr<MetalOperator> pipe = std::move(scan);

    struct ScalarAtomicLookup : MetalUnaryOperator {
        std::string buf_, key_, var_;
        ScalarAtomicLookup(std::unique_ptr<MetalOperator> c, std::string buf,
                           std::string key, std::string var)
            : MetalUnaryOperator(std::move(c)), buf_(std::move(buf)),
              key_(std::move(key)), var_(std::move(var)) {}
        void produce(MetalCodegen& cg, ConsumerFn consume) override {
            cg.addBufferParam(buf_, "const atomic_uint", "", false, 0);
            child_->produce(cg, [&]() {
                cg.addLine("uint " + var_ + " = atomic_load_explicit(&" +
                           buf_ + "[" + key_ + "], memory_order_relaxed);");
                consume();
            });
        }
        std::string describe() const override { return "ScalarAtomicLookup"; }
        void iusUsed(std::vector<IU>& out) const override {
            appendIUsFromExpr(key_, out);
        }
    };

    for (const auto& info : scalarLookups) {
        if (info.kind == ScalarLookupInfo::AvgByKey) {
            std::string keyExpr = scalarLookupKeyExpr(info, 0, idxVar, table, aq.schema);
            if (!info.countBuffer.empty() && !info.cntVar.empty()) {
                pipe = std::make_unique<ScalarAtomicLookup>(
                    std::move(pipe), info.countBuffer, keyExpr, info.cntVar);
            }
            if (!info.sumBuffer.empty() && !info.sumVar.empty()) {
                pipe = std::make_unique<ScalarAtomicLookup>(
                    std::move(pipe), info.sumBuffer, keyExpr, info.sumVar);
            }
        }
    }

    filterCond = rewriteScalarSentinels(filterCond, idxVar, scalarLookups, table, aq.schema);
    pipe = maybeSelect(std::move(pipe), filterCond);

    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(pipe), "d_adhoc_result_count", "1");

    const std::string outputSize = tableSizeName(table);
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        DataType type = inferExprDataType(target.expr);
        std::string displayName = displayNameForTarget(target, targetIndex);
        std::string bufferName = "d_adhoc_" + std::to_string(targetIndex) + "_" +
                                 sanitizeIdentifier(displayName);
        int stringLen = fixedStringLenForExpr(target.expr, aq.schema);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        materialize->addColumn(bufferName, metalTypeForDataType(type),
                               materializeValueExpr(target.expr, idxVar, aq.schema),
                               displayName, sizeExpr, stringLen);
        materializedCols.push_back({displayName, bufferName,
                                    metalTypeForDataType(type), stringLen, 0, false});
    }

    auto& matPhase = appendPhase(plan, "ADHOC_single_table_scalar_lookup_materialize",
                                 std::move(materialize));
    if (referencesScalarLookupBuffer(filterCond, scalarLookups))
        attachScalarLookupBuffers(matPhase, scalarLookups);

    if (!aq.orderBy.empty() || aq.limit >= 0) {
        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < (int)aq.orderBy.size(); ++oi) {
            auto column = resolveOrderColumn(aq.orderBy[oi].expr, oi, aq.orderBy, aq.targets);
            if (column) sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
        }
        const std::string sortRowsSym = "n_gpu_sort_single_scalar_lookup_rows";
        attachMaterializedCountHook(matPhase, "d_adhoc_result_count", sortRowsSym);
        if (!appendGenericGpuSort(plan, "single_scalar_lookup", sortRowsSym,
                                  outputSize, materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
    }
    return plan;
}

std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan_impl(
    const AnalyzedQuery& aq, std::string* error) {

    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.fromSubqueryAggs.empty())
        return fail("Multi-table planner: grouped FROM-subquery shapes are handled by the generic IR route.");

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
    // EXISTS inner tables should never be the probe — they are semi-join filters.
    std::set<std::string> existsInnerTables;
    for (const auto& jc : aq.joins) {
        if (jc.semi && !jc.innerTable.empty()) {
            existsInnerTables.insert(jc.innerTable);
            for (size_t ti = 0; ti < aq.tables.size(); ++ti) {
                const std::string alias = ti < aq.tableAliases.size() ? aq.tableAliases[ti] : aq.tables[ti];
                if (alias == jc.innerTable) {
                    existsInnerTables.insert(alias);
                } else if (aq.tables[ti] == jc.innerTable) {
                    existsInnerTables.insert(aq.tables[ti]);
                    if (alias == aq.tables[ti])
                        existsInnerTables.insert(alias);
                }
            }
        }
    }
    int probeIdx = 0;
    int bestPrio = -1;
    // LEFT OUTER JOIN: the left table must be the probe root.
    bool hasLeftOuter = false;
    std::string leftOuterTable;
    for (const auto& jc : aq.joins) {
        if (jc.leftOuter) { hasLeftOuter = true; leftOuterTable = jc.leftTable; break; }
    }
    for (size_t i = 0; i < aq.tables.size(); ++i) {
        const std::string alias = i < aq.tableAliases.size() ? aq.tableAliases[i] : aq.tables[i];
        if (existsInnerTables.count(aq.tables[i]) || existsInnerTables.count(alias)) continue;
        if (hasLeftOuter && aq.tables[i] == leftOuterTable) { probeIdx = (int)i; bestPrio = INT_MAX; break; }
        int p = multiTableProbePriority(aq.tables[i], aq.schema);
        if (p > bestPrio) { bestPrio = p; probeIdx = (int)i; }
    }
    const std::string& probeTable = aq.tables[probeIdx];

    // ---------- Build join tree ----------
    std::vector<MultiTableTreeNode> nodes;
    if (!multiTableBuildJoinTree(aq, probeIdx, nodes, error)) return std::nullopt;

    // Build column equivalence map from all join edges for diamond-edge
    // probe resolution.  If cols (A, B) join, A and B are in the same
    // equivalence class.  The canonical representative is the column
    // that appears as a bitmap key (keyOnSelf) of some node.
    std::unordered_map<std::string, std::string> colEquiv; // "table.col" → "canonical_table.canonical_col"
    auto canonicalCol = [&](const std::string& table, const std::string& col) -> std::string {
        std::string key = table + "." + col;
        std::string cur = key;
        int steps = 0;
        while (colEquiv.count(cur) && colEquiv[cur] != key && steps++ < 20)
            cur = colEquiv[cur];
        return cur;
    };
    for (const auto& jc : aq.joins) {
        std::string leftCanon = canonicalCol(jc.leftTable, jc.leftCol);
        std::string rightCanon = canonicalCol(jc.rightTable, jc.rightCol);
        colEquiv[jc.leftTable + "." + jc.leftCol] = rightCanon;
        colEquiv[jc.rightTable + "." + jc.rightCol] = leftCanon;
    }

    // ---------- Collect needed (carried) columns per table ----------
    // Compute BEFORE join validation so we know which edges are SemiJoin-only.
    std::map<std::string, std::set<std::string>> neededByTable;

    auto addNeededFromExpr = [&](const ExprPtr& e) {
        if (!e) return;
        std::map<std::string, std::string> colToTable;
        collectColumnTables(e, colToTable);
        for (const auto& [col, owner] : colToTable)
            neededByTable[owner].insert(col);
    };

    for (const auto& t : aq.targets) {
        if (t.isAgg) {
            if (t.agg && !t.agg->isStar) {
                if (t.agg->innerExpr) addNeededFromExpr(t.agg->innerExpr);
                else if (t.expr) addNeededFromExpr(t.expr); // complex agg (e.g. SUM(...)/SUM(...))
            }
        } else {
            addNeededFromExpr(t.expr);
        }
    }
    for (const auto& g : aq.groupBy) addNeededFromExpr(g);

    // For diamond edges, add join columns to neededByTable if they differ
    // from the node's bitmap key — they need to be carried to the probe
    // for per-row constraint checking.  Skip hash-join nodes (composite
    // key) which can only carry one column.
    for (const auto& jc : aq.joins) {
        int idxL = -1, idxR = -1;
        for (int k = 0; k < (int)nodes.size(); ++k) {
            if (aq.tables[k] == jc.leftTable) idxL = k;
            if (aq.tables[k] == jc.rightTable) idxR = k;
        }
        if (idxL < 0 || idxR < 0) continue;
        if (nodes[idxL].keyOnSelf != jc.leftCol && !nodes[idxL].composite())
            neededByTable[jc.leftTable].insert(jc.leftCol);
        if (nodes[idxR].keyOnSelf != jc.rightCol && !nodes[idxR].composite())
            neededByTable[jc.rightTable].insert(jc.rightCol);
    }

    // ---------- Per-table filters ----------
    std::map<std::string, std::vector<PredPtr>> filtersByTable;
    std::vector<PredPtr> crossFilters;  // multi-table filters applied in probe phase
    // Detect multi-instance tables (same base name, different aliases)
    std::set<std::string> multiInstanceTables;
    {
        std::map<std::string, int> baseCount;
        for (const auto& t : aq.tables) baseCount[t]++;
        for (const auto& [t, cnt] : baseCount)
            if (cnt >= 2) multiInstanceTables.insert(t);
    }
    for (const auto& f : aq.filters) {
        std::map<std::string, std::string> colToTable;
        collectColumnTables(f, colToTable);
        std::set<std::string> tbls;
        for (const auto& [col, owner] : colToTable)
            tbls.insert(owner);
        bool isCross = (tbls.size() != 1);
        std::set<std::string> aliases;
        // For single-table filters on multi-instance tables, check if the
        // predicate actually references multiple instances via column aliases.
        if (!isCross && tbls.size() == 1) {
            std::string singleTable = *tbls.begin();
            if (multiInstanceTables.count(singleTable)) {
                // Walk the predicate to count distinct table aliases
                std::function<void(const PredPtr&)> collectPredAliases;
                collectPredAliases = [&](const PredPtr& p) {
                    if (!p) return;
                    std::visit([&](auto&& node) {
                        using T = std::decay_t<decltype(node)>;
                        if constexpr (std::is_same_v<T, Comparison>) {
                            std::function<void(const ExprPtr&)> collectExprAliases;
                            collectExprAliases = [&](const ExprPtr& e) {
                                if (!e) return;
                                if (auto* cr = std::get_if<ColRef>(&e->node)) {
                                    if (!cr->tableAlias.empty())
                                        aliases.insert(cr->tableAlias);
                                } else if (auto* be = std::get_if<BinaryExpr>(&e->node)) {
                                    collectExprAliases(be->left);
                                    collectExprAliases(be->right);
                                }
                            };
                            collectExprAliases(node.left);
                            collectExprAliases(node.right);
                        } else if constexpr (std::is_same_v<T, LogicalAnd> || std::is_same_v<T, LogicalOr>) {
                            for (auto& c : node.children) collectPredAliases(c);
                        } else if constexpr (std::is_same_v<T, LogicalNot>) {
                            collectPredAliases(node.child);
                        }
                    }, p->node);
                };
                collectPredAliases(f);
                if (aliases.size() >= 2) isCross = true;
            }
        }
        if (isCross) {
            crossFilters.push_back(f);
            continue;
        }
        // For single-table filters on multi-instance tables, key by alias so
        // the filter only applies to the correct instance (e.g. l1, not l2/l3).
        std::string filterKey = *tbls.begin();
        if (multiInstanceTables.count(filterKey) && aliases.size() == 1)
            filterKey = *aliases.begin();
        filtersByTable[filterKey].push_back(f);
    }

    // Add cross-filter columns to neededByTable so carries are created
    // for cross-table comparisons (e.g. l2.l_suppkey <> l1.l_suppkey in Q21).
    for (const auto& cf : crossFilters) {
        std::map<std::string, std::string> colToTable;
        collectColumnTables(cf, colToTable);
        for (const auto& [col, owner] : colToTable)
            neededByTable[owner].insert(col);
    }

    std::map<int, ExistsDistinctInfo> existsDistinctByChild;
    std::set<size_t> consumedCrossFilters;
    for (int child = 0; child < (int)nodes.size(); ++child) {
        if (!nodes[child].semi || nodes[child].parent != probeIdx || nodes[child].composite())
            continue;
        int parent = nodes[child].parent;
        if (parent < 0) continue;
        for (size_t fi = 0; fi < crossFilters.size(); ++fi) {
            if (consumedCrossFilters.count(fi)) continue;
            auto* cmp = crossFilters[fi] ? std::get_if<Comparison>(&crossFilters[fi]->node) : nullptr;
            if (!cmp || cmp->op != CmpOp::NE) continue;
            auto* left = cmp->left ? std::get_if<ColRef>(&cmp->left->node) : nullptr;
            auto* right = cmp->right ? std::get_if<ColRef>(&cmp->right->node) : nullptr;
            if (!left || !right) continue;

            const ColRef* childCol = nullptr;
            const ColRef* parentCol = nullptr;
            if (colRefMatchesNode(*left, nodes[child]) &&
                colRefMatchesNode(*right, nodes[parent])) {
                childCol = left;
                parentCol = right;
            } else if (colRefMatchesNode(*right, nodes[child]) &&
                       colRefMatchesNode(*left, nodes[parent])) {
                childCol = right;
                parentCol = left;
            } else {
                continue;
            }

            DataType childType = aq.schema->columnType(nodes[child].baseTable, childCol->column);
            DataType parentType = aq.schema->columnType(nodes[parent].baseTable, parentCol->column);
            if ((childType != DataType::INT && childType != DataType::DATE &&
                 childType != DataType::CHAR1) ||
                childType != parentType) {
                continue;
            }

            ExistsDistinctInfo info;
            info.childIdx = child;
            info.filterIndex = fi;
            info.childValueCol = childCol->column;
            info.parentValueCol = parentCol->column;
            info.firstBuffer = "d_exists_" + nodes[child].table + "_" +
                               sanitizeIdentifier(childCol->column) + "_first";
            info.stateBuffer = "d_exists_" + nodes[child].table + "_" +
                               sanitizeIdentifier(childCol->column) + "_state";
            info.multiBitmap = "d_exists_" + nodes[child].table + "_" +
                               sanitizeIdentifier(childCol->column) + "_multi";
            info.anti = nodes[child].anti;
            existsDistinctByChild[child] = info;
            consumedCrossFilters.insert(fi);
            neededByTable[nodes[child].baseTable].insert(info.childValueCol);
            neededByTable[nodes[parent].baseTable].insert(info.parentValueCol);
            break;
        }
    }
    if (!consumedCrossFilters.empty()) {
        std::vector<PredPtr> residualCrossFilters;
        for (size_t fi = 0; fi < crossFilters.size(); ++fi) {
            if (!consumedCrossFilters.count(fi))
                residualCrossFilters.push_back(crossFilters[fi]);
        }
        crossFilters.swap(residualCrossFilters);
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
        auto pk = multiTablePkInfo(nodes[i].baseTable, aq.schema);
        if (!pk)
            return fail("Multi-table planner: table '" + nodes[i].table + "' has no PK descriptor.");
    }

    // ---------- Carried columns: per-non-probe-table local + subtree ----------
    std::map<int, std::vector<CarriedKey>> localCarry;       // owned by this node
    std::map<int, std::vector<CarriedKey>> subtreeCarry;     // local + descendants
    std::function<void(int)> dfs = [&](int u) {
        const std::string& tname = nodes[u].baseTable;
        // Local carries: columns from this table needed at probe, EXCEPT
        // the join key itself when the value is implicitly carried by
        // probe's lookup key.
        if (u != probeIdx) {
            const auto& need = neededByTable[tname];
            for (const auto& c : need) {
                CarriedKey ck{nodes[u].table, c, tname};
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
    // Add intermediate join keys for grandchild CHAR_FIXED carries
    // BEFORE DFS so localCarry and subtreeCarry pick them up.
    // For each node that has CHAR_FIXED needed columns, walk up its
    // parent chain and add the join keys at each intermediate level.
    for (int i = 0; i < (int)nodes.size(); ++i) {
        if (i == probeIdx) continue;
        if (nodes[i].parent < 0) continue;
        const std::string& tname = nodes[i].baseTable;
        auto nit = neededByTable.find(tname);
        if (nit == neededByTable.end()) continue;
        bool hasCharFixed = false;
        for (const auto& c : nit->second) {
            if (aq.schema->columnType(tname, c) == DataType::CHAR_FIXED) {
                hasCharFixed = true; break;
            }
        }
        if (!hasCharFixed) continue;
        int cur = i;
        while (cur >= 0 && nodes[cur].parent != probeIdx && nodes[cur].parent >= 0) {
            int parent = nodes[cur].parent;
            std::string keyOnParent = nodes[cur].keyOnParent;
            neededByTable[aq.tables[parent]].insert(keyOnParent);
            cur = parent;
        }
    }
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

    auto scalarLookups = buildCorrelatedScalarPreAggs(aq, plan);
    const size_t firstNormalPhase = plan.phases.size();
    std::set<std::string> scalarConsumerPhases;
    auto finalizePlan = [&]() -> MetalQueryPlan {
        for (size_t pi = firstNormalPhase; pi < plan.phases.size(); ++pi) {
            if (scalarConsumerPhases.count(plan.phases[pi].name))
                attachScalarLookupBuffers(plan.phases[pi], scalarLookups);
        }
        return std::move(plan);
    };

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
    std::set<int> builtBitmaps;
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

        // Collect columns from per-table filters (needed for EXISTS inner WHERE)
        for (const auto& f : filtersByTable[tname])
            collectColumns(f, scanCols);
        if (tag != tname) {
            auto itFi = filtersByTable.find(tag);
            if (itFi != filtersByTable.end())
                for (const auto& f : itFi->second)
                    collectColumns(f, scanCols);
        }
        auto itInst = aq.instanceFilters.find(tag);
        if (itInst != aq.instanceFilters.end())
            for (const auto& f : itInst->second)
                collectColumns(f, scanCols);

        auto scan = makeScanForCols(tname, idxVar, scanCols, aq.schema);
        std::unique_ptr<MetalOperator> pipe = std::move(scan);

        // Per-table filters + alias-specific + instance-specific (EXISTS inner WHERE)
        auto buildFilters = filtersByTable[tname];
        if (tag != tname) {
            auto itFi = filtersByTable.find(tag);
            if (itFi != filtersByTable.end())
                buildFilters.insert(buildFilters.end(), itFi->second.begin(), itFi->second.end());
        }
        if (itInst != aq.instanceFilters.end())
            buildFilters.insert(buildFilters.end(), itInst->second.begin(), itInst->second.end());
        std::string filterCond = combineFilters(buildFilters, idxVar, aq.schema);
        bool buildUsesScalarLookup =
            referencesScalarSentinel(filterCond, scalarLookups);
        filterCond = rewriteScalarSentinels(filterCond, idxVar, scalarLookups, tname, aq.schema);
        buildUsesScalarLookup =
            buildUsesScalarLookup || referencesScalarLookupBuffer(filterCond, scalarLookups);
        pipe = maybeSelect(std::move(pipe), filterCond);

        // For each child of u, attach probe (BitmapProbe or ArrayLookup
        // for each carried column from that child's subtree).
        for (int c : nodes[u].children) {
            const std::string& probeKey = nodes[c].keyOnParent + "[" + idxVar + "]";
            const auto& subC = subtreeCarry[c];
            // Hash-join children (composite or non-PK) are validated to
            // attach only to the probe, so we don't need a HashMapLookup
            // branch here.
            // Always add bitmap probe for semi-join filtering.
            // Left-outer: skip the probe (rows always survive).
            if (nodes[c].leftOuter) {
                // No bitmap probe — row always passes. Carries default to fill value.
            } else if (nodes[c].anti) {
                pipe = std::make_unique<MetalAntiBitmapProbe>(
                    std::move(pipe), "d_bitmap_" + nodes[c].table, probeKey);
            } else {
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), "d_bitmap_" + nodes[c].table, probeKey);
            }
            // ArrayLookup for non-CHAR_FIXED carries.
            // Anti-join / left-outer: rows survive even when no carry stored,
            // so sentinel must differ from ArrayStore fill value (0xFF = -1 for int).
            int carrSentinel = (nodes[c].anti || nodes[c].leftOuter) ? -2 : -1;
            for (const auto& ck : subC) {
                DataType ckType = aq.schema->columnType(ck.baseTable.empty() ? ck.table : ck.baseTable, ck.column);
                if (ckType == DataType::CHAR_FIXED) continue;
                pipe = std::make_unique<MetalArrayLookup>(
                    std::move(pipe), ck.storageArray(nodes[c].table),
                    probeKey, ck.varName(), "int", carrSentinel);
            }
        }

        // Diamond-edge probes: for each join edge where this table
        // participates, check if the join column differs from keyOnSelf
        // and probe an already-built bitmap with the equivalent key.
        // Skip if a child already probes the same bitmap (avoid duplicate params).
        std::set<std::string> childProbeBits; // bitmaps already probed by children
        for (int c : nodes[u].children) childProbeBits.insert("d_bitmap_" + nodes[c].table);
         for (const auto& jc : aq.joins) {
            std::string colOnThis, colOnOther;
            std::string otherTable;
            if (jc.leftTable == tname) { colOnThis = jc.leftCol; colOnOther = jc.rightCol; otherTable = jc.rightTable; }
            else if (jc.rightTable == tname) { colOnThis = jc.rightCol; colOnOther = jc.leftCol; otherTable = jc.leftTable; }
            else continue;
            if (colOnThis == nodes[u].keyOnSelf) continue; // already handled by bitmap
            std::string canonOther = canonicalCol(otherTable, colOnOther);
            // Find a built node whose base table matches and keyOnSelf matches
            // the canonical other column (unqualified).
            int probeNode = -1;
            std::string canonColName = canonOther.substr(canonOther.find('.') + 1);
            for (int nb : builtBitmaps) {
                if (nodes[nb].keyOnSelf == canonColName) { probeNode = nb; break; }
            }
            if (probeNode < 0) continue;
            std::string probeBitmap = "d_bitmap_" + nodes[probeNode].table;
            if (childProbeBits.count(probeBitmap)) continue;
            scanCols.insert(colOnThis);
            std::string probeKey = colOnThis + "[" + idxVar + "]";
            pipe = std::make_unique<MetalBitmapProbe>(
                std::move(pipe), probeBitmap, probeKey);
        }

        // Now emit storage for parent.
        const std::string storeKey = nodes[u].keyOnSelf + "[" + idxVar + "]";

        // Check if this table is an IN subquery with GROUP BY + HAVING (e.g. Q18).
        // If so, use MetalAtomicAgg + a bitmap-conversion phase instead of
        // a simple bitmap build.
        const AnalyzedQuery::InSubqueryAggInfo* subAgg = nullptr;
        for (auto& sa : aq.inSubAggs) {
            if (sa.tableIndex >= 0 && sa.tableIndex == u) { subAgg = &sa; break; }
            if ((sa.alias == tag || sa.baseTable == tname) && sa.tableIndex < 0) { subAgg = &sa; break; }
        }
        if (subAgg && !subAgg->groupCol.empty()) {
            const std::string aggArrayName = "d_" + tag + "_agg";
            auto pkOpt = multiTablePkInfo(tname, aq.schema);
            const std::string sizeSym = pkOpt ? pkOpt->sizeSym : tableSizeName(tname);
            const std::string bucketExpr = subAgg->groupCol + "[" + idxVar + "]";
            const std::string valExpr = subAgg->aggExpr + "[" + idxVar + "]";

            // Phase A: atomic aggregation
            auto aggPipe = std::make_unique<MetalAtomicAgg>(
                std::move(pipe), aggArrayName, bucketExpr, valExpr,
                sizeSym, "atomic_uint", "float");
            const std::string aggPhaseName = "ADHOC_multi_agg_" + tag;
            appendPhase(plan, aggPhaseName, std::move(aggPipe));
            if (buildUsesScalarLookup) scalarConsumerPhases.insert(aggPhaseName);

            // Phase B: range scan + HAVING filter + bitmap build
            auto rscan = std::make_unique<MetalRangeScan>(sizeSym, idxVar);
            std::string havingCond;
            const std::string aggBits =
                "atomic_load_explicit(&" + aggArrayName + "[" + idxVar +
                "], memory_order_relaxed)";
            const std::string aggRef = "as_type<float>(" + aggBits + ")";
            if (subAgg->havingPred) {
                // Extract threshold from HAVING comparison: agg > literal
                if (auto* cmp = std::get_if<Comparison>(&subAgg->havingPred->node)) {
                    if (auto* lit = cmp->right ? std::get_if<Literal>(&cmp->right->node) : nullptr) {
                        std::visit([&](auto&& v) {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, int>) {
                                havingCond = aggRef + " > " + std::to_string(v);
                            } else if constexpr (std::is_same_v<T, float>) {
                                havingCond = aggRef + " > " + std::to_string(v) + "f";
                            }
                        }, lit->value);
                    }
                }
            }
            if (havingCond.empty()) havingCond = aggRef + " > 0";

            auto filterPipe = std::make_unique<MetalSelection>(
                std::move(rscan), havingCond);
            std::unique_ptr<MetalOperator> bmpBuild = std::make_unique<MetalBitmapBuild>(
                std::move(filterPipe), "d_bitmap_" + tag, idxVar,
                "(" + sizeSym + " + 31) / 32");

            // Also store aggregation result as carry for parent to access.
            // Reinterpret float as int to match the int carry type.
            const auto& sub = subtreeCarry[u];
            for (const auto& ck : sub) {
                DataType ckType = aq.schema->columnType(ck.baseTable.empty() ? ck.table : ck.baseTable, ck.column);
                if (ckType == DataType::CHAR_FIXED) continue;
                bmpBuild = std::make_unique<MetalArrayStore>(
                    std::move(bmpBuild), ck.storageArray(tag),
                    idxVar, "as_type<int>(" + aggBits + ")",
                    "int", sizeSym);
            }

            const std::string buildPhaseName = "ADHOC_multi_build_" + tag;
            appendPhase(plan, buildPhaseName, std::move(bmpBuild));

            // The bitmap-conversion phase reads the agg array. Register it as
            // a side buffer since MetalSelection doesn't auto-register params.
            {   auto& ph = plan.phases.back();
                ph.extraBuffers.push_back({aggArrayName, "atomic_uint", true, false});
            }

            builtBitmaps.insert(u);
            continue;
        }

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
                DataType origType = aq.schema->columnType(ck.baseTable.empty() ? ck.table : ck.baseTable, ck.column);
                valExpr = encodeCarryValue(origType,
                                           ck.column + "[" + idxVar + "]");
            }
            pipe = std::make_unique<MetalHashMapBuild>(
                std::move(pipe), mapName, k1, k2, valExpr, capExpr);
            const std::string buildPhaseName = "ADHOC_multi_build_" + tag;
            appendPhase(plan, buildPhaseName, std::move(pipe));
            if (buildUsesScalarLookup) scalarConsumerPhases.insert(buildPhaseName);
            builtBitmaps.insert(u);
            continue;
        }

        auto pkU = multiTablePkInfo(tname, aq.schema);
        std::string sizeSym = maxKeySymbolForColumn(tname, nodes[u].keyOnSelf, aq.schema);
        if (sizeSym.empty()) {
            sizeSym = pkU ? pkU->sizeSym : tableSizeName(tname);
        }

        auto existsDistinctIt = existsDistinctByChild.find(u);
        if (existsDistinctIt != existsDistinctByChild.end()) {
            const auto& info = existsDistinctIt->second;
            pipe = std::make_unique<MetalExistsDistinctBuild>(
                std::move(pipe),
                info.firstBuffer,
                info.stateBuffer,
                info.multiBitmap,
                storeKey,
                info.childValueCol + "[" + idxVar + "]",
                sizeSym);
            const std::string buildPhaseName = "ADHOC_multi_build_" + tag;
            appendPhase(plan, buildPhaseName, std::move(pipe));
            if (buildUsesScalarLookup) scalarConsumerPhases.insert(buildPhaseName);
            continue;
        }

        const auto& sub = subtreeCarry[u];
        // Always create a bitmap for the SemiJoin filter.
        pipe = std::make_unique<MetalBitmapBuild>(
            std::move(pipe), "d_bitmap_" + tag, storeKey,
            "(" + sizeSym + " + 31) / 32");

        // For non-CHAR_FIXED carries, also create ArrayStores for value propagation.
        for (const auto& ck : sub) {
            DataType ckType = aq.schema->columnType(ck.baseTable.empty() ? ck.table : ck.baseTable, ck.column);
            if (ckType == DataType::CHAR_FIXED) continue;
            std::string valExpr;
            if (ck.table == tname || ck.table == tag) {
                DataType origType = aq.schema->columnType(tname, ck.column);
                valExpr = encodeCarryValue(origType,
                                           ck.column + "[" + idxVar + "]");
            } else {
                valExpr = ck.varName();
            }
            pipe = std::make_unique<MetalArrayStore>(
                std::move(pipe), ck.storageArray(tag),
                storeKey, valExpr, "int", sizeSym);
        }

        const std::string buildPhaseName = "ADHOC_multi_build_" + tag;
        appendPhase(plan, buildPhaseName, std::move(pipe));
        if (buildUsesScalarLookup) scalarConsumerPhases.insert(buildPhaseName);
        builtBitmaps.insert(u);
    }

    // Build map from build-table to probe-side join key for CHAR_FIXED direct access.
    std::map<std::string, CharFixedAccess> charFixedJoinKey;  // tableName → access descriptor
    for (int c : nodes[probeIdx].children) {
        int offset = 0;
        auto pkInfo = multiTablePkInfo(aq.tables[c], aq.schema);
        if (pkInfo) {
            // If sizeSym starts with "max", it's a 1-based PK table.
            // "25", "5" etc. are 0-based dimension tables.
            offset = pkInfo->sizeSym.find("max") == 0 ? -1 : 0;
        }
        charFixedJoinKey[nodes[c].table] = {nodes[c].keyOnParent, true, offset, nodes[c].baseTable};
    }
    // Grandchild CHAR_FIXED columns: add carry-variable based access.
    // For duplicate base tables (e.g. nation n1/n2), add an entry per instance
    // keyed by the node's alias so rewriteForProbe can disambiguate.
    for (int i = 0; i < (int)nodes.size(); ++i) {
        if (i == probeIdx) continue;
        if (nodes[i].parent < 0) continue;
        // For direct children, already handled above.
        bool isDirectChild = false;
        for (int c : nodes[probeIdx].children)
            if (c == i) { isDirectChild = true; break; }
        if (isDirectChild) continue;
        const std::string& alias = nodes[i].table;
        if (charFixedJoinKey.count(alias)) continue;
        const std::string& baseTbl = nodes[i].baseTable;
        // Check if this table has any CHAR_FIXED columns needed by the probe
        auto nit = neededByTable.find(baseTbl);
        if (nit == neededByTable.end()) {
            // Also check under alias (for duplicated tables with alias-aware neededByTable)
            nit = neededByTable.find(alias);
            if (nit == neededByTable.end()) continue;
        }
        bool hasCharFixed = false;
        for (const auto& c : nit->second) {
            if (aq.schema->columnType(baseTbl, c) == DataType::CHAR_FIXED) {
                hasCharFixed = true; break;
            }
        }
        if (!hasCharFixed) continue;
        int intermIdx = nodes[i].parent;
        std::string intermTable = aq.tables[intermIdx];
        std::string intermKeyCol = nodes[i].keyOnParent;
        CarriedKey ck{intermTable, intermKeyCol, intermTable};
        int goffset = 0;
        auto goPkInfo = multiTablePkInfo(baseTbl, aq.schema);
        if (goPkInfo) {
            goffset = goPkInfo->sizeSym.find("max") == 0 ? -1 : 0;
        }
        charFixedJoinKey[alias] = {ck.varName(), false, goffset, baseTbl};
    }

    // Collect build-side CHAR_FIXED columns needed by cross-table filters
    // so they can be registered as probe-phase extra buffers.
    std::vector<std::pair<std::string, std::string>> crossExtraCols; // (colName, colType)
    {
        std::map<std::string, std::string> colToTable;
        for (auto& cf : crossFilters) collectColumnTables(cf, colToTable);
        for (const auto& [c, owner] : colToTable) {
            if (owner != probeTable) {
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
        std::map<std::string, std::string> colToTable;
        for (auto& cf : crossFilters) collectColumnTables(cf, colToTable);
        for (const auto& [c, owner] : colToTable) {
            if (owner == probeTable) probeScanCols.insert(c);
        }
    }
    {
        const std::string& probeAlias = nodes[probeIdx].table;
        if (probeAlias != probeTable) {
            auto itFi = filtersByTable.find(probeAlias);
            if (itFi != filtersByTable.end())
                for (const auto& f : itFi->second)
                    collectColumns(f, probeScanCols);
        }
        auto itInst = aq.instanceFilters.find(probeAlias);
        if (itInst != aq.instanceFilters.end())
            for (const auto& f : itInst->second)
                collectColumns(f, probeScanCols);
    }
    auto probeScan = makeScanForCols(probeTable, idxVar, probeScanCols, aq.schema);
    std::unique_ptr<MetalOperator> probePipe = std::move(probeScan);

    // Probe's own filters + alias-specific + instance-specific filters.
    auto probeFilters = filtersByTable[probeTable];
    {
        const std::string& probeAlias = nodes[probeIdx].table;
        if (probeAlias != probeTable) {
            auto itFi = filtersByTable.find(probeAlias);
            if (itFi != filtersByTable.end())
                probeFilters.insert(probeFilters.end(), itFi->second.begin(), itFi->second.end());
        }
        auto itInst = aq.instanceFilters.find(probeAlias);
        if (itInst != aq.instanceFilters.end())
            probeFilters.insert(probeFilters.end(), itInst->second.begin(), itInst->second.end());
    }
    std::string probeFilterCond = combineFilters(probeFilters, idxVar, aq.schema);
    bool probeUsesScalarLookup =
        referencesScalarSentinel(probeFilterCond, scalarLookups);
    probeFilterCond = rewriteScalarSentinels(probeFilterCond, idxVar, scalarLookups, probeTable, aq.schema);
    probeUsesScalarLookup =
        probeUsesScalarLookup || referencesScalarLookupBuffer(probeFilterCond, scalarLookups);

    // Insert MetalArrayLookup operators for scalar subquery carries BEFORE
    // the filter, so the filter expression can reference the carry variables.
    // Use a custom terminal that reads atomic_uint buffers via atomic_load_explicit,
    // since regular MetalArrayLookup registers buffers as plain int which can't
    // read atomic_uint on Apple GPU.
    for (const auto& info : scalarLookups) {
        if (info.kind == ScalarLookupInfo::AvgByKey) {
            std::string keyExpr = scalarLookupKeyExpr(info, 0, idxVar, probeTable, aq.schema);
            // Custom operator: reads atomic_uint buffer, stores as int variable
            struct ScalarAtomicLookup : MetalUnaryOperator {
                std::string buf_, key_, var_, idx_;
                ScalarAtomicLookup(std::unique_ptr<MetalOperator> c, std::string buf,
                                   std::string key, std::string var, std::string idx)
                    : MetalUnaryOperator(std::move(c)), buf_(buf), key_(key),
                      var_(var), idx_(idx) {}
                void produce(MetalCodegen& cg, ConsumerFn consume) override {
                    cg.addBufferParam(buf_, "const atomic_uint", "", false, 0);
                    child_->produce(cg, [&]() {
                        cg.addLine("uint " + var_ + " = atomic_load_explicit(&" +
                                   buf_ + "[" + key_ + "], memory_order_relaxed);");
                        consume();
                    });
                }
                std::string describe() const override { return "ScalarAtomicLookup"; }
            };
            if (!info.cntVar.empty())
                probePipe = std::make_unique<ScalarAtomicLookup>(
                    std::move(probePipe), info.countBuffer, keyExpr, info.cntVar, idxVar);
            if (!info.sumVar.empty())
                probePipe = std::make_unique<ScalarAtomicLookup>(
                    std::move(probePipe), info.sumBuffer, keyExpr, info.sumVar, idxVar);
        }
    }

    probePipe = maybeSelect(std::move(probePipe), probeFilterCond);

    // Probe each direct child.
    std::map<CarriedKey, std::string> carryVar; // for expression rewrite
    for (int c : nodes[probeIdx].children) {
        const std::string& probeKey = nodes[c].keyOnParent + "[" + idxVar + "]";
        const auto& subC = subtreeCarry[c];

        auto existsDistinctIt = existsDistinctByChild.find(c);
        if (existsDistinctIt != existsDistinctByChild.end()) {
            const auto& info = existsDistinctIt->second;
            probePipe = std::make_unique<MetalExistsDistinctProbe>(
                std::move(probePipe),
                info.firstBuffer,
                info.stateBuffer,
                info.multiBitmap,
                probeKey,
                info.parentValueCol + "[" + idxVar + "]",
                info.anti);
            continue;
        }

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
        // Left-outer: skip the probe (rows always survive).
        if (nodes[c].leftOuter) {
            // No bitmap probe — row always passes. Carries default to fill value.
        } else if (nodes[c].anti) {
            probePipe = std::make_unique<MetalAntiBitmapProbe>(
                std::move(probePipe), "d_bitmap_" + nodes[c].table, probeKey);
            } else {
                probePipe = std::make_unique<MetalBitmapProbe>(
                std::move(probePipe), "d_bitmap_" + nodes[c].table, probeKey);
        }

        // For non-CHAR_FIXED carries, create ArrayLookups for value propagation.
        int carrSentinel = (nodes[c].anti || nodes[c].leftOuter) ? -2 : -1;
        for (const auto& ck : subC) {
            DataType ckType = aq.schema->columnType(ck.baseTable.empty() ? ck.table : ck.baseTable, ck.column);
            if (ckType == DataType::CHAR_FIXED) continue;
            probePipe = std::make_unique<MetalArrayLookup>(
                std::move(probePipe), ck.storageArray(nodes[c].table),
                probeKey, ck.varName(), "int", carrSentinel);
            carryVar[ck] = ck.varName();
        }
    }

    // Apply cross-table filters (e.g. Q19's OR branches) after all probes.
    // Build-side column references are rewritten to use the join key index.
    if (!crossFilters.empty()) {
        // Collect CHAR_FIXED aliases from cross-filter predicates for alias-priority rewrite
        std::set<std::string> cfAliases;
        {
            std::function<void(const PredPtr&)> collectPredAliases;
            collectPredAliases = [&](const PredPtr& p) {
                if (!p) return;
                std::visit([&](auto&& node) {
                    using T = std::decay_t<decltype(node)>;
                    if constexpr (std::is_same_v<T, Comparison>) {
                        std::function<void(const ExprPtr&)> collectExprAliases;
                        collectExprAliases = [&](const ExprPtr& e) {
                            if (!e) return;
                            if (auto* cr = std::get_if<ColRef>(&e->node)) {
                                if (cr->dataType == DataType::CHAR_FIXED && !cr->tableAlias.empty())
                                    cfAliases.insert(cr->tableAlias);
                            } else if (auto* be = std::get_if<BinaryExpr>(&e->node)) {
                                collectExprAliases(be->left);
                                collectExprAliases(be->right);
                            }
                        };
                        collectExprAliases(node.left);
                        collectExprAliases(node.right);
                    } else if constexpr (std::is_same_v<T, LogicalAnd> || std::is_same_v<T, LogicalOr>) {
                        for (auto& c : node.children) collectPredAliases(c);
                    } else if constexpr (std::is_same_v<T, LogicalNot>) {
                        collectPredAliases(node.child);
                    }
                }, p->node);
            };
            for (const auto& f : crossFilters) collectPredAliases(f);
        }
        auto cfKeys = charFixedJoinKey;
        std::string cond = combineFilters(crossFilters, idxVar, aq.schema);
        probeUsesScalarLookup =
            probeUsesScalarLookup || referencesScalarSentinel(cond, scalarLookups);
        cond = rewriteScalarSentinels(cond, idxVar, scalarLookups, probeTable, aq.schema);
        probeUsesScalarLookup =
            probeUsesScalarLookup || referencesScalarLookupBuffer(cond, scalarLookups);
        cond = rewriteForProbe(cond, idxVar, carryVar, cfKeys, aq.schema);
        // Rewrite build-side INT/DATE column indices.
        for (const auto& [tname, jk] : charFixedJoinKey) {
            // INT/DATE: `col[i]` → `col[jk[i]]` (only for array access)
            if (jk.isArray) {
                for (auto& [colName, colType] : crossExtraCols) {
                    if (colType != "int") continue;
                    std::string from = colName + "[" + idxVar + "]";
                    std::string to = colName + "[" + jk.expr + "[" + idxVar + "]]";
                    size_t pos = 0;
                    while ((pos = cond.find(from, pos)) != std::string::npos) {
                        cond.replace(pos, from.size(), to);
                        pos += to.size();
                    }
                }
            }
        }
        probePipe = maybeSelect(std::move(probePipe), cond);
    }

    // Per-row diamond-edge constraints: for skipped join edges where
    // both columns were carried to the probe, add an equality check.
    for (const auto& jc : aq.joins) {
        int idxL = -1, idxR = -1;
        for (int k = 0; k < (int)nodes.size(); ++k) {
            if (aq.tables[k] == jc.leftTable) idxL = k;
            if (aq.tables[k] == jc.rightTable) idxR = k;
        }
        if (idxL < 0 || idxR < 0) continue;
        // Skip if this edge is already handled by the tree structure
        if (nodes[idxL].parent == idxR || nodes[idxR].parent == idxL) continue;
        // Check if both columns were carried (in carryVar)
        CarriedKey ckL{jc.leftTable, jc.leftCol, jc.leftTable};
        CarriedKey ckR{jc.rightTable, jc.rightCol, jc.rightTable};
        auto itL = carryVar.find(ckL);
        auto itR = carryVar.find(ckR);
        if (itL == carryVar.end() || itR == carryVar.end()) continue;
        std::string eqCond = itL->second + " == " + itR->second;
        probePipe = maybeSelect(std::move(probePipe), eqCond);
    }

    // ---------- Terminal operators ----------
    // Materialize path: no aggregation, no GROUP BY → emit joined rows directly.
    if (!aq.hasAggregation() && !aq.hasGroupBy()) {
        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < (int)aq.orderBy.size(); ++oi) {
            auto column = resolveOrderColumn(aq.orderBy[oi].expr, oi, aq.orderBy, aq.targets);
            if (!column) return fail("ORDER BY column not found in SELECT targets.");
            sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
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
        std::vector<GenericMatColumnDesc> materializedCols;

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
            expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
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
                            const auto& jk = jkIt->second;
                            std::string to = col->column + " + " + jk.expr +
                                (jk.isArray ? ("[" + idxVar + "]") : "") +
                                " * " + std::to_string(len);
                            size_t pos = expr.find(from);
                            if (pos != std::string::npos)
                                expr.replace(pos, from.size(), to);
                        }
                    }
                }
            }
            materialize->addColumn(bufferName, metalTypeForDataType(type),
                                   expr, displayName, sizeExpr, stringLen);
            materializedCols.push_back({displayName, bufferName,
                                        metalTypeForDataType(type), stringLen});
        }

        const std::string materializePhaseName = "ADHOC_multi_materialize";
        auto& matPhase = appendPhase(plan, materializePhaseName, std::move(materialize));
        if (probeUsesScalarLookup) scalarConsumerPhases.insert(materializePhaseName);
        // Register build-side CHAR_FIXED column buffers as read-only extra buffers.
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType cdType = aq.schema->columnType(tname, c);
                if (cdType == DataType::CHAR_FIXED) {
                    matPhase.extraBuffers.push_back({c, "char", true, false});
                }
            }
        }
        // Also add columns needed by cross-table filters.
        for (auto& [c, t] : crossExtraCols) {
            matPhase.extraBuffers.push_back({c, t, true, false});
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            const std::string sortRowsSym = "n_gpu_sort_multi_materialize_rows";
            attachMaterializedCountHook(matPhase, "d_adhoc_multi_result_count", sortRowsSym);
            if (!appendGenericGpuSort(plan, "multi_materialize", sortRowsSym,
                                      outputSize, materializedCols, sortSpec, error)) {
                return std::nullopt;
            }
        }
        return finalizePlan();
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
            std::vector<const FuncCall*> aggCalls;
            bool isDivision = false;
            AggCallWalker::walk(target.expr, aggCalls, isDivision);
            ExprPtr projectionExpr = scalarProjectionExprForSingleAgg(target);
            if (aggCalls.size() == 1 && !projectionExpr)
                return fail("Scalar aggregation: unsupported aggregate projection expression.");

            if (func == AggFunc::COUNT) {
                int idx = reduce->addAccumulator(accName, "1", "long");
                reduce->setAccumulatorResultAlias(alias, idx, 0, projectionExpr);
                continue;
            }

            // Complex aggregate with nested aggregates (e.g. const * SUM(a) / SUM(b)).
            // Decompose into individual accumulators, combine with setAverageResultAlias.
            if (target.expr && func == AggFunc::SUM) {
                // Only enter if the expression tree contains multiple aggregate calls;
                // simple SUM(x) is handled by the single-accumulator path below.
                if (aggCalls.size() >= 2 && isDivision) {
                    double outerConst = 1.0;
                    // Pattern: const * SUM(a) / SUM(b) → two accumulators + AVG alias
                    // Extract the constant multiplier from the expression tree.
                    if (auto* be = std::get_if<BinaryExpr>(&target.expr->node)) {
                        if (be->op == ExprOp::DIV) {
                            // Check left side: may be MUL(const, SUM(a)) or just SUM(a)
                            if (auto* leftBe = std::get_if<BinaryExpr>(&be->left->node)) {
                                if (leftBe->op == ExprOp::MUL) {
                                    if (auto* litL = std::get_if<Literal>(&leftBe->left->node)) {
                                        if (auto* iv = std::get_if<int>(&litL->value)) outerConst = (double)*iv;
                                        else if (auto* fv = std::get_if<float>(&litL->value)) outerConst = (double)*fv;
                                    }
                                    if (auto* litR = std::get_if<Literal>(&leftBe->right->node)) {
                                        if (auto* iv = std::get_if<int>(&litR->value)) outerConst *= (double)*iv;
                                        else if (auto* fv = std::get_if<float>(&litR->value)) outerConst *= (double)*fv;
                                    }
                                }
                            }
                            // Also check right side for MUL pattern
                            if (auto* rightBe = std::get_if<BinaryExpr>(&be->right->node)) {
                                if (rightBe->op == ExprOp::MUL) {
                                    if (auto* litL = std::get_if<Literal>(&rightBe->left->node)) {
                                        if (auto* iv = std::get_if<int>(&litL->value)) outerConst /= (double)*iv;
                                        else if (auto* fv = std::get_if<float>(&litL->value)) outerConst /= (double)*fv;
                                    }
                                }
                            }
                            auto leftCall = aggCalls[0];
                            auto rightCall = aggCalls[1];
                            if (!leftCall->args.empty() && !rightCall->args.empty()) {
                                auto numExpr = leftCall->args[0];
                                auto denExpr = rightCall->args[0];
                                std::string numRaw = exprToMetal(numExpr, idxVar, aq.schema);
                                std::string denRaw = exprToMetal(denExpr, idxVar, aq.schema);
                                if (outerConst != 1.0)
                                    numRaw = "(" + numRaw + " * " + std::to_string(outerConst) + "f)";
                                std::string numFinal = rewriteForProbe(numRaw, idxVar, carryVar, charFixedJoinKey, aq.schema);
                                std::string denFinal = rewriteForProbe(denRaw, idxVar, carryVar, charFixedJoinKey, aq.schema);
                                int numIdx = reduce->addAccumulator(accName + "_num", numFinal, "float");
                                int denIdx = reduce->addAccumulator(accName + "_den", denFinal, "float");
                                reduce->setAverageResultAlias(alias, numIdx, denIdx, 0);
                                continue;
                            }
                        }
                    }
                }
                // Simple aggregate (single FuncCall): fall through to single-accum path.
                // Complex unhandled pattern (multiple FuncCalls without division): fail.
                if (aggCalls.size() >= 2) {
                    return fail("Complex aggregate expression not decomposable.");
                }
            }
            if (aggCalls.size() >= 2)
                return fail("Complex aggregate expression not decomposable.");

            if (!target.agg->innerExpr || !exprSupported(target.agg->innerExpr, false))
                return fail("Aggregate expression not supported on GPU.");
            if (!isNumericLike(inferExprDataType(target.agg->innerExpr)))
                return fail("Aggregate expression must be numeric.");

            std::string finalExpr;
            if (auto carriedAgg = inSubqueryAggregateCarryExpr(aq, target, carryVar)) {
                finalExpr = *carriedAgg;
            } else {
                std::string raw = exprToMetal(target.agg->innerExpr, idxVar, aq.schema);
                finalExpr = rewriteForProbe(raw, idxVar, carryVar, charFixedJoinKey, aq.schema);
            }

            if (func == AggFunc::AVG) {
                int sumIdx = reduce->addAccumulator(accName + "_sum", finalExpr, "float");
                int cntIdx = reduce->addAccumulator(accName + "_count", "1.0f", "float");
                reduce->setAverageResultAlias(alias, sumIdx, cntIdx, 0, projectionExpr);
            } else {
                DataType vt = inferExprDataType(target.agg->innerExpr);
                std::string outType = (vt == DataType::FLOAT) ? "float" : "long";
                MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
                if (func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
                else if (func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
                if (op != MetalTGReduce::ReduceOp::SUM && vt != DataType::FLOAT) outType = "int";
                int idx = reduce->addAccumulator(accName, finalExpr, outType, "", "", op);
                reduce->setAccumulatorResultAlias(alias, idx, 0, projectionExpr);
            }
        }
        const std::string scalarPhaseName = "ADHOC_multi_probe_scalar";
        auto& phaseRef = appendPhase(plan, scalarPhaseName, std::move(reduce));
        if (probeUsesScalarLookup) scalarConsumerPhases.insert(scalarPhaseName);
        // Register build-side CHAR_FIXED columns as extra buffers
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType cdType = aq.schema->columnType(tname, c);
                if (cdType == DataType::CHAR_FIXED) {
                    phaseRef.extraBuffers.push_back({c, "char", true, false});
                }
            }
        }
        for (auto& [c, t] : crossExtraCols) {
            phaseRef.extraBuffers.push_back({c, "char", true, false});
        }
        if (aq.limit == 0)
            return fail("Scalar aggregation: LIMIT 0 is not supported without CPU post-processing.");
        return finalizePlan();
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
            kd.keyExpr = char1BucketExpr(*gc, idxVar, kd.numValues, aq.schema);
            if (kd.numValues == 0) { canUseGpuKeyedAgg = false; break; }
        } else {
            auto d = smallIntGroupDomain(*gc, aq.schema);
            if (!d || d->maxValue < d->minValue) { canUseGpuKeyedAgg = false; break; }
            kd.numValues = d->maxValue - d->minValue + 1;
            std::string groupValue = gc->column + "[i]";
            std::string keySource;
            if (gc->table == probeTable) {
                keySource = groupValue;
            } else {
                CarriedKey ck{gc->table, gc->column, gc->table};
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
        // ---------- MaterializeAgg fallback: emit raw rows on GPU, group generically ----------
        // Build a materialize plan that outputs group-by keys and aggregate inputs.
        // Generic GPU hash GROUP BY happens after materialization.
        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < (int)aq.orderBy.size(); ++oi) {
            auto column = resolveOrderColumn(aq.orderBy[oi].expr, oi, aq.orderBy, aq.targets);
            if (!column) return fail("ORDER BY column not found in SELECT targets.");
            sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
        }

        GenericGroupSpec groupSpec;
        for (const auto& g : aq.groupBy) {
            auto* gcRef = std::get_if<ColRef>(&g->node);
            if (gcRef) {
                groupSpec.keyColumns.push_back(displayNameForTargetByCol(aq, *gcRef));
            } else {
                // Non-ColRef GROUP BY (FuncCall, BinaryExpr, etc.):
                // match by position against non-aggregate targets.
                // Count how many GROUP BY items (ColRef + non) precede this one;
                // that's the index into the non-aggregate SELECT targets.
                int preceding = 0;
                for (auto& gbItem : aq.groupBy) {
                    if (&gbItem == &g) break;
                    preceding++;
                }
                for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
                    if (!aq.targets[ti].isAgg) {
                        if (preceding == 0) {
                            groupSpec.keyColumns.push_back(displayNameForTarget(aq.targets[ti], ti));
                            break;
                        }
                        preceding--;
                    }
                }
            }
        }
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            if (target.isAgg && target.agg) {
                bool complex = false;
                if (target.agg->func == AggFunc::SUM && target.expr) {
                    std::vector<const FuncCall*> aggCalls;
                    bool isDivision = false;
                    AggCallWalker::walk(target.expr, aggCalls, isDivision);
                    if (aggCalls.size() >= 2 && isDivision) {
                        std::string dn = displayNameForTarget(target, ti);
                        groupSpec.aggColumns.push_back(dn);
                        groupSpec.aggFuncs.push_back("RATIO");
                        groupSpec.aggColumns.push_back("__hidden_" + dn + "_den");
                        groupSpec.aggFuncs.push_back("RATIO_DEN");
                        complex = true;
                    }
                }
                if (!complex) {
                    groupSpec.aggColumns.push_back(displayNameForTarget(target, ti));
                    groupSpec.aggFuncs.push_back(aggFuncName(target.agg->func));
                }
            }
        }

        // Detect scalar subquery HAVING: compute threshold from grouped data.
        if (aq.having) {
            if (auto* cmp = std::get_if<Comparison>(&aq.having->node)) {
                if (auto* lit = cmp->right ? std::get_if<Literal>(&cmp->right->node) : nullptr) {
                    std::visit([&](auto&& v) {
                        using T = std::decay_t<decltype(v)>;
                        if constexpr (std::is_same_v<T, int>) {
                            int sentinel = v;
                                if (sentinel < -1000000) { // INT_MIN range → scalar subquery sentinel
                                    int sqIdx = sentinel - INT_MIN;
                                if (sqIdx >= 0 && sqIdx < (int)aq.subqueries.size()) {
                                    auto& sq = aq.subqueries[sqIdx];
                                    if (sq.type == AnalyzedQuery::Subquery::SCALAR_SUBQUERY) {
                                        // Find index of first SUM aggregate in aggColumns
                                        int aggIdx = -1;
                                        int aggCount = 0;
                                        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
                                            if (aq.targets[ti].isAgg && aq.targets[ti].agg) {
                                                if (aggFuncName(aq.targets[ti].agg->func) != "RATIO"
                                                    && aggFuncName(aq.targets[ti].agg->func) != "RATIO_DEN") {
                                                    if (aggIdx < 0) aggIdx = aggCount;
                                                }
                                                aggCount++;
                                            }
                                        }
                                        if (aggIdx < 0 && aggCount > 0) aggIdx = 0;
                                        // Extract factor from subquery JSON (e.g. 0.0001)
                                        double factor = 0.0;
                                        try {
                                            nlohmann::json sqJson = nlohmann::json::parse(sq.sql);
                                            if (sqJson.contains("SelectStmt")) {
                                                auto& ss = sqJson["SelectStmt"];
                                                if (ss.contains("targetList") && ss["targetList"].is_array()) {
                                                    for (auto& t : ss["targetList"]) {
                                                        if (!t.contains("ResTarget")) continue;
                                                        auto& rt = t["ResTarget"];
                                                        if (!rt.contains("val")) continue;
                                                        auto& val = rt["val"];
                                                        if (!val.contains("A_Expr")) continue;
                                                        auto& ae = val["A_Expr"];
                                                        if (!ae.contains("rexpr")) continue;
                                                        auto& rexpr = ae["rexpr"];
                                                        if (!rexpr.contains("A_Const")) continue;
                                                        auto& ac = rexpr["A_Const"];
                                                        // fval can be "Float" or direct "fval" key
                                                        if (ac.contains("val") && ac["val"].contains("Float")) {
                                                            factor = ac["val"]["Float"].value("fval", 0.0);
                                                        } else if (ac.contains("ival") && ac["ival"].contains("ival")) {
                                                            factor = std::stod(ac["ival"]["ival"].get<std::string>());
                                                        } else if (ac.contains("fval") && ac["fval"].contains("fval")) {
                                                            factor = std::stod(ac["fval"]["fval"].get<std::string>());
                                                        }
                                                        if (factor != 0.0) break;
                                                    }
                                                }
                                            }
                                        } catch (...) {}
                                        groupSpec.havingAggIdx = aggIdx;
                                        groupSpec.havingMultiplier = factor;
                                        groupSpec.havingSentinel = sentinel;
                                    }
                                }
                            }
                        }
                    }, lit->value);
                }
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
        std::vector<GenericMatColumnDesc> materializedCols;

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
            std::string distinctDomainSymbol;
            int inputScaleDown = 0;

            std::string expr;
            ExprPtr denExpr;
            if (target.isAgg && target.agg) {
                // Check for complex aggregate (e.g. SUM(CASE...) / SUM(volume))
                bool complex = false;
                if (target.agg->func == AggFunc::SUM && target.expr) {
                    std::vector<const FuncCall*> aggCalls;
                    bool isDivision = false;
                    AggCallWalker::walk(target.expr, aggCalls, isDivision);
                    if (aggCalls.size() >= 2 && isDivision && !aggCalls[0]->args.empty() && !aggCalls[1]->args.empty()) {
                        // Emit num column with the ACTUAL display name
                        ExprPtr numExpr = aggCalls[0]->args[0];
                        expr = exprToMetal(numExpr, idxVar);
                        // For CHAR_FIXED columns in multi-instance tables (e.g. n1/n2),
                        // prioritize the charFixedKeys entry matching the ColRef's alias
                        // so the correct carry variable is used for indexing.
                        {
                            std::set<std::string> exprAliases;
                            std::function<void(const ExprPtr&)> collectAliases;
                            collectAliases = [&](const ExprPtr& e) {
                                if (!e) return;
                                if (auto* cr = std::get_if<ColRef>(&e->node)) {
                                    if (cr->dataType == DataType::CHAR_FIXED && !cr->tableAlias.empty())
                                        exprAliases.insert(cr->tableAlias);
                                } else if (auto* be = std::get_if<BinaryExpr>(&e->node)) {
                                    collectAliases(be->left);
                                    collectAliases(be->right);
                                } else if (auto* cw = std::get_if<CaseWhen>(&e->node)) {
                                    for (auto& b : cw->branches) {
                                        if (b.condition) {
                                            if (auto* cmp = std::get_if<Comparison>(&b.condition->node)) {
                                                collectAliases(cmp->left);
                                                collectAliases(cmp->right);
                                            }
                                        }
                                        collectAliases(b.result);
                                    }
                                    if (cw->elseResult) collectAliases(cw->elseResult);
                                } else if (auto* fc = std::get_if<FuncCall>(&e->node)) {
                                    for (auto& a : fc->args) collectAliases(a);
                                }
                            };
                            collectAliases(numExpr);
                            // Alias-based matching in rewriteForProbe handles
                            // disambiguation — no reordering needed.
                            expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
                        }
                        type = DataType::FLOAT;
                        inputScaleDown = numericScaleForExpr(numExpr, aq.schema);
                        int numLen = fixedStringLenForExpr(aggCalls[0]->args[0], aq.schema);
                        std::string numMetalType = metalTypeForDataType(type);
                        materialize->addColumn(bufferName, numMetalType,
                                               expr, displayName, sizeExpr, numLen);
                        materializedCols.push_back({displayName, bufferName, numMetalType,
                                                    numLen, inputScaleDown, false, ""});
                        matColIdx++;
                        // Set up denominator column (hidden name)
                        bufferName = "d_adhoc_matgb_" + std::to_string(matColIdx) + "_" + sanitizeIdentifier("__hidden_" + displayName + "_den");
                        std::string denDisplayName = "__hidden_" + displayName + "_den";
                        displayName = denDisplayName;
                        denExpr = aggCalls[1]->args[0];
                        expr = exprToMetal(denExpr, idxVar);
                        expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
                        type = DataType::FLOAT;
                        stringLen = fixedStringLenForExpr(aggCalls[1]->args[0], aq.schema);
                        sizeExpr = outputSize;
                        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
                        complex = true;
                    }
                }
                if (!complex) {
                    // Standard aggregate: emit raw input value for host group-by.
                    if (target.agg->func == AggFunc::COUNT_DISTINCT) {
                        if (!target.agg->innerExpr)
                            return fail("COUNT(DISTINCT) requires an inner expression.");
                        distinctDomainSymbol = distinctDomainSymbolForExpr(target.agg->innerExpr, aq.schema);
                        if (distinctDomainSymbol.empty())
                            return fail("COUNT(DISTINCT) has no schema distinct-domain metadata.");
                        expr = exprToMetal(target.agg->innerExpr, idxVar);
                        expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
                        type = inferExprDataType(target.agg->innerExpr);
                        inputScaleDown = numericScaleForExpr(target.agg->innerExpr, aq.schema);
                    } else if (target.agg->func == AggFunc::COUNT || target.agg->isStar) {
                        expr = "1.0f";
                        type = DataType::FLOAT;
                    } else if (target.agg->innerExpr) {
                        if (auto carriedAgg = inSubqueryAggregateCarryExpr(aq, target, carryVar)) {
                            expr = *carriedAgg;
                        } else {
                            expr = exprToMetal(target.agg->innerExpr, idxVar);
                            expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
                        }
                        type = DataType::FLOAT;
                        inputScaleDown = numericScaleForExpr(target.agg->innerExpr, aq.schema);
                    } else {
                        expr = "0";
                    }
                }
            } else {
                expr = materializeValueExpr(target.expr, idxVar, aq.schema);
                expr = rewriteForProbe(expr, idxVar, carryVar, charFixedJoinKey, aq.schema);
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
                            const auto& jk = jkIt->second;
                            std::string to = col->column + " + " + jk.expr +
                                (jk.isArray ? ("[" + idxVar + "]") : "") +
                                " * " + std::to_string(len);
                            size_t pos = expr.find(from);
                            if (pos != std::string::npos)
                                expr.replace(pos, from.size(), to);
                        }
                    }
                }
            }

            std::string outMetalType = metalTypeForDataType(type);
            materialize->addColumn(bufferName, outMetalType,
                                   expr, displayName, sizeExpr, stringLen);
            materializedCols.push_back({displayName, bufferName, outMetalType,
                                        stringLen, inputScaleDown, false,
                                        distinctDomainSymbol});
            matColIdx++;
        }

        const std::string matgbPhaseName = "ADHOC_multi_matgb";
        auto& matPhase = appendPhase(plan, matgbPhaseName, std::move(materialize));
        if (probeUsesScalarLookup) scalarConsumerPhases.insert(matgbPhaseName);

        // Register build-side CHAR_FIXED extra buffers on the newly created phase.
        for (const auto& [tname, cols] : neededByTable) {
            if (tname == probeTable) continue;
            for (const auto& c : cols) {
                DataType cdType = aq.schema->columnType(tname, c);
                if (cdType == DataType::CHAR_FIXED) {
                    matPhase.extraBuffers.push_back({c, "char", true, false});
                }
            }
        }
        const std::string groupTag = "multi_" + sanitizeIdentifier(probeTable);
        GenericGpuGroupSpec gbSpec;
        gbSpec.tag = groupTag;
        gbSpec.inputCounter = "d_adhoc_multi_result_count";
        gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
        gbSpec.capacityExpr = "next_pow2(" + outputSize + " * 2)";
        gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
        gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
        gbSpec.inputColumns = std::move(materializedCols);
        gbSpec.groupBy = groupSpec;
        attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
        appendGenericGpuGroupBy(plan, gbSpec);
        attachMaterializedCountHook(plan.phases.back(), gbSpec.outputCounter,
                                    "n_gpu_sort_" + groupTag + "_rows");
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            if (!appendGenericGpuSort(plan, "group_" + groupTag,
                                      "n_gpu_sort_" + groupTag + "_rows",
                                      gbSpec.capacityExpr,
                                      genericGpuGroupOutputColumns(gbSpec),
                                      sortSpec, error)) {
                return std::nullopt;
            }
        }
        return finalizePlan();
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
            d.charMap = aq.schema ? aq.schema->charDomain(gc->table, gc->column)
                                  : std::vector<char>{};
        } else {
            auto sd = smallIntGroupDomain(*gc, aq.schema);
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

        std::string finalExpr;
        if (auto carriedAgg = inSubqueryAggregateCarryExpr(aq, target, carryVar)) {
            finalExpr = *carriedAgg;
        } else {
            std::string raw = exprToMetal(target.agg->innerExpr, idxVar);
            finalExpr = rewriteForProbe(raw, idxVar, carryVar, charFixedJoinKey, aq.schema);
        }

        if (func == AggFunc::AVG) {
            bool isFloat = (vt == DataType::FLOAT);
            int fixedScale = isFloat ? numericScaleForExpr(target.agg->innerExpr, aq.schema) : 0;
            PendingAgg sumA;
            sumA.display = display;
            sumA.name = "a" + std::to_string(ti) + "_" + sname + "_sum";
            sumA.offset = valuesPerBucket;
            sumA.valueExpr = fixedScale > 0 ? scaledLongExpr(finalExpr, fixedScale) : finalExpr;
            sumA.scaleDown = fixedScale > 0 ? -fixedScale : -1;
            sumA.atomicOp = "add";
            sumA.funcName = "AVG";
            sumA.innerColumn = extractInnerCol(target.agg->innerExpr);
            if (isFloat && fixedScale <= 0) {
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
            std::string maxExpr = distinctDomainSymbolForExpr(target.agg->innerExpr, aq.schema);
            if (maxExpr.empty())
                return fail("COUNT(DISTINCT) on column '" + innerCol->column + "' — no schema distinct-domain metadata.");

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
            const int fixedScale = numericScaleForExpr(target.agg->innerExpr, aq.schema);
            if (fixedScale > 0) {
                p.valueExpr = scaledLongExpr(finalExpr, fixedScale);
                p.isLongPair = true;
                p.scaleDown = fixedScale;
                valuesPerBucket += 2;
            } else {
                p.isFloatSum = true;
                valuesPerBucket += 1;
            }
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

    const std::string groupPhaseName = "ADHOC_multi_probe_group";
    appendPhase(plan, groupPhaseName, std::move(agg));
    if (probeUsesScalarLookup) scalarConsumerPhases.insert(groupPhaseName);

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

    if (!aq.orderBy.empty() || aq.limit >= 0) {
        return fail("Grouped aggregation: ORDER BY/LIMIT on legacy keyed aggregation requires the IR GPU sort route.");
    }
    return finalizePlan();
}

} // namespace

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
        const AnalyzedQuery& aq,
        MetalQueryPlan& plan) {
    return buildCorrelatedScalarPreAggs(aq, plan);
}

std::string genericScalarLookupKeyExpr(
        const GenericScalarLookupInfo& info,
        size_t keyIndex,
        const std::string& idxVar,
        const std::string& probeTable,
        const SchemaProvider* schema) {
    return scalarLookupKeyExpr(info, keyIndex, idxVar, probeTable, schema);
}

std::string rewriteGenericScalarSentinels(
        const std::string& cond,
        const std::string& idxVar,
        const std::vector<GenericScalarLookupInfo>& lookups,
        const std::string& probeTable,
        const SchemaProvider* schema) {
    return rewriteScalarSentinels(cond, idxVar, lookups, probeTable, schema);
}

bool referencesGenericScalarSentinel(
        const std::string& text,
        const std::vector<GenericScalarLookupInfo>& lookups) {
    return referencesScalarSentinel(text, lookups);
}

bool referencesGenericScalarLookupBuffer(
        const std::string& text,
        const std::vector<GenericScalarLookupInfo>& lookups) {
    return referencesScalarLookupBuffer(text, lookups);
}

void attachGenericScalarLookupBuffers(
        MetalQueryPlan::Phase& phase,
        const std::vector<GenericScalarLookupInfo>& lookups) {
    attachScalarLookupBuffers(phase, lookups);
}

std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<MetalQueryPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (!aq.isSingleTable()) return fail("Single-table planner: query references multiple tables.");
    if (aq.tables.empty()) return fail("Single-table planner: no tables in FROM clause.");
    if (aq.tables[0] == "__subquery__") return fail("Single-table planner: subqueries not supported.");
    if (!filtersSupported(aq.filters)) return fail("Single-table planner: WHERE clause contains expressions not supported on GPU.");

    std::string subError;
    if (auto scalar = buildScalarAggPlan(aq, &subError)) {
        return scalar;
    }
    if (auto grouped = buildGroupedAggPlan(aq, &subError)) {
        return grouped;
    }
    if (auto scalarLookupMat = buildSingleTableScalarLookupMaterializePlan(aq, &subError)) {
        return scalarLookupMat;
    }
    auto mat = buildMaterializePlan(aq, error);
    return mat;
}

std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan(
    const AnalyzedQuery& aq, std::string* error) {
    return buildGenericMultiTableAdhocPlan_impl(aq, error);
}

} // namespace codegen
