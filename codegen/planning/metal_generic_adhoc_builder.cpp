#include "metal_generic_adhoc_builder.h"
#include "metal_plan_common.h"
#include "tpch_schema.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
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

bool exprIsColumn(const ExprPtr& expr, const ColRef& expected) {
    auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr;
    return col && col->table == expected.table && col->column == expected.column;
}

struct GroupDomain {
    int minValue = 0;
    int maxValue = 0;
};

std::optional<GroupDomain> smallIntGroupDomain(const ColRef& col) {
    if (col.dataType != DataType::INT && col.dataType != DataType::DATE) return std::nullopt;

    if (col.column == "c_nationkey" || col.column == "s_nationkey" ||
        col.column == "n_nationkey") {
        return GroupDomain{0, 24};
    }
    if (col.column == "n_regionkey" || col.column == "r_regionkey") {
        return GroupDomain{0, 4};
    }
    if (col.column == "p_size") {
        return GroupDomain{1, 50};
    }
    if (col.column == "l_linenumber") {
        return GroupDomain{1, 7};
    }
    if (col.column == "o_shippriority") {
        return GroupDomain{0, 0};
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
    if (!orderCol) return std::nullopt;

    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        std::string displayName = displayNameForTarget(target, i);
        if (displayName == orderCol->column) return displayName;
        if (auto* targetCol = target.expr ? std::get_if<ColRef>(&target.expr->node) : nullptr) {
            if (targetCol->column == orderCol->column) return displayName;
        }
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
    if (!value) return true;
    if (type == DataType::DATE) return isDateLiteralString(*value);
    if (type == DataType::CHAR1) return value->size() == 1;
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

int fixedStringLenForExpr(const ExprPtr& expr) {
    if (!expr) return 0;
    auto* col = std::get_if<ColRef>(&expr->node);
    if (!col) return 0;
    if (col->dataType == DataType::CHAR1) return 1;
    if (col->dataType != DataType::CHAR_FIXED) return 0;
    const auto& cdef = TPCHSchema::instance().table(col->table).col(col->column);
    return cdef.fixedWidth;
}

std::string materializeValueExpr(const ExprPtr& expr, const std::string& idxVar) {
    if (auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr) {
        if (col->dataType == DataType::CHAR1) {
            return col->column + " + " + idxVar;
        }
        if (col->dataType == DataType::CHAR_FIXED) {
            int len = fixedStringLenForExpr(expr);
            return col->column + " + " + idxVar + " * " + std::to_string(len);
        }
    }
    return exprToMetal(expr, idxVar);
}

bool predSupported(const PredPtr& pred) {
    if (!pred) return true;
    return std::visit([&](auto&& node) -> bool {
        using Node = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<Node, Comparison>) {
            if (node.op == CmpOp::EQ || node.op == CmpOp::NE) {
                if (fixedStringCompareSupported(node.left, node.right) ||
                    fixedStringCompareSupported(node.right, node.left)) return true;
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
        const auto& table = TPCHSchema::instance().table(aq.tables[0]);
        if (!table.columns.empty()) scanColumns.insert(table.columns.front().name);
    }
    auto scan = makeScanForCols(aq.tables[0], idxVar, scanColumns);
    return maybeSelect(std::move(scan), combineFilters(aq.filters, idxVar));
}

std::optional<MetalQueryPlan> buildScalarAggPlan(const AnalyzedQuery& aq) {
    if (!aq.hasAggregation()) return std::nullopt;
    if (aq.hasGroupBy() || aq.having || !aq.orderBy.empty() || aq.limit >= 0) return std::nullopt;

    std::set<std::string> usedColumns;
    for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);

    for (const auto& target : aq.targets) {
        if (!target.isAgg || !target.agg) return std::nullopt;
        const AggFunc func = target.agg->func;
        if (func != AggFunc::SUM && func != AggFunc::COUNT && func != AggFunc::AVG &&
            func != AggFunc::MIN && func != AggFunc::MAX) return std::nullopt;
        if (func != AggFunc::COUNT) {
            if (!target.agg->innerExpr || !exprSupported(target.agg->innerExpr, false))
                return std::nullopt;
            if (!isNumericLike(inferExprDataType(target.agg->innerExpr)))
                return std::nullopt;
            collectColumns(target.agg->innerExpr, usedColumns);
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

std::optional<MetalQueryPlan> buildGroupedAggPlan(const AnalyzedQuery& aq) {
    if (!aq.hasAggregation() || !aq.hasGroupBy()) return std::nullopt;
    if (aq.groupBy.size() != 1 || aq.having || !aq.orderBy.empty() || aq.limit >= 0)
        return std::nullopt;

    auto* groupCol = aq.groupBy[0] ? std::get_if<ColRef>(&aq.groupBy[0]->node) : nullptr;
    if (!groupCol) return std::nullopt;
    auto domain = smallIntGroupDomain(*groupCol);
    if (!domain) return std::nullopt;

    struct PendingAgg {
        std::string displayName;
        std::string name;
        int offset = 0;
        std::string valueExpr;
        bool isLongPair = false;
        int scaleDown = 0;
    };

    std::set<std::string> usedColumns;
    for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);
    collectColumns(aq.groupBy[0], usedColumns);

    std::vector<PendingAgg> pending;
    int valuesPerBucket = 0;
    bool sawGroupTarget = false;
    bool sawCount = false;
    std::string keyDisplayName = groupCol->column;
    const std::string idxVar = "i";

    for (size_t targetIndex = 0; targetIndex < aq.targets.size(); ++targetIndex) {
        const auto& target = aq.targets[targetIndex];
        if (!target.isAgg) {
            if (!exprIsColumn(target.expr, *groupCol) || sawGroupTarget) return std::nullopt;
            sawGroupTarget = true;
            keyDisplayName = displayNameForTarget(target, targetIndex);
            continue;
        }

        if (!target.agg) return std::nullopt;
        const AggFunc func = target.agg->func;
        if (func == AggFunc::COUNT) {
            PendingAgg agg;
            agg.displayName = displayNameForTarget(target, targetIndex);
            agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
            agg.offset = valuesPerBucket++;
            agg.valueExpr = "1u";
            pending.push_back(std::move(agg));
            sawCount = true;
            continue;
        }

        if (func != AggFunc::SUM) return std::nullopt;
        if (!target.agg->innerExpr || !exprSupported(target.agg->innerExpr, false)) return std::nullopt;
        DataType valueType = inferExprDataType(target.agg->innerExpr);
        if (valueType != DataType::INT && valueType != DataType::DATE) return std::nullopt;
        collectColumns(target.agg->innerExpr, usedColumns);

        PendingAgg agg;
        agg.displayName = displayNameForTarget(target, targetIndex);
        agg.name = "a" + std::to_string(targetIndex) + "_" + sanitizeIdentifier(agg.displayName);
        agg.offset = valuesPerBucket;
        agg.valueExpr = exprToMetal(target.agg->innerExpr, idxVar);
        agg.isLongPair = true;
        valuesPerBucket += 2;
        pending.push_back(std::move(agg));
    }

    if (!sawGroupTarget || pending.empty() || !sawCount) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_GROUP";
    auto filtered = makeFilteredScan(aq, usedColumns, idxVar);

    const std::string groupValue = groupCol->column + "[" + idxVar + "]";
    const std::string guard = "(" + groupValue + " >= " + std::to_string(domain->minValue) +
                              " && " + groupValue + " <= " + std::to_string(domain->maxValue) + ")";
    filtered = maybeSelect(std::move(filtered), guard);

    std::string bucketExpr = groupValue;
    if (domain->minValue != 0) {
        bucketExpr = "(" + bucketExpr + " - " + std::to_string(domain->minValue) + ")";
    }
    const int numBuckets = domain->maxValue - domain->minValue + 1;
    auto agg = std::make_unique<MetalKeyedAgg>(
        std::move(filtered), "d_adhoc_group_aggs", bucketExpr,
        numBuckets, valuesPerBucket, std::to_string(numBuckets * valuesPerBucket));
    agg->setKeyResult(keyDisplayName, domain->minValue);

    for (const auto& pendingAgg : pending) {
        agg->addAggregate(pendingAgg.displayName, pendingAgg.offset, pendingAgg.valueExpr,
                          "add", pendingAgg.isLongPair, pendingAgg.scaleDown);
    }

    appendPhase(plan, "ADHOC_single_table_group", std::move(agg));
    return plan;
}

std::optional<MetalQueryPlan> buildMaterializePlan(const AnalyzedQuery& aq) {
    if (aq.hasAggregation() || aq.hasGroupBy() || aq.having)
        return std::nullopt;
    if (aq.targets.empty()) return std::nullopt;

    MetalQueryPlan::CpuSort cpuSort;
    cpuSort.limit = aq.limit;
    for (const auto& order : aq.orderBy) {
        auto column = orderColumnForExpr(order.expr, aq.targets);
        if (!column) return std::nullopt;
        cpuSort.keys.push_back({*column, order.descending});
    }

    std::set<std::string> usedColumns;
    for (const auto& filter : aq.filters) collectColumns(filter, usedColumns);
    for (const auto& target : aq.targets) {
        if (!target.expr || !materializeExprSupported(target.expr)) return std::nullopt;
        collectColumns(target.expr, usedColumns);
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_SINGLE_TABLE_MATERIALIZE";
    if (!cpuSort.keys.empty() || cpuSort.limit >= 0) plan.cpuSort = cpuSort;
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
        int stringLen = fixedStringLenForExpr(target.expr);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        materialize->addColumn(bufferName, metalTypeForDataType(type),
                       materializeValueExpr(target.expr, idxVar), displayName,
                       sizeExpr, stringLen);
    }

    appendPhase(plan, "ADHOC_single_table_materialize", std::move(materialize));
    return plan;
}

} // namespace

std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq) {
    if (!aq.isSingleTable()) return std::nullopt;
    if (aq.tables.empty() || aq.tables[0] == "__subquery__") return std::nullopt;
    if (!filtersSupported(aq.filters)) return std::nullopt;

    if (auto scalar = buildScalarAggPlan(aq)) return scalar;
    if (auto grouped = buildGroupedAggPlan(aq)) return grouped;
    return buildMaterializePlan(aq);
}

} // namespace codegen