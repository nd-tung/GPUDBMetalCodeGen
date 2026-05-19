#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "metal_plan_common.h"
#include "tpch_schema.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

std::optional<std::pair<std::string, TypeInfo>> scanAnchorColumn(
        const GenericRelPlan& ir,
        const GenericScanDetail& scan) {
    if (const auto* inst = ir.findRelationInstance(scan.relationInstance)) {
        if (const auto* rel = ir.findRelation(inst->relation)) {
            if (!rel->primaryKeyColumn.empty()) {
                TypeInfo type{DataType::INT, 0};
                try {
                    const auto& col = TPCHSchema::instance()
                        .table(scan.table)
                        .col(rel->primaryKeyColumn);
                    type = TypeInfo{col.type, col.fixedWidth};
                } catch (...) {}
                return std::make_pair(rel->primaryKeyColumn, type);
            }
        }
    }
    try {
        const auto& table = TPCHSchema::instance().table(scan.table);
        if (!table.columns.empty()) {
            const auto& col = table.columns.front();
            return std::make_pair(col.name, TypeInfo{col.type, col.fixedWidth});
        }
    } catch (...) {}
    return std::nullopt;
}

std::optional<double> numericLiteralValueLocal(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    return std::visit([](const auto& value) -> std::optional<double> {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, int64_t>)
            return static_cast<double>(value);
        else if constexpr (std::is_same_v<V, double>)
            return value;
        else
            return std::nullopt;
    }, lit->value);
}

bool scaleNeutralFactor(const GenericExprPtr& expr) {
    if (!expr) return false;
    if (numericLiteralValueLocal(expr)) return true;
    return expr->type.type == DataType::INT;
}

int scalarSumFixedPointScale(const GenericExprPtr& expr) {
    const int directScale = numericScaleForExpr(expr);
    if (directScale > 0) return directScale;
    if (!expr) return 0;
    auto* bin = std::get_if<GenericBinaryExpr>(&expr->node);
    if (!bin) return 0;

    const int leftScale = scalarSumFixedPointScale(bin->left);
    const int rightScale = scalarSumFixedPointScale(bin->right);
    switch (bin->op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
            if (leftScale > 0 && leftScale == rightScale) return leftScale;
            if (leftScale > 0 && numericLiteralValueLocal(bin->right)) return leftScale;
            if (rightScale > 0 && numericLiteralValueLocal(bin->left)) return rightScale;
            return 0;
        case ExprOp::MUL:
            if (leftScale > 0 && rightScale > 0)
                return std::max(leftScale, rightScale);
            if (leftScale > 0 && scaleNeutralFactor(bin->right)) return leftScale;
            if (rightScale > 0 && scaleNeutralFactor(bin->left)) return rightScale;
            return 0;
        case ExprOp::DIV:
            return 0;
    }
    return 0;
}

} // namespace

std::optional<MetalQueryPlan> lowerSingleTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto shape = parseSingleTableScalarAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!scan || !aggregate)
        return fail(error, "IR scalar aggregate lowerer: malformed scan/aggregate detail.");
    if (!aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR scalar aggregate lowerer: no aggregate outputs.");
    if (aggregate->having)
        return fail(error, "IR scalar aggregate lowerer: HAVING is not supported.");

    const std::string idxVar = "i";
    auto scanOp = makeAutoScan(scan->table, idxVar);
    auto* scanRoot = scanOp.get();
    std::unique_ptr<MetalOperator> pipe = std::move(scanOp);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate)) {
            return fail(error, "IR scalar aggregate lowerer: filter predicate is not supported.");
        }
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }

    auto reduce = std::make_unique<MetalTGReduce>(std::move(pipe), "d_ir_scalar");
    bool hasInputColumnReference = shape->filter != nullptr;
    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        if (!projection.expr)
            return fail(error, "IR scalar aggregate lowerer: null aggregate expression.");
        auto* agg = std::get_if<GenericAggregateExpr>(&projection.expr->node);
        if (!agg)
            return fail(error, "IR scalar aggregate lowerer: projection is not an aggregate.");

        const std::string alias = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        const std::string accName = "a" + std::to_string(i) + "_" +
                                    sanitizeIdentifier(alias);

        if (agg->func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(alias, accIndex, 0, nullptr);
            continue;
        }

        if (!agg->arg)
            return fail(error, "IR scalar aggregate lowerer: aggregate '" +
                               aggFuncName(agg->func) + "' requires an argument.");
        if (!materializeExprSupported(agg->arg))
            return fail(error, "IR scalar aggregate lowerer: aggregate argument is not supported.");

        hasInputColumnReference = true;
        std::string valueExpr = genericExprToMetal(agg->arg, idxVar);
        if (agg->func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum", valueExpr, "float");
            int countIndex = reduce->addAccumulator(accName + "_count", "1.0f", "float");
            reduce->setAverageResultAlias(alias, sumIndex, countIndex, 0, nullptr);
            continue;
        }

        int outputScale = 0;
        std::string outputType = agg->arg->type.type == DataType::FLOAT ? "float" : "long";
        MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
        if (agg->func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
        else if (agg->func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
        else if (agg->func != AggFunc::SUM) {
            return fail(error, "IR scalar aggregate lowerer: unsupported aggregate '" +
                               aggFuncName(agg->func) + "'.");
        }
        if (op == MetalTGReduce::ReduceOp::SUM &&
            agg->arg->type.type == DataType::FLOAT) {
            outputScale = scalarSumFixedPointScale(agg->arg);
            if (outputScale > 0) {
                valueExpr = scaledLongExpr(valueExpr, outputScale);
                outputType = "long";
            }
        } else if (op != MetalTGReduce::ReduceOp::SUM &&
                   agg->arg->type.type != DataType::FLOAT) {
            outputType = "int";
        }

        int accIndex = reduce->addAccumulator(accName, valueExpr, outputType, "", "", op);
        reduce->setAccumulatorResultAlias(alias, accIndex, outputScale, nullptr);
    }

    if (!hasInputColumnReference) {
        auto anchor = scanAnchorColumn(ir, *scan);
        if (!anchor) {
            return fail(error, "IR scalar aggregate lowerer: count-only scan has no anchor column.");
        }
        scanRoot->addColumn(anchor->first, metalTypeForType(anchor->second));
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SINGLE_TABLE_SCALAR";
    plan.chunkable = true;
    appendPhase(plan, "GENERIC_ir_single_table_scalar", std::move(reduce));
    return plan;
}

} // namespace codegen
