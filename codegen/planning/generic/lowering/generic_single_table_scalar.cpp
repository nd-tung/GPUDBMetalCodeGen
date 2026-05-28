#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_cost_model.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "core/schema_provider.h"
#include "metal_plan_common.h"

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
        const GenericRelNode& scanNode,
        const GenericScanDetail& scan) {
    auto columnFromScanOutput = [&](const std::string& name)
            -> std::optional<std::pair<std::string, TypeInfo>> {
        for (const auto& col : scanNode.output.columns) {
            if (col.relationInstance.value != scan.relationInstance.value)
                continue;
            if (!name.empty() && col.name != name)
                continue;
            if (col.name.empty())
                continue;
            return std::make_pair(col.name, col.type);
        }
        return std::nullopt;
    };
    auto columnFromSchema = [&](const std::string& name)
            -> std::optional<std::pair<std::string, TypeInfo>> {
        if (!ir.schema || name.empty() || !ir.schema->hasColumn(scan.table, name))
            return std::nullopt;
        return std::make_pair(
            name,
            TypeInfo{ir.schema->columnType(scan.table, name),
                     ir.schema->columnFixedWidth(scan.table, name)});
    };

    if (const auto* inst = ir.findRelationInstance(scan.relationInstance)) {
        if (const auto* rel = ir.findRelation(inst->relation)) {
            if (!rel->primaryKeyColumn.empty()) {
                if (auto anchor = columnFromScanOutput(rel->primaryKeyColumn))
                    return anchor;
                if (auto anchor = columnFromSchema(rel->primaryKeyColumn))
                    return anchor;
            }
        }
    }
    if (ir.schema) {
        for (const auto& name : ir.schema->columnNames(scan.table)) {
            if (auto anchor = columnFromSchema(name))
                return anchor;
        }
    }
    return columnFromScanOutput("");
}

MetalTGReduce::ReduceOp scalarReduceOpToMetal(
        ScalarReduceAccumulatorSpec::Op op) {
    switch (op) {
        case ScalarReduceAccumulatorSpec::Op::Min:
            return MetalTGReduce::ReduceOp::MIN;
        case ScalarReduceAccumulatorSpec::Op::Max:
            return MetalTGReduce::ReduceOp::MAX;
        case ScalarReduceAccumulatorSpec::Op::Sum:
            return MetalTGReduce::ReduceOp::SUM;
    }
    return MetalTGReduce::ReduceOp::SUM;
}

} // namespace

std::optional<MetalQueryPlan> lowerSingleTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto finish = [&](MetalQueryPlan&& plan) -> std::optional<MetalQueryPlan> {
        return attachGenericCostTrace(
            std::optional<MetalQueryPlan>{std::move(plan)}, ir,
            "single_table_scalar_aggregate");
    };

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

        auto reduceSpec = buildScalarReduceAccumulatorSpec(
            agg->func, agg->arg, valueExpr);
        if (!reduceSpec) {
            return fail(error, "IR scalar aggregate lowerer: unsupported aggregate '" +
                               aggFuncName(agg->func) + "'.");
        }

        int accIndex = reduce->addAccumulator(
            accName, reduceSpec->valueExpr, reduceSpec->metalType, "", "",
            scalarReduceOpToMetal(reduceSpec->op));
        reduce->setAccumulatorResultAlias(
            alias, accIndex, reduceSpec->outputScale, nullptr);
    }

    if (!hasInputColumnReference) {
        auto anchor = scanAnchorColumn(ir, *shape->scan, *scan);
        if (!anchor) {
            return fail(error, "IR scalar aggregate lowerer: count-only scan has no anchor column.");
        }
        scanRoot->addColumn(anchor->first, metalTypeForType(anchor->second));
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SINGLE_TABLE_SCALAR";
    plan.chunkable = true;
    appendPhase(plan, "GENERIC_ir_single_table_scalar", std::move(reduce));
    return finish(std::move(plan));
}

} // namespace codegen
