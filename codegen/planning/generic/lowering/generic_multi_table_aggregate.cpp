#include "generic/lowering/generic_ir_physical_planner.h"

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_join_carry.h"
#include "generic/lowering/generic_multi_table_checks.h"
#include "generic/lowering/generic_multi_table_join_lowering.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_scalar_lookup.h"
#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

void prependPlanPhases(MetalQueryPlan& target, MetalQueryPlan& prefix) {
    for (auto& helper : prefix.helpers) {
        if (std::find(target.helpers.begin(), target.helpers.end(), helper) ==
            target.helpers.end()) {
            target.helpers.push_back(std::move(helper));
        }
    }
    if (!prefix.phases.empty()) {
        target.phases.insert(
            target.phases.begin(),
            std::make_move_iterator(prefix.phases.begin()),
            std::make_move_iterator(prefix.phases.end()));
    }
}

} // namespace

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableGroupedAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!aggregate)
        return fail(error, "IR multi-table grouped aggregate lowerer: malformed aggregate detail.");
    if (aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR multi-table grouped aggregate lowerer: no aggregate outputs.");

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& group : aggregate->groupBy)
        neededExprs.push_back(group);
    for (const auto& projection : aggregate->aggregates) {
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table grouped aggregate lowerer: non-aggregate projection.");
        if (agg->arg)
            neededExprs.push_back(agg->arg);
    }

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq) && groupedAggregateNeedsScalarPreAgg(*shape)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_GROUP_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table grouped aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "GENERIC_IR_MULTI_TABLE_GROUP", aq,
        scalarLookups.empty() ? nullptr : &scalarLookups, error);
    if (!lowering) return std::nullopt;
    prependPlanPhases(lowering->plan, scalarPreAggPlan);

    const std::string idxVar = "i";
    const std::string resultCounter = "d_ir_multi_group_input_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(lowering->probePipe), resultCounter, "1");

    std::vector<GenericMatColumnDesc> materializedCols;
    GenericGroupSpec groupSpec;
    std::vector<IrGroupKeyDesc> groupKeys;
    int matColIdx = 0;

    auto addInputColumn = [&](const std::string& displayName,
                              const TypeInfo& type,
                              const GenericExprPtr& expr,
                              int scaleDown,
                              const std::string& distinctDomainSymbol) -> bool {
        if (!materializeExprSupported(expr)) {
            if (error)
                *error = "IR multi-table grouped aggregate lowerer: input expression '" +
                         displayName + "' is not supported.";
            return false;
        }
        if (exprNeedsCarriedString(expr, lowering->carryMap)) {
            if (error)
                *error = "IR multi-table grouped aggregate lowerer: carried string input '" +
                         displayName + "' is not supported yet.";
            return false;
        }
        int stringLen = materializedStringLenForExpr(expr, lowering->carryMap);
        std::string sizeExpr = lowering->outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_multi_group_" + std::to_string(matColIdx++) +
                                 "_" + sanitizeIdentifier(displayName);
        std::string metalType = metalTypeForType(type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetalWithCarryMap(
                                   expr, idxVar, lowering->carryMap),
                               displayName, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            displayName, bufferName, metalType, stringLen, scaleDown, false,
            distinctDomainSymbol});
        return true;
    };

    for (size_t i = 0; i < aggregate->groupBy.size(); ++i) {
        const auto& group = aggregate->groupBy[i];
        const std::string displayName = groupDisplayNameForAggregate(*aggregate, i);
        groupSpec.keyColumns.push_back(displayName);
        IrGroupKeyDesc key;
        key.displayName = displayName;
        groupKeys.push_back(std::move(key));
        if (!addInputColumn(displayName, group ? group->type : TypeInfo{DataType::INT, 0},
                            group, 0, "")) {
            return std::nullopt;
        }
    }

    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table grouped aggregate lowerer: non-aggregate projection.");
        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;

        GenericExprPtr inputExpr;
        TypeInfo inputType{DataType::FLOAT, 0};
        int inputScaleDown = 0;
        std::string distinctDomainSymbol;
        std::string funcName = aggregateOutputFuncFor(*aggregate, i, agg->func);

        if (agg->func == AggFunc::COUNT) {
            GenericExpr lit;
            lit.type = inputType;
            lit.node = GenericLiteralExpr{1.0, inputType};
            inputExpr = std::make_shared<GenericExpr>(std::move(lit));
            funcName = "COUNT";
        } else {
            if (!agg->arg)
                return fail(error, "IR multi-table grouped aggregate lowerer: aggregate '" +
                                   aggFuncName(agg->func) + "' requires an argument.");
            inputExpr = agg->arg;
            if (agg->func == AggFunc::COUNT_DISTINCT) {
                distinctDomainSymbol = distinctDomainSymbolForExpr(agg->arg);
                if (distinctDomainSymbol.empty())
                    return fail(error, "IR multi-table grouped aggregate lowerer: COUNT(DISTINCT) has no schema distinct-domain metadata.");
                inputType = agg->arg->type;
                funcName = "COUNT_DISTINCT";
            } else if (agg->func == AggFunc::SUM || agg->func == AggFunc::AVG) {
                inputScaleDown = numericScaleForExpr(agg->arg);
            } else if (agg->func != AggFunc::MIN && agg->func != AggFunc::MAX) {
                return fail(error, "IR multi-table grouped aggregate lowerer: unsupported aggregate " +
                                   aggFuncName(agg->func) + ".");
            }
        }

        groupSpec.aggColumns.push_back(displayName);
        groupSpec.aggFuncs.push_back(funcName);
        if (!addInputColumn(displayName, inputType, inputExpr, inputScaleDown,
                            distinctDomainSymbol)) {
            return std::nullopt;
        }
    }

    groupSpec.outputColumns = aggregate->outputOrder;
    if (!configureAggregateHaving(*aggregate, groupSpec, aq, &*shape, error))
        return std::nullopt;

    auto& matPhase = appendPhase(lowering->plan, "GENERIC_ir_multi_table_group_materialize",
                                 std::move(materialize));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(matPhase, scalarLookups);

    const std::string groupTag = "ir_multi_table_group";
    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = "next_pow2(" + lowering->outputSize + " * 2)";
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
    gbSpec.inputColumns = std::move(materializedCols);
    gbSpec.groupBy = std::move(groupSpec);
    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
    appendGenericGpuGroupBy(lowering->plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(lowering->plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, *aggregate, groupKeys);
            if (!name)
                return fail(error, "IR multi-table grouped aggregate lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        if (!appendGenericGpuSort(lowering->plan, "group_" + groupTag,
                                  sortRowsSym, gbSpec.capacityExpr,
                                  genericGpuGroupOutputColumns(gbSpec),
                                  sortSpec, error)) {
            return std::nullopt;
        }
    }

    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableGroupedAggregateIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableGroupedAggregateIRToMetalImpl(ir, &aq, error);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableScalarAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!aggregate)
        return fail(error, "IR multi-table scalar aggregate lowerer: malformed aggregate detail.");
    if (!aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR multi-table scalar aggregate lowerer: no aggregate outputs.");
    if (aggregate->having)
        return fail(error, "IR multi-table scalar aggregate lowerer: HAVING is not supported.");

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& projection : aggregate->aggregates) {
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table scalar aggregate lowerer: non-aggregate projection.");
        if (agg->arg)
            neededExprs.push_back(agg->arg);
    }

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_SCALAR_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table scalar aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "GENERIC_IR_MULTI_TABLE_SCALAR", aq,
        scalarLookups.empty() ? nullptr : &scalarLookups, error);
    if (!lowering) return std::nullopt;
    prependPlanPhases(lowering->plan, scalarPreAggPlan);

    auto expressionSupported = [&](const GenericExprPtr& expr,
                                   const std::string& displayName) -> bool {
        if (!expr) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' requires an argument.";
            return false;
        }
        if (!materializeExprSupported(expr)) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' argument is not supported.";
            return false;
        }
        if (exprNeedsCarriedString(expr, lowering->carryMap)) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: carried string aggregate input '" +
                         displayName + "' is not supported yet.";
            return false;
        }
        if (expr->type.type != DataType::INT &&
            expr->type.type != DataType::DATE &&
            expr->type.type != DataType::FLOAT) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' argument must be numeric.";
            return false;
        }
        return true;
    };

    const std::string idxVar = "i";
    auto reduce = std::make_unique<MetalTGReduce>(
        std::move(lowering->probePipe), "d_ir_multi_scalar");
    std::vector<bool> consumed(aggregate->aggregates.size(), false);

    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        if (consumed[i]) continue;
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table scalar aggregate lowerer: non-aggregate projection.");

        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        const std::string accName = "a" + std::to_string(i) + "_" +
                                    sanitizeIdentifier(displayName);
        const std::string outputFunc = aggregateOutputFuncFor(*aggregate, i,
                                                              agg->func);

        if (outputFunc == "RATIO_DEN") {
            consumed[i] = true;
            continue;
        }

        if (outputFunc == "RATIO") {
            if (i + 1 >= aggregate->aggregates.size())
                return fail(error, "IR multi-table scalar aggregate lowerer: RATIO denominator is missing.");
            const auto& denProjection = aggregate->aggregates[i + 1];
            auto* denAgg = denProjection.expr
                ? std::get_if<GenericAggregateExpr>(&denProjection.expr->node)
                : nullptr;
            if (!denAgg || aggregateOutputFuncFor(*aggregate, i + 1,
                                                  denAgg->func) != "RATIO_DEN") {
                return fail(error, "IR multi-table scalar aggregate lowerer: RATIO denominator metadata is invalid.");
            }
            if (!expressionSupported(agg->arg, displayName) ||
                !expressionSupported(denAgg->arg, denProjection.name)) {
                return std::nullopt;
            }
            std::string numExpr = materializeExprToMetalWithCarryMap(
                agg->arg, idxVar, lowering->carryMap);
            std::string denExpr = materializeExprToMetalWithCarryMap(
                denAgg->arg, idxVar, lowering->carryMap);
            int numIdx = reduce->addAccumulator(accName + "_num", numExpr, "float");
            int denIdx = reduce->addAccumulator(accName + "_den", denExpr, "float");
            reduce->setAverageResultAlias(displayName, numIdx, denIdx, 0, nullptr);
            consumed[i] = true;
            consumed[i + 1] = true;
            continue;
        }

        if (agg->func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(displayName, accIndex, 0, nullptr);
            consumed[i] = true;
            continue;
        }

        if (!expressionSupported(agg->arg, displayName))
            return std::nullopt;
        std::string valueExpr = materializeExprToMetalWithCarryMap(
            agg->arg, idxVar, lowering->carryMap);

        if (agg->func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum",
                                                  valueExpr, "float");
            int countIndex = reduce->addAccumulator(accName + "_count",
                                                    "1.0f", "float");
            reduce->setAverageResultAlias(displayName, sumIndex, countIndex,
                                          0, nullptr);
            consumed[i] = true;
            continue;
        }

        std::string outputType =
            agg->arg->type.type == DataType::FLOAT ? "float" : "long";
        MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
        if (agg->func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
        else if (agg->func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
        else if (agg->func != AggFunc::SUM) {
            return fail(error, "IR multi-table scalar aggregate lowerer: unsupported aggregate '" +
                               aggFuncName(agg->func) + "'.");
        }
        if (op != MetalTGReduce::ReduceOp::SUM &&
            agg->arg->type.type != DataType::FLOAT) {
            outputType = "int";
        }
        int accIndex = reduce->addAccumulator(accName, valueExpr, outputType,
                                              "", "", op);
        reduce->setAccumulatorResultAlias(displayName, accIndex, 0, nullptr);
        consumed[i] = true;
    }

    auto& scalarPhase = appendPhase(lowering->plan, "GENERIC_ir_multi_table_scalar",
                                    std::move(reduce));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(scalarPhase, scalarLookups);
    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetalImpl(ir, &aq, error);
}

} // namespace codegen
