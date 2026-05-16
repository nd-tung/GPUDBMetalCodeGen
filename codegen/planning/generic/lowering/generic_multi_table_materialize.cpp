#include "generic/lowering/generic_ir_physical_planner.h"

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
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

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableMaterializeShape(ir, error);
    if (!shape) return std::nullopt;

    auto* project = projectDetail(shape->project);
    if (!project || project->projections.empty())
        return fail(error, "IR multi-table materialize lowerer: no projection columns.");
    if (materializeHasEmptyInListPlaceholder(*shape))
        return std::nullopt;

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq) && materializeNeedsScalarPreAgg(*shape)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_MATERIALIZE_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table materialize lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& projection : project->projections)
        neededExprs.push_back(projection.expr);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys)
            neededExprs.push_back(key.expr);
    }

    std::string sharedLowerError;
    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "GENERIC_IR_MULTI_TABLE_MATERIALIZE", aq,
        scalarLookups.empty() ? nullptr : &scalarLookups, &sharedLowerError);
    if (!lowering) {
        return fail(error, sharedLowerError.empty()
            ? "IR multi-table materialize lowerer: unsupported join shape."
            : sharedLowerError);
    }

    prependPlanPhases(lowering->plan, scalarPreAggPlan);
    const std::string idxVar = "i";
    const std::string resultCounter = "d_ir_multi_table_result_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(lowering->probePipe), resultCounter, "1");
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t i = 0; i < project->projections.size(); ++i) {
        const auto& projection = project->projections[i];
        if (!materializeExprSupported(projection.expr))
            return std::nullopt;
        if (exprNeedsCarriedString(projection.expr, lowering->carryMap))
            return std::nullopt;
        int stringLen = materializedStringLenForExpr(projection.expr,
                                                     lowering->carryMap);
        std::string sizeExpr = lowering->outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        const std::string bufferName = "d_ir_multi_table_" +
            std::to_string(i) + "_" + sanitizeIdentifier(projection.name);
        const std::string metalType = metalTypeForType(projection.type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetalWithCarryMap(
                                   projection.expr, idxVar,
                                   lowering->carryMap),
                               projection.name, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            projection.name, bufferName, metalType, stringLen});
    }

    auto& matPhase = appendPhase(lowering->plan,
                                 "GENERIC_ir_multi_table_materialize",
                                 std::move(materialize));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(matPhase, scalarLookups);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayName(key, *project);
            if (!name)
                return fail(error, "IR multi-table materialize lowerer: ORDER BY key is not projected.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }
    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_multi_table_rows";
        attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
        if (!appendGenericGpuSort(lowering->plan, "ir_multi_table_materialize",
                                  rowsSym, lowering->outputSize,
                                  materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
    }
    return std::move(lowering->plan);
}

} // namespace

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableMaterializeIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableMaterializeIRToMetalImpl(ir, &aq, error);
}

} // namespace codegen
