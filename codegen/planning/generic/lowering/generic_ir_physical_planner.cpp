#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "metal_plan_common.h"

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

} // namespace

std::optional<MetalQueryPlan> lowerSingleTableHashGroupedAggregateIRToMetal(
        const GenericScanDetail& scan,
        const GenericAggregateDetail& aggregate,
        const GenericRelNode* filterNode,
        const GenericRelNode* sortNode,
        const GenericRelNode* limitNode,
        std::string* error) {
    const std::string idxVar = "i";
    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan.table, idxVar);
    if (auto* filter = filterDetail(filterNode)) {
        if (!predicateSupported(filter->predicate))
            return fail(error, "IR single-table hash group lowerer: filter predicate unsupported.");
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }

    const std::string resultCounter = "d_ir_single_hash_group_input_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(pipe), resultCounter, "1");
    const std::string outputSize = tableSizeName(scan.table);

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
                *error = "IR single-table hash group lowerer: input expression '" +
                         displayName + "' is not supported.";
            return false;
        }
        int stringLen = fixedStringLenForExpr(expr);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_single_hash_group_" +
                                 std::to_string(matColIdx++) + "_" +
                                 sanitizeIdentifier(displayName);
        std::string metalType = metalTypeForType(type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetal(expr, idxVar),
                               displayName, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            displayName, bufferName, metalType, stringLen, scaleDown, false,
            distinctDomainSymbol});
        return true;
    };

    if (!buildAggregateInputGroupSpec(
            aggregate, "IR single-table hash group lowerer", groupSpec,
            groupKeys, addInputColumn, error)) {
        return std::nullopt;
    }
    if (!configureAggregateHaving(aggregate, groupSpec, nullptr, nullptr, error))
        return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SINGLE_TABLE_HASH_GROUP";
    auto& matPhase = appendPhase(plan, "GENERIC_ir_single_table_hash_group_materialize",
                                 std::move(materialize));

    const std::string groupTag = "ir_single_table_hash_group";
    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = "next_pow2(" + outputSize + " * 2)";
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.maxOutputRowsExpr = outputSize;
    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
    gbSpec.inputColumns = std::move(materializedCols);
    gbSpec.groupBy = std::move(groupSpec);
    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
    appendGenericGpuGroupBy(plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(limitNode);
    if (auto* sort = sortDetail(sortNode)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, aggregate, groupKeys);
            if (!name)
                return fail(error, "IR single-table hash group lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        if (!appendBestGenericGpuOrder(plan, "group_" + groupTag,
                                       sortRowsSym, gbSpec.maxOutputRowsExpr,
                                       genericGpuGroupOutputColumns(gbSpec),
                                       sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

} // namespace codegen
