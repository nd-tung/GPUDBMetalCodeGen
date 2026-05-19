#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/lowering/generic_cost_model.h"
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

std::optional<MetalQueryPlan> lowerSingleTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto finish = [&](MetalQueryPlan&& plan) -> std::optional<MetalQueryPlan> {
        return attachGenericCostTrace(
            std::optional<MetalQueryPlan>{std::move(plan)}, ir,
            "single_table_materialize");
    };

    auto shape = parseSingleTableMaterializeShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* project = projectDetail(shape->project);
    if (!scan || !project)
        return fail(error, "IR materialize lowerer: malformed scan/project detail.");
    if (project->projections.empty())
        return fail(error, "IR materialize lowerer: no projection columns.");

    const std::string idxVar = "i";
    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan->table, idxVar);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate)) {
            return fail(error, "IR materialize lowerer: filter predicate is not supported.");
        }
        std::string predicate = genericPredicateToMetal(filter->predicate, idxVar);
        pipe = maybeSelect(std::move(pipe), predicate);
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SINGLE_TABLE_MATERIALIZE";

    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(pipe), "d_generic_result_count", "1");

    const std::string outputSize = tableSizeName(scan->table);
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t i = 0; i < project->projections.size(); ++i) {
        const auto& projection = project->projections[i];
        if (!materializeExprSupported(projection.expr)) {
            return fail(error, "IR materialize lowerer: projection '" +
                               projection.name + "' is not supported.");
        }

        int stringLen = fixedStringLenForExpr(projection.expr);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_generic_" + std::to_string(i) + "_" +
                                 sanitizeIdentifier(projection.name);
        std::string metalType = metalTypeForType(projection.type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetal(projection.expr, idxVar),
                               projection.name, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            projection.name, bufferName, metalType, stringLen});
    }

    auto& matPhase = appendPhase(plan, "GENERIC_ir_single_table_materialize",
                                 std::move(materialize));

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayName(key, *project);
            if (!name) {
                return fail(error, "IR materialize lowerer: ORDER BY key is not projected.");
            }
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_single_rows";
        attachMaterializedCountHook(matPhase, "d_generic_result_count", rowsSym);
        if (!appendGenericGpuSort(plan, "ir_single_materialize", rowsSym,
                                  outputSize, materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
    }

    return finish(std::move(plan));
}

} // namespace codegen
