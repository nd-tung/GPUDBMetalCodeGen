#include "generic/lowering/generic_ir_physical_planner.h"

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_cost_model.h"
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
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
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
    if (!prefix.costTraces.empty()) {
        target.costTraces.insert(
            target.costTraces.begin(),
            std::make_move_iterator(prefix.costTraces.begin()),
            std::make_move_iterator(prefix.costTraces.end()));
    }
}

const IrCarryColumn* findCarry(const IrCarryMap& carries,
                               const GenericColumnExpr& col) {
    auto relIt = carries.find(col.relationInstance.value);
    if (relIt == carries.end()) return nullptr;
    auto colIt = relIt->second.find(col.column);
    if (colIt == relIt->second.end()) return nullptr;
    return &colIt->second;
}

void collectExprColumns(const GenericExprPtr& expr,
                        std::vector<GenericColumnExpr>& out,
                        bool& hasScalarLookup) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            out.push_back(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectExprColumns(node.left, out, hasScalarLookup);
            collectExprColumns(node.right, out, hasScalarLookup);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectExprColumns(branch.result, out, hasScalarLookup);
            }
            collectExprColumns(node.elseResult, out, hasScalarLookup);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectExprColumns(arg, out, hasScalarLookup);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectExprColumns(node.arg, out, hasScalarLookup);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            hasScalarLookup = true;
        }
    }, expr->node);
}

size_t metalColumnWidth(const std::string& metalType, int stringLen) {
    if (stringLen > 0) return static_cast<size_t>(stringLen);
    if (metalType == "char") return 1;
    if (metalType == "long" || metalType == "ulong") return 8;
    return 4;
}

size_t projectionWidth(const GenericProjection& projection,
                       const IrCarryMap& carries) {
    return metalColumnWidth(metalTypeForType(projection.type),
                            materializedStringLenForExpr(projection.expr,
                                                         carries));
}

bool canReloadColumnsAfterTopK(
        const GenericProjectDetail& project,
        const GenericScanDetail& probeScan,
        const IrCarryMap& carries) {
    bool hasScalarLookup = false;
    std::vector<GenericColumnExpr> columns;
    for (const auto& projection : project.projections) {
        collectExprColumns(projection.expr, columns, hasScalarLookup);
    }
    if (hasScalarLookup) return false;
    for (const auto& col : columns) {
        if (col.relationInstance.value == probeScan.relationInstance.value)
            continue;
        const auto* carry = findCarry(carries, col);
        if (!carry || carry->bufferName.empty() ||
            carry->lookupKeyColumn.empty() ||
            carry->lookupKeyMetalType.empty()) {
            return false;
        }
    }
    return true;
}

bool shouldUseLateMaterialization(const GenericProjectDetail& project,
                                  const IrCarryMap& carries,
                                  const std::vector<size_t>& sortProjectionIdxs,
                                  int limit) {
    if (limit <= 0 || sortProjectionIdxs.empty() || limit > 4096)
        return false;

    size_t fullWidth = 0;
    for (const auto& projection : project.projections)
        fullWidth += projectionWidth(projection, carries);

    size_t keyWidth = sizeof(uint32_t);
    std::set<size_t> seen;
    for (size_t idx : sortProjectionIdxs) {
        if (idx >= project.projections.size() || !seen.insert(idx).second)
            continue;
        keyWidth += projectionWidth(project.projections[idx], carries);
    }

    if (fullWidth <= keyWidth) return false;
    const size_t savedPerCandidate = fullWidth - keyWidth;
    const size_t gatherBytes = fullWidth * static_cast<size_t>(limit);
    return savedPerCandidate >= 32 && gatherBytes <= fullWidth * 4096;
}

struct LateProjection {
    std::string displayName;
    std::string bufferName;
    std::string metalType;
    GenericExprPtr expr;
    int stringLen = 0;
};

class MetalLateMaterializeGather : public MetalOperator {
public:
    MetalLateMaterializeGather(std::string sortedIndexBuffer,
                               std::string rowIdBuffer,
                               std::string nRowsSymbol,
                               std::string rowIdCapacityExpr,
                               std::string outputCounter,
                               std::string probeTable,
                               GenericRelationInstanceId probeRelation,
                               IrCarryMap carries,
                               std::vector<LateProjection> projections,
                               int limit)
        : sortedIndexBuffer_(std::move(sortedIndexBuffer)),
          rowIdBuffer_(std::move(rowIdBuffer)),
          nRowsSymbol_(std::move(nRowsSymbol)),
          rowIdCapacityExpr_(std::move(rowIdCapacityExpr)),
          outputCounter_(std::move(outputCounter)),
          probeTable_(std::move(probeTable)),
          probeRelation_(probeRelation),
          carries_(std::move(carries)),
          projections_(std::move(projections)),
          limit_(limit) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string suffix = sanitizeIdentifier(outputCounter_);
        const std::string limitParam = "n_late_limit_" + suffix;
        const std::string probeRow = "_late_probe_row";

        cg.addScalarParam(nRowsSymbol_, "uint");
        cg.addResolvedScalarParam(limitParam, "uint", std::to_string(limit_));
        cg.addBufferParam(sortedIndexBuffer_, "int", "", false);
        cg.addBufferParam(rowIdBuffer_, "uint", rowIdCapacityExpr_, false);
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");

        registerInputColumns(cg);
        registerCarryInputs(cg);

        for (const auto& projection : projections_) {
            std::string sizeExpr = std::to_string(limit_);
            if (projection.stringLen > 0)
                sizeExpr += " * " + std::to_string(projection.stringLen);
            cg.addBufferParam(projection.bufferName, projection.metalType,
                              sizeExpr, false);
        }

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& projection : projections_) {
            cg.registerOutputColumn(projection.displayName,
                                    projection.bufferName,
                                    projection.metalType,
                                    projection.stringLen);
        }

        cg.addIf("tid == 0", [&]() {
            cg.addLine("uint _late_count = min(" + nRowsSymbol_ + ", " +
                       limitParam + ");");
            cg.addLine("atomic_store_explicit(&" + outputCounter_ +
                       "[0], _late_count, memory_order_relaxed);");
        });
        cg.addBlock("for (uint _rank = tid; _rank < " + limitParam +
                    " && _rank < " + nRowsSymbol_ + "; _rank += tpg)", [&]() {
            cg.addLine("int _candidate_pos_i = " + sortedIndexBuffer_ +
                       "[_rank];");
            cg.addIf("_candidate_pos_i >= 0 && (uint)_candidate_pos_i < " +
                     nRowsSymbol_, [&]() {
                cg.addLine("uint _candidate_pos = (uint)_candidate_pos_i;");
                cg.addLine("uint " + probeRow + " = " + rowIdBuffer_ +
                           "[_candidate_pos];");
                cg.addLine("bool _late_valid = true;");
                emitCarryLoads(cg, probeRow);
                cg.addIf("_late_valid", [&]() {
                    emitProjectionWrites(cg, probeRow);
                });
            });
        });
    }

    std::string describe() const override {
        return "LateMaterializeGather(" +
               std::to_string(projections_.size()) + " columns)";
    }

private:
    std::string sortedIndexBuffer_;
    std::string rowIdBuffer_;
    std::string nRowsSymbol_;
    std::string rowIdCapacityExpr_;
    std::string outputCounter_;
    std::string probeTable_;
    GenericRelationInstanceId probeRelation_;
    IrCarryMap carries_;
    std::vector<LateProjection> projections_;
    int limit_ = 0;

    void collectProjectionColumns(std::vector<GenericColumnExpr>& columns) const {
        bool hasScalarLookup = false;
        for (const auto& projection : projections_)
            collectExprColumns(projection.expr, columns, hasScalarLookup);
    }

    void registerInputColumns(MetalCodegen& cg) const {
        std::vector<GenericColumnExpr> columns;
        collectProjectionColumns(columns);
        std::set<std::string> seen;
        for (const auto& col : columns) {
            if (col.relationInstance.value != probeRelation_.value)
                continue;
            if (!seen.insert(col.column).second) continue;
            cg.addColumnParam(col.column, metalTypeForType(col.type),
                              col.table.empty() ? probeTable_ : col.table);
        }
    }

    void registerCarryInputs(MetalCodegen& cg) const {
        std::set<std::string> seenBuffers;
        std::set<std::string> seenProbeKeys;
        std::set<std::string> seenSourceColumns;
        forEachUsedCarry([&](const IrCarryColumn& carry) {
            if (seenProbeKeys.insert(carry.lookupKeyColumn).second) {
                cg.addColumnParam(carry.lookupKeyColumn,
                                  carry.lookupKeyMetalType,
                                  probeTable_);
            }
            if (seenBuffers.insert(carry.bufferName).second) {
                if (carry.column.type.type == DataType::CHAR_FIXED) {
                    cg.addBufferParam(carry.bufferName, "uint", "", false);
                } else {
                    cg.addBufferParam(carry.bufferName,
                                      metalTypeForType(carry.column.type),
                                      "", false);
                }
            }
            if (carry.column.type.type == DataType::CHAR_FIXED &&
                seenSourceColumns.insert(carry.column.table + "." +
                                         carry.column.column).second) {
                cg.addColumnParam(carry.column.column, "char",
                                  carry.column.table);
            }
        });
    }

    template <typename Fn>
    void forEachUsedCarry(Fn fn) const {
        std::vector<GenericColumnExpr> columns;
        collectProjectionColumns(columns);
        std::set<std::string> seen;
        for (const auto& col : columns) {
            const auto* carry = findCarry(carries_, col);
            if (!carry) continue;
            std::string key = std::to_string(col.relationInstance.value) +
                              ":" + col.column;
            if (!seen.insert(key).second) continue;
            fn(*carry);
        }
    }

    void emitCarryLoads(MetalCodegen& cg,
                        const std::string& probeRow) const {
        forEachUsedCarry([&](const IrCarryColumn& carry) {
            const std::string suffix = sanitizeIdentifier(carry.varName);
            const std::string keyVar = "_late_key_" + suffix;
            cg.addLine("uint " + keyVar + " = (uint)(" +
                       carry.lookupKeyColumn + "[" + probeRow + "]);");
            if (carry.column.type.type == DataType::CHAR_FIXED) {
                int width = carry.column.type.fixedWidth > 0
                    ? carry.column.type.fixedWidth
                    : 1;
                cg.addLine("uint " + carry.rowVarName + " = " +
                           carry.bufferName + "[" + keyVar + "];");
                cg.addLine("_late_valid = _late_valid && " +
                           carry.rowVarName + " != 0xFFFFFFFFu;");
                cg.addLine("const device char* " + carry.varName + " = " +
                           carry.column.column + " + " + carry.rowVarName +
                           " * " + std::to_string(width) + "u;");
            } else {
                cg.addLine(metalTypeForType(carry.column.type) + " " +
                           carry.varName + " = " + carry.bufferName +
                           "[" + keyVar + "];");
            }
        });
    }

    void emitProjectionWrites(MetalCodegen& cg,
                              const std::string& probeRow) const {
        for (const auto& projection : projections_) {
            std::string valueExpr = materializeExprToMetalWithCarryMap(
                projection.expr, probeRow, carries_);
            if (projection.stringLen > 0) {
                cg.addBlock("for (uint _ci = 0; _ci < " +
                            std::to_string(projection.stringLen) +
                            "u; ++_ci)", [&]() {
                    cg.addLine(projection.bufferName + "[(ulong)_rank * " +
                               std::to_string(projection.stringLen) +
                               "ul + _ci] = (" + valueExpr + ")[_ci];");
                });
            } else {
                cg.addLine(projection.bufferName + "[_rank] = " +
                           valueExpr + ";");
            }
        }
    }
};

std::vector<LateProjection> makeLateProjections(
        const GenericProjectDetail& project,
        const IrCarryMap& carries) {
    std::vector<LateProjection> out;
    out.reserve(project.projections.size());
    for (size_t i = 0; i < project.projections.size(); ++i) {
        const auto& projection = project.projections[i];
        const std::string metalType = metalTypeForType(projection.type);
        out.push_back(LateProjection{
            projection.name,
            "d_ir_multi_table_late_" + std::to_string(i) + "_" +
                sanitizeIdentifier(projection.name),
            metalType,
            projection.expr,
            materializedStringLenForExpr(projection.expr, carries),
        });
    }
    return out;
}

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetalImpl(
        const GenericRelPlan& ir,
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
    if (hasScalarSubqueries(ir) && materializeNeedsScalarPreAgg(*shape)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_MATERIALIZE_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(ir, scalarPreAggPlan);
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
        "GENERIC_IR_MULTI_TABLE_MATERIALIZE",
        scalarLookups.empty() ? nullptr : &scalarLookups, &sharedLowerError);
    if (!lowering) {
        return fail(error, sharedLowerError.empty()
            ? "IR multi-table materialize lowerer: unsupported join shape."
            : sharedLowerError);
    }

    prependPlanPhases(lowering->plan, scalarPreAggPlan);

    for (const auto& projection : project->projections) {
        if (!materializeExprSupported(projection.expr))
            return std::nullopt;
        if (exprNeedsCarriedString(projection.expr, lowering->carryMap))
            return std::nullopt;
    }

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    std::vector<size_t> sortProjectionIdxs;
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayName(key, *project);
            if (!name)
                return fail(error, "IR multi-table materialize lowerer: ORDER BY key is not projected.");
            sortSpec.keys.push_back({*name, key.descending});
            auto it = std::find_if(
                project->projections.begin(), project->projections.end(),
                [&](const GenericProjection& projection) {
                    return projection.name == *name;
                });
            if (it != project->projections.end()) {
                sortProjectionIdxs.push_back(static_cast<size_t>(
                    std::distance(project->projections.begin(), it)));
            }
        }
    }

    const bool useLateMaterialization =
        !sortSpec.keys.empty() &&
        sortSpec.limit > 0 &&
        lowering->probeScan &&
        canReloadColumnsAfterTopK(*project, *lowering->probeScan,
                                  lowering->carryMap) &&
        shouldUseLateMaterialization(*project, lowering->carryMap,
                                     sortProjectionIdxs, sortSpec.limit);

    const std::string idxVar = "i";
    const std::string resultCounter = "d_ir_multi_table_result_count";

    if (useLateMaterialization) {
        auto keyMaterialize = std::make_unique<MetalMaterialize>(
            std::move(lowering->probePipe), resultCounter, "1");
        std::vector<GenericMatColumnDesc> keyCols;
        std::set<size_t> emittedKeys;
        for (size_t idx : sortProjectionIdxs) {
            if (idx >= project->projections.size() ||
                !emittedKeys.insert(idx).second) {
                continue;
            }
            const auto& projection = project->projections[idx];
            const int stringLen = materializedStringLenForExpr(
                projection.expr, lowering->carryMap);
            std::string sizeExpr = lowering->outputSize;
            if (stringLen > 0)
                sizeExpr += " * " + std::to_string(stringLen);
            const std::string bufferName = "d_ir_multi_table_key_" +
                std::to_string(idx) + "_" + sanitizeIdentifier(projection.name);
            const std::string metalType = metalTypeForType(projection.type);
            keyMaterialize->addColumn(
                bufferName, metalType,
                materializeExprToMetalWithCarryMap(
                    projection.expr, idxVar, lowering->carryMap),
                projection.name, sizeExpr, stringLen);
            keyCols.push_back(GenericMatColumnDesc{
                projection.name, bufferName, metalType, stringLen});
        }

        const std::string rowIdBuffer = "d_ir_multi_table_late_rowid";
        keyMaterialize->addColumn(rowIdBuffer, "uint", "(uint)" + idxVar,
                                  "__rowid", lowering->outputSize, 0);

        auto& keyPhase = appendPhase(lowering->plan,
                                     "GENERIC_ir_multi_table_key_materialize",
                                     std::move(keyMaterialize));
        if (!scalarLookups.empty())
            attachGenericScalarLookupBuffers(keyPhase, scalarLookups);

        const std::string rowsSym = "n_gpu_sort_ir_multi_table_rows";
        attachMaterializedCountHook(keyPhase, resultCounter, rowsSym);
        if (!appendBestGenericGpuOrder(
                lowering->plan, "ir_multi_table_late_keys",
                rowsSym, lowering->outputSize, keyCols, sortSpec, error)) {
            return std::nullopt;
        }

        if (!lowering->plan.gpuSort) {
            return fail(error, "IR multi-table materialize lowerer: late materialization requires GPU order metadata.");
        }
        auto sortInfo = *lowering->plan.gpuSort;
        auto gather = std::make_unique<MetalLateMaterializeGather>(
            sortInfo.sortedIndexBuffer, rowIdBuffer, rowsSym,
            lowering->outputSize,
            "d_ir_multi_table_late_result_count",
            lowering->probeScan->table,
            lowering->probeScan->relationInstance,
            lowering->carryMap,
            makeLateProjections(*project, lowering->carryMap),
            sortSpec.limit);
        appendPhase(lowering->plan,
                    "GENERIC_ir_multi_table_late_materialize",
                    std::move(gather));
        lowering->plan.gpuSort.reset();
        return std::move(lowering->plan);
    }

    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(lowering->probePipe), resultCounter, "1");
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t i = 0; i < project->projections.size(); ++i) {
        const auto& projection = project->projections[i];
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

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_multi_table_rows";
        attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
        if (!appendBestGenericGpuOrder(
                lowering->plan, "ir_multi_table_materialize",
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
    return attachGenericCostTrace(
        lowerMultiTableMaterializeIRToMetalImpl(ir, error),
        ir, "multi_table_materialize");
}

} // namespace codegen
