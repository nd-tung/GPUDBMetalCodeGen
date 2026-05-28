#include "metal_adhoc_plan_api.h"
#include "generic/ir/generic_relational_ir.h"
#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/ir/generic_ir_validator.h"

namespace codegen {

namespace {

void appendPlannerReason(std::string& out,
                         const std::string& stage,
                         const std::string& reason) {
    if (reason.empty()) return;
    if (!out.empty()) out += "; ";
    out += stage + ": " + reason;
}

bool validateGenericIrForPlanning(const GenericRelPlan& ir, std::string* error) {
    auto validation = validateGenericRelationalIR(ir);
    if (!validation.ok()) {
        if (error)
            *error = "Generic relational IR preflight failed: " + validation.message();
        return false;
    }
    return true;
}

size_t scanCount(const GenericRelPlan& ir) {
    size_t count = 0;
    for (const auto& node : ir.nodes) {
        if (node.op == GenericRelOp::Scan) ++count;
    }
    return count;
}

bool hasJoinNodes(const GenericRelPlan& ir) {
    for (const auto& node : ir.nodes) {
        if (node.op == GenericRelOp::Join ||
            node.op == GenericRelOp::SemiJoin ||
            node.op == GenericRelOp::AntiJoin) {
            return true;
        }
    }
    return false;
}

bool isSingleTablePlan(const GenericRelPlan& ir) {
    return scanCount(ir) == 1 && !hasJoinNodes(ir);
}

bool hasAggregateNode(const GenericRelPlan& ir) {
    for (const auto& node : ir.nodes) {
        if (node.op == GenericRelOp::Aggregate) return true;
    }
    return false;
}

bool hasOnlyScalarSubqueries(const GenericRelPlan& ir) {
    for (const auto& sq : ir.source.subqueries) {
        if (sq.type != GenericSourceSubquery::SCALAR_SUBQUERY)
            return false;
    }
    return true;
}

} // namespace

std::optional<MetalQueryPlan> buildAdhocGenericPlan(const GenericRelPlan& ir,
                                                    const std::string& label,
                                                    std::string* error) {
    if (!validateGenericIrForPlanning(ir, error))
        return std::nullopt;

    const bool singleTable = isSingleTablePlan(ir);
    const size_t scans = scanCount(ir);
    const bool hasAggregation = hasAggregateNode(ir);

    const GenericRelPlan* singleTableIr = nullptr;
    if (singleTable) {
        singleTableIr = &ir;
    }

    const GenericRelPlan* multiTableMaterializeIr = nullptr;
    if (!singleTable && scans >= 2 &&
        !hasAggregation &&
        ir.source.fromSubqueryAggs.empty() &&
        hasOnlyScalarSubqueries(ir) &&
        ir.source.inSubAggs.empty()) {
        multiTableMaterializeIr = &ir;
    }
    const GenericRelPlan* multiTableAggregateIr = nullptr;
    if (!singleTable && scans >= 2 &&
        hasAggregation &&
        ir.source.fromSubqueryAggs.empty()) {
        multiTableAggregateIr = &ir;
    }
    const GenericRelPlan* fromSubqueryIr = nullptr;
    if (!ir.source.fromSubqueryAggs.empty()) {
        fromSubqueryIr = &ir;
    }

    std::string lowerErrors;
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        if (singleTableIr) {
            std::string irLowerError;
            if (auto p = lowerSingleTableGroupedAggregateIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "single-table grouped aggregate lowerer", irLowerError);
            irLowerError.clear();
            if (auto p = lowerSingleTableScalarAggregateIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "single-table scalar aggregate lowerer", irLowerError);
            irLowerError.clear();
            if (auto p = lowerSingleTableMaterializeIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "single-table materialize lowerer", irLowerError);
        }
        if (multiTableMaterializeIr) {
            std::string irLowerError;
            if (auto p = lowerMultiTableMaterializeIRToMetal(*multiTableMaterializeIr,
                                                             &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table materialize lowerer", irLowerError);
        }
        if (multiTableAggregateIr) {
            std::string irLowerError;
            if (auto p = lowerMultiTableGroupedAggregateIRToMetal(*multiTableAggregateIr,
                                                                  &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table grouped aggregate lowerer", irLowerError);
            irLowerError.clear();
            if (auto p = lowerMultiTableScalarAggregateIRToMetal(*multiTableAggregateIr,
                                                                 &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table scalar aggregate lowerer", irLowerError);
        }
        if (fromSubqueryIr) {
            std::string irLowerError;
            if (auto p = lowerFromSubqueryAggregateIRToMetal(*fromSubqueryIr, &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "FROM-subquery aggregate lowerer", irLowerError);
        }
        return std::nullopt;
    };

    auto plan = dispatch();
    if (!plan && error) {
        *error = "Generic SQL route: query is not implemented by Generic IR GPU lowerers";
        if (!lowerErrors.empty()) *error += " (" + lowerErrors + ")";
        *error += ". The ad-hoc SQL route does not use fallback query builders.";
    }
    if (!plan) return plan;

    if (!label.empty()) plan->name = label;
    if (label.rfind("MB", 0) == 0) plan->chunkable = true;
    return plan;
}

} // namespace codegen
