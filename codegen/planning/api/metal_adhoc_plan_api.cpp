#include "metal_adhoc_plan_api.h"
#include "generic/ir/generic_ir_builder.h"
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

std::optional<GenericRelPlan> buildValidatedGenericIr(const AnalyzedQuery& aq,
                                                      std::string* error) {
    std::string buildError;
    auto ir = buildGenericRelationalIR(aq, &buildError);
    if (!ir) {
        if (error) *error = "Generic relational IR preflight failed: " + buildError;
        return std::nullopt;
    }

    auto validation = validateGenericRelationalIR(*ir);
    if (!validation.ok()) {
        if (error)
            *error = "Generic relational IR preflight failed: " + validation.message();
        return std::nullopt;
    }
    return ir;
}

std::optional<GenericRelPlan> buildValidatedSingleTableIr(const AnalyzedQuery& aq,
                                                          std::string* error) {
    if (!aq.isSingleTable()) return std::nullopt;
    return buildValidatedGenericIr(aq, error);
}

bool hasOnlyScalarSubqueries(const AnalyzedQuery& aq) {
    for (const auto& sq : aq.subqueries) {
        if (sq.type != AnalyzedQuery::Subquery::SCALAR_SUBQUERY)
            return false;
    }
    return true;
}

} // namespace

std::optional<MetalQueryPlan> buildAdhocSQLPlan(const AnalyzedQuery& aq,
                                                const std::string& label,
                                                std::string* error) {
    std::optional<GenericRelPlan> singleTableIr;
    if (aq.isSingleTable()) {
        singleTableIr = buildValidatedSingleTableIr(aq, error);
        if (!singleTableIr) return std::nullopt;
    }

    std::string planningErrors;
    std::optional<GenericRelPlan> multiTableMaterializeIr;
    if (!aq.isSingleTable() && aq.tables.size() >= 2 &&
        !aq.hasAggregation() && !aq.hasGroupBy() &&
        aq.fromSubqueryAggs.empty() &&
        hasOnlyScalarSubqueries(aq) &&
        aq.inSubAggs.empty()) {
        std::string irError;
        multiTableMaterializeIr = buildValidatedGenericIr(aq, &irError);
        if (!multiTableMaterializeIr)
            appendPlannerReason(planningErrors, "multi-table materialize IR", irError);
    }
    std::optional<GenericRelPlan> multiTableAggregateIr;
    if (!aq.isSingleTable() && aq.tables.size() >= 2 &&
        (aq.hasAggregation() || aq.hasGroupBy()) &&
        aq.fromSubqueryAggs.empty()) {
        std::string irError;
        multiTableAggregateIr = buildValidatedGenericIr(aq, &irError);
        if (!multiTableAggregateIr)
            appendPlannerReason(planningErrors, "multi-table aggregate IR", irError);
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
                                                             aq,
                                                             &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table materialize lowerer", irLowerError);
        }
        if (multiTableAggregateIr) {
            std::string irLowerError;
            if (auto p = lowerMultiTableGroupedAggregateIRToMetal(*multiTableAggregateIr,
                                                                  aq,
                                                                  &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table grouped aggregate lowerer", irLowerError);
            irLowerError.clear();
            if (auto p = lowerMultiTableScalarAggregateIRToMetal(*multiTableAggregateIr,
                                                                 aq,
                                                                 &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "multi-table scalar aggregate lowerer", irLowerError);
        }
        if (!aq.fromSubqueryAggs.empty()) {
            std::string irLowerError;
            if (auto p = lowerFromSubqueryAggregateIRToMetal(aq, &irLowerError)) {
                return p;
            }
            appendPlannerReason(lowerErrors, "FROM-subquery aggregate lowerer", irLowerError);
        }
        return std::nullopt;
    };

    auto plan = dispatch();
    if (!plan && error) {
        std::string detail = !lowerErrors.empty() ? lowerErrors : planningErrors;
        *error = "Strict generic SQL: query is not implemented by Generic IR GPU lowerers";
        if (!detail.empty()) *error += " (" + detail + ")";
        *error += ". Legacy generic fallback builders are disabled on the ad-hoc SQL route.";
    }
    if (!plan) return plan;

    if (!label.empty()) plan->name = label;
    if (label.rfind("MB", 0) == 0) plan->chunkable = true;
    return plan;
}

} // namespace codegen
