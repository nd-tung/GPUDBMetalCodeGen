#include "metal_adhoc_plan_api.h"
#include "generic_ir_builder.h"
#include "generic_ir_physical_planner.h"
#include "generic_ir_validator.h"
#include "metal_generic_adhoc_builder.h"

#include <sstream>
#include <vector>

namespace codegen {

namespace {

bool validateStrictGenericPlan(const MetalQueryPlan& plan, std::string* error) {
    std::vector<std::string> fallbacks;
    if (plan.cpuSort) fallbacks.push_back("cpuSort");
    if (plan.cpuGroupBy) fallbacks.push_back("cpuGroupBy");
    if (plan.cpuScalarAgg) fallbacks.push_back("cpuScalarAgg");
    if (fallbacks.empty()) return true;

    if (error) {
        std::ostringstream oss;
        oss << "Strict generic SQL plan contains CPU relational fallback";
        if (fallbacks.size() > 1) oss << "s";
        oss << ": ";
        for (size_t i = 0; i < fallbacks.size(); ++i) {
            if (i) oss << ", ";
            oss << fallbacks[i];
        }
        oss << ". This SQL shape must be implemented with GPU generic operators or rejected as unsupported.";
        *error = oss.str();
    }
    return false;
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
    std::optional<GenericRelPlan> multiTableMaterializeIr;
    if (!aq.isSingleTable() && aq.tables.size() >= 2 &&
        !aq.hasAggregation() && !aq.hasGroupBy() &&
        aq.fromSubqueryAggs.empty() &&
        hasOnlyScalarSubqueries(aq) &&
        aq.inSubAggs.empty()) {
        std::string irError;
        multiTableMaterializeIr = buildValidatedGenericIr(aq, &irError);
    }
    std::optional<GenericRelPlan> multiTableAggregateIr;
    if (!aq.isSingleTable() && aq.tables.size() >= 2 &&
        (aq.hasAggregation() || aq.hasGroupBy()) &&
        aq.fromSubqueryAggs.empty()) {
        std::string irError;
        multiTableAggregateIr = buildValidatedGenericIr(aq, &irError);
    }

    std::string singleError, multiError;
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        if (singleTableIr) {
            std::string irLowerError;
            if (auto p = lowerSingleTableGroupedAggregateIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
            if (auto p = lowerSingleTableScalarAggregateIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
            if (auto p = lowerSingleTableMaterializeIRToMetal(*singleTableIr, &irLowerError)) {
                return p;
            }
        }
        if (multiTableMaterializeIr) {
            std::string irLowerError;
            if (auto p = lowerMultiTableMaterializeIRToMetal(*multiTableMaterializeIr,
                                                             aq,
                                                             &irLowerError)) {
                return p;
            }
        }
        if (multiTableAggregateIr) {
            std::string irLowerError;
            if (auto p = lowerMultiTableGroupedAggregateIRToMetal(*multiTableAggregateIr,
                                                                  aq,
                                                                  &irLowerError)) {
                return p;
            }
            if (auto p = lowerMultiTableScalarAggregateIRToMetal(*multiTableAggregateIr,
                                                                 aq,
                                                                 &irLowerError)) {
                return p;
            }
        }
        if (!aq.fromSubqueryAggs.empty()) {
            std::string irLowerError;
            if (auto p = lowerFromSubqueryAggregateIRToMetal(aq, &irLowerError)) {
                return p;
            }
        }
        if (auto p = buildGenericSingleTableAdhocPlan(aq, &singleError)) {
            return p;
        }
        if (auto p = buildGenericMultiTableAdhocPlan(aq, &multiError)) {
            return p;
        }
        return std::nullopt;
    };

    auto plan = dispatch();
    if (!plan && error) {
        if (!multiError.empty())
            *error = multiError;
        else if (!singleError.empty())
            *error = singleError;
        else
            *error = "Ad-hoc SQL: query does not match any supported pattern.";
    }
    if (!plan) return plan;

    if (!validateStrictGenericPlan(*plan, error))
        return std::nullopt;

    if (!label.empty()) plan->name = label;
    if (label.rfind("MB", 0) == 0) plan->chunkable = true;
    return plan;
}

} // namespace codegen
