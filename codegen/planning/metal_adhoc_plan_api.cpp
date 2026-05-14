#include "metal_adhoc_plan_api.h"
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

} // namespace

std::optional<MetalQueryPlan> buildAdhocSQLPlan(const AnalyzedQuery& aq,
                                                const std::string& label,
                                                std::string* error) {
    std::string singleError, multiError;
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
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
