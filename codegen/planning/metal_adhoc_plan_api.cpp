#include "metal_adhoc_plan_api.h"
#include "metal_generic_adhoc_builder.h"

namespace codegen {

std::optional<MetalQueryPlan> buildAdhocSQLPlan(const AnalyzedQuery& aq,
                                                const std::string& label,
                                                std::string* error) {
    std::string singleError, multiError;
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        if (auto p = buildGenericSingleTableAdhocPlan(aq, &singleError)) {
            if (getenv("GEN_DEBUG")) fprintf(stderr, "[DISPATCH] -> buildGenericSingleTableAdhocPlan\n");
            return p;
        }
        if (auto p = buildGenericMultiTableAdhocPlan(aq, &multiError)) {
            if (getenv("GEN_DEBUG")) fprintf(stderr, "[DISPATCH] -> buildGenericMultiTableAdhocPlan\n");
            return p;
        }
        if (getenv("GEN_DEBUG")) fprintf(stderr, "[DISPATCH] no builder matched. singleErr=%s multiErr=%s\n",
                                         singleError.c_str(), multiError.c_str());
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

    if (!label.empty()) plan->name = label;
    if (label.rfind("MB", 0) == 0) plan->chunkable = true;
    return plan;
}

} // namespace codegen