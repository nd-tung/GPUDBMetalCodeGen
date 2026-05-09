#include "metal_adhoc_plan_api.h"
#include "metal_generic_adhoc_builder.h"
#include "metal_tpch_query_builders.h"

namespace codegen {

std::optional<MetalQueryPlan> buildAdhocSQLPlan(const AnalyzedQuery& aq,
                                                const std::string& label) {
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        if (auto p = buildQ6Plan(aq)) return p;
        if (auto p = buildQ1Plan(aq)) return p;
        if (auto p = buildQ14Plan(aq)) return p;
        if (auto p = buildQ4Plan(aq)) return p;
        if (auto p = buildQ12Plan(aq)) return p;
        if (auto p = buildQ10Plan(aq)) return p;
        if (auto p = buildQ7Plan(aq)) return p;
        if (auto p = buildGenericSingleTableAdhocPlan(aq)) return p;
        return std::nullopt;
    };

    auto plan = dispatch();
    if (!plan) return plan;

    if (!label.empty()) plan->name = label;
    if (label.rfind("MB", 0) == 0) plan->chunkable = true;
    return plan;
}

} // namespace codegen