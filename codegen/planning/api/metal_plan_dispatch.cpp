#include "metal_plan_builder.h"
#include "metal_adhoc_plan_api.h"
#include "metal_tpch_plan_api.h"

namespace codegen {

std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName) {
    if (isPredefinedTPCHQueryName(queryName)) {
        return buildPredefinedTPCHPlan(queryName);
    }
    return buildAdhocSQLPlan(aq, queryName);
}

} // namespace codegen