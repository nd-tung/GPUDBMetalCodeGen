#pragma once

#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

// Build a plan from analyzed SQL using only structural pattern planners.
// This route is for ad-hoc SQL and must not use query-name hand-tuned TPC-H
// builders. It is intentionally a supported-pattern API, not a full SQL
// planner.
std::optional<MetalQueryPlan> buildAdhocSQLPlan(const AnalyzedQuery& aq,
                                                const std::string& label = "");

} // namespace codegen