#pragma once

#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

struct GenericRelPlan;

// Build a plan from Generic relational IR using only structural pattern planners.
// This route is for ad-hoc SQL and must not use query-name hand-tuned TPC-H
// builders. It is intentionally a supported-pattern API, not a full SQL
// planner.
std::optional<MetalQueryPlan> buildAdhocGenericPlan(const GenericRelPlan& ir,
                                                    const std::string& label = "",
                                                    std::string* error = nullptr);

} // namespace codegen
