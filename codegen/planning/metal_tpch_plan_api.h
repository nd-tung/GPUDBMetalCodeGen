#pragma once

#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

bool isPredefinedTPCHQueryName(const std::string& queryName);

// Build a predefined TPC-H plan by query name. q1..q22 do not go through SQL
// parsing/planning, and may use query-name hand-tuned builders. The ad-hoc SQL
// API owns analyzer-based planning for SQL text.
std::optional<MetalQueryPlan> buildPredefinedTPCHPlan(const std::string& queryName);

} // namespace codegen