#pragma once

#include "metal_plan_builder.h"

#include <optional>

namespace codegen {

std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq);

} // namespace codegen