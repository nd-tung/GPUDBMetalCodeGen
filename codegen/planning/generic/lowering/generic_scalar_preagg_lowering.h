#pragma once

#include "generic/lowering/generic_scalar_lookup.h"
#include "metal_plan_builder.h"

#include <vector>

namespace codegen {

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
    const AnalyzedQuery& aq,
    MetalQueryPlan& plan);

} // namespace codegen
