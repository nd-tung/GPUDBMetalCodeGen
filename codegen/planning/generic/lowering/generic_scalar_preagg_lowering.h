#pragma once

#include "generic/ir/generic_relational_ir.h"
#include "generic/lowering/generic_scalar_lookup.h"
#include "metal_plan_builder.h"

#include <vector>

namespace codegen {

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
    const GenericRelPlan& ir,
    MetalQueryPlan& plan);

} // namespace codegen
