#pragma once

#include "metal_generic_executor.h"
#include "metal_plan_builder.h"
#include <string>
#include <vector>

namespace codegen {

// Applies plan-declared host finalization metadata to the collected result.
// The same path is used for predefined plans that expose compact GPU buffers
// and for ordinary materialized results that only need GPU-order remapping.
void finalizeHostResult(const MetalQueryPlan& plan,
                        MetalGenericExecutor& executor,
                        GenericResult& result,
                        std::vector<std::string>* hostOps = nullptr);

} // namespace codegen
