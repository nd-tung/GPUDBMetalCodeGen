#include "metal_plan_builder.h"

// Metal planning is split by responsibility:
//   - metal_plan_common.cpp: shared plan-builder helpers
//   - api/metal_plan_dispatch.cpp: dispatch wrapper
//   - metal_plan_codegen.cpp: MetalQueryPlan -> MetalCodegen lowering
//   - tpch/queries/tpch_q*.cpp: hand-tuned per-query TPC-H plan builders
