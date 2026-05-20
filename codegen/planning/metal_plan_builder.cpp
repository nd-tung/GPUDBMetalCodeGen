#include "metal_plan_builder.h"

// Metal planning is split by responsibility:
//   - metal_plan_common.cpp: shared plan-builder helpers
//   - metal_plan_codegen.cpp: MetalQueryPlan -> MetalCodegen lowering
//   - tpch/queries/tpch_q*.cpp: hand-tuned per-query TPC-H plan builders
