#pragma once

#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

// Returns nullopt + writes diagnostics to `error` when the query cannot be
// planned.  The error parameter may be nullptr.
std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq,
                                                                std::string* error = nullptr);

// Multi-table generic ad-hoc plan.  Handles any number of tables connected by
// equi-joins.  The plan uses SemiJoin (bitmap) steps for filter-only joins
// and falls back to IndexJoin (array) when build-side columns are referenced
// in the output.  Returns nullopt + writes diagnostics to `error` when the
// query cannot be planned.
std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan(
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

} // namespace codegen