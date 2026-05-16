#pragma once

#include "generic_relational_ir.h"
#include "query_analyzer.h"

#include <optional>
#include <string>

namespace codegen {

// Builds a logical, GPU-neutral relational IR from the analyzer output.
//
// Phase 1 uses this as compile-only infrastructure. The existing generic
// Metal builder remains the execution path until individual routes are
// migrated to IR lowering.
std::optional<GenericRelPlan> buildGenericRelationalIR(const AnalyzedQuery& aq,
                                                       std::string* error = nullptr);

} // namespace codegen
