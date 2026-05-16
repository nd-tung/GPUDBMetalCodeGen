#pragma once

#include "generic_relational_ir.h"
#include "query_analyzer.h"

#include <optional>
#include <string>

namespace codegen {

// Builds the GPU-neutral relational IR used by generic lowering routes.
std::optional<GenericRelPlan> buildGenericRelationalIR(const AnalyzedQuery& aq,
                                                       std::string* error = nullptr);

} // namespace codegen
