#pragma once

#include "generic_relational_ir.h"

#include <optional>
#include <string>

namespace codegen {

class SchemaProvider;

// Parses SQL and builds the GPU-neutral relational IR used by generic lowering routes.
std::optional<GenericRelPlan> buildGenericRelationalIRFromSQL(
    const std::string& sql,
    const SchemaProvider& schema,
    std::string* error = nullptr);

} // namespace codegen
