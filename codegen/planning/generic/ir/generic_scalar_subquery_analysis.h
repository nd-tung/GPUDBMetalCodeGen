#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

std::optional<DecorrelatedScalarSubquery> parseDecorrelatedScalarSubquery(
    const std::string& sqlJson,
    const SchemaProvider* schema,
    int sqIdx);

} // namespace codegen
