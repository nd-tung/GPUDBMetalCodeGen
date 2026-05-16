#pragma once

#include "generic_relational_ir.h"

#include <string>
#include <vector>

namespace codegen {

struct GenericIrValidationResult {
    std::vector<std::string> errors;

    bool ok() const { return errors.empty(); }
    std::string message() const;
};

GenericIrValidationResult validateGenericRelationalIR(const GenericRelPlan& plan);

} // namespace codegen
