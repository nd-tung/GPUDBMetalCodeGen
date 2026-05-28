#pragma once

#include "query_plan.h"

#include <string>

namespace codegen {

inline std::string scalarSubqueryPlaceholderToken(int scalarSubqueryIndex) {
    return "__generic_scalar_lookup_" + std::to_string(scalarSubqueryIndex) + "__";
}

} // namespace codegen
