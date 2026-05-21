#pragma once

#include "query_plan.h"

#include <string>

namespace codegen {

inline std::string scalarSubqueryPlaceholderToken(int scalarSubqueryIndex) {
    return "__generic_scalar_lookup_" + std::to_string(scalarSubqueryIndex) + "__";
}

bool analyzedExprReferencesScalarSubquery(const ExprPtr& expr);
bool analyzedPredicateReferencesScalarSubquery(const PredPtr& pred);
bool analyzedExprIsScalarSubqueryRef(const ExprPtr& expr, int scalarSubqueryIndex);
bool analyzedPredicateReferencesScalarSubquery(const PredPtr& pred,
                                               int scalarSubqueryIndex);

} // namespace codegen
