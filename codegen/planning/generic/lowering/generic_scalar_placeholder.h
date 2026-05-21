#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <optional>

namespace codegen {

bool isScalarSubqueryPlaceholderExpr(const GenericExprPtr& expr);
std::optional<int> scalarSubqueryIndexFromExpr(const GenericExprPtr& expr);
bool exprReferencesScalarSubqueryPlaceholder(const GenericExprPtr& expr);
bool predicateReferencesScalarSubqueryPlaceholder(const GenericPredicatePtr& pred);

} // namespace codegen
