#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <optional>

namespace codegen {

bool isScalarSubquerySentinelLiteral(const GenericExprPtr& expr);
bool exprReferencesScalarSentinel(const GenericExprPtr& expr);
bool predicateReferencesScalarSentinel(const GenericPredicatePtr& pred);
std::optional<int> scalarSubqueryIndexFromSentinelLiteral(
    const GenericExprPtr& expr);

} // namespace codegen
