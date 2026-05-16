#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <set>
#include <vector>

namespace codegen {

const GenericRelation* relationForScan(const GenericRelPlan& ir,
                                       const GenericRelNode* scanNode);

void collectExprRelations(const GenericExprPtr& expr,
                          std::set<int>& relationInstances);

void collectPredicateRelations(const GenericPredicatePtr& pred,
                               std::set<int>& relationInstances);

void splitConjuncts(const GenericPredicatePtr& pred,
                    std::vector<GenericPredicatePtr>& out);

} // namespace codegen
