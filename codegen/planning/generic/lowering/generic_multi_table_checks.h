#pragma once

#include "generic/lowering/generic_plan_shapes.h"

namespace codegen {

bool hasScalarSubqueries(const GenericRelPlan& ir);
bool groupedAggregateNeedsScalarPreAgg(
    const MultiTableGroupedAggShape& shape);
bool materializeNeedsScalarPreAgg(const MultiTableMaterializeShape& shape);
bool materializeHasEmptyInListPlaceholder(
    const MultiTableMaterializeShape& shape);

} // namespace codegen
