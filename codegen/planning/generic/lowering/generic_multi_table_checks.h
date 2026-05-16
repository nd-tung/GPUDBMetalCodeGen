#pragma once

#include "generic/lowering/generic_plan_shapes.h"

namespace codegen {

struct AnalyzedQuery;

bool hasScalarSubqueries(const AnalyzedQuery* aq);
bool groupedAggregateNeedsScalarPreAgg(
    const MultiTableGroupedAggShape& shape);
bool materializeNeedsScalarPreAgg(const MultiTableMaterializeShape& shape);
bool materializeHasEmptyInListPlaceholder(
    const MultiTableMaterializeShape& shape);

} // namespace codegen
