#pragma once

#include "generic/ir/generic_relational_ir.h"
#include "generic/lowering/generic_join_carry.h"
#include "generic/lowering/generic_scalar_lookup.h"
#include "metal_plan_builder.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

struct MultiTableJoinLowering {
    MetalQueryPlan plan;
    std::unique_ptr<MetalOperator> probePipe;
    const GenericScanDetail* probeScan = nullptr;
    std::string outputSize;
    IrCarryMap carryMap;
};

std::optional<MultiTableJoinLowering> buildMultiTableJoinLowering(
    const GenericRelPlan& ir,
    const std::vector<const GenericRelNode*>& scans,
    const std::vector<const GenericRelNode*>& joins,
    const GenericRelNode* filterNode,
    const std::vector<GenericExprPtr>& neededExprs,
    const std::string& planName,
    const AnalyzedQuery* aq,
    const std::vector<GenericScalarLookupInfo>* scalarLookups,
    std::string* error);

} // namespace codegen
