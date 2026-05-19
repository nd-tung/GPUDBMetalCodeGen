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

struct GenericJoinDomainBitmapInfo {
    int relationInstance = -1;
    std::string storageTable;
    std::string storageAlias;
    std::string keyColumn;
    std::string bitmapName;
    std::string keyDomain;
    std::string buildPhaseName;
};

struct GenericInSubAggInfo {
    std::string table;
    std::string keyColumn;
    std::string valueColumn;
    std::string aggFunc;
    std::string aggBuffer;
    std::string keyDomain;
    std::string keyListBuffer;
    std::string keyListCountBuffer;
};

struct MultiTableJoinLowering {
    MetalQueryPlan plan;
    std::unique_ptr<MetalOperator> probePipe;
    const GenericScanDetail* probeScan = nullptr;
    std::string outputSize;
    IrCarryMap carryMap;
    std::vector<GenericJoinDomainBitmapInfo> domainBitmaps;
    std::vector<GenericInSubAggInfo> inSubAggs;
    bool probeUsesScalarLookupBuffer = false;
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
