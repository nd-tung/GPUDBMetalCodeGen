#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace codegen {

struct SingleTableShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* project = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct SingleTableScalarAggShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
};

struct SingleTableGroupedAggShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct MultiTableMaterializeShape {
    std::vector<const GenericRelNode*> scans;
    std::vector<const GenericRelNode*> joins;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* project = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct MultiTableGroupedAggShape {
    std::vector<const GenericRelNode*> scans;
    std::vector<const GenericRelNode*> joins;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

std::optional<SingleTableShape> parseSingleTableMaterializeShape(
    const GenericRelPlan& ir,
    std::string* error);

std::optional<SingleTableScalarAggShape> parseSingleTableScalarAggShape(
    const GenericRelPlan& ir,
    std::string* error);

std::optional<SingleTableGroupedAggShape> parseSingleTableGroupedAggShape(
    const GenericRelPlan& ir,
    std::string* error);

std::optional<MultiTableMaterializeShape> parseMultiTableMaterializeShape(
    const GenericRelPlan& ir,
    std::string* error);

std::optional<MultiTableGroupedAggShape> parseMultiTableGroupedAggShape(
    const GenericRelPlan& ir,
    std::string* error);

std::optional<MultiTableGroupedAggShape> parseMultiTableScalarAggShape(
    const GenericRelPlan& ir,
    std::string* error);

void collectScanRelationInstances(const GenericRelPlan& ir,
                                  const GenericRelNode* node,
                                  std::set<int>& relationInstances);

const GenericScanDetail* scanDetail(const GenericRelNode* node);
const GenericProjectDetail* projectDetail(const GenericRelNode* node);
const GenericFilterDetail* filterDetail(const GenericRelNode* node);
const GenericSortDetail* sortDetail(const GenericRelNode* node);
int limitValue(const GenericRelNode* node);
const GenericAggregateDetail* aggregateDetail(const GenericRelNode* node);

bool projectedColumnMatches(const GenericProjection& projection,
                            const GenericColumnExpr& sortColumn);

std::optional<std::string> sortKeyDisplayName(
    const GenericSortKey& key,
    const GenericProjectDetail& project);

} // namespace codegen
