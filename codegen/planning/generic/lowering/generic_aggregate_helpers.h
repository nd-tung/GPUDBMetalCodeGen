#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <optional>
#include <string>
#include <vector>

namespace codegen {

struct AnalyzedQuery;
struct GenericGroupSpec;
struct MultiTableGroupedAggShape;

struct IrGroupKeyDesc {
    std::string displayName;
    std::string keyExpr;
    int numValues = 0;
    std::string numValuesExpr;
    int stride = 1;
    std::vector<char> charMap;
    int keyBase = 0;
    std::vector<std::string> stringMap;
    int stringLen = 0;
};

struct IrPendingAgg {
    std::string displayName;
    int offset = 0;
    std::string valueExpr;
    bool isLongPair = false;
    int scaleDown = 0;
    bool isFloatSum = false;
    bool isMinMax = false;
    std::string atomicOp = "add";
    std::string funcName;
    std::string innerColumn;
};

bool genericExprEquivalent(const GenericExprPtr& left,
                           const GenericExprPtr& right);

std::optional<std::string> sortKeyDisplayNameForGroupedAgg(
    const GenericSortKey& key,
    const GenericAggregateDetail& aggregate,
    const std::vector<IrGroupKeyDesc>& groupKeys);

std::string char1BucketExpr(const GenericColumnExpr& col,
                            const std::string& idxVar);
std::string scaledLongExpr(const std::string& rawExpr, int scale);
int numericScaleForExpr(const GenericExprPtr& expr);
std::string distinctDomainSymbolForExpr(const GenericExprPtr& expr);
std::string innerColumnName(const GenericExprPtr& expr);

std::string groupDisplayNameForAggregate(
    const GenericAggregateDetail& aggregate,
    size_t index);

std::string aggregateOutputFuncFor(
    const GenericAggregateDetail& aggregate,
    size_t index,
    AggFunc fallback);

bool aggregateNeedsHashGroupOutput(const GenericAggregateDetail& aggregate);
bool canUseKeyedSingleTableGroup(const GenericAggregateDetail& aggregate);
bool configureAggregateHaving(const GenericAggregateDetail& aggregate,
                              GenericGroupSpec& groupSpec,
                              const AnalyzedQuery* aq,
                              const MultiTableGroupedAggShape* shape,
                              std::string* error);

} // namespace codegen
