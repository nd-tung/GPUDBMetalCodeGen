#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

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
    bool stringRowRef = false;
    std::string stringSourceTable;
    std::string stringSourceColumn;
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

struct ScalarReduceAccumulatorSpec {
    enum class Op { Sum, Min, Max };

    std::string valueExpr;
    std::string metalType = "float";
    int outputScale = 0;
    Op op = Op::Sum;
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
std::optional<ScalarReduceAccumulatorSpec> buildScalarReduceAccumulatorSpec(
    AggFunc func,
    const GenericExprPtr& arg,
    std::string valueExpr);
std::string distinctDomainSymbolForExpr(const GenericExprPtr& expr);
std::string innerColumnName(const GenericExprPtr& expr);

std::string groupDisplayNameForAggregate(
    const GenericAggregateDetail& aggregate,
    size_t index);

std::string aggregateOutputFuncFor(
    const GenericAggregateDetail& aggregate,
    size_t index,
    AggFunc fallback);

using AggregateInputColumnBuilder = std::function<bool(
    const std::string& displayName,
    const TypeInfo& type,
    const GenericExprPtr& expr,
    int scaleDown,
    const std::string& distinctDomainSymbol)>;

bool buildAggregateInputGroupSpec(
    const GenericAggregateDetail& aggregate,
    const std::string& errorContext,
    GenericGroupSpec& groupSpec,
    std::vector<IrGroupKeyDesc>& groupKeys,
    const AggregateInputColumnBuilder& addInputColumn,
    std::string* error);

bool aggregateNeedsHashGroupOutput(const GenericAggregateDetail& aggregate);
bool canUseKeyedSingleTableGroup(const GenericAggregateDetail& aggregate);
bool configureAggregateHaving(const GenericAggregateDetail& aggregate,
                              GenericGroupSpec& groupSpec,
                              const GenericRelPlan* ir,
                              const MultiTableGroupedAggShape* shape,
                              std::string* error);

} // namespace codegen
