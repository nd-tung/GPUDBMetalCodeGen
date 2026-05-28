#pragma once

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/ir/generic_relational_ir.h"
#include "metal_plan_builder.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

struct GenericCostRelationEstimate {
    int relationId = -1;
    std::string name;
    std::string rowBoundExpr;
    std::optional<double> rowBound;
    std::string primaryKeyColumn;
    std::string primaryKeyDomainExpr;
    int probePriority = 0;
    bool virtualRelation = false;
};

struct GenericCostRelationInstanceEstimate {
    int relationInstanceId = -1;
    int relationId = -1;
    std::string baseName;
    std::string alias;
    std::string rowBoundExpr;
    std::optional<double> rowBound;
};

struct GenericCostColumnEstimate {
    int relationInstanceId = -1;
    std::string table;
    std::string alias;
    std::string column;
    TypeInfo type{DataType::INT, 0};
    size_t byteWidth = 4;
    std::optional<int64_t> finiteValueCount;
    std::string keyDomainExpr;
    std::string distinctDomainExpr;
};

struct GenericCostAlternativeTrace {
    std::string name;
    double cost = 0.0;
    std::string reason;
};

struct GenericCostDecisionTrace {
    std::string operatorName;
    std::string tag;
    std::string chosen;
    double chosenCost = 0.0;
    std::map<std::string, std::string> estimates;
    std::vector<GenericCostAlternativeTrace> rejected;
};

struct GenericCostContext {
    bool traceEnabled = false;
    std::vector<GenericCostRelationEstimate> relations;
    std::vector<GenericCostRelationInstanceEstimate> relationInstances;
    std::vector<GenericCostColumnEstimate> columns;
    std::vector<GenericCostDecisionTrace> decisions;

    const GenericCostRelationEstimate* relation(int relationId) const;
    const GenericCostRelationInstanceEstimate* relationInstance(
        int relationInstanceId) const;
    const GenericCostColumnEstimate* column(int relationInstanceId,
                                            const std::string& columnName) const;
};

struct DenseGroupCostChoice {
    bool useDense = false;
    double denseCost = 0.0;
    double hashCost = 0.0;
    std::string reason;
    GenericCostDecisionTrace trace;
};

struct FdTopKLateMaterializationChoice {
    bool useLateMaterialization = false;
    size_t fullWidth = 0;
    size_t narrowWidth = 0;
    double groupBound = 0.0;
    double limitRows = 0.0;
    double fullCompactBytes = 0.0;
    double lateMaterializeBytes = 0.0;
    double gatherBytes = 0.0;
    double savedBytes = 0.0;
    double requiredSavings = 0.0;
    std::string reason;
    GenericCostDecisionTrace trace;
};

struct GenericAggregationCandidateCostInput {
    std::string name;
    bool available = true;
    std::string reason;
    std::vector<GenericMatColumnDesc> outputColumns;
    int aggregateSlots = 0;
    int denseBuckets = 0;
    std::string denseBucketsExpr;
    bool dynamicDenseDomain = false;
    bool directInputFused = false;
    int directPipelineCarriedStringRowRefs = 0;
    int directPipelineExtraBuffers = 0;
    bool activeBucketCompaction = false;
    bool fdLateTopK = false;
};

struct GenericAggregationCostInput {
    std::string tag;
    std::string inputRowsExpr;
    std::string outputRowsExpr;
    std::vector<GenericMatColumnDesc> materializedInputColumns;
    std::vector<GenericMatColumnDesc> outputColumns;
    size_t groupKeyCount = 0;
    size_t aggregateCount = 0;
    int aggregateSlots = 0;
    int denseBuckets = 0;
    std::string denseBucketsExpr;
    bool dynamicDenseDomain = false;
    bool materializedHashAvailable = true;
    bool directDenseAvailable = false;
    bool directInputFused = false;
    bool directMaterializedAvailable = false;
    int directPipelineCarriedStringRowRefs = 0;
    int directPipelineExtraBuffers = 0;
    bool activeBucketCompaction = false;
    bool fdKeyedAvailable = false;
    bool fdLateTopK = false;
    bool countDistinctAvailable = false;
    int sortLimit = -1;
    std::vector<GenericAggregationCandidateCostInput> candidates;
};

struct MultiTableAggregationCostChoice {
    std::string chosenCandidate;
    double chosenCost = 0.0;
    GenericCostDecisionTrace trace;
};

struct DirectDenseInputModeChoice {
    bool usePipeline = false;
    double pipelineCost = 0.0;
    double materializedCost = 0.0;
    double requiredWin = 0.0;
    std::string reason;
    GenericCostDecisionTrace trace;
};

struct ActiveBucketCompactionCostChoice {
    bool useActiveList = false;
    double denseCompactCost = 0.0;
    double activeCompactCost = 0.0;
    double estimatedActiveBuckets = 0.0;
    double denseBuckets = 0.0;
    double activeFraction = 1.0;
    double requiredWin = 0.0;
    std::string reason;
    GenericCostDecisionTrace trace;
};

struct KeysetPropagationCostInput {
    std::string tag;
    std::string buildRowsExpr;
    std::string targetRowsExpr;
    std::string keyDomainExpr;
    size_t keyByteWidth = 4;
    size_t targetRowByteWidth = 16;
    double estimatedActiveKeyFraction = 0.5;
    int propagationDepth = 0;
    int reuseCount = 1;
    bool hasSourceBitmap = false;
};

struct KeysetPropagationCostChoice {
    bool useKeyset = false;
    double setupCost = 0.0;
    double probeCost = 0.0;
    double savedBytes = 0.0;
    double requiredSavings = 0.0;
    std::string reason;
    GenericCostDecisionTrace trace;
};

bool genericCostTraceEnabled();
class GenericCostSymbolScope {
public:
    explicit GenericCostSymbolScope(const GenericRelPlan& ir);
    ~GenericCostSymbolScope();
    GenericCostSymbolScope(const GenericCostSymbolScope&) = delete;
    GenericCostSymbolScope& operator=(const GenericCostSymbolScope&) = delete;
};
size_t genericCostTypeByteWidth(const TypeInfo& type);
std::optional<double> parseGenericCostPositiveNumber(const std::string& expr);
std::optional<double> resolveGenericCostExpression(const std::string& expr);
GenericCostContext buildGenericCostContext(const GenericRelPlan& ir);
std::string formatGenericCostDecision(const GenericCostDecisionTrace& decision);
std::string formatGenericCostContextSummary(const GenericCostContext& context,
                                            const std::string& route);
void appendGenericCostTrace(MetalQueryPlan& plan,
                            const GenericCostContext& context,
                            const std::string& route);
void appendGenericCostDecisionTrace(MetalQueryPlan& plan,
                                    const GenericCostDecisionTrace& decision);
std::optional<MetalQueryPlan> attachGenericCostTrace(
    std::optional<MetalQueryPlan>&& plan,
    const GenericRelPlan& ir,
    const std::string& route);
DenseGroupCostChoice chooseDenseGroupPlan(
    const std::vector<IrGroupKeyDesc>& keys,
    const std::vector<IrPendingAgg>& pending,
    int totalBuckets,
    bool dynamicDomain,
    const KeyedCompactHavingSpec& havingSpec,
    const std::string& tag = {});
int genericMatColumnByteWidthEstimate(const GenericMatColumnDesc& col);
size_t genericMatRowByteWidthEstimate(
    const std::vector<GenericMatColumnDesc>& columns,
    bool includeHidden = false);
double fdTopKGroupBoundEstimate(const std::string& outputBoundExpr,
                                const std::string& keyDomainExpr);
FdTopKLateMaterializationChoice chooseFdTopKLateMaterialization(
    const GenericSortSpec& sortSpec,
    const std::vector<GenericMatColumnDesc>& fullOutputs,
    const std::vector<GenericMatColumnDesc>& narrowOutputs,
    const std::string& outputBoundExpr,
    const std::string& keyDomainExpr,
    const std::string& tag = {});
bool shouldUseFdTopKLateMaterialization(
    const GenericSortSpec& sortSpec,
    const std::vector<GenericMatColumnDesc>& fullOutputs,
    const std::vector<GenericMatColumnDesc>& narrowOutputs,
    const std::string& outputBoundExpr,
    const std::string& keyDomainExpr);
MultiTableAggregationCostChoice chooseMultiTableAggregationPlan(
    const GenericAggregationCostInput& input,
    const std::string& preferredCandidate = {});
DirectDenseInputModeChoice chooseDirectDenseInputMode(
    const GenericAggregationCostInput& input,
    const std::string& tag = {});
ActiveBucketCompactionCostChoice chooseActiveBucketCompaction(
    const GenericAggregationCostInput& input,
    const std::string& tag = {});
KeysetPropagationCostChoice chooseKeysetPropagation(
    const KeysetPropagationCostInput& input);

} // namespace codegen
