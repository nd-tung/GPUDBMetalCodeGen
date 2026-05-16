#include "generic/lowering/generic_ir_physical_planner.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "metal_plan_common.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

std::optional<MetalQueryPlan> lowerSingleTableHashGroupedAggregateIRToMetal(
    const GenericScanDetail& scan,
    const GenericAggregateDetail& aggregate,
    const GenericRelNode* filterNode,
    const GenericRelNode* sortNode,
    const GenericRelNode* limitNode,
    std::string* error);

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

} // namespace

std::optional<MetalQueryPlan> lowerSingleTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto shape = parseSingleTableGroupedAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!scan || !aggregate)
        return fail(error, "IR grouped aggregate lowerer: malformed scan/aggregate detail.");
    if (aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->having)
        return fail(error, "IR grouped aggregate lowerer: HAVING is not supported yet.");
    if (aggregate->aggregates.empty())
        return fail(error, "IR grouped aggregate lowerer: no aggregate outputs.");

    if (aggregateNeedsHashGroupOutput(*aggregate) ||
        !canUseKeyedSingleTableGroup(*aggregate)) {
        return lowerSingleTableHashGroupedAggregateIRToMetal(
            *scan, *aggregate, shape->filter, shape->sort, shape->limit, error);
    }

    const std::string idxVar = "i";
    std::vector<IrGroupKeyDesc> groupKeys;
    int totalBuckets = 1;
    for (size_t i = 0; i < aggregate->groupBy.size(); ++i) {
        auto* col = std::get_if<GenericColumnExpr>(&aggregate->groupBy[i]->node);
        if (!col)
            return std::nullopt;

        IrGroupKeyDesc key;
        key.displayName = groupDisplayNameForAggregate(*aggregate, i);
        key.stride = totalBuckets;
        if (col->type.type == DataType::CHAR1) {
            key.keyExpr = char1BucketExpr(*col, idxVar);
            key.numValues = static_cast<int>(col->charDomain.size());
            key.charMap = col->charDomain;
            if (key.keyExpr.empty() || key.numValues <= 0)
                return fail(error, "IR grouped aggregate lowerer: CHAR1 group key has no schema char domain.");
        } else {
            if (col->domainMin > col->domainMax)
                return fail(error, "IR grouped aggregate lowerer: group key has no schema domain.");
            key.numValues = col->domainMax - col->domainMin + 1;
            key.keyBase = col->domainMin;
            std::string raw = col->column + "[" + idxVar + "]";
            if (col->domainMin != 0)
                key.keyExpr = "(" + raw + " - " + std::to_string(col->domainMin) + ")";
            else
                key.keyExpr = raw;
            key.keyExpr = "clamp(" + key.keyExpr + ", 0, " +
                          std::to_string(key.numValues - 1) + ")";
        }
        if (key.numValues <= 0)
            return fail(error, "IR grouped aggregate lowerer: invalid group key domain.");
        totalBuckets *= key.numValues;
        groupKeys.push_back(std::move(key));
    }
    if (totalBuckets > 4096)
        return fail(error, "IR grouped aggregate lowerer: bucket count exceeds 4096.");

    std::string bucketExpr = "(" + groupKeys.front().keyExpr + ")";
    for (size_t i = 1; i < groupKeys.size(); ++i) {
        bucketExpr = "(" + bucketExpr + " + (" + groupKeys[i].keyExpr + ") * " +
                     std::to_string(groupKeys[i].stride) + ")";
    }

    std::vector<IrPendingAgg> pending;
    int valuesPerBucket = 0;
    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr ? std::get_if<GenericAggregateExpr>(&projection.expr->node) : nullptr;
        if (!agg)
            return fail(error, "IR grouped aggregate lowerer: non-aggregate projection.");

        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        if (agg->func == AggFunc::COUNT) {
            IrPendingAgg out;
            out.displayName = displayName;
            out.offset = valuesPerBucket++;
            out.valueExpr = "1u";
            out.funcName = "COUNT";
            pending.push_back(std::move(out));
            continue;
        }
        if (!agg->arg)
            return fail(error, "IR grouped aggregate lowerer: aggregate argument required.");
        if (!materializeExprSupported(agg->arg))
            return fail(error, "IR grouped aggregate lowerer: aggregate argument unsupported.");

        if (agg->func == AggFunc::AVG) {
            const bool isFloat = agg->arg->type.type == DataType::FLOAT;
            const int fixedScale = isFloat ? numericScaleForExpr(agg->arg) : 0;
            IrPendingAgg sum;
            sum.displayName = displayName;
            sum.offset = valuesPerBucket;
            std::string valueExpr = genericExprToMetal(agg->arg, idxVar);
            if (isFloat && fixedScale > 0) {
                sum.valueExpr = scaledLongExpr(valueExpr, fixedScale);
                sum.isLongPair = true;
                sum.scaleDown = -fixedScale;
                valuesPerBucket += 2;
            } else if (isFloat) {
                sum.valueExpr = valueExpr;
                sum.isFloatSum = true;
                sum.scaleDown = -1;
                valuesPerBucket += 1;
            } else {
                sum.valueExpr = valueExpr;
                sum.isLongPair = true;
                sum.scaleDown = -1;
                valuesPerBucket += 2;
            }
            sum.funcName = "AVG";
            sum.innerColumn = innerColumnName(agg->arg);
            pending.push_back(std::move(sum));

            IrPendingAgg cnt;
            cnt.displayName = displayName + "_cnt";
            cnt.offset = valuesPerBucket++;
            cnt.valueExpr = "1u";
            cnt.funcName = "AVG";
            pending.push_back(std::move(cnt));
            continue;
        }

        IrPendingAgg out;
        out.displayName = displayName;
        out.offset = valuesPerBucket;
        out.valueExpr = genericExprToMetal(agg->arg, idxVar);
        out.funcName = aggFuncName(agg->func);
        out.innerColumn = innerColumnName(agg->arg);
        if (agg->func == AggFunc::MIN || agg->func == AggFunc::MAX) {
            out.atomicOp = agg->func == AggFunc::MIN ? "min" : "max";
            out.isMinMax = true;
            if (agg->arg->type.type == DataType::FLOAT)
                out.isFloatSum = true;
            valuesPerBucket += 1;
        } else if (agg->func == AggFunc::SUM) {
            if (agg->arg->type.type == DataType::FLOAT) {
                const int fixedScale = numericScaleForExpr(agg->arg);
                if (fixedScale > 0) {
                    out.valueExpr = scaledLongExpr(out.valueExpr, fixedScale);
                    out.isLongPair = true;
                    out.scaleDown = fixedScale;
                    valuesPerBucket += 2;
                } else {
                    out.isFloatSum = true;
                    valuesPerBucket += 1;
                }
            } else {
                out.isLongPair = true;
                valuesPerBucket += 2;
            }
        } else {
            return fail(error, "IR grouped aggregate lowerer: unsupported aggregate " +
                               aggFuncName(agg->func) + ".");
        }
        pending.push_back(std::move(out));
    }
    if (pending.empty())
        return fail(error, "IR grouped aggregate lowerer: no aggregate slots.");

    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan->table, idxVar);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate))
            return fail(error, "IR grouped aggregate lowerer: filter predicate unsupported.");
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }
    pipe = maybeSelect(std::move(pipe), "(" + bucketExpr + " >= 0 && " +
                       bucketExpr + " < " + std::to_string(totalBuckets) + ")");

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SINGLE_TABLE_GROUP";
    plan.chunkable = true;
    auto keyed = std::make_unique<MetalKeyedAgg>(
        std::move(pipe), "d_ir_group_aggs", bucketExpr,
        totalBuckets, valuesPerBucket, std::to_string(totalBuckets * valuesPerBucket));

    std::vector<std::string> keyNames;
    std::vector<GroupKeyDecode> decodeInfo;
    for (const auto& key : groupKeys) {
        keyNames.push_back(key.displayName);
        GroupKeyDecode d;
        d.name = key.displayName;
        d.numValues = key.numValues;
        d.stride = key.stride;
        d.charMap = key.charMap;
        d.keyBase = key.keyBase;
        decodeInfo.push_back(std::move(d));
    }
    keyed->setMultiKeyResult(keyNames, decodeInfo, totalBuckets);

    for (const auto& agg : pending) {
        keyed->addAggregateWithMeta(agg.displayName, agg.offset, agg.valueExpr,
                                    agg.atomicOp, agg.isLongPair, agg.scaleDown,
                                    agg.isFloatSum, agg.isMinMax,
                                    agg.funcName, agg.innerColumn);
    }

    auto& groupPhase = appendPhase(plan, "GENERIC_ir_single_table_group", std::move(keyed));

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, *aggregate, groupKeys);
            if (!name)
                return fail(error, "IR grouped aggregate lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    (void)groupPhase;

    std::vector<KeyedCompactKeySpec> compactKeys;
    std::vector<GenericMatColumnDesc> compactCols;
    for (const auto& key : groupKeys) {
        compactKeys.push_back({key.displayName, key.numValues, key.stride,
                               key.charMap, key.keyBase, {}, 0});
        std::string buf = "d_ir_keyed_out_" + std::to_string(compactCols.size()) +
                          "_" + sanitizeIdentifier(key.displayName);
        compactCols.push_back(GenericMatColumnDesc{
            key.displayName, buf, key.charMap.empty() ? "int" : "char"});
    }

    std::vector<KeyedCompactAggSpec> compactAggs;
    for (size_t pi = 0; pi < pending.size(); ++pi) {
        const auto& p = pending[pi];
        KeyedCompactAggSpec out;
        out.displayName = p.displayName;
        out.offset = p.offset;
        out.isLongPair = p.isLongPair;
        out.scaleDown = p.scaleDown;
        out.isFloatSum = p.isFloatSum;
        out.isMinMax = p.isMinMax;
        out.atomicOp = p.atomicOp;
        out.avgSumIsLongPair = p.isLongPair;

        std::string metalType = "uint";
        int outScale = 0;
        bool outLongPair = false;
        if (p.scaleDown < 0 && pi + 1 < pending.size()) {
            const auto& cnt = pending[pi + 1];
            out.isAvg = true;
            out.countOffset = cnt.offset;
            out.countIsFloat = cnt.isFloatSum;
            metalType = "float";
            ++pi;
        } else if (p.isLongPair) {
            metalType = "uint";
            outScale = p.scaleDown > 0 ? p.scaleDown : 0;
            outLongPair = true;
            out.isLongPair = true;
        } else if (p.isFloatSum || p.isMinMax) {
            metalType = "float";
        }
        std::string buf = "d_ir_keyed_out_" + std::to_string(compactCols.size()) +
                          "_" + sanitizeIdentifier(out.displayName);
        compactAggs.push_back(out);
        compactCols.push_back(GenericMatColumnDesc{
            out.displayName, buf, metalType, 0, outScale, outLongPair});
    }

    const std::string compactCounter = "d_ir_keyed_result_count";
    auto& compactPhase = appendPhase(plan, "GENERIC_ir_single_table_group_compact",
        makeKeyedAggCompactOperator(
            "d_ir_group_aggs", compactCounter, totalBuckets, valuesPerBucket,
            compactKeys, compactAggs, compactCols));

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string sortRowsSym = "n_gpu_sort_ir_single_keyed_rows";
        attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);
        if (!appendGenericGpuSort(plan, "ir_single_keyed", sortRowsSym,
                                  std::to_string(totalBuckets), compactCols,
                                  sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

} // namespace codegen
