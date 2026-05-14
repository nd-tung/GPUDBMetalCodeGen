#pragma once

#include "metal_plan_builder.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

struct GenericMatColumnDesc {
    GenericMatColumnDesc() = default;
    GenericMatColumnDesc(std::string displayName,
                         std::string bufferName,
                         std::string metalType,
                         int stringLen = 0,
                         int scaleDown = 0,
                         bool isLongPair = false,
                         std::string distinctDomainSymbol = {})
        : displayName(std::move(displayName)),
          bufferName(std::move(bufferName)),
          metalType(std::move(metalType)),
          stringLen(stringLen),
          scaleDown(scaleDown),
          isLongPair(isLongPair),
          distinctDomainSymbol(std::move(distinctDomainSymbol)) {}

    std::string displayName;
    std::string bufferName;
    std::string metalType;
    int stringLen = 0;
    int scaleDown = 0;
    bool isLongPair = false;
    std::string distinctDomainSymbol;
};

struct GenericGpuGroupSpec {
    std::string tag;
    std::string inputCounter;
    std::string inputRowsSymbol;
    std::string capacityExpr;
    std::string capacitySymbol;
    std::string outputCounter;
    std::vector<GenericMatColumnDesc> inputColumns;
    MetalQueryPlan::CpuGroupBy groupBy;
};

struct KeyedCompactKeySpec {
    std::string displayName;
    int numValues = 0;
    int stride = 1;
    std::vector<char> charMap;
    int keyBase = 0;
};

struct KeyedCompactAggSpec {
    std::string displayName;
    int offset = 0;
    bool isLongPair = false;
    int scaleDown = 0;
    bool isFloatSum = false;
    bool isMinMax = false;
    std::string atomicOp = "add";
    bool isAvg = false;
    bool avgSumIsLongPair = false;
    int countOffset = -1;
    bool countIsFloat = false;
};

std::vector<GenericMatColumnDesc> genericGpuGroupOutputColumns(
    const GenericGpuGroupSpec& spec);

void attachMaterializedCountHook(MetalQueryPlan::Phase& phase,
                                 std::string counterName,
                                 std::string symbolName);

void appendGenericGpuGroupBy(MetalQueryPlan& plan,
                             const GenericGpuGroupSpec& spec);

std::unique_ptr<MetalOperator> makeKeyedAggCompactOperator(
    std::string inputBuffer,
    std::string outputCounter,
    int numBuckets,
    int valuesPerBucket,
    std::vector<KeyedCompactKeySpec> keys,
    std::vector<KeyedCompactAggSpec> aggs,
    std::vector<GenericMatColumnDesc> outputs);

bool appendGenericGpuSort(MetalQueryPlan& plan,
                          const std::string& tag,
                          const std::string& nRowsSymbol,
                          const std::string& capacityExpr,
                          const std::vector<GenericMatColumnDesc>& columns,
                          const MetalQueryPlan::CpuSort& cpuSort,
                          std::string* error);

} // namespace codegen
