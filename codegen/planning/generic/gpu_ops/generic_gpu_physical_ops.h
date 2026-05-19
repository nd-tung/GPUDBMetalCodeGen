#pragma once

#include "metal_plan_builder.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

// Materialized GPU column metadata shared by group, sort, and collection.
struct GenericMatColumnDesc {
    GenericMatColumnDesc() = default;
    GenericMatColumnDesc(std::string displayName,
                         std::string bufferName,
                         std::string metalType,
                         int stringLen = 0,
                         int scaleDown = 0,
                         bool isLongPair = false,
                         std::string distinctDomainSymbol = {},
                         bool stringRowRef = false,
                         std::string stringSourceTable = {},
                         std::string stringSourceColumn = {})
        : displayName(std::move(displayName)),
          bufferName(std::move(bufferName)),
          metalType(std::move(metalType)),
          stringLen(stringLen),
          scaleDown(scaleDown),
          isLongPair(isLongPair),
          distinctDomainSymbol(std::move(distinctDomainSymbol)),
          stringRowRef(stringRowRef),
          stringSourceTable(std::move(stringSourceTable)),
          stringSourceColumn(std::move(stringSourceColumn)) {}

    std::string displayName;
    std::string bufferName;
    std::string metalType;
    // stringLen > 0 marks fixed-width char data.
    int stringLen = 0;
    int scaleDown = 0;
    bool isLongPair = false;
    std::string distinctDomainSymbol;
    // stringRowRef stores source row ids instead of copied bytes.
    bool stringRowRef = false;
    std::string stringSourceTable;
    std::string stringSourceColumn;
};

struct GenericSortSpec {
    struct SortKey {
        std::string column;
        bool descending = false;
    };
    std::vector<SortKey> keys;
    int limit = -1;
};

struct GenericGroupSpec {
    std::vector<std::string> keyColumns;
    std::vector<std::string> aggColumns;
    std::vector<std::string> aggFuncs;
    std::vector<std::string> outputColumns;
    std::vector<std::pair<int,int>> ratioPairs;
    int havingAggIdx = -1;
    double havingMultiplier = 0;
    int havingSentinel = 0;
    std::string havingScalarCompareOp = ">";
    int havingCompareAggIdx = -1;
    std::string havingCompareOp;
    double havingCompareValue = 0;
};

struct GenericGpuGroupSpec {
    std::string tag;
    std::string inputCounter;
    std::string inputRowsSymbol;
    // capacityExpr sizes hash storage; maxOutputRowsExpr sizes output.
    std::string capacityExpr;
    std::string capacitySymbol;
    std::string maxOutputRowsExpr;
    std::string outputCounter;
    std::vector<GenericMatColumnDesc> inputColumns;
    GenericGroupSpec groupBy;
};

struct KeyedCompactKeySpec {
    KeyedCompactKeySpec() = default;
    KeyedCompactKeySpec(std::string displayName,
                        int numValues = 0,
                        std::string numValuesExpr = {},
                        int stride = 1,
                        std::vector<char> charMap = {},
                        int keyBase = 0,
                        std::vector<std::string> stringMap = {},
                        int stringLen = 0,
                        bool stringRowRef = false,
                        std::string stringSourceTable = {},
                        std::string stringSourceColumn = {})
        : displayName(std::move(displayName)),
          numValues(numValues),
          numValuesExpr(std::move(numValuesExpr)),
          stride(stride),
          charMap(std::move(charMap)),
          keyBase(keyBase),
          stringMap(std::move(stringMap)),
          stringLen(stringLen),
          stringRowRef(stringRowRef),
          stringSourceTable(std::move(stringSourceTable)),
          stringSourceColumn(std::move(stringSourceColumn)) {}

    std::string displayName;
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

struct KeyedCompactAggSpec {
    std::string displayName;
    int offset = 0;
    // Result collection uses these flags to reconstruct aggregate values.
    bool isLongPair = false;
    int scaleDown = 0;
    bool isFloatSum = false;
    bool isMinMax = false;
    std::string atomicOp = "add";
    bool isAvg = false;
    bool avgSumIsLongPair = false;
    int countOffset = -1;
    bool countIsFloat = false;
    bool isRatio = false;
    int ratioDenOffset = -1;
    bool ratioDenIsLongPair = false;
    bool ratioDenIsFloatSum = false;
    int ratioDenScaleDown = 0;
};

struct KeyedCompactHavingSpec {
    int scalarAggOffset = -1;
    bool scalarAggIsLongPair = false;
    bool scalarAggIsFloatSum = false;
    int scalarAggScaleDown = 0;
    std::string scalarTotalBuffer;
    std::string scalarCompareOp = ">";
    double scalarMultiplier = 0.0;

    int compareAggOffset = -1;
    bool compareAggIsLongPair = false;
    bool compareAggIsFloatSum = false;
    int compareAggScaleDown = 0;
    std::string compareOp;
    double compareValue = 0.0;
};

std::vector<GenericMatColumnDesc> genericGpuGroupOutputColumns(
    const GenericGpuGroupSpec& spec);

void attachMaterializedCountHook(MetalQueryPlan::Phase& phase,
                                 std::string counterName,
                                 std::string symbolName);

// Appends build, aggregate, optional HAVING, and materialize phases.
void appendGenericGpuGroupBy(MetalQueryPlan& plan,
                             const GenericGpuGroupSpec& spec);

std::unique_ptr<MetalOperator> makeKeyedAggCompactOperator(
    std::string inputBuffer,
    std::string outputCounter,
    int numBuckets,
    int valuesPerBucket,
    std::vector<KeyedCompactKeySpec> keys,
    std::vector<KeyedCompactAggSpec> aggs,
    std::vector<GenericMatColumnDesc> outputs,
    std::string bucketCountExpr = {},
    std::string bucketCountSymbol = {},
    KeyedCompactHavingSpec having = {});

bool appendGenericGpuSort(MetalQueryPlan& plan,
                          const std::string& tag,
                          const std::string& nRowsSymbol,
                          const std::string& capacityExpr,
                          const std::vector<GenericMatColumnDesc>& columns,
                          const GenericSortSpec& sortSpec,
                          std::string* error);

bool appendGenericGpuSmallSort(MetalQueryPlan& plan,
                               const std::string& tag,
                               const std::string& nRowsSymbol,
                               int maxRows,
                               const std::vector<GenericMatColumnDesc>& columns,
                               const GenericSortSpec& sortSpec,
                               std::string* error);

bool appendGenericGpuTopK(MetalQueryPlan& plan,
                          const std::string& tag,
                          const std::string& nRowsSymbol,
                          const std::string& capacityExpr,
                          const std::vector<GenericMatColumnDesc>& columns,
                          const GenericSortSpec& sortSpec,
                          std::string* error);

bool appendGenericGpuTopKSelection(MetalQueryPlan& plan,
                                   const std::string& tag,
                                   const std::string& nRowsSymbol,
                                   const std::string& capacityExpr,
                                   const std::vector<GenericMatColumnDesc>& columns,
                                   const GenericSortSpec& sortSpec,
                                   std::string* error);

} // namespace codegen
