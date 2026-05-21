#pragma once

#include "metal_plan_builder.h"

#include <string>
#include <vector>

namespace codegen {

class SchemaProvider;

struct GenericScalarLookupInfo {
    enum Kind {
        SumByKey, AvgByKey, MinByKey, MaxByKey, CountByKey,
        SumByCompositeHash, AvgByCompositeHash, CountByCompositeHash,
        GlobalSum, GlobalAvg, GlobalMin, GlobalMax, GlobalCount
    };

    int scalarSubqueryIndex = -1;
    Kind kind = SumByKey;
    std::string valueTable;
    std::string keyCol;
    std::string keyCol2;
    std::vector<std::string> keyCols;
    std::vector<std::string> outerKeyCols;
    std::string valueCol;
    float multiplier = 1.0f;
    std::string sizeSymbol;
    std::string sumBuffer;
    std::string countBuffer;
    std::string minBuffer;
    std::string maxBuffer;
    std::string stateBuffer;
    std::string hashMap;
    std::string countHashMap;
    std::string hashCapacityExpr;
    std::string htFlags;
    std::string htKeys;
    std::string htVals;
    std::string cntVar;
    std::string sumVar;
    std::string scalarName;
};

struct GenericScalarRewriteResult {
    std::string text;
    bool referencedScalarLookup = false;
    bool needsScalarLookupLoads = false;
    bool needsScalarLookupBuffers = false;
};

std::string genericScalarLookupKeyExpr(
    const GenericScalarLookupInfo& info,
    size_t keyIndex,
    const std::string& idxVar,
    const std::string& probeTable,
    const SchemaProvider* schema);

std::string rewriteGenericScalarPlaceholders(
    const std::string& cond,
    const std::string& idxVar,
    const std::vector<GenericScalarLookupInfo>& lookups,
    const std::string& probeTable,
    const SchemaProvider* schema);

GenericScalarRewriteResult rewriteGenericScalarPlaceholdersWithDeps(
    const std::string& cond,
    const std::string& idxVar,
    const std::vector<GenericScalarLookupInfo>& lookups,
    const std::string& probeTable,
    const SchemaProvider* schema);

void attachGenericScalarLookupBuffers(
    MetalQueryPlan::Phase& phase,
    const std::vector<GenericScalarLookupInfo>& lookups);

} // namespace codegen
