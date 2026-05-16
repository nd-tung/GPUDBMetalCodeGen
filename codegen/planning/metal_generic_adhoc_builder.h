#pragma once

#include "metal_plan_builder.h"

#include <optional>
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

    int sentinel = 0;
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
};

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
    const AnalyzedQuery& aq,
    MetalQueryPlan& plan);

std::string genericScalarLookupKeyExpr(
    const GenericScalarLookupInfo& info,
    size_t keyIndex,
    const std::string& idxVar,
    const std::string& probeTable,
    const SchemaProvider* schema);

std::string rewriteGenericScalarSentinels(
    const std::string& cond,
    const std::string& idxVar,
    const std::vector<GenericScalarLookupInfo>& lookups,
    const std::string& probeTable,
    const SchemaProvider* schema);

bool referencesGenericScalarSentinel(
    const std::string& text,
    const std::vector<GenericScalarLookupInfo>& lookups);

bool referencesGenericScalarLookupBuffer(
    const std::string& text,
    const std::vector<GenericScalarLookupInfo>& lookups);

void attachGenericScalarLookupBuffers(
    MetalQueryPlan::Phase& phase,
    const std::vector<GenericScalarLookupInfo>& lookups);

// Returns nullopt + writes diagnostics to `error` when the query cannot be
// planned.  The error parameter may be nullptr.
// `schema` selects the schema provider; defaults to TPCHSchemaProvider.
std::optional<MetalQueryPlan> buildGenericSingleTableAdhocPlan(const AnalyzedQuery& aq,
                                                                std::string* error = nullptr);

// Multi-table generic ad-hoc plan.  Handles any number of tables connected by
// equi-joins.  The plan uses SemiJoin (bitmap) steps for filter-only joins
// and falls back to IndexJoin (array) when build-side columns are referenced
// in the output.  Returns nullopt + writes diagnostics to `error` when the
// query cannot be planned.
std::optional<MetalQueryPlan> buildGenericMultiTableAdhocPlan(
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

} // namespace codegen
