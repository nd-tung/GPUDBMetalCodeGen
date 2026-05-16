#pragma once

#include "query_analyzer.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

struct DecorrCol {
    std::string table;
    std::string column;
    std::string qualifier;
    bool inner = false;
};

struct DecorrJoin {
    DecorrCol left;
    DecorrCol right;
};

struct DecorrCorrelation {
    DecorrCol inner;
    DecorrCol outer;
};

struct DecorrelatedScalarSubquery {
    int sqIdx = 0;
    AggFunc func = AggFunc::SUM;
    bool countStar = false;
    float multiplier = 1.0f;
    std::string valueTable;
    std::string valueCol;
    std::vector<std::string> tables;
    std::map<std::string, std::string> aliases;
    std::vector<DecorrJoin> joins;
    std::vector<DecorrCorrelation> correlations;
    std::map<std::string, std::vector<PredPtr>> filtersByTable;
};

std::optional<DecorrelatedScalarSubquery> parseDecorrelatedScalarSubquery(
    const std::string& sqlJson,
    const AnalyzedQuery& aq,
    int sqIdx);

} // namespace codegen
