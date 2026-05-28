#pragma once

#include "query_plan.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace codegen {

class SchemaProvider;

struct AnalyzedQuery {
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;
    std::vector<JoinClause> joins;
    std::vector<PredPtr> filters;
    std::map<std::string, std::vector<PredPtr>> instanceFilters;

    struct InSubqueryAggInfo {
        std::string alias;
        std::string baseTable;
        int tableIndex = -1;
        std::string groupCol;
        std::string aggFunc;
        std::string aggExpr;
        PredPtr havingPred;
    };
    std::vector<InSubqueryAggInfo> inSubAggs;
    std::vector<FromSubqueryAggInfo> fromSubqueryAggs;
    std::vector<SelectTarget> targets;
    std::vector<ExprPtr> groupBy;
    PredPtr having;
    std::vector<OrderByItem> orderBy;
    int limit = -1;
    std::vector<std::string> scalarSubquerySql;

    const SchemaProvider* schema = nullptr;
    std::unordered_map<std::string, std::string> aliasMap;
};

AnalyzedQuery collectAnalyzedQuery(const std::string& sql,
                                   const SchemaProvider& schema);

} // namespace codegen
