#pragma once
#include "query_plan.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <unordered_map>

namespace codegen {

class SchemaProvider;
class Catalog;

// --- Analyzed Query ---

struct JoinClause {
    std::string leftTable, rightTable;
    std::string leftCol, rightCol;
    bool anti = false;       // NOT EXISTS anti-semi join.
    bool leftOuter = false;  // LEFT OUTER JOIN.
    bool semi = false;       // EXISTS semi join.
    std::string innerTable;  // EXISTS/NOT EXISTS inner table.
};

struct AggTarget {
    AggFunc func;
    ExprPtr innerExpr;  // Aggregate input expression.
    std::string alias;
    bool isStar = false; // COUNT(*)
};

struct SelectTarget {
    ExprPtr expr;
    std::string alias;
    bool isAgg = false;
    std::optional<AggTarget> agg;
};

struct FromSubqueryAggInfo {
    std::string alias;  // FROM-subquery alias.
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;
    std::vector<JoinClause> joins;
    std::vector<PredPtr> filters;
    std::vector<SelectTarget> targets;
    std::vector<ExprPtr> groupBy;
};

struct OrderByItem {
    ExprPtr expr;
    bool descending = false;
};

struct AnalyzedQuery {
    // FROM clause
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;

    // Equi-joins extracted from WHERE or explicit JOIN ON.
    std::vector<JoinClause> joins;

    // Non-join WHERE predicates.
    std::vector<PredPtr> filters;

    // Alias-scoped filters for duplicate table instances.
    std::map<std::string, std::vector<PredPtr>> instanceFilters;

    // IN subqueries with GROUP BY/HAVING need aggregate build metadata.
    struct InSubqueryAggInfo {
        std::string alias;        // Duplicate table alias.
        std::string baseTable;    // Base table name.
        int tableIndex = -1;      // Index in tables.
        std::string groupCol;     // GROUP BY column.
        std::string aggFunc;      // SUM, COUNT, AVG, MIN, or MAX.
        std::string aggExpr;      // Aggregate input column.
        PredPtr havingPred;       // HAVING predicate.
    };
    std::vector<InSubqueryAggInfo> inSubAggs;

    // Grouped FROM subqueries keep their aggregate boundary here.
    std::vector<FromSubqueryAggInfo> fromSubqueryAggs;

    // SELECT target list
    std::vector<SelectTarget> targets;

    // GROUP BY
    std::vector<ExprPtr> groupBy;

    // HAVING
    PredPtr having;

    // ORDER BY
    std::vector<OrderByItem> orderBy;

    // LIMIT
    int limit = -1;

    // Subqueries (IN, EXISTS, scalar)
    struct Subquery {
        enum Type { IN_SUBQUERY, EXISTS_SUBQUERY, NOT_EXISTS_SUBQUERY, SCALAR_SUBQUERY };
        Type type;
        std::string sql; // Raw SQL or serialized AST for re-analysis.
        ExprPtr outerExpr; // IN subquery test expression.
        AnalyzedQuery* analyzed = nullptr; // Filled by later planning.
    };
    std::vector<Subquery> subqueries;

    // Schema provider selected for this analysis.
    const SchemaProvider* schema = nullptr;

    // Catalog derived from the schema provider.
    const Catalog* catalog = nullptr;

    // FROM-subquery column aliases -> source column.
    std::unordered_map<std::string, ColRef> subqueryColMap;
    // FROM-subquery column aliases -> source expression.
    std::unordered_map<std::string, ExprPtr> subqueryExprMap;

    // Table alias -> base table name.
    std::unordered_map<std::string, std::string> aliasMap;

    // Helpers
    bool isSingleTable() const { return tables.size() == 1 && joins.empty(); }
    bool hasAggregation() const {
        for (auto& t : targets) if (t.isAgg) return true;
        return false;
    }
    bool hasGroupBy() const { return !groupBy.empty(); }
};

// --- Public API ---

// Parse SQL into planner-facing structure; throws std::runtime_error on failure.
AnalyzedQuery analyzeSQL(const std::string& sql,
                         const SchemaProvider* schema = nullptr);

// Default schema provider for callers that do not inject one.
extern SchemaProvider& defaultSchemaProvider();

} // namespace codegen
