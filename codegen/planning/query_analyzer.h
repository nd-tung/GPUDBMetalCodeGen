#pragma once
#include "query_plan.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <unordered_map>

namespace codegen {

class SchemaProvider;  // fwd
class Catalog;          // fwd

// ===================================================================
// ANALYZED QUERY — Extracted from SQL AST
// ===================================================================

struct JoinClause {
    std::string leftTable, rightTable;
    std::string leftCol, rightCol;
    bool anti = false;       // true for NOT EXISTS → anti-semi-join
    bool leftOuter = false;  // true for LEFT OUTER JOIN
    bool semi = false;       // true for EXISTS → semi-join (inner table = child)
    std::string innerTable;  // for semi joins: the EXISTS inner table
};

struct AggTarget {
    AggFunc func;
    ExprPtr innerExpr;  // the expression inside the aggregate
    std::string alias;
    bool isStar = false; // COUNT(*)
};

struct SelectTarget {
    ExprPtr expr;
    std::string alias;
    bool isAgg = false;
    std::optional<AggTarget> agg;
};

struct OrderByItem {
    ExprPtr expr;
    bool descending = false;
};

struct AnalyzedQuery {
    // FROM clause
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;

    // JOIN conditions (equi-joins extracted from WHERE or explicit JOIN ON)
    std::vector<JoinClause> joins;

    // WHERE predicates (non-join predicates, per-table filters)
    std::vector<PredPtr> filters;

    // Per-instance filters keyed by alias (e.g. l3-specific filters from
    // NOT EXISTS subquery — should NOT apply to other instances of same base table).
    std::map<std::string, std::vector<PredPtr>> instanceFilters;

    // IN subquery with GROUP BY + HAVING (e.g. Q18): aggregate metadata
    // so the builder creates an AtomicAgg build phase instead of a plain bitmap.
    struct InSubqueryAggInfo {
        std::string alias;        // table alias (e.g. "lineitem" duplicate)
        std::string baseTable;    // base table name
        int tableIndex = -1;      // index in aq.tables (to identify the dup)
        std::string groupCol;     // GROUP BY column
        std::string aggFunc;      // "SUM", "COUNT", "AVG"
        std::string aggExpr;      // expression inside aggregate (e.g. "l_quantity")
        PredPtr havingPred;       // HAVING predicate (e.g. SUM > 300)
    };
    std::vector<InSubqueryAggInfo> inSubAggs;

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
        std::string sql; // raw SQL for re-parsing
        ExprPtr outerExpr; // for IN: the outer expression being tested
        AnalyzedQuery* analyzed = nullptr; // filled later
    };
    std::vector<Subquery> subqueries;

    // Schema provider (injected — defaults to TPCHSchemaProvider).
    const SchemaProvider* schema = nullptr;

    // Catalog built from the schema provider (provides table/column metadata).
    const Catalog* catalog = nullptr;

    // FROM-clause subquery column aliases → source column (for simple col refs).
    std::unordered_map<std::string, ColRef> subqueryColMap;
    // FROM-clause subquery column aliases → source expression (for computed cols).
    std::unordered_map<std::string, ExprPtr> subqueryExprMap;

    // Table alias → real table name mapping (e.g. "n1" → "nation").
    std::unordered_map<std::string, std::string> aliasMap;

    // Helpers
    bool isSingleTable() const { return tables.size() == 1 && joins.empty(); }
    bool hasAggregation() const {
        for (auto& t : targets) if (t.isAgg) return true;
        return false;
    }
    bool hasGroupBy() const { return !groupBy.empty(); }
};

// ===================================================================
// PUBLIC API
// ===================================================================

// Parse a SQL string and extract structural information.
// Returns an AnalyzedQuery, or throws std::runtime_error on parse failure.
// `schema` selects the schema provider; defaults to TPCHSchemaProvider.
AnalyzedQuery analyzeSQL(const std::string& sql,
                         const SchemaProvider* schema = nullptr);

// Default schema provider used when no schema is specified.
extern SchemaProvider& defaultSchemaProvider();

} // namespace codegen
