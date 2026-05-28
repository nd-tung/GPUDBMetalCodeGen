#pragma once

#include "query_plan.h"
#include "../../../../third_party/nlohmann/json.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace codegen {

class SchemaProvider;

struct GenericRelationId {
    int value = -1;
    bool valid() const { return value >= 0; }
};

struct GenericRelationInstanceId {
    int value = -1;
    bool valid() const { return value >= 0; }
};

struct GenericColumnId {
    int value = -1;
    bool valid() const { return value >= 0; }
};

struct GenericExprId {
    int value = -1;
    bool valid() const { return value >= 0; }
};

struct GenericNodeId {
    int value = -1;
    bool valid() const { return value >= 0; }
};

struct GenericRelation {
    GenericRelationId id;
    std::string name;
    bool virtualRelation = false;
    std::string maxKeySymbol;
    std::string primaryKeyColumn;
    std::string primaryKeyDomainSymbol;
    int probePriority = 0;
};

struct GenericRelationInstance {
    GenericRelationInstanceId id;
    GenericRelationId relation;
    std::string baseName;
    std::string alias;
};

struct GenericColumn {
    GenericColumnId id;
    GenericRelationInstanceId relationInstance;
    std::string name;
    std::string displayName;
    TypeInfo type;
};

struct GenericOutputSchema {
    std::vector<GenericColumn> columns;

    const GenericColumn* findByDisplayName(const std::string& name) const;
};

struct GenericExpr;
struct GenericPredicate;
using GenericExprPtr = std::shared_ptr<GenericExpr>;
using GenericPredicatePtr = std::shared_ptr<GenericPredicate>;

struct GenericColumnExpr {
    GenericRelationInstanceId relationInstance;
    std::string table;
    std::string alias;
    std::string column;
    TypeInfo type;
    bool hasGroupDomain = false;
    int domainMin = -1;
    int domainMax = -1;
    std::vector<char> charDomain;
    int numericScale = 0;
    std::string keyDomainSymbol;
    std::string distinctDomainSymbol;
};

struct GenericLiteralExpr {
    std::variant<int64_t, double, std::string> value;
    TypeInfo type;
};

struct GenericBinaryExpr {
    ExprOp op;
    GenericExprPtr left;
    GenericExprPtr right;
    TypeInfo type;
};

struct GenericCaseBranch {
    GenericPredicatePtr condition;
    GenericExprPtr result;
};

struct GenericCaseExpr {
    std::vector<GenericCaseBranch> branches;
    GenericExprPtr elseResult;
    TypeInfo type;
};

struct GenericFunctionExpr {
    std::string name;
    std::vector<GenericExprPtr> args;
    TypeInfo type;
};

struct GenericAggregateExpr {
    AggFunc func;
    GenericExprPtr arg;
    bool star = false;
    bool distinct = false;
    std::string alias;
    TypeInfo type;
};

struct GenericScalarSubqueryExpr {
    int index = -1;
    TypeInfo type;
};

struct GenericScalarLookupExpr {
    GenericNodeId source;
    std::string outputName;
    std::vector<GenericExprPtr> keys;
    TypeInfo type;
};

struct GenericExpr {
    GenericExprId id;
    TypeInfo type;
    std::variant<GenericColumnExpr,
                 GenericLiteralExpr,
                 GenericBinaryExpr,
                 GenericCaseExpr,
                 GenericFunctionExpr,
                 GenericAggregateExpr,
                 GenericScalarSubqueryExpr,
                 GenericScalarLookupExpr> node;
};

struct GenericComparisonPred {
    CmpOp op;
    GenericExprPtr left;
    GenericExprPtr right;
};

struct GenericBetweenPred {
    GenericExprPtr expr;
    GenericExprPtr low;
    GenericExprPtr high;
};

struct GenericInListPred {
    GenericExprPtr expr;
    std::vector<GenericExprPtr> values;
};

struct GenericLikePred {
    GenericExprPtr expr;
    std::string pattern;
    bool negated = false;
};

struct GenericLogicalPred {
    enum class Op { And, Or, Not };
    Op op = Op::And;
    std::vector<GenericPredicatePtr> children;
};

struct GenericExistsPred {
    bool negated = false;
    int subqueryIndex = -1;
};

struct GenericPredicate {
    std::variant<GenericComparisonPred,
                 GenericBetweenPred,
                 GenericInListPred,
                 GenericLikePred,
                 GenericLogicalPred,
                 GenericExistsPred> node;
};

enum class GenericRelOp {
    Scan,
    Filter,
    Project,
    Join,
    SemiJoin,
    AntiJoin,
    Aggregate,
    Sort,
    Limit,
    Materialize
};

enum class GenericJoinKind {
    Inner,
    LeftOuter,
    Semi,
    Anti
};

struct GenericProjection {
    std::string name;
    GenericExprPtr expr;
    TypeInfo type;
};

struct GenericSortKey {
    GenericExprPtr expr;
    bool descending = false;
};

struct GenericScanDetail {
    GenericRelationInstanceId relationInstance;
    std::string table;
    std::string alias;
};

struct GenericFilterDetail {
    GenericPredicatePtr predicate;
};

struct GenericProjectDetail {
    std::vector<GenericProjection> projections;
};

struct GenericJoinDetail {
    GenericJoinKind kind = GenericJoinKind::Inner;
    GenericPredicatePtr predicate;
};

struct GenericAggregateDetail {
    std::vector<GenericExprPtr> groupBy;
    std::vector<std::string> groupNames;
    std::vector<GenericProjection> aggregates;
    std::vector<std::string> aggregateOutputFuncs;
    std::vector<std::string> outputOrder;
    GenericPredicatePtr having;
};

struct GenericSortDetail {
    std::vector<GenericSortKey> keys;
};

struct GenericLimitDetail {
    int limit = -1;
};

struct GenericMaterializeDetail {
    std::string outputName;
};

struct GenericInSubqueryHaving {
    CmpOp op = CmpOp::GT;
    double literal = 0.0;
};

struct GenericScalarSubqueryAggTarget {
    AggFunc func = AggFunc::SUM;
    bool star = false;
    double multiplier = 1.0;
    std::string argSignature;
};

struct GenericScalarHavingSubquerySummary {
    GenericScalarSubqueryAggTarget aggregate;
    std::vector<std::string> tables;
    std::vector<std::string> predicateSignatures;
};

struct GenericFromSubqueryScalarExtremum {
    std::string sourceAlias;
    AggFunc func = AggFunc::MAX;
    std::string argAlias;
};

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
    std::map<std::string, std::vector<GenericPredicatePtr>> filtersByTable;
};

struct GenericSourceSubquery {
    enum Type { IN_SUBQUERY, EXISTS_SUBQUERY, NOT_EXISTS_SUBQUERY, SCALAR_SUBQUERY };
    Type type = SCALAR_SUBQUERY;
    std::optional<GenericScalarHavingSubquerySummary> scalarHavingSummary;
    std::vector<GenericFromSubqueryScalarExtremum> fromSubqueryScalarExtrema;
    std::optional<DecorrelatedScalarSubquery> decorrelatedScalar;
};

struct GenericInSubqueryAggInfo {
    std::string alias;        // Duplicate table alias.
    std::string baseTable;    // Base table name.
    int tableIndex = -1;      // Source table index from SQL analysis.
    std::string groupCol;     // GROUP BY column.
    std::string aggFunc;      // SUM, COUNT, AVG, MIN, or MAX.
    std::string aggExpr;      // Aggregate input column.
    bool hasHavingPred = false;
    std::optional<GenericInSubqueryHaving> having;
};

struct GenericFromSubqueryJoin {
    std::string leftTable;
    std::string rightTable;
    std::string leftCol;
    std::string rightCol;
    bool leftOuter = false;
};

struct GenericFromSubqueryAggTarget {
    std::string name;
    AggFunc func = AggFunc::COUNT;
    GenericExprPtr arg;
    bool star = false;
    TypeInfo type;
};

struct GenericFromSubqueryAggInfo {
    std::string alias;
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;
    std::vector<GenericFromSubqueryJoin> joins;
    std::vector<GenericPredicatePtr> filters;
    std::vector<GenericFromSubqueryAggTarget> aggregates;
    std::vector<GenericExprPtr> groupBy;
};

struct GenericSourceQueryInfo {
    std::vector<GenericInSubqueryAggInfo> inSubAggs;
    std::vector<GenericFromSubqueryAggInfo> fromSubqueryAggs;
    std::vector<GenericSourceSubquery> subqueries;
};

struct GenericRelNode {
    GenericNodeId id;
    GenericRelOp op = GenericRelOp::Scan;
    std::vector<GenericNodeId> inputs;
    GenericOutputSchema output;
    std::variant<GenericScanDetail,
                 GenericFilterDetail,
                 GenericProjectDetail,
                 GenericJoinDetail,
                 GenericAggregateDetail,
                 GenericSortDetail,
                 GenericLimitDetail,
                 GenericMaterializeDetail> detail;
};

struct GenericRelPlan {
    std::vector<GenericRelation> relations;
    std::vector<GenericRelationInstance> relationInstances;
    std::vector<GenericRelNode> nodes;
    GenericNodeId root;
    const SchemaProvider* schema = nullptr;
    GenericSourceQueryInfo source;

    const GenericRelation* findRelation(GenericRelationId id) const;
    const GenericRelationInstance* findRelationInstance(GenericRelationInstanceId id) const;
    const GenericRelNode* findNode(GenericNodeId id) const;
    nlohmann::json toJSON() const;
};

class GenericRelPlanBuilder {
public:
    GenericRelationId addRelation(const std::string& name,
                                  bool virtualRelation = false,
                                  std::string maxKeySymbol = {},
                                  std::string primaryKeyColumn = {},
                                  std::string primaryKeyDomainSymbol = {},
                                  int probePriority = 0);
    GenericRelationInstanceId addRelationInstance(GenericRelationId relation,
                                                  const std::string& baseName,
                                                  const std::string& alias);
    GenericNodeId addNode(GenericRelOp op,
                          std::vector<GenericNodeId> inputs,
                          GenericOutputSchema output,
                          std::variant<GenericScanDetail,
                                       GenericFilterDetail,
                                       GenericProjectDetail,
                                       GenericJoinDetail,
                                       GenericAggregateDetail,
                                       GenericSortDetail,
                                       GenericLimitDetail,
                                       GenericMaterializeDetail> detail);
    void setSchema(const SchemaProvider* schema);
    void setSourceQuery(GenericSourceQueryInfo source);
    GenericExprId nextExprId();
    GenericColumnId nextColumnId();
    GenericRelPlan finish(GenericNodeId root);

private:
    GenericRelPlan plan_;
    int nextExprId_ = 0;
    int nextColumnId_ = 0;
};

std::string dataTypeName(DataType type);
std::string exprOpName(ExprOp op);
std::string aggFuncName(AggFunc func);
std::string cmpOpName(CmpOp op);
std::string genericRelOpName(GenericRelOp op);
std::string genericJoinKindName(GenericJoinKind kind);

nlohmann::json genericExprToJSON(const GenericExprPtr& expr);
nlohmann::json genericPredicateToJSON(const GenericPredicatePtr& pred);
nlohmann::json genericSchemaToJSON(const GenericOutputSchema& schema);
nlohmann::json genericNodeToJSON(const GenericRelNode& node);

} // namespace codegen
