#include "generic/ir/analyzed_query.h"
#include "generic/ir/analyzed_query_context.h"
#include "metal_plan_common.h"

extern "C" {
#include "pg_query.h"
}
#include "../../../../third_party/nlohmann/json.hpp"

#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <set>
#include <unordered_map>

using json = nlohmann::json;

namespace codegen {

// --- Internal Helpers ---

namespace {

using analyzed_query_internal::AnalyzeContext;
using analyzed_query_internal::AnalyzeScope;
using analyzed_query_internal::NameResolver;
using analyzed_query_internal::ScalarSubqueryInfo;
using analyzed_query_internal::rebuildAliasMap;

std::string astSnippet(const json& node, std::size_t limit = 200) {
    std::string dump = node.dump();
    if (dump.size() > limit) {
        dump.resize(limit);
        dump += "...";
    }
    return dump;
}

[[noreturn]] void unsupportedSql(const std::string& context, const json& node) {
    throw std::runtime_error("Unsupported SQL " + context + ": " + astSnippet(node));
}

bool isInlinedSubqueryPlaceholder(const PredPtr& pred) {
    if (!pred) return false;
    if (std::holds_alternative<ExistsPred>(pred->node)) return true;
    if (auto* notPred = std::get_if<LogicalNot>(&pred->node)) {
        if (!notPred->child) return false;
        if (std::holds_alternative<ExistsPred>(notPred->child->node))
            return true;
        if (auto* inList = std::get_if<InList>(&notPred->child->node))
            return inList->values.empty();
        return false;
    }
    if (auto* inList = std::get_if<InList>(&pred->node))
        return inList->values.empty();
    return false;
}

} // anonymous namespace

// Parse YYYY-MM-DD into YYYYMMDD.
int parseDateLiteral(const std::string& s) {
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        int y = std::stoi(s.substr(0, 4));
        int m = std::stoi(s.substr(5, 2));
        int d = std::stoi(s.substr(8, 2));
        return y * 10000 + m * 100 + d;
    }
    throw std::runtime_error("Invalid date literal: " + s);
}

// --- AST Walking ---

// Forward declarations
ExprPtr walkExpr(AnalyzeContext& ctx, const json& node,
                 const AnalyzeScope& scope);
PredPtr walkPredicate(AnalyzeContext& ctx, const json& node,
                      const AnalyzeScope& scope);

ExprPtr walkColumnRef(AnalyzeContext& ctx, const json& node,
                      const AnalyzeScope& scope) {
    auto& fields = node["fields"];
    std::string colName;
    std::string tblQualifier;
    if (fields.size() == 1) {
        colName = fields[0]["String"]["sval"].get<std::string>();
    } else if (fields.size() == 2) {
        tblQualifier = fields[0]["String"]["sval"].get<std::string>();
        colName = fields[1]["String"]["sval"].get<std::string>();
    }
    NameResolver resolver(ctx, scope);
    auto resolved = resolver.resolveColumn(colName, tblQualifier);
    if (!resolved) {
        // FROM-subquery SELECT-list aliases.
        auto sqit = ctx.subqueryAliasMap.find(colName);
        if (sqit != ctx.subqueryAliasMap.end())
            return Expr::col(sqit->second.table, sqit->second.column,
                             sqit->second.colIndex, sqit->second.dataType,
                             sqit->second.fixedWidth, sqit->second.tableAlias);
        // Computed aliases visible to WHERE/HAVING.
        auto eqit = ctx.subqueryExprMap.find(colName);
        if (eqit != ctx.subqueryExprMap.end()) return eqit->second;
        // Preserve unresolved SELECT aliases and derived columns.
        return Expr::col("", colName, -1, DataType::INT);
    }

    return Expr::col(resolved->table, resolved->column, -1, resolved->type,
                     resolved->fixedWidth, resolved->tableAlias);
}

ExprPtr walkConst(const json& node) {
    if (node.contains("ival")) {
        auto& iv = node["ival"];
        if (iv.is_object() && iv.contains("ival"))
            return Expr::lit(iv["ival"].get<int>());
        return Expr::lit(0); // empty ival = integer 0
    }
    if (node.contains("fval")) {
        auto& fv = node["fval"];
        if (fv.is_object() && fv.contains("fval"))
            return Expr::litf(std::stof(fv["fval"].get<std::string>()));
        return Expr::litf(0.0f);
    }
    if (node.contains("sval")) {
        auto& sv = node["sval"];
        if (sv.is_object() && sv.contains("sval"))
            return Expr::lits(sv["sval"].get<std::string>());
        return Expr::lits("");
    }
    unsupportedSql("constant", node);
}

ExprPtr walkTypeCast(AnalyzeContext& ctx, const json& node,
                     const AnalyzeScope& scope) {
    auto& typeName = node["typeName"];
    std::string typStr;
    if (typeName.contains("names")) {
        for (auto& n : typeName["names"]) {
            if (n.contains("String"))
                typStr = n["String"]["sval"].get<std::string>();
        }
    }

    auto arg = walkExpr(ctx, node["arg"], scope);
    // DATE casts normalize string literals to YYYYMMDD.
    if (typStr == "date") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) {
                return Expr::lit(parseDateLiteral(*sv));
            }
        }
    }
    // INTERVAL casts keep the raw value; walkAExpr reads typmods for the unit.
    if (typStr == "interval") {
        int intervalValue = 0;
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* iv = std::get_if<int>(&lit->value))
                intervalValue = *iv;
            else if (auto* sv = std::get_if<std::string>(&lit->value))
                intervalValue = std::stoi(*sv);
        }
        return Expr::lit(intervalValue);
    }
    return arg;
}

// Apply day offsets while preserving YYYYMMDD encoding.
static int computeDateArithDays(int yyyymmdd, int days, bool isAdd) {
    int dir = isAdd ? 1 : -1;
    int y = yyyymmdd / 10000;
    int m = (yyyymmdd / 100) % 100;
    int d = yyyymmdd % 100;

    auto isLeap = [](int yr) { return (yr % 4 == 0 && yr % 100 != 0) || yr % 400 == 0; };
    auto daysInMonth = [&](int yr, int mo) -> int {
        static const int dim[] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
        if (mo == 2 && isLeap(yr)) return 29;
        return dim[mo];
    };

    d += dir * days;
    while (d > daysInMonth(y, m)) {
        d -= daysInMonth(y, m);
        m++;
        if (m > 12) { m = 1; y++; }
    }
    while (d < 1) {
        m--;
        if (m < 1) { m = 12; y--; }
        d += daysInMonth(y, m);
    }

    return y * 10000 + m * 100 + d;
}

enum class IntervalUnit { UNKNOWN, YEAR, MONTH, DAY };

// Extract interval unit from TypeCast typmods.
static IntervalUnit extractIntervalUnit(const json& typeCastNode) {
    if (!typeCastNode.contains("typeName")) return IntervalUnit::UNKNOWN;
    auto& tn = typeCastNode["typeName"];
    bool isInterval = false;
    if (tn.contains("names")) {
        for (auto& n : tn["names"]) {
            if (n.contains("String") && n["String"]["sval"] == "interval")
                isInterval = true;
        }
    }
    if (!isInterval) return IntervalUnit::UNKNOWN;

    int typmods = 0;
    if (tn.contains("typmods")) {
        for (auto& tm : tn["typmods"]) {
            if (tm.contains("Integer"))
                typmods = tm["Integer"]["ival"].get<int>();
            else if (tm.contains("A_Const") && tm["A_Const"].contains("ival"))
                typmods = tm["A_Const"]["ival"]["ival"].get<int>();
        }
    }
    // PostgreSQL interval mask: YEAR=4, MONTH=2, DAY=8.
    if (typmods & 4) return IntervalUnit::YEAR;
    if (typmods & 2) return IntervalUnit::MONTH;
    if (typmods & 8) return IntervalUnit::DAY;
    return IntervalUnit::UNKNOWN;
}

// Compute DATE +/- INTERVAL for YEAR, MONTH, and DAY units.
static int computeDateArith(int yyyymmdd, int intervalVal, bool isAdd, IntervalUnit unit) {
    int dir = isAdd ? 1 : -1;
    int y = yyyymmdd / 10000;
    int m = (yyyymmdd / 100) % 100;
    int d = yyyymmdd % 100;

    switch (unit) {
        case IntervalUnit::YEAR:
            y += dir * intervalVal;
            return y * 10000 + m * 100 + d;
        case IntervalUnit::MONTH: {
            m += dir * intervalVal;
            while (m > 12) { m -= 12; y++; }
            while (m < 1)  { m += 12; y--; }
            return y * 10000 + m * 100 + d;
        }
        case IntervalUnit::DAY:
        default:
            return computeDateArithDays(yyyymmdd, intervalVal, isAdd);
    }
}

ExprPtr walkFuncCall(AnalyzeContext& ctx, const json& node,
                     const AnalyzeScope& scope) {
    std::string funcName;
    if (node.contains("funcname")) {
        for (auto& n : node["funcname"]) {
            if (n.contains("String"))
                funcName = n["String"]["sval"].get<std::string>();
        }
    }
    std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);

    FuncCall fc;
    fc.name = funcName;
    if (node.contains("args")) {
        for (auto& a : node["args"])
            fc.args.push_back(walkExpr(ctx, a, scope));
    }

    auto e = std::make_shared<Expr>();
    e->node = fc;
    return e;
}

ExprPtr walkAExpr(AnalyzeContext& ctx, const json& node,
                  const AnalyzeScope& scope) {
    std::string kind = node.value("kind", "AEXPR_OP");
    std::string opName;
    if (node.contains("name")) {
        for (auto& n : node["name"]) {
            if (n.contains("String"))
                opName = n["String"]["sval"].get<std::string>();
        }
    }

    if (kind == "AEXPR_OP") {
        // Arithmetic operators in expression context
        ExprOp exOp;
        if      (opName == "+") exOp = ExprOp::ADD;
        else if (opName == "-") exOp = ExprOp::SUB;
        else if (opName == "*") exOp = ExprOp::MUL;
        else if (opName == "/") exOp = ExprOp::DIV;
        else {
            unsupportedSql("expression operator '" + opName + "'", node);
        }
        auto left = node.contains("lexpr") ? walkExpr(ctx, node["lexpr"], scope) : Expr::lit(0);
        if (!node.contains("rexpr"))
            unsupportedSql("expression missing right operand", node);
        auto right = walkExpr(ctx, node["rexpr"], scope);

        // Fold DATE +/- INTERVAL when both sides are literals.
        if (exOp == ExprOp::ADD || exOp == ExprOp::SUB) {
            auto* litL = std::get_if<Literal>(&left->node);
            auto* litR = std::get_if<Literal>(&right->node);
            if (litL && litR) {
                auto* dateVal = std::get_if<int>(&litL->value);
                auto* intVal  = std::get_if<int>(&litR->value);
                if (dateVal && intVal && *dateVal > 19000101 && *dateVal < 21001231) {
                    bool isAdd = (exOp == ExprOp::ADD);
                    IntervalUnit unit = IntervalUnit::DAY;
                    if (node.contains("rexpr") && node["rexpr"].contains("TypeCast"))
                        unit = extractIntervalUnit(node["rexpr"]["TypeCast"]);
                    int result = computeDateArith(*dateVal, *intVal, isAdd, unit);
                    return Expr::lit(result);
                }
            }
        }

        return Expr::binary(exOp, left, right);
    }
    unsupportedSql("expression kind '" + kind + "'", node);
}

ExprPtr walkExpr(AnalyzeContext& ctx, const json& node,
                 const AnalyzeScope& scope) {
    if (node.contains("ColumnRef"))
        return walkColumnRef(ctx, node["ColumnRef"], scope);
    if (node.contains("fields")) {
        // Bare ColumnRef without outer "ColumnRef" key (e.g., GROUP BY items)
        return walkColumnRef(ctx, node, scope);
    }
    if (node.contains("A_Const"))
        return walkConst(node["A_Const"]);
    if (node.contains("TypeCast"))
        return walkTypeCast(ctx, node["TypeCast"], scope);
    if (node.contains("FuncCall")) {
        return walkFuncCall(ctx, node["FuncCall"], scope);
    }
    if (node.contains("A_Expr"))
        return walkAExpr(ctx, node["A_Expr"], scope);
    if (node.contains("SubLink")) {
        // Scalar subqueries are materialized later.
        auto& sl = node["SubLink"];
        std::string subType = sl.value("subLinkType", "EXISTS_SUBLINK");
        if (subType == "EXPR_SUBLINK" && sl.contains("subselect")) {
            int idx = (int)ctx.scalarSubqueries.size();
            ScalarSubqueryInfo sq;
            sq.sql = sl["subselect"].dump();
            ctx.scalarSubqueries.push_back(sq);
            return Expr::scalarSubquery(idx);
        }
        unsupportedSql("scalar subquery type '" + subType + "'", node);
    }
    if (node.contains("BoolExpr")) {
        unsupportedSql("boolean expression used as scalar expression", node);
    }
    if (node.contains("CaseExpr")) {
        auto& ce = node["CaseExpr"];
        CaseWhen cw;
        if (ce.contains("args")) {
            for (auto& when : ce["args"]) {
                if (when.contains("CaseWhen")) {
                    auto& caseWhen = when["CaseWhen"];
                    CaseWhen::Branch br;
                    br.condition = walkPredicate(ctx, caseWhen["expr"], scope);
                    br.result = walkExpr(ctx, caseWhen["result"], scope);
                    cw.branches.push_back(std::move(br));
                }
            }
        }
        if (ce.contains("defresult"))
            cw.elseResult = walkExpr(ctx, ce["defresult"], scope);
        auto e = std::make_shared<Expr>();
        e->node = std::move(cw);
        return e;
    }
    unsupportedSql("expression node", node);
}

// --- Predicate Walking ---

CmpOp parseCmpOp(const std::string& op) {
    if (op == "=")  return CmpOp::EQ;
    if (op == "<>") return CmpOp::NE;
    if (op == "!=") return CmpOp::NE;
    if (op == "<")  return CmpOp::LT;
    if (op == "<=") return CmpOp::LE;
    if (op == ">")  return CmpOp::GT;
    if (op == ">=") return CmpOp::GE;
    throw std::runtime_error("Unknown CmpOp: " + op);
}

PredPtr walkAExprPred(AnalyzeContext& ctx, const json& node,
                      const AnalyzeScope& scope) {
    std::string kind = node.value("kind", "AEXPR_OP");
    std::string opName;
    if (node.contains("name")) {
        for (auto& n : node["name"])
            if (n.contains("String"))
                opName = n["String"]["sval"].get<std::string>();
    }

    if (kind == "AEXPR_OP") {
        auto left = walkExpr(ctx, node["lexpr"], scope);
        auto right = walkExpr(ctx, node["rexpr"], scope);
        return Predicate::cmp(parseCmpOp(opName), left, right);
    }
    if (kind == "AEXPR_BETWEEN" || kind == "AEXPR_NOT_BETWEEN") {
        auto expr = walkExpr(ctx, node["lexpr"], scope);
        auto& list = node["rexpr"]["List"]["items"];
        auto lo = walkExpr(ctx, list[0], scope);
        auto hi = walkExpr(ctx, list[1], scope);
        auto p = Predicate::between(expr, lo, hi);
        if (kind == "AEXPR_NOT_BETWEEN")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_IN") {
        auto expr = walkExpr(ctx, node["lexpr"], scope);
        std::vector<ExprPtr> vals;
        if (node["rexpr"].contains("List")) {
            for (auto& item : node["rexpr"]["List"]["items"])
                vals.push_back(walkExpr(ctx, item, scope));
        }
        auto p = Predicate::inList(expr, std::move(vals));
        if (opName == "<>")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
        auto expr = walkExpr(ctx, node["lexpr"], scope);
        auto patExpr = walkExpr(ctx, node["rexpr"], scope);
        std::string pat;
        if (auto* lit = std::get_if<Literal>(&patExpr->node))
            if (auto* sv = std::get_if<std::string>(&lit->value))
                pat = *sv;
        bool negated = (opName == "!~~" || opName == "!~~*");
        return Predicate::like(expr, pat, negated);
    }
    if (kind == "AEXPR_NOT_DISTINCT") {
        auto left = walkExpr(ctx, node["lexpr"], scope);
        auto right = walkExpr(ctx, node["rexpr"], scope);
        return Predicate::cmp(CmpOp::EQ, left, right);
    }

    unsupportedSql("predicate expression kind '" + kind + "'", node);
}

PredPtr walkPredicate(AnalyzeContext& ctx, const json& node,
                      const AnalyzeScope& scope) {
    if (node.contains("BoolExpr")) {
        auto& be = node["BoolExpr"];
        std::string boolop = be["boolop"].get<std::string>();
        if (boolop == "AND_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(ctx, arg, scope));
            return Predicate::logAnd(std::move(children));
        }
        if (boolop == "OR_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(ctx, arg, scope));
            return Predicate::logOr(std::move(children));
        }
        if (boolop == "NOT_EXPR") {
            return Predicate::logNot(walkPredicate(ctx, be["args"][0], scope));
        }
    }
    if (node.contains("A_Expr")) {
        return walkAExprPred(ctx, node["A_Expr"], scope);
    }
    if (node.contains("NullTest")) {
        unsupportedSql("NULL predicate", node);
    }
    if (node.contains("SubLink")) {
        // Subquery predicates are expanded into joins after WHERE extraction.
        auto& sl = node["SubLink"];
        std::string subType = sl.value("subLinkType", "EXISTS_SUBLINK");
        bool isExists = (subType == "EXISTS_SUBLINK");
        if (isExists || subType == "ALL_SUBLINK" || subType == "ANY_SUBLINK") {
            if (isExists) {
                auto p = std::make_shared<Predicate>();
                p->node = ExistsPred{false, -1};
                return p;
            }
            if (subType == "ALL_SUBLINK" || subType == "ANY_SUBLINK") {
                if (sl.contains("testexpr")) {
                    auto expr = walkExpr(ctx, sl["testexpr"], scope);
                    return Predicate::inList(expr, {}); // Marker for expanded ANY_SUBLINK.
                }
                unsupportedSql("subquery predicate missing test expression", node);
            }
        }
        unsupportedSql("subquery predicate type '" + subType + "'", node);
    }
    throw std::runtime_error("Unknown predicate node: " + node.dump().substr(0, 100));
}

// --- FROM Clause Extraction ---

void extractTables(AnalyzeContext& ctx, const json& fromItem,
                   std::vector<std::string>& tables,
                   std::vector<std::string>& aliases) {
    if (fromItem.contains("RangeVar")) {
        auto& rv = fromItem["RangeVar"];
        std::string name = rv["relname"].get<std::string>();
        // Inline CREATE VIEW definitions by exposing their base tables.
        auto vit = ctx.views.find(name);
        if (vit != ctx.views.end()) {
            auto& [viewBody, viewCols] = vit->second;
            if (viewBody.contains("fromClause")) {
                for (auto& item : viewBody["fromClause"]) {
                    // Keep scalar subquery view expansion from duplicating base tables.
                    std::vector<std::string> newTables, newAliases;
                    extractTables(ctx, item, newTables, newAliases);
                    for (size_t ni = 0; ni < newTables.size(); ++ni) {
                        if (std::find(tables.begin(), tables.end(), newTables[ni]) == tables.end()) {
                            tables.push_back(newTables[ni]);
                            if (ni < newAliases.size()) aliases.push_back(newAliases[ni]);
                            else aliases.push_back(newTables[ni]);
                        }
                    }
                }
            }
            return;
        }
        tables.push_back(name);
        if (rv.contains("alias")) {
            auto& a = rv["alias"];
            if (a.contains("Alias"))
                aliases.push_back(a["Alias"]["aliasname"].get<std::string>());
            else if (a.contains("aliasname"))
                aliases.push_back(a["aliasname"].get<std::string>());
            else
                aliases.push_back(name);
        } else {
            aliases.push_back(name);
        }
    }
    if (fromItem.contains("JoinExpr")) {
        auto& je = fromItem["JoinExpr"];
        extractTables(ctx, je["larg"], tables, aliases);
        extractTables(ctx, je["rarg"], tables, aliases);
    }
    if (fromItem.contains("RangeSubselect")) {
        auto& rs = fromItem["RangeSubselect"];
        if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
            auto& sub = rs["subquery"]["SelectStmt"];
            if (sub.contains("fromClause")) {
                for (auto& item : sub["fromClause"])
                    extractTables(ctx, item, tables, aliases);
                return;
            }
        }

        tables.push_back("__subquery__");
        if (rs.contains("alias")) {
            auto& a = rs["alias"];
            if (a.contains("Alias"))
                aliases.push_back(a["Alias"]["aliasname"].get<std::string>());
            else if (a.contains("aliasname"))
                aliases.push_back(a["aliasname"].get<std::string>());
            else
                aliases.push_back("__subquery__");
        } else {
            aliases.push_back("__subquery__");
        }
    }
}

// --- Join Extraction ---

// Column equality predicates become join clauses.
bool isJoinCondition(const PredPtr& pred, JoinClause& jc) {
    auto* cmp = std::get_if<Comparison>(&pred->node);
    if (!cmp || cmp->op != CmpOp::EQ) return false;

    auto* leftCol = std::get_if<ColRef>(&cmp->left->node);
    auto* rightCol = std::get_if<ColRef>(&cmp->right->node);
    if (!leftCol || !rightCol) return false;
    // Self-joins rely on aliases for instance disambiguation.

    jc.leftTable = leftCol->tableAlias.empty() ? leftCol->table : leftCol->tableAlias;
    jc.leftCol = leftCol->column;
    jc.rightTable = rightCol->tableAlias.empty() ? rightCol->table : rightCol->tableAlias;
    jc.rightCol = rightCol->column;
    return true;
}

// Flatten AND predicates and split joins from filters.
void separatePredicates(const PredPtr& pred, const std::vector<std::string>& tables,
                        std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (auto* la = std::get_if<LogicalAnd>(&pred->node)) {
        for (auto& child : la->children)
            separatePredicates(child, tables, joins, filters);
        return;
    }
    if (auto* lo = std::get_if<LogicalOr>(&pred->node)) {
        // Only joins present in every OR branch can be hoisted.
        std::vector<JoinClause> commonJoins;
        std::vector<PredPtr> strippedBranches;
        bool first = true;
        for (auto& child : lo->children) {
            std::vector<JoinClause> branchJoins;
            std::vector<PredPtr> branchFilters;
            separatePredicates(child, tables, branchJoins, branchFilters);
            // Rebuild each branch after removing hoisted joins.
            if (!branchFilters.empty()) {
                if (branchFilters.size() == 1)
                    strippedBranches.push_back(branchFilters[0]);
                else
                    strippedBranches.push_back(Predicate::logAnd(branchFilters));
            }
            if (first) {
                commonJoins = std::move(branchJoins);
                first = false;
            } else {
                commonJoins.erase(
                    std::remove_if(commonJoins.begin(), commonJoins.end(),
                        [&](const JoinClause& jc) {
                            for (auto& bj : branchJoins)
                                if (jc.leftTable == bj.leftTable && jc.rightTable == bj.rightTable &&
                                    jc.leftCol == bj.leftCol && jc.rightCol == bj.rightCol)
                                    return false;
                            return true;
                        }),
                    commonJoins.end());
            }
        }
        for (auto& jc : commonJoins) joins.push_back(jc);
        if (!strippedBranches.empty()) {
            if (strippedBranches.size() == 1)
                filters.push_back(strippedBranches[0]);
            else
                filters.push_back(Predicate::logOr(strippedBranches));
        }
        return;
    }
    JoinClause jc;
    if (isJoinCondition(pred, jc))
        joins.push_back(jc);
    else
        filters.push_back(pred);
}

// --- Target Extraction ---

AggFunc parseAggFunc(const std::string& name) {
    if (name == "sum")   return AggFunc::SUM;
    if (name == "count") return AggFunc::COUNT;
    if (name == "avg")   return AggFunc::AVG;
    if (name == "min")   return AggFunc::MIN;
    if (name == "max")   return AggFunc::MAX;
    throw std::runtime_error("Unknown aggregate: " + name);
}

// Recursively detect aggregate functions in a target expression.
static bool containsAggregateJson(const json& node) {
    if (node.contains("FuncCall")) {
        auto& fc = node["FuncCall"];
        std::string fn;
        if (fc.contains("funcname"))
            for (auto& n : fc["funcname"])
                if (n.contains("String")) fn = n["String"]["sval"].get<std::string>();
        std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);
        if (fn == "sum" || fn == "count" || fn == "avg" || fn == "min" || fn == "max")
            return true;
    }
    if (node.contains("A_Expr")) {
        auto& ae = node["A_Expr"];
        return containsAggregateJson(ae.value("lexpr", json{})) ||
               containsAggregateJson(ae.value("rexpr", json{}));
    }
    if (node.contains("CaseExpr")) {
        auto& ce = node["CaseExpr"];
        if (ce.contains("args"))
            for (auto& a : ce["args"])
                if (a.contains("CaseWhen")) {
                    auto& cw = a["CaseWhen"];
                    if (containsAggregateJson(cw.value("expr", json{}))) return true;
                    if (containsAggregateJson(cw.value("result", json{}))) return true;
                }
        if (containsAggregateJson(ce.value("defresult", json{}))) return true;
    }
    if (node.contains("TypeCast"))
        return containsAggregateJson(node["TypeCast"].value("arg", json{}));
    return false;
}

SelectTarget extractTarget(AnalyzeContext& ctx, const json& resTarget,
                           const AnalyzeScope& scope) {
    SelectTarget st;
    st.alias = resTarget.value("name", "");
    auto& val = resTarget["val"];
    auto sqlVisibleColumnName = [](const json& expr) -> std::string {
        if (!expr.contains("ColumnRef")) return "";
        const auto& cr = expr["ColumnRef"];
        if (!cr.contains("fields") || !cr["fields"].is_array() || cr["fields"].empty())
            return "";
        const auto& field = cr["fields"].back();
        if (field.contains("String") && field["String"].contains("sval"))
            return field["String"]["sval"].get<std::string>();
        if (field.contains("sval"))
            return field["sval"].get<std::string>();
        return "";
    };
    if (st.alias.empty()) {
        // Preserve display names before subquery aliases are resolved.
        st.alias = sqlVisibleColumnName(val);
    }
    if (val.contains("FuncCall")) {
        auto& fc = val["FuncCall"];
        std::string funcName;
        if (fc.contains("funcname"))
            for (auto& n : fc["funcname"])
                if (n.contains("String"))
                    funcName = n["String"]["sval"].get<std::string>();
        std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);

        // Top-level aggregate target.
        if (funcName == "sum" || funcName == "count" || funcName == "avg" ||
            funcName == "min" || funcName == "max") {
            st.isAgg = true;
            AggTarget at;
            at.func = parseAggFunc(funcName);
            at.alias = st.alias;
            if (fc.contains("agg_star") && fc["agg_star"].get<bool>()) {
                at.isStar = true;
            } else if (fc.contains("args") && !fc["args"].empty()) {
                at.innerExpr = walkExpr(ctx, fc["args"][0], scope);
            }
            if (fc.contains("agg_distinct") && fc["agg_distinct"].get<bool>()) {
                at.func = AggFunc::COUNT_DISTINCT;
            }
            st.agg = at;
            st.expr = walkExpr(ctx, val, scope); // Full FuncCall as expr
        } else {
            st.expr = walkExpr(ctx, val, scope);
        }
    } else {
        // Non-FuncCall targets may still contain nested aggregates.
        st.expr = walkExpr(ctx, val, scope);
        if (containsAggregateJson(val)) {
            st.isAgg = true;
            AggTarget at;
            at.alias = st.alias;
            at.isStar = false;
            // Extract the first aggregate call inside a computed target.
            std::function<bool(const ExprPtr&)> tryExtract;
            tryExtract = [&](const ExprPtr& e) -> bool {
                if (!e) return false;
                if (auto* fc = std::get_if<FuncCall>(&e->node)) {
                    if (fc->name == "sum" || fc->name == "count" || fc->name == "avg" || fc->name == "min" || fc->name == "max") {
                        if (fc->name == "sum") at.func = AggFunc::SUM;
                        else if (fc->name == "count") at.func = AggFunc::COUNT;
                        else if (fc->name == "avg") at.func = AggFunc::AVG;
                        else if (fc->name == "min") at.func = AggFunc::MIN;
                        else at.func = AggFunc::MAX;
                        if (!fc->args.empty()) at.innerExpr = fc->args[0];
                        return true;
                    }
                }
                if (auto* be = std::get_if<BinaryExpr>(&e->node))
                    return tryExtract(be->left) || tryExtract(be->right);
                return false;
            };
            tryExtract(st.expr);
            st.agg = at;
        }
    }
    return st;
}

bool resTargetIsStar(const json& resTarget) {
    if (!resTarget.contains("val")) return false;
    const auto& val = resTarget["val"];
    if (!val.contains("ColumnRef")) return false;
    const auto& cr = val["ColumnRef"];
    if (!cr.contains("fields") || !cr["fields"].is_array() || cr["fields"].empty())
        return false;
    return cr["fields"].back().contains("A_Star");
}

std::string starQualifier(const json& resTarget) {
    if (!resTargetIsStar(resTarget)) return "";
    const auto& fields = resTarget["val"]["ColumnRef"]["fields"];
    if (fields.size() < 2) return "";
    const auto& first = fields.front();
    if (first.contains("String") && first["String"].contains("sval"))
        return first["String"]["sval"].get<std::string>();
    if (first.contains("sval"))
        return first["sval"].get<std::string>();
    return "";
}

void appendStarTargets(AnalyzeContext& ctx, const json& resTarget,
                       AnalyzedQuery& aq, const AnalyzeScope& scope) {
    const std::string qualifier = starQualifier(resTarget);
    bool matched = false;
    for (size_t i = 0; i < scope.tables.size(); ++i) {
        const std::string& table = scope.tables[i];
        std::string alias = table;
        if (i < scope.aliases.size() && !scope.aliases[i].empty())
            alias = scope.aliases[i];
        if (!qualifier.empty() && qualifier != table && qualifier != alias)
            continue;

        auto* tdef = ctx.catalog.findTable(table);
        if (!tdef) continue;
        for (const auto& col : tdef->columns) {
            SelectTarget st;
            st.alias = col.name;
            st.expr = Expr::col(table, col.name, -1, col.type,
                                col.fixedWidth, alias == table ? "" : alias);
            aq.targets.push_back(std::move(st));
        }
        matched = true;
    }
    if (!matched) {
        throw std::runtime_error(
            qualifier.empty()
                ? "SELECT * target has no FROM relation to expand"
                : "SELECT target qualifier not found for *: " + qualifier);
    }
}

// --- Explicit JOIN ON Extraction ---

void extractJoinOns(AnalyzeContext& ctx, const json& fromItem,
                    const AnalyzeScope& scope,
                    std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (fromItem.contains("RangeSubselect")) {
        auto& rs = fromItem["RangeSubselect"];
        if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
            auto& sub = rs["subquery"]["SelectStmt"];
            if (sub.contains("fromClause")) {
                for (auto& item : sub["fromClause"])
                    extractJoinOns(ctx, item, scope, joins, filters);
            }
        }
    }
    if (fromItem.contains("JoinExpr")) {
        auto& je = fromItem["JoinExpr"];
        size_t joinCountBefore = joins.size();
        if (je.contains("quals")) {
            auto pred = walkPredicate(ctx, je["quals"], scope);
            separatePredicates(pred, scope.tables, joins, filters);
        }
        // Mark join clauses produced by LEFT OUTER JOIN.
        if (je.contains("jointype")) {
            bool isLeftJoin = false;
            const auto& joinType = je["jointype"];
            if (joinType.is_number_integer()) {
                isLeftJoin = (joinType.get<int>() == 2);
            } else if (joinType.is_string()) {
                isLeftJoin = (joinType.get<std::string>() == "JOIN_LEFT");
            } else {
                unsupportedSql("JOIN jointype", je);
            }
            if (isLeftJoin) {
                for (size_t j = joinCountBefore; j < joins.size(); ++j)
                    joins[j].leftOuter = true;
            }
        }
        extractJoinOns(ctx, je["larg"], scope, joins, filters);
        extractJoinOns(ctx, je["rarg"], scope, joins, filters);
    }
}

std::string rangeSubselectAlias(const json& rs) {
    if (!rs.contains("alias")) return "";
    const auto& a = rs["alias"];
    if (a.contains("Alias"))
        return a["Alias"].value("aliasname", "");
    if (a.contains("aliasname"))
        return a.value("aliasname", "");
    return "";
}

std::optional<FromSubqueryAggInfo> extractGroupedSelectAggInfo(
        AnalyzeContext& ctx,
        const json& sub,
        const std::string& alias,
        const std::vector<std::string>& targetAliases = {}) {
    if (!sub.contains("groupClause") || !sub.contains("fromClause"))
        return std::nullopt;

    FromSubqueryAggInfo info;
    info.alias = alias;

    for (const auto& item : sub["fromClause"])
        extractTables(ctx, item, info.tables, info.tableAliases);
    if (info.tables.empty())
        return std::nullopt;
    AnalyzeScope scope = AnalyzeScope::fromTables(info.tables, info.tableAliases);

    for (const auto& item : sub["fromClause"])
        extractJoinOns(ctx, item, scope, info.joins, info.filters);

    if (sub.contains("whereClause")) {
        auto pred = walkPredicate(ctx, sub["whereClause"], scope);
        separatePredicates(pred, scope.tables, info.joins, info.filters);
    }

    if (sub.contains("targetList")) {
        size_t targetIndex = 0;
        for (const auto& t : sub["targetList"]) {
            if (t.contains("ResTarget")) {
                auto target = extractTarget(ctx, t["ResTarget"], scope);
                if (targetIndex < targetAliases.size() && !targetAliases[targetIndex].empty()) {
                    target.alias = targetAliases[targetIndex];
                    if (target.agg) target.agg->alias = target.alias;
                }
                info.targets.push_back(std::move(target));
                targetIndex++;
            }
        }
    }

    bool hasAggTarget = false;
    for (const auto& target : info.targets) {
        if (target.isAgg) {
            hasAggTarget = true;
            break;
        }
    }
    if (!hasAggTarget)
        return std::nullopt;

    for (const auto& g : sub["groupClause"])
        info.groupBy.push_back(walkExpr(ctx, g, scope));

    return info;
}

std::optional<FromSubqueryAggInfo> extractGroupedFromSubqueryInfo(
        AnalyzeContext& ctx, const json& rs) {
    if (!rs.contains("subquery") || !rs["subquery"].contains("SelectStmt"))
        return std::nullopt;
    return extractGroupedSelectAggInfo(ctx, rs["subquery"]["SelectStmt"],
                                       rangeSubselectAlias(rs));
}

void collectReferencedViews(const AnalyzeContext& ctx, const json& fromItem,
                            std::set<std::string>& views) {
    if (fromItem.contains("RangeVar")) {
        const auto& rv = fromItem["RangeVar"];
        std::string name = rv.value("relname", "");
        if (!name.empty() && ctx.views.find(name) != ctx.views.end())
            views.insert(name);
    }
    if (fromItem.contains("RangeSubselect")) {
        const auto& rs = fromItem["RangeSubselect"];
        if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
            const auto& sub = rs["subquery"]["SelectStmt"];
            if (sub.contains("fromClause"))
                for (const auto& item : sub["fromClause"])
                    collectReferencedViews(ctx, item, views);
        }
    }
    if (fromItem.contains("JoinExpr")) {
        const auto& je = fromItem["JoinExpr"];
        collectReferencedViews(ctx, je["larg"], views);
        collectReferencedViews(ctx, je["rarg"], views);
    }
}

void collectPredicateAliases(const PredPtr& pred, std::set<std::string>& aliases) {
    if (!pred) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            for (auto* expr : {&node.left, &node.right}) {
                if (*expr && std::get_if<ColRef>(&(*expr)->node)) {
                    auto& col = std::get<ColRef>((*expr)->node);
                    if (!col.tableAlias.empty()) aliases.insert(col.tableAlias);
                }
            }
        } else if constexpr (std::is_same_v<T, Between>) {
            for (auto* expr : {&node.expr, &node.low, &node.high}) {
                if (*expr && std::get_if<ColRef>(&(*expr)->node)) {
                    auto& col = std::get<ColRef>((*expr)->node);
                    if (!col.tableAlias.empty()) aliases.insert(col.tableAlias);
                }
            }
        } else if constexpr (std::is_same_v<T, InList>) {
            if (node.expr && std::get_if<ColRef>(&node.expr->node)) {
                auto& col = std::get<ColRef>(node.expr->node);
                if (!col.tableAlias.empty()) aliases.insert(col.tableAlias);
            }
        } else if constexpr (std::is_same_v<T, Like>) {
            if (node.expr && std::get_if<ColRef>(&node.expr->node)) {
                auto& col = std::get<ColRef>(node.expr->node);
                if (!col.tableAlias.empty()) aliases.insert(col.tableAlias);
            }
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (auto& child : node.children) collectPredicateAliases(child, aliases);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            collectPredicateAliases(node.child, aliases);
        }
    }, pred->node);
}

void relocateInnerInstanceFilters(AnalyzedQuery& aq,
                                  size_t tablesBefore,
                                  size_t filterCountBefore) {
    for (size_t ti = tablesBefore; ti < aq.tables.size(); ++ti) {
        const std::string& alias = (ti < aq.tableAliases.size())
            ? aq.tableAliases[ti]
            : aq.tables[ti];
        const std::string& innerTable = aq.tables[ti];
        for (size_t fi = filterCountBefore; fi < aq.filters.size(); ) {
            std::map<std::string, std::string> colToTable;
            collectColumnTables(aq.filters[fi], colToTable);
            if (colToTable.size() == 1 && colToTable.begin()->second == innerTable) {
                std::set<std::string> filterAliases;
                collectPredicateAliases(aq.filters[fi], filterAliases);
                if (filterAliases.size() <= 1) {
                    aq.instanceFilters[alias].push_back(aq.filters[fi]);
                    aq.filters.erase(aq.filters.begin() + fi);
                    continue;
                }
            }
            ++fi;
        }
    }
}

// --- SQL Build-State Collection ---

AnalyzedQuery collectAnalyzedQuery(const std::string& sql, const SchemaProvider& schema) {
    PgQueryParseResult result = pg_query_parse(sql.c_str());
    if (result.error) {
        std::string msg = result.error->message;
        pg_query_free_parse_result(result);
        throw std::runtime_error("SQL parse error: " + msg);
    }

    json ast;
    try {
        ast = json::parse(result.parse_tree);
    } catch (const std::exception& e) {
        pg_query_free_parse_result(result);
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    pg_query_free_parse_result(result);

    AnalyzedQuery aq;
    aq.schema = &schema;
    AnalyzeContext ctx(schema);

    // Collect view definitions and use the last SELECT as the main query.
    json* selPtr = nullptr;
    struct ViewDef { std::string name; json selectBody; std::vector<std::string> cols; };
    std::vector<ViewDef> views;
    for (auto& s : ast["stmts"]) {
        if (s["stmt"].contains("ViewStmt")) {
            auto& vs = s["stmt"]["ViewStmt"];
            ViewDef vd;
            if (vs.contains("view"))
                vd.name = vs["view"].value("relname", "");
            if (vd.name.empty()) continue;
            vd.selectBody = vs["query"]["SelectStmt"];
            if (vs.contains("aliases")) {
                for (auto& a : vs["aliases"]) {
                    if (a.contains("String") && a["String"].contains("sval"))
                        vd.cols.push_back(a["String"]["sval"].get<std::string>());
                    else if (a.contains("aliasname"))
                        vd.cols.push_back(a["aliasname"].get<std::string>());
                }
            }
            ctx.views[vd.name] = {vd.selectBody, vd.cols};
            views.push_back(std::move(vd));
        }
        if (s["stmt"].contains("SelectStmt")) {
            selPtr = &s["stmt"]["SelectStmt"];
        }
    }
    if (!selPtr)
        throw std::runtime_error("Expected SELECT statement");
    auto& sel = *selPtr;

    AnalyzeScope topScope;

    // Extract FROM tables and explicit JOIN ON predicates.
    if (sel.contains("fromClause")) {
        for (auto& item : sel["fromClause"])
            extractTables(ctx, item, aq.tables, aq.tableAliases);
        // Build alias -> base table map.
        rebuildAliasMap(ctx, aq, false);
        topScope = AnalyzeScope::fromAnalyzed(aq);
        for (auto& item : sel["fromClause"])
            extractJoinOns(ctx, item, topScope, aq.joins, aq.filters);

        // Pull filters and aliases out of FROM-subqueries.
	        std::function<void(const json&)> extractSubWhere = [&](const json& fromItem) {
            if (fromItem.contains("RangeSubselect")) {
                auto& rs = fromItem["RangeSubselect"];
                if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
                    auto& sub = rs["subquery"]["SelectStmt"];
                    if (auto groupedInfo = extractGroupedFromSubqueryInfo(ctx, rs))
                        aq.fromSubqueryAggs.push_back(std::move(*groupedInfo));
                    if (sub.contains("whereClause")) {
                        auto pred = walkPredicate(ctx, sub["whereClause"], topScope);
                        separatePredicates(pred, topScope.tables, aq.joins, aq.filters);
                    }
                    // Make FROM-subquery target aliases visible to outer clauses.
                    if (sub.contains("targetList")) {
                        for (auto& t : sub["targetList"]) {
                            if (!t.contains("ResTarget")) continue;
                            auto& rt = t["ResTarget"];
                            std::string alias = rt.value("name", "");
                            if (alias.empty()) continue;
                            if (!rt.contains("val")) continue;
                            auto& val = rt["val"];
                            if (val.contains("ColumnRef")) {
                                auto cre = walkColumnRef(ctx, val["ColumnRef"], topScope);
                                if (auto* cr = cre ? std::get_if<ColRef>(&cre->node) : nullptr) {
                                    ctx.subqueryAliasMap[alias] = *cr;
                                }
                            } else {
                                auto expr = walkExpr(ctx, val, topScope);
                                if (expr) ctx.subqueryExprMap[alias] = expr;
                            }
                        }
                    }
                }
            }
            if (fromItem.contains("JoinExpr")) {
                auto& je = fromItem["JoinExpr"];
                extractSubWhere(je["larg"]);
                extractSubWhere(je["rarg"]);
            }
        };
        for (auto& item : sel["fromClause"])
            extractSubWhere(item);

        std::set<std::string> referencedViews;
        for (auto& item : sel["fromClause"])
            collectReferencedViews(ctx, item, referencedViews);
        for (const auto& viewName : referencedViews) {
            auto vit = ctx.views.find(viewName);
            if (vit == ctx.views.end()) continue;
            auto& [viewBody, viewCols] = vit->second;
            if (auto groupedInfo = extractGroupedSelectAggInfo(ctx, viewBody, viewName, viewCols))
                aq.fromSubqueryAggs.push_back(std::move(*groupedInfo));
        }
    }

    // Process inlined view filters and target aliases.
    for (auto& [name, vp] : ctx.views) {
        auto& [viewBody, viewCols] = vp;
        if (viewBody.contains("whereClause")) {
            auto pred = walkPredicate(ctx, viewBody["whereClause"], topScope);
            separatePredicates(pred, topScope.tables, aq.joins, aq.filters);
        }
        if (viewBody.contains("targetList")) {
            size_t ci = 0;
            for (auto& t : viewBody["targetList"]) {
                if (!t.contains("ResTarget")) continue;
                auto& rt = t["ResTarget"];
                std::string alias;
                if (ci < viewCols.size()) alias = viewCols[ci++];
                else alias = rt.value("name", "");
                if (alias.empty()) continue;
                if (!rt.contains("val")) continue;
                auto expr = walkExpr(ctx, rt["val"], topScope);
                if (!expr) continue;
                if (auto* cr = std::get_if<ColRef>(&expr->node))
                    ctx.subqueryAliasMap[alias] = *cr;
                else
                    ctx.subqueryExprMap[alias] = expr;
            }
        }
    }

    // Extract top-level WHERE predicates.
    if (sel.contains("whereClause")) {
        auto wherePred = walkPredicate(ctx, sel["whereClause"], topScope);
        separatePredicates(wherePred, topScope.tables, aq.joins, aq.filters);
    }

    // Inline EXISTS and IN subqueries into join/filter metadata.
    {
        std::function<void(const json&, bool)> inlineExists = [&](const json& node, bool negated) {
            if (node.is_object() && node.contains("SubLink")) {
                auto& sl = node["SubLink"];
                std::string subType = sl.value("subLinkType", "");
	                if (subType == "EXISTS_SUBLINK" && sl.contains("subselect")) {
	                    auto& sub = sl["subselect"]["SelectStmt"];
	                    // Newly added tables belong to this EXISTS branch.
	                    size_t tablesBefore = aq.tables.size();
	                    if (sub.contains("fromClause")) {
	                        for (auto& item : sub["fromClause"]) {
	                            extractTables(ctx, item, aq.tables, aq.tableAliases);
	                        }
	                        // Inner WHERE resolution needs the updated alias map.
	                        rebuildAliasMap(ctx, aq);
	                    }
	                    AnalyzeScope subqueryScope =
	                        AnalyzeScope::fromSubqueryFirst(aq, tablesBefore);
	                    // Identify inner tables so generated joins can be marked.
	                    std::set<std::string> existsTables;
                    for (size_t ti = tablesBefore; ti < aq.tables.size(); ++ti) {
                        existsTables.insert(aq.tables[ti]);
                        if (ti < aq.tableAliases.size())
                            existsTables.insert(aq.tableAliases[ti]);
                    }
	                    if (sub.contains("whereClause")) {
	                        // Only joins added by this branch get EXISTS semantics.
	                        size_t joinCountBefore = aq.joins.size();
	                        size_t filterCountBefore = aq.filters.size();
	                        auto innerPred = walkPredicate(ctx, sub["whereClause"], subqueryScope);
	                        separatePredicates(innerPred, subqueryScope.tables, aq.joins, aq.filters);
	                        // Keep single-table inner filters scoped to the new alias.
	                        relocateInnerInstanceFilters(aq, tablesBefore, filterCountBefore);
                        if (negated) {
                            for (size_t j = joinCountBefore; j < aq.joins.size(); ++j) {
                                aq.joins[j].anti = true;
                                aq.joins[j].semi = true;
                                if (existsTables.count(aq.joins[j].leftTable))
                                    aq.joins[j].innerTable = aq.joins[j].leftTable;
                                else if (existsTables.count(aq.joins[j].rightTable))
                                    aq.joins[j].innerTable = aq.joins[j].rightTable;
                            }
                        } else {
                            for (size_t j = joinCountBefore; j < aq.joins.size(); ++j) {
                                aq.joins[j].semi = true;
                                if (existsTables.count(aq.joins[j].leftTable))
                                    aq.joins[j].innerTable = aq.joins[j].leftTable;
                                else if (existsTables.count(aq.joins[j].rightTable))
                                    aq.joins[j].innerTable = aq.joins[j].rightTable;
                            }
                        }
                    }
                }
                // Convert ANY_SUBLINK into inner tables plus a semi-join.
	                if (subType == "ANY_SUBLINK" && sl.contains("subselect") && sl.contains("testexpr")) {
	                    auto& sub = sl["subselect"]["SelectStmt"];
	                    auto aliasRewriteBefore = ctx.aliasRewriteMap;
	                    size_t tablesBefore = aq.tables.size();
	                    // Allow duplicate inner tables; each instance gets its own build phase.
	                    if (sub.contains("fromClause")) {
	                        for (auto& item : sub["fromClause"]) {
	                            std::vector<std::string> newTables, newAliases;
	                            extractTables(ctx, item, newTables, newAliases);
	                            for (size_t ni = 0; ni < newTables.size(); ++ni) {
	                                aq.tables.push_back(newTables[ni]);
                                std::string originalAlias = ni < newAliases.size() ? newAliases[ni] : newTables[ni];
                                std::string alias = originalAlias;
                                // Keep duplicate instances addressable in later phases.
	                                if (tablesBefore < aq.tables.size()) {
	                                    alias += "_IN";
	                                    if (!originalAlias.empty())
	                                        ctx.aliasRewriteMap[originalAlias] = alias;
	                                }
	                                aq.tableAliases.push_back(alias);
	                            }
	                        }
	                        rebuildAliasMap(ctx, aq);
	                    }
	                    AnalyzeScope outerScope =
	                        AnalyzeScope::fromTables(
	                            std::vector<std::string>(aq.tables.begin(), aq.tables.begin() + tablesBefore),
	                            std::vector<std::string>(aq.tableAliases.begin(), aq.tableAliases.begin() + std::min(tablesBefore, aq.tableAliases.size())));
	                    AnalyzeScope subqueryScope =
	                        AnalyzeScope::fromSubqueryFirst(aq, tablesBefore);
	                    auto testExpr = walkExpr(ctx, sl["testexpr"], outerScope);
                    auto* testCol = testExpr ? std::get_if<ColRef>(&testExpr->node) : nullptr;
                    // The first inner SELECT column is the semi-join key.
                    ColRef innerCol;
                    if (sub.contains("targetList")) {
                        for (auto& t : sub["targetList"]) {
                            if (!t.contains("ResTarget")) continue;
                            auto& rt = t["ResTarget"];
                            if (!rt.contains("val")) continue;
	                            auto innerExpr = walkExpr(ctx, rt["val"], subqueryScope);
                            if (auto* ic = innerExpr ? std::get_if<ColRef>(&innerExpr->node) : nullptr) {
                                innerCol = *ic;
                                break;
                            }
                        }
                    }
                    if (testCol && !innerCol.table.empty()) {
                        std::string innerJoinName = innerCol.tableAlias;
                        if (innerJoinName.empty()) {
                            for (size_t ti = tablesBefore; ti < aq.tables.size() && ti < aq.tableAliases.size(); ++ti) {
                                if (aq.tables[ti] == innerCol.table) {
                                    innerJoinName = aq.tableAliases[ti];
                                    break;
                                }
                            }
                        }
                        if (innerJoinName.empty()) innerJoinName = innerCol.table;
                        JoinClause jc;
                        jc.leftTable = testCol->table;
                        jc.leftCol = testCol->column;
                        jc.rightTable = innerJoinName;
                        jc.rightCol = innerCol.column;
                        jc.anti = negated;
                        jc.semi = true;
                        jc.innerTable = innerJoinName;
                        aq.joins.push_back(jc);
                    }
                    // Attach inner WHERE predicates to the duplicate table instance.
	                    if (sub.contains("whereClause")) {
	                        size_t filterCountBefore = aq.filters.size();
	                        auto innerPred = walkPredicate(ctx, sub["whereClause"], subqueryScope);
	                        separatePredicates(innerPred, subqueryScope.tables, aq.joins, aq.filters);
	                        relocateInnerInstanceFilters(aq, tablesBefore, filterCountBefore);
                    }
                    // Preserve grouped IN-subquery aggregate metadata.
                    if (sub.contains("groupClause") && sub.contains("havingClause")) {
                        std::string innerAlias;
                        for (size_t ti = tablesBefore; ti < aq.tables.size() && ti < aq.tableAliases.size(); ++ti) {
                            if (aq.tables[ti] == innerCol.table) { innerAlias = aq.tableAliases[ti]; break; }
                        }
                        if (innerAlias.empty()) innerAlias = innerCol.table;
                        std::string groupCol;
                        const auto& gb = sub["groupClause"];
                        if (gb.is_array() && !gb.empty()) {
                            for (const auto& gitem : gb) {
                                const json* colNode = &gitem;
                                if (gitem.contains("ColumnRef")) colNode = &gitem["ColumnRef"];
	                                auto colExpr = walkExpr(ctx, *colNode, subqueryScope);
                                if (auto* cr = colExpr ? std::get_if<ColRef>(&colExpr->node) : nullptr) {
                                    groupCol = cr->column;
                                    break;
                                }
                            }
                        }
                        // Find the aggregate referenced by HAVING.
                        std::string aggFunc, aggExpr;
                        PredPtr havingPred;
                        {
                            const auto& hc = sub["havingClause"];
	                            auto hPred = walkPredicate(ctx, hc, subqueryScope);
                            if (hPred) {
                                havingPred = hPred;
                                std::function<void(const PredPtr&)> findAgg;
                                findAgg = [&](const PredPtr& p) {
                                    if (!p || !aggFunc.empty()) return;
                                    std::visit([&](auto&& n) {
                                        using T = std::decay_t<decltype(n)>;
                                        if constexpr (std::is_same_v<T, Comparison>) {
                                            for (auto* e : {&n.left, &n.right}) {
                                                if (*e && std::get_if<FuncCall>(&(*e)->node)) {
                                                    auto& fc = std::get<FuncCall>((*e)->node);
                                                    aggFunc = fc.name;
                                                    if (!fc.args.empty() && fc.args[0]) {
                                                        if (auto* cr = std::get_if<ColRef>(&fc.args[0]->node))
                                                            aggExpr = cr->column;
                                                    }
                                                }
                                            }
                                        }
                                    }, p->node);
                                };
                                findAgg(havingPred);
                            }
                        }
                        if (!groupCol.empty() && !aggFunc.empty()) {
                            AnalyzedQuery::InSubqueryAggInfo info;
                            info.alias = innerAlias;
                            info.baseTable = innerCol.table;
                            info.tableIndex = (int)tablesBefore;
                            info.groupCol = groupCol;
                            info.aggFunc = aggFunc;
                            info.aggExpr = aggExpr;
                            info.havingPred = havingPred;
                            aq.inSubAggs.push_back(info);
                        }
                    }
	                    ctx.aliasRewriteMap = std::move(aliasRewriteBefore);
	                }
            }
            // NOT flips EXISTS/IN join semantics.
            bool innerNegated = negated;
            if (node.is_object() && node.contains("BoolExpr")) {
                auto& be = node["BoolExpr"];
                if (be.value("boolop", "") == "NOT_EXPR") {
                    innerNegated = !negated;
                }
            }
            // FROM-subquery WHERE clauses can contain nested EXISTS/IN.
            if (node.is_object() && node.contains("RangeSubselect")) {
                auto& rs = node["RangeSubselect"];
                if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
                    auto& sub = rs["subquery"]["SelectStmt"];
                    if (sub.contains("whereClause")) {
                        inlineExists(sub["whereClause"], innerNegated);
                    }
                    if (sub.contains("fromClause")) {
                        for (auto& item : sub["fromClause"]) {
                            inlineExists(item, innerNegated);
                        }
                    }
                }
                return;
            }
            if (node.is_object()) {
                for (auto& [k, v] : node.items()) inlineExists(v, innerNegated);
            } else if (node.is_array()) {
                for (auto& i : node) inlineExists(i, innerNegated);
            }
        };
        if (sel.contains("whereClause"))
            inlineExists(sel["whereClause"], false);
        if (sel.contains("fromClause")) {
            for (auto& item : sel["fromClause"])
                inlineExists(item, false);
        }

        // Drop predicates that were replaced by join metadata.
        aq.filters.erase(
            std::remove_if(aq.filters.begin(), aq.filters.end(),
                isInlinedSubqueryPlaceholder),
            aq.filters.end());
        for (auto& [_, filters] : aq.instanceFilters) {
            filters.erase(
                std::remove_if(filters.begin(), filters.end(),
                               isInlinedSubqueryPlaceholder),
                filters.end());
        }
    }

	    // Rebuild alias map after subquery inlining adds table instances.
	    rebuildAliasMap(ctx, aq);
	    aq.aliasMap = ctx.aliasMap;
	    topScope = AnalyzeScope::fromAnalyzed(aq);

    // Extract SELECT targets.
    if (sel.contains("targetList")) {
        for (auto& t : sel["targetList"]) {
            if (t.contains("ResTarget")) {
	                if (resTargetIsStar(t["ResTarget"]))
	                    appendStarTargets(ctx, t["ResTarget"], aq, topScope);
	                else
	                    aq.targets.push_back(extractTarget(ctx, t["ResTarget"], topScope));
            }
        }
    }

	    if (sel.contains("groupClause")) {
	        for (auto& g : sel["groupClause"])
	            aq.groupBy.push_back(walkExpr(ctx, g, topScope));
	    }

	    if (sel.contains("havingClause"))
	        aq.having = walkPredicate(ctx, sel["havingClause"], topScope);

    if (sel.contains("sortClause")) {
        for (auto& s : sel["sortClause"]) {
            if (s.contains("SortBy")) {
                auto& sb = s["SortBy"];
	                OrderByItem obi;
	                obi.expr = walkExpr(ctx, sb["node"], topScope);
                if (sb.contains("sortby_dir")) {
                    std::string dir = sb["sortby_dir"].get<std::string>();
                    obi.descending = (dir == "SORTBY_DESC");
                }
                aq.orderBy.push_back(obi);
            }
        }
    }

    if (sel.contains("limitCount")) {
        auto& lc = sel["limitCount"];
        if (lc.contains("A_Const") && lc["A_Const"].contains("ival"))
            aq.limit = lc["A_Const"]["ival"]["ival"].get<int>();
    }

	    auto subqueryColMap = std::move(ctx.subqueryAliasMap);
	    auto subqueryExprMap = std::move(ctx.subqueryExprMap);
	    ctx.subqueryAliasMap.clear();
	    ctx.subqueryExprMap.clear();

    // Resolve subquery aliases inside target expressions.
    auto resolveRecursive = [&](ExprPtr& expr, auto&& self) -> void {
        if (!expr) return;
        if (auto* cr = std::get_if<ColRef>(&expr->node)) {
            if (cr->table.empty()) {
                auto it = subqueryColMap.find(cr->column);
                if (it != subqueryColMap.end()) {
                    expr = Expr::col(it->second.table, it->second.column,
                                     it->second.colIndex, it->second.dataType,
                                     it->second.fixedWidth, it->second.tableAlias);
                    return;
                }
                auto eit = subqueryExprMap.find(cr->column);
                if (eit != subqueryExprMap.end()) {
                    expr = eit->second;
                    return;
                }
            }
            return;
        }
        if (auto* bin = std::get_if<BinaryExpr>(&expr->node)) {
            self(bin->left, self);
            self(bin->right, self);
            return;
        }
        if (auto* cw = std::get_if<CaseWhen>(&expr->node)) {
            for (auto& b : cw->branches) { self(b.result, self); }
            if (cw->elseResult) self(cw->elseResult, self);
            return;
        }
        if (auto* fc = std::get_if<FuncCall>(&expr->node)) {
            for (auto& a : fc->args) self(a, self);
            return;
        }
    };
    for (auto& t : aq.targets) {
        std::string origAlias;
        if (t.expr && std::holds_alternative<ColRef>(t.expr->node) && t.alias.empty()) {
            origAlias = std::get_if<ColRef>(&t.expr->node)->column;
        }
        resolveRecursive(t.expr, resolveRecursive);
        if (t.agg) resolveRecursive(t.agg->innerExpr, resolveRecursive);
        if (!origAlias.empty() && t.alias.empty())
            t.alias = origAlias;
    }
    // GROUP BY/ORDER BY keep display aliases; targets use resolved expressions.

    // Transfer scalar subqueries into Generic source metadata input.
	    for (auto& sq : ctx.scalarSubqueries) {
	        aq.scalarSubquerySql.push_back(sq.sql);
	    }
	    ctx.scalarSubqueries.clear();

    return aq;
}

} // namespace codegen
