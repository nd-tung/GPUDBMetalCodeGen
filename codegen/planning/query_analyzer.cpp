#include "query_analyzer.h"

extern "C" {
#include "pg_query.h"
}
#include "../../third_party/nlohmann/json.hpp"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;

namespace codegen {

// ===================================================================
// INTERNAL HELPERS
// ===================================================================

namespace {

// File-scope alias map: alias -> real table name (e.g. "l1" -> "lineitem")
std::unordered_map<std::string, std::string> g_aliasMap;

// Subquery column alias → source ColRef.  Populated when a RangeSubselect's
// targetList aliases a column reference (e.g. "n2.n_name AS nation").
// Used to resolve outer-query column references that don't exist in physical
// tables but were aliased in a FROM-clause subquery.
std::unordered_map<std::string, ColRef> g_subqueryAliasMap;
// Subquery column alias → source expression (for non-column expressions).
std::unordered_map<std::string, ExprPtr> g_subqueryExprMap;

// File-scope: view definitions inlined during analysis.
std::map<std::string, std::pair<json, std::vector<std::string>>> g_views; // name → (selectBody, columnAliases)

const auto& schema() { return TPCHSchema::instance(); }

// Resolve an unqualified column name to (table, column).
// If multiple tables have the column, we need the table list to disambiguate.
// Returns ("", colName) if not found in any table (could be a SELECT alias).
std::pair<std::string, std::string> resolveColumn(const std::string& colName,
                                                   const std::vector<std::string>& tables) {
    for (auto& t : tables) {
        auto it = schema().tables.find(t);
        if (it == schema().tables.end()) continue;
        if (it->second.nameToIdx.count(colName)) return {t, colName};
    }
    return {"", colName}; // alias or derived column
}

// Parse a date string like "1994-01-01" to YYYYMMDD integer
int parseDateLiteral(const std::string& s) {
    // Expecting YYYY-MM-DD
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        int y = std::stoi(s.substr(0, 4));
        int m = std::stoi(s.substr(5, 2));
        int d = std::stoi(s.substr(8, 2));
        return y * 10000 + m * 100 + d;
    }
    throw std::runtime_error("Invalid date literal: " + s);
}

// ===================================================================
// AST WALKING
// ===================================================================

// Forward declarations
ExprPtr walkExpr(const json& node, const std::vector<std::string>& tables);
PredPtr walkPredicate(const json& node, const std::vector<std::string>& tables);

ExprPtr walkColumnRef(const json& node, const std::vector<std::string>& tables) {
    auto& fields = node["fields"];
    std::string colName;
    std::string tblQualifier;
    if (fields.size() == 1) {
        colName = fields[0]["String"]["sval"].get<std::string>();
    } else if (fields.size() == 2) {
        tblQualifier = fields[0]["String"]["sval"].get<std::string>();
        colName = fields[1]["String"]["sval"].get<std::string>();
    }

    std::string resolvedTable;
    if (!tblQualifier.empty()) {
        // Resolve alias to real table name if needed
        auto ait = g_aliasMap.find(tblQualifier);
        resolvedTable = (ait != g_aliasMap.end()) ? ait->second : tblQualifier;
    } else {
        auto [t, c] = resolveColumn(colName, tables);
        resolvedTable = t;
    }

    if (resolvedTable.empty()) {
        // Check subquery column aliases (FROM-clause subquery SELECT list).
        auto sqit = g_subqueryAliasMap.find(colName);
        if (sqit != g_subqueryAliasMap.end())
            return Expr::col(sqit->second.table, sqit->second.column,
                             sqit->second.colIndex, sqit->second.dataType);
        // This is a SELECT alias (e.g., "revenue" in ORDER BY) — return as unresolved ColRef
        return Expr::col("", colName, -1, DataType::INT);
    }

    auto& tbl = schema().table(resolvedTable);
    auto& col = tbl.col(colName);
    return Expr::col(resolvedTable, colName, col.index, col.type);
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
    std::cerr << "WARN: unknown A_Const type: " << node.dump().substr(0, 200) << std::endl;
    return Expr::lit(0);
}

ExprPtr walkTypeCast(const json& node, const std::vector<std::string>& tables) {
    auto& typeName = node["typeName"];
    std::string typStr;
    if (typeName.contains("names")) {
        for (auto& n : typeName["names"]) {
            if (n.contains("String"))
                typStr = n["String"]["sval"].get<std::string>();
        }
    }

    auto arg = walkExpr(node["arg"], tables);
    // DATE cast: convert string literal to integer
    if (typStr == "date") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) {
                return Expr::lit(parseDateLiteral(*sv));
            }
        }
    }
    // INTERVAL cast: return raw integer value (NOT scaled to YYYYMMDD offset).
    // The unit (YEAR/MONTH/DAY) is resolved at the point of use in walkAExpr
    // by re-inspecting the AST node's typmods.
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
    return arg; // For other casts, pass through
}

// Helper: proper date arithmetic in YYYYMMDD format for DAY intervals
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

// Interval unit enum
enum class IntervalUnit { UNKNOWN, YEAR, MONTH, DAY };

// Extract interval unit from a TypeCast AST node's typmods
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
    // PostgreSQL datetime.h: YEAR=2, MONTH=1, DAY=3
    // INTERVAL_MASK(X) = 1 << X → YEAR=4, MONTH=2, DAY=8
    if (typmods & 4) return IntervalUnit::YEAR;
    if (typmods & 2) return IntervalUnit::MONTH;
    if (typmods & 8) return IntervalUnit::DAY;
    return IntervalUnit::UNKNOWN;
}

// Compute DATE ± INTERVAL with proper unit handling (YEAR/MONTH/DAY)
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

ExprPtr walkFuncCall(const json& node, const std::vector<std::string>& tables) {
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
            fc.args.push_back(walkExpr(a, tables));
    }

    auto e = std::make_shared<Expr>();
    e->node = fc;
    return e;
}

ExprPtr walkAExpr(const json& node, const std::vector<std::string>& tables) {
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
            // Comparison operators — shouldn't be called from walkExpr, but handle gracefully
            // Return a dummy expression
            if (node.contains("lexpr"))
                return walkExpr(node["lexpr"], tables);
            return Expr::lit(0);
        }
        auto left = node.contains("lexpr") ? walkExpr(node["lexpr"], tables) : Expr::lit(0);
        auto right = node.contains("rexpr") ? walkExpr(node["rexpr"], tables) : Expr::lit(0);

        // Pre-compute date ± interval when both sides are literals
        if (exOp == ExprOp::ADD || exOp == ExprOp::SUB) {
            auto* litL = std::get_if<Literal>(&left->node);
            auto* litR = std::get_if<Literal>(&right->node);
            if (litL && litR) {
                auto* dateVal = std::get_if<int>(&litL->value);
                auto* intVal  = std::get_if<int>(&litR->value);
                if (dateVal && intVal && *dateVal > 19000101 && *dateVal < 21001231) {
                    bool isAdd = (exOp == ExprOp::ADD);
                    // Determine interval unit from the original AST rexpr
                    IntervalUnit unit = IntervalUnit::DAY; // default
                    if (node.contains("rexpr") && node["rexpr"].contains("TypeCast"))
                        unit = extractIntervalUnit(node["rexpr"]["TypeCast"]);
                    int result = computeDateArith(*dateVal, *intVal, isAdd, unit);
                    return Expr::lit(result);
                }
            }
        }

        return Expr::binary(exOp, left, right);
    }
    // Non-arithmetic A_Expr types (LIKE, BETWEEN, IN) in expression context
    // Return lexpr as a pass-through — the planner doesn't inspect these deeply
    if (node.contains("lexpr"))
        return walkExpr(node["lexpr"], tables);
    return Expr::lit(0);
}

ExprPtr walkExpr(const json& node, const std::vector<std::string>& tables) {
    if (node.contains("ColumnRef"))
        return walkColumnRef(node["ColumnRef"], tables);
    if (node.contains("A_Const"))
        return walkConst(node["A_Const"]);
    if (node.contains("TypeCast"))
        return walkTypeCast(node["TypeCast"], tables);
    if (node.contains("FuncCall"))
        return walkFuncCall(node["FuncCall"], tables);
    if (node.contains("A_Expr"))
        return walkAExpr(node["A_Expr"], tables);
    if (node.contains("SubLink")) {
        // Subquery expression — return placeholder
        return Expr::lit(0);
    }
    if (node.contains("BoolExpr")) {
        // Boolean expression in expression context (e.g., in CASE WHEN)
        // Return a placeholder literal
        return Expr::lit(0);
    }
    if (node.contains("CaseExpr")) {
        auto& ce = node["CaseExpr"];
        CaseWhen cw;
        if (ce.contains("args")) {
            for (auto& when : ce["args"]) {
                if (when.contains("CaseWhen")) {
                    auto& caseWhen = when["CaseWhen"];
                    CaseWhen::Branch br;
                    br.condition = walkPredicate(caseWhen["expr"], tables);
                    br.result = walkExpr(caseWhen["result"], tables);
                    cw.branches.push_back(std::move(br));
                }
            }
        }
        if (ce.contains("defresult"))
            cw.elseResult = walkExpr(ce["defresult"], tables);
        auto e = std::make_shared<Expr>();
        e->node = std::move(cw);
        return e;
    }
    std::cerr << "WARN: unhandled expr node, returning lit(0): " << node.dump().substr(0, 200) << std::endl;
    return Expr::lit(0);
}

// ===================================================================
// PREDICATE WALKING
// ===================================================================

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

PredPtr walkAExprPred(const json& node, const std::vector<std::string>& tables) {
    std::string kind = node.value("kind", "AEXPR_OP");
    std::string opName;
    if (node.contains("name")) {
        for (auto& n : node["name"])
            if (n.contains("String"))
                opName = n["String"]["sval"].get<std::string>();
    }

    if (kind == "AEXPR_OP") {
        auto left = walkExpr(node["lexpr"], tables);
        auto right = walkExpr(node["rexpr"], tables);
        return Predicate::cmp(parseCmpOp(opName), left, right);
    }
    if (kind == "AEXPR_BETWEEN" || kind == "AEXPR_NOT_BETWEEN") {
        auto expr = walkExpr(node["lexpr"], tables);
        auto& list = node["rexpr"]["List"]["items"];
        auto lo = walkExpr(list[0], tables);
        auto hi = walkExpr(list[1], tables);
        auto p = Predicate::between(expr, lo, hi);
        if (kind == "AEXPR_NOT_BETWEEN")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_IN") {
        auto expr = walkExpr(node["lexpr"], tables);
        std::vector<ExprPtr> vals;
        if (node["rexpr"].contains("List")) {
            for (auto& item : node["rexpr"]["List"]["items"])
                vals.push_back(walkExpr(item, tables));
        }
        auto p = Predicate::inList(expr, std::move(vals));
        if (opName == "<>")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
        auto expr = walkExpr(node["lexpr"], tables);
        auto patExpr = walkExpr(node["rexpr"], tables);
        std::string pat;
        if (auto* lit = std::get_if<Literal>(&patExpr->node))
            if (auto* sv = std::get_if<std::string>(&lit->value))
                pat = *sv;
        bool negated = (opName == "!~~" || opName == "!~~*");
        return Predicate::like(expr, pat, negated);
    }
    if (kind == "AEXPR_NOT_DISTINCT") {
        // Treat as equality
        auto left = walkExpr(node["lexpr"], tables);
        auto right = walkExpr(node["rexpr"], tables);
        return Predicate::cmp(CmpOp::EQ, left, right);
    }

    // Fallback: treat as comparison
    auto left = node.contains("lexpr") ? walkExpr(node["lexpr"], tables) : Expr::lit(0);
    auto right = node.contains("rexpr") ? walkExpr(node["rexpr"], tables) : Expr::lit(0);
    return Predicate::cmp(CmpOp::EQ, left, right);
}

PredPtr walkPredicate(const json& node, const std::vector<std::string>& tables) {
    if (node.contains("BoolExpr")) {
        auto& be = node["BoolExpr"];
        std::string boolop = be["boolop"].get<std::string>();
        if (boolop == "AND_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(arg, tables));
            return Predicate::logAnd(std::move(children));
        }
        if (boolop == "OR_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(arg, tables));
            return Predicate::logOr(std::move(children));
        }
        if (boolop == "NOT_EXPR") {
            return Predicate::logNot(walkPredicate(be["args"][0], tables));
        }
    }
    if (node.contains("A_Expr")) {
        return walkAExprPred(node["A_Expr"], tables);
    }
    if (node.contains("NullTest")) {
        auto& nt = node["NullTest"];
        auto expr = walkExpr(nt["arg"], tables);
        std::string kind = nt.value("nulltesttype", "IS_NULL");
        if (kind == "IS_NOT_NULL")
            return Predicate::cmp(CmpOp::NE, expr, Expr::lit(0)); // approximate
        return Predicate::cmp(CmpOp::EQ, expr, Expr::lit(0)); // approximate
    }
    if (node.contains("SubLink")) {
        // EXISTS / NOT EXISTS / IN subquery — store subquery SQL for later inlining
        auto& sl = node["SubLink"];
        std::string subType = sl.value("subLinkType", "EXISTS_SUBLINK");
        bool isExists = (subType == "EXISTS_SUBLINK");
        if (isExists || subType == "ALL_SUBLINK" || subType == "ANY_SUBLINK") {
            // Extract subquery SQL from the SubLink's subselect field
            std::string subSql;
            if (sl.contains("subselect")) {
                // Reconstruct SQL from the subquery's AST. For simplicity,
                // store the raw subselect JSON so it can be re-analyzed later.
            }
            if (isExists) {
                auto p = std::make_shared<Predicate>();
                p->node = ExistsPred{false, -1};
                return p;
            }
            if (subType == "ALL_SUBLINK" || subType == "ANY_SUBLINK") {
                if (sl.contains("testexpr")) {
                    auto expr = walkExpr(sl["testexpr"], tables);
                    return Predicate::inList(expr, {}); // placeholder
                }
            }
        }
        auto p = std::make_shared<Predicate>();
        p->node = ExistsPred{false, -1};
        return p;
    }
    throw std::runtime_error("Unknown predicate node: " + node.dump().substr(0, 100));
}

// ===================================================================
// EXTRACT TABLES FROM FROM CLAUSE
// ===================================================================

void extractTables(const json& fromItem, std::vector<std::string>& tables,
                   std::vector<std::string>& aliases) {
    if (fromItem.contains("RangeVar")) {
        auto& rv = fromItem["RangeVar"];
        std::string name = rv["relname"].get<std::string>();
        // Inline CREATE VIEW definitions: extract view's FROM tables.
        auto vit = g_views.find(name);
        if (vit != g_views.end()) {
            auto& [viewBody, viewCols] = vit->second;
            if (viewBody.contains("fromClause")) {
                for (auto& item : viewBody["fromClause"]) {
                    // Only add tables not already present (dedup for scalar subqueries).
                    std::vector<std::string> newTables, newAliases;
                    extractTables(item, newTables, newAliases);
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
        extractTables(je["larg"], tables, aliases);
        extractTables(je["rarg"], tables, aliases);
    }
    if (fromItem.contains("RangeSubselect")) {
        auto& rs = fromItem["RangeSubselect"];
        if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
            auto& sub = rs["subquery"]["SelectStmt"];
            if (sub.contains("fromClause")) {
                for (auto& item : sub["fromClause"])
                    extractTables(item, tables, aliases);
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

// ===================================================================
// EXTRACT JOIN CONDITIONS FROM WHERE
// ===================================================================

// Check if a predicate is a join condition (column = column across different tables)
bool isJoinCondition(const PredPtr& pred, JoinClause& jc) {
    auto* cmp = std::get_if<Comparison>(&pred->node);
    if (!cmp || cmp->op != CmpOp::EQ) return false;

    auto* leftCol = std::get_if<ColRef>(&cmp->left->node);
    auto* rightCol = std::get_if<ColRef>(&cmp->right->node);
    if (!leftCol || !rightCol) return false;
    // Self-joins are valid (e.g. l1.l_orderkey = l2.l_orderkey in EXISTS subqueries).
    // The multi-table builder disambiguates via the visited-node BFS.

    jc.leftTable = leftCol->table;
    jc.leftCol = leftCol->column;
    jc.rightTable = rightCol->table;
    jc.rightCol = rightCol->column;
    return true;
}

// Flatten AND predicates and separate join conditions from filters
void separatePredicates(const PredPtr& pred, const std::vector<std::string>& tables,
                        std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (auto* la = std::get_if<LogicalAnd>(&pred->node)) {
        for (auto& child : la->children)
            separatePredicates(child, tables, joins, filters);
        return;
    }
    if (auto* lo = std::get_if<LogicalOr>(&pred->node)) {
        // OR branches: extract join conditions that appear in ALL branches
        // (e.g. Q19: all branches have p_partkey = l_partkey).
        std::vector<JoinClause> commonJoins;
        std::vector<PredPtr> strippedBranches; // branches with joins removed
        bool first = true;
        for (auto& child : lo->children) {
            std::vector<JoinClause> branchJoins;
            std::vector<PredPtr> branchFilters;
            separatePredicates(child, tables, branchJoins, branchFilters);
            // Reconstruct branch without extracted joins.
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

// ===================================================================
// EXTRACT TARGET LIST
// ===================================================================

AggFunc parseAggFunc(const std::string& name) {
    if (name == "sum")   return AggFunc::SUM;
    if (name == "count") return AggFunc::COUNT;
    if (name == "avg")   return AggFunc::AVG;
    if (name == "min")   return AggFunc::MIN;
    if (name == "max")   return AggFunc::MAX;
    throw std::runtime_error("Unknown aggregate: " + name);
}

SelectTarget extractTarget(const json& resTarget, const std::vector<std::string>& tables) {
    SelectTarget st;
    st.alias = resTarget.value("name", "");
    auto& val = resTarget["val"];
    if (val.contains("FuncCall")) {
        auto& fc = val["FuncCall"];
        std::string funcName;
        if (fc.contains("funcname"))
            for (auto& n : fc["funcname"])
                if (n.contains("String"))
                    funcName = n["String"]["sval"].get<std::string>();
        std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);

        // Check if it's an aggregate function
        if (funcName == "sum" || funcName == "count" || funcName == "avg" ||
            funcName == "min" || funcName == "max") {
            st.isAgg = true;
            AggTarget at;
            at.func = parseAggFunc(funcName);
            at.alias = st.alias;
            if (fc.contains("agg_star") && fc["agg_star"].get<bool>()) {
                at.isStar = true;
            } else if (fc.contains("args") && !fc["args"].empty()) {
                at.innerExpr = walkExpr(fc["args"][0], tables);
            }
            if (fc.contains("agg_distinct") && fc["agg_distinct"].get<bool>()) {
                at.func = AggFunc::COUNT_DISTINCT;
            }
            st.agg = at;
            st.expr = walkExpr(val, tables); // Full FuncCall as expr
        } else {
            st.expr = walkExpr(val, tables);
        }
    } else {
        st.expr = walkExpr(val, tables);
    }
    return st;
}

// ===================================================================
// EXTRACT JOIN CONDITIONS FROM EXPLICIT JOIN ON
// ===================================================================

void extractJoinOns(const json& fromItem, const std::vector<std::string>& tables,
                    std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (fromItem.contains("JoinExpr")) {
        auto& je = fromItem["JoinExpr"];
        size_t joinCountBefore = joins.size();
        if (je.contains("quals")) {
            auto pred = walkPredicate(je["quals"], tables);
            separatePredicates(pred, tables, joins, filters);
        }
        // Detect LEFT OUTER JOIN: mark newly added join clauses.
        if (je.contains("jointype")) {
            try {
                if (je["jointype"].get<int>() == 1) {
                    for (size_t j = joinCountBefore; j < joins.size(); ++j)
                        joins[j].leftOuter = true;
                }
            } catch (...) {}
        }
        extractJoinOns(je["larg"], tables, joins, filters);
        extractJoinOns(je["rarg"], tables, joins, filters);
    }
}

} // anonymous namespace

// ===================================================================
// PUBLIC: analyzeSQL
// ===================================================================

AnalyzedQuery analyzeSQL(const std::string& sql) {
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
    g_aliasMap.clear();
    g_subqueryAliasMap.clear();
    g_subqueryExprMap.clear();

    // Navigate to the SelectStmt.  Handle CREATE VIEW statements by
    // inlining the view definition into the main query's FROM clause.
    // Find the last SelectStmt; also collect ViewStmt definitions.
    json* selPtr = nullptr;
    struct ViewDef { std::string name; json selectBody; std::vector<std::string> cols; };
    std::vector<ViewDef> views;
    for (auto& s : ast["stmts"]) {
        if (s["stmt"].contains("ViewStmt")) {
            auto& vs = s["stmt"]["ViewStmt"];
            ViewDef vd;
            if (vs.contains("view") && vs["view"].contains("RangeVar"))
                vd.name = vs["view"]["RangeVar"].value("relname", "");
            if (vd.name.empty()) continue;
            vd.selectBody = vs["query"]["SelectStmt"];
            if (vs.contains("aliases")) {
                for (auto& a : vs["aliases"]) {
                    if (a.contains("String") && a["String"].contains("str"))
                        vd.cols.push_back(a["String"]["str"].get<std::string>());
                    else if (a.contains("aliasname"))
                        vd.cols.push_back(a["aliasname"].get<std::string>());
                }
            }
            views.push_back(std::move(vd));
            g_views[vd.name] = {vd.selectBody, vd.cols};
        }
        if (s["stmt"].contains("SelectStmt")) {
            selPtr = &s["stmt"]["SelectStmt"];
        }
    }
    if (!selPtr)
        throw std::runtime_error("Expected SELECT statement");
    auto& sel = *selPtr;

    // 1. Extract tables from FROM clause
    if (sel.contains("fromClause")) {
        for (auto& item : sel["fromClause"])
            extractTables(item, aq.tables, aq.tableAliases);
        // Build alias -> real table name map
        g_aliasMap.clear();
        for (size_t i = 0; i < aq.tables.size(); ++i) {
            if (i < aq.tableAliases.size() && aq.tableAliases[i] != aq.tables[i])
                g_aliasMap[aq.tableAliases[i]] = aq.tables[i];
        }
        // Extract explicit JOIN ON conditions
        for (auto& item : sel["fromClause"])
            extractJoinOns(item, aq.tables, aq.joins, aq.filters);

        // Extract WHERE conditions from RangeSubselect subqueries
        // (e.g. Q8/Q9/Q13/Q19 — FROM-clause subqueries whose join
        //  conditions and filters reside in the inner WHERE clause).
        std::function<void(const json&)> extractSubWhere = [&](const json& fromItem) {
            if (fromItem.contains("RangeSubselect")) {
                auto& rs = fromItem["RangeSubselect"];
                if (rs.contains("subquery") && rs["subquery"].contains("SelectStmt")) {
                    auto& sub = rs["subquery"]["SelectStmt"];
                    if (sub.contains("whereClause")) {
                        auto pred = walkPredicate(sub["whereClause"], aq.tables);
                        separatePredicates(pred, aq.tables, aq.joins, aq.filters);
                    }
                    // Also extract JOIN ON conditions from the subquery's FROM clause.
                    if (sub.contains("fromClause")) {
                        for (auto& item : sub["fromClause"])
                            extractJoinOns(item, aq.tables, aq.joins, aq.filters);
                    }
                    // Extract subquery column aliases for outer-query resolution.
                    // e.g. "n2.n_name AS nation" → g_subqueryAliasMap["nation"] = ColRef(nation, n_name)
                    if (sub.contains("targetList")) {
                        for (auto& t : sub["targetList"]) {
                            if (!t.contains("ResTarget")) continue;
                            auto& rt = t["ResTarget"];
                            std::string alias = rt.value("name", "");
                            if (alias.empty()) continue;
                            if (!rt.contains("val")) continue;
                            auto& val = rt["val"];
                            if (val.contains("ColumnRef")) {
                                auto cre = walkColumnRef(val["ColumnRef"], aq.tables);
                                if (auto* cr = cre ? std::get_if<ColRef>(&cre->node) : nullptr) {
                                    g_subqueryAliasMap[alias] = *cr;
                                }
                            } else {
                                // Non-column expression (FuncCall, BinaryExpr, etc.)
                                auto expr = walkExpr(val, aq.tables);
                                if (expr) g_subqueryExprMap[alias] = expr;
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
    }

    // 1c. Process inlined view definitions: extract view's WHERE clause
    // and map view column aliases to their source expressions.
    for (auto& [name, vp] : g_views) {
        auto& [viewBody, viewCols] = vp;
        if (viewBody.contains("whereClause")) {
            auto pred = walkPredicate(viewBody["whereClause"], aq.tables);
            separatePredicates(pred, aq.tables, aq.joins, aq.filters);
        }
        // Map view column aliases to SELECT target expressions.
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
                auto expr = walkExpr(rt["val"], aq.tables);
                if (!expr) continue;
                if (auto* cr = std::get_if<ColRef>(&expr->node))
                    g_subqueryAliasMap[alias] = *cr;
                else
                    g_subqueryExprMap[alias] = expr;
            }
        }
    }

    // 2. Extract WHERE clause predicates
    if (sel.contains("whereClause")) {
        auto wherePred = walkPredicate(sel["whereClause"], aq.tables);
        separatePredicates(wherePred, aq.tables, aq.joins, aq.filters);
    }

    // 2b. Inline EXISTS subqueries: extract inner tables and correlation joins.
    // Walk the JSON AST for EXISTS_SUBLINK nodes, extract the inner query's
    // FROM table and WHERE correlation predicate, and merge into the main query.
    // Then remove the placeholder ExistsPred from filters.
    if (sel.contains("whereClause")) {
        std::function<void(const json&, bool)> inlineExists = [&](const json& node, bool negated) {
            if (node.is_object() && node.contains("SubLink")) {
                auto& sl = node["SubLink"];
                std::string subType = sl.value("subLinkType", "");
                if (subType == "EXISTS_SUBLINK" && sl.contains("subselect")) {
                    auto& sub = sl["subselect"]["SelectStmt"];
                    if (sub.contains("fromClause")) {
                        for (auto& item : sub["fromClause"]) {
                            extractTables(item, aq.tables, aq.tableAliases);
                        }
                        // Rebuild alias map so column refs in the inner WHERE
                        // can resolve qualified names (e.g. l2.l_orderkey).
                        g_aliasMap.clear();
                        for (size_t i = 0; i < aq.tables.size() && i < aq.tableAliases.size(); ++i) {
                            g_aliasMap[aq.tableAliases[i]] = aq.tables[i];
                        }
                    }
                    if (sub.contains("whereClause")) {
                        // Record join count before adding: new joins from NOT EXISTS
                        // are anti-joins.
                        size_t joinCountBefore = aq.joins.size();
                        auto innerPred = walkPredicate(sub["whereClause"], aq.tables);
                        separatePredicates(innerPred, aq.tables, aq.joins, aq.filters);
                        if (negated) {
                            for (size_t j = joinCountBefore; j < aq.joins.size(); ++j) {
                                aq.joins[j].anti = true;
                            }
                        }
                    }
                }
            }
            // Track negated context: BoolExpr(NOT_EXPR) toggles the flag.
            bool innerNegated = negated;
            if (node.is_object() && node.contains("BoolExpr")) {
                auto& be = node["BoolExpr"];
                if (be.value("boolop", "") == "NOT_EXPR") {
                    innerNegated = !negated;
                }
            }
            // Recurse into RangeSubselect inner WHERE clauses (e.g. Q22's
            // NOT EXISTS inside a FROM-clause subquery).
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
            }
            if (node.is_object()) {
                for (auto& [k, v] : node.items()) inlineExists(v, innerNegated);
            } else if (node.is_array()) {
                for (auto& i : node) inlineExists(i, innerNegated);
            }
        };
        inlineExists(sel["whereClause"], false);

        // Remove placeholder ExistsPred entries from filters (they've been inlined).
        // Also handle NOT EXISTS: LogicalNot(ExistsPred) must be removed too since
        // the negation will be applied as an anti-join or anti-bitmap-probe.
        aq.filters.erase(
            std::remove_if(aq.filters.begin(), aq.filters.end(),
                [](const PredPtr& p) {
                    if (std::holds_alternative<ExistsPred>(p->node)) return true;
                    if (auto* ln = std::get_if<LogicalNot>(&p->node)) {
                        return ln->child && std::holds_alternative<ExistsPred>(ln->child->node);
                    }
                    return false;
                }),
            aq.filters.end());
    }

    // Rebuild alias map after inlineExists may have added new tables
    // with aliases (e.g. Q21's l2, l3 from EXISTS subqueries).
    g_aliasMap.clear();
    for (size_t i = 0; i < aq.tables.size() && i < aq.tableAliases.size(); ++i) {
        g_aliasMap[aq.tableAliases[i]] = aq.tables[i];
    }

    // 3. Extract SELECT targets
    if (sel.contains("targetList")) {
        for (auto& t : sel["targetList"]) {
            if (t.contains("ResTarget"))
                aq.targets.push_back(extractTarget(t["ResTarget"], aq.tables));
        }
    }

    if (sel.contains("groupClause")) {
        for (auto& g : sel["groupClause"])
            aq.groupBy.push_back(walkExpr(g, aq.tables));
    }

    // 5. Extract HAVING
    if (sel.contains("havingClause"))
        aq.having = walkPredicate(sel["havingClause"], aq.tables);

    // 6. Extract ORDER BY
    if (sel.contains("sortClause")) {
        for (auto& s : sel["sortClause"]) {
            if (s.contains("SortBy")) {
                auto& sb = s["SortBy"];
                OrderByItem obi;
                obi.expr = walkExpr(sb["node"], aq.tables);
                if (sb.contains("sortby_dir")) {
                    std::string dir = sb["sortby_dir"].get<std::string>();
                    obi.descending = (dir == "SORTBY_DESC");
                }
                aq.orderBy.push_back(obi);
            }
        }
    }

    // 7. Extract LIMIT
    if (sel.contains("limitCount")) {
        auto& lc = sel["limitCount"];
        if (lc.contains("A_Const") && lc["A_Const"].contains("ival"))
            aq.limit = lc["A_Const"]["ival"]["ival"].get<int>();
    }

    // Copy subquery alias maps to AnalyzedQuery for builder access.
    aq.subqueryColMap = std::move(g_subqueryAliasMap);
    aq.subqueryExprMap = std::move(g_subqueryExprMap);
    g_subqueryAliasMap.clear();
    g_subqueryExprMap.clear();

    // Recursively resolve subquery aliases in an expression tree.
    auto resolveRecursive = [&](ExprPtr& expr, auto&& self) -> void {
        if (!expr) return;
        if (auto* cr = std::get_if<ColRef>(&expr->node)) {
            if (cr->table.empty()) {
                auto it = aq.subqueryColMap.find(cr->column);
                if (it != aq.subqueryColMap.end()) {
                    expr = Expr::col(it->second.table, it->second.column,
                                     it->second.colIndex, it->second.dataType);
                    return;
                }
                auto eit = aq.subqueryExprMap.find(cr->column);
                if (eit != aq.subqueryExprMap.end()) {
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
    // GROUP BY and ORDER BY keep their original ColRef — the subquery alias
    // name is used for display-name matching in orderColumnForExpr and
    // materialize column naming; the resolved expression is used in targets.

    return aq;
}

} // namespace codegen
