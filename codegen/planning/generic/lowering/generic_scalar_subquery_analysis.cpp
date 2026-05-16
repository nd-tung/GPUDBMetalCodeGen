#include "generic/lowering/generic_scalar_subquery_analysis.h"

#include "core/schema_provider.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "../../../../third_party/nlohmann/json.hpp"

namespace codegen {

namespace {

bool isAggregateFuncCallName(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name == "sum" || name == "avg" || name == "count" ||
           name == "min" || name == "max";
}

static std::optional<std::string> jsonStringValue(const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") && node["String"].contains("sval"))
        return node["String"]["sval"].get<std::string>();
    return std::nullopt;
}

static std::string jsonAExprOp(const nlohmann::json& ae) {
    if (!ae.contains("name") || !ae["name"].is_array() || ae["name"].empty()) return {};
    if (auto s = jsonStringValue(ae["name"][0])) return *s;
    return {};
}

static std::optional<DecorrCol> jsonRawColumnRef(const nlohmann::json& node) {
    const nlohmann::json* cr = nullptr;
    if (node.is_object() && node.contains("ColumnRef")) cr = &node["ColumnRef"];
    else if (node.is_object() && node.contains("fields")) cr = &node;
    if (!cr || !cr->contains("fields") || !(*cr)["fields"].is_array()) return std::nullopt;
    std::vector<std::string> fields;
    for (const auto& f : (*cr)["fields"]) {
        if (auto s = jsonStringValue(f)) fields.push_back(*s);
    }
    if (fields.empty()) return std::nullopt;
    DecorrCol out;
    out.column = fields.back();
    if (fields.size() >= 2) out.qualifier = fields[fields.size() - 2];
    return out;
}

static std::string jsonFuncName(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() || fc["funcname"].empty())
        return {};
    auto s = jsonStringValue(fc["funcname"].back());
    if (!s) return {};
    std::string name = *s;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return name;
}

static std::optional<double> jsonNumericConst(const nlohmann::json& node) {
    const nlohmann::json* ac = nullptr;
    if (node.is_object() && node.contains("A_Const")) ac = &node["A_Const"];
    else if (node.is_object()) ac = &node;
    if (!ac) return std::nullopt;
    auto readNum = [](const nlohmann::json& v) -> std::optional<double> {
        if (v.is_number()) return v.get<double>();
        if (v.is_string()) return std::stod(v.get<std::string>());
        return std::nullopt;
    };
    try {
        if (ac->contains("fval")) {
            const auto& f = (*ac)["fval"];
            if (f.is_object() && f.contains("fval")) return readNum(f["fval"]);
            return readNum(f);
        }
        if (ac->contains("ival")) {
            const auto& i = (*ac)["ival"];
            if (i.is_object() && i.contains("ival")) return readNum(i["ival"]);
            return readNum(i);
        }
        if (ac->contains("val")) {
            const auto& v = (*ac)["val"];
            if (v.contains("Float")) return readNum(v["Float"].at("fval"));
            if (v.contains("Integer")) return readNum(v["Integer"].at("ival"));
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

static int parseDateLiteralLocal(const std::string& s) {
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        return std::stoi(s.substr(0, 4)) * 10000 +
               std::stoi(s.substr(5, 2)) * 100 +
               std::stoi(s.substr(8, 2));
    }
    return 0;
}

static int dateAddYearsLocal(int yyyymmdd, int years) {
    int y = yyyymmdd / 10000;
    int md = yyyymmdd % 10000;
    return (y + years) * 10000 + md;
}

static DecorrCol resolveDecorrCol(DecorrCol col,
                                  const DecorrelatedScalarSubquery& dsq,
                                  const SchemaProvider* schema) {
    if (!col.qualifier.empty()) {
        auto it = dsq.aliases.find(col.qualifier);
        if (it != dsq.aliases.end()) {
            col.table = it->second;
            col.inner = true;
            return col;
        }
        if (schema && schema->hasColumn(col.qualifier, col.column)) {
            col.table = col.qualifier;
            col.inner = true;
            return col;
        }
        col.table = col.qualifier;
        col.inner = false;
        return col;
    }

    std::string match;
    for (const auto& table : dsq.tables) {
        if (schema && schema->hasColumn(table, col.column)) {
            if (!match.empty()) {
                col.inner = false;
                return col;
            }
            match = table;
        }
    }
    if (!match.empty()) {
        col.table = match;
        col.inner = true;
    }
    return col;
}

static ExprPtr jsonExprToExpr(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema);

static ExprPtr jsonConstToExpr(const nlohmann::json& ac) {
    if (ac.contains("ival")) {
        const auto& iv = ac["ival"];
        if (iv.is_object() && iv.contains("ival")) return Expr::lit(iv["ival"].get<int>());
        return Expr::lit(0);
    }
    if (ac.contains("fval")) {
        const auto& fv = ac["fval"];
        if (fv.is_object() && fv.contains("fval"))
            return Expr::litf(std::stof(fv["fval"].get<std::string>()));
        return Expr::litf(0.0f);
    }
    if (ac.contains("sval")) {
        const auto& sv = ac["sval"];
        if (sv.is_object() && sv.contains("sval"))
            return Expr::lits(sv["sval"].get<std::string>());
    }
    return Expr::lit(0);
}

static ExprPtr jsonTypeCastToExpr(const nlohmann::json& tc,
                                  const DecorrelatedScalarSubquery& dsq,
                                  const SchemaProvider* schema) {
    std::string typ;
    if (tc.contains("typeName") && tc["typeName"].contains("names")) {
        for (const auto& n : tc["typeName"]["names"]) {
            if (auto s = jsonStringValue(n)) typ = *s;
        }
    }
    auto arg = jsonExprToExpr(tc.value("arg", nlohmann::json{}), dsq, schema);
    if (typ == "date") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value))
                return Expr::lit(parseDateLiteralLocal(*sv));
        }
    }
    if (typ == "interval") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) return Expr::lit(std::stoi(*sv));
            if (auto* iv = std::get_if<int>(&lit->value)) return Expr::lit(*iv);
        }
    }
    return arg;
}

static ExprPtr jsonExprToExpr(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema) {
    if (node.contains("ColumnRef")) {
        auto raw = jsonRawColumnRef(node);
        if (!raw) return Expr::lit(0);
        auto col = resolveDecorrCol(*raw, dsq, schema);
        DataType dt = (col.inner && schema) ? schema->columnType(col.table, col.column) : DataType::INT;
        int fw = (dt == DataType::CHAR_FIXED && schema) ? schema->columnFixedWidth(col.table, col.column) : 0;
        return Expr::col(col.inner ? col.table : "", col.column, -1, dt, fw, col.qualifier);
    }
    if (node.contains("A_Const")) return jsonConstToExpr(node["A_Const"]);
    if (node.contains("TypeCast")) return jsonTypeCastToExpr(node["TypeCast"], dsq, schema);
    if (node.contains("FuncCall")) {
        FuncCall fc;
        fc.name = jsonFuncName(node["FuncCall"]);
        if (node["FuncCall"].contains("args")) {
            for (const auto& arg : node["FuncCall"]["args"])
                fc.args.push_back(jsonExprToExpr(arg, dsq, schema));
        }
        auto out = std::make_shared<Expr>();
        out->node = std::move(fc);
        return out;
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string op = jsonAExprOp(ae);
        if (op == "+" || op == "-" || op == "*" || op == "/") {
            auto left = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto right = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
            ExprOp eop = ExprOp::ADD;
            if (op == "-") eop = ExprOp::SUB;
            else if (op == "*") eop = ExprOp::MUL;
            else if (op == "/") eop = ExprOp::DIV;
            if ((eop == ExprOp::ADD || eop == ExprOp::SUB)) {
                auto* l = std::get_if<Literal>(&left->node);
                auto* r = std::get_if<Literal>(&right->node);
                if (l && r) {
                    auto* dateVal = std::get_if<int>(&l->value);
                    auto* intervalVal = std::get_if<int>(&r->value);
                    if (dateVal && intervalVal && *dateVal > 19000101 && *dateVal < 21001231) {
                        int years = *intervalVal;
                        if (eop == ExprOp::SUB) years = -years;
                        return Expr::lit(dateAddYearsLocal(*dateVal, years));
                    }
                }
            }
            return Expr::binary(eop, left, right);
        }
    }
    return Expr::lit(0);
}

static CmpOp cmpOpFromJson(const std::string& op) {
    if (op == "=") return CmpOp::EQ;
    if (op == "<>" || op == "!=") return CmpOp::NE;
    if (op == "<") return CmpOp::LT;
    if (op == "<=") return CmpOp::LE;
    if (op == ">") return CmpOp::GT;
    if (op == ">=") return CmpOp::GE;
    return CmpOp::EQ;
}

static PredPtr jsonPredToPred(const nlohmann::json& node,
                              const DecorrelatedScalarSubquery& dsq,
                              const SchemaProvider* schema) {
    if (node.contains("BoolExpr")) {
        const auto& be = node["BoolExpr"];
        std::string op = be.value("boolop", "AND_EXPR");
        std::vector<PredPtr> children;
        if (be.contains("args")) {
            for (const auto& arg : be["args"])
                children.push_back(jsonPredToPred(arg, dsq, schema));
        }
        if (op == "OR_EXPR") return Predicate::logOr(std::move(children));
        if (op == "NOT_EXPR" && !children.empty()) return Predicate::logNot(children.front());
        return Predicate::logAnd(std::move(children));
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string kind = ae.value("kind", "AEXPR_OP");
        std::string op = jsonAExprOp(ae);
        if (kind == "AEXPR_IN") {
            auto expr = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            std::vector<ExprPtr> values;
            if (ae.contains("rexpr") && ae["rexpr"].contains("List") &&
                ae["rexpr"]["List"].contains("items")) {
                for (const auto& item : ae["rexpr"]["List"]["items"])
                    values.push_back(jsonExprToExpr(item, dsq, schema));
            }
            auto pred = Predicate::inList(expr, std::move(values));
            if (op == "<>" || op == "!=") return Predicate::logNot(pred);
            return pred;
        }
        if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
            auto expr = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto patExpr = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
            std::string pat;
            if (auto* lit = std::get_if<Literal>(&patExpr->node))
                if (auto* sv = std::get_if<std::string>(&lit->value)) pat = *sv;
            return Predicate::like(expr, pat, op == "!~~" || op == "!~~*");
        }
        auto left = jsonExprToExpr(ae.value("lexpr", nlohmann::json{}), dsq, schema);
        auto right = jsonExprToExpr(ae.value("rexpr", nlohmann::json{}), dsq, schema);
        return Predicate::cmp(cmpOpFromJson(op), left, right);
    }
    return Predicate::cmp(CmpOp::EQ, Expr::lit(1), Expr::lit(1));
}

static void collectJsonConjuncts(const nlohmann::json& node,
                                 std::vector<nlohmann::json>& out) {
    if (node.contains("BoolExpr") && node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        for (const auto& arg : node["BoolExpr"]["args"]) collectJsonConjuncts(arg, out);
        return;
    }
    out.push_back(node);
}

static void collectPredTables(const PredPtr& pred, std::set<std::string>& tables) {
    std::map<std::string, std::string> colToTable;
    collectColumnTables(pred, colToTable);
    for (const auto& [_, table] : colToTable) {
        if (!table.empty()) tables.insert(table);
    }
}


static bool extractDecorrelatedAggTarget(const nlohmann::json& node,
                                         DecorrelatedScalarSubquery& dsq,
                                         const SchemaProvider* schema,
                                         float multiplier = 1.0f) {
    if (!node.is_object()) return false;
    if (node.contains("TypeCast"))
        return extractDecorrelatedAggTarget(node["TypeCast"].value("arg", nlohmann::json{}),
                                            dsq, schema, multiplier);
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        std::string name = jsonFuncName(fc);
        if (!isAggregateFuncCallName(name)) return false;
        if (name == "sum") dsq.func = AggFunc::SUM;
        else if (name == "avg") dsq.func = AggFunc::AVG;
        else if (name == "min") dsq.func = AggFunc::MIN;
        else if (name == "max") dsq.func = AggFunc::MAX;
        else if (name == "count") dsq.func = AggFunc::COUNT;
        dsq.multiplier = multiplier;
        dsq.countStar = !fc.contains("args") || fc["args"].empty();
        if (!dsq.countStar) {
            auto raw = jsonRawColumnRef(fc["args"][0]);
            if (!raw) return false;
            auto col = resolveDecorrCol(*raw, dsq, schema);
            if (!col.inner) return false;
            dsq.valueTable = col.table;
            dsq.valueCol = col.column;
        }
        return true;
    }
    if (!node.contains("A_Expr")) return false;
    const auto& ae = node["A_Expr"];
    std::string op = jsonAExprOp(ae);
    if (op == "*") {
        if (auto lit = jsonNumericConst(ae.value("lexpr", nlohmann::json{})))
            return extractDecorrelatedAggTarget(ae.value("rexpr", nlohmann::json{}),
                                                dsq, schema, multiplier * (float)*lit);
        if (auto lit = jsonNumericConst(ae.value("rexpr", nlohmann::json{})))
            return extractDecorrelatedAggTarget(ae.value("lexpr", nlohmann::json{}),
                                                dsq, schema, multiplier * (float)*lit);
    }
    if (op == "/") {
        if (auto lit = jsonNumericConst(ae.value("rexpr", nlohmann::json{}))) {
            if (*lit != 0.0)
                return extractDecorrelatedAggTarget(ae.value("lexpr", nlohmann::json{}),
                                                    dsq, schema, multiplier / (float)*lit);
        }
    }
    return false;
}


} // namespace

std::optional<DecorrelatedScalarSubquery> parseDecorrelatedScalarSubquery(
        const std::string& sqlJson,
        const AnalyzedQuery& aq,
        int sqIdx) {
    nlohmann::json root;
    try { root = nlohmann::json::parse(sqlJson); } catch (...) { return std::nullopt; }
    if (!root.contains("SelectStmt")) return std::nullopt;
    const auto& ss = root["SelectStmt"];

    DecorrelatedScalarSubquery dsq;
    dsq.sqIdx = sqIdx;
    if (!ss.contains("fromClause") || !ss["fromClause"].is_array()) return std::nullopt;
    for (const auto& from : ss["fromClause"]) {
        if (!from.contains("RangeVar")) continue;
        const auto& rv = from["RangeVar"];
        std::string rel = rv.value("relname", "");
        if (rel.empty()) continue;
        dsq.tables.push_back(rel);
        dsq.aliases[rel] = rel;
        if (rv.contains("alias")) {
            if (rv["alias"].contains("Alias"))
                dsq.aliases[rv["alias"]["Alias"].value("aliasname", rel)] = rel;
            else if (rv["alias"].contains("aliasname"))
                dsq.aliases[rv["alias"].value("aliasname", rel)] = rel;
        }
    }
    if (dsq.tables.empty()) return std::nullopt;

    if (!ss.contains("targetList") || !ss["targetList"].is_array() || ss["targetList"].empty())
        return std::nullopt;
    bool foundAgg = false;
    for (const auto& target : ss["targetList"]) {
        if (!target.contains("ResTarget") || !target["ResTarget"].contains("val")) continue;
        if (extractDecorrelatedAggTarget(target["ResTarget"]["val"], dsq, aq.schema)) {
            foundAgg = true;
            break;
        }
    }
    if (!foundAgg) return std::nullopt;

    std::vector<nlohmann::json> conjuncts;
    if (ss.contains("whereClause")) collectJsonConjuncts(ss["whereClause"], conjuncts);
    for (const auto& predJson : conjuncts) {
        bool classified = false;
        if (predJson.contains("A_Expr") && jsonAExprOp(predJson["A_Expr"]) == "=") {
            auto leftRaw = jsonRawColumnRef(predJson["A_Expr"].value("lexpr", nlohmann::json{}));
            auto rightRaw = jsonRawColumnRef(predJson["A_Expr"].value("rexpr", nlohmann::json{}));
            if (leftRaw && rightRaw) {
                auto left = resolveDecorrCol(*leftRaw, dsq, aq.schema);
                auto right = resolveDecorrCol(*rightRaw, dsq, aq.schema);
                if (left.inner && right.inner) {
                    if (left.table != right.table || left.column != right.column)
                        dsq.joins.push_back({left, right});
                    classified = true;
                } else if (left.inner != right.inner) {
                    dsq.correlations.push_back(left.inner
                        ? DecorrCorrelation{left, right}
                        : DecorrCorrelation{right, left});
                    classified = true;
                }
            }
        }
        if (classified) continue;

        auto pred = jsonPredToPred(predJson, dsq, aq.schema);
        std::set<std::string> predTables;
        collectPredTables(pred, predTables);
        if (predTables.size() != 1) return std::nullopt;
        dsq.filtersByTable[*predTables.begin()].push_back(pred);
    }

    if (dsq.correlations.size() > 2) return std::nullopt;
    if (dsq.countStar && dsq.valueTable.empty())
        dsq.valueTable = dsq.correlations.empty() ? dsq.tables.front()
                                                  : dsq.correlations.front().inner.table;
    if (dsq.valueTable.empty()) return std::nullopt;
    return dsq;
}


} // namespace codegen
