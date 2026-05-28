#include "generic/ir/generic_scalar_subquery_analysis.h"

#include "core/schema_provider.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
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

static TypeInfo intType() {
    return TypeInfo{DataType::INT, 0};
}

static TypeInfo floatType() {
    return TypeInfo{DataType::FLOAT, 0};
}

static TypeInfo stringType(size_t len) {
    return TypeInfo{DataType::CHAR_FIXED, static_cast<int>(len)};
}

template <typename Node>
static GenericExprPtr makeGenericExpr(TypeInfo type, Node node) {
    auto out = std::make_shared<GenericExpr>();
    out->type = type;
    out->node = std::move(node);
    return out;
}

template <typename Node>
static GenericPredicatePtr makeGenericPredicate(Node node) {
    auto out = std::make_shared<GenericPredicate>();
    out->node = std::move(node);
    return out;
}

static GenericExprPtr genericIntLiteral(int64_t value) {
    TypeInfo type = intType();
    return makeGenericExpr(type, GenericLiteralExpr{value, type});
}

static GenericExprPtr genericDateLiteral(int64_t value) {
    TypeInfo type{DataType::DATE, 0};
    return makeGenericExpr(type, GenericLiteralExpr{value, type});
}

static GenericExprPtr genericFloatLiteral(double value) {
    TypeInfo type = floatType();
    return makeGenericExpr(type, GenericLiteralExpr{value, type});
}

static GenericExprPtr genericStringLiteral(std::string value) {
    TypeInfo type = stringType(value.size());
    return makeGenericExpr(type, GenericLiteralExpr{std::move(value), type});
}

static std::optional<std::string> genericStringLiteralValue(
        const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    auto* value = std::get_if<std::string>(&lit->value);
    if (!value) return std::nullopt;
    return *value;
}

static std::optional<int64_t> genericIntLiteralValue(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    auto* value = std::get_if<int64_t>(&lit->value);
    if (!value) return std::nullopt;
    return *value;
}

static TypeInfo typeForDecorrCol(const DecorrCol& col,
                                 const SchemaProvider* schema) {
    if (col.inner && schema && schema->hasColumn(col.table, col.column)) {
        DataType dt = schema->columnType(col.table, col.column);
        int fixedWidth = dt == DataType::CHAR_FIXED
            ? schema->columnFixedWidth(col.table, col.column)
            : 0;
        return TypeInfo{dt, fixedWidth};
    }
    return intType();
}

static GenericExprPtr genericColExpr(const DecorrCol& col,
                                     const SchemaProvider* schema) {
    GenericColumnExpr out;
    out.table = col.inner ? col.table : "";
    out.alias = col.qualifier;
    out.column = col.column;
    out.type = typeForDecorrCol(col, schema);
    if (col.inner && schema && schema->hasColumn(col.table, col.column)) {
        if (auto gd = schema->groupDomain(col.table, col.column)) {
            out.hasGroupDomain = true;
            out.domainMin = gd->minValue;
            out.domainMax = gd->maxValue;
        }
        out.charDomain = schema->charDomain(col.table, col.column);
        out.numericScale = schema->numericScale(col.table, col.column);
        out.keyDomainSymbol = schema->keyDomainSymbol(col.table, col.column);
        out.distinctDomainSymbol =
            schema->distinctDomainSymbol(col.table, col.column);
    }
    return makeGenericExpr(out.type, std::move(out));
}

static GenericExprPtr jsonExprToGenericExpr(
    const nlohmann::json& node,
    const DecorrelatedScalarSubquery& dsq,
    const SchemaProvider* schema);

static GenericExprPtr jsonConstToGenericExpr(const nlohmann::json& ac) {
    if (ac.contains("ival")) {
        const auto& iv = ac["ival"];
        if (iv.is_object() && iv.contains("ival"))
            return genericIntLiteral(iv["ival"].get<int64_t>());
        if (iv.is_number_integer()) return genericIntLiteral(iv.get<int64_t>());
        return genericIntLiteral(0);
    }
    if (ac.contains("fval")) {
        const auto& fv = ac["fval"];
        if (fv.is_object() && fv.contains("fval"))
            return genericFloatLiteral(std::stod(fv["fval"].get<std::string>()));
        if (fv.is_number()) return genericFloatLiteral(fv.get<double>());
        return genericFloatLiteral(0.0);
    }
    if (ac.contains("sval")) {
        const auto& sv = ac["sval"];
        if (sv.is_object() && sv.contains("sval"))
            return genericStringLiteral(sv["sval"].get<std::string>());
        if (sv.is_string()) return genericStringLiteral(sv.get<std::string>());
    }
    if (ac.contains("val")) {
        const auto& val = ac["val"];
        if (val.contains("Integer")) {
            const auto& iv = val["Integer"].at("ival");
            if (iv.is_number_integer()) return genericIntLiteral(iv.get<int64_t>());
            if (iv.is_string()) return genericIntLiteral(std::stoll(iv.get<std::string>()));
        }
        if (val.contains("Float")) {
            const auto& fv = val["Float"].at("fval");
            if (fv.is_number()) return genericFloatLiteral(fv.get<double>());
            if (fv.is_string()) return genericFloatLiteral(std::stod(fv.get<std::string>()));
        }
        if (val.contains("String")) {
            return genericStringLiteral(val["String"].at("sval").get<std::string>());
        }
    }
    return genericIntLiteral(0);
}

static GenericExprPtr jsonTypeCastToGenericExpr(
        const nlohmann::json& tc,
        const DecorrelatedScalarSubquery& dsq,
        const SchemaProvider* schema) {
    std::string typ;
    if (tc.contains("typeName") && tc["typeName"].contains("names")) {
        for (const auto& n : tc["typeName"]["names"]) {
            if (auto s = jsonStringValue(n)) typ = *s;
        }
    }
    auto arg = jsonExprToGenericExpr(tc.value("arg", nlohmann::json{}),
                                     dsq, schema);
    if (typ == "date") {
        if (auto sv = genericStringLiteralValue(arg))
            return genericDateLiteral(parseDateLiteralLocal(*sv));
    }
    if (typ == "interval") {
        if (auto sv = genericStringLiteralValue(arg))
            return genericIntLiteral(std::stoi(*sv));
        if (auto iv = genericIntLiteralValue(arg)) return genericIntLiteral(*iv);
    }
    return arg;
}

static TypeInfo binaryExprType(ExprOp op,
                               const GenericExprPtr& left,
                               const GenericExprPtr& right) {
    TypeInfo leftType = left ? left->type : intType();
    TypeInfo rightType = right ? right->type : intType();
    if (op == ExprOp::ADD || op == ExprOp::SUB) {
        if (leftType.type == DataType::DATE || rightType.type == DataType::DATE)
            return TypeInfo{DataType::DATE, 0};
    }
    if (leftType.type == DataType::FLOAT || rightType.type == DataType::FLOAT)
        return floatType();
    return leftType;
}

static GenericExprPtr jsonExprToGenericExpr(
        const nlohmann::json& node,
        const DecorrelatedScalarSubquery& dsq,
        const SchemaProvider* schema) {
    if (node.contains("ColumnRef")) {
        auto raw = jsonRawColumnRef(node);
        if (!raw) return genericIntLiteral(0);
        auto col = resolveDecorrCol(*raw, dsq, schema);
        return genericColExpr(col, schema);
    }
    if (node.contains("A_Const"))
        return jsonConstToGenericExpr(node["A_Const"]);
    if (node.contains("TypeCast"))
        return jsonTypeCastToGenericExpr(node["TypeCast"], dsq, schema);
    if (node.contains("FuncCall")) {
        GenericFunctionExpr fc;
        fc.name = jsonFuncName(node["FuncCall"]);
        if (node["FuncCall"].contains("args")) {
            for (const auto& arg : node["FuncCall"]["args"])
                fc.args.push_back(jsonExprToGenericExpr(arg, dsq, schema));
        }
        fc.type = intType();
        return makeGenericExpr(fc.type, std::move(fc));
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string op = jsonAExprOp(ae);
        if (op == "+" || op == "-" || op == "*" || op == "/") {
            auto left = jsonExprToGenericExpr(
                ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto right = jsonExprToGenericExpr(
                ae.value("rexpr", nlohmann::json{}), dsq, schema);
            ExprOp eop = ExprOp::ADD;
            if (op == "-") eop = ExprOp::SUB;
            else if (op == "*") eop = ExprOp::MUL;
            else if (op == "/") eop = ExprOp::DIV;
            if ((eop == ExprOp::ADD || eop == ExprOp::SUB)) {
                auto dateVal = genericIntLiteralValue(left);
                auto intervalVal = genericIntLiteralValue(right);
                if (dateVal && intervalVal &&
                    *dateVal > 19000101 && *dateVal < 21001231) {
                    int years = static_cast<int>(*intervalVal);
                    if (eop == ExprOp::SUB) years = -years;
                    return genericDateLiteral(dateAddYearsLocal(
                        static_cast<int>(*dateVal), years));
                }
            }
            TypeInfo type = binaryExprType(eop, left, right);
            return makeGenericExpr(
                type, GenericBinaryExpr{eop, std::move(left), std::move(right), type});
        }
    }
    return genericIntLiteral(0);
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

static GenericPredicatePtr jsonPredToGenericPred(
        const nlohmann::json& node,
        const DecorrelatedScalarSubquery& dsq,
        const SchemaProvider* schema) {
    if (node.contains("BoolExpr")) {
        const auto& be = node["BoolExpr"];
        std::string op = be.value("boolop", "AND_EXPR");
        std::vector<GenericPredicatePtr> children;
        if (be.contains("args")) {
            for (const auto& arg : be["args"])
                children.push_back(jsonPredToGenericPred(arg, dsq, schema));
        }
        GenericLogicalPred pred;
        if (op == "OR_EXPR") pred.op = GenericLogicalPred::Op::Or;
        else if (op == "NOT_EXPR") pred.op = GenericLogicalPred::Op::Not;
        else pred.op = GenericLogicalPred::Op::And;
        pred.children = std::move(children);
        return makeGenericPredicate(std::move(pred));
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        std::string kind = ae.value("kind", "AEXPR_OP");
        std::string op = jsonAExprOp(ae);
        if (kind == "AEXPR_IN") {
            auto expr = jsonExprToGenericExpr(
                ae.value("lexpr", nlohmann::json{}), dsq, schema);
            std::vector<GenericExprPtr> values;
            if (ae.contains("rexpr") && ae["rexpr"].contains("List") &&
                ae["rexpr"]["List"].contains("items")) {
                for (const auto& item : ae["rexpr"]["List"]["items"])
                    values.push_back(jsonExprToGenericExpr(item, dsq, schema));
            }
            auto pred = makeGenericPredicate(
                GenericInListPred{std::move(expr), std::move(values)});
            if (op == "<>" || op == "!=") {
                GenericLogicalPred notPred;
                notPred.op = GenericLogicalPred::Op::Not;
                notPred.children.push_back(pred);
                return makeGenericPredicate(std::move(notPred));
            }
            return pred;
        }
        if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
            auto expr = jsonExprToGenericExpr(
                ae.value("lexpr", nlohmann::json{}), dsq, schema);
            auto patExpr = jsonExprToGenericExpr(
                ae.value("rexpr", nlohmann::json{}), dsq, schema);
            std::string pat = genericStringLiteralValue(patExpr).value_or("");
            return makeGenericPredicate(
                GenericLikePred{std::move(expr), pat, op == "!~~" || op == "!~~*"});
        }
        auto left = jsonExprToGenericExpr(
            ae.value("lexpr", nlohmann::json{}), dsq, schema);
        auto right = jsonExprToGenericExpr(
            ae.value("rexpr", nlohmann::json{}), dsq, schema);
        return makeGenericPredicate(
            GenericComparisonPred{cmpOpFromJson(op), std::move(left), std::move(right)});
    }
    return makeGenericPredicate(
        GenericComparisonPred{CmpOp::EQ, genericIntLiteral(1), genericIntLiteral(1)});
}

static void collectJsonConjuncts(const nlohmann::json& node,
                                 std::vector<nlohmann::json>& out) {
    if (node.contains("BoolExpr") && node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        for (const auto& arg : node["BoolExpr"]["args"]) collectJsonConjuncts(arg, out);
        return;
    }
    out.push_back(node);
}

static void collectGenericExprTables(const GenericExprPtr& expr,
                                     std::set<std::string>& tables);

static void collectGenericPredTables(const GenericPredicatePtr& pred,
                                     std::set<std::string>& tables) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectGenericExprTables(node.left, tables);
            collectGenericExprTables(node.right, tables);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectGenericExprTables(node.expr, tables);
            collectGenericExprTables(node.low, tables);
            collectGenericExprTables(node.high, tables);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectGenericExprTables(node.expr, tables);
            for (const auto& value : node.values)
                collectGenericExprTables(value, tables);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectGenericExprTables(node.expr, tables);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectGenericPredTables(child, tables);
        }
    }, pred->node);
}

static void collectGenericExprTables(const GenericExprPtr& expr,
                                     std::set<std::string>& tables) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (!node.table.empty()) tables.insert(node.table);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectGenericExprTables(node.left, tables);
            collectGenericExprTables(node.right, tables);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectGenericPredTables(branch.condition, tables);
                collectGenericExprTables(branch.result, tables);
            }
            collectGenericExprTables(node.elseResult, tables);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectGenericExprTables(arg, tables);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectGenericExprTables(node.arg, tables);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectGenericExprTables(key, tables);
        }
    }, expr->node);
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
        const SchemaProvider* schema,
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
        if (extractDecorrelatedAggTarget(target["ResTarget"]["val"], dsq, schema)) {
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
                auto left = resolveDecorrCol(*leftRaw, dsq, schema);
                auto right = resolveDecorrCol(*rightRaw, dsq, schema);
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

        auto pred = jsonPredToGenericPred(predJson, dsq, schema);
        std::set<std::string> predTables;
        collectGenericPredTables(pred, predTables);
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
