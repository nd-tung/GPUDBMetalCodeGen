#include "generic/ir/generic_ir_builder.h"
#include "generic/ir/analyzed_query.h"
#include "generic/ir/generic_scalar_subquery_analysis.h"
#include "core/schema_provider.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>

namespace codegen {

namespace {

TypeInfo intType() {
    return TypeInfo{DataType::INT, 0};
}

TypeInfo floatType() {
    return TypeInfo{DataType::FLOAT, 0};
}

TypeInfo typeFromColumn(const ColRef& col) {
    return TypeInfo{col.dataType, col.fixedWidth};
}

GenericColumnExpr columnExprFromColRef(const ColRef& col,
                                       GenericRelationInstanceId inst,
                                       const SchemaProvider* schema) {
    TypeInfo type = typeFromColumn(col);
    GenericColumnExpr out;
    out.relationInstance = inst;
    out.table = col.table;
    out.alias = col.tableAlias;
    out.column = col.column;
    out.type = type;
    if (schema && !col.table.empty() && schema->hasColumn(col.table, col.column)) {
        if (auto domain = schema->groupDomain(col.table, col.column)) {
            out.hasGroupDomain = true;
            out.domainMin = domain->minValue;
            out.domainMax = domain->maxValue;
        }
        out.charDomain = schema->charDomain(col.table, col.column);
        out.numericScale = schema->numericScale(col.table, col.column);
        out.keyDomainSymbol = schema->keyDomainSymbol(col.table, col.column);
        out.distinctDomainSymbol = schema->distinctDomainSymbol(col.table, col.column);
    }
    return out;
}

TypeInfo typeFromLiteral(const Literal& lit) {
    if (std::holds_alternative<float>(lit.value)) return floatType();
    if (std::holds_alternative<std::string>(lit.value)) {
        const auto& s = std::get<std::string>(lit.value);
        return TypeInfo{DataType::CHAR_FIXED, static_cast<int>(s.size())};
    }
    return intType();
}

TypeInfo typeFromExpr(const ExprPtr& expr) {
    if (!expr) return intType();
    return std::visit([&](const auto& node) -> TypeInfo {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            return typeFromColumn(node);
        } else if constexpr (std::is_same_v<T, Literal>) {
            return typeFromLiteral(node);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            TypeInfo lt = typeFromExpr(node.left);
            TypeInfo rt = typeFromExpr(node.right);
            if (node.op == ExprOp::DIV ||
                lt.type == DataType::FLOAT ||
                rt.type == DataType::FLOAT) {
                return floatType();
            }
            return intType();
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            if (node.elseResult) return typeFromExpr(node.elseResult);
            if (!node.branches.empty() && node.branches.front().result)
                return typeFromExpr(node.branches.front().result);
            return intType();
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            std::string name = node.name;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name == "avg") return floatType();
            return intType();
        } else if constexpr (std::is_same_v<T, ScalarSubqueryRef>) {
            return floatType();
        }
        return intType();
    }, expr->node);
}

TypeInfo typeFromAgg(AggFunc func, const ExprPtr& expr) {
    if (func == AggFunc::COUNT || func == AggFunc::COUNT_DISTINCT)
        return intType();
    if (func == AggFunc::AVG)
        return floatType();
    if (expr) return typeFromExpr(expr);
    return intType();
}

std::string targetName(const SelectTarget& target, size_t index) {
    if (!target.alias.empty()) return target.alias;
    if (target.agg && !target.agg->alias.empty()) return target.agg->alias;
    if (target.expr) {
        if (auto* col = std::get_if<ColRef>(&target.expr->node))
            return col->column;
        if (auto* fn = std::get_if<FuncCall>(&target.expr->node)) {
            std::string fnName = fn->name;
            std::transform(fnName.begin(), fnName.end(), fnName.begin(), ::tolower);
            const bool aggregate =
                fnName == "sum" || fnName == "count" || fnName == "avg" ||
                fnName == "min" || fnName == "max";
            if (aggregate && fn->args.empty()) return fnName + "(*)";
            if (aggregate && fn->args.size() == 1 && fn->args.front()) {
                if (auto* argCol = std::get_if<ColRef>(&fn->args.front()->node))
                    return fnName + "(" + argCol->column + ")";
            }
            return fn->name;
        }
    }
    return "expr_" + std::to_string(index);
}

std::string groupName(const ExprPtr& expr, size_t index) {
    if (expr) {
        if (auto* col = std::get_if<ColRef>(&expr->node))
            return col->column;
        if (auto* fn = std::get_if<FuncCall>(&expr->node))
            return fn->name;
    }
    return "group_" + std::to_string(index);
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

bool sqlExprEquivalent(const ExprPtr& left, const ExprPtr& right) {
    if (!left || !right) return !left && !right;
    return std::visit([&](const auto& lnode) -> bool {
        using L = std::decay_t<decltype(lnode)>;
        auto* rnode = std::get_if<L>(&right->node);
        if (!rnode) return false;

        if constexpr (std::is_same_v<L, ColRef>) {
            if (lnode.column != rnode->column) return false;
            if (!lnode.table.empty() && !rnode->table.empty() &&
                lnode.table != rnode->table) {
                return false;
            }
            if (!lnode.tableAlias.empty() && !rnode->tableAlias.empty() &&
                lnode.tableAlias != rnode->tableAlias) {
                return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, Literal>) {
            return lnode.value == rnode->value;
        } else if constexpr (std::is_same_v<L, BinaryExpr>) {
            return lnode.op == rnode->op &&
                   sqlExprEquivalent(lnode.left, rnode->left) &&
                   sqlExprEquivalent(lnode.right, rnode->right);
        } else if constexpr (std::is_same_v<L, FuncCall>) {
            if (lowerAscii(lnode.name) != lowerAscii(rnode->name) ||
                lnode.args.size() != rnode->args.size()) {
                return false;
            }
            for (size_t i = 0; i < lnode.args.size(); ++i) {
                if (!sqlExprEquivalent(lnode.args[i], rnode->args[i]))
                    return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, CaseWhen>) {
            return false;
        } else if constexpr (std::is_same_v<L, ScalarSubqueryRef>) {
            return lnode.index == rnode->index;
        }
        return false;
    }, left->node);
}

std::string groupOutputNameForExpr(const AnalyzedQuery& aq,
                                   const ExprPtr& expr,
                                   size_t groupIndex) {
    for (size_t i = 0; i < aq.targets.size(); ++i) {
        const auto& target = aq.targets[i];
        if (target.isAgg) continue;
        if (sqlExprEquivalent(target.expr, expr))
            return targetName(target, i);
    }
    return groupName(expr, groupIndex);
}

bool aggregateFuncName(std::string name, AggFunc& out) {
    name = lowerAscii(std::move(name));
    if (name == "sum") {
        out = AggFunc::SUM;
        return true;
    }
    if (name == "count") {
        out = AggFunc::COUNT;
        return true;
    }
    if (name == "avg") {
        out = AggFunc::AVG;
        return true;
    }
    if (name == "min") {
        out = AggFunc::MIN;
        return true;
    }
    if (name == "max") {
        out = AggFunc::MAX;
        return true;
    }
    return false;
}

std::string formatSignatureDouble(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::optional<std::string> jsonStringValueForGenericSource(
        const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") &&
        node["String"].contains("sval")) {
        return node["String"]["sval"].get<std::string>();
    }
    return std::nullopt;
}

std::string jsonAExprOpForGenericSource(const nlohmann::json& ae) {
    if (!ae.contains("name") || !ae["name"].is_array() || ae["name"].empty())
        return {};
    if (auto s = jsonStringValueForGenericSource(ae["name"][0])) return *s;
    return {};
}

std::string jsonFuncNameForGenericSource(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() ||
        fc["funcname"].empty()) {
        return {};
    }
    auto s = jsonStringValueForGenericSource(fc["funcname"].back());
    return s ? lowerAscii(*s) : "";
}

std::optional<double> jsonNumericConstForGenericSource(
        const nlohmann::json& node) {
    const nlohmann::json* ac = nullptr;
    if (node.is_object() && node.contains("A_Const")) ac = &node["A_Const"];
    else if (node.is_object()) ac = &node;
    if (!ac) return std::nullopt;

    auto readNum = [](const nlohmann::json& v) -> std::optional<double> {
        try {
            if (v.is_number()) return v.get<double>();
            if (v.is_string()) return std::stod(v.get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
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

std::string resolveJsonColumnTableForGenericSource(
        const std::string& qualifier,
        const std::string& column,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    if (!qualifier.empty()) {
        auto it = aliases.find(qualifier);
        return it == aliases.end() ? qualifier : it->second;
    }

    if (!schema) return "";
    std::string match;
    for (const auto& table : tables) {
        if (!schema->hasColumn(table, column)) continue;
        if (!match.empty()) return "";
        match = table;
    }
    return match;
}

struct JsonColumnRefForGenericSource {
    std::string qualifier;
    std::string column;
};

std::optional<JsonColumnRefForGenericSource> jsonRawColumnRefForGenericSource(
        const nlohmann::json& node) {
    const nlohmann::json* cr = nullptr;
    if (node.is_object() && node.contains("ColumnRef")) cr = &node["ColumnRef"];
    else if (node.is_object() && node.contains("fields")) cr = &node;
    if (!cr || !cr->contains("fields") || !(*cr)["fields"].is_array())
        return std::nullopt;

    std::vector<std::string> fields;
    for (const auto& field : (*cr)["fields"]) {
        if (auto s = jsonStringValueForGenericSource(field)) fields.push_back(*s);
    }
    if (fields.empty()) return std::nullopt;
    JsonColumnRefForGenericSource out;
    out.column = fields.back();
    if (fields.size() >= 2) out.qualifier = fields[fields.size() - 2];
    return out;
}

std::optional<std::string> jsonColumnSignatureForGenericSource(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    auto col = jsonRawColumnRefForGenericSource(node);
    if (!col) return std::nullopt;
    const std::string table = resolveJsonColumnTableForGenericSource(
        col->qualifier, col->column, aliases, tables, schema);
    return table.empty() ? "col:" + col->column
                         : "col:" + table + "." + col->column;
}

std::string combineBinarySignature(std::string op,
                                   std::string left,
                                   std::string right) {
    if (op == "+" || op == "*") {
        if (right < left) std::swap(left, right);
    }
    return "bin:" + op + "(" + left + "," + right + ")";
}

std::optional<std::string> jsonExprSignatureForGenericSource(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    if (!node.is_object()) return std::nullopt;
    if (node.contains("TypeCast")) {
        return jsonExprSignatureForGenericSource(
            node["TypeCast"].value("arg", nlohmann::json{}),
            aliases, tables, schema);
    }
    if (node.contains("ColumnRef") || node.contains("fields"))
        return jsonColumnSignatureForGenericSource(node, aliases, tables, schema);
    if (node.contains("A_Const")) {
        const auto& ac = node["A_Const"];
        try {
            auto readStringConst = [](const nlohmann::json& value)
                    -> std::optional<std::string> {
                if (value.is_string()) return value.get<std::string>();
                if (value.is_object() && value.contains("sval"))
                    return value["sval"].get<std::string>();
                return std::nullopt;
            };
            auto readIntConst = [](const nlohmann::json& value)
                    -> std::optional<int64_t> {
                if (value.is_number_integer()) return value.get<int64_t>();
                if (value.is_string()) return std::stoll(value.get<std::string>());
                if (value.is_object() && value.contains("ival")) {
                    const auto& inner = value["ival"];
                    if (inner.is_number_integer()) return inner.get<int64_t>();
                    if (inner.is_string()) return std::stoll(inner.get<std::string>());
                }
                return std::nullopt;
            };
            auto readFloatConst = [](const nlohmann::json& value)
                    -> std::optional<double> {
                if (value.is_number()) return value.get<double>();
                if (value.is_string()) return std::stod(value.get<std::string>());
                if (value.is_object() && value.contains("fval")) {
                    const auto& inner = value["fval"];
                    if (inner.is_number()) return inner.get<double>();
                    if (inner.is_string()) return std::stod(inner.get<std::string>());
                }
                return std::nullopt;
            };
            if (ac.contains("sval")) {
                if (auto s = readStringConst(ac["sval"])) return "lit:s:" + *s;
            }
            if (ac.contains("ival")) {
                if (auto i = readIntConst(ac["ival"]))
                    return "lit:i:" + std::to_string(*i);
            }
            if (ac.contains("fval")) {
                if (auto f = readFloatConst(ac["fval"]))
                    return "lit:f:" + formatSignatureDouble(*f);
            }
            if (ac.contains("val") && ac["val"].contains("String"))
                return "lit:s:" + ac["val"]["String"].at("sval").get<std::string>();
            if (ac.contains("val") && ac["val"].contains("Integer"))
                return "lit:i:" + std::to_string(
                    ac["val"]["Integer"].at("ival").get<int64_t>());
            if (ac.contains("val") && ac["val"].contains("Float"))
                return "lit:f:" + formatSignatureDouble(std::stod(
                    ac["val"]["Float"].at("fval").get<std::string>()));
        } catch (...) {
            return std::nullopt;
        }
        return std::nullopt;
    }
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        const std::string name = jsonFuncNameForGenericSource(fc);
        std::vector<std::string> args;
        if (fc.contains("args") && fc["args"].is_array()) {
            for (const auto& arg : fc["args"]) {
                if (arg.is_object() && arg.contains("A_Star")) {
                    args.push_back("*");
                    continue;
                }
                auto sig = jsonExprSignatureForGenericSource(
                    arg, aliases, tables, schema);
                if (!sig) return std::nullopt;
                args.push_back(*sig);
            }
        }
        std::ostringstream oss;
        oss << "fn:" << name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i) oss << ",";
            oss << args[i];
        }
        oss << ")";
        return oss.str();
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        const std::string op = jsonAExprOpForGenericSource(ae);
        auto left = jsonExprSignatureForGenericSource(
            ae.value("lexpr", nlohmann::json{}), aliases, tables, schema);
        auto right = jsonExprSignatureForGenericSource(
            ae.value("rexpr", nlohmann::json{}), aliases, tables, schema);
        if (!left || !right || op.empty()) return std::nullopt;
        return combineBinarySignature(op, *left, *right);
    }
    return std::nullopt;
}

std::string predicateSignatureFromComparison(std::string op,
                                             std::string left,
                                             std::string right) {
    if (op == "=") op = "==";
    if (op == "<>") op = "!=";
    if (op == "==") {
        if (right < left) std::swap(left, right);
    }
    return "cmp:" + op + "(" + left + "," + right + ")";
}

bool collectJsonPredicateAtomSignaturesForGenericSource(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        std::vector<std::string>& out) {
    if (!node.is_object()) return false;
    if (node.contains("BoolExpr") &&
        node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        const auto& args = node["BoolExpr"].value("args", nlohmann::json::array());
        for (const auto& arg : args) {
            if (!collectJsonPredicateAtomSignaturesForGenericSource(
                    arg, aliases, tables, schema, out)) {
                return false;
            }
        }
        return true;
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        const std::string op = jsonAExprOpForGenericSource(ae);
        auto left = jsonExprSignatureForGenericSource(
            ae.value("lexpr", nlohmann::json{}), aliases, tables, schema);
        auto right = jsonExprSignatureForGenericSource(
            ae.value("rexpr", nlohmann::json{}), aliases, tables, schema);
        if (!left || !right || op.empty()) return false;
        out.push_back(predicateSignatureFromComparison(op, *left, *right));
        return true;
    }
    return false;
}

bool extractJsonScalarAggTargetForGenericSource(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        GenericScalarSubqueryAggTarget& out,
        double multiplier = 1.0) {
    if (!node.is_object()) return false;
    if (node.contains("TypeCast")) {
        return extractJsonScalarAggTargetForGenericSource(
            node["TypeCast"].value("arg", nlohmann::json{}), aliases, tables,
            schema, out, multiplier);
    }
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        AggFunc func = AggFunc::SUM;
        if (!aggregateFuncName(jsonFuncNameForGenericSource(fc), func))
            return false;
        out.func = func;
        out.multiplier = multiplier;
        out.star = !fc.contains("args") || !fc["args"].is_array() ||
                   fc["args"].empty();
        if (!out.star) {
            const auto& arg = fc["args"][0];
            out.star = arg.is_object() && arg.contains("A_Star");
        }
        if (!out.star) {
            auto sig = jsonExprSignatureForGenericSource(
                fc["args"][0], aliases, tables, schema);
            if (!sig) return false;
            out.argSignature = *sig;
        }
        return true;
    }
    if (!node.contains("A_Expr")) return false;

    const auto& ae = node["A_Expr"];
    const std::string op = jsonAExprOpForGenericSource(ae);
    if (op == "*") {
        if (auto lit = jsonNumericConstForGenericSource(
                ae.value("lexpr", nlohmann::json{}))) {
            return extractJsonScalarAggTargetForGenericSource(
                ae.value("rexpr", nlohmann::json{}), aliases, tables, schema,
                out, multiplier * *lit);
        }
        if (auto lit = jsonNumericConstForGenericSource(
                ae.value("rexpr", nlohmann::json{}))) {
            return extractJsonScalarAggTargetForGenericSource(
                ae.value("lexpr", nlohmann::json{}), aliases, tables, schema,
                out, multiplier * *lit);
        }
    }
    if (op == "/") {
        if (auto lit = jsonNumericConstForGenericSource(
                ae.value("rexpr", nlohmann::json{}))) {
            if (*lit != 0.0) {
                return extractJsonScalarAggTargetForGenericSource(
                    ae.value("lexpr", nlohmann::json{}), aliases, tables,
                    schema, out, multiplier / *lit);
            }
        }
    }
    return false;
}

std::optional<GenericScalarHavingSubquerySummary>
scalarHavingSummaryFromSqlJson(const std::string& sqlJson,
                               const SchemaProvider* schema) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(sqlJson);
    } catch (...) {
        return std::nullopt;
    }
    if (!root.contains("SelectStmt")) return std::nullopt;
    const auto& ss = root["SelectStmt"];
    if (ss.contains("groupClause") && !ss["groupClause"].is_null())
        return std::nullopt;
    if (ss.contains("havingClause") && !ss["havingClause"].is_null())
        return std::nullopt;
    if (ss.contains("limitCount") && !ss["limitCount"].is_null())
        return std::nullopt;

    GenericScalarHavingSubquerySummary summary;
    std::map<std::string, std::string> aliases;
    if (!ss.contains("fromClause") || !ss["fromClause"].is_array())
        return std::nullopt;
    for (const auto& from : ss["fromClause"]) {
        if (!from.contains("RangeVar")) return std::nullopt;
        const auto& rv = from["RangeVar"];
        const std::string rel = rv.value("relname", "");
        if (rel.empty()) return std::nullopt;
        summary.tables.push_back(rel);
        aliases[rel] = rel;
        if (rv.contains("alias")) {
            if (rv["alias"].contains("Alias")) {
                aliases[rv["alias"]["Alias"].value("aliasname", rel)] = rel;
            } else if (rv["alias"].contains("aliasname")) {
                aliases[rv["alias"].value("aliasname", rel)] = rel;
            }
        }
    }
    if (summary.tables.empty()) return std::nullopt;

    if (!ss.contains("targetList") || !ss["targetList"].is_array())
        return std::nullopt;
    bool foundAgg = false;
    for (const auto& target : ss["targetList"]) {
        if (!target.contains("ResTarget") ||
            !target["ResTarget"].contains("val")) {
            continue;
        }
        if (extractJsonScalarAggTargetForGenericSource(
                target["ResTarget"]["val"], aliases, summary.tables, schema,
                summary.aggregate)) {
            foundAgg = true;
            break;
        }
    }
    if (!foundAgg) return std::nullopt;

    if (ss.contains("whereClause") && !ss["whereClause"].is_null()) {
        if (!collectJsonPredicateAtomSignaturesForGenericSource(
                ss["whereClause"], aliases, summary.tables, schema,
                summary.predicateSignatures)) {
            return std::nullopt;
        }
        std::sort(summary.predicateSignatures.begin(),
                  summary.predicateSignatures.end());
    }
    std::sort(summary.tables.begin(), summary.tables.end());
    return summary;
}

std::vector<GenericFromSubqueryScalarExtremum>
fromSubqueryScalarExtremaFromSqlJson(const std::string& sqlJson) {
    std::vector<GenericFromSubqueryScalarExtremum> out;
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(sqlJson);
    } catch (...) {
        return out;
    }
    if (!root.contains("SelectStmt")) return out;
    const auto& ss = root["SelectStmt"];
    if (!ss.contains("fromClause") || !ss["fromClause"].is_array())
        return out;
    if (!ss.contains("targetList") || !ss["targetList"].is_array() ||
        ss["targetList"].empty()) {
        return out;
    }

    std::vector<std::string> sourceAliases;
    for (const auto& from : ss["fromClause"]) {
        if (!from.contains("RangeVar")) continue;
        const auto& rv = from["RangeVar"];
        std::string rel = rv.value("relname", "");
        if (!rel.empty()) sourceAliases.push_back(std::move(rel));
    }
    if (sourceAliases.empty()) return out;

    for (const auto& target : ss["targetList"]) {
        if (!target.contains("ResTarget")) continue;
        const auto& rt = target["ResTarget"];
        if (!rt.contains("val") || !rt["val"].contains("FuncCall"))
            continue;
        const auto& fc = rt["val"]["FuncCall"];
        const std::string funcName = jsonFuncNameForGenericSource(fc);
        if (funcName != "max" && funcName != "min") continue;
        if (!fc.contains("args") || !fc["args"].is_array() ||
            fc["args"].empty()) {
            continue;
        }
        auto arg = jsonRawColumnRefForGenericSource(fc["args"][0]);
        if (!arg) continue;
        for (const auto& sourceAlias : sourceAliases) {
            out.push_back(GenericFromSubqueryScalarExtremum{
                sourceAlias,
                funcName == "max" ? AggFunc::MAX : AggFunc::MIN,
                arg->column
            });
        }
    }
    return out;
}

CmpOp reverseCmpOpForGenericSource(CmpOp op) {
    switch (op) {
        case CmpOp::LT: return CmpOp::GT;
        case CmpOp::LE: return CmpOp::GE;
        case CmpOp::GT: return CmpOp::LT;
        case CmpOp::GE: return CmpOp::LE;
        default: return op;
    }
}

std::optional<double> numericLiteralValueForGenericSource(const Literal& lit) {
    if (auto* value = std::get_if<int>(&lit.value))
        return static_cast<double>(*value);
    if (auto* value = std::get_if<float>(&lit.value))
        return static_cast<double>(*value);
    return std::nullopt;
}

bool sqlExprIsInSubAggCall(
        const ExprPtr& expr,
        const AnalyzedQuery::InSubqueryAggInfo& info) {
    if (!expr) return false;
    auto* call = std::get_if<FuncCall>(&expr->node);
    if (!call || lowerAscii(call->name) != lowerAscii(info.aggFunc))
        return false;
    if (lowerAscii(info.aggFunc) == "count")
        return true;
    if (call->args.empty() || !call->args.front())
        return info.aggExpr.empty();
    auto* col = std::get_if<ColRef>(&call->args.front()->node);
    return col && col->column == info.aggExpr;
}

std::optional<GenericInSubqueryHaving> inSubqueryHavingFromAnalyzedQuery(
        const AnalyzedQuery::InSubqueryAggInfo& info) {
    auto* cmp = info.havingPred
        ? std::get_if<Comparison>(&info.havingPred->node)
        : nullptr;
    if (!cmp) return std::nullopt;

    CmpOp op = cmp->op;
    const Literal* literal = nullptr;
    if (sqlExprIsInSubAggCall(cmp->left, info)) {
        literal = cmp->right ? std::get_if<Literal>(&cmp->right->node) : nullptr;
    } else if (sqlExprIsInSubAggCall(cmp->right, info)) {
        literal = cmp->left ? std::get_if<Literal>(&cmp->left->node) : nullptr;
        op = reverseCmpOpForGenericSource(cmp->op);
    }
    if (!literal) return std::nullopt;
    auto value = numericLiteralValueForGenericSource(*literal);
    if (!value) return std::nullopt;
    return GenericInSubqueryHaving{op, *value};
}

bool analyzedQueryHasAggregation(const AnalyzedQuery& aq) {
    for (const auto& target : aq.targets) {
        if (target.isAgg) return true;
    }
    return false;
}

GenericInSubqueryAggInfo inSubqueryAggFromAnalyzedQuery(
        const AnalyzedQuery::InSubqueryAggInfo& info) {
    return GenericInSubqueryAggInfo{
        info.alias,
        info.baseTable,
        info.tableIndex,
        info.groupCol,
        info.aggFunc,
        info.aggExpr,
        static_cast<bool>(info.havingPred),
        inSubqueryHavingFromAnalyzedQuery(info)
    };
}

const FuncCall* aggregateCallForExpr(const ExprPtr& expr, AggFunc& func) {
    if (!expr) return nullptr;
    auto* call = std::get_if<FuncCall>(&expr->node);
    if (!call) return nullptr;
    return aggregateFuncName(call->name, func) ? call : nullptr;
}

struct RatioAggregateParts {
    AggFunc numeratorFunc = AggFunc::SUM;
    ExprPtr numeratorArg;
    AggFunc denominatorFunc = AggFunc::SUM;
    ExprPtr denominatorArg;
};

struct ScaledAggregateParts {
    AggFunc func = AggFunc::SUM;
    ExprPtr arg;
    double factor = 1.0;
};

std::optional<double> numericLiteralValue(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<Literal>(&expr->node);
    if (!lit) return std::nullopt;
    if (auto* i = std::get_if<int>(&lit->value)) return static_cast<double>(*i);
    if (auto* f = std::get_if<float>(&lit->value)) return static_cast<double>(*f);
    return std::nullopt;
}

bool extractAggregateSide(const ExprPtr& expr,
                          const FuncCall*& call,
                          AggFunc& func,
                          double& factor) {
    factor = 1.0;
    call = aggregateCallForExpr(expr, func);
    if (call) return true;

    auto* binary = expr ? std::get_if<BinaryExpr>(&expr->node) : nullptr;
    if (!binary || binary->op != ExprOp::MUL) return false;

    if (auto leftFactor = numericLiteralValue(binary->left)) {
        call = aggregateCallForExpr(binary->right, func);
        if (call) {
            factor = *leftFactor;
            return true;
        }
    }
    if (auto rightFactor = numericLiteralValue(binary->right)) {
        call = aggregateCallForExpr(binary->left, func);
        if (call) {
            factor = *rightFactor;
            return true;
        }
    }
    return false;
}

ExprPtr applyAggregateInputFactor(const ExprPtr& arg, double factor) {
    if (!arg || factor == 1.0) return arg;
    return Expr::binary(ExprOp::MUL, Expr::litf(static_cast<float>(factor)), arg);
}

bool extractRatioAggregateParts(const ExprPtr& expr,
                                RatioAggregateParts& out) {
    if (!expr) return false;
    auto* binary = std::get_if<BinaryExpr>(&expr->node);
    if (!binary || binary->op != ExprOp::DIV) return false;

    const FuncCall* numerator = nullptr;
    const FuncCall* denominator = nullptr;
    double numeratorFactor = 1.0;
    double denominatorFactor = 1.0;
    if (!extractAggregateSide(binary->left, numerator, out.numeratorFunc,
                              numeratorFactor) ||
        !extractAggregateSide(binary->right, denominator, out.denominatorFunc,
                              denominatorFactor) ||
        numerator->args.empty() || denominator->args.empty()) {
        return false;
    }

    out.numeratorArg = applyAggregateInputFactor(numerator->args[0],
                                                 numeratorFactor);
    out.denominatorArg = applyAggregateInputFactor(denominator->args[0],
                                                   denominatorFactor);
    return true;
}

bool extractScaledAggregateParts(const ExprPtr& expr,
                                 ScaledAggregateParts& out) {
    if (!expr) return false;
    const FuncCall* call = nullptr;
    double factor = 1.0;
    if (extractAggregateSide(expr, call, out.func, factor)) {
        if (!call || call->args.empty() || factor == 1.0) return false;
        if (out.func != AggFunc::SUM && out.func != AggFunc::AVG)
            return false;
        out.arg = applyAggregateInputFactor(call->args[0], factor);
        out.factor = factor;
        return true;
    }

    auto* binary = std::get_if<BinaryExpr>(&expr->node);
    if (!binary || binary->op != ExprOp::DIV) return false;
    auto divisor = numericLiteralValue(binary->right);
    if (!divisor || *divisor == 0.0) return false;

    if (!extractAggregateSide(binary->left, call, out.func, factor) ||
        !call || call->args.empty()) {
        return false;
    }
    if (out.func != AggFunc::SUM && out.func != AggFunc::AVG)
        return false;

    out.factor = factor / *divisor;
    if (out.factor == 1.0) return false;
    out.arg = applyAggregateInputFactor(call->args[0], out.factor);
    return true;
}

class IrBuildContext {
public:
    explicit IrBuildContext(const AnalyzedQuery& aq) : aq_(aq) {}

    GenericRelationInstanceId addTableInstance(size_t tableIndex) {
        const std::string& table = aq_.tables[tableIndex];
        std::string alias = table;
        if (tableIndex < aq_.tableAliases.size() && !aq_.tableAliases[tableIndex].empty())
            alias = aq_.tableAliases[tableIndex];

        GenericRelationId rel;
        auto rit = relationByName_.find(table);
        if (rit == relationByName_.end()) {
            std::string maxKeySymbol;
            std::string pkColumn;
            std::string pkDomainSymbol;
            int probePriority = 0;
            if (aq_.schema) {
                maxKeySymbol = aq_.schema->maxKeySymbol(table);
                if (auto pk = aq_.schema->pkInfo(table)) {
                    pkColumn = pk->first;
                    pkDomainSymbol = pk->second;
                }
                probePriority = aq_.schema->tableProbePriority(table);
            }
            rel = builder.addRelation(table, false, std::move(maxKeySymbol),
                                      std::move(pkColumn),
                                      std::move(pkDomainSymbol),
                                      probePriority);
            relationByName_[table] = rel;
        } else {
            rel = rit->second;
        }

        auto inst = builder.addRelationInstance(rel, table, alias);
        instanceByAlias_[alias] = inst;
        if (!instanceByAlias_.count(table))
            instanceByAlias_[table] = inst;
        return inst;
    }

    GenericRelationInstanceId resolveInstance(const std::string& name) const {
        if (name.empty()) return GenericRelationInstanceId{};
        auto it = instanceByAlias_.find(name);
        if (it != instanceByAlias_.end()) return it->second;
        auto ait = aq_.aliasMap.find(name);
        if (ait != aq_.aliasMap.end()) {
            auto baseIt = instanceByAlias_.find(ait->second);
            if (baseIt != instanceByAlias_.end()) return baseIt->second;
        }
        return GenericRelationInstanceId{};
    }

    GenericExprPtr makeExpr(GenericExpr expr) {
        expr.id = builder.nextExprId();
        auto ptr = std::make_shared<GenericExpr>();
        *ptr = std::move(expr);
        return ptr;
    }

    GenericExprPtr convertExpr(const ExprPtr& expr) {
        if (!expr) return nullptr;
        return std::visit([&](const auto& node) -> GenericExprPtr {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                GenericRelationInstanceId inst =
                    resolveInstance(!node.tableAlias.empty() ? node.tableAlias : node.table);
                TypeInfo type = typeFromColumn(node);
                GenericExpr out;
                out.type = type;
                out.node = columnExprFromColRef(node, inst, aq_.schema);
                return makeExpr(std::move(out));
            } else if constexpr (std::is_same_v<T, Literal>) {
                GenericLiteralExpr lit;
                lit.type = typeFromLiteral(node);
                if (std::holds_alternative<int>(node.value))
                    lit.value = static_cast<int64_t>(std::get<int>(node.value));
                else if (std::holds_alternative<float>(node.value))
                    lit.value = static_cast<double>(std::get<float>(node.value));
                else
                    lit.value = std::get<std::string>(node.value);
                GenericExpr out;
                out.type = lit.type;
                out.node = std::move(lit);
                return makeExpr(std::move(out));
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                GenericBinaryExpr bin;
                bin.op = node.op;
                bin.left = convertExpr(node.left);
                bin.right = convertExpr(node.right);
                bin.type = typeFromExpr(expr);
                GenericExpr out;
                out.type = bin.type;
                out.node = std::move(bin);
                return makeExpr(std::move(out));
            } else if constexpr (std::is_same_v<T, CaseWhen>) {
                GenericCaseExpr c;
                c.type = typeFromExpr(expr);
                for (const auto& branch : node.branches)
                    c.branches.push_back({convertPredicate(branch.condition),
                                          convertExpr(branch.result)});
                c.elseResult = convertExpr(node.elseResult);
                GenericExpr out;
                out.type = c.type;
                out.node = std::move(c);
                return makeExpr(std::move(out));
            } else if constexpr (std::is_same_v<T, FuncCall>) {
                GenericFunctionExpr fn;
                fn.name = node.name;
                fn.type = typeFromExpr(expr);
                for (const auto& arg : node.args)
                    fn.args.push_back(convertExpr(arg));
                GenericExpr out;
                out.type = fn.type;
                out.node = std::move(fn);
                return makeExpr(std::move(out));
            } else if constexpr (std::is_same_v<T, ScalarSubqueryRef>) {
                GenericScalarSubqueryExpr scalar;
                scalar.index = node.index;
                scalar.type = typeFromExpr(expr);
                GenericExpr out;
                out.type = scalar.type;
                out.node = std::move(scalar);
                return makeExpr(std::move(out));
            }
            return nullptr;
        }, expr->node);
    }

    GenericPredicatePtr convertPredicate(const PredPtr& pred) {
        if (!pred) return nullptr;
        auto out = std::make_shared<GenericPredicate>();
        std::visit([&](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, Comparison>) {
                out->node = GenericComparisonPred{node.op, convertExpr(node.left),
                                                  convertExpr(node.right)};
            } else if constexpr (std::is_same_v<T, Between>) {
                out->node = GenericBetweenPred{convertExpr(node.expr),
                                               convertExpr(node.low),
                                               convertExpr(node.high)};
            } else if constexpr (std::is_same_v<T, InList>) {
                GenericInListPred in;
                in.expr = convertExpr(node.expr);
                for (const auto& value : node.values)
                    in.values.push_back(convertExpr(value));
                out->node = std::move(in);
            } else if constexpr (std::is_same_v<T, Like>) {
                out->node = GenericLikePred{convertExpr(node.expr), node.pattern,
                                            node.negated};
            } else if constexpr (std::is_same_v<T, LogicalAnd>) {
                GenericLogicalPred logical;
                logical.op = GenericLogicalPred::Op::And;
                for (const auto& child : node.children)
                    logical.children.push_back(convertPredicate(child));
                out->node = std::move(logical);
            } else if constexpr (std::is_same_v<T, LogicalOr>) {
                GenericLogicalPred logical;
                logical.op = GenericLogicalPred::Op::Or;
                for (const auto& child : node.children)
                    logical.children.push_back(convertPredicate(child));
                out->node = std::move(logical);
            } else if constexpr (std::is_same_v<T, LogicalNot>) {
                GenericLogicalPred logical;
                logical.op = GenericLogicalPred::Op::Not;
                logical.children.push_back(convertPredicate(node.child));
                out->node = std::move(logical);
            } else if constexpr (std::is_same_v<T, ExistsPred>) {
                out->node = GenericExistsPred{node.negated, node.subqueryIdx};
            }
        }, pred->node);
        return out;
    }

    GenericPredicatePtr combineAnd(const std::vector<GenericPredicatePtr>& preds) {
        if (preds.empty()) return nullptr;
        if (preds.size() == 1) return preds.front();
        auto out = std::make_shared<GenericPredicate>();
        out->node = GenericLogicalPred{GenericLogicalPred::Op::And, preds};
        return out;
    }

    GenericPredicatePtr joinPredicate(const JoinClause& join) {
        auto left = makeColumnExpr(join.leftTable, join.leftCol);
        auto right = makeColumnExpr(join.rightTable, join.rightCol);
        auto out = std::make_shared<GenericPredicate>();
        out->node = GenericComparisonPred{CmpOp::EQ, left, right};
        return out;
    }

    GenericExprPtr makeColumnExpr(const std::string& tableOrAlias,
                                  const std::string& column) {
        std::string base = tableOrAlias;
        auto ait = aq_.aliasMap.find(tableOrAlias);
        if (ait != aq_.aliasMap.end()) base = ait->second;
        TypeInfo type = intType();
        if (aq_.schema && aq_.schema->hasColumn(base, column)) {
            DataType dt = aq_.schema->columnType(base, column);
            type = TypeInfo{dt, aq_.schema->columnFixedWidth(base, column)};
        }
        GenericExpr out;
        out.type = type;
        GenericColumnExpr col;
        col.relationInstance = resolveInstance(tableOrAlias);
        col.table = base;
        col.alias = tableOrAlias;
        col.column = column;
        col.type = type;
        if (aq_.schema && !base.empty() && aq_.schema->hasColumn(base, column)) {
            if (auto domain = aq_.schema->groupDomain(base, column)) {
                col.hasGroupDomain = true;
                col.domainMin = domain->minValue;
                col.domainMax = domain->maxValue;
            }
            col.charDomain = aq_.schema->charDomain(base, column);
            col.numericScale = aq_.schema->numericScale(base, column);
            col.keyDomainSymbol = aq_.schema->keyDomainSymbol(base, column);
            col.distinctDomainSymbol = aq_.schema->distinctDomainSymbol(base, column);
        }
        out.node = std::move(col);
        return makeExpr(std::move(out));
    }

    GenericOutputSchema outputForProjections(const std::vector<GenericProjection>& projections) {
        GenericOutputSchema schema;
        for (const auto& proj : projections) {
            schema.columns.push_back(GenericColumn{builder.nextColumnId(),
                                                   GenericRelationInstanceId{},
                                                   proj.name,
                                                   proj.name,
                                                   proj.type});
        }
        return schema;
    }

    GenericOutputSchema appendSchema(const GenericOutputSchema& left,
                                     const GenericOutputSchema& right) {
        GenericOutputSchema out = left;
        out.columns.insert(out.columns.end(), right.columns.begin(), right.columns.end());
        return out;
    }

    GenericRelPlanBuilder builder;

private:
    const AnalyzedQuery& aq_;
    std::unordered_map<std::string, GenericRelationId> relationByName_;
    std::unordered_map<std::string, GenericRelationInstanceId> instanceByAlias_;
};

GenericFromSubqueryJoin fromSubqueryJoinFromAnalyzedQuery(const JoinClause& join) {
    return GenericFromSubqueryJoin{
        join.leftTable,
        join.rightTable,
        join.leftCol,
        join.rightCol,
        join.leftOuter
    };
}

std::optional<GenericFromSubqueryAggTarget> fromSubqueryAggTargetFromAnalyzedQuery(
        IrBuildContext& ctx,
        const SelectTarget& target,
        size_t targetIndex) {
    if (!target.isAgg || !target.agg) return std::nullopt;
    GenericFromSubqueryAggTarget out;
    out.name = targetName(target, targetIndex);
    out.func = target.agg->func;
    out.arg = ctx.convertExpr(target.agg->innerExpr);
    out.star = target.agg->isStar;
    out.type = typeFromAgg(target.agg->func, target.agg->innerExpr);
    return out;
}

GenericFromSubqueryAggInfo fromSubqueryAggFromAnalyzedQuery(
        IrBuildContext& ctx,
        const FromSubqueryAggInfo& info) {
    GenericFromSubqueryAggInfo out;
    out.alias = info.alias;
    out.tables = info.tables;
    out.tableAliases = info.tableAliases;
    for (const auto& join : info.joins)
        out.joins.push_back(fromSubqueryJoinFromAnalyzedQuery(join));
    for (const auto& filter : info.filters)
        out.filters.push_back(ctx.convertPredicate(filter));
    for (size_t i = 0; i < info.targets.size(); ++i) {
        if (auto target = fromSubqueryAggTargetFromAnalyzedQuery(ctx, info.targets[i], i))
            out.aggregates.push_back(std::move(*target));
    }
    for (const auto& expr : info.groupBy)
        out.groupBy.push_back(ctx.convertExpr(expr));
    return out;
}

GenericSourceQueryInfo sourceQueryFromAnalyzedQuery(IrBuildContext& ctx,
                                               const AnalyzedQuery& aq) {
    GenericSourceQueryInfo source;
    for (const auto& info : aq.inSubAggs)
        source.inSubAggs.push_back(inSubqueryAggFromAnalyzedQuery(info));
    for (const auto& info : aq.fromSubqueryAggs)
        source.fromSubqueryAggs.push_back(
            fromSubqueryAggFromAnalyzedQuery(ctx, info));
    for (const auto& scalarSql : aq.scalarSubquerySql) {
        GenericSourceSubquery out;
        out.type = GenericSourceSubquery::SCALAR_SUBQUERY;
        out.scalarHavingSummary =
            scalarHavingSummaryFromSqlJson(scalarSql, aq.schema);
        out.fromSubqueryScalarExtrema =
            fromSubqueryScalarExtremaFromSqlJson(scalarSql);
        out.decorrelatedScalar = parseDecorrelatedScalarSubquery(
            scalarSql, aq.schema, static_cast<int>(source.subqueries.size()));
        source.subqueries.push_back(std::move(out));
    }
    return source;
}

} // namespace

static std::optional<GenericRelPlan> buildGenericRelPlanFromAnalyzedQuery(
        const AnalyzedQuery& aq,
        std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<GenericRelPlan> {
        if (error) *error = msg;
        return std::nullopt;
    };

    if (aq.tables.empty())
        return fail("Generic IR builder: query has no FROM relation.");

    IrBuildContext ctx(aq);
    std::vector<GenericNodeId> scanNodes;
    for (size_t i = 0; i < aq.tables.size(); ++i) {
        auto inst = ctx.addTableInstance(i);
        std::string alias = aq.tables[i];
        if (i < aq.tableAliases.size() && !aq.tableAliases[i].empty())
            alias = aq.tableAliases[i];
        GenericScanDetail detail{inst, aq.tables[i], alias};
        scanNodes.push_back(ctx.builder.addNode(GenericRelOp::Scan, {},
                                                GenericOutputSchema{},
                                                std::move(detail)));
    }

    std::map<std::string, int> baseTableCounts;
    for (const auto& table : aq.tables)
        baseTableCounts[table]++;

    auto namesForTableIndex = [&](size_t idx) {
        std::set<std::string> names;
        const std::string& base = aq.tables[idx];
        if (idx < aq.tableAliases.size() && !aq.tableAliases[idx].empty())
            names.insert(aq.tableAliases[idx]);
        if (baseTableCounts[base] <= 1 ||
            (idx < aq.tableAliases.size() && aq.tableAliases[idx] == base)) {
            names.insert(base);
        }
        return names;
    };
    auto containsName = [](const std::set<std::string>& names,
                           const std::string& value) {
        return names.find(value) != names.end();
    };

    GenericNodeId root = scanNodes.front();
    GenericOutputSchema rootSchema;
    std::set<std::string> joinedNames = namesForTableIndex(0);
    for (size_t i = 1; i < scanNodes.size(); ++i) {
        std::set<std::string> newNames = namesForTableIndex(i);
        std::vector<GenericPredicatePtr> joinPreds;
        GenericJoinDetail detail;
        detail.kind = GenericJoinKind::Inner;
        for (const auto& join : aq.joins) {
            bool leftNew = containsName(newNames, join.leftTable);
            bool rightNew = containsName(newNames, join.rightTable);
            bool leftJoined = containsName(joinedNames, join.leftTable);
            bool rightJoined = containsName(joinedNames, join.rightTable);
            if (!((leftNew && rightJoined) || (rightNew && leftJoined)))
                continue;
            joinPreds.push_back(ctx.joinPredicate(join));
            if (join.leftOuter) detail.kind = GenericJoinKind::LeftOuter;
            if (join.semi) detail.kind = GenericJoinKind::Semi;
            if (join.anti) detail.kind = GenericJoinKind::Anti;
        }

        detail.predicate = ctx.combineAnd(joinPreds);

        GenericRelOp op = GenericRelOp::Join;
        if (detail.kind == GenericJoinKind::Semi) op = GenericRelOp::SemiJoin;
        else if (detail.kind == GenericJoinKind::Anti) op = GenericRelOp::AntiJoin;

        rootSchema = ctx.appendSchema(rootSchema, GenericOutputSchema{});
        root = ctx.builder.addNode(op, {root, scanNodes[i]},
                                   rootSchema, std::move(detail));
        joinedNames.insert(newNames.begin(), newNames.end());
    }

    if (!aq.filters.empty() || !aq.instanceFilters.empty()) {
        std::vector<GenericPredicatePtr> filters;
        for (const auto& filter : aq.filters)
            filters.push_back(ctx.convertPredicate(filter));
        for (const auto& [_, instanceFilters] : aq.instanceFilters) {
            for (const auto& filter : instanceFilters)
                filters.push_back(ctx.convertPredicate(filter));
        }
        GenericFilterDetail detail{ctx.combineAnd(filters)};
        root = ctx.builder.addNode(GenericRelOp::Filter, {root},
                                   rootSchema, std::move(detail));
    }

    if (analyzedQueryHasAggregation(aq) || !aq.groupBy.empty()) {
        GenericAggregateDetail detail;
        std::vector<GenericProjection> outputs;
        std::map<std::string, GenericProjection> outputByName;

        for (size_t i = 0; i < aq.groupBy.size(); ++i) {
            auto expr = ctx.convertExpr(aq.groupBy[i]);
            std::string name = groupOutputNameForExpr(aq, aq.groupBy[i], i);
            GenericProjection groupProjection{name,
                                              expr, expr ? expr->type : intType()};
            detail.groupNames.push_back(name);
            outputByName[groupProjection.name] = groupProjection;
            detail.groupBy.push_back(std::move(expr));
        }

        for (size_t i = 0; i < aq.targets.size(); ++i) {
            const auto& target = aq.targets[i];
            if (!target.isAgg || !target.agg) continue;
            auto appendAggregate = [&](const std::string& name,
                                       AggFunc func,
                                       const ExprPtr& inner,
                                       bool star,
                                       const std::string& outputFunc,
                                       TypeInfo outputType) {
                GenericAggregateExpr agg;
                agg.func = func;
                agg.arg = ctx.convertExpr(inner);
                agg.star = star;
                agg.distinct = func == AggFunc::COUNT_DISTINCT;
                agg.alias = name;
                agg.type = typeFromAgg(func, inner);

                GenericExpr out;
                out.type = agg.type;
                out.node = std::move(agg);
                auto aggExpr = ctx.makeExpr(std::move(out));
                detail.aggregates.push_back(GenericProjection{name, aggExpr, outputType});
                detail.aggregateOutputFuncs.push_back(outputFunc);
                outputByName[detail.aggregates.back().name] = detail.aggregates.back();
            };

            const std::string name = targetName(target, i);
            RatioAggregateParts ratio;
            if (extractRatioAggregateParts(target.expr, ratio)) {
                appendAggregate(name, ratio.numeratorFunc, ratio.numeratorArg, false,
                                "RATIO", floatType());
                appendAggregate("__hidden_" + name + "_den", ratio.denominatorFunc,
                                ratio.denominatorArg, false, "RATIO_DEN",
                                typeFromAgg(ratio.denominatorFunc,
                                            ratio.denominatorArg));
                continue;
            }

            ScaledAggregateParts scaled;
            if (extractScaledAggregateParts(target.expr, scaled)) {
                appendAggregate(name, scaled.func, scaled.arg, false, "",
                                typeFromAgg(scaled.func, scaled.arg));
                continue;
            }

            appendAggregate(name, target.agg->func, target.agg->innerExpr,
                            target.agg->isStar, "",
                            typeFromAgg(target.agg->func, target.agg->innerExpr));
        }

        for (size_t i = 0; i < aq.targets.size(); ++i) {
            const auto& target = aq.targets[i];
            std::string name;
            if (target.isAgg && target.agg) {
                name = targetName(target, i);
            } else {
                name = targetName(target, i);
                if (!outputByName.count(name))
                    name = groupName(target.expr, i);
            }
            auto it = outputByName.find(name);
            if (it != outputByName.end()) {
                outputs.push_back(it->second);
                detail.outputOrder.push_back(name);
            }
        }

        if (outputs.empty()) {
            for (const auto& [name, projection] : outputByName) {
                outputs.push_back(projection);
                detail.outputOrder.push_back(name);
            }
        }

        detail.having = ctx.convertPredicate(aq.having);
        GenericOutputSchema output = ctx.outputForProjections(outputs);
        rootSchema = output;
        root = ctx.builder.addNode(GenericRelOp::Aggregate, {root},
                                   std::move(output), std::move(detail));
    } else if (!aq.targets.empty()) {
        GenericProjectDetail detail;
        for (size_t i = 0; i < aq.targets.size(); ++i) {
            auto expr = ctx.convertExpr(aq.targets[i].expr);
            detail.projections.push_back(GenericProjection{targetName(aq.targets[i], i),
                                                           expr,
                                                           expr ? expr->type : intType()});
        }
        GenericOutputSchema output = ctx.outputForProjections(detail.projections);
        rootSchema = output;
        root = ctx.builder.addNode(GenericRelOp::Project, {root},
                                   std::move(output), std::move(detail));
    }

    if (!aq.orderBy.empty()) {
        GenericSortDetail detail;
        for (const auto& item : aq.orderBy)
            detail.keys.push_back(GenericSortKey{ctx.convertExpr(item.expr),
                                                 item.descending});
        root = ctx.builder.addNode(GenericRelOp::Sort, {root},
                                   rootSchema, std::move(detail));
    }

    if (aq.limit >= 0) {
        GenericLimitDetail detail{aq.limit};
        root = ctx.builder.addNode(GenericRelOp::Limit, {root},
                                   rootSchema, std::move(detail));
    }

    ctx.builder.setSchema(aq.schema);
    ctx.builder.setSourceQuery(sourceQueryFromAnalyzedQuery(ctx, aq));
    return ctx.builder.finish(root);
}

std::optional<GenericRelPlan> buildGenericRelationalIRFromSQL(
        const std::string& sql,
        const SchemaProvider& schema,
        std::string* error) {
    try {
        auto analyzed = collectAnalyzedQuery(sql, schema);
        return buildGenericRelPlanFromAnalyzedQuery(analyzed, error);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return std::nullopt;
    }
}

} // namespace codegen
