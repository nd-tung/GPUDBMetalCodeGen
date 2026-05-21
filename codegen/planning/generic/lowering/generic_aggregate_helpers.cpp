#include "generic/lowering/generic_aggregate_helpers.h"

#include "core/schema_provider.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_scalar_placeholder.h"
#include "query_analyzer.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>

namespace codegen {

namespace {

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::string groupDisplayNameForExpr(const GenericExprPtr& expr, size_t index) {
    if (expr) {
        if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
            return col->column.empty() ? "group_" + std::to_string(index) : col->column;
        if (auto* fn = std::get_if<GenericFunctionExpr>(&expr->node))
            return fn->name.empty() ? "group_" + std::to_string(index) : fn->name;
    }
    return "group_" + std::to_string(index);
}


std::optional<AggFunc> aggregateFuncFromName(const std::string& name) {
    const std::string lower = lowerAscii(name);
    if (lower == "sum") return AggFunc::SUM;
    if (lower == "count") return AggFunc::COUNT;
    if (lower == "avg") return AggFunc::AVG;
    if (lower == "min") return AggFunc::MIN;
    if (lower == "max") return AggFunc::MAX;
    return std::nullopt;
}

bool havingExprMatchesAggregate(const GenericExprPtr& expr,
                                const GenericProjection& projection) {
    if (!expr || !projection.expr) return false;
    auto* projected = std::get_if<GenericAggregateExpr>(&projection.expr->node);
    if (!projected) return false;

    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        return !projection.name.empty() && col->column == projection.name;
    }

    if (auto* agg = std::get_if<GenericAggregateExpr>(&expr->node)) {
        return agg->func == projected->func &&
               agg->star == projected->star &&
               agg->distinct == projected->distinct &&
               genericExprEquivalent(agg->arg, projected->arg);
    }

    auto* fn = std::get_if<GenericFunctionExpr>(&expr->node);
    if (!fn) return false;
    auto func = aggregateFuncFromName(fn->name);
    if (!func || *func != projected->func) return false;

    const bool star = *func == AggFunc::COUNT && fn->args.empty();
    if (star || projected->star)
        return star == projected->star;
    if (fn->args.empty()) return !projected->arg;
    return genericExprEquivalent(fn->args.front(), projected->arg);
}

std::optional<int> aggregateIndexForHavingExpr(
        const GenericExprPtr& expr,
        const GenericAggregateDetail& aggregate) {
    for (size_t i = 0; i < aggregate.aggregates.size(); ++i) {
        if (havingExprMatchesAggregate(expr, aggregate.aggregates[i]))
            return static_cast<int>(i);
    }
    return std::nullopt;
}

std::optional<double> numericLiteralValue(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    return std::visit([](const auto& value) -> std::optional<double> {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, int64_t>)
            return static_cast<double>(value);
        else if constexpr (std::is_same_v<V, double>)
            return value;
        else
            return std::nullopt;
    }, lit->value);
}

bool scaleNeutralFactor(const GenericExprPtr& expr) {
    if (!expr) return false;
    if (numericLiteralValue(expr)) return true;
    return expr->type.type == DataType::INT;
}

int scalarSumFixedPointScaleForExpr(const GenericExprPtr& expr) {
    const int directScale = numericScaleForExpr(expr);
    if (directScale > 0) return directScale;
    if (!expr) return 0;
    auto* bin = std::get_if<GenericBinaryExpr>(&expr->node);
    if (!bin) return 0;

    const int leftScale = scalarSumFixedPointScaleForExpr(bin->left);
    const int rightScale = scalarSumFixedPointScaleForExpr(bin->right);
    switch (bin->op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
            if (leftScale > 0 && leftScale == rightScale) return leftScale;
            if (leftScale > 0 && numericLiteralValue(bin->right)) return leftScale;
            if (rightScale > 0 && numericLiteralValue(bin->left)) return rightScale;
            return 0;
        case ExprOp::MUL:
            if (leftScale > 0 && rightScale > 0)
                return std::max(leftScale, rightScale);
            if (leftScale > 0 && scaleNeutralFactor(bin->right)) return leftScale;
            if (rightScale > 0 && scaleNeutralFactor(bin->left)) return rightScale;
            return 0;
        case ExprOp::DIV:
            return 0;
    }
    return 0;
}

CmpOp reverseCmpOp(CmpOp op) {
    switch (op) {
        case CmpOp::LT: return CmpOp::GT;
        case CmpOp::LE: return CmpOp::GE;
        case CmpOp::GT: return CmpOp::LT;
        case CmpOp::GE: return CmpOp::LE;
        case CmpOp::EQ: return CmpOp::EQ;
        case CmpOp::NE: return CmpOp::NE;
    }
    return op;
}

std::string formatSignatureDouble(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::string aggFuncSignatureName(AggFunc func) {
    return lowerAscii(aggFuncName(func));
}

std::string exprOpSignatureToken(ExprOp op) {
    switch (op) {
        case ExprOp::ADD: return "+";
        case ExprOp::SUB: return "-";
        case ExprOp::MUL: return "*";
        case ExprOp::DIV: return "/";
    }
    return "?";
}

std::optional<std::string> jsonStringValueForIr(const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") && node["String"].contains("sval"))
        return node["String"]["sval"].get<std::string>();
    return std::nullopt;
}

std::string jsonAExprOpForIr(const nlohmann::json& ae) {
    if (!ae.contains("name") || !ae["name"].is_array() || ae["name"].empty())
        return {};
    if (auto s = jsonStringValueForIr(ae["name"][0])) return *s;
    return {};
}

std::string jsonFuncNameForIr(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() || fc["funcname"].empty())
        return {};
    auto s = jsonStringValueForIr(fc["funcname"].back());
    return s ? lowerAscii(*s) : "";
}

std::optional<double> jsonNumericConstForIr(const nlohmann::json& node) {
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

std::string resolveJsonColumnTable(const std::string& qualifier,
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

std::optional<std::string> jsonColumnSignatureForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    const nlohmann::json* cr = nullptr;
    if (node.is_object() && node.contains("ColumnRef")) cr = &node["ColumnRef"];
    else if (node.is_object() && node.contains("fields")) cr = &node;
    if (!cr || !cr->contains("fields") || !(*cr)["fields"].is_array())
        return std::nullopt;

    std::vector<std::string> fields;
    for (const auto& field : (*cr)["fields"]) {
        if (auto s = jsonStringValueForIr(field)) fields.push_back(*s);
    }
    if (fields.empty()) return std::nullopt;
    const std::string column = fields.back();
    const std::string qualifier = fields.size() >= 2 ? fields[fields.size() - 2] : "";
    const std::string table = resolveJsonColumnTable(qualifier, column, aliases, tables, schema);
    return table.empty() ? "col:" + column : "col:" + table + "." + column;
}

std::string genericColumnSignatureForIr(const GenericColumnExpr& col) {
    return col.table.empty() ? "col:" + col.column : "col:" + col.table + "." + col.column;
}

std::optional<std::string> genericExprSignatureForIr(const GenericExprPtr& expr);

std::string combineBinarySignature(const std::string& op,
                                   std::string left,
                                   std::string right) {
    if (op == "+" || op == "*") {
        if (right < left) std::swap(left, right);
    }
    return "bin:" + op + "(" + left + "," + right + ")";
}

std::optional<std::string> genericExprSignatureForIr(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<std::string> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return genericColumnSignatureForIr(node);
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return std::visit([](const auto& value) -> std::string {
                using V = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<V, int64_t>)
                    return "lit:i:" + std::to_string(value);
                else if constexpr (std::is_same_v<V, double>)
                    return "lit:f:" + formatSignatureDouble(value);
                else
                    return "lit:s:" + value;
            }, node.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            auto left = genericExprSignatureForIr(node.left);
            auto right = genericExprSignatureForIr(node.right);
            if (!left || !right) return std::nullopt;
            return combineBinarySignature(exprOpSignatureToken(node.op), *left, *right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            std::vector<std::string> args;
            for (const auto& arg : node.args) {
                auto sig = genericExprSignatureForIr(arg);
                if (!sig) return std::nullopt;
                args.push_back(*sig);
            }
            std::ostringstream oss;
            oss << "fn:" << lowerAscii(node.name) << "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i) oss << ",";
                oss << args[i];
            }
            oss << ")";
            return oss.str();
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            if (node.star) return "agg:" + aggFuncSignatureName(node.func) + "(*)";
            auto arg = genericExprSignatureForIr(node.arg);
            if (!arg) return std::nullopt;
            return "agg:" + aggFuncSignatureName(node.func) + "(" + *arg + ")";
        } else if constexpr (std::is_same_v<T, GenericScalarSubqueryExpr>) {
            return "scalar_subquery:" + std::to_string(node.index);
        } else {
            return std::nullopt;
        }
    }, expr->node);
}

std::optional<std::string> jsonExprSignatureForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    if (!node.is_object()) return std::nullopt;
    if (node.contains("TypeCast"))
        return jsonExprSignatureForIr(node["TypeCast"].value("arg", nlohmann::json{}),
                                      aliases, tables, schema);
    if (node.contains("ColumnRef") || node.contains("fields"))
        return jsonColumnSignatureForIr(node, aliases, tables, schema);
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
                if (auto i = readIntConst(ac["ival"])) return "lit:i:" + std::to_string(*i);
            }
            if (ac.contains("fval")) {
                if (auto f = readFloatConst(ac["fval"]))
                    return "lit:f:" + formatSignatureDouble(*f);
            }
            if (ac.contains("val") && ac["val"].contains("String"))
                return "lit:s:" + ac["val"]["String"].at("sval").get<std::string>();
            if (ac.contains("val") && ac["val"].contains("Integer"))
                return "lit:i:" + std::to_string(ac["val"]["Integer"].at("ival").get<int64_t>());
            if (ac.contains("val") && ac["val"].contains("Float"))
                return "lit:f:" + formatSignatureDouble(
                    std::stod(ac["val"]["Float"].at("fval").get<std::string>()));
        } catch (...) {
            return std::nullopt;
        }
        return std::nullopt;
    }
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        const std::string name = jsonFuncNameForIr(fc);
        std::vector<std::string> args;
        if (fc.contains("args") && fc["args"].is_array()) {
            for (const auto& arg : fc["args"]) {
                if (arg.is_object() && arg.contains("A_Star")) {
                    args.push_back("*");
                    continue;
                }
                auto sig = jsonExprSignatureForIr(arg, aliases, tables, schema);
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
        const std::string op = jsonAExprOpForIr(ae);
        auto left = jsonExprSignatureForIr(ae.value("lexpr", nlohmann::json{}),
                                           aliases, tables, schema);
        auto right = jsonExprSignatureForIr(ae.value("rexpr", nlohmann::json{}),
                                            aliases, tables, schema);
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

std::optional<std::string> genericPredicateSignatureForIr(const GenericPredicatePtr& pred) {
    if (!pred) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<std::string> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            auto left = genericExprSignatureForIr(node.left);
            auto right = genericExprSignatureForIr(node.right);
            if (!left || !right) return std::nullopt;
            return predicateSignatureFromComparison(cmpOpToMetal(node.op), *left, *right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            auto low = genericExprSignatureForIr(node.low);
            auto high = genericExprSignatureForIr(node.high);
            if (!expr || !low || !high) return std::nullopt;
            return "between:" + *expr + "(" + *low + "," + *high + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            if (!expr) return std::nullopt;
            std::vector<std::string> values;
            for (const auto& value : node.values) {
                auto sig = genericExprSignatureForIr(value);
                if (!sig) return std::nullopt;
                values.push_back(*sig);
            }
            std::sort(values.begin(), values.end());
            std::ostringstream oss;
            oss << "in:" << *expr << "(";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) oss << ",";
                oss << values[i];
            }
            oss << ")";
            return oss.str();
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            if (!expr) return std::nullopt;
            return std::string(node.negated ? "notlike:" : "like:") + *expr + ":" + node.pattern;
        } else {
            return std::nullopt;
        }
    }, pred->node);
}

bool collectGenericPredicateAtomSignatures(const GenericPredicatePtr& pred,
                                           std::vector<std::string>& out) {
    if (!pred) return true;
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op == GenericLogicalPred::Op::And) {
            for (const auto& child : logical->children) {
                if (!collectGenericPredicateAtomSignatures(child, out))
                    return false;
            }
            return true;
        }
    }
    auto sig = genericPredicateSignatureForIr(pred);
    if (!sig) return false;
    out.push_back(*sig);
    return true;
}

bool collectJsonPredicateAtomSignatures(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        std::vector<std::string>& out) {
    if (!node.is_object()) return false;
    if (node.contains("BoolExpr") && node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        const auto& args = node["BoolExpr"].value("args", nlohmann::json::array());
        for (const auto& arg : args) {
            if (!collectJsonPredicateAtomSignatures(arg, aliases, tables, schema, out))
                return false;
        }
        return true;
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        const std::string op = jsonAExprOpForIr(ae);
        auto left = jsonExprSignatureForIr(ae.value("lexpr", nlohmann::json{}),
                                           aliases, tables, schema);
        auto right = jsonExprSignatureForIr(ae.value("rexpr", nlohmann::json{}),
                                            aliases, tables, schema);
        if (!left || !right || op.empty()) return false;
        out.push_back(predicateSignatureFromComparison(op, *left, *right));
        return true;
    }
    return false;
}

struct JsonScalarAggTarget {
    AggFunc func = AggFunc::SUM;
    bool star = false;
    double multiplier = 1.0;
    std::string argSignature;
};

bool extractJsonScalarAggTargetForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        JsonScalarAggTarget& out,
        double multiplier = 1.0) {
    if (!node.is_object()) return false;
    if (node.contains("TypeCast"))
        return extractJsonScalarAggTargetForIr(
            node["TypeCast"].value("arg", nlohmann::json{}), aliases, tables,
            schema, out, multiplier);
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        auto func = aggregateFuncFromName(jsonFuncNameForIr(fc));
        if (!func) return false;
        out.func = *func;
        out.multiplier = multiplier;
        out.star = !fc.contains("args") || !fc["args"].is_array() || fc["args"].empty();
        if (!out.star) {
            const auto& arg = fc["args"][0];
            out.star = arg.is_object() && arg.contains("A_Star");
        }
        if (!out.star) {
            auto sig = jsonExprSignatureForIr(fc["args"][0], aliases, tables, schema);
            if (!sig) return false;
            out.argSignature = *sig;
        }
        return true;
    }
    if (!node.contains("A_Expr")) return false;

    const auto& ae = node["A_Expr"];
    const std::string op = jsonAExprOpForIr(ae);
    if (op == "*") {
        if (auto lit = jsonNumericConstForIr(ae.value("lexpr", nlohmann::json{})))
            return extractJsonScalarAggTargetForIr(ae.value("rexpr", nlohmann::json{}),
                                                   aliases, tables, schema, out,
                                                   multiplier * *lit);
        if (auto lit = jsonNumericConstForIr(ae.value("rexpr", nlohmann::json{})))
            return extractJsonScalarAggTargetForIr(ae.value("lexpr", nlohmann::json{}),
                                                   aliases, tables, schema, out,
                                                   multiplier * *lit);
    }
    if (op == "/") {
        if (auto lit = jsonNumericConstForIr(ae.value("rexpr", nlohmann::json{}))) {
            if (*lit != 0.0)
                return extractJsonScalarAggTargetForIr(ae.value("lexpr", nlohmann::json{}),
                                                       aliases, tables, schema, out,
                                                       multiplier / *lit);
        }
    }
    return false;
}

struct ScalarHavingSubquerySummary {
    JsonScalarAggTarget aggregate;
    std::vector<std::string> tables;
    std::vector<std::string> predicateSignatures;
};

std::optional<ScalarHavingSubquerySummary> parseScalarHavingSubquerySummary(
        const AnalyzedQuery& aq,
        int sqIdx) {
    if (sqIdx < 0 || sqIdx >= static_cast<int>(aq.subqueries.size()))
        return std::nullopt;
    const auto& sq = aq.subqueries[(size_t)sqIdx];
    if (sq.type != AnalyzedQuery::Subquery::SCALAR_SUBQUERY)
        return std::nullopt;

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(sq.sql);
    } catch (...) {
        return std::nullopt;
    }
    if (!root.contains("SelectStmt")) return std::nullopt;
    const auto& ss = root["SelectStmt"];
    if (ss.contains("groupClause") && !ss["groupClause"].is_null()) return std::nullopt;
    if (ss.contains("havingClause") && !ss["havingClause"].is_null()) return std::nullopt;
    if (ss.contains("limitCount") && !ss["limitCount"].is_null()) return std::nullopt;

    ScalarHavingSubquerySummary summary;
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
        if (!target.contains("ResTarget") || !target["ResTarget"].contains("val"))
            continue;
        if (extractJsonScalarAggTargetForIr(target["ResTarget"]["val"], aliases,
                                            summary.tables, aq.schema,
                                            summary.aggregate)) {
            foundAgg = true;
            break;
        }
    }
    if (!foundAgg) return std::nullopt;

    if (ss.contains("whereClause") && !ss["whereClause"].is_null()) {
        if (!collectJsonPredicateAtomSignatures(ss["whereClause"], aliases,
                                                summary.tables, aq.schema,
                                                summary.predicateSignatures)) {
            return std::nullopt;
        }
        std::sort(summary.predicateSignatures.begin(),
                  summary.predicateSignatures.end());
    }
    std::sort(summary.tables.begin(), summary.tables.end());
    return summary;
}

std::vector<std::string> scanTableSignaturesForHaving(
        const std::vector<const GenericRelNode*>& scans) {
    std::vector<std::string> tables;
    for (const auto* scanNode : scans) {
        if (auto* scan = scanDetail(scanNode))
            tables.push_back(scan->table);
    }
    std::sort(tables.begin(), tables.end());
    return tables;
}

std::optional<std::vector<std::string>> outerPredicateSignaturesForHaving(
        const MultiTableGroupedAggShape& shape) {
    std::vector<std::string> signatures;
    for (const auto* joinNode : shape.joins) {
        auto* join = joinNode ? std::get_if<GenericJoinDetail>(&joinNode->detail) : nullptr;
        if (!join) return std::nullopt;
        if (!collectGenericPredicateAtomSignatures(join->predicate, signatures))
            return std::nullopt;
    }
    if (auto* filter = filterDetail(shape.filter)) {
        if (!collectGenericPredicateAtomSignatures(filter->predicate, signatures))
            return std::nullopt;
    }
    std::sort(signatures.begin(), signatures.end());
    return signatures;
}

bool validateHavingAggregateIndex(const GenericAggregateDetail& aggregate,
                                  int aggIdx,
                                  const GenericAggregateExpr*& agg,
                                  std::string* error) {
    if (aggIdx < 0 || aggIdx >= static_cast<int>(aggregate.aggregates.size())) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING aggregate index is out of range.";
        return false;
    }

    agg = aggregate.aggregates[(size_t)aggIdx].expr
        ? std::get_if<GenericAggregateExpr>(&aggregate.aggregates[(size_t)aggIdx].expr->node)
        : nullptr;
    if (!agg || agg->func == AggFunc::COUNT_DISTINCT) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING over COUNT(DISTINCT) is not supported yet.";
        return false;
    }
    return true;
}

bool configureAggregateScalarHaving(const GenericAggregateDetail& aggregate,
                                    int aggIdx,
                                    CmpOp op,
                                    int sqIdx,
                                    const AnalyzedQuery* aq,
                                    const MultiTableGroupedAggShape* shape,
                                    GenericGroupSpec& groupSpec,
                                    std::string* error) {
    if (!aq || !shape) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING requires analyzed subquery metadata.";
        return false;
    }

    const GenericAggregateExpr* agg = nullptr;
    if (!validateHavingAggregateIndex(aggregate, aggIdx, agg, error))
        return false;
    auto summary = parseScalarHavingSubquerySummary(*aq, sqIdx);
    if (!summary) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING is not a supported ungrouped aggregate subquery.";
        return false;
    }
    if (summary->aggregate.multiplier <= 0.0) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING multiplier must be positive.";
        return false;
    }
    if (summary->aggregate.func != agg->func) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING aggregate function differs from grouped aggregate.";
        return false;
    }
    if (summary->aggregate.star != agg->star) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING COUNT(*) shape differs from grouped aggregate.";
        return false;
    }
    if (!agg->star) {
        auto argSig = genericExprSignatureForIr(agg->arg);
        if (!argSig || *argSig != summary->aggregate.argSignature) {
            if (error)
                *error = "IR grouped aggregate lowerer: scalar-subquery HAVING aggregate input differs from grouped aggregate.";
            return false;
        }
    }

    if (scanTableSignaturesForHaving(shape->scans) != summary->tables) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING table set differs from outer grouped input.";
        return false;
    }

    auto outerPredicates = outerPredicateSignaturesForHaving(*shape);
    if (!outerPredicates || *outerPredicates != summary->predicateSignatures) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING predicates differ from outer grouped input.";
        return false;
    }

    groupSpec.havingAggIdx = aggIdx;
    groupSpec.havingMultiplier = summary->aggregate.multiplier;
    groupSpec.havingScalarCompareOp = cmpOpToMetal(op);
    return true;
}

} // namespace

bool configureAggregateHaving(const GenericAggregateDetail& aggregate,
                              GenericGroupSpec& groupSpec,
                              const AnalyzedQuery* aq,
                              const MultiTableGroupedAggShape* shape,
                              std::string* error) {
    if (!aggregate.having) return true;

    auto* cmp = std::get_if<GenericComparisonPred>(&aggregate.having->node);
    if (!cmp) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING currently supports aggregate comparisons only.";
        return false;
    }

    auto aggIdx = aggregateIndexForHavingExpr(cmp->left, aggregate);
    CmpOp op = cmp->op;
    auto sqIdx = scalarSubqueryIndexFromExpr(cmp->right);

    if (!aggIdx || !sqIdx) {
        aggIdx = aggregateIndexForHavingExpr(cmp->right, aggregate);
        op = reverseCmpOp(cmp->op);
        sqIdx = scalarSubqueryIndexFromExpr(cmp->left);
    }

    if (aggIdx && sqIdx) {
        return configureAggregateScalarHaving(aggregate, *aggIdx, op, *sqIdx,
                                              aq, shape, groupSpec, error);
    }

    aggIdx = aggregateIndexForHavingExpr(cmp->left, aggregate);
    auto literal = numericLiteralValue(cmp->right);
    op = cmp->op;

    if (!aggIdx || !literal) {
        aggIdx = aggregateIndexForHavingExpr(cmp->right, aggregate);
        literal = numericLiteralValue(cmp->left);
        op = reverseCmpOp(cmp->op);
    }

    if ((literal && isScalarSubqueryPlaceholderExpr(cmp->left)) ||
        (literal && isScalarSubqueryPlaceholderExpr(cmp->right))) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING requires decorrelation.";
        return false;
    }

    if (!aggIdx || !literal) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING must compare a projected aggregate with a numeric literal.";
        return false;
    }
    const GenericAggregateExpr* agg = nullptr;
    if (!validateHavingAggregateIndex(aggregate, *aggIdx, agg, error))
        return false;

    groupSpec.havingCompareAggIdx = *aggIdx;
    groupSpec.havingCompareOp = cmpOpToMetal(op);
    groupSpec.havingCompareValue = *literal;
    return true;
}


bool genericExprEquivalent(const GenericExprPtr& left, const GenericExprPtr& right) {
    if (!left || !right) return !left && !right;
    return std::visit([&](const auto& lnode) -> bool {
        using L = std::decay_t<decltype(lnode)>;
        auto* rnode = std::get_if<L>(&right->node);
        if (!rnode) return false;

        if constexpr (std::is_same_v<L, GenericColumnExpr>) {
            if (lnode.column != rnode->column) return false;
            if (lnode.relationInstance.valid() && rnode->relationInstance.valid())
                return lnode.relationInstance.value == rnode->relationInstance.value;
            if (!lnode.table.empty() && !rnode->table.empty() &&
                lnode.table != rnode->table) {
                return false;
            }
            if (!lnode.alias.empty() && !rnode->alias.empty() &&
                lnode.alias != rnode->alias) {
                return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericLiteralExpr>) {
            return lnode.value == rnode->value;
        } else if constexpr (std::is_same_v<L, GenericBinaryExpr>) {
            return lnode.op == rnode->op &&
                   genericExprEquivalent(lnode.left, rnode->left) &&
                   genericExprEquivalent(lnode.right, rnode->right);
        } else if constexpr (std::is_same_v<L, GenericFunctionExpr>) {
            if (lowerAscii(lnode.name) != lowerAscii(rnode->name) ||
                lnode.args.size() != rnode->args.size()) {
                return false;
            }
            for (size_t i = 0; i < lnode.args.size(); ++i) {
                if (!genericExprEquivalent(lnode.args[i], rnode->args[i]))
                    return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericAggregateExpr>) {
            return lnode.func == rnode->func &&
                   lnode.star == rnode->star &&
                   lnode.distinct == rnode->distinct &&
                   genericExprEquivalent(lnode.arg, rnode->arg);
        } else if constexpr (std::is_same_v<L, GenericScalarSubqueryExpr>) {
            return lnode.index == rnode->index;
        } else if constexpr (std::is_same_v<L, GenericScalarLookupExpr>) {
            if (lnode.source.value != rnode->source.value ||
                lnode.outputName != rnode->outputName ||
                lnode.keys.size() != rnode->keys.size()) {
                return false;
            }
            for (size_t i = 0; i < lnode.keys.size(); ++i) {
                if (!genericExprEquivalent(lnode.keys[i], rnode->keys[i]))
                    return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericCaseExpr>) {
            return false;
        }
        return false;
    }, left->node);
}

std::optional<std::string> sortKeyDisplayNameForGroupedAgg(
        const GenericSortKey& key,
        const GenericAggregateDetail& aggregate,
        const std::vector<IrGroupKeyDesc>& groupKeys) {
    if (!key.expr) return std::nullopt;
    for (size_t i = 0; i < aggregate.groupBy.size() && i < groupKeys.size(); ++i) {
        if (genericExprEquivalent(key.expr, aggregate.groupBy[i]))
            return groupKeys[i].displayName;
    }
    if (auto* col = std::get_if<GenericColumnExpr>(&key.expr->node)) {
        for (size_t i = 0; i < aggregate.groupNames.size() && i < groupKeys.size(); ++i) {
            if (col->column == aggregate.groupNames[i])
                return groupKeys[i].displayName;
        }
        for (size_t i = 0; i < aggregate.groupBy.size() && i < groupKeys.size(); ++i) {
            auto* groupCol = std::get_if<GenericColumnExpr>(&aggregate.groupBy[i]->node);
            if (!groupCol) continue;
            if (groupCol->column == col->column &&
                (col->table.empty() || groupCol->table == col->table) &&
                (col->alias.empty() || groupCol->alias == col->alias)) {
                return groupKeys[i].displayName;
            }
        }
        for (const auto& agg : aggregate.aggregates) {
            if (agg.name == col->column) return agg.name;
        }
    }
    return std::nullopt;
}

std::string char1BucketExpr(const GenericColumnExpr& col, const std::string& idxVar) {
    if (col.charDomain.empty()) return "";
    if (col.charDomain.size() == 1) return "0";
    std::string expr = col.column + "[" + idxVar + "]";
    std::string result;
    for (size_t i = 0; i + 1 < col.charDomain.size(); ++i) {
        result += "(" + expr + " == '" + col.charDomain[i] + "' ? " +
                  std::to_string(i) + " : ";
    }
    result += std::to_string(col.charDomain.size() - 1);
    for (size_t i = 0; i + 1 < col.charDomain.size(); ++i)
        result += ")";
    return result;
}

std::string scaledLongExpr(const std::string& rawExpr, int scale) {
    return "(long)round((" + rawExpr + ") * " + std::to_string(scale) + ".0f)";
}

int numericScaleForExpr(const GenericExprPtr& expr) {
    if (!expr) return 0;
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->numericScale;
    if (auto* bin = std::get_if<GenericBinaryExpr>(&expr->node)) {
        const int leftScale = numericScaleForExpr(bin->left);
        const int rightScale = numericScaleForExpr(bin->right);
        switch (bin->op) {
            case ExprOp::ADD:
            case ExprOp::SUB:
                return leftScale == rightScale ? leftScale : 0;
            case ExprOp::MUL:
                if (leftScale > 0 &&
                    bin->right && bin->right->type.type == DataType::INT) {
                    return leftScale;
                }
                if (rightScale > 0 &&
                    bin->left && bin->left->type.type == DataType::INT) {
                    return rightScale;
                }
                return 0;
            case ExprOp::DIV:
                return 0;
        }
    }
    return 0;
}

std::optional<ScalarReduceAccumulatorSpec> buildScalarReduceAccumulatorSpec(
        AggFunc func,
        const GenericExprPtr& arg,
        std::string valueExpr) {
    if (!arg) return std::nullopt;

    ScalarReduceAccumulatorSpec spec;
    spec.valueExpr = std::move(valueExpr);
    spec.metalType = arg->type.type == DataType::FLOAT ? "float" : "long";

    if (func == AggFunc::MIN) {
        spec.op = ScalarReduceAccumulatorSpec::Op::Min;
    } else if (func == AggFunc::MAX) {
        spec.op = ScalarReduceAccumulatorSpec::Op::Max;
    } else if (func != AggFunc::SUM) {
        return std::nullopt;
    }

    if (spec.op == ScalarReduceAccumulatorSpec::Op::Sum &&
        arg->type.type == DataType::FLOAT) {
        spec.outputScale = scalarSumFixedPointScaleForExpr(arg);
        if (spec.outputScale > 0) {
            spec.valueExpr = scaledLongExpr(spec.valueExpr, spec.outputScale);
            spec.metalType = "long";
        }
    } else if (spec.op != ScalarReduceAccumulatorSpec::Op::Sum &&
               arg->type.type != DataType::FLOAT) {
        spec.metalType = "int";
    }

    return spec;
}

std::string distinctDomainSymbolForExpr(const GenericExprPtr& expr) {
    if (!expr) return "";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->distinctDomainSymbol;
    return "";
}

std::string innerColumnName(const GenericExprPtr& expr) {
    if (!expr) return "";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->column;
    return "";
}

std::string groupDisplayNameForAggregate(const GenericAggregateDetail& aggregate,
                                         size_t index) {
    if (index < aggregate.groupNames.size() && !aggregate.groupNames[index].empty())
        return aggregate.groupNames[index];
    if (index < aggregate.groupBy.size())
        return groupDisplayNameForExpr(aggregate.groupBy[index], index);
    return "group_" + std::to_string(index);
}

std::string aggregateOutputFuncFor(const GenericAggregateDetail& aggregate,
                                   size_t index,
                                   AggFunc fallback) {
    if (index < aggregate.aggregateOutputFuncs.size() &&
        !aggregate.aggregateOutputFuncs[index].empty()) {
        return aggregate.aggregateOutputFuncs[index];
    }
    return aggFuncName(fallback);
}

bool buildAggregateInputGroupSpec(
        const GenericAggregateDetail& aggregate,
        const std::string& errorContext,
        GenericGroupSpec& groupSpec,
        std::vector<IrGroupKeyDesc>& groupKeys,
        const AggregateInputColumnBuilder& addInputColumn,
        std::string* error) {
    for (size_t i = 0; i < aggregate.groupBy.size(); ++i) {
        const auto& group = aggregate.groupBy[i];
        const std::string displayName = groupDisplayNameForAggregate(aggregate, i);
        groupSpec.keyColumns.push_back(displayName);
        IrGroupKeyDesc key;
        key.displayName = displayName;
        groupKeys.push_back(std::move(key));
        if (!addInputColumn(displayName,
                            group ? group->type : TypeInfo{DataType::INT, 0},
                            group, 0, "")) {
            return false;
        }
    }

    for (size_t i = 0; i < aggregate.aggregates.size(); ++i) {
        const auto& projection = aggregate.aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg) {
            if (error) *error = errorContext + ": non-aggregate projection.";
            return false;
        }
        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;

        GenericExprPtr inputExpr;
        TypeInfo inputType{DataType::FLOAT, 0};
        int inputScaleDown = 0;
        std::string distinctDomainSymbol;
        std::string funcName = aggregateOutputFuncFor(aggregate, i, agg->func);

        if (agg->func == AggFunc::COUNT) {
            GenericExpr lit;
            lit.type = inputType;
            lit.node = GenericLiteralExpr{1.0, inputType};
            inputExpr = std::make_shared<GenericExpr>(std::move(lit));
            funcName = "COUNT";
        } else {
            if (!agg->arg) {
                if (error) {
                    *error = errorContext + ": aggregate '" +
                             aggFuncName(agg->func) + "' requires an argument.";
                }
                return false;
            }
            inputExpr = agg->arg;
            if (agg->func == AggFunc::COUNT_DISTINCT) {
                if (agg->arg->type.type == DataType::CHAR_FIXED) {
                    if (error) {
                        *error = errorContext +
                                 ": COUNT(DISTINCT) over fixed strings is not supported yet.";
                    }
                    return false;
                }
                distinctDomainSymbol = distinctDomainSymbolForExpr(agg->arg);
                inputType = agg->arg->type;
                funcName = "COUNT_DISTINCT";
            } else if (agg->func == AggFunc::SUM || agg->func == AggFunc::AVG) {
                inputScaleDown = numericScaleForExpr(agg->arg);
            } else if (agg->func != AggFunc::MIN && agg->func != AggFunc::MAX) {
                if (error) {
                    *error = errorContext + ": unsupported aggregate " +
                             aggFuncName(agg->func) + ".";
                }
                return false;
            }
        }

        groupSpec.aggColumns.push_back(displayName);
        groupSpec.aggFuncs.push_back(funcName);
        if (!addInputColumn(displayName, inputType, inputExpr, inputScaleDown,
                            distinctDomainSymbol)) {
            return false;
        }
    }

    groupSpec.outputColumns = aggregate.outputOrder;
    return true;
}

bool aggregateNeedsHashGroupOutput(const GenericAggregateDetail& aggregate) {
    for (const auto& func : aggregate.aggregateOutputFuncs) {
        if (func == "RATIO" || func == "RATIO_DEN")
            return true;
    }
    return false;
}

bool canUseKeyedSingleTableGroup(const GenericAggregateDetail& aggregate) {
    int totalBuckets = 1;
    for (const auto& group : aggregate.groupBy) {
        auto* col = group ? std::get_if<GenericColumnExpr>(&group->node) : nullptr;
        if (!col) return false;

        int numValues = 0;
        if (col->type.type == DataType::CHAR1) {
            numValues = static_cast<int>(col->charDomain.size());
        } else if ((col->type.type == DataType::INT ||
                    col->type.type == DataType::DATE) &&
                   col->hasGroupDomain &&
                   col->domainMax >= col->domainMin) {
            numValues = col->domainMax - col->domainMin + 1;
        } else {
            return false;
        }

        if (numValues <= 0) return false;
        totalBuckets *= numValues;
        if (totalBuckets > 4096) return false;
    }
    return true;
}

} // namespace codegen
