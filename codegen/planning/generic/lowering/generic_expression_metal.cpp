#include "generic/lowering/generic_expression_metal.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <type_traits>

namespace codegen {

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front())))
        name = "c_" + name;
    return name;
}

std::string metalTypeForType(const TypeInfo& type) {
    switch (type.type) {
        case DataType::INT:
        case DataType::DATE:
            return "int";
        case DataType::FLOAT:
            return "float";
        case DataType::CHAR1:
        case DataType::CHAR_FIXED:
            return "char";
    }
    return "int";
}

int fixedStringLenForExpr(const GenericExprPtr& expr) {
    if (!expr) return 0;
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (col->type.type == DataType::CHAR1) return 1;
        if (col->type.type == DataType::CHAR_FIXED) return col->type.fixedWidth;
    }
    return 0;
}

std::string literalToMetal(const GenericLiteralExpr& lit) {
    return std::visit([](const auto& value) -> std::string {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, int64_t>) {
            return std::to_string(value);
        } else if constexpr (std::is_same_v<V, double>) {
            return std::to_string(value) + "f";
        } else {
            std::string escaped;
            escaped.reserve(value.size() + 2);
            escaped.push_back('"');
            for (char ch : value) {
                if (ch == '\\' || ch == '"') escaped.push_back('\\');
                escaped.push_back(ch);
            }
            escaped.push_back('"');
            return escaped;
        }
    }, lit.value);
}

std::optional<std::string> stringLiteralValue(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    auto* value = std::get_if<std::string>(&lit->value);
    if (!value) return std::nullopt;
    return *value;
}

std::optional<int64_t> integerStringLiteralValue(const GenericExprPtr& expr) {
    auto value = stringLiteralValue(expr);
    if (!value || value->empty()) return std::nullopt;
    size_t pos = 0;
    if ((*value)[0] == '-' || (*value)[0] == '+') pos = 1;
    if (pos >= value->size()) return std::nullopt;
    for (size_t i = pos; i < value->size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>((*value)[i])))
            return std::nullopt;
    }
    try {
        return std::stoll(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::string genericMetalCharLiteral(char ch) {
    if (ch == '\\') return "'\\\\'";
    if (ch == '\'') return "'\\\''";
    if (ch == '\0') return "'\\0'";
    return std::string("'") + ch + "'";
}

std::string fixedStringEqMetalFromPointer(const std::string& basePtr,
                                          int width,
                                          const std::string& literal) {
    int cmpLen = std::min(static_cast<int>(literal.size()), width);
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += "(" + basePtr + ")[" + std::to_string(i) + "] == " +
                genericMetalCharLiteral(literal[(size_t)i]);
    }
    for (int i = cmpLen; i < width; ++i) {
        if (!cond.empty()) cond += " && ";
        std::string slot = "(" + basePtr + ")[" + std::to_string(i) + "]";
        cond += "(" + slot + " == '\\0' || " + slot + " == ' ')";
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

namespace {

std::string fixedStringEqMetalImpl(const GenericColumnExpr& col,
                                   const std::string& literal,
                                   const std::string& idxVar) {
    int width = col.type.fixedWidth > 0 ? col.type.fixedWidth : 1;
    int cmpLen = std::min(static_cast<int>(literal.size()), width);
    std::string base = col.column + "[" + idxVar + " * " +
                       std::to_string(width) + " + ";
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += base + std::to_string(i) + "] == " +
                genericMetalCharLiteral(literal[(size_t)i]);
    }
    for (int i = cmpLen; i < width; ++i) {
        if (!cond.empty()) cond += " && ";
        std::string slot = base + std::to_string(i) + "]";
        cond += "(" + slot + " == '\\0' || " + slot + " == ' ')";
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

std::optional<std::string> charStringComparisonToMetal(
        const GenericExprPtr& left,
        CmpOp op,
        const GenericExprPtr& right,
        const std::string& idxVar) {
    if (op != CmpOp::EQ && op != CmpOp::NE) return std::nullopt;

    const GenericColumnExpr* col = left ? std::get_if<GenericColumnExpr>(&left->node) : nullptr;
    auto literal = stringLiteralValue(right);
    if (!col || !literal) {
        col = right ? std::get_if<GenericColumnExpr>(&right->node) : nullptr;
        literal = stringLiteralValue(left);
    }
    if (!col || !literal) return std::nullopt;

    std::string eq;
    if (col->type.type == DataType::CHAR_FIXED) {
        eq = fixedStringEqMetal(*col, *literal, idxVar);
    } else if (col->type.type == DataType::CHAR1) {
        if (literal->empty()) return std::nullopt;
        eq = "(" + col->column + "[" + idxVar + "] == " +
             genericMetalCharLiteral(literal->front()) + ")";
    } else {
        return std::nullopt;
    }
    return op == CmpOp::NE ? "!(" + eq + ")" : eq;
}

std::string genericExprToMetalImpl(const GenericExprPtr& expr,
                                   const std::string& idxVar);
std::string genericPredicateToMetalImpl(const GenericPredicatePtr& pred,
                                        const std::string& idxVar);

std::string inListValueToMetal(const GenericExprPtr& value,
                               const TypeInfo& exprType,
                               const std::string& idxVar) {
    if ((exprType.type == DataType::INT || exprType.type == DataType::DATE) &&
        integerStringLiteralValue(value)) {
        return std::to_string(*integerStringLiteralValue(value));
    }
    return genericExprToMetalImpl(value, idxVar);
}

std::string functionExprToMetalImpl(const GenericFunctionExpr& fn,
                                    const std::string& idxVar) {
    std::string name = fn.name;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });

    if (name == "date_part" || name == "extract") {
        std::string unit;
        if (!fn.args.empty() && fn.args[0]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[0]->node)) {
                if (auto* s = std::get_if<std::string>(&lit->value))
                    unit = *s;
            }
        }
        std::string col = fn.args.size() > 1 ? genericExprToMetalImpl(fn.args[1], idxVar) : "0";
        if (unit == "year") return "(" + col + " / 10000)";
        if (unit == "month") return "((" + col + " / 100) % 100)";
        if (unit == "day") return "(" + col + " % 100)";
        return col;
    }

    if (name == "substring") {
        int start = 1;
        int len = 1;
        if (fn.args.size() > 1 && fn.args[1]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[1]->node)) {
                if (auto* v = std::get_if<int64_t>(&lit->value))
                    start = (int)*v;
            }
        }
        if (fn.args.size() > 2 && fn.args[2]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[2]->node)) {
                if (auto* v = std::get_if<int64_t>(&lit->value))
                    len = (int)*v;
            }
        }

        auto* col = fn.args.empty() || !fn.args[0]
            ? nullptr
            : std::get_if<GenericColumnExpr>(&fn.args[0]->node);
        if (!col) return "0";
        int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
        std::string result = "(";
        for (int i = 0; i < len; ++i) {
            if (i) result += " + ";
            int pos = start - 1 + i;
            int weight = 1;
            for (int w = 0; w < len - 1 - i; ++w) weight *= 10;
            std::string access = col->column + "[" + idxVar + " * " +
                                 std::to_string(width) + " + " +
                                 std::to_string(pos) + "]";
            if (weight > 1)
                result += "(" + access + " - '0') * " + std::to_string(weight);
            else
                result += "(" + access + " - '0')";
        }
        result += ")";
        return result;
    }

    return "0";
}

std::string genericExprToMetalImpl(const GenericExprPtr& expr,
                                   const std::string& idxVar) {
    if (!expr) return "0";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return node.column + "[" + idxVar + "]";
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return literalToMetal(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            std::string left = genericExprToMetalImpl(node.left, idxVar);
            std::string right = genericExprToMetalImpl(node.right, idxVar);
            switch (node.op) {
                case ExprOp::ADD: return "(" + left + " + " + right + ")";
                case ExprOp::SUB: return "(" + left + " - " + right + ")";
                case ExprOp::MUL: return "(" + left + " * " + right + ")";
                case ExprOp::DIV: return "(" + left + " / " + right + ")";
            }
            return left;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.branches.empty())
                return genericExprToMetalImpl(node.elseResult, idxVar);
            std::string result;
            for (const auto& branch : node.branches) {
                result += "((" + genericPredicateToMetalImpl(branch.condition, idxVar) +
                          ") ? (" + genericExprToMetalImpl(branch.result, idxVar) +
                          ") : ";
            }
            result += "(" + genericExprToMetalImpl(node.elseResult, idxVar) + ")";
            for (size_t i = 0; i < node.branches.size(); ++i)
                result += ")";
            return result;
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            return functionExprToMetalImpl(node, idxVar);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return node.arg ? genericExprToMetalImpl(node.arg, idxVar) : "1";
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            return node.outputName;
        }
        return "0";
    }, expr->node);
}

} // namespace

std::string fixedStringEqMetal(const GenericColumnExpr& col,
                               const std::string& literal,
                               const std::string& idxVar) {
    return fixedStringEqMetalImpl(col, literal, idxVar);
}

std::string functionExprToMetal(const GenericFunctionExpr& fn,
                                const std::string& idxVar) {
    return functionExprToMetalImpl(fn, idxVar);
}

std::optional<std::string> fixedStringLikeMetal(
        const GenericLikePred& like,
        const std::string& idxVar) {
    auto* col = like.expr ? std::get_if<GenericColumnExpr>(&like.expr->node) : nullptr;
    if (!col || col->type.type != DataType::CHAR_FIXED ||
        like.pattern.find('_') != std::string::npos ||
        like.pattern.find('\\') != std::string::npos) {
        return std::nullopt;
    }
    int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;

    if (like.pattern.find('%') == std::string::npos) {
        std::string exact = fixedStringEqMetal(*col, like.pattern, idxVar);
        return like.negated ? "!(" + exact + ")" : exact;
    }

    return fixedStringLikeDataMetal(col->column, "(uint)(" + idxVar + ")",
                                    width, like.pattern, like.negated);
}

std::string genericExprToMetal(const GenericExprPtr& expr,
                               const std::string& idxVar) {
    return genericExprToMetalImpl(expr, idxVar);
}

std::string materializeExprToMetal(const GenericExprPtr& expr,
                                   const std::string& idxVar) {
    if (!expr) return "0";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (col->type.type == DataType::CHAR1)
            return col->column + " + " + idxVar;
        if (col->type.type == DataType::CHAR_FIXED) {
            int len = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
            return col->column + " + " + idxVar + " * " + std::to_string(len);
        }
    }
    return genericExprToMetalImpl(expr, idxVar);
}

std::string cmpOpToMetal(CmpOp op) {
    switch (op) {
        case CmpOp::EQ: return "==";
        case CmpOp::NE: return "!=";
        case CmpOp::LT: return "<";
        case CmpOp::LE: return "<=";
        case CmpOp::GT: return ">";
        case CmpOp::GE: return ">=";
    }
    return "==";
}

std::string genericPredicateToMetal(const GenericPredicatePtr& pred,
                                    const std::string& idxVar) {
    return genericPredicateToMetalImpl(pred, idxVar);
}

namespace {

std::string genericPredicateToMetalImpl(const GenericPredicatePtr& pred,
                                        const std::string& idxVar) {
    if (!pred) return "true";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (auto cmp = charStringComparisonToMetal(node.left, node.op,
                                                       node.right, idxVar)) {
                return *cmp;
            }
            return "(" + genericExprToMetalImpl(node.left, idxVar) + " " +
                   cmpOpToMetal(node.op) + " " +
                   genericExprToMetalImpl(node.right, idxVar) + ")";
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            std::string expr = genericExprToMetalImpl(node.expr, idxVar);
            return "(" + expr + " >= " + genericExprToMetalImpl(node.low, idxVar) +
                   " && " + expr + " <= " + genericExprToMetalImpl(node.high, idxVar) + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (node.values.empty()) return "true";
            std::string expr = genericExprToMetalImpl(node.expr, idxVar);
            std::string out = "(";
            for (size_t i = 0; i < node.values.size(); ++i) {
                if (i) out += " || ";
                if (auto cmp = charStringComparisonToMetal(node.expr, CmpOp::EQ,
                                                           node.values[i], idxVar)) {
                    out += *cmp;
                } else {
                    out += expr + " == " +
                           inListValueToMetal(node.values[i],
                                               node.expr ? node.expr->type : TypeInfo{DataType::INT, 0},
                                               idxVar);
                }
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            if (auto like = fixedStringLikeMetal(node, idxVar))
                return *like;
            return node.negated ? "true" : "false";
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op == GenericLogicalPred::Op::Not) {
                return "!(" + (node.children.empty()
                    ? std::string("true")
                    : genericPredicateToMetalImpl(node.children.front(), idxVar)) + ")";
            }
            if (node.children.empty()) {
                return node.op == GenericLogicalPred::Op::And ? "true" : "false";
            }
            std::string joiner = node.op == GenericLogicalPred::Op::And ? " && " : " || ";
            std::string out = "(";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i) out += joiner;
                out += genericPredicateToMetalImpl(node.children[i], idxVar);
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericExistsPred>) {
            return node.negated ? "true" : "false";
        }
        return "true";
    }, pred->node);
}

} // namespace

bool materializeExprSupported(const GenericExprPtr& expr) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return true;
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return !std::holds_alternative<std::string>(node.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return materializeExprSupported(node.left) &&
                   materializeExprSupported(node.right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            std::string name = node.name;
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return name == "date_part" || name == "extract" || name == "substring";
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.type.type != DataType::INT &&
                node.type.type != DataType::DATE &&
                node.type.type != DataType::FLOAT) {
                return false;
            }
            for (const auto& branch : node.branches) {
                if (!predicateSupported(branch.condition) ||
                    !materializeExprSupported(branch.result)) {
                    return false;
                }
            }
            return materializeExprSupported(node.elseResult);
        } else {
            return false;
        }
    }, expr->node);
}

bool predicateSupported(const GenericPredicatePtr& pred) {
    if (!pred) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (charStringComparisonToMetal(node.left, node.op, node.right, "i"))
                return true;
            return materializeExprSupported(node.left) &&
                   materializeExprSupported(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return materializeExprSupported(node.expr) &&
                   materializeExprSupported(node.low) &&
                   materializeExprSupported(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            auto* inCol = node.expr
                ? std::get_if<GenericColumnExpr>(&node.expr->node)
                : nullptr;
            if (inCol && (inCol->type.type == DataType::CHAR_FIXED ||
                          inCol->type.type == DataType::CHAR1)) {
                for (const auto& value : node.values) {
                    if (!stringLiteralValue(value)) return false;
                }
                return true;
            }
            if (!materializeExprSupported(node.expr)) return false;
            for (const auto& value : node.values) {
                if (materializeExprSupported(value)) continue;
                if (node.expr &&
                    (node.expr->type.type == DataType::INT ||
                     node.expr->type.type == DataType::DATE) &&
                    integerStringLiteralValue(value)) {
                    continue;
                }
                return false;
            }
            return true;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return fixedStringLikeMetal(node, "i").has_value();
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (!predicateSupported(child)) return false;
            }
            return true;
        } else {
            return false;
        }
    }, pred->node);
}

} // namespace codegen
