#include "generic/lowering/generic_join_carry.h"
#include "generic/lowering/generic_expression_metal.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace codegen {

namespace {

const IrCarryColumn* findCarry(const IrCarryMap& carries,
                               const GenericColumnExpr& col) {
    auto relIt = carries.find(col.relationInstance.value);
    if (relIt == carries.end()) return nullptr;
    auto colIt = relIt->second.find(col.column);
    if (colIt == relIt->second.end()) return nullptr;
    return &colIt->second;
}

bool isCarriedStringColumn(const GenericColumnExpr& col,
                           const IrCarryMap& carries) {
    if (!findCarry(carries, col)) return false;
    return col.type.type == DataType::CHAR_FIXED ||
           col.type.type == DataType::CHAR1;
}

bool exprContainsCarriedString(const GenericExprPtr& expr,
                               const IrCarryMap& carries,
                               bool allowDirectColumn);

bool predContainsCarriedString(const GenericPredicatePtr& pred,
                               const IrCarryMap& carries);

bool exprContainsCarriedString(const GenericExprPtr& expr,
                               const IrCarryMap& carries,
                               bool allowDirectColumn) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return !allowDirectColumn && isCarriedStringColumn(node, carries);
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return false;
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return exprContainsCarriedString(node.left, carries, false) ||
                   exprContainsCarriedString(node.right, carries, false);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (predContainsCarriedString(branch.condition, carries) ||
                    exprContainsCarriedString(branch.result, carries, false)) {
                    return true;
                }
            }
            return exprContainsCarriedString(node.elseResult, carries, false);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (exprContainsCarriedString(arg, carries, false))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return exprContainsCarriedString(node.arg, carries, false);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys) {
                if (exprContainsCarriedString(key, carries, false))
                    return true;
            }
            return false;
        } else {
            return false;
        }
    }, expr->node);
}

bool predContainsCarriedString(const GenericPredicatePtr& pred,
                               const IrCarryMap& carries) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return exprContainsCarriedString(node.left, carries, false) ||
                   exprContainsCarriedString(node.right, carries, false);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return exprContainsCarriedString(node.expr, carries, false) ||
                   exprContainsCarriedString(node.low, carries, false) ||
                   exprContainsCarriedString(node.high, carries, false);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (exprContainsCarriedString(node.expr, carries, false))
                return true;
            for (const auto& value : node.values) {
                if (exprContainsCarriedString(value, carries, false))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return exprContainsCarriedString(node.expr, carries, false);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predContainsCarriedString(child, carries))
                    return true;
            }
            return false;
        } else {
            return false;
        }
    }, pred->node);
}

[[noreturn]] void failUnsupportedCarryPredicateEmitter(
        const std::string& detail) {
    throw std::logic_error(
        "genericPredicateToMetalWithCarryMap: unsupported predicate " +
        detail + "; call predicateSupported before emitting Metal.");
}

std::optional<std::string> charStringComparisonToMetalWithCarryMap(
        const GenericExprPtr& left,
        CmpOp op,
        const GenericExprPtr& right,
        const std::string& idxVar,
        const IrCarryMap& carries) {
    if (op != CmpOp::EQ && op != CmpOp::NE) return std::nullopt;

    const GenericColumnExpr* col = left ? std::get_if<GenericColumnExpr>(&left->node) : nullptr;
    auto literal = stringLiteralValue(right);
    if (!col || !literal) {
        col = right ? std::get_if<GenericColumnExpr>(&right->node) : nullptr;
        literal = stringLiteralValue(left);
    }
    if (!col || !literal) return std::nullopt;

    std::string eq;
    const IrCarryColumn* carry = findCarry(carries, *col);
    if (col->type.type == DataType::CHAR_FIXED) {
        int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
        eq = carry
            ? fixedStringEqMetalFromPointer(carry->varName, width, *literal)
            : fixedStringEqMetal(*col, *literal, idxVar);
    } else if (col->type.type == DataType::CHAR1) {
        if (literal->empty()) return std::nullopt;
        std::string valueExpr = carry
            ? carry->varName
            : col->column + "[" + idxVar + "]";
        eq = "(" + valueExpr + " == " + genericMetalCharLiteral(literal->front()) + ")";
    } else {
        return std::nullopt;
    }
    return op == CmpOp::NE ? "!(" + eq + ")" : eq;
}

std::optional<std::string> fixedStringLikeMetalWithCarryMap(
        const GenericLikePred& like,
        const std::string& idxVar,
        const IrCarryMap& carries) {
    auto* col = like.expr ? std::get_if<GenericColumnExpr>(&like.expr->node) : nullptr;
    if (!col || col->type.type != DataType::CHAR_FIXED ||
        like.pattern.find('_') != std::string::npos ||
        like.pattern.find('\\') != std::string::npos) {
        return std::nullopt;
    }

    const IrCarryColumn* carry = findCarry(carries, *col);
    if (!carry)
        return fixedStringLikeMetal(like, idxVar);

    int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
    if (like.pattern.find('%') == std::string::npos) {
        std::string exact = fixedStringEqMetalFromPointer(
            carry->varName, width, like.pattern);
        return like.negated ? "!(" + exact + ")" : exact;
    }

    return fixedStringLikeDataMetal(carry->varName, "0u", width,
                                    like.pattern, like.negated);
}

std::string functionExprToMetalWithCarryMap(const GenericFunctionExpr& fn,
                                            const std::string& idxVar,
                                            const IrCarryMap& carries) {
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
        std::string col = fn.args.size() > 1
            ? genericExprToMetalWithCarryMap(fn.args[1], idxVar, carries)
            : "0";
        if (unit == "year") return "(" + col + " / 10000)";
        if (unit == "month") return "((" + col + " / 100) % 100)";
        if (unit == "day") return "(" + col + " % 100)";
        return col;
    }
    return functionExprToMetal(fn, idxVar);
}

} // namespace

int materializedStringLenForExpr(const GenericExprPtr& expr,
                                 const IrCarryMap& carries) {
    if (expr) {
        if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
            if (findCarry(carries, *col) && col->type.type == DataType::CHAR1)
                return 0;
        }
    }
    return fixedStringLenForExpr(expr);
}

std::string genericExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                           const std::string& idxVar,
                                           const IrCarryMap& carries) {
    if (!expr) return "0";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (const auto* carry = findCarry(carries, node))
                return carry->varName;
            return node.column + "[" + idxVar + "]";
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return literalToMetal(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            std::string left = genericExprToMetalWithCarryMap(node.left, idxVar, carries);
            std::string right = genericExprToMetalWithCarryMap(node.right, idxVar, carries);
            switch (node.op) {
                case ExprOp::ADD: return "(" + left + " + " + right + ")";
                case ExprOp::SUB: return "(" + left + " - " + right + ")";
                case ExprOp::MUL: return "(" + left + " * " + right + ")";
                case ExprOp::DIV: return "(" + left + " / " + right + ")";
            }
            return left;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.branches.empty())
                return genericExprToMetalWithCarryMap(node.elseResult, idxVar, carries);
            std::string result;
            for (const auto& branch : node.branches) {
                result += "((" +
                    genericPredicateToMetalWithCarryMap(branch.condition, idxVar, carries) +
                    ") ? (" +
                    genericExprToMetalWithCarryMap(branch.result, idxVar, carries) +
                    ") : ";
            }
            result += "(" + genericExprToMetalWithCarryMap(node.elseResult, idxVar, carries) + ")";
            for (size_t i = 0; i < node.branches.size(); ++i)
                result += ")";
            return result;
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            return functionExprToMetalWithCarryMap(node, idxVar, carries);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return node.arg ? genericExprToMetalWithCarryMap(node.arg, idxVar, carries) : "1";
        } else {
            return genericExprToMetal(expr, idxVar);
        }
    }, expr->node);
}

std::string materializeExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                               const std::string& idxVar,
                                               const IrCarryMap& carries) {
    if (!expr) return "0";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (const auto* carry = findCarry(carries, *col))
            return carry->varName;
        return materializeExprToMetal(expr, idxVar);
    }
    return genericExprToMetalWithCarryMap(expr, idxVar, carries);
}

std::string genericPredicateToMetalWithCarryMap(const GenericPredicatePtr& pred,
                                                const std::string& idxVar,
                                                const IrCarryMap& carries) {
    if (!pred) return "true";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (auto cmp = charStringComparisonToMetalWithCarryMap(
                    node.left, node.op, node.right, idxVar, carries)) {
                return *cmp;
            }
            return "(" + genericExprToMetalWithCarryMap(node.left, idxVar, carries) +
                   " " + cmpOpToMetal(node.op) + " " +
                   genericExprToMetalWithCarryMap(node.right, idxVar, carries) + ")";
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            std::string expr = genericExprToMetalWithCarryMap(node.expr, idxVar, carries);
            return "(" + expr + " >= " +
                   genericExprToMetalWithCarryMap(node.low, idxVar, carries) +
                   " && " + expr + " <= " +
                   genericExprToMetalWithCarryMap(node.high, idxVar, carries) + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (node.values.empty()) return "true";
            std::string expr = genericExprToMetalWithCarryMap(node.expr, idxVar, carries);
            std::string out = "(";
            for (size_t i = 0; i < node.values.size(); ++i) {
                if (i) out += " || ";
                if (auto cmp = charStringComparisonToMetalWithCarryMap(
                        node.expr, CmpOp::EQ, node.values[i], idxVar, carries)) {
                    out += *cmp;
                } else {
                    if (node.expr &&
                        (node.expr->type.type == DataType::INT ||
                         node.expr->type.type == DataType::DATE) &&
                        integerStringLiteralValue(node.values[i])) {
                        out += expr + " == " +
                               std::to_string(*integerStringLiteralValue(node.values[i]));
                    } else {
                        out += expr + " == " +
                               genericExprToMetalWithCarryMap(node.values[i], idxVar, carries);
                    }
                }
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op == GenericLogicalPred::Op::Not) {
                return "!(" + (node.children.empty()
                    ? std::string("true")
                    : genericPredicateToMetalWithCarryMap(node.children.front(),
                                                          idxVar, carries)) + ")";
            }
            if (node.children.empty()) {
                return node.op == GenericLogicalPred::Op::And ? "true" : "false";
            }
            std::string joiner = node.op == GenericLogicalPred::Op::And ? " && " : " || ";
            std::string out = "(";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i) out += joiner;
                out += genericPredicateToMetalWithCarryMap(node.children[i],
                                                           idxVar, carries);
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            if (auto like = fixedStringLikeMetalWithCarryMap(node, idxVar, carries))
                return *like;
            failUnsupportedCarryPredicateEmitter("LIKE shape");
        } else if constexpr (std::is_same_v<T, GenericExistsPred>) {
            failUnsupportedCarryPredicateEmitter(
                node.negated ? "NOT EXISTS" : "EXISTS");
        } else {
            failUnsupportedCarryPredicateEmitter("variant");
        }
    }, pred->node);
}

bool exprNeedsCarriedString(const GenericExprPtr& expr,
                            const IrCarryMap& carries) {
    return exprContainsCarriedString(expr, carries, true);
}

} // namespace codegen
