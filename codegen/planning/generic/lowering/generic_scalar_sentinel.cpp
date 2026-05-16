#include "generic/lowering/generic_scalar_sentinel.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace codegen {

bool isScalarSubquerySentinelLiteral(const GenericExprPtr& expr) {
    if (!expr) return false;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return false;
    auto* value = std::get_if<int64_t>(&lit->value);
    return value && *value < -1000000;
}

bool predicateReferencesScalarSentinel(const GenericPredicatePtr& pred);

bool exprReferencesScalarSentinel(const GenericExprPtr& expr) {
    if (!expr) return false;
    if (isScalarSubquerySentinelLiteral(expr)) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return exprReferencesScalarSentinel(node.left) ||
                   exprReferencesScalarSentinel(node.right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (exprReferencesScalarSentinel(arg)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (predicateReferencesScalarSentinel(branch.condition) ||
                    exprReferencesScalarSentinel(branch.result)) {
                    return true;
                }
            }
            return exprReferencesScalarSentinel(node.elseResult);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return exprReferencesScalarSentinel(node.arg);
        }
        return false;
    }, expr->node);
}

bool predicateReferencesScalarSentinel(const GenericPredicatePtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return exprReferencesScalarSentinel(node.left) ||
                   exprReferencesScalarSentinel(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return exprReferencesScalarSentinel(node.expr) ||
                   exprReferencesScalarSentinel(node.low) ||
                   exprReferencesScalarSentinel(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (exprReferencesScalarSentinel(node.expr)) return true;
            for (const auto& value : node.values) {
                if (exprReferencesScalarSentinel(value)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return exprReferencesScalarSentinel(node.expr);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predicateReferencesScalarSentinel(child)) return true;
            }
            return false;
        }
        return false;
    }, pred->node);
}

std::optional<int> scalarSubqueryIndexFromSentinelLiteral(
        const GenericExprPtr& expr) {
    if (!isScalarSubquerySentinelLiteral(expr)) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    auto* value = lit ? std::get_if<int64_t>(&lit->value) : nullptr;
    if (!value) return std::nullopt;
    const int64_t idx = *value -
        static_cast<int64_t>(std::numeric_limits<int>::min());
    if (idx < 0 || idx > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return std::nullopt;
    return static_cast<int>(idx);
}

} // namespace codegen
