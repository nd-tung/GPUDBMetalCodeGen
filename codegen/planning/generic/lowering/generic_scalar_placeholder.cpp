#include "generic/lowering/generic_scalar_placeholder.h"

#include <type_traits>

namespace codegen {

bool isScalarSubqueryPlaceholderExpr(const GenericExprPtr& expr) {
    return expr && std::holds_alternative<GenericScalarSubqueryExpr>(expr->node);
}

std::optional<int> scalarSubqueryIndexFromExpr(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* scalar = std::get_if<GenericScalarSubqueryExpr>(&expr->node);
    if (!scalar || scalar->index < 0) return std::nullopt;
    return scalar->index;
}

bool predicateReferencesScalarSubqueryPlaceholder(const GenericPredicatePtr& pred);

bool exprReferencesScalarSubqueryPlaceholder(const GenericExprPtr& expr) {
    if (!expr) return false;
    if (isScalarSubqueryPlaceholderExpr(expr)) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return exprReferencesScalarSubqueryPlaceholder(node.left) ||
                   exprReferencesScalarSubqueryPlaceholder(node.right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (exprReferencesScalarSubqueryPlaceholder(arg)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (predicateReferencesScalarSubqueryPlaceholder(branch.condition) ||
                    exprReferencesScalarSubqueryPlaceholder(branch.result)) {
                    return true;
                }
            }
            return exprReferencesScalarSubqueryPlaceholder(node.elseResult);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return exprReferencesScalarSubqueryPlaceholder(node.arg);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys) {
                if (exprReferencesScalarSubqueryPlaceholder(key)) return true;
            }
            return false;
        }
        return false;
    }, expr->node);
}

bool predicateReferencesScalarSubqueryPlaceholder(const GenericPredicatePtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return exprReferencesScalarSubqueryPlaceholder(node.left) ||
                   exprReferencesScalarSubqueryPlaceholder(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return exprReferencesScalarSubqueryPlaceholder(node.expr) ||
                   exprReferencesScalarSubqueryPlaceholder(node.low) ||
                   exprReferencesScalarSubqueryPlaceholder(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (exprReferencesScalarSubqueryPlaceholder(node.expr)) return true;
            for (const auto& value : node.values) {
                if (exprReferencesScalarSubqueryPlaceholder(value)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return exprReferencesScalarSubqueryPlaceholder(node.expr);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predicateReferencesScalarSubqueryPlaceholder(child)) return true;
            }
            return false;
        }
        return false;
    }, pred->node);
}

} // namespace codegen
