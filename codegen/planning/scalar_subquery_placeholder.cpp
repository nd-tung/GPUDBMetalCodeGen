#include "scalar_subquery_placeholder.h"

#include <type_traits>

namespace codegen {

bool analyzedPredicateReferencesScalarSubquery(const PredPtr& pred);

bool analyzedExprReferencesScalarSubquery(const ExprPtr& expr) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ScalarSubqueryRef>) {
            return true;
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            return analyzedExprReferencesScalarSubquery(node.left) ||
                   analyzedExprReferencesScalarSubquery(node.right);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (const auto& arg : node.args) {
                if (analyzedExprReferencesScalarSubquery(arg)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (const auto& branch : node.branches) {
                if (analyzedPredicateReferencesScalarSubquery(branch.condition) ||
                    analyzedExprReferencesScalarSubquery(branch.result)) {
                    return true;
                }
            }
            return analyzedExprReferencesScalarSubquery(node.elseResult);
        }
        return false;
    }, expr->node);
}

bool analyzedPredicateReferencesScalarSubquery(const PredPtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            return analyzedExprReferencesScalarSubquery(node.left) ||
                   analyzedExprReferencesScalarSubquery(node.right);
        } else if constexpr (std::is_same_v<T, Between>) {
            return analyzedExprReferencesScalarSubquery(node.expr) ||
                   analyzedExprReferencesScalarSubquery(node.low) ||
                   analyzedExprReferencesScalarSubquery(node.high);
        } else if constexpr (std::is_same_v<T, InList>) {
            if (analyzedExprReferencesScalarSubquery(node.expr)) return true;
            for (const auto& value : node.values) {
                if (analyzedExprReferencesScalarSubquery(value)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, Like>) {
            return analyzedExprReferencesScalarSubquery(node.expr);
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children) {
                if (analyzedPredicateReferencesScalarSubquery(child)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            return analyzedPredicateReferencesScalarSubquery(node.child);
        }
        return false;
    }, pred->node);
}

bool analyzedExprIsScalarSubqueryRef(const ExprPtr& expr,
                                     int scalarSubqueryIndex) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ScalarSubqueryRef>) {
            return node.index == scalarSubqueryIndex;
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            return analyzedExprIsScalarSubqueryRef(node.left, scalarSubqueryIndex) ||
                   analyzedExprIsScalarSubqueryRef(node.right, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (const auto& arg : node.args) {
                if (analyzedExprIsScalarSubqueryRef(arg, scalarSubqueryIndex))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (const auto& branch : node.branches) {
                if (analyzedPredicateReferencesScalarSubquery(
                        branch.condition, scalarSubqueryIndex) ||
                    analyzedExprIsScalarSubqueryRef(
                        branch.result, scalarSubqueryIndex)) {
                    return true;
                }
            }
            return analyzedExprIsScalarSubqueryRef(node.elseResult,
                                                  scalarSubqueryIndex);
        }
        return false;
    }, expr->node);
}

bool analyzedPredicateReferencesScalarSubquery(const PredPtr& pred,
                                               int scalarSubqueryIndex) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            return analyzedExprIsScalarSubqueryRef(node.left, scalarSubqueryIndex) ||
                   analyzedExprIsScalarSubqueryRef(node.right, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, Between>) {
            return analyzedExprIsScalarSubqueryRef(node.expr, scalarSubqueryIndex) ||
                   analyzedExprIsScalarSubqueryRef(node.low, scalarSubqueryIndex) ||
                   analyzedExprIsScalarSubqueryRef(node.high, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, InList>) {
            if (analyzedExprIsScalarSubqueryRef(node.expr, scalarSubqueryIndex))
                return true;
            for (const auto& candidate : node.values) {
                if (analyzedExprIsScalarSubqueryRef(candidate, scalarSubqueryIndex))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, Like>) {
            return analyzedExprIsScalarSubqueryRef(node.expr, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children) {
                if (analyzedPredicateReferencesScalarSubquery(child,
                                                              scalarSubqueryIndex)) {
                    return true;
                }
            }
            return false;
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            return analyzedPredicateReferencesScalarSubquery(node.child,
                                                            scalarSubqueryIndex);
        }
        return false;
    }, pred->node);
}

} // namespace codegen
