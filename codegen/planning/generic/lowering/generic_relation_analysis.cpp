#include "generic/lowering/generic_relation_analysis.h"

#include "generic/lowering/generic_plan_shapes.h"

#include <type_traits>

namespace codegen {

const GenericRelation* relationForScan(const GenericRelPlan& ir,
                                       const GenericRelNode* scanNode) {
    auto* scan = scanDetail(scanNode);
    if (!scan) return nullptr;
    auto* inst = ir.findRelationInstance(scan->relationInstance);
    if (!inst) return nullptr;
    return ir.findRelation(inst->relation);
}

void collectPredicateRelations(const GenericPredicatePtr& pred,
                               std::set<int>& relationInstances);

void collectExprRelations(const GenericExprPtr& expr,
                          std::set<int>& relationInstances) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (node.relationInstance.valid())
                relationInstances.insert(node.relationInstance.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectExprRelations(node.left, relationInstances);
            collectExprRelations(node.right, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectPredicateRelations(branch.condition, relationInstances);
                collectExprRelations(branch.result, relationInstances);
            }
            collectExprRelations(node.elseResult, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectExprRelations(arg, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectExprRelations(node.arg, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectExprRelations(key, relationInstances);
        }
    }, expr->node);
}

void collectPredicateRelations(const GenericPredicatePtr& pred,
                               std::set<int>& relationInstances) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectExprRelations(node.left, relationInstances);
            collectExprRelations(node.right, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectExprRelations(node.expr, relationInstances);
            collectExprRelations(node.low, relationInstances);
            collectExprRelations(node.high, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectExprRelations(node.expr, relationInstances);
            for (const auto& value : node.values)
                collectExprRelations(value, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectExprRelations(node.expr, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectPredicateRelations(child, relationInstances);
        }
    }, pred->node);
}

void splitConjuncts(const GenericPredicatePtr& pred,
                    std::vector<GenericPredicatePtr>& out) {
    if (!pred) return;
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op == GenericLogicalPred::Op::And) {
            for (const auto& child : logical->children)
                splitConjuncts(child, out);
            return;
        }
    }
    out.push_back(pred);
}

} // namespace codegen
