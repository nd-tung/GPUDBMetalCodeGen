#include "generic/lowering/generic_multi_table_checks.h"

#include "generic/lowering/generic_scalar_placeholder.h"

#include <type_traits>

namespace codegen {

namespace {

bool predicateContainsEmptyInList(const GenericPredicatePtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericInListPred>) {
            return node.values.empty();
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predicateContainsEmptyInList(child)) return true;
            }
            return false;
        }
        return false;
    }, pred->node);
}

const GenericJoinDetail* joinDetail(const GenericRelNode* node) {
    return node ? std::get_if<GenericJoinDetail>(&node->detail) : nullptr;
}

} // namespace

bool hasScalarSubqueries(const GenericRelPlan& ir) {
    for (const auto& sq : ir.source.subqueries) {
        if (sq.type == GenericSourceSubquery::SCALAR_SUBQUERY)
            return true;
    }
    return false;
}

bool groupedAggregateNeedsScalarPreAgg(
        const MultiTableGroupedAggShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateReferencesScalarSubqueryPlaceholder(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinDetail(joinNode);
        if (join && predicateReferencesScalarSubqueryPlaceholder(join->predicate))
            return true;
    }
    return false;
}

bool materializeNeedsScalarPreAgg(const MultiTableMaterializeShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateReferencesScalarSubqueryPlaceholder(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinDetail(joinNode);
        if (join && predicateReferencesScalarSubqueryPlaceholder(join->predicate))
            return true;
    }
    return false;
}

bool materializeHasEmptyInListPlaceholder(
        const MultiTableMaterializeShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateContainsEmptyInList(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinDetail(joinNode);
        if (join && predicateContainsEmptyInList(join->predicate))
            return true;
    }
    return false;
}

} // namespace codegen
