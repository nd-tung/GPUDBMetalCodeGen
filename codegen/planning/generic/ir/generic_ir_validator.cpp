#include "generic_ir_validator.h"

#include <sstream>
#include <type_traits>

namespace codegen {

namespace {

void addError(GenericIrValidationResult& result, const std::string& msg) {
    result.errors.push_back(msg);
}

bool detailMatchesOp(const GenericRelNode& node) {
    switch (node.op) {
        case GenericRelOp::Scan:
            return std::holds_alternative<GenericScanDetail>(node.detail);
        case GenericRelOp::Filter:
            return std::holds_alternative<GenericFilterDetail>(node.detail);
        case GenericRelOp::Project:
            return std::holds_alternative<GenericProjectDetail>(node.detail);
        case GenericRelOp::Join:
        case GenericRelOp::SemiJoin:
        case GenericRelOp::AntiJoin:
            return std::holds_alternative<GenericJoinDetail>(node.detail);
        case GenericRelOp::Aggregate:
            return std::holds_alternative<GenericAggregateDetail>(node.detail);
        case GenericRelOp::Sort:
            return std::holds_alternative<GenericSortDetail>(node.detail);
        case GenericRelOp::Limit:
            return std::holds_alternative<GenericLimitDetail>(node.detail);
        case GenericRelOp::Materialize:
            return std::holds_alternative<GenericMaterializeDetail>(node.detail);
    }
    return false;
}

size_t expectedInputCount(const GenericRelNode& node) {
    switch (node.op) {
        case GenericRelOp::Scan:
            return 0;
        case GenericRelOp::Join:
        case GenericRelOp::SemiJoin:
        case GenericRelOp::AntiJoin:
            return 2;
        case GenericRelOp::Filter:
        case GenericRelOp::Project:
        case GenericRelOp::Aggregate:
        case GenericRelOp::Sort:
        case GenericRelOp::Limit:
        case GenericRelOp::Materialize:
            return 1;
    }
    return 0;
}

void validateExpr(const GenericRelPlan& plan,
                  const GenericExprPtr& expr,
                  GenericIrValidationResult& result,
                  const std::string& path);

void validatePredicate(const GenericRelPlan& plan,
                       const GenericPredicatePtr& pred,
                       GenericIrValidationResult& result,
                       const std::string& path) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            validateExpr(plan, node.left, result, path + ".left");
            validateExpr(plan, node.right, result, path + ".right");
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            validateExpr(plan, node.expr, result, path + ".expr");
            validateExpr(plan, node.low, result, path + ".low");
            validateExpr(plan, node.high, result, path + ".high");
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            validateExpr(plan, node.expr, result, path + ".expr");
            for (size_t i = 0; i < node.values.size(); ++i)
                validateExpr(plan, node.values[i], result,
                             path + ".values[" + std::to_string(i) + "]");
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            validateExpr(plan, node.expr, result, path + ".expr");
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op == GenericLogicalPred::Op::Not && node.children.size() != 1)
                addError(result, path + ": NOT predicate must have exactly one child");
            for (size_t i = 0; i < node.children.size(); ++i)
                validatePredicate(plan, node.children[i], result,
                                  path + ".children[" + std::to_string(i) + "]");
        } else if constexpr (std::is_same_v<T, GenericExistsPred>) {
            if (node.subqueryIndex < 0)
                addError(result, path + ": EXISTS predicate has no subquery index");
        }
    }, pred->node);
}

void validateExpr(const GenericRelPlan& plan,
                  const GenericExprPtr& expr,
                  GenericIrValidationResult& result,
                  const std::string& path) {
    if (!expr) {
        addError(result, path + ": null expression");
        return;
    }

    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (node.relationInstance.valid() &&
                !plan.findRelationInstance(node.relationInstance)) {
                addError(result, path + ": column references unknown relation instance");
            }
            if (!node.relationInstance.valid() &&
                (!node.table.empty() || !node.alias.empty())) {
                addError(result, path + ": column '" + node.column +
                                  "' did not resolve to a relation instance");
            }
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            validateExpr(plan, node.left, result, path + ".left");
            validateExpr(plan, node.right, result, path + ".right");
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (size_t i = 0; i < node.branches.size(); ++i) {
                validatePredicate(plan, node.branches[i].condition, result,
                                  path + ".branches[" + std::to_string(i) + "].condition");
                validateExpr(plan, node.branches[i].result, result,
                             path + ".branches[" + std::to_string(i) + "].result");
            }
            if (node.elseResult)
                validateExpr(plan, node.elseResult, result, path + ".else");
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (size_t i = 0; i < node.args.size(); ++i)
                validateExpr(plan, node.args[i], result,
                             path + ".args[" + std::to_string(i) + "]");
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            if (!node.star)
                validateExpr(plan, node.arg, result, path + ".arg");
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            if (node.source.valid() && !plan.findNode(node.source))
                addError(result, path + ": scalar lookup references unknown source node");
            for (size_t i = 0; i < node.keys.size(); ++i)
                validateExpr(plan, node.keys[i], result,
                             path + ".keys[" + std::to_string(i) + "]");
        }
    }, expr->node);
}

void validateNodeDetail(const GenericRelPlan& plan,
                        const GenericRelNode& node,
                        GenericIrValidationResult& result) {
    const std::string path = "node[" + std::to_string(node.id.value) + "]";
    std::visit([&](const auto& detail) {
        using T = std::decay_t<decltype(detail)>;
        if constexpr (std::is_same_v<T, GenericScanDetail>) {
            if (!plan.findRelationInstance(detail.relationInstance))
                addError(result, path + ": scan references unknown relation instance");
        } else if constexpr (std::is_same_v<T, GenericFilterDetail>) {
            validatePredicate(plan, detail.predicate, result, path + ".filter");
        } else if constexpr (std::is_same_v<T, GenericProjectDetail>) {
            for (size_t i = 0; i < detail.projections.size(); ++i)
                validateExpr(plan, detail.projections[i].expr, result,
                             path + ".project[" + std::to_string(i) + "]");
        } else if constexpr (std::is_same_v<T, GenericJoinDetail>) {
            validatePredicate(plan, detail.predicate, result, path + ".join");
        } else if constexpr (std::is_same_v<T, GenericAggregateDetail>) {
            for (size_t i = 0; i < detail.groupBy.size(); ++i)
                validateExpr(plan, detail.groupBy[i], result,
                             path + ".groupBy[" + std::to_string(i) + "]");
            for (size_t i = 0; i < detail.aggregates.size(); ++i)
                validateExpr(plan, detail.aggregates[i].expr, result,
                             path + ".aggregate[" + std::to_string(i) + "]");
            validatePredicate(plan, detail.having, result, path + ".having");
        } else if constexpr (std::is_same_v<T, GenericSortDetail>) {
            for (size_t i = 0; i < detail.keys.size(); ++i)
                validateExpr(plan, detail.keys[i].expr, result,
                             path + ".sort[" + std::to_string(i) + "]");
        } else if constexpr (std::is_same_v<T, GenericLimitDetail>) {
            if (detail.limit < 0)
                addError(result, path + ": limit node has negative limit");
        } else if constexpr (std::is_same_v<T, GenericMaterializeDetail>) {
            if (detail.outputName.empty())
                addError(result, path + ": materialize node has empty output name");
        }
    }, node.detail);
}

} // namespace

std::string GenericIrValidationResult::message() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i) out << "; ";
        out << errors[i];
    }
    return out.str();
}

GenericIrValidationResult validateGenericRelationalIR(const GenericRelPlan& plan) {
    GenericIrValidationResult result;

    if (!plan.root.valid()) {
        addError(result, "plan root is invalid");
    } else if (!plan.findNode(plan.root)) {
        addError(result, "plan root references unknown node");
    }

    for (const auto& inst : plan.relationInstances) {
        if (!plan.findRelation(inst.relation))
            addError(result, "relation instance " + std::to_string(inst.id.value) +
                             " references unknown relation");
    }

    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        const auto& node = plan.nodes[i];
        if (node.id.value != static_cast<int>(i)) {
            addError(result, "node at vector index " + std::to_string(i) +
                             " has mismatched id " + std::to_string(node.id.value));
        }
        if (!detailMatchesOp(node)) {
            addError(result, "node " + std::to_string(node.id.value) +
                             " detail does not match op " + genericRelOpName(node.op));
        }
        if (node.inputs.size() != expectedInputCount(node)) {
            addError(result, "node " + std::to_string(node.id.value) +
                             " has " + std::to_string(node.inputs.size()) +
                             " inputs; expected " +
                             std::to_string(expectedInputCount(node)));
        }
        for (const auto& input : node.inputs) {
            if (!plan.findNode(input))
                addError(result, "node " + std::to_string(node.id.value) +
                                 " references unknown input node " +
                                 std::to_string(input.value));
        }
        validateNodeDetail(plan, node, result);
    }

    return result;
}

} // namespace codegen
