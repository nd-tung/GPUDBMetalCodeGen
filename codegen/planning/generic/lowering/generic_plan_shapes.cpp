#include "generic/lowering/generic_plan_shapes.h"

#include <variant>

namespace codegen {

namespace {

template <typename Shape>
std::optional<Shape> shapeFail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

bool isSupportedJoinTreeOp(GenericRelOp op) {
    return op == GenericRelOp::Join ||
           op == GenericRelOp::SemiJoin ||
           op == GenericRelOp::AntiJoin;
}

bool collectInnerJoinScans(const GenericRelPlan& ir,
                           const GenericRelNode* node,
                           std::vector<const GenericRelNode*>& scans,
                           std::vector<const GenericRelNode*>& joins,
                           std::string* error) {
    if (!node) {
        if (error) *error = "IR multi-table materialize lowerer: null join-tree node.";
        return false;
    }
    if (node->op == GenericRelOp::Scan) {
        scans.push_back(node);
        return true;
    }
    if (!isSupportedJoinTreeOp(node->op)) {
        if (error) *error = "IR multi-table materialize lowerer: join tree contains " +
                            genericRelOpName(node->op) + ".";
        return false;
    }
    auto* detail = std::get_if<GenericJoinDetail>(&node->detail);
    if (!detail || detail->kind == GenericJoinKind::LeftOuter) {
        if (error) *error = "IR multi-table materialize lowerer: only inner/semi/anti joins are supported.";
        return false;
    }
    if (node->inputs.size() != 2) {
        if (error) *error = "IR multi-table materialize lowerer: join must have two inputs.";
        return false;
    }
    joins.push_back(node);
    return collectInnerJoinScans(ir, ir.findNode(node->inputs[0]), scans, joins, error) &&
           collectInnerJoinScans(ir, ir.findNode(node->inputs[1]), scans, joins, error);
}

} // namespace

std::optional<SingleTableShape> parseSingleTableMaterializeShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Project)
        return std::nullopt;
    shape.project = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableShape>(
            error, "IR materialize lowerer: expected Scan under Project/Filter.");
    }
    shape.scan = node;
    return shape;
}

std::optional<SingleTableScalarAggShape> parseSingleTableScalarAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableScalarAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableScalarAggShape>(
            error, "IR scalar aggregate lowerer: expected Scan under Aggregate/Filter.");
    }
    shape.scan = node;
    return shape;
}

std::optional<SingleTableGroupedAggShape> parseSingleTableGroupedAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableGroupedAggShape>(
            error, "IR grouped aggregate lowerer: expected Scan under Aggregate/Filter.");
    }
    shape.scan = node;
    return shape;
}

std::optional<MultiTableMaterializeShape> parseMultiTableMaterializeShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableMaterializeShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Project)
        return std::nullopt;
    shape.project = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableMaterializeShape>(
            error, "IR multi-table materialize lowerer: expected at least two scans.");
    return shape;
}

std::optional<MultiTableGroupedAggShape> parseMultiTableGroupedAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableGroupedAggShape>(
            error, "IR multi-table grouped aggregate lowerer: expected at least two scans.");
    return shape;
}

std::optional<MultiTableGroupedAggShape> parseMultiTableScalarAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableGroupedAggShape>(
            error, "IR multi-table scalar aggregate lowerer: expected at least two scans.");
    return shape;
}

void collectScanRelationInstances(const GenericRelPlan& ir,
                                  const GenericRelNode* node,
                                  std::set<int>& relationInstances) {
    if (!node) return;
    if (node->op == GenericRelOp::Scan) {
        if (auto* scan = std::get_if<GenericScanDetail>(&node->detail)) {
            if (scan->relationInstance.valid())
                relationInstances.insert(scan->relationInstance.value);
        }
        return;
    }
    for (const auto& input : node->inputs)
        collectScanRelationInstances(ir, ir.findNode(input), relationInstances);
}

const GenericScanDetail* scanDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericScanDetail>(&node->detail);
}

const GenericProjectDetail* projectDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericProjectDetail>(&node->detail);
}

const GenericFilterDetail* filterDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericFilterDetail>(&node->detail);
}

const GenericSortDetail* sortDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericSortDetail>(&node->detail);
}

int limitValue(const GenericRelNode* node) {
    if (!node) return -1;
    if (auto* detail = std::get_if<GenericLimitDetail>(&node->detail))
        return detail->limit;
    return -1;
}

const GenericAggregateDetail* aggregateDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericAggregateDetail>(&node->detail);
}

bool projectedColumnMatches(const GenericProjection& projection,
                            const GenericColumnExpr& sortColumn) {
    if (projection.name == sortColumn.column) return true;
    if (!projection.expr) return false;
    auto* col = std::get_if<GenericColumnExpr>(&projection.expr->node);
    if (!col) return false;
    bool sameColumn = col->column == sortColumn.column;
    bool sameTable = sortColumn.table.empty() || col->table == sortColumn.table;
    bool sameAlias = sortColumn.alias.empty() || col->alias == sortColumn.alias;
    return sameColumn && sameTable && sameAlias;
}

std::optional<std::string> sortKeyDisplayName(
        const GenericSortKey& key,
        const GenericProjectDetail& project) {
    if (!key.expr) return std::nullopt;
    if (auto* col = std::get_if<GenericColumnExpr>(&key.expr->node)) {
        for (const auto& projection : project.projections) {
            if (projectedColumnMatches(projection, *col))
                return projection.name;
        }
    }
    return std::nullopt;
}

} // namespace codegen
