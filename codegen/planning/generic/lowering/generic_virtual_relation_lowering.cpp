#include "generic/lowering/generic_ir_physical_planner.h"

#include "core/schema_provider.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "metal_plan_common.h"
#include "scalar_subquery_placeholder.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

bool genericExprReferencesScalarSubquery(const GenericExprPtr& expr,
                                         int scalarSubqueryIndex);

bool genericPredicateReferencesScalarSubquery(const GenericPredicatePtr& pred,
                                              int scalarSubqueryIndex) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return genericExprReferencesScalarSubquery(node.left, scalarSubqueryIndex) ||
                   genericExprReferencesScalarSubquery(node.right, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return genericExprReferencesScalarSubquery(node.expr, scalarSubqueryIndex) ||
                   genericExprReferencesScalarSubquery(node.low, scalarSubqueryIndex) ||
                   genericExprReferencesScalarSubquery(node.high, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (genericExprReferencesScalarSubquery(node.expr, scalarSubqueryIndex))
                return true;
            for (const auto& value : node.values) {
                if (genericExprReferencesScalarSubquery(value, scalarSubqueryIndex))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return genericExprReferencesScalarSubquery(node.expr, scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (genericPredicateReferencesScalarSubquery(child,
                                                             scalarSubqueryIndex)) {
                    return true;
                }
            }
            return false;
        } else {
            return false;
        }
    }, pred->node);
}

bool genericExprReferencesScalarSubquery(const GenericExprPtr& expr,
                                         int scalarSubqueryIndex) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericScalarSubqueryExpr>) {
            return node.index == scalarSubqueryIndex;
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return genericExprReferencesScalarSubquery(node.left,
                                                       scalarSubqueryIndex) ||
                   genericExprReferencesScalarSubquery(node.right,
                                                       scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (genericPredicateReferencesScalarSubquery(
                        branch.condition, scalarSubqueryIndex) ||
                    genericExprReferencesScalarSubquery(branch.result,
                                                        scalarSubqueryIndex)) {
                    return true;
                }
            }
            return genericExprReferencesScalarSubquery(node.elseResult,
                                                       scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (genericExprReferencesScalarSubquery(arg,
                                                        scalarSubqueryIndex)) {
                    return true;
                }
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return genericExprReferencesScalarSubquery(node.arg,
                                                       scalarSubqueryIndex);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys) {
                if (genericExprReferencesScalarSubquery(key,
                                                        scalarSubqueryIndex)) {
                    return true;
                }
            }
            return false;
        } else {
            return false;
        }
    }, expr->node);
}

bool genericFiltersReferenceScalarSubquery(const GenericRelPlan& ir,
                                           int scalarSubqueryIndex) {
    for (const auto& node : ir.nodes) {
        if (node.op != GenericRelOp::Filter) continue;
        auto* filter = std::get_if<GenericFilterDetail>(&node.detail);
        if (!filter) continue;
        if (genericPredicateReferencesScalarSubquery(filter->predicate,
                                                     scalarSubqueryIndex)) {
            return true;
        }
    }
    return false;
}

void collectGenericJoinEqualities(
        const GenericPredicatePtr& pred,
        std::vector<std::pair<GenericColumnExpr, GenericColumnExpr>>& out) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (node.op != CmpOp::EQ || !node.left || !node.right) return;
            auto* left = std::get_if<GenericColumnExpr>(&node.left->node);
            auto* right = std::get_if<GenericColumnExpr>(&node.right->node);
            if (left && right) out.push_back({*left, *right});
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op != GenericLogicalPred::Op::And) return;
            for (const auto& child : node.children)
                collectGenericJoinEqualities(child, out);
        }
    }, pred->node);
}

std::vector<std::pair<GenericColumnExpr, GenericColumnExpr>>
genericOuterJoinEqualities(const GenericRelPlan& ir) {
    std::vector<std::pair<GenericColumnExpr, GenericColumnExpr>> out;
    for (const auto& node : ir.nodes) {
        if (node.op != GenericRelOp::Join &&
            node.op != GenericRelOp::SemiJoin &&
            node.op != GenericRelOp::AntiJoin) {
            continue;
        }
        auto* join = std::get_if<GenericJoinDetail>(&node.detail);
        if (!join) continue;
        collectGenericJoinEqualities(join->predicate, out);
    }
    return out;
}

std::string genericColumnQualifierForFromMatch(const GenericColumnExpr& col) {
    if (!col.alias.empty()) return col.alias;
    return col.table;
}

struct FromSubqueryProjectShape {
    const GenericProjectDetail* project = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct FromSubqueryAggregateShape {
    const GenericAggregateDetail* aggregate = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

std::optional<FromSubqueryProjectShape> parseFromSubqueryProjectShape(
        const GenericRelPlan& ir) {
    FromSubqueryProjectShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = sortDetail(node) ? node : nullptr;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Project)
        return std::nullopt;
    shape.project = projectDetail(node);
    if (!shape.project) return std::nullopt;
    return shape;
}

std::optional<FromSubqueryAggregateShape> parseFromSubqueryAggregateShape(
        const GenericRelPlan& ir) {
    FromSubqueryAggregateShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = sortDetail(node) ? node : nullptr;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = aggregateDetail(node);
    if (!shape.aggregate) return std::nullopt;
    return shape;
}

bool genericColumnMatchesTable(const GenericColumnExpr& col,
                               const std::string& table) {
    return col.table == table || col.alias == table;
}

void collectGenericColumnsForTable(const GenericExprPtr& expr,
                                   const std::string& table,
                                   std::set<std::string>& out);

void collectGenericPredicateColumnsForTable(const GenericPredicatePtr& pred,
                                            const std::string& table,
                                            std::set<std::string>& out) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectGenericColumnsForTable(node.left, table, out);
            collectGenericColumnsForTable(node.right, table, out);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectGenericColumnsForTable(node.expr, table, out);
            collectGenericColumnsForTable(node.low, table, out);
            collectGenericColumnsForTable(node.high, table, out);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectGenericColumnsForTable(node.expr, table, out);
            for (const auto& value : node.values)
                collectGenericColumnsForTable(value, table, out);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectGenericColumnsForTable(node.expr, table, out);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectGenericPredicateColumnsForTable(child, table, out);
        }
    }, pred->node);
}

void collectGenericColumnsForTable(const GenericExprPtr& expr,
                                   const std::string& table,
                                   std::set<std::string>& out) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (genericColumnMatchesTable(node, table))
                out.insert(node.column);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectGenericColumnsForTable(node.left, table, out);
            collectGenericColumnsForTable(node.right, table, out);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectGenericPredicateColumnsForTable(
                    branch.condition, table, out);
                collectGenericColumnsForTable(branch.result, table, out);
            }
            collectGenericColumnsForTable(node.elseResult, table, out);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectGenericColumnsForTable(arg, table, out);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectGenericColumnsForTable(node.arg, table, out);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectGenericColumnsForTable(key, table, out);
        }
    }, expr->node);
}

void collectGenericColumns(const GenericExprPtr& expr,
                           std::set<std::string>& out);

void collectGenericPredicateColumns(const GenericPredicatePtr& pred,
                                    std::set<std::string>& out) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectGenericColumns(node.left, out);
            collectGenericColumns(node.right, out);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectGenericColumns(node.expr, out);
            collectGenericColumns(node.low, out);
            collectGenericColumns(node.high, out);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectGenericColumns(node.expr, out);
            for (const auto& value : node.values)
                collectGenericColumns(value, out);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectGenericColumns(node.expr, out);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectGenericPredicateColumns(child, out);
        }
    }, pred->node);
}

void collectGenericColumns(const GenericExprPtr& expr,
                           std::set<std::string>& out) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            out.insert(node.column);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectGenericColumns(node.left, out);
            collectGenericColumns(node.right, out);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectGenericPredicateColumns(branch.condition, out);
                collectGenericColumns(branch.result, out);
            }
            collectGenericColumns(node.elseResult, out);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectGenericColumns(arg, out);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectGenericColumns(node.arg, out);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectGenericColumns(key, out);
        }
    }, expr->node);
}

void collectGenericExprTables(const GenericExprPtr& expr,
                              std::set<std::string>& tables);

void collectGenericPredicateTables(const GenericPredicatePtr& pred,
                                   std::set<std::string>& tables) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectGenericExprTables(node.left, tables);
            collectGenericExprTables(node.right, tables);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectGenericExprTables(node.expr, tables);
            collectGenericExprTables(node.low, tables);
            collectGenericExprTables(node.high, tables);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectGenericExprTables(node.expr, tables);
            for (const auto& value : node.values)
                collectGenericExprTables(value, tables);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectGenericExprTables(node.expr, tables);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectGenericPredicateTables(child, tables);
        }
    }, pred->node);
}

void collectGenericExprTables(const GenericExprPtr& expr,
                              std::set<std::string>& tables) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            std::string table = node.table.empty() ? node.alias : node.table;
            if (!table.empty()) tables.insert(std::move(table));
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectGenericExprTables(node.left, tables);
            collectGenericExprTables(node.right, tables);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectGenericPredicateTables(branch.condition, tables);
                collectGenericExprTables(branch.result, tables);
            }
            collectGenericExprTables(node.elseResult, tables);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectGenericExprTables(arg, tables);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectGenericExprTables(node.arg, tables);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectGenericExprTables(key, tables);
        }
    }, expr->node);
}

std::string combineGenericFilters(
        const std::vector<GenericPredicatePtr>& filters,
        const std::string& idxVar) {
    std::string cond;
    for (const auto& filter : filters) {
        if (!filter) continue;
        if (!cond.empty()) cond += " && ";
        cond += "(" + genericPredicateToMetal(filter, idxVar) + ")";
    }
    return cond;
}

const GenericAggregateExpr* genericAggregateForProjection(
        const GenericProjection& projection) {
    if (!projection.expr) return nullptr;
    return std::get_if<GenericAggregateExpr>(&projection.expr->node);
}

std::string keyDomainSymbolForAnalyzedColumn(const std::string& table,
                                             const std::string& column,
                                             const SchemaProvider* schema) {
    if (!schema) return "";
    auto keySym = schema->keyDomainSymbol(table, column);
    if (!keySym.empty()) return keySym;
    if (auto gd = schema->groupDomain(table, column))
        return std::to_string(gd->maxValue + 1);
    auto pk = schema->pkInfo(table);
    if (pk && pk->first == column) return pk->second;
    return schema->maxKeySymbol(table);
}

std::string fromSubqueryBaseForKey(const GenericFromSubqueryAggInfo& info,
                                   const std::string& tableKey) {
    for (size_t i = 0; i < info.tables.size(); ++i) {
        if (info.tables[i] == tableKey) return info.tables[i];
        if (i < info.tableAliases.size() && info.tableAliases[i] == tableKey)
            return info.tables[i];
    }
    return tableKey;
}

bool fromSubqueryColMatches(const GenericFromSubqueryAggInfo& info,
                            const GenericColumnExpr& col,
                            const std::string& tableKey,
                            const std::string& column) {
    if (col.column != column) return false;
    if (col.table == tableKey) return true;
    if (!col.alias.empty() && col.alias == tableKey) return true;
    return fromSubqueryBaseForKey(info, tableKey) == col.table;
}

struct FromSubqueryScalarExtremumForIr {
    int sqIdx = -1;
    AggFunc func = AggFunc::MAX;
    std::string argAlias;
};

std::optional<FromSubqueryScalarExtremumForIr>
parseFromSubqueryScalarExtremumForIr(const GenericRelPlan& ir,
                                     const GenericSourceQueryInfo& aq,
                                     const GenericFromSubqueryAggInfo& fsq,
                                     const std::string& aggregateAlias) {
    if (fsq.alias.empty() || aggregateAlias.empty()) return std::nullopt;
    for (size_t sqIdx = 0; sqIdx < aq.subqueries.size(); ++sqIdx) {
        const auto& sq = aq.subqueries[sqIdx];
        if (sq.type != GenericSourceSubquery::SCALAR_SUBQUERY) continue;
        if (!genericFiltersReferenceScalarSubquery(
                ir, static_cast<int>(sqIdx))) {
            continue;
        }
        for (const auto& extremum : sq.fromSubqueryScalarExtrema) {
            if (extremum.sourceAlias != fsq.alias ||
                extremum.argAlias != aggregateAlias) {
                continue;
            }
            FromSubqueryScalarExtremumForIr out;
            out.sqIdx = static_cast<int>(sqIdx);
            out.func = extremum.func;
            out.argAlias = aggregateAlias;
            return out;
        }
    }
    return std::nullopt;
}

class MetalIrAtomicFloatSumWithSeen : public MetalUnaryOperator {
public:
    MetalIrAtomicFloatSumWithSeen(std::unique_ptr<MetalOperator> child,
                                    std::string inputArray,
                                    std::string seenArray,
                                    std::string bucketExpr,
                                    std::string valueExpr,
                                    std::string sizeExpr)
        : MetalUnaryOperator(std::move(child)),
          inputArray_(std::move(inputArray)),
          seenArray_(std::move(seenArray)),
          bucketExpr_(std::move(bucketExpr)),
          valueExpr_(std::move(valueExpr)),
          sizeExpr_(std::move(sizeExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addAtomicBufferParam(inputArray_, "atomic_float", sizeExpr_);
        cg.addAtomicBufferParam(seenArray_, "atomic_uint", sizeExpr_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_from_subquery_bucket = (uint)(" +
                       bucketExpr_ + ");");
            cg.addLine("atomic_fetch_add_explicit(&" + inputArray_ +
                       "[_ir_from_subquery_bucket], (float)(" + valueExpr_ +
                       "), memory_order_relaxed);");
            cg.addLine("atomic_store_explicit(&" + seenArray_ +
                       "[_ir_from_subquery_bucket], 1u, memory_order_relaxed);");
            consume();
        });
    }

    std::string describe() const override {
        return "IrAtomicFloatSumWithSeen(" + inputArray_ + "[" +
               bucketExpr_ + "])";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(bucketExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string inputArray_;
    std::string seenArray_;
    std::string bucketExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
};

class MetalIrAtomicExtremumFloatArray : public MetalUnaryOperator {
public:
    MetalIrAtomicExtremumFloatArray(std::unique_ptr<MetalOperator> child,
                                    std::string inputArray,
                                    std::string seenArray,
                                    std::string outputScalar,
                                    std::string stateScalar,
                                    std::string indexExpr,
                                    bool useMax)
        : MetalUnaryOperator(std::move(child)),
          inputArray_(std::move(inputArray)),
          seenArray_(std::move(seenArray)),
          outputScalar_(std::move(outputScalar)),
          stateScalar_(std::move(stateScalar)),
          indexExpr_(std::move(indexExpr)),
          useMax_(useMax) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(inputArray_, "const atomic_float", "", false, 0);
        cg.addBufferParam(seenArray_, "const atomic_uint", "", false, 0);
        cg.addAtomicBufferParam(outputScalar_, "atomic_uint", "1");
        cg.addAtomicBufferParam(stateScalar_, "atomic_uint", "1");
        child_->produce(cg, [&]() {
            cg.addIf("atomic_load_explicit(&" + seenArray_ + "[" +
                     indexExpr_ + "], memory_order_relaxed) != 0u", [&]() {
                cg.addLine("float _ir_from_subquery_value = atomic_load_explicit(&" +
                           inputArray_ + "[" + indexExpr_ +
                           "], memory_order_relaxed);");
                if (useMax_) {
                    cg.addLine("atomic_max_float(&" + outputScalar_ +
                               "[0], _ir_from_subquery_value);");
                    cg.addLine("atomic_store_explicit(&" + stateScalar_ +
                               "[0], 2u, memory_order_relaxed);");
                } else {
                    cg.addLine("atomic_min_float_seen(&" + outputScalar_ +
                               "[0], &" + stateScalar_ +
                               "[0], _ir_from_subquery_value);");
                }
            });
            consume();
        });
    }

    std::string describe() const override {
        return std::string(useMax_ ? "IrAtomicMax" : "IrAtomicMin") +
               "FloatArray(" + inputArray_ + ")";
    }

private:
    std::string inputArray_;
    std::string seenArray_;
    std::string outputScalar_;
    std::string stateScalar_;
    std::string indexExpr_;
    bool useMax_ = true;
};

class MetalIrCountHistogram : public MetalUnaryOperator {
public:
    MetalIrCountHistogram(std::unique_ptr<MetalOperator> child,
                          std::string countBuffer,
                          std::string histBuffer,
                          std::string groupKeyExpr,
                          int bucketCap,
                          int localBucketCap)
        : MetalUnaryOperator(std::move(child)),
          countBuffer_(std::move(countBuffer)),
          histBuffer_(std::move(histBuffer)),
          groupKeyExpr_(std::move(groupKeyExpr)),
          bucketCap_(bucketCap),
          localBucketCap_(localBucketCap) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const int bucketCap = std::max(1, bucketCap_);
        const int localBucketCap =
            std::max(0, std::min(localBucketCap_, bucketCap));
        cg.addBufferParam(countBuffer_, "const atomic_uint", "", false, 0);
        cg.addAtomicBufferParam(histBuffer_, "atomic_uint",
                                std::to_string(bucketCap));

        const std::string suffix = sanitizeIdentifier(histBuffer_);
        if (localBucketCap > 0) {
            const std::string localSize = std::to_string(localBucketCap);
            cg.setPhaseMaxThreadgroups(1024);
            cg.addLine("threadgroup uint _tg_hist_" + suffix + "[" +
                       localSize + "];");
            cg.addBlock("for (uint _h = lid; _h < " + localSize +
                        "u; _h += tg_size)", [&]() {
                cg.addLine("_tg_hist_" + suffix + "[_h] = 0u;");
            });
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        }

        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_hist_group_" + suffix + " = (uint)(" +
                       groupKeyExpr_ + ");");
            cg.addLine("uint _ir_hist_bucket_" + suffix +
                       " = atomic_load_explicit(&" + countBuffer_ +
                       "[_ir_hist_group_" + suffix +
                       "], memory_order_relaxed);");
            cg.addIf("_ir_hist_bucket_" + suffix + " >= " +
                     std::to_string(bucketCap) + "u", [&]() {
                cg.addLine("_ir_hist_bucket_" + suffix + " = " +
                           std::to_string(bucketCap - 1) + "u;");
            });
            if (localBucketCap > 0) {
                cg.addIf("_ir_hist_bucket_" + suffix + " < " +
                         std::to_string(localBucketCap) + "u", [&]() {
                    cg.addLine(
                        "atomic_fetch_add_explicit((threadgroup atomic_uint*)&_tg_hist_" +
                        suffix + "[_ir_hist_bucket_" + suffix +
                        "], 1u, memory_order_relaxed);");
                });
                cg.addIf("_ir_hist_bucket_" + suffix + " >= " +
                         std::to_string(localBucketCap) + "u", [&]() {
                    cg.addLine("atomic_fetch_add_explicit(&" + histBuffer_ +
                               "[_ir_hist_bucket_" + suffix +
                               "], 1u, memory_order_relaxed);");
                });
            } else {
                cg.addLine("atomic_fetch_add_explicit(&" + histBuffer_ +
                           "[_ir_hist_bucket_" + suffix +
                           "], 1u, memory_order_relaxed);");
            }
            consume();
        });

        if (localBucketCap > 0) {
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            cg.addBlock("for (uint _h = lid; _h < " +
                        std::to_string(localBucketCap) +
                        "u; _h += tg_size)", [&]() {
                cg.addIf("_tg_hist_" + suffix + "[_h] > 0u", [&]() {
                    cg.addLine("atomic_fetch_add_explicit(&" + histBuffer_ +
                               "[_h], _tg_hist_" + suffix +
                               "[_h], memory_order_relaxed);");
                });
            });
        }
    }

    std::string describe() const override {
        return "IrCountHistogram(" + countBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(groupKeyExpr_, out);
    }

private:
    std::string countBuffer_;
    std::string histBuffer_;
    std::string groupKeyExpr_;
    int bucketCap_ = 1;
    int localBucketCap_ = 0;
};

std::optional<MetalQueryPlan> lowerFromSubqueryTopScalarIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    const auto& source = ir.source;
    const auto* schema = ir.schema;
    if (source.fromSubqueryAggs.size() != 1) return std::nullopt;
    auto shape = parseFromSubqueryProjectShape(ir);
    if (!shape || !shape->project) return std::nullopt;
    const auto& project = *shape->project;

    const auto& fsq = source.fromSubqueryAggs[0];
    if (fsq.tables.size() != 1 || fsq.groupBy.size() != 1)
        return std::nullopt;
    auto* groupCol = fsq.groupBy[0]
        ? std::get_if<GenericColumnExpr>(&fsq.groupBy[0]->node)
        : nullptr;
    if (!groupCol) return std::nullopt;

    const GenericFromSubqueryAggTarget* innerAgg = nullptr;
    FromSubqueryScalarExtremumForIr scalarExtremum;
    for (const auto& target : fsq.aggregates) {
        auto parsed = parseFromSubqueryScalarExtremumForIr(
            ir, source, fsq, target.name);
        if (!parsed) continue;
        innerAgg = &target;
        scalarExtremum = *parsed;
        break;
    }
    if (!innerAgg) return std::nullopt;
    if (scalarExtremum.func != AggFunc::MAX) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer currently supports scalar MAX.");
    }
    if (innerAgg->func != AggFunc::SUM) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer currently supports SUM aggregate values.");
    }
    TypeInfo aggValueType = innerAgg->arg ? innerAgg->arg->type : innerAgg->type;
    if (!innerAgg->arg ||
        (aggValueType.type != DataType::INT &&
         aggValueType.type != DataType::FLOAT &&
         aggValueType.type != DataType::DATE)) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer requires a numeric aggregate expression.");
    }

    std::string viewBase = fromSubqueryBaseForKey(
        fsq, groupCol->alias.empty()
                 ? groupCol->table
                 : groupCol->alias);
    if (viewBase.empty()) viewBase = groupCol->table;
    if (viewBase.empty()) return std::nullopt;

    bool foundOuterJoin = false;
    std::string outerTable;
    std::string outerKeyCol;
    for (const auto& [leftCol, rightCol] : genericOuterJoinEqualities(ir)) {
        const std::string leftKey = genericColumnQualifierForFromMatch(leftCol);
        const std::string rightKey = genericColumnQualifierForFromMatch(rightCol);
        const bool leftIsViewKey =
            fromSubqueryColMatches(fsq, *groupCol, leftKey, leftCol.column);
        const bool rightIsViewKey =
            fromSubqueryColMatches(fsq, *groupCol, rightKey, rightCol.column);
        if (leftIsViewKey == rightIsViewKey) continue;
        const std::string candidateTable = leftIsViewKey
            ? fromSubqueryBaseForKey(fsq, rightKey)
            : fromSubqueryBaseForKey(fsq, leftKey);
        const std::string candidateKey =
            leftIsViewKey ? rightCol.column : leftCol.column;
        if (std::find(fsq.tables.begin(), fsq.tables.end(),
                      candidateTable) != fsq.tables.end()) {
            continue;
        }
        foundOuterJoin = true;
        outerTable = candidateTable;
        outerKeyCol = candidateKey;
        break;
    }
    if (!foundOuterJoin || outerTable.empty() || outerKeyCol.empty())
        return std::nullopt;

    const std::string sizeSymbol = keyDomainSymbolForAnalyzedColumn(
        viewBase, groupCol->column, schema);
    if (sizeSymbol.empty()) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer: group key has no schema domain.");
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_FROM_SUBQUERY_TOP_SCALAR";
    const std::string idxVar = "i";
    const std::string tag = sanitizeIdentifier(
        fsq.alias.empty() ? "from_subquery" : fsq.alias);
    const std::string aggAlias = innerAgg->name;
    const std::string aggBuffer = "d_ir_from_subquery_" + tag + "_" +
        sanitizeIdentifier(aggAlias);
    const std::string aggSeenBuffer = aggBuffer + "_seen_keys";
    const std::string extremumBuffer = aggBuffer + "_" +
        (scalarExtremum.func == AggFunc::MAX ? "max" : "min");
    const std::string extremumState = extremumBuffer + "_seen";

    {
        std::set<std::string> scanCols{groupCol->column};
        collectGenericColumns(innerAgg->arg, scanCols);
        for (const auto& filter : fsq.filters)
            collectGenericPredicateColumns(filter, scanCols);
        auto scan = makeScanForCols(viewBase, idxVar, scanCols, schema);
        auto filtered = maybeSelect(
            std::move(scan), combineGenericFilters(fsq.filters, idxVar));
        const std::string valueExpr =
            genericExprToMetal(innerAgg->arg, idxVar);
        auto agg = std::make_unique<MetalIrAtomicFloatSumWithSeen>(
            std::move(filtered), aggBuffer, aggSeenBuffer,
            groupCol->column + "[" + idxVar + "]", valueExpr, sizeSymbol);
        appendPhase(plan, "GENERIC_ir_from_subquery_aggregate_" + tag,
                    std::move(agg));
    }

    {
        auto range = std::make_unique<MetalRangeScan>(sizeSymbol, idxVar);
        auto extremum = std::make_unique<MetalIrAtomicExtremumFloatArray>(
            std::move(range), aggBuffer, aggSeenBuffer, extremumBuffer,
            extremumState, idxVar, scalarExtremum.func == AggFunc::MAX);
        appendPhase(plan, "GENERIC_ir_from_subquery_extremum_" + tag,
                    std::move(extremum));
    }

    {
        std::set<std::string> scanCols{outerKeyCol};
        for (const auto& projection : project.projections) {
            if (projection.name == aggAlias)
                continue;
            collectGenericColumnsForTable(projection.expr, outerTable, scanCols);
        }
        auto scan = makeScanForCols(outerTable, idxVar, scanCols, schema);
        const std::string outerKeyExpr = outerKeyCol + "[" + idxVar + "]";
        const std::string aggValueExpr = "atomic_load_explicit(&" +
            aggBuffer + "[" + outerKeyExpr + "], memory_order_relaxed)";
        const std::string extremumExpr =
            "as_type<float>(atomic_load_explicit(&" + extremumBuffer +
            "[0], memory_order_relaxed))";
        const std::string stateReady = "(atomic_load_explicit(&" +
            extremumState + "[0], memory_order_relaxed) == 2u)";
        auto filtered = maybeSelect(
            std::move(scan),
            stateReady + " && (" + aggValueExpr + " == " +
                extremumExpr + ")");

        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(filtered), "d_ir_from_subquery_" + tag + "_result_count",
            "1");
        const std::string outputSize = tableSizeName(outerTable);
        std::vector<GenericMatColumnDesc> materializedCols;

        for (size_t ti = 0; ti < project.projections.size(); ++ti) {
            const auto& projection = project.projections[ti];
            const std::string displayName = projection.name;
            const std::string bufferName = "d_ir_from_subquery_" + tag +
                "_" + std::to_string(ti) + "_" +
                sanitizeIdentifier(displayName);
            if (displayName == aggAlias) {
                materialize->addColumn(bufferName, "float", aggValueExpr,
                                       displayName, outputSize, 0);
                materializedCols.push_back(
                    {displayName, bufferName, "float", 0, 0, false});
                continue;
            }
            if (!projection.expr ||
                !materializeExprSupported(projection.expr)) {
                return fail(error,
                    "IR grouped FROM-view scalar lowerer target expression is not supported.");
            }
            TypeInfo type = projection.type;
            const int stringLen = fixedStringLenForExpr(projection.expr);
            std::string sizeExpr = outputSize;
            if (stringLen > 0)
                sizeExpr += " * " + std::to_string(stringLen);
            const std::string outType = metalTypeForType(type);
            const std::string valueExpr =
                materializeExprToMetal(projection.expr, idxVar);
            materialize->addColumn(bufferName, outType, valueExpr,
                                   displayName, sizeExpr, stringLen);
            materializedCols.push_back(
                {displayName, bufferName, outType, stringLen, 0, false});
        }

        auto& phase = appendPhase(
            plan, "GENERIC_ir_from_subquery_materialize_" + tag,
            std::move(materialize));
        phase.extraBuffers.push_back(
            {aggBuffer, "atomic_float", true, false});
        phase.extraBuffers.push_back(
            {extremumBuffer, "atomic_uint", true, false});
        phase.extraBuffers.push_back(
            {extremumState, "atomic_uint", true, false});

        GenericSortSpec sortSpec;
        sortSpec.limit = limitValue(shape->limit);
        if (auto* sort = sortDetail(shape->sort)) {
            for (const auto& key : sort->keys) {
                auto column = sortKeyDisplayName(key, project);
                if (!column) {
                    return fail(error,
                        "IR grouped FROM-view scalar lowerer: ORDER BY key is not projected.");
                }
                sortSpec.keys.push_back({*column, key.descending});
            }
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            const std::string sortRowsSym =
                "n_gpu_sort_ir_from_subquery_" + tag + "_rows";
            attachMaterializedCountHook(
                phase, "d_ir_from_subquery_" + tag + "_result_count",
                sortRowsSym);
            if (!appendBestGenericGpuOrder(plan, "ir_from_subquery_" + tag,
                                           sortRowsSym, outputSize,
                                           materializedCols, sortSpec, error)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}

std::optional<MetalQueryPlan> lowerFromSubqueryHistogramIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    const auto& source = ir.source;
    const auto* schema = ir.schema;
    if (source.fromSubqueryAggs.size() != 1) return std::nullopt;
    auto shape = parseFromSubqueryAggregateShape(ir);
    if (!shape || !shape->aggregate) return std::nullopt;
    const auto& outerAggregate = *shape->aggregate;
    if (outerAggregate.groupBy.size() != 1) return std::nullopt;

    const auto& fsq = source.fromSubqueryAggs[0];
    auto* outerGroupCol = outerAggregate.groupBy[0]
        ? std::get_if<GenericColumnExpr>(&outerAggregate.groupBy[0]->node)
        : nullptr;
    std::string innerAggAlias = !outerAggregate.groupNames.empty()
        ? outerAggregate.groupNames.front()
        : (outerGroupCol ? outerGroupCol->column : "");
    if (innerAggAlias.empty()) return std::nullopt;

    bool hasOuterGroupProjection =
        std::find(outerAggregate.outputOrder.begin(),
                  outerAggregate.outputOrder.end(),
                  innerAggAlias) != outerAggregate.outputOrder.end();
    if (outerAggregate.outputOrder.empty()) {
        hasOuterGroupProjection =
            std::find(outerAggregate.groupNames.begin(),
                      outerAggregate.groupNames.end(),
                      innerAggAlias) != outerAggregate.groupNames.end();
    }

    const GenericProjection* outerCount = nullptr;
    for (const auto& projection : outerAggregate.aggregates) {
        auto* agg = genericAggregateForProjection(projection);
        if (agg && agg->func == AggFunc::COUNT) {
            outerCount = &projection;
            break;
        }
    }
    if (!hasOuterGroupProjection || !outerCount) return std::nullopt;

    const GenericFromSubqueryAggTarget* innerAgg = nullptr;
    for (const auto& target : fsq.aggregates) {
        if (target.name == innerAggAlias) {
            innerAgg = &target;
            break;
        }
    }
    if (!innerAgg || innerAgg->func != AggFunc::COUNT) {
        return std::nullopt;
    }
    if (fsq.groupBy.size() != 1) return std::nullopt;
    auto* innerGroupCol = fsq.groupBy[0]
        ? std::get_if<GenericColumnExpr>(&fsq.groupBy[0]->node)
        : nullptr;
    if (!innerGroupCol) return std::nullopt;

    const GenericFromSubqueryJoin* leftOuterJoin = nullptr;
    for (const auto& jc : fsq.joins) {
        if (!jc.leftOuter) continue;
        if (fromSubqueryColMatches(fsq, *innerGroupCol,
                                   jc.leftTable, jc.leftCol)) {
            leftOuterJoin = &jc;
            break;
        }
    }
    if (!leftOuterJoin) return std::nullopt;

    const std::string groupBase =
        fromSubqueryBaseForKey(fsq, leftOuterJoin->leftTable);
    const std::string aggBase =
        fromSubqueryBaseForKey(fsq, leftOuterJoin->rightTable);
    const std::string groupJoinCol = leftOuterJoin->leftCol;
    const std::string aggJoinCol = leftOuterJoin->rightCol;
    if (groupBase.empty() || aggBase.empty() ||
        groupJoinCol.empty() || aggJoinCol.empty()) {
        return std::nullopt;
    }

    std::vector<GenericPredicatePtr> aggFilters;
    for (const auto& filter : fsq.filters) {
        std::set<std::string> filterTables;
        collectGenericPredicateTables(filter, filterTables);
        bool appliesToAgg = filterTables.empty();
        if (!filterTables.empty()) {
            appliesToAgg = std::all_of(
                filterTables.begin(), filterTables.end(),
                [&](const std::string& table) { return table == aggBase; });
        }
        if (!appliesToAgg) return std::nullopt;
        aggFilters.push_back(filter);
    }

    std::string countSize =
        keyDomainSymbolForAnalyzedColumn(groupBase, groupJoinCol, schema);
    if (countSize.empty()) {
        return fail(error, "IR FROM-subquery histogram lowerer: group key has no schema domain.");
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_FROM_SUBQUERY_HISTOGRAM";
    const std::string idxVar = "i";
    const std::string tag = sanitizeIdentifier(fsq.alias.empty()
        ? "from_subquery"
        : fsq.alias);
    const std::string countBuffer = "d_ir_from_subquery_" + tag + "_count";

    {
        std::set<std::string> scanCols{aggJoinCol};
        for (const auto& filter : aggFilters)
            collectGenericPredicateColumns(filter, scanCols);
        auto scan = makeScanForCols(aggBase, idxVar, scanCols, schema);
        auto filtered = maybeSelect(
            std::move(scan), combineGenericFilters(aggFilters, idxVar));
        auto count = std::make_unique<MetalAtomicCount>(
            std::move(filtered), countBuffer, aggJoinCol + "[" + idxVar + "]",
            countSize);
        appendPhase(plan, "GENERIC_ir_from_subquery_count_" + tag,
                    std::move(count));
    }

    {
        std::set<std::string> scanCols{groupJoinCol};
        auto scan = makeScanForCols(groupBase, idxVar, scanCols, schema);
        const std::string groupKeyExpr = groupJoinCol + "[" + idxVar + "]";
        const std::string outerCountName = outerCount->name;
        constexpr int kHistogramBucketCap = 65536;
        constexpr int kLocalHistogramBucketCap = 256;
        const std::string groupTag = "ir_from_subquery_hist_" + tag;
        const std::string histBuffer = "d_ir_from_subquery_" + tag + "_hist";
        auto hist = std::make_unique<MetalIrCountHistogram>(
            std::move(scan), countBuffer, histBuffer, groupKeyExpr,
            kHistogramBucketCap, kLocalHistogramBucketCap);
        auto& histPhase = appendPhase(
            plan, "GENERIC_ir_from_subquery_histogram_" + tag,
            std::move(hist));
        (void)histPhase;

        const std::string compactCounter =
            "d_ir_from_subquery_" + tag + "_hist_result_count";
        std::vector<KeyedCompactKeySpec> compactKeys = {
            {innerAggAlias, kHistogramBucketCap, {}, 1, {}, 0, {}, 0}
        };
        std::vector<KeyedCompactAggSpec> compactAggs;
        KeyedCompactAggSpec countOut;
        countOut.displayName = outerCountName;
        countOut.offset = 0;
        compactAggs.push_back(countOut);

        std::vector<GenericMatColumnDesc> compactCols;
        const std::string countCol = "d_ir_from_subquery_" + tag + "_0_" +
            sanitizeIdentifier(innerAggAlias);
        const std::string outerCountCol = "d_ir_from_subquery_" + tag + "_1_" +
            sanitizeIdentifier(outerCountName);
        compactCols.push_back({innerAggAlias, countCol, "int", 0, 0, false});
        compactCols.push_back({outerCountName, outerCountCol, "uint", 0, 0, false});

        auto& compactPhase = appendPhase(
            plan, "GENERIC_ir_from_subquery_histogram_compact_" + tag,
            makeKeyedAggCompactOperator(
                histBuffer, compactCounter, kHistogramBucketCap, 1,
                compactKeys, compactAggs, compactCols));
        const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
        attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);

        GenericSortSpec sortSpec;
        sortSpec.limit = limitValue(shape->limit);
        if (auto* sort = sortDetail(shape->sort)) {
            IrGroupKeyDesc groupKey;
            groupKey.displayName = innerAggAlias;
            std::vector<IrGroupKeyDesc> groupKeys{groupKey};
            for (const auto& key : sort->keys) {
                auto column = sortKeyDisplayNameForGroupedAgg(
                    key, outerAggregate, groupKeys);
                if (!column) {
                    return fail(error, "IR FROM-subquery histogram lowerer: ORDER BY key is not projected.");
                }
                sortSpec.keys.push_back({*column, key.descending});
            }
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            if (!appendBestGenericGpuOrder(plan, "group_" + groupTag,
                                           sortRowsSym,
                                           std::to_string(kHistogramBucketCap),
                                           compactCols,
                                           sortSpec, error)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}



} // namespace

std::optional<MetalQueryPlan> lowerFromSubqueryAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    if (auto p = lowerFromSubqueryHistogramIRToMetal(ir, error))
        return p;
    return lowerFromSubqueryTopScalarIRToMetal(ir, error);
}

} // namespace codegen
