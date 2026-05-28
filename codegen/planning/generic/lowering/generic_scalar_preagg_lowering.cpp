#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "generic/lowering/generic_cost_model.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_scalar_preagg_ops.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "execution/metal_generic_executor.h"
#include "metal_plan_common.h"
#include "scalar_subquery_placeholder.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>


namespace codegen {

namespace {

using ScalarLookupInfo = GenericScalarLookupInfo;

static float scalarFloatFromBits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

static void attachGlobalScalarValueHook(MetalQueryPlan::Phase& phase,
                                        ScalarLookupInfo info) {
    phase.postDispatchHook = [info = std::move(info)](
            MetalGenericExecutor& executor) {
        float value = std::numeric_limits<float>::quiet_NaN();
        auto readFloat = [&](const std::string& bufferName) -> std::optional<float> {
            auto* buffer = executor.getAllocatedBuffer(bufferName);
            if (!buffer) return std::nullopt;
            return *static_cast<const float*>(buffer->contents());
        };
        auto readUint = [&](const std::string& bufferName) -> std::optional<uint32_t> {
            auto* buffer = executor.getAllocatedBuffer(bufferName);
            if (!buffer) return std::nullopt;
            return *static_cast<const uint32_t*>(buffer->contents());
        };

        switch (info.kind) {
            case ScalarLookupInfo::GlobalCount:
                if (auto count = readUint(info.countBuffer))
                    value = info.multiplier * static_cast<float>(*count);
                break;
            case ScalarLookupInfo::GlobalSum:
                if (auto sum = readFloat(info.sumBuffer))
                    value = info.multiplier * *sum;
                break;
            case ScalarLookupInfo::GlobalAvg: {
                auto sum = readFloat(info.sumBuffer);
                auto count = readUint(info.countBuffer);
                if (sum && count && *count > 0)
                    value = info.multiplier * *sum / static_cast<float>(*count);
                break;
            }
            case ScalarLookupInfo::GlobalMin:
                if (auto seen = readUint(info.stateBuffer); seen && *seen != 0u) {
                    if (auto bits = readUint(info.minBuffer))
                        value = info.multiplier * scalarFloatFromBits(*bits);
                }
                break;
            case ScalarLookupInfo::GlobalMax:
                if (auto seen = readUint(info.stateBuffer); seen && *seen != 0u) {
                    if (auto bits = readUint(info.maxBuffer))
                        value = info.multiplier * scalarFloatFromBits(*bits);
                }
                break;
            default:
                break;
        }
        executor.registerScalarFloat(info.scalarName, value);
        return 0.0;
    };
}

static std::string maxKeySymbolForColumn(const std::string& table,
                                         const std::string& col,
                                         const SchemaProvider* schema) {
    if (schema) {
        auto keySym = schema->keyDomainSymbol(table, col);
        if (!keySym.empty()) return keySym;
        if (auto gd = schema->groupDomain(table, col))
            return std::to_string(gd->maxValue + 1);
        auto pk = schema->pkInfo(table);
        if (pk && pk->first == col) return pk->second;
        auto tableSym = schema->maxKeySymbol(table);
        if (!tableSym.empty()) return tableSym;
    }
    return "";
}

static size_t scalarCostColumnWidth(const std::string& table,
                                    const std::string& col,
                                    const SchemaProvider* schema) {
    if (!schema || !schema->hasColumn(table, col)) return 4;
    TypeInfo type{schema->columnType(table, col),
                  schema->columnFixedWidth(table, col)};
    return genericCostTypeByteWidth(type);
}

static std::string scalarCostTableRowsExpr(const std::string& table,
                                           const SchemaProvider* schema) {
    if (schema) {
        size_t rows = schema->tableRowCount(table);
        if (rows > 0) return std::to_string(rows);
    }
    return tableSizeName(table);
}

struct DecorrelatedBitmapState {
    std::string table;
    std::string column;
    std::string bitmap;
    bool externalToTable = false;
};

struct ResolvedCorrelation {
    DecorrCol inner;
    DecorrCol outer;
};

struct OuterFilterBinding {
    std::string table;
    std::string alias;
};

struct OuterRelation {
    std::string node;
    std::string table;
    std::string alias;
    int relationInstance = -1;
};

struct OuterKeysetState {
    std::string node;
    std::string table;
    std::string column;
    std::string bitmap;
    double estimatedActiveKeyFraction = 0.5;
    int propagationDepth = 0;
};

struct ValueExternalProbe {
    std::string bitmap;
    std::string keyColumn;
};

static std::vector<OuterRelation> outerRelationsForIr(const GenericRelPlan& ir) {
    std::vector<OuterRelation> out;
    for (const auto& inst : ir.relationInstances) {
        std::string alias = inst.alias.empty() ? inst.baseName : inst.alias;
        std::string node = alias.empty() ? inst.baseName : alias;
        out.push_back({std::move(node), inst.baseName, std::move(alias),
                       inst.id.value});
    }
    return out;
}

static const OuterRelation* outerRelationByInstance(
        const std::vector<OuterRelation>& relations,
        int relationInstance) {
    for (const auto& relation : relations) {
        if (relation.relationInstance == relationInstance)
            return &relation;
    }
    return nullptr;
}

static std::optional<OuterFilterBinding> resolveOuterFilterBinding(
        const DecorrCol& outer,
        const std::vector<OuterRelation>& relations,
        const SchemaProvider* schema) {
    std::vector<OuterFilterBinding> matches;
    auto addMatch = [&](std::string table, std::string alias) {
        if (!schema || !schema->hasColumn(table, outer.column)) return;
        for (const auto& existing : matches) {
            if (existing.table == table && existing.alias == alias) return;
        }
        matches.push_back({std::move(table), std::move(alias)});
    };

    if (!outer.table.empty()) {
        for (const auto& relation : relations) {
            if (relation.alias == outer.table || relation.table == outer.table)
                addMatch(relation.table, relation.alias);
        }
        if (matches.empty()) addMatch(outer.table, "");
    } else {
        for (const auto& relation : relations)
            addMatch(relation.table,
                     relation.alias == relation.table ? "" : relation.alias);
    }

    if (matches.size() != 1) return std::nullopt;
    return matches.front();
}

static void collectGenericExprRelationRefs(const GenericExprPtr& expr,
                                           std::set<int>& relationInstances,
                                           std::set<std::string>& tables);

static bool genericExprHasScalarSubqueryRef(const GenericExprPtr& expr,
                                            int sqIdx);

static bool genericPredHasScalarSubqueryRef(const GenericPredicatePtr& pred,
                                            int sqIdx) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return genericExprHasScalarSubqueryRef(node.left, sqIdx) ||
                   genericExprHasScalarSubqueryRef(node.right, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return genericExprHasScalarSubqueryRef(node.expr, sqIdx) ||
                   genericExprHasScalarSubqueryRef(node.low, sqIdx) ||
                   genericExprHasScalarSubqueryRef(node.high, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (genericExprHasScalarSubqueryRef(node.expr, sqIdx)) return true;
            for (const auto& value : node.values) {
                if (genericExprHasScalarSubqueryRef(value, sqIdx)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return genericExprHasScalarSubqueryRef(node.expr, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (genericPredHasScalarSubqueryRef(child, sqIdx)) return true;
            }
            return false;
        } else {
            return false;
        }
    }, pred->node);
}

static bool genericExprHasScalarSubqueryRef(const GenericExprPtr& expr,
                                            int sqIdx) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericScalarSubqueryExpr>) {
            return node.index == sqIdx;
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return genericExprHasScalarSubqueryRef(node.left, sqIdx) ||
                   genericExprHasScalarSubqueryRef(node.right, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (genericPredHasScalarSubqueryRef(branch.condition, sqIdx) ||
                    genericExprHasScalarSubqueryRef(branch.result, sqIdx)) {
                    return true;
                }
            }
            return genericExprHasScalarSubqueryRef(node.elseResult, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (genericExprHasScalarSubqueryRef(arg, sqIdx)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return genericExprHasScalarSubqueryRef(node.arg, sqIdx);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys) {
                if (genericExprHasScalarSubqueryRef(key, sqIdx)) return true;
            }
            return false;
        } else {
            return false;
        }
    }, expr->node);
}

static void collectGenericPredRelationRefs(const GenericPredicatePtr& pred,
                                           std::set<int>& relationInstances,
                                           std::set<std::string>& tables) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectGenericExprRelationRefs(node.left, relationInstances, tables);
            collectGenericExprRelationRefs(node.right, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectGenericExprRelationRefs(node.expr, relationInstances, tables);
            collectGenericExprRelationRefs(node.low, relationInstances, tables);
            collectGenericExprRelationRefs(node.high, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectGenericExprRelationRefs(node.expr, relationInstances, tables);
            for (const auto& value : node.values)
                collectGenericExprRelationRefs(value, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectGenericExprRelationRefs(node.expr, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectGenericPredRelationRefs(child, relationInstances, tables);
        }
    }, pred->node);
}

static void collectGenericExprRelationRefs(const GenericExprPtr& expr,
                                           std::set<int>& relationInstances,
                                           std::set<std::string>& tables) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (node.relationInstance.valid())
                relationInstances.insert(node.relationInstance.value);
            if (!node.table.empty())
                tables.insert(node.table);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectGenericExprRelationRefs(node.left, relationInstances, tables);
            collectGenericExprRelationRefs(node.right, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectGenericPredRelationRefs(branch.condition, relationInstances, tables);
                collectGenericExprRelationRefs(branch.result, relationInstances, tables);
            }
            collectGenericExprRelationRefs(node.elseResult, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectGenericExprRelationRefs(arg, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectGenericExprRelationRefs(node.arg, relationInstances, tables);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectGenericExprRelationRefs(key, relationInstances, tables);
        }
    }, expr->node);
}

static bool genericPredicateOnlyReferencesBinding(
        const GenericPredicatePtr& pred,
        const OuterFilterBinding& binding,
        const std::vector<OuterRelation>& relations) {
    std::set<int> instances;
    std::set<std::string> tables;
    collectGenericPredRelationRefs(pred, instances, tables);
    if (instances.size() == 1) {
        auto* relation = outerRelationByInstance(relations, *instances.begin());
        if (!relation || relation->table != binding.table) return false;
        return binding.alias.empty() ||
               relation->alias == binding.alias ||
               relation->node == binding.alias;
    }
    return instances.empty() &&
           tables.size() == 1 &&
           *tables.begin() == binding.table;
}

static void collectGenericConjuncts(const GenericPredicatePtr& pred,
                                    std::vector<GenericPredicatePtr>& out) {
    if (!pred) return;
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node);
        logical && logical->op == GenericLogicalPred::Op::And) {
        for (const auto& child : logical->children)
            collectGenericConjuncts(child, out);
        return;
    }
    out.push_back(pred);
}

static std::vector<GenericPredicatePtr> outerFilterPredicatesForIr(
        const GenericRelPlan& ir) {
    std::vector<GenericPredicatePtr> out;
    for (const auto& node : ir.nodes) {
        if (node.op != GenericRelOp::Filter) continue;
        auto* filter = std::get_if<GenericFilterDetail>(&node.detail);
        if (!filter) continue;
        collectGenericConjuncts(filter->predicate, out);
    }
    return out;
}

static std::string combineGenericFilters(
        const std::vector<GenericPredicatePtr>& filters,
        const std::string& idxVar) {
    if (filters.empty()) return "";
    if (filters.size() == 1)
        return genericPredicateToMetal(filters.front(), idxVar);

    std::string cond;
    const char* sep = filters.size() > 2 ? "\n               && " : " && ";
    for (size_t i = 0; i < filters.size(); ++i) {
        if (i) cond += sep;
        cond += "(" + genericPredicateToMetal(filters[i], idxVar) + ")";
    }
    return cond;
}

static std::optional<std::string> outerNodeForPredicate(
        const GenericPredicatePtr& pred,
        const std::vector<OuterRelation>& relations,
        const std::map<std::string, std::string>& uniqueOuterNodeForBase) {
    std::set<int> instances;
    std::set<std::string> tables;
    collectGenericPredRelationRefs(pred, instances, tables);
    if (instances.size() == 1) {
        if (auto* relation = outerRelationByInstance(relations, *instances.begin()))
            return relation->node;
    }
    if (instances.empty() && tables.size() == 1) {
        auto uniqueIt = uniqueOuterNodeForBase.find(*tables.begin());
        if (uniqueIt != uniqueOuterNodeForBase.end())
            return uniqueIt->second;
    }
    return std::nullopt;
}

static std::string outerNodeForColumn(
        const GenericColumnExpr& col,
        const std::vector<OuterRelation>& relations,
        const std::map<std::string, std::string>& uniqueOuterNodeForBase,
        const std::map<std::string, std::string>& outerNodeTable) {
    if (col.relationInstance.valid()) {
        if (auto* relation = outerRelationByInstance(relations,
                                                     col.relationInstance.value)) {
            return relation->node;
        }
    }
    if (!col.alias.empty() && outerNodeTable.count(col.alias))
        return col.alias;
    if (!col.table.empty()) {
        auto uniqueIt = uniqueOuterNodeForBase.find(col.table);
        if (uniqueIt != uniqueOuterNodeForBase.end())
            return uniqueIt->second;
    }
    return {};
}

static void collectJoinEqualityColumns(
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
                collectJoinEqualityColumns(child, out);
        }
    }, pred->node);
}

static std::vector<std::pair<GenericColumnExpr, GenericColumnExpr>>
outerJoinEqualitiesForIr(const GenericRelPlan& ir) {
    std::vector<std::pair<GenericColumnExpr, GenericColumnExpr>> out;
    for (const auto& node : ir.nodes) {
        if (node.op != GenericRelOp::Join &&
            node.op != GenericRelOp::SemiJoin &&
            node.op != GenericRelOp::AntiJoin) {
            continue;
        }
        auto* join = std::get_if<GenericJoinDetail>(&node.detail);
        if (!join) continue;
        if (join->kind == GenericJoinKind::Anti ||
            join->kind == GenericJoinKind::LeftOuter) {
            continue;
        }
        collectJoinEqualityColumns(join->predicate, out);
    }
    return out;
}

static std::optional<ScalarLookupInfo> buildDecorrelatedScalarPreAgg(
        const DecorrelatedScalarSubquery& dsq,
        const GenericRelPlan& ir,
        const SchemaProvider* schema,
        MetalQueryPlan& plan) {
    const std::string idxVar = "i";
    const std::vector<OuterRelation> outerRelations = outerRelationsForIr(ir);
    const std::vector<GenericPredicatePtr> outerFilters =
        outerFilterPredicatesForIr(ir);
    std::map<std::string, std::set<std::string>> relevantCols;
    for (const auto& j : dsq.joins) {
        relevantCols[j.left.table].insert(j.left.column);
        relevantCols[j.right.table].insert(j.right.column);
    }
    for (const auto& c : dsq.correlations)
        relevantCols[c.inner.table].insert(c.inner.column);

    std::map<std::string, std::vector<GenericPredicatePtr>> filtersByTable =
        dsq.filtersByTable;
    std::map<std::string, std::vector<GenericPredicatePtr>> genericFiltersByTable;
    std::vector<ResolvedCorrelation> resolvedCorrelations;
    std::vector<std::pair<OuterFilterBinding, DecorrCol>> outerBindingCols;
    std::set<const GenericPredicate*> copiedOuterFilters;
    std::set<std::pair<std::string, std::string>> outerFilterSeedCols;
    // Bitmap state is one path per column; keep outer-filter seeding to
    // correlation-only subqueries so inner join constraints stay intact.
    const bool canSeedOuterFilters = dsq.joins.empty();
    for (const auto& c : dsq.correlations) {
        DecorrCol outer = c.outer;
        if (auto binding = resolveOuterFilterBinding(c.outer, outerRelations, schema)) {
            outer.table = binding->table;
            relevantCols[outer.table].insert(outer.column);
            outerBindingCols.push_back({*binding, outer});

            if (canSeedOuterFilters) {
                bool hasOuterFilterForBinding = false;
                for (const auto& pred : outerFilters) {
                    if (!pred) continue;
                    if (genericPredHasScalarSubqueryRef(pred, dsq.sqIdx)) continue;
                    if (!genericPredicateOnlyReferencesBinding(pred, *binding,
                                                               outerRelations)) {
                        continue;
                    }
                    hasOuterFilterForBinding = true;
                    if (!copiedOuterFilters.count(pred.get())) {
                        genericFiltersByTable[binding->table].push_back(pred);
                        copiedOuterFilters.insert(pred.get());
                    }
                }
                if (hasOuterFilterForBinding)
                    outerFilterSeedCols.insert({outer.table, outer.column});
            }
        }
        if (!outer.table.empty())
            resolvedCorrelations.push_back({c.inner, outer});
    }

    auto filterCondFor = [&](const std::string& table) {
        auto it = filtersByTable.find(table);
        std::string filterCond = it == filtersByTable.end()
            ? std::string()
            : combineGenericFilters(it->second, idxVar);
        auto genericIt = genericFiltersByTable.find(table);
        if (genericIt != genericFiltersByTable.end()) {
            std::string seedCond = combineGenericFilters(genericIt->second, idxVar);
            if (!seedCond.empty()) {
                filterCond = filterCond.empty()
                    ? std::move(seedCond)
                    : "(" + filterCond + ") && (" + seedCond + ")";
            }
        }
        return filterCond.empty() ? std::string("true") : filterCond;
    };
    auto makeFilteredScan = [&](const std::string& table) -> std::unique_ptr<MetalOperator> {
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(table, idxVar);
        return maybeSelect(std::move(pipe), filterCondFor(table));
    };

    std::map<std::pair<std::string, std::string>, DecorrelatedBitmapState> states;
    auto addState = [&](const std::string& table, const std::string& col,
                        DecorrelatedBitmapState state) {
        states[{table, col}] = std::move(state);
    };
    auto hasState = [&](const std::string& table, const std::string& col) {
        return states.count({table, col}) != 0;
    };
    auto stateName = [&](const std::string& table, const std::string& col,
                         const std::string& suffix) {
        return "d_scalar_" + std::to_string(dsq.sqIdx) + "_" +
               sanitizeIdentifier(table + "_" + col + "_" + suffix) + "_bmp";
    };
    auto stateFeedsInnerJoin = [&](const std::string& table,
                                   const std::string& col) {
        for (const auto& j : dsq.joins) {
            if ((j.left.table == table && j.left.column == col) ||
                (j.right.table == table && j.right.column == col)) {
                return true;
            }
        }
        return false;
    };
    auto keyDomainsMatch = [&](const std::string& leftTable,
                               const std::string& leftCol,
                               const std::string& rightTable,
                               const std::string& rightCol) {
        std::string leftDomain = maxKeySymbolForColumn(leftTable, leftCol, schema);
        std::string rightDomain = maxKeySymbolForColumn(rightTable, rightCol, schema);
        return !leftDomain.empty() && leftDomain == rightDomain;
    };
    auto canDirectProbeValueTableFromOuterSeed =
            [&](const std::string& table, const std::string& col) {
        for (const auto& c : resolvedCorrelations) {
            if (c.outer.table == table && c.outer.column == col &&
                c.inner.table == dsq.valueTable &&
                !stateFeedsInnerJoin(c.inner.table, c.inner.column) &&
                keyDomainsMatch(table, col, c.inner.table, c.inner.column)) {
                return true;
            }
        }
        return false;
    };

    std::map<std::string, std::string> outerNodeTable;
    std::map<std::string, int> outerBaseCounts;
    std::map<std::string, std::string> uniqueOuterNodeForBase;
    for (const auto& relation : outerRelations) {
        outerNodeTable[relation.node] = relation.table;
        outerBaseCounts[relation.table]++;
    }
    for (const auto& [node, table] : outerNodeTable) {
        if (outerBaseCounts[table] == 1)
            uniqueOuterNodeForBase[table] = node;
    }
    auto resolveOuterNode = [&](const std::string& name) -> std::string {
        if (outerNodeTable.count(name)) return name;
        auto uniqueIt = uniqueOuterNodeForBase.find(name);
        if (uniqueIt != uniqueOuterNodeForBase.end()) return uniqueIt->second;
        return {};
    };

    struct OuterJoinEdge {
        std::string leftNode;
        std::string leftTable;
        std::string leftCol;
        std::string rightNode;
        std::string rightTable;
        std::string rightCol;
    };
    std::vector<OuterJoinEdge> outerJoinEdges;
    std::map<std::string, std::set<std::string>> outerRelevantCols;
    for (const auto& [leftCol, rightCol] : outerJoinEqualitiesForIr(ir)) {
        std::string leftNode = outerNodeForColumn(
            leftCol, outerRelations, uniqueOuterNodeForBase, outerNodeTable);
        std::string rightNode = outerNodeForColumn(
            rightCol, outerRelations, uniqueOuterNodeForBase, outerNodeTable);
        if (leftNode.empty() || rightNode.empty()) continue;
        const std::string& leftTable = outerNodeTable[leftNode];
        const std::string& rightTable = outerNodeTable[rightNode];
        if (outerBaseCounts[leftTable] != 1 || outerBaseCounts[rightTable] != 1)
            continue;
        outerJoinEdges.push_back({leftNode, leftTable, leftCol.column,
                                  rightNode, rightTable, rightCol.column});
        outerRelevantCols[leftNode].insert(leftCol.column);
        outerRelevantCols[rightNode].insert(rightCol.column);
    }

    std::set<std::string> outerCorrelationNodes;
    for (const auto& [binding, outerCol] : outerBindingCols) {
        std::string node = !binding.alias.empty()
            ? resolveOuterNode(binding.alias)
            : resolveOuterNode(binding.table);
        if (node.empty()) continue;
        outerCorrelationNodes.insert(node);
        outerRelevantCols[node].insert(outerCol.column);
    }

    std::map<std::string, std::vector<GenericPredicatePtr>> outerFiltersByNode;
    for (const auto& pred : outerFilters) {
        if (!pred) continue;
        if (genericPredHasScalarSubqueryRef(pred, dsq.sqIdx)) continue;
        auto node = outerNodeForPredicate(pred, outerRelations,
                                          uniqueOuterNodeForBase);
        if (!node) continue;
        const std::string& table = outerNodeTable[*node];
        if (outerBaseCounts[table] != 1) continue;
        outerFiltersByNode[*node].push_back(pred);
    }

    auto outerFilterCondFor = [&](const std::string& node) {
        auto it = outerFiltersByNode.find(node);
        if (it == outerFiltersByNode.end()) return std::string("true");
        return combineGenericFilters(it->second, idxVar);
    };
    auto makeOuterFilteredScan = [&](const std::string& node) -> std::unique_ptr<MetalOperator> {
        auto tableIt = outerNodeTable.find(node);
        if (tableIt == outerNodeTable.end()) return nullptr;
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(tableIt->second, idxVar);
        return maybeSelect(std::move(pipe), outerFilterCondFor(node));
    };
    std::map<std::pair<std::string, std::string>, OuterKeysetState> outerStates;
    auto hasOuterState = [&](const std::string& node, const std::string& col) {
        return outerStates.count({node, col}) != 0;
    };
    auto addOuterState = [&](std::string node,
                             std::string table,
                             std::string col,
                             std::string bitmap,
                             double estimatedActiveKeyFraction,
                             int propagationDepth) {
        std::pair<std::string, std::string> key{node, col};
        if (outerStates.count(key)) return false;
        outerStates[key] = {node, table, col, bitmap,
                            estimatedActiveKeyFraction, propagationDepth};
        auto relevantIt = relevantCols.find(table);
        if (relevantIt != relevantCols.end() && relevantIt->second.count(col) &&
            !hasState(table, col)) {
            addState(table, col, {table, col, bitmap, true});
        }
        return true;
    };

    auto outerKeysetTargetWidth = [&](const std::string& node,
                                      const std::string& keyCol) {
        auto tableIt = outerNodeTable.find(node);
        if (tableIt == outerNodeTable.end()) return size_t{64};
        const std::string& table = tableIt->second;
        size_t width = scalarCostColumnWidth(table, keyCol, schema);
        auto colsIt = outerRelevantCols.find(node);
        if (colsIt != outerRelevantCols.end()) {
            for (const auto& col : colsIt->second)
                width += scalarCostColumnWidth(table, col, schema);
        }
        return std::max<size_t>(width, 64);
    };
    auto chooseOuterKeyset = [&](const std::string& tag,
                                 const OuterKeysetState& source,
                                 const std::string& targetNode,
                                 const std::string& targetTable,
                                 const std::string& targetCol,
                                 double activeFraction,
                                 int propagationDepth) {
        KeysetPropagationCostInput input;
        input.tag = tag;
        input.buildRowsExpr = scalarCostTableRowsExpr(targetTable, schema);
        input.targetRowsExpr = input.buildRowsExpr;
        input.keyDomainExpr = maxKeySymbolForColumn(targetTable, targetCol,
                                                    schema);
        input.keyByteWidth =
            scalarCostColumnWidth(targetTable, targetCol, schema);
        input.targetRowByteWidth =
            outerKeysetTargetWidth(targetNode, targetCol);
        input.estimatedActiveKeyFraction = activeFraction;
        input.propagationDepth = propagationDepth;
        input.hasSourceBitmap = true;
        auto choice = chooseKeysetPropagation(input);
        appendGenericCostDecisionTrace(plan, choice.trace);
        (void)source;
        return choice;
    };

    for (const auto& [node, filters] : outerFiltersByNode) {
        if (outerCorrelationNodes.count(node)) continue;
        auto colsIt = outerRelevantCols.find(node);
        if (colsIt == outerRelevantCols.end()) continue;
        const std::string& table = outerNodeTable[node];
        for (const auto& col : colsIt->second) {
            std::string bitmap = stateName(table, col, "outer_seed");
            auto pipe = makeOuterFilteredScan(node);
            if (!pipe) continue;
            auto build = std::make_unique<MetalBitmapBuild>(
                std::move(pipe), bitmap, col + "[" + idxVar + "]",
                maxKeySymbolForColumn(table, col, schema));
            appendPhase(plan, "GENERIC_scalar_outer_keyset_" +
                              std::to_string(dsq.sqIdx) + "_" +
                              sanitizeIdentifier(node + "_" + col + "_seed"),
                        std::move(build));
            addOuterState(node, table, col, bitmap, 0.25, 0);
        }
    }

    bool outerChanged = true;
    int outerGuard = 0;
    while (outerChanged && outerGuard++ < 64) {
        outerChanged = false;
        std::vector<OuterKeysetState> snapshot;
        for (const auto& [_, state] : outerStates) snapshot.push_back(state);
        for (const auto& st : snapshot) {
            auto colsIt = outerRelevantCols.find(st.node);
            if (colsIt != outerRelevantCols.end()) {
                for (const auto& outCol : colsIt->second) {
                    if (outCol == st.column || hasOuterState(st.node, outCol)) continue;
                    std::string bitmap = stateName(st.table, outCol, "outer_xfer");
                    auto pipe = makeOuterFilteredScan(st.node);
                    if (!pipe) continue;
                    const double activeFraction =
                        std::min(0.95, std::max(0.05,
                            st.estimatedActiveKeyFraction));
                    const int propagationDepth = st.propagationDepth + 1;
                    auto choice = chooseOuterKeyset(
                        "scalar_outer_xfer_" +
                            sanitizeIdentifier(st.node + "_" + outCol),
                        st, st.node, st.table, outCol, activeFraction,
                        propagationDepth);
                    if (!choice.useKeyset) continue;
                    pipe = std::make_unique<MetalBitmapProbe>(
                        std::move(pipe), st.bitmap, st.column + "[" + idxVar + "]");
                    auto build = std::make_unique<MetalBitmapBuild>(
                        std::move(pipe), bitmap, outCol + "[" + idxVar + "]",
                        maxKeySymbolForColumn(st.table, outCol, schema));
                    appendPhase(plan, "GENERIC_scalar_outer_keyset_" +
                                      std::to_string(dsq.sqIdx) + "_" +
                                      sanitizeIdentifier(st.node + "_" + outCol + "_xfer"),
                                std::move(build));
                    outerChanged =
                        addOuterState(st.node, st.table, outCol, bitmap,
                                      activeFraction, propagationDepth) ||
                        outerChanged;
                }
            }

            for (const auto& edge : outerJoinEdges) {
                const bool fromLeft =
                    edge.leftNode == st.node && edge.leftCol == st.column;
                const bool fromRight =
                    edge.rightNode == st.node && edge.rightCol == st.column;
                if (!fromLeft && !fromRight) continue;
                const std::string& dstNode = fromLeft ? edge.rightNode : edge.leftNode;
                const std::string& dstTable = fromLeft ? edge.rightTable : edge.leftTable;
                const std::string& dstCol = fromLeft ? edge.rightCol : edge.leftCol;
                if (hasOuterState(dstNode, dstCol)) continue;
                std::string bitmap = stateName(dstTable, dstCol, "outer_join");
                auto pipe = makeOuterFilteredScan(dstNode);
                if (!pipe) continue;
                const double activeFraction =
                    std::min(0.95, std::max(0.05,
                        st.estimatedActiveKeyFraction));
                const int propagationDepth = st.propagationDepth + 1;
                auto choice = chooseOuterKeyset(
                    "scalar_outer_join_" +
                        sanitizeIdentifier(dstNode + "_" + dstCol),
                    st, dstNode, dstTable, dstCol, activeFraction,
                    propagationDepth);
                if (!choice.useKeyset) continue;
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, dstCol + "[" + idxVar + "]");
                auto build = std::make_unique<MetalBitmapBuild>(
                    std::move(pipe), bitmap, dstCol + "[" + idxVar + "]",
                    maxKeySymbolForColumn(dstTable, dstCol, schema));
                appendPhase(plan, "GENERIC_scalar_outer_keyset_" +
                                  std::to_string(dsq.sqIdx) + "_" +
                                  sanitizeIdentifier(dstNode + "_" + dstCol + "_join"),
                            std::move(build));
                outerChanged =
                    addOuterState(dstNode, dstTable, dstCol, bitmap,
                                  activeFraction, propagationDepth) ||
                    outerChanged;
            }
        }
    }

    for (const auto& [table, filters] : filtersByTable) {
        auto colsIt = relevantCols.find(table);
        if (colsIt == relevantCols.end()) continue;
        for (const auto& col : colsIt->second) {
            if (hasState(table, col)) continue;
            if (table == dsq.valueTable && !stateFeedsInnerJoin(table, col))
                continue;
            std::string bitmap = stateName(table, col, "seed");
            auto pipe = makeFilteredScan(table);
            auto build = std::make_unique<MetalBitmapBuild>(
                std::move(pipe), bitmap, col + "[" + idxVar + "]",
                maxKeySymbolForColumn(table, col, schema));
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              "_" + sanitizeIdentifier(table + "_" + col + "_seed"),
                        std::move(build));
            const bool externalSeed =
                outerFilterSeedCols.count({table, col}) != 0 &&
                canDirectProbeValueTableFromOuterSeed(table, col);
            addState(table, col, {table, col, bitmap, externalSeed});
        }
    }

    bool changed = true;
    int guard = 0;
    while (changed && guard++ < 64) {
        changed = false;
        std::vector<DecorrelatedBitmapState> snapshot;
        for (const auto& [_, state] : states) snapshot.push_back(state);

        for (const auto& st : snapshot) {
            auto colsIt = relevantCols.find(st.table);
            if (colsIt != relevantCols.end()) {
                for (const auto& outCol : colsIt->second) {
                    if (outCol == st.column || hasState(st.table, outCol)) continue;
                    std::string bitmap = stateName(st.table, outCol, "xfer");
                    std::unique_ptr<MetalOperator> pipe = makeAutoScan(st.table, idxVar);
                    pipe = std::make_unique<MetalBitmapProbe>(
                        std::move(pipe), st.bitmap, st.column + "[" + idxVar + "]");
                    pipe = maybeSelect(std::move(pipe), filterCondFor(st.table));
                    auto build = std::make_unique<MetalBitmapBuild>(
                        std::move(pipe), bitmap, outCol + "[" + idxVar + "]",
                        maxKeySymbolForColumn(st.table, outCol, schema));
                    appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                                      "_" + sanitizeIdentifier(st.table + "_" + outCol + "_xfer"),
                                std::move(build));
                    addState(st.table, outCol, {st.table, outCol, bitmap, st.externalToTable});
                    changed = true;
                }
            }

            for (const auto& j : dsq.joins) {
                const DecorrCol* src = nullptr;
                const DecorrCol* dst = nullptr;
                if (j.left.table == st.table && j.left.column == st.column) {
                    src = &j.left; dst = &j.right;
                } else if (j.right.table == st.table && j.right.column == st.column) {
                    src = &j.right; dst = &j.left;
                }
                if (!src || !dst || hasState(dst->table, dst->column)) continue;
                std::string bitmap = stateName(dst->table, dst->column, "join");
                std::unique_ptr<MetalOperator> pipe = makeAutoScan(dst->table, idxVar);
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, dst->column + "[" + idxVar + "]");
                pipe = maybeSelect(std::move(pipe), filterCondFor(dst->table));
                auto build = std::make_unique<MetalBitmapBuild>(
                    std::move(pipe), bitmap, dst->column + "[" + idxVar + "]",
                    maxKeySymbolForColumn(dst->table, dst->column, schema));
                appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                                  "_" + sanitizeIdentifier(dst->table + "_" + dst->column + "_join"),
                            std::move(build));
                addState(dst->table, dst->column, {dst->table, dst->column, bitmap, true});
                changed = true;
            }

            for (const auto& c : resolvedCorrelations) {
                const DecorrCol* dst = nullptr;
                if (c.inner.table == st.table && c.inner.column == st.column) {
                    dst = &c.outer;
                } else if (c.outer.table == st.table && c.outer.column == st.column) {
                    dst = &c.inner;
                }
                if (!dst || dst->table.empty() || hasState(dst->table, dst->column)) continue;
                const bool directValueProbeAvailable =
                    st.externalToTable &&
                    c.outer.table == st.table &&
                    c.outer.column == st.column &&
                    c.inner.table == dsq.valueTable &&
                    !stateFeedsInnerJoin(c.inner.table, c.inner.column) &&
                    keyDomainsMatch(st.table, st.column,
                                    c.inner.table, c.inner.column);
                if (directValueProbeAvailable) continue;
                std::string bitmap = stateName(dst->table, dst->column, "corr");
                std::unique_ptr<MetalOperator> pipe = makeAutoScan(dst->table, idxVar);
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, dst->column + "[" + idxVar + "]");
                pipe = maybeSelect(std::move(pipe), filterCondFor(dst->table));
                auto build = std::make_unique<MetalBitmapBuild>(
                    std::move(pipe), bitmap, dst->column + "[" + idxVar + "]",
                    maxKeySymbolForColumn(dst->table, dst->column, schema));
                appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                                  "_" + sanitizeIdentifier(dst->table + "_" + dst->column + "_corr"),
                            std::move(build));
                addState(dst->table, dst->column, {dst->table, dst->column, bitmap, true});
                changed = true;
            }
        }
    }

    std::vector<ValueExternalProbe> valueExternalProbes;
    std::set<std::pair<std::string, std::string>> valueExternalProbeKeys;
    auto addValueExternalProbe = [&](const std::string& bitmap,
                                     const std::string& keyColumn) {
        std::pair<std::string, std::string> key{bitmap, keyColumn};
        if (valueExternalProbeKeys.insert(key).second)
            valueExternalProbes.push_back({bitmap, keyColumn});
    };
    for (const auto& [_, st] : states) {
        if (!st.externalToTable) continue;
        for (const auto& c : resolvedCorrelations) {
            if (c.outer.table == st.table && c.outer.column == st.column &&
                c.inner.table == dsq.valueTable) {
                addValueExternalProbe(st.bitmap, c.inner.column);
            }
        }
    }

    auto makeAggInput = [&]() -> std::unique_ptr<MetalOperator> {
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(dsq.valueTable, idxVar);
        pipe = maybeSelect(std::move(pipe), filterCondFor(dsq.valueTable));
        for (const auto& probe : valueExternalProbes) {
            pipe = std::make_unique<MetalBitmapProbe>(
                std::move(pipe), probe.bitmap, probe.keyColumn + "[" + idxVar + "]");
        }
        for (const auto& [_, st] : states) {
            if (st.table == dsq.valueTable && st.externalToTable) {
                pipe = std::make_unique<MetalBitmapProbe>(
                    std::move(pipe), st.bitmap, st.column + "[" + idxVar + "]");
            }
        }
        return pipe;
    };

    ScalarLookupInfo info;
    info.scalarSubqueryIndex = dsq.sqIdx;
    info.valueTable = dsq.valueTable;
    info.valueCol = dsq.valueCol;
    info.multiplier = dsq.multiplier;
    for (const auto& c : dsq.correlations) {
        info.keyCols.push_back(c.inner.column);
        info.outerKeyCols.push_back(c.outer.column);
    }
    info.keyCol = info.keyCols.empty() ? "" : info.keyCols[0];
    info.keyCol2 = info.keyCols.size() > 1 ? info.keyCols[1] : "";

    const std::string base = "d_scalar_" + std::to_string(dsq.sqIdx) + "_" +
                             sanitizeIdentifier(dsq.valueTable + "_" + info.keyCol +
                                                (info.keyCol2.empty() ? "" : "_" + info.keyCol2) +
                                                "_" + (dsq.valueCol.empty() ? "star" : dsq.valueCol));
    if (info.keyCols.empty()) {
        info.scalarName = "s_scalar_global_" + std::to_string(dsq.sqIdx);
        const std::string valueExpr = dsq.countStar ? "1.0f" : dsq.valueCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::COUNT) {
            info.kind = ScalarLookupInfo::GlobalCount;
            info.countBuffer = base + "_cnt";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, "0u", "1");
            auto& phase = appendPhase(
                plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_count",
                std::move(count), 256);
            attachGlobalScalarValueHook(phase, info);
            return info;
        }
        if (dsq.func == AggFunc::AVG) {
            info.kind = ScalarLookupInfo::GlobalAvg;
            info.countBuffer = base + "_cnt";
            info.sumBuffer = base + "_sum";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, "0u", "1");
            appendPhase(plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_avg_count",
                        std::move(count), 256);
            auto sum = makeScalarGlobalFloatAgg(
                makeAggInput(), "sum", info.sumBuffer, "", valueExpr);
            auto& phase = appendPhase(
                plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_avg_sum",
                std::move(sum), 256);
            attachGlobalScalarValueHook(phase, info);
            return info;
        }
        if (dsq.func == AggFunc::SUM) {
            info.kind = ScalarLookupInfo::GlobalSum;
            info.sumBuffer = base + "_sum";
            auto sum = makeScalarGlobalFloatAgg(
                makeAggInput(), "sum", info.sumBuffer, "", valueExpr);
            auto& phase = appendPhase(
                plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_sum",
                std::move(sum), 256);
            attachGlobalScalarValueHook(phase, info);
            return info;
        }
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) {
            bool isMin = dsq.func == AggFunc::MIN;
            info.kind = isMin ? ScalarLookupInfo::GlobalMin : ScalarLookupInfo::GlobalMax;
            if (isMin) info.minBuffer = base + "_min";
            else info.maxBuffer = base + "_max";
            info.stateBuffer = base + "_seen";
            auto minmax = makeScalarGlobalFloatAgg(
                makeAggInput(), isMin ? "min" : "max",
                isMin ? info.minBuffer : info.maxBuffer,
                info.stateBuffer, valueExpr);
            auto& phase = appendPhase(
                plan,
                "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) +
                    (isMin ? "_min" : "_max"),
                std::move(minmax), 256);
            attachGlobalScalarValueHook(phase, info);
            return info;
        }
        return std::nullopt;
    }

    if (info.keyCols.size() == 1) {
        info.sizeSymbol = maxKeySymbolForColumn(dsq.valueTable, info.keyCol, schema);
        const std::string keyExpr = info.keyCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::COUNT) {
            info.kind = ScalarLookupInfo::CountByKey;
            info.countBuffer = base + "_cnt";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, keyExpr, info.sizeSymbol);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_count",
                        std::move(count));
            return info;
        }

        const std::string valueExpr = dsq.countStar ? "1.0f" : dsq.valueCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::AVG) {
            info.kind = ScalarLookupInfo::AvgByKey;
            info.countBuffer = base + "_cnt";
            info.sumBuffer = base + "_sum";
            info.cntVar = "_scalar_" + std::to_string(dsq.sqIdx) + "_cnt";
            info.sumVar = "_scalar_" + std::to_string(dsq.sqIdx) + "_sum";
            auto avg = makeScalarDirectAvgAgg(
                makeAggInput(), info.countBuffer, info.sumBuffer, keyExpr,
                valueExpr, info.sizeSymbol);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" +
                              std::to_string(dsq.sqIdx) + "_avg",
                        std::move(avg));
            return info;
        }
        if (dsq.func == AggFunc::SUM) {
            info.kind = ScalarLookupInfo::SumByKey;
            info.sumBuffer = base + "_sum";
            info.stateBuffer = base + "_seen";
            auto sum = makeScalarDirectFloatAgg(
                makeAggInput(), "sum", info.sumBuffer, info.stateBuffer, keyExpr, valueExpr,
                info.sizeSymbol);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_sum",
                        std::move(sum));
            return info;
        }
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) {
            bool isMin = dsq.func == AggFunc::MIN;
            info.kind = isMin ? ScalarLookupInfo::MinByKey : ScalarLookupInfo::MaxByKey;
            if (isMin) info.minBuffer = base + "_min";
            else info.maxBuffer = base + "_max";
            info.stateBuffer = base + "_seen";
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min_init" : "_max_init"),
                        makeScalarFillFloatBuffer(
                            isMin ? info.minBuffer : info.maxBuffer,
                            info.sizeSymbol,
                            isMin ? "3.402823466e38f" : "-3.402823466e38f"));
            auto minmax = makeScalarDirectFloatAgg(
                makeAggInput(), isMin ? "min" : "max", isMin ? info.minBuffer : info.maxBuffer,
                info.stateBuffer, keyExpr, valueExpr, info.sizeSymbol);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min" : "_max"),
                        std::move(minmax));
            return info;
        }
        return std::nullopt;
    }

    if (info.keyCols.size() == 2) {
        if (dsq.func == AggFunc::MIN || dsq.func == AggFunc::MAX) return std::nullopt;
        info.hashCapacityExpr = "next_pow2(" + tableSizeName(dsq.valueTable) + " * 2)";
        std::string k1 = "(uint)(" + info.keyCol + "[" + idxVar + "])";
        std::string k2 = "(uint)(" + info.keyCol2 + "[" + idxVar + "])";
        std::string valueExpr = dsq.countStar ? "1u" : dsq.valueCol + "[" + idxVar + "]";

        ensureScalarCompositeHashHelpers(plan);

        if (dsq.func == AggFunc::COUNT || dsq.func == AggFunc::AVG) {
            info.countHashMap = "hm_scalar_" + std::to_string(dsq.sqIdx) + "_count";
            auto count = makeScalarCompositeHashAgg(
                makeAggInput(), info.countHashMap, k1, k2, "1u", info.hashCapacityExpr, false);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_hash_count",
                        std::move(count));
        }
        if (dsq.func == AggFunc::SUM || dsq.func == AggFunc::AVG) {
            info.hashMap = "hm_scalar_" + std::to_string(dsq.sqIdx) + "_sum";
            auto sum = makeScalarCompositeHashAgg(
                makeAggInput(), info.hashMap, k1, k2, valueExpr, info.hashCapacityExpr, true);
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) + "_hash_sum",
                        std::move(sum));
        }
        if (dsq.func == AggFunc::AVG) info.kind = ScalarLookupInfo::AvgByCompositeHash;
        else if (dsq.func == AggFunc::COUNT) info.kind = ScalarLookupInfo::CountByCompositeHash;
        else info.kind = ScalarLookupInfo::SumByCompositeHash;
        return info;
    }

    return std::nullopt;
}

static std::vector<ScalarLookupInfo> buildCorrelatedScalarPreAggs(
        const GenericRelPlan& ir,
        MetalQueryPlan& plan) {
    const auto& aq = ir.source;
    const auto* schema = ir.schema;
    std::vector<ScalarLookupInfo> result;
    int sqIdx = 0;
    for (const auto& sq : aq.subqueries) {
        if (sq.type == GenericSourceSubquery::SCALAR_SUBQUERY &&
            sq.decorrelatedScalar) {
            auto dsq = *sq.decorrelatedScalar;
            dsq.sqIdx = sqIdx;
            if (auto info = buildDecorrelatedScalarPreAgg(dsq, ir, schema, plan)) {
                result.push_back(std::move(*info));
            }
        }
        sqIdx++;
    }
    return result;
}

} // namespace

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
        const GenericRelPlan& ir,
        MetalQueryPlan& plan) {
    return buildCorrelatedScalarPreAggs(ir, plan);
}

} // namespace codegen
