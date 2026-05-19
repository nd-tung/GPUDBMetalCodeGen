#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "generic/lowering/generic_cost_model.h"
#include "generic/lowering/generic_scalar_preagg_ops.h"
#include "generic/lowering/generic_scalar_subquery_analysis.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "execution/metal_generic_executor.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <climits>
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

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front()))) {
        name = "c_" + name;
    }
    return name;
}

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

static std::optional<OuterFilterBinding> resolveOuterFilterBinding(
        const DecorrCol& outer,
        const AnalyzedQuery& aq) {
    std::vector<OuterFilterBinding> matches;
    auto addMatch = [&](std::string table, std::string alias) {
        if (!aq.schema || !aq.schema->hasColumn(table, outer.column)) return;
        for (const auto& existing : matches) {
            if (existing.table == table && existing.alias == alias) return;
        }
        matches.push_back({std::move(table), std::move(alias)});
    };

    if (!outer.table.empty()) {
        auto aliasIt = aq.aliasMap.find(outer.table);
        if (aliasIt != aq.aliasMap.end()) {
            addMatch(aliasIt->second, outer.table);
        }
        for (size_t i = 0; i < aq.tables.size(); ++i) {
            const std::string alias = i < aq.tableAliases.size()
                ? aq.tableAliases[i]
                : aq.tables[i];
            if (alias == outer.table || aq.tables[i] == outer.table)
                addMatch(aq.tables[i], alias);
        }
        addMatch(outer.table, "");
    } else {
        for (size_t i = 0; i < aq.tables.size(); ++i) {
            const std::string alias = i < aq.tableAliases.size()
                ? aq.tableAliases[i]
                : aq.tables[i];
            addMatch(aq.tables[i], alias == aq.tables[i] ? "" : alias);
        }
    }

    if (matches.size() != 1) return std::nullopt;
    return matches.front();
}

static bool predicateOnlyReferencesTable(const PredPtr& pred,
                                         const std::string& table) {
    std::map<std::string, std::string> colToTable;
    collectColumnTables(pred, colToTable);
    if (colToTable.empty()) return false;
    for (const auto& [_, refTable] : colToTable) {
        if (refTable != table) return false;
    }
    return true;
}

static bool exprHasScalarSentinel(const ExprPtr& expr, int sqIdx);

static bool predHasScalarSentinel(const PredPtr& pred, int sqIdx) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            return exprHasScalarSentinel(node.left, sqIdx) ||
                   exprHasScalarSentinel(node.right, sqIdx);
        } else if constexpr (std::is_same_v<T, Between>) {
            return exprHasScalarSentinel(node.expr, sqIdx) ||
                   exprHasScalarSentinel(node.low, sqIdx) ||
                   exprHasScalarSentinel(node.high, sqIdx);
        } else if constexpr (std::is_same_v<T, InList>) {
            if (exprHasScalarSentinel(node.expr, sqIdx)) return true;
            for (const auto& value : node.values)
                if (exprHasScalarSentinel(value, sqIdx)) return true;
            return false;
        } else if constexpr (std::is_same_v<T, Like>) {
            return exprHasScalarSentinel(node.expr, sqIdx);
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children)
                if (predHasScalarSentinel(child, sqIdx)) return true;
            return false;
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            return predHasScalarSentinel(node.child, sqIdx);
        } else {
            return false;
        }
    }, pred->node);
}

static bool exprHasScalarSentinel(const ExprPtr& expr, int sqIdx) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Literal>) {
            if (auto value = std::get_if<int>(&node.value))
                return *value == INT_MIN + sqIdx;
            return false;
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            return exprHasScalarSentinel(node.left, sqIdx) ||
                   exprHasScalarSentinel(node.right, sqIdx);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (const auto& arg : node.args)
                if (exprHasScalarSentinel(arg, sqIdx)) return true;
            return false;
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (const auto& branch : node.branches) {
                if (predHasScalarSentinel(branch.condition, sqIdx) ||
                    exprHasScalarSentinel(branch.result, sqIdx)) {
                    return true;
                }
            }
            return exprHasScalarSentinel(node.elseResult, sqIdx);
        } else {
            return false;
        }
    }, expr->node);
}

static std::optional<ScalarLookupInfo> buildDecorrelatedScalarPreAgg(
        const DecorrelatedScalarSubquery& dsq,
        const AnalyzedQuery& aq,
        MetalQueryPlan& plan) {
    const std::string idxVar = "i";
    std::map<std::string, std::set<std::string>> relevantCols;
    for (const auto& j : dsq.joins) {
        relevantCols[j.left.table].insert(j.left.column);
        relevantCols[j.right.table].insert(j.right.column);
    }
    for (const auto& c : dsq.correlations)
        relevantCols[c.inner.table].insert(c.inner.column);

    std::map<std::string, std::vector<PredPtr>> filtersByTable =
        dsq.filtersByTable;
    std::vector<ResolvedCorrelation> resolvedCorrelations;
    std::vector<std::pair<OuterFilterBinding, DecorrCol>> outerBindingCols;
    std::set<const Predicate*> copiedOuterFilters;
    std::set<std::pair<std::string, std::string>> outerFilterSeedCols;
    // Bitmap state is one path per column; keep outer-filter seeding to
    // correlation-only subqueries so inner join constraints stay intact.
    const bool canSeedOuterFilters = dsq.joins.empty();
    for (const auto& c : dsq.correlations) {
        DecorrCol outer = c.outer;
        if (auto binding = resolveOuterFilterBinding(c.outer, aq)) {
            outer.table = binding->table;
            relevantCols[outer.table].insert(outer.column);
            outerBindingCols.push_back({*binding, outer});

            if (canSeedOuterFilters) {
                bool hasOuterFilterForBinding = false;
                for (const auto& pred : aq.filters) {
                    if (!pred) continue;
                    if (predHasScalarSentinel(pred, dsq.sqIdx)) continue;
                    if (!predicateOnlyReferencesTable(pred, binding->table)) continue;
                    hasOuterFilterForBinding = true;
                    if (!copiedOuterFilters.count(pred.get())) {
                        filtersByTable[binding->table].push_back(pred);
                        copiedOuterFilters.insert(pred.get());
                    }
                }
                if (!binding->alias.empty()) {
                    auto instIt = aq.instanceFilters.find(binding->alias);
                    if (instIt != aq.instanceFilters.end()) {
                        for (const auto& pred : instIt->second) {
                            if (!pred) continue;
                            if (predHasScalarSentinel(pred, dsq.sqIdx)) continue;
                            hasOuterFilterForBinding = true;
                            if (!copiedOuterFilters.count(pred.get())) {
                                filtersByTable[binding->table].push_back(pred);
                                copiedOuterFilters.insert(pred.get());
                            }
                        }
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
        if (it == filtersByTable.end()) return std::string("true");
        return combineFilters(it->second, idxVar, aq.schema);
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
        std::string leftDomain = maxKeySymbolForColumn(leftTable, leftCol, aq.schema);
        std::string rightDomain = maxKeySymbolForColumn(rightTable, rightCol, aq.schema);
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
    for (size_t i = 0; i < aq.tables.size(); ++i) {
        const std::string& table = aq.tables[i];
        std::string node = table;
        if (i < aq.tableAliases.size() && !aq.tableAliases[i].empty())
            node = aq.tableAliases[i];
        outerNodeTable[node] = table;
        outerBaseCounts[table]++;
    }
    for (const auto& [node, table] : outerNodeTable) {
        if (outerBaseCounts[table] == 1)
            uniqueOuterNodeForBase[table] = node;
    }
    auto resolveOuterNode = [&](const std::string& name) -> std::string {
        if (outerNodeTable.count(name)) return name;
        auto aliasIt = aq.aliasMap.find(name);
        if (aliasIt != aq.aliasMap.end()) {
            auto uniqueIt = uniqueOuterNodeForBase.find(aliasIt->second);
            if (uniqueIt != uniqueOuterNodeForBase.end()) return uniqueIt->second;
        }
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
    for (const auto& join : aq.joins) {
        if (join.anti || join.leftOuter) continue;
        std::string leftNode = resolveOuterNode(join.leftTable);
        std::string rightNode = resolveOuterNode(join.rightTable);
        if (leftNode.empty() || rightNode.empty()) continue;
        const std::string& leftTable = outerNodeTable[leftNode];
        const std::string& rightTable = outerNodeTable[rightNode];
        if (outerBaseCounts[leftTable] != 1 || outerBaseCounts[rightTable] != 1)
            continue;
        outerJoinEdges.push_back({leftNode, leftTable, join.leftCol,
                                  rightNode, rightTable, join.rightCol});
        outerRelevantCols[leftNode].insert(join.leftCol);
        outerRelevantCols[rightNode].insert(join.rightCol);
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

    std::map<std::string, std::vector<PredPtr>> outerFiltersByNode;
    for (const auto& pred : aq.filters) {
        if (!pred) continue;
        if (predHasScalarSentinel(pred, dsq.sqIdx)) continue;
        std::map<std::string, std::string> colToTable;
        collectColumnTables(pred, colToTable);
        if (colToTable.empty()) continue;
        std::set<std::string> tables;
        for (const auto& [_, table] : colToTable) tables.insert(table);
        if (tables.size() != 1) continue;
        const std::string& table = *tables.begin();
        if (outerBaseCounts[table] != 1) continue;
        auto nodeIt = uniqueOuterNodeForBase.find(table);
        if (nodeIt != uniqueOuterNodeForBase.end())
            outerFiltersByNode[nodeIt->second].push_back(pred);
    }
    for (const auto& [alias, filters] : aq.instanceFilters) {
        std::string node = resolveOuterNode(alias);
        if (node.empty()) continue;
        const std::string& table = outerNodeTable[node];
        if (outerBaseCounts[table] != 1) continue;
        for (const auto& pred : filters) {
            if (predHasScalarSentinel(pred, dsq.sqIdx)) continue;
            if (pred) outerFiltersByNode[node].push_back(pred);
        }
    }

    auto outerFilterCondFor = [&](const std::string& node) {
        auto it = outerFiltersByNode.find(node);
        if (it == outerFiltersByNode.end()) return std::string("true");
        return combineFilters(it->second, idxVar, aq.schema);
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
        size_t width = scalarCostColumnWidth(table, keyCol, aq.schema);
        auto colsIt = outerRelevantCols.find(node);
        if (colsIt != outerRelevantCols.end()) {
            for (const auto& col : colsIt->second)
                width += scalarCostColumnWidth(table, col, aq.schema);
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
        input.buildRowsExpr = scalarCostTableRowsExpr(targetTable, aq.schema);
        input.targetRowsExpr = input.buildRowsExpr;
        input.keyDomainExpr = maxKeySymbolForColumn(targetTable, targetCol,
                                                    aq.schema);
        input.keyByteWidth =
            scalarCostColumnWidth(targetTable, targetCol, aq.schema);
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
                maxKeySymbolForColumn(table, col, aq.schema));
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
                        maxKeySymbolForColumn(st.table, outCol, aq.schema));
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
                    maxKeySymbolForColumn(dstTable, dstCol, aq.schema));
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
                maxKeySymbolForColumn(table, col, aq.schema));
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
                        maxKeySymbolForColumn(st.table, outCol, aq.schema));
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
                    maxKeySymbolForColumn(dst->table, dst->column, aq.schema));
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
                    maxKeySymbolForColumn(dst->table, dst->column, aq.schema));
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
    info.sentinel = INT_MIN + dsq.sqIdx;
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
        info.sizeSymbol = maxKeySymbolForColumn(dsq.valueTable, info.keyCol, aq.schema);
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

static std::vector<ScalarLookupInfo> buildCorrelatedScalarPreAggs(const AnalyzedQuery& aq,
                                                                   MetalQueryPlan& plan) {
    std::vector<ScalarLookupInfo> result;
    int sqIdx = 0;
    for (const auto& sq : aq.subqueries) {
        if (sq.type == AnalyzedQuery::Subquery::SCALAR_SUBQUERY) {
            if (auto dsq = parseDecorrelatedScalarSubquery(sq.sql, aq, sqIdx)) {
                if (auto info = buildDecorrelatedScalarPreAgg(*dsq, aq, plan))
                    result.push_back(std::move(*info));
            }
        }
        sqIdx++;
    }
    return result;
}

} // namespace

std::vector<GenericScalarLookupInfo> buildGenericScalarPreAggs(
        const AnalyzedQuery& aq,
        MetalQueryPlan& plan) {
    return buildCorrelatedScalarPreAggs(aq, plan);
}

} // namespace codegen
