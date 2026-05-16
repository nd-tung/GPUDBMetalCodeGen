#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "generic/lowering/generic_scalar_preagg_ops.h"
#include "generic/lowering/generic_scalar_subquery_analysis.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "metal_plan_common.h"

#include <cctype>
#include <climits>
#include <map>
#include <optional>
#include <set>
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
    std::set<const Predicate*> copiedOuterFilters;
    // Bitmap state is one path per column; keep outer-filter seeding to
    // correlation-only subqueries so inner join constraints stay intact.
    const bool canSeedOuterFilters = dsq.joins.empty();
    for (const auto& c : dsq.correlations) {
        DecorrCol outer = c.outer;
        if (auto binding = resolveOuterFilterBinding(c.outer, aq)) {
            outer.table = binding->table;
            relevantCols[outer.table].insert(outer.column);

            if (canSeedOuterFilters) {
                for (const auto& pred : aq.filters) {
                    if (!pred || copiedOuterFilters.count(pred.get())) continue;
                    if (!predicateOnlyReferencesTable(pred, binding->table)) continue;
                    filtersByTable[binding->table].push_back(pred);
                    copiedOuterFilters.insert(pred.get());
                }
                if (!binding->alias.empty()) {
                    auto instIt = aq.instanceFilters.find(binding->alias);
                    if (instIt != aq.instanceFilters.end()) {
                        for (const auto& pred : instIt->second) {
                            if (!pred || copiedOuterFilters.count(pred.get())) continue;
                            filtersByTable[binding->table].push_back(pred);
                            copiedOuterFilters.insert(pred.get());
                        }
                    }
                }
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

    for (const auto& [table, filters] : filtersByTable) {
        auto colsIt = relevantCols.find(table);
        if (colsIt == relevantCols.end()) continue;
        for (const auto& col : colsIt->second) {
            std::string bitmap = stateName(table, col, "seed");
            auto pipe = makeFilteredScan(table);
            auto build = std::make_unique<MetalBitmapBuild>(
                std::move(pipe), bitmap, col + "[" + idxVar + "]",
                maxKeySymbolForColumn(table, col, aq.schema));
            appendPhase(plan, "GENERIC_scalar_decorrelate_" + std::to_string(dsq.sqIdx) +
                              "_" + sanitizeIdentifier(table + "_" + col + "_seed"),
                        std::move(build));
            addState(table, col, {table, col, bitmap, false});
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

    auto makeAggInput = [&]() -> std::unique_ptr<MetalOperator> {
        std::unique_ptr<MetalOperator> pipe = makeAutoScan(dsq.valueTable, idxVar);
        pipe = maybeSelect(std::move(pipe), filterCondFor(dsq.valueTable));
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
        const std::string valueExpr = dsq.countStar ? "1.0f" : dsq.valueCol + "[" + idxVar + "]";
        if (dsq.func == AggFunc::COUNT) {
            info.kind = ScalarLookupInfo::GlobalCount;
            info.countBuffer = base + "_cnt";
            auto count = std::make_unique<MetalAtomicCount>(
                makeAggInput(), info.countBuffer, "0u", "1");
            appendPhase(plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_count",
                        std::move(count), 256);
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
            appendPhase(plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_avg_sum",
                        std::move(sum), 256);
            return info;
        }
        if (dsq.func == AggFunc::SUM) {
            info.kind = ScalarLookupInfo::GlobalSum;
            info.sumBuffer = base + "_sum";
            auto sum = makeScalarGlobalFloatAgg(
                makeAggInput(), "sum", info.sumBuffer, "", valueExpr);
            appendPhase(plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) + "_sum",
                        std::move(sum), 256);
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
            appendPhase(plan, "GENERIC_scalar_global_" + std::to_string(dsq.sqIdx) +
                              (isMin ? "_min" : "_max"),
                        std::move(minmax), 256);
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
