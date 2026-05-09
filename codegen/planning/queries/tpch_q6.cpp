#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q6 Plan Builder
// ===================================================================

std::optional<MetalQueryPlan> buildQ6Plan(const AnalyzedQuery& aq) {
    // Q6: single-table lineitem, SUM aggregate, no GROUP BY
    if (!aq.isSingleTable()) return std::nullopt;
    if (aq.tables[0] != "lineitem") return std::nullopt;
    if (!aq.hasAggregation() || aq.hasGroupBy()) return std::nullopt;

    // Should have exactly 1 SUM aggregate
    bool hasSumAgg = false;
    for (const auto& t : aq.targets) {
        if (t.isAgg && t.agg && t.agg->func == AggFunc::SUM)
            hasSumAgg = true;
    }
    if (!hasSumAgg) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q6";

    // Collect all referenced columns from filters and aggregates
    std::set<std::string> usedCols;
    for (const auto& f : aq.filters) collectColumns(f, usedCols);
    for (const auto& t : aq.targets) {
        if (t.agg && t.agg->innerExpr) collectColumns(t.agg->innerExpr, usedCols);
    }

    // Build the aggregate expression using columnar indexing
    std::string idxVar = "i";
    std::string aggExpr;
    std::string alias = "revenue";
    for (const auto& t : aq.targets) {
        if (t.isAgg && t.agg && t.agg->func == AggFunc::SUM) {
            aggExpr = exprToMetal(t.agg->innerExpr, idxVar);
            if (!t.alias.empty()) alias = t.alias;
        }
    }

    // Build filter predicate using columnar indexing
    std::string filterCond = combineFilters(aq.filters, idxVar);

    auto filtered = maybeSelect(makeScanForCols("lineitem", idxVar, usedCols), filterCond);

    auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), tableDataName(alias));
    // For Q6, use long (fixed-point 100x) accumulation for precision
    reduce->addAccumulator(alias, "(long)(" + aggExpr + " * 100.0f)", "long");

    // Register result schema: scalar aggregate, 1 column, scale down by 100
    reduce->setResultAlias(alias, 100);

    appendPhase(plan, "Q6_reduce", std::move(reduce));

    return plan;
}

} // namespace codegen
