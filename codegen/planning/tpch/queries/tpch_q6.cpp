#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ6PlanForShape(const std::set<std::string>& /*usedCols*/,
                                                  const std::string& filterCond,
                                                  const std::string& aggExpr,
                                                  const std::string& alias) {
    MetalQueryPlan plan;
    plan.name = "Q6";

    std::string idxVar = "i";
    auto filtered = maybeSelect(makeAutoScan("lineitem", idxVar), filterCond);

    auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), tableDataName(alias));
    reduce->addAccumulator(alias, "(long)(" + aggExpr + " * 100.0f)", "long");
    reduce->setResultAlias(alias, 100);

    appendPhase(plan, "Q6_reduce", std::move(reduce));
    return plan;
}

} // namespace

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

    return buildQ6PlanForShape(usedCols, filterCond, aggExpr, alias);
}

std::optional<MetalQueryPlan> buildQ6Plan_byName() {
    return buildQ6PlanForShape(
        {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"},
        "l_shipdate[i] >= 19940101 && l_shipdate[i] < 19950101 && "
        "(l_discount[i] >= 0.050000f && l_discount[i] <= 0.070000f) && "
        "l_quantity[i] < 24.000000f",
        "(l_extendedprice[i] * l_discount[i])",
        "revenue");
}

} // namespace codegen
