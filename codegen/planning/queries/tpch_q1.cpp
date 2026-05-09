#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ1PlanForShape(const std::string& filterCond,
                                                  const std::set<std::string>& usedCols) {
    MetalQueryPlan plan;
    plan.name = "Q1";

    std::string idxVar = "i";
    auto filtered = maybeSelect(makeScanForCols("lineitem", idxVar, usedCols), filterCond);

    std::string bucketExpr = "((l_returnflag[" + idxVar + "] == 'A' ? 0 : (l_returnflag[" + idxVar + "] == 'N' ? 2 : 4)) + (l_linestatus[" + idxVar + "] == 'F' ? 0 : 1))";

    auto agg = std::make_unique<MetalKeyedAgg>(
        std::move(filtered), "d_q1_aggs", bucketExpr,
        /*numBuckets=*/6, /*valuesPerBucket=*/11, "66");

    agg->addAggregate("sum_qty", 0, "(uint)(l_quantity[" + idxVar + "] * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_base_price", 2, "(uint)(l_extendedprice[" + idxVar + "] * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_disc_price", 4,
                      "(uint)(l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "]) * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_charge", 6,
                      "(uint)(l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "]) * (1.0f + l_tax[" + idxVar + "]) * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_disc", 8, "(uint)(l_discount[" + idxVar + "] * 10000.0f)", "add", true, 0);
    agg->addAggregate("count_order", 10, "1u", "add", false, 0);

    appendPhase(plan, "Q1_reduce", std::move(agg));
    return plan;
}

} // namespace

// ===================================================================
// Q1 Plan Builder
// ===================================================================

std::optional<MetalQueryPlan> buildQ1Plan(const AnalyzedQuery& aq) {
    // Q1: single-table lineitem, GROUP BY l_returnflag, l_linestatus (6 bins)
    if (!aq.isSingleTable()) return std::nullopt;
    if (aq.tables[0] != "lineitem") return std::nullopt;
    if (!aq.hasAggregation() || !aq.hasGroupBy()) return std::nullopt;

    // Check for 2 GROUP BY columns (returnflag + linestatus)
    if (aq.groupBy.size() != 2) return std::nullopt;

    std::string idxVar = "i";
    std::string filterCond = combineFilters(aq.filters, idxVar);

    // Collect all columns used in filters, group by, and aggregates
    std::set<std::string> usedCols;
    for (const auto& f : aq.filters) collectColumns(f, usedCols);
    for (const auto& g : aq.groupBy) collectColumns(g, usedCols);
    for (const auto& t : aq.targets) {
        if (t.agg && t.agg->innerExpr) collectColumns(t.agg->innerExpr, usedCols);
    }

    return buildQ1PlanForShape(filterCond, usedCols);
}

std::optional<MetalQueryPlan> buildQ1Plan_byName() {
    return buildQ1PlanForShape(
        "l_shipdate[i] <= 19980902",
        {"l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax"});
}

} // namespace codegen
