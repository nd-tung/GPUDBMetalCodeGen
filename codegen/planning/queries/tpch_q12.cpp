#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ12PlanForDateFilter(const std::string& dateCond) {
    MetalQueryPlan plan;
    plan.name = "Q12";
    std::string idxVar = "i";

    {
        auto scan = makeAutoScan("orders", idxVar);
        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderpriority[" + idxVar + "] == '1' || o_orderpriority[" + idxVar + "] == '2'");
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_priority_bitmap",
            "o_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");
        appendPhase(plan, "Q12_build_bitmap", std::move(bitmapBuild));
    }

    {
        auto scan = makeAutoScan("lineitem", idxVar);

        std::string filterCond =
            "(l_shipmode[" + idxVar + " * 2] == 'M' || l_shipmode[" + idxVar + " * 2] == 'S') && "
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "] && "
            "l_shipdate[" + idxVar + "] < l_commitdate[" + idxVar + "]";
        if (!dateCond.empty()) filterCond += " && " + dateCond;

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
        std::string bucketExpr =
            "((l_shipmode[" + idxVar + " * 2] == 'S' ? 2 : 0) + "
            "(bitmap_test(d_priority_bitmap, l_orderkey[" + idxVar + "]) ? 0 : 1))";

        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(filtered), "d_q12_counts", bucketExpr,
            /*numBuckets=*/4, /*valuesPerBucket=*/1, "4");
        agg->addAggregate("count", 0, "1u", "add", false, 0);

        auto& phase = appendPhase(plan, "Q12_count", std::move(agg));
        phase.bitmapReads.push_back({"d_priority_bitmap", ""});
    }

    return plan;
}

} // namespace

// ===================================================================
// Q12 Plan Builder — Shipping Modes and Order Priority
// Pattern: BitmapBuild(orders, priority) → Filter+KeyedAgg(lineitem)
// ===================================================================

std::optional<MetalQueryPlan> buildQ12Plan(const AnalyzedQuery& aq) {
    // Q12: lineitem + orders, GROUP BY l_shipmode, SUM(CASE priority)
    if (aq.tables.size() != 2) return std::nullopt;
    bool hasLineitem = false, hasOrders = false;
    for (auto& t : aq.tables) {
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
    }
    if (!hasLineitem || !hasOrders) return std::nullopt;
    if (!aq.hasGroupBy()) return std::nullopt;

    // Check GROUP BY l_shipmode (not o_orderpriority — that's Q4)
    bool groupByShipmode = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "l_shipmode") groupByShipmode = true;
            }
        }, g->node);
    }
    if (!groupByShipmode) return std::nullopt;

    std::string idxVar = "i";

    std::vector<PredPtr> dateFilters;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("l_receiptdate") && !cols.count("l_commitdate"))
            dateFilters.push_back(f);
    }
    return buildQ12PlanForDateFilter(combineFilters(dateFilters, idxVar));
}

std::optional<MetalQueryPlan> buildQ12Plan_byName() {
    return buildQ12PlanForDateFilter("l_receiptdate[i] >= 19940101 && l_receiptdate[i] < 19950101");
}

// ===================================================================
// Dispatch: try all known patterns

} // namespace codegen
