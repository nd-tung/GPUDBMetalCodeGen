#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ4PlanForDateFilter(const std::string& filterCond) {
    MetalQueryPlan plan;
    plan.name = "Q4";
    std::string idxVar = "i";

    {
        auto scan = makeAutoScan("lineitem", idxVar);

        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "]");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_late_bitmap",
            "l_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");

        appendPhase(plan, "Q4_build_bitmap", std::move(bitmapBuild));
    }

    {
        auto scan = makeAutoScan("orders", idxVar);

        auto filtered = maybeSelect(std::move(scan), filterCond);
        auto probed = std::make_unique<MetalBitmapProbe>(
            std::move(filtered), "d_late_bitmap",
            "o_orderkey[" + idxVar + "]");

        std::string bucketExpr = "(o_orderpriority[" + idxVar + "] - '1')";
        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(probed), "d_q4_counts", bucketExpr,
            /*numBuckets=*/5, /*valuesPerBucket=*/1, "5");
        agg->addAggregate("order_count", 0, "1u", "add", false, 0);

        appendPhase(plan, "Q4_count", std::move(agg));
    }

    return plan;
}

} // namespace

// ===================================================================
// Q4 Plan Builder — Order Priority Checking
// Pattern: BitmapBuild(lineitem, late) → Filter+KeyedAgg(orders, date)
// ===================================================================

std::optional<MetalQueryPlan> buildQ4Plan(const AnalyzedQuery& aq) {
    // Q4: orders table with GROUP BY o_orderpriority
    // + EXISTS subquery on lineitem (commitdate < receiptdate)
    if (aq.tables.size() < 1) return std::nullopt;
    bool hasOrders = false;
    for (auto& t : aq.tables) if (t == "orders") hasOrders = true;
    if (!hasOrders) return std::nullopt;
    if (!aq.hasGroupBy()) return std::nullopt;

    // Check for EXISTS subquery and o_orderpriority GROUP BY
    bool hasExists = false;
    for (auto& f : aq.filters) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ExistsPred>) hasExists = true;
        }, f->node);
    }
    if (!hasExists) return std::nullopt;

    // Verify GROUP BY o_orderpriority
    bool groupByPriority = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "o_orderpriority") groupByPriority = true;
            }
        }, g->node);
    }
    if (!groupByPriority) return std::nullopt;

    std::string idxVar = "i";

    std::vector<PredPtr> dateFilters;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("o_orderdate")) dateFilters.push_back(f);
    }
    return buildQ4PlanForDateFilter(combineFilters(dateFilters, idxVar));
}

std::optional<MetalQueryPlan> buildQ4Plan_byName() {
    return buildQ4PlanForDateFilter("o_orderdate[i] >= 19930701 && o_orderdate[i] < 19931001");
}

} // namespace codegen
