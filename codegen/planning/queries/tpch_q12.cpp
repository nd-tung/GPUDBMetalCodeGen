#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

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

    MetalQueryPlan plan;
    plan.name = "Q12";
    std::string idxVar = "i";

    // Phase 1: Build priority bitmap from orders
    // Set bit for o_orderkey where o_orderpriority is '1-URGENT' or '2-HIGH'
    {
        auto scan = makeScan("orders", idxVar, {{"o_orderkey", "int"}, {"o_orderpriority", "char"}});

        // o_orderpriority CHAR1, first char: '1' or '2' → high priority
        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderpriority[" + idxVar + "] == '1' || o_orderpriority[" + idxVar + "] == '2'");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_priority_bitmap",
            "o_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");

        appendPhase(plan, "Q12_build_bitmap", std::move(bitmapBuild));
    }

    // Phase 2: Scan lineitem, filter by shipmode + date constraints, probe bitmap, 4-bin count
    {
        auto scan = makeScan("lineitem", idxVar, {
            {"l_orderkey", "int"}, {"l_shipmode", "char"}, {"l_shipdate", "int"},
            {"l_commitdate", "int"}, {"l_receiptdate", "int"}
        });

        // Lineitem filters:
        // 1. l_shipmode IN ('MAIL', 'SHIP') → first char 'M' or 'S'
        // 2. l_commitdate < l_receiptdate
        // 3. l_shipdate < l_commitdate
        // 4. l_receiptdate >= start_date AND l_receiptdate < end_date
        // l_shipmode is CHAR_FIXED(2), first char at l_shipmode[i * 2]

        // Extract date filters from analyzed query
        std::vector<PredPtr> dateFilters;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("l_receiptdate") && !cols.count("l_commitdate"))
                dateFilters.push_back(f);
        }
        std::string dateCond = combineFilters(dateFilters, idxVar);

        std::string filterCond =
            "(l_shipmode[" + idxVar + " * 2] == 'M' || l_shipmode[" + idxVar + " * 2] == 'S') && "
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "] && "
            "l_shipdate[" + idxVar + "] < l_commitdate[" + idxVar + "]";
        if (!dateCond.empty()) filterCond += " && " + dateCond;

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);

        // 4-bin keyed agg:
        // shipmode MAIL(0)/SHIP(2), crossed with high(+0)/low(+1) priority
        // Bucket = (l_shipmode first char == 'S' ? 2 : 0) + (bitmap_test ? 0 : 1)
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

// ===================================================================
// Dispatch: try all known patterns

} // namespace codegen
