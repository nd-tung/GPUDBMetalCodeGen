#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ12PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q12";
    std::string idxVar = "i";
    const std::string dateCond =
        "l_receiptdate[i] >= 19940101 && l_receiptdate[i] < 19950101";

    // --- Priority Bitmap ---
    // Mark high-priority orders before scanning lineitem.
    {
        auto scan = makeAutoScan("orders", idxVar);
        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderpriority[" + idxVar + "] == '1' || o_orderpriority[" + idxVar + "] == '2'");
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_priority_bitmap",
            "o_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");
        appendPhase(plan, "Q12_build_bitmap", std::move(bitmapBuild));
    }

    // --- Shipmode Counts ---
    // Four buckets encode shipmode and high/low priority.
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
            "(bitmap_test_atomic(d_priority_bitmap, l_orderkey[" + idxVar + "]) ? 0 : 1))";

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

// Q12: Shipping Modes and Order Priority.
std::optional<MetalQueryPlan> buildQ12Plan_byName() {
    return buildQ12PredefinedPlan();
}

} // namespace codegen
