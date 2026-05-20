#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ4PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q4";
    std::string idxVar = "i";
    const std::string filterCond =
        "o_orderdate[i] >= 19930701 && o_orderdate[i] < 19931001";

    // --- Late Orders Bitmap ---
    // Mark orders that have at least one late lineitem.
    {
        auto scan = makeAutoScan("lineitem", idxVar);

        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "]");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_late_bitmap",
            "l_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");

        appendPhase(plan, "Q4_build_bitmap", std::move(bitmapBuild));
    }

    // --- Priority Counts ---
    // Count date-filtered orders by priority after the late-order probe.
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

// Q4: Order Priority Checking.
std::optional<MetalQueryPlan> buildQ4Plan_byName() {
    return buildQ4PredefinedPlan();
}

} // namespace codegen
