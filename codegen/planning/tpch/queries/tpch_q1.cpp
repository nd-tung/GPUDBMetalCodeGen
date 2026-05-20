#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ1PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q1";

    // --- Returnflag/Status Aggregate ---
    // Six buckets cover returnflag and linestatus combinations.
    std::string idxVar = "i";
    auto filtered = maybeSelect(makeAutoScan("lineitem", idxVar),
                                "l_shipdate[i] <= 19980902");

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

// Q1: Pricing Summary Report.
std::optional<MetalQueryPlan> buildQ1Plan_byName() {
    return buildQ1PredefinedPlan();
}

} // namespace codegen
