#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ14PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q14";
    std::string idxVar = "i";
    const std::string filterCond =
        "l_shipdate[i] >= 19950901 && l_shipdate[i] < 19951001";

    // --- Promo Part Bitmap ---
    // Match fixed-width p_type values with the PROMO prefix.
    {
        auto scan = makeAutoScan("part", idxVar);
        std::string promoFilter =
            "p_type[" + idxVar + " * 25] == 'P' && "
            "p_type[" + idxVar + " * 25 + 1] == 'R' && "
            "p_type[" + idxVar + " * 25 + 2] == 'O' && "
            "p_type[" + idxVar + " * 25 + 3] == 'M' && "
            "p_type[" + idxVar + " * 25 + 4] == 'O'";

        auto filter = std::make_unique<MetalSelection>(std::move(scan), promoFilter);
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_promo_bitmap",
            "p_partkey[" + idxVar + "]", "(maxPartkey + 31) / 32 + 1");

        appendPhase(plan, "Q14_build_bitmap", std::move(bitmapBuild));
    }

    // --- Revenue Ratio Reduction ---
    // Reduce promo and total revenue so the collector can compute the ratio.
    {
        auto scan = makeAutoScan("lineitem", idxVar);

        auto filtered = maybeSelect(std::move(scan), filterCond);
        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "] * 0.01f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_q14");
        reduce->addAccumulator("promo",
            "bitmap_test_atomic(d_promo_bitmap, l_partkey[" + idxVar + "]) ? " + revenue + " : 0.0f", "float");
        reduce->addAccumulator("total", revenue, "float");
        reduce->setResultAlias("promo_revenue", 0);
        reduce->setResultAlias("total_revenue", 0);

        auto& phase = appendPhase(plan, "Q14_reduce", std::move(reduce));
        phase.bitmapReads.push_back({"d_promo_bitmap", ""});
    }

    return plan;
}

} // namespace

// Q14: Promotion Effect.
std::optional<MetalQueryPlan> buildQ14Plan_byName() {
    return buildQ14PredefinedPlan();
}

} // namespace codegen
