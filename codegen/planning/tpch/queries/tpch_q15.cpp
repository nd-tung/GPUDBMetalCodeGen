#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q15: Top Supplier.
std::optional<MetalQueryPlan> buildQ15Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Supplier Revenue ---
    // Accumulate revenue for the target shipping window.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19960101 && l_shipdate[" + idx + "] < 19960401");

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(filtered), "d_supp_revenue",
            "l_suppkey[" + idx + "]", revenue, "maxSuppkey",
            "atomic_float", "float");

        appendPhase(plan, "Q15_aggregate", std::move(agg));
    }

    // --- Maximum Revenue ---
    // Reduce per-supplier revenue to the global maximum.
    {
        auto scan = std::make_unique<MetalRangeScan>("maxSuppkey", idx);
        auto reduce = std::make_unique<MetalTGReduce>(std::move(scan), "d_q15_max");
        reduce->addAccumulator("revenue", "d_supp_revenue[" + idx + "]", "float",
                               "", "", MetalTGReduce::ReduceOp::MAX);
        auto& phase = appendPhase(plan, "Q15_max_revenue", std::move(reduce));
        phase.extraBuffers.push_back({"d_supp_revenue", "float", true, false});
    }

    // --- Top Supplier Materialization ---
    // The reduce output is compared through its raw float bits.
    {
        auto scan = std::make_unique<MetalRangeScan>("maxSuppkey", idx);
        auto filtered = std::make_unique<MetalSelection>(
            std::move(scan),
            "d_supp_revenue[" + idx + "] >= as_type<float>(d_q15_max_revenue_lo[0]) - 0.01f");
        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(filtered), "d_q15_result_count");
        materialize->addColumn("d_q15_result_suppkey", "int",
                               "(int)" + idx, "s_suppkey", "maxSuppkey");
        materialize->addColumn("d_q15_result_revenue", "float",
                               "d_supp_revenue[" + idx + "]",
                               "total_revenue", "maxSuppkey");
        auto& phase = appendPhase(plan, "Q15_materialize_top_supplier",
                                  std::move(materialize));
        phase.extraBuffers.push_back({"d_supp_revenue", "float", true, false});
        phase.extraBuffers.push_back({"d_q15_max_revenue_lo", "uint", true, false});
    }

    return plan;
}

} // namespace codegen
