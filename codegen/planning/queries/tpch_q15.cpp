#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q15: Top Supplier — 1 GPU phase + CPU max scan
// ===================================================================
std::optional<MetalQueryPlan> buildQ15Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    {
        auto scan = makeScan("lineitem", idx, {
            {"l_suppkey", "int"}, {"l_shipdate", "int"},
            {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19960101 && l_shipdate[" + idx + "] < 19960401");

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(filtered), "d_supp_revenue",
            "l_suppkey[" + idx + "]", revenue, "maxSuppkey",
            "atomic_float", "float");

        appendPhase(plan, "Q15_aggregate", std::move(agg));
    }

    return plan;
}

} // namespace codegen
