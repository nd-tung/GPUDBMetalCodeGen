#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q11: Important Stock Identification — 2 phases
// ===================================================================
std::optional<MetalQueryPlan> buildQ11Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 1: Build supplier bitmap for GERMANY
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == germany_nk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_supp_bitmap",
            "s_suppkey[" + idx + "]", "(maxSuppkey + 31) / 32");

        auto& phase = appendPhase(plan, "Q11_build_supp_bitmap", std::move(bitmap), 256);
        phase.scalarParams = {{"germany_nk", "int"}};
    }

    // Phase 2: Scan partsupp → bitmap probe → per-part value aggregation
    {
        auto scan = makeAutoScan("partsupp", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_supp_bitmap", "ps_suppkey[" + idx + "]");

        std::string valueExpr = "ps_supplycost[" + idx + "] * (float)ps_availqty[" + idx + "]";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(probed), "d_part_value",
            "ps_partkey[" + idx + "]", valueExpr, "maxPartkey",
            "atomic_float", "float");

        appendPhase(plan, "Q11_aggregate", std::move(agg), 256);
    }

    return plan;
}

} // namespace codegen
