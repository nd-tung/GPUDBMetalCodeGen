#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q5: Local Supplier Volume — 5 phases
// ===================================================================
// Phase 0: Scan nation → filter(n_regionkey == asia_rk) → BitmapBuild(nation_bitmap)
// Phase 1: Scan customer → BitmapProbe(nation_bitmap, c_nationkey) → ArrayStore(cust_nation_map)
// Phase 2: Scan supplier → BitmapProbe(nation_bitmap, s_nationkey) → ArrayStore(supp_nation_map)
// Phase 3: Scan orders → date filter → ArrayLookup(cust_nation_map) → ArrayStore(orders_nation_map)
// Phase 4: Scan lineitem → ArrayLookup(orders_nation_map) → ArrayLookup(supp_nation_map)
//          → same-nation filter → AtomicFloatAgg(nation_revenue[nationkey])
// Result: 25-element array indexed by nationkey

std::optional<MetalQueryPlan> buildQ5Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 0: Build nation bitmap (ASIA nations only)
    {
        auto scan = makeAutoScan("nation", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "n_regionkey[" + idx + "] == asia_rk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_nation_bitmap",
            "n_nationkey[" + idx + "]", "(25 + 31) / 32");

        auto& phase = appendPhase(plan, "Q5_build_nation_bitmap", std::move(bitmap), 32);
        phase.scalarParams = {{"asia_rk", "int"}};
    }

    // Phase 1: Build customer nation map (ASIA customers only)
    {
        auto scan = makeAutoScan("customer", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "c_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        appendPhase(plan, "Q5_build_cust_map", std::move(store), 256);
    }

    // Phase 2: Build supplier nation map (ASIA suppliers only)
    {
        auto scan = makeAutoScan("supplier", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "s_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        appendPhase(plan, "Q5_build_supp_map", std::move(store), 256);
    }

    // Phase 3: Build orders nation map (date-filtered, customer-in-ASIA)
    {
        auto scan = makeAutoScan("orders", idx);

        // Date filter: 1994-01-01 to 1994-12-31
        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] >= 19940101 && o_orderdate[" + idx + "] < 19950101");

        // Lookup customer nation map
        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_cust_nation_map",
            "o_custkey[" + idx + "]",
            "_cust_nk", "int", -1);

        // Store: orders_nation_map[orderkey] = cust_nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(lookup), "d_orders_nation_map",
            "o_orderkey[" + idx + "]", "_cust_nk",
            "int", "maxOrderkey");

        appendPhase(plan, "Q5_build_orders_map", std::move(store));
    }

    // Phase 4: Probe lineitem → same-nation check → aggregate
    {
        auto scan = makeAutoScan("lineitem", idx);

        // Lookup: cust_nk = orders_nation_map[l_orderkey]
        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_orders_nation_map",
            "l_orderkey[" + idx + "]",
            "_cust_nk", "int", -1);

        // Lookup: supp_nk = supp_nation_map[l_suppkey]
        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        // Same-nation filter: customer and supplier must be in same nation
        auto sameNation = std::make_unique<MetalSelection>(std::move(lookupSupp),
            "_cust_nk == _supp_nk");

        // AtomicFloatAgg: nation_revenue[nationkey] += revenue
        std::string valueExpr = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(sameNation), "d_nation_revenue",
            "_cust_nk", valueExpr, "25",
            "atomic_float", "float");

        appendPhase(plan, "Q5_probe_aggregate", std::move(agg));
    }

    return plan;
}

} // namespace codegen
