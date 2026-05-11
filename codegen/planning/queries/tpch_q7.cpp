#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q7: Volume Shipping — 4 phases
// ===================================================================
// Phase 1: Scan supplier → filter (FRANCE | GERMANY) → ArrayStore(supp_nation_map)
// Phase 2: Scan customer → filter (FRANCE | GERMANY) → ArrayStore(cust_nation_map)
// Phase 3: Scan orders → ArrayStore(orders_map[orderkey] = custkey)
// Phase 4: Scan lineitem → date filter → 3 ArrayLookups → pair filter → AtomicFloatAgg(4 bins)
// Result: 4 bins = 2 nation pairs × 2 years
//   bin = pair_idx * 2 + year_idx
//   pair 0: FRANCE→GERMANY, pair 1: GERMANY→FRANCE
//   year 0: 1995, year 1: 1996

std::optional<MetalQueryPlan> buildQ7Plan(const AnalyzedQuery& aq) {
    // Detection: 6 table refs (supplier, lineitem, orders, customer, nation×2),
    // GROUP BY with l_year, and joins linking all tables
    bool hasNation = false, hasLineitem = false, hasOrders = false;
    bool hasSupplier = false, hasCustomer = false;
    int nationCount = 0;
    for (auto& t : aq.tables) {
        if (t == "nation") { hasNation = true; nationCount++; }
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
        if (t == "supplier") hasSupplier = true;
        if (t == "customer") hasCustomer = true;
    }
    if (!(hasNation && hasLineitem && hasOrders && hasSupplier && hasCustomer))
        return std::nullopt;
    if (nationCount < 2) return std::nullopt;

    if (aq.groupBy.size() < 3) return std::nullopt;

    return buildQ7Plan_byName();
}

std::optional<MetalQueryPlan> buildQ7Plan_byName() {
    std::string idx = "i";

    MetalQueryPlan plan;

    // Phase 1: Build supplier nation map
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == france_nk || s_nationkey[" + idx + "] == germany_nk");

        // ArrayStore: supp_nation_map[suppkey] = nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        auto& phase = appendPhase(plan, "Q7_build_supp_map", std::move(store), 256);
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
    }

    // Phase 2: Build customer nation map
    {
        auto scan = makeAutoScan("customer", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_nationkey[" + idx + "] == france_nk || c_nationkey[" + idx + "] == germany_nk");

        // ArrayStore: cust_nation_map[custkey] = nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        auto& phase = appendPhase(plan, "Q7_build_cust_map", std::move(store), 256);
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
    }

    // Phase 3: Build orders map (orderkey → custkey)
    {
        auto scan = makeAutoScan("orders", idx);

        // ArrayStore: orders_map[orderkey] = custkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_orders_map",
            "o_orderkey[" + idx + "]", "o_custkey[" + idx + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q7_build_orders_map", std::move(store), 256);
    }

    // Phase 4: Probe lineitem → cascaded lookups → aggregate into 4 bins
    {
        auto scan = makeAutoScan("lineitem", idx);

        // Date filter: 1995-01-01 to 1996-12-31
        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19950101 && l_shipdate[" + idx + "] <= 19961231");

        // ArrayLookup: ck = orders_map[l_orderkey]
        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_orders_map",
            "l_orderkey[" + idx + "]",
            "_ck", "int", -1);

        // ArrayLookup: supp_nk = supp_nation_map[l_suppkey]
        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        // ArrayLookup: cust_nk = cust_nation_map[ck]
        auto lookupCust = std::make_unique<MetalArrayLookup>(
            std::move(lookupSupp), "d_cust_nation_map",
            "_ck",
            "_cust_nk", "int", -1);

        // Pair filter: (FRANCE→GERMANY) or (GERMANY→FRANCE)
        auto pairFiltered = std::make_unique<MetalSelection>(std::move(lookupCust),
            "((_supp_nk == france_nk && _cust_nk == germany_nk) || "
            "(_supp_nk == germany_nk && _cust_nk == france_nk))");

        // AtomicFloatAgg: revenue_bins[pair_idx * 2 + year_idx]
        // pair_idx: 0 if supp=FRANCE, 1 if supp=GERMANY
        // year_idx: shipdate/10000 - 1995
        std::string bucketExpr = "((_supp_nk == france_nk) ? 0 : 1) * 2 + (l_shipdate[" + idx + "] / 10000 - 1995)";
        std::string valueExpr = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(pairFiltered), "d_revenue_bins",
            bucketExpr, valueExpr, "4",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q7_probe_aggregate", std::move(agg));
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
    }

    return plan;
}

} // namespace codegen
