#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q7: Volume Shipping.
std::optional<MetalQueryPlan> buildQ7Plan(const AnalyzedQuery& aq) {
    // Match the canonical five-table, two-nation Q7 shape.
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

    // --- Nation Maps ---
    // Build supplier nation map.
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == france_nk || s_nationkey[" + idx + "] == germany_nk");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        auto& phase = appendPhase(plan, "Q7_build_supp_map", std::move(store), 256);
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
    }

    // Build customer nation map.
    {
        auto scan = makeAutoScan("customer", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_nationkey[" + idx + "] == france_nk || c_nationkey[" + idx + "] == germany_nk");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        auto& phase = appendPhase(plan, "Q7_build_cust_map", std::move(store), 256);
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
    }

    // --- Orders Map ---
    // Build orderkey to customer map.
    {
        auto scan = makeAutoScan("orders", idx);

        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_orders_map",
            "o_orderkey[" + idx + "]", "o_custkey[" + idx + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q7_build_orders_map", std::move(store), 256);
    }

    // --- Revenue Aggregate ---
    // Aggregate revenue by nation pair and year.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19950101 && l_shipdate[" + idx + "] <= 19961231");

        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_orders_map",
            "l_orderkey[" + idx + "]",
            "_ck", "int", -1);

        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        auto lookupCust = std::make_unique<MetalArrayLookup>(
            std::move(lookupSupp), "d_cust_nation_map",
            "_ck",
            "_cust_nk", "int", -1);

        auto pairFiltered = std::make_unique<MetalSelection>(std::move(lookupCust),
            "((_supp_nk == france_nk && _cust_nk == germany_nk) || "
            "(_supp_nk == germany_nk && _cust_nk == france_nk))");

        // Four bins encode supplier nation direction and ship year.
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
