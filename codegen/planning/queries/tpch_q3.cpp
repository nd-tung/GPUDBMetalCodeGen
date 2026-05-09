#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q3: Shipping Priority — 3 phases
// ===================================================================
std::optional<MetalQueryPlan> buildQ3Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 1: Build customer bitmap (BUILDING segment)
    {
        auto scan = makeScan("customer", idx, {{"c_custkey", "int"}, {"c_mktsegment", "char"}});

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_mktsegment[" + idx + "] == 'B'");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_cust_bitmap",
            "c_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        appendPhase(plan, "Q3_build_cust_bitmap", std::move(bitmap), 256);
    }

    // Phase 2: Build orders maps (date + priority, dual ArrayStore)
    {
        auto scan = makeScan("orders", idx, {
            {"o_orderkey", "int"}, {"o_custkey", "int"},
            {"o_orderdate", "int"}, {"o_shippriority", "int"}
        });

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] < 19950315");

        auto custProbed = std::make_unique<MetalBitmapProbe>(std::move(dateFiltered),
            "d_cust_bitmap", "o_custkey[" + idx + "]");

        auto storeDate = std::make_unique<MetalArrayStore>(
            std::move(custProbed), "d_orders_date_map",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "]",
            "int", "maxOrderkey");

        auto storePrio = std::make_unique<MetalArrayStore>(
            std::move(storeDate), "d_orders_prio_map",
            "o_orderkey[" + idx + "]", "o_shippriority[" + idx + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q3_build_orders_maps", std::move(storePrio), 256);
    }

    // Phase 3: Probe lineitem → aggregate revenue per orderkey
    {
        auto scan = makeScan("lineitem", idx, {
            {"l_orderkey", "int"}, {"l_shipdate", "int"},
            {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] > 19950315");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_orders_date_map",
            "l_orderkey[" + idx + "]",
            "_odate", "int", -1);

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_order_revenue",
            "l_orderkey[" + idx + "]", revenue, "maxOrderkey",
            "atomic_float", "float");

        appendPhase(plan, "Q3_probe_aggregate", std::move(agg));
    }

    // Phase 4: GPU compact-emit qualifying (orderkey, revenue) pairs.
    // Range scan over [0, n_q3_oks) reads dense d_order_revenue and
    // atomic-appends to a compact list. CPU joins with the existing
    // d_orders_date_map / d_orders_prio_map for date+prio and partial-
    // sorts to top 10.
    plan.helpers.push_back(R"(
static void q3_compact_emit(device atomic_uint* counter,
                             device uint* out_ok,
                             device float* out_rev,
                             const device float* d_order_revenue,
                             uint q3_compact_cap,
                             uint ok) {
    float r = d_order_revenue[ok];
    if (!(r > 0.0f)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q3_compact_cap) {
        out_ok[slot] = ok;
        out_rev[slot] = r;
    }
}
)");
    {
        auto rscan = std::make_unique<MetalRangeScan>("q3_oks", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q3_unused", "int",
            "(q3_compact_emit(d_q3_compact_count, d_q3_compact_ok, "
            "d_q3_compact_rev, d_order_revenue, q3_compact_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q3_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q3_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q3_compact_ok",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q3_compact_rev",   "float",       false, false});
        phase.extraBuffers.push_back({"d_order_revenue",    "float",       true,  false});
        phase.scalarParams.push_back({"q3_compact_cap", "uint"});
    }

    return plan;
}

} // namespace codegen
