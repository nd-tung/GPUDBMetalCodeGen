#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q10 Plan Builder — Returned Item Reporting
// Pattern: ArrayStore(orders) → ArrayLookup+AtomicFloatAgg(lineitem)
// ===================================================================

std::optional<MetalQueryPlan> buildQ10Plan(const AnalyzedQuery& aq) {
    // Q10: 4 tables (customer, orders, lineitem, nation), GROUP BY c_custkey + many cols
    // Detect: has lineitem+orders, GROUP BY includes c_custkey, SUM of revenue
    if (aq.tables.size() < 2) return std::nullopt;
    bool hasLineitem = false, hasOrders = false;
    for (auto& t : aq.tables)  {
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
    }
    if (!hasLineitem || !hasOrders) return std::nullopt;

    // Check for l_returnflag filter and GROUP BY c_custkey
    bool hasReturnflagFilter = false;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("l_returnflag")) hasReturnflagFilter = true;
    }
    if (!hasReturnflagFilter) return std::nullopt;

    bool groupByCustkey = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "c_custkey") groupByCustkey = true;
            }
        }, g->node);
    }
    if (!groupByCustkey) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q10";
    std::string idxVar = "i";

    // Phase 1: Build orders direct-address map
    // orders_map[orderkey] = custkey (filtered by date)
    {
        auto scan = makeScan("orders", idxVar, {
            {"o_orderkey", "int"}, {"o_custkey", "int"}, {"o_orderdate", "int"}
        });

        // Extract date filters from analyzed query
        std::vector<PredPtr> dateFilters;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("o_orderdate")) dateFilters.push_back(f);
        }
        std::string filterCond = combineFilters(dateFilters, idxVar);

        auto filtered = maybeSelect(std::move(scan), filterCond);

        // ArrayStore: orders_map[orderkey] = custkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_orders_map",
            "o_orderkey[" + idxVar + "]",
            "o_custkey[" + idxVar + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q10_build_orders_map", std::move(store));
    }

    // Phase 2: Probe lineitem, aggregate revenue per custkey
    {
        auto scan = makeScan("lineitem", idxVar, {
            {"l_orderkey", "int"}, {"l_returnflag", "char"},
            {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        // Filter: l_returnflag = 'R'
        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_returnflag[" + idxVar + "] == 'R'");

        // ArrayLookup: custkey = orders_map[l_orderkey]
        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_orders_map",
            "l_orderkey[" + idxVar + "]",
            "_custkey", "int", -1);

        // AtomicFloatAgg: cust_revenue[custkey] += revenue
        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_cust_revenue",
            "_custkey", revenue, "maxCustkey",
            "atomic_float", "float");

        appendPhase(plan, "Q10_probe_aggregate", std::move(agg));
    }

    // Phase 3: GPU compact-emit qualifying customers (rev > 0). Range
    // scan over [0, n_q10_cks) reads dense d_cust_revenue and atomic-
    // appends (custkey, revenue) pairs into a compact list. Replaces
    // the maxCustkey-sized CPU loop with iteration over only the
    // (typically all-customers) qualifying set.
    plan.helpers.push_back(R"(
static void q10_compact_emit(device atomic_uint* counter,
                              device uint* out_ck,
                              device float* out_rev,
                              const device float* d_cust_revenue,
                              uint q10_compact_cap,
                              uint ck) {
    float r = d_cust_revenue[ck];
    if (!(r > 0.0f)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q10_compact_cap) {
        out_ck[slot] = ck;
        out_rev[slot] = r;
    }
}
)");
    {
        auto rscan = std::make_unique<MetalRangeScan>("q10_cks", idxVar);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q10_unused", "int",
            "(q10_compact_emit(d_q10_compact_count, d_q10_compact_ck, "
            "d_q10_compact_rev, d_cust_revenue, q10_compact_cap, " + idxVar + "), 0)");
        auto& phase = appendPhase(plan, "Q10_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q10_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q10_compact_ck",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q10_compact_rev",   "float",       false, false});
        phase.extraBuffers.push_back({"d_cust_revenue",      "float",       true,  false});
        phase.scalarParams.push_back({"q10_compact_cap", "uint"});
    }

    return plan;
}

} // namespace codegen
