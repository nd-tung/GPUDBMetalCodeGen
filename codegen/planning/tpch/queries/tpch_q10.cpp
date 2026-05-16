#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ10PlanForDateFilter(const std::string& filterCond) {
    MetalQueryPlan plan;
    plan.name = "Q10";
    std::string idxVar = "i";

    // --- Orders Map ---
    // Map selected orders to customers before probing from lineitem.
    {
        auto scan = makeAutoScan("orders", idxVar);

        auto filtered = maybeSelect(std::move(scan), filterCond);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_orders_map",
            "o_orderkey[" + idxVar + "]",
            "o_custkey[" + idxVar + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q10_build_orders_map", std::move(store));
    }

    // --- Revenue Aggregate ---
    // Sum returned-line revenue by customer key.
    {
        auto scan = makeAutoScan("lineitem", idxVar);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_returnflag[" + idxVar + "] == 'R'");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_orders_map",
            "l_orderkey[" + idxVar + "]",
            "_custkey", "int", -1);

        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_cust_revenue",
            "_custkey", revenue, "maxCustkey",
            "atomic_float", "float");

        appendPhase(plan, "Q10_probe_aggregate", std::move(agg));
    }

    // --- Compact Results ---
    // Keep only customers with positive revenue for the final GPU order.
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

    const std::string resultRows = "q10_result_rows";
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
        attachMaterializedCountHook(phase, "d_q10_compact_count", resultRows);
    }

    // --- Result Order ---
    // Prefer TopK; fall back to full sort when the shape is unsupported.
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("c_custkey", "d_q10_compact_ck", "int"),
            GenericMatColumnDesc("revenue", "d_q10_compact_rev", "float"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"revenue", true});
        sortSpec.keys.push_back({"c_custkey", false});
        sortSpec.limit = 20;
        std::string topKError;
        if (!appendGenericGpuTopK(plan, "q10_result", resultRows,
                                  "maxCustkey", columns, sortSpec, &topKError)) {
            appendGenericGpuSort(plan, "q10_result", resultRows,
                                 "maxCustkey", columns, sortSpec, &topKError);
        }
    }

    return plan;
}

} // namespace

std::optional<MetalQueryPlan> buildQ10Plan(const AnalyzedQuery& aq) {
    // Match Q10 only when the analyzed shape has the required join, filter, and group key.
    if (aq.tables.size() < 2) return std::nullopt;
    bool hasLineitem = false, hasOrders = false;
    for (auto& t : aq.tables)  {
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
    }
    if (!hasLineitem || !hasOrders) return std::nullopt;

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

    std::string idxVar = "i";
    std::vector<PredPtr> dateFilters;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("o_orderdate")) dateFilters.push_back(f);
    }
    return buildQ10PlanForDateFilter(combineFilters(dateFilters, idxVar));
}

std::optional<MetalQueryPlan> buildQ10Plan_byName() {
    return buildQ10PlanForDateFilter("o_orderdate[i] >= 19931001 && o_orderdate[i] < 19940101");
}

} // namespace codegen
