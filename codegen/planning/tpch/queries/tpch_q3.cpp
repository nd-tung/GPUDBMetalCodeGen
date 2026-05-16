#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q3: Shipping Priority.
std::optional<MetalQueryPlan> buildQ3Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Customer Filter ---
    // Build customer bitmap for BUILDING segment.
    {
        auto scan = makeAutoScan("customer", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_mktsegment[" + idx + "] == 'B'");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_cust_bitmap",
            "c_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        appendPhase(plan, "Q3_build_cust_bitmap", std::move(bitmap), 256);
    }

    // --- Order Maps ---
    // Build order date and priority maps.
    {
        auto scan = makeAutoScan("orders", idx);

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

    // --- Revenue Aggregate ---
    // Aggregate lineitem revenue per orderkey.
    {
        auto scan = makeAutoScan("lineitem", idx);

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

    // --- Compact Results ---
    // Compact qualifying orders for GPU top-k.
    plan.helpers.push_back(R"(
static void q3_compact_emit(device atomic_uint* counter,
                             device uint* out_ok,
                             device float* out_rev,
                             device int* out_date,
                             device int* out_prio,
                             const device float* d_order_revenue,
                             const device int* d_orders_date_map,
                             const device int* d_orders_prio_map,
                             uint q3_compact_cap,
                             uint ok) {
    float r = d_order_revenue[ok];
    if (!(r > 0.0f)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q3_compact_cap) {
        out_ok[slot] = ok;
        out_rev[slot] = r;
        out_date[slot] = d_orders_date_map[ok];
        out_prio[slot] = d_orders_prio_map[ok];
    }
}
)");
    const std::string resultRows = "q3_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q3_oks", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q3_unused", "int",
            "(q3_compact_emit(d_q3_compact_count, d_q3_compact_ok, "
            "d_q3_compact_rev, d_q3_compact_date, d_q3_compact_prio, "
            "d_order_revenue, d_orders_date_map, d_orders_prio_map, "
            "q3_compact_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q3_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q3_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q3_compact_ok",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q3_compact_rev",   "float",       false, false});
        phase.extraBuffers.push_back({"d_q3_compact_date",  "int",         false, false});
        phase.extraBuffers.push_back({"d_q3_compact_prio",  "int",         false, false});
        phase.extraBuffers.push_back({"d_order_revenue",    "float",       true,  false});
        phase.extraBuffers.push_back({"d_orders_date_map",  "int",         true,  false});
        phase.extraBuffers.push_back({"d_orders_prio_map",  "int",         true,  false});
        phase.scalarParams.push_back({"q3_compact_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q3_compact_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("l_orderkey", "d_q3_compact_ok", "uint"),
            GenericMatColumnDesc("revenue", "d_q3_compact_rev", "float"),
            GenericMatColumnDesc("o_orderdate", "d_q3_compact_date", "int"),
            GenericMatColumnDesc("o_shippriority", "d_q3_compact_prio", "int"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"revenue", true});
        sortSpec.keys.push_back({"o_orderdate", false});
        sortSpec.limit = 10;
        std::string topKError;
        if (!appendGenericGpuTopK(plan, "q3_result", resultRows,
                                  "maxOrderkey", columns, sortSpec, &topKError)) {
            appendGenericGpuSort(plan, "q3_result", resultRows,
                                 "maxOrderkey", columns, sortSpec, &topKError);
        }
    }

    return plan;
}

} // namespace codegen
