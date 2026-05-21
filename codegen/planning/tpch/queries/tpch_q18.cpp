#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q18: Large Volume Customer.
std::optional<MetalQueryPlan> buildQ18Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Compact writes are capped by q18_compact_cap.
    plan.helpers.push_back(R"(
static void q18_compact_emit(device atomic_uint* counter,
                              device uint* out_ok,
                              device char* out_c_name,
                              device int* out_custkey,
                              device float* out_totalprice,
                              device int* out_orderdate,
                              device float* out_qty,
                              const device float* d_order_qty,
                              const device int* d_q18_ok_lookup,
                              const device int* d_q18_customer_idx,
                              const device char* c_name,
                              const device int* o_custkey,
                              const device float* o_totalprice,
                              const device int* o_orderdate,
                              uint q18_compact_cap,
                              uint ok) {
    float q = d_order_qty[ok];
    if (!(q > 300.0f)) return;
    int order_idx = d_q18_ok_lookup[ok];
    if (order_idx < 0) return;
    int ck = o_custkey[order_idx];
    int ci = d_q18_customer_idx[ck];
    if (ci < 0) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q18_compact_cap) {
        out_ok[slot] = ok;
        out_custkey[slot] = ck;
        out_totalprice[slot] = o_totalprice[order_idx];
        out_orderdate[slot] = o_orderdate[order_idx];
        out_qty[slot] = q;
        for (uint c = 0; c < 25u; ++c) {
            out_c_name[slot * 25u + c] = c_name[(uint)ci * 25u + c];
        }
    }
}
)");

    // Build orderkey to orders-row lookup.
    {
        auto scan = makeAutoScan("orders", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q18_ok_lookup",
            "o_orderkey[" + idx + "]", "(int)" + idx,
            "int", "maxOrderkey", 0xFF);
        appendPhase(plan, "Q18_build_ok_lookup", std::move(store), 256);
    }
    {
        auto scan = makeAutoScan("customer", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q18_customer_idx",
            "c_custkey[" + idx + "]", "(int)" + idx,
            "int", "maxCustkey", 0xFF);
        appendPhase(plan, "Q18_build_customer_idx", std::move(store), 256);
    }

    // Sum quantity per orderkey.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(scan), "d_order_qty",
            "l_orderkey[" + idx + "]", "l_quantity[" + idx + "]", "maxOrderkey",
            "atomic_float", "float");

        appendPhase(plan, "Q18_aggregate", std::move(agg));
    }

    // Compact qualifying orders for GPU top-k.
    const std::string resultRows = "q18_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q18_oks", idx);
        rscan->addSideColumn("orders", "o_custkey", "int");
        rscan->addSideColumn("orders", "o_totalprice", "float");
        rscan->addSideColumn("orders", "o_orderdate", "int");
        rscan->addSideColumn("customer", "c_name", "char");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q18_unused", "int",
            "(q18_compact_emit(d_q18_compact_count, d_q18_compact_ok, "
            "d_q18_compact_name, d_q18_compact_custkey, d_q18_compact_totalprice, "
            "d_q18_compact_orderdate, d_q18_compact_qty, d_order_qty, "
            "d_q18_ok_lookup, d_q18_customer_idx, c_name, "
            "o_custkey, o_totalprice, o_orderdate, "
            "q18_compact_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q18_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q18_compact_count",      "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q18_compact_ok",         "uint",        false, false});
        phase.extraBuffers.push_back({"d_q18_compact_name",       "char",        false, false});
        phase.extraBuffers.push_back({"d_q18_compact_custkey",    "int",         false, false});
        phase.extraBuffers.push_back({"d_q18_compact_totalprice", "float",       false, false});
        phase.extraBuffers.push_back({"d_q18_compact_orderdate",  "int",         false, false});
        phase.extraBuffers.push_back({"d_q18_compact_qty",        "float",       false, false});
        phase.extraBuffers.push_back({"d_order_qty",              "float",       true,  false});
        phase.extraBuffers.push_back({"d_q18_ok_lookup",          "int",         true,  false});
        phase.extraBuffers.push_back({"d_q18_customer_idx",       "int",         true,  false});
        phase.scalarParams.push_back({"q18_compact_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q18_compact_count", resultRows);
    }

    // --- Result Order ---
    // Use TopK for the required top 100 orders when available.
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("o_orderkey", "d_q18_compact_ok", "uint"),
            GenericMatColumnDesc("c_custkey", "d_q18_compact_custkey", "int"),
            GenericMatColumnDesc("o_totalprice", "d_q18_compact_totalprice", "float"),
            GenericMatColumnDesc("o_orderdate", "d_q18_compact_orderdate", "int"),
            GenericMatColumnDesc("sum(l_quantity)", "d_q18_compact_qty", "float"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"o_totalprice", true});
        sortSpec.keys.push_back({"o_orderdate", false});
        sortSpec.limit = 100;
        std::string orderError;
        if (!appendBestGenericGpuOrder(plan, "q18_result", resultRows,
                                       "maxOrderkey", columns, sortSpec,
                                       &orderError)) {
            return std::nullopt;
        }
    }

    return plan;
}

} // namespace codegen
