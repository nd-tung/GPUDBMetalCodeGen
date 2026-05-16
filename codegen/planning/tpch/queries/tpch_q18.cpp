#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q18: Large Volume Customer — 3 GPU phases (incl. compact emit), CPU sort
// ===================================================================
std::optional<MetalQueryPlan> buildQ18Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: per-orderkey filter + atomic-append into compact list.
    // Caps writes at q18CompactCap to avoid out-of-bounds on extreme inputs;
    // the tight cap is sized in preprocessing (1<<20 slots).
    plan.helpers.push_back(R"(
static void q18_compact_emit(device atomic_uint* counter,
                              device uint* out_ok,
                              device float* out_qty,
                              const device float* d_order_qty,
                              uint q18_compact_cap,
                              uint ok) {
    float q = d_order_qty[ok];
    if (!(q > 300.0f)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q18_compact_cap) {
        out_ok[slot] = ok;
        out_qty[slot] = q;
    }
}
)");

    // Phase 1: build orderkey -> orders-row-index lookup on GPU
    // (replaces the 1.5M sequential CPU writes that previously dominated
    // Q18 preprocessing). fillByte = 0xFF gives -1 sentinel for missing
    // orderkeys; orderkeys are unique by FK so no atomics needed.
    {
        auto scan = makeAutoScan("orders", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q18_ok_lookup",
            "o_orderkey[" + idx + "]", "(int)" + idx,
            "int", "maxOrderkey", 0xFF);
        appendPhase(plan, "Q18_build_ok_lookup", std::move(store), 256);
    }

    // Phase 2: per-orderkey sum(l_quantity)
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(scan), "d_order_qty",
            "l_orderkey[" + idx + "]", "l_quantity[" + idx + "]", "maxOrderkey",
            "atomic_float", "float");

        appendPhase(plan, "Q18_aggregate", std::move(agg));
    }

    // Phase 3: GPU compact-emit qualifying orderkeys (qty > 300).
    // Range scan over [0, n_q18_oks) reads the dense d_order_qty
    // direct-address array and appends compact (ok, qty) pairs. The CPU
    // post then iterates only the small qualifying set.
    {
        auto rscan = std::make_unique<MetalRangeScan>("q18_oks", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q18_unused", "int",
            "(q18_compact_emit(d_q18_compact_count, d_q18_compact_ok, "
            "d_q18_compact_qty, d_order_qty, q18_compact_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q18_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q18_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q18_compact_ok",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q18_compact_qty",   "float",       false, false});
        phase.extraBuffers.push_back({"d_order_qty",         "float",       true,  false});
        phase.scalarParams.push_back({"q18_compact_cap", "uint"});
    }

    return plan;
}

} // namespace codegen
