#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q2: Minimum Cost Supplier
// CPU pre-processing builds: qualifying part bitmap (size=15, type ends BRASS),
//   EUROPE supplier bitmap.
// GPU: scan partsupp → bitmap probes → atomic_min(supplycost as uint).
// ===================================================================
// Q2: Minimum Cost Supplier
// Phase 1 (GPU): Scan part → build qualifying part bitmap (size=15, type ends BRASS)
// Phase 2 (GPU): scan partsupp → bitmap probes → atomic_min(supplycost as uint).
// CPU pre: EUROPE supplier bitmap (tiny tables).
// CPU post: read min_cost array, match suppliers, join strings, sort, top 100.
// ===================================================================
std::optional<MetalQueryPlan> buildQ2Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: check if p_type ends with "BRASS" (fixed-width 25-char field)
    plan.helpers.push_back(R"(
static bool q2_type_ends_brass(const device char* p_type, uint idx) {
    const device char* tp = p_type + (uint)idx * 25u;
    int len = 25;
    while (len > 0 && (tp[len-1] == ' ' || tp[len-1] == '\0')) len--;
    return len >= 5 && tp[len-5]=='B' && tp[len-4]=='R' &&
           tp[len-3]=='A' && tp[len-2]=='S' && tp[len-1]=='S';
}
)");

    // Helper: atomic min for float (using uint reinterpretation)
    // For positive floats, as_type<uint>(f) preserves ordering.
    plan.helpers.push_back(R"(
static void q2_atomic_min(device atomic_uint* min_cost, uint partkey, float cost) {
    uint cost_uint = as_type<uint>(cost);
    atomic_fetch_min_explicit(&min_cost[partkey], cost_uint, memory_order_relaxed);
}
)");

    // Phase 1: Build part bitmap on GPU (size=15, type ends BRASS)
    {
        auto scan = makeAutoScan("part", idx);

        auto sizeFilter = std::make_unique<MetalSelection>(
            std::move(scan), "p_size[" + idx + "] == 15");

        auto typeFilter = std::make_unique<MetalSelection>(
            std::move(sizeFilter),
            "q2_type_ends_brass(p_type, " + idx + ")");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(typeFilter), "d_q2_part_bitmap", "p_partkey[" + idx + "]", "");

        appendPhase(plan, "Q2_build_part_bitmap", std::move(bitmapBuild));
    }

    // Phase 2: Find min cost (existing logic)
    {
        auto scan = makeAutoScan("partsupp", idx);

        // BitmapProbe: qualifying parts (size=15, type ends BRASS)
        auto partProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q2_part_bitmap", "ps_partkey[" + idx + "]");

        // BitmapProbe: EUROPE suppliers
        auto suppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(partProbe), "d_q2_supp_bitmap", "ps_suppkey[" + idx + "]");

        // ComputeExpr: atomic min on min_cost[ps_partkey]
        auto atomicMin = std::make_unique<MetalComputeExpr>(
            std::move(suppProbe), "_unused", "int",
            "(q2_atomic_min(d_q2_min_cost, (uint)ps_partkey[" + idx + "], "
            "ps_supplycost[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q2_find_min_cost", std::move(atomicMin));
        phase.extraBuffers.push_back({"d_q2_min_cost", "atomic_uint", false});
    }

    // Phase 3: GPU compact-emit. Scan partsupp once more with both
    // bitmap probes AND the supplycost==min_cost equality test, atomic-
    // append (partkey, suppkey, ps_idx) to a small list. Replaces the
    // 80M-row CPU loop in the post block.
    plan.helpers.push_back(R"(
static void q2_compact_emit(device atomic_uint* counter,
                             device uint* out_pk, device uint* out_sk, device uint* out_psi,
                             const device uint* d_q2_min_cost,
                             uint cap, uint pk, uint sk, float supplycost, uint i) {
    uint minU = d_q2_min_cost[pk];
    if (minU == 0xFFFFFFFFu) return;
    float minCost = as_type<float>(minU);
    if (supplycost != minCost) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        out_pk[slot] = pk;
        out_sk[slot] = sk;
        out_psi[slot] = i;
    }
}
)");
    {
        auto scan = makeAutoScan("partsupp", idx);
        auto partProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q2_part_bitmap", "ps_partkey[" + idx + "]");
        auto suppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(partProbe), "d_q2_supp_bitmap", "ps_suppkey[" + idx + "]");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(suppProbe), "_q2_unused", "int",
            "(q2_compact_emit(d_q2_compact_count, d_q2_compact_pk, "
            "d_q2_compact_sk, d_q2_compact_psi, d_q2_min_cost, "
            "q2_compact_cap, "
            "(uint)ps_partkey[" + idx + "], (uint)ps_suppkey[" + idx + "], "
            "ps_supplycost[" + idx + "], " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q2_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q2_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q2_compact_pk",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q2_compact_sk",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q2_compact_psi",   "uint",        false, false});
        // Re-bind the existing d_q2_min_cost buffer as read-only `uint`.
        // The same MTL::Buffer is bound; the type is just for kernel
        // compilation since this phase only reads (no atomics).
        phase.extraBuffers.push_back({"d_q2_min_cost",      "uint",        true,  false});
        phase.scalarParams.push_back({"q2_compact_cap", "uint"});
    }

    return plan;
}

} // namespace codegen
