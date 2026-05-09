#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q21: Suppliers Who Kept Orders Waiting
// Phase 0 (GPU): Scan supplier → build SA-supplier bitmap (sa_nk scalar)
// Phase 1 (GPU): Scan orders → build F-orders bitmap
// Phase 2 (GPU): Scan lineitem → atomicCAS to build multi_supp/multi_late bitmaps
// Phase 3 (GPU): Scan lineitem → filter → AtomicCount per supplier
// CPU pre: nation→SAUDI ARABIA key lookup (25 rows), s_suppkey/s_name mirror
//          for post-formatting; allocate first_supp/first_late counter arrays.
// CPU post: read per-supp counts, join names, sort, top 100.
// ===================================================================
std::optional<MetalQueryPlan> buildQ21Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 0: Build SAUDI-supplier bitmap on GPU. `sa_nk` scalar comes from
    // the host's tiny nation lookup (registerNameKey) — keeping the lookup
    // on CPU is cheaper than a 25-row dispatch.
    {
        auto scan = makeScan("supplier", idx,
                             {{"s_suppkey", "int"}, {"s_nationkey", "int"}});
        auto filter = std::make_unique<MetalSelection>(
            std::move(scan), "s_nationkey[" + idx + "] == sa_nk");
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q21_sa_supp", "s_suppkey[" + idx + "]",
            "(maxSuppkey + 31) / 32");
        auto& phase = appendPhase(plan, "Q21_build_sa_supp", std::move(bitmapBuild));
        phase.scalarParams.push_back({"sa_nk", "int"});
    }

    // Helper for Phase 2: atomic CAS to detect multi-supplier/multi-late orders
    plan.helpers.push_back(R"(
static void q21_track_supplier(device atomic_int* first_supp,
                                device atomic_uint* multi_supp_bmp,
                                device atomic_int* first_late,
                                device atomic_uint* multi_late_bmp,
                                int ok, int sk, bool is_late) {
    // Track multi-supplier orders
    int expected = -1;
    bool was_first = atomic_compare_exchange_weak_explicit(
        &first_supp[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
    if (!was_first && expected != sk) {
        atomic_fetch_or_explicit(&multi_supp_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
    }
    // Track multi-late orders
    if (is_late) {
        expected = -1;
        was_first = atomic_compare_exchange_weak_explicit(
            &first_late[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
        if (!was_first && expected != sk) {
            atomic_fetch_or_explicit(&multi_late_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
        }
    }
}
)");

    // Phase 1: Build F-orders bitmap on GPU
    {
        auto scan = makeScan("orders", idx, {{"o_orderkey", "int"}, {"o_orderstatus", "char"}});

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan), "o_orderstatus[" + idx + "] == 'F'");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q21_f_orders", "o_orderkey[" + idx + "]", "");

        appendPhase(plan, "Q21_build_f_orders", std::move(bitmapBuild));
    }

    // Phase 2: Build multi_supp and multi_late bitmaps on GPU
    {
        auto scan = makeScan("lineitem", idx, {
            {"l_orderkey", "int"}, {"l_suppkey", "int"},
            {"l_receiptdate", "int"}, {"l_commitdate", "int"}
        });

        // BitmapProbe: only process F-orders
        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

        // ComputeExpr: atomicCAS tracking (side-effect)
        auto trackExpr = std::make_unique<MetalComputeExpr>(
            std::move(fOrderProbe), "_unused", "int",
            "(q21_track_supplier(d_q21_first_supp, d_q21_multi_supp, "
            "d_q21_first_late, d_q21_multi_late, "
            "l_orderkey[" + idx + "], l_suppkey[" + idx + "], "
            "l_receiptdate[" + idx + "] > l_commitdate[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q21_build_bitmaps", std::move(trackExpr));
        phase.extraBuffers.push_back({"d_q21_first_supp", "atomic_int", false});
        phase.extraBuffers.push_back({"d_q21_first_late", "atomic_int", false});
        phase.extraBuffers.push_back({"d_q21_multi_supp", "atomic_uint", false});
        phase.extraBuffers.push_back({"d_q21_multi_late", "atomic_uint", false});
    }

    // Phase 3: Count qualifying suppliers
    {
        auto scan = makeScan("lineitem", idx, {
            {"l_orderkey", "int"}, {"l_suppkey", "int"},
            {"l_receiptdate", "int"}, {"l_commitdate", "int"}
        });

        // BitmapProbe: F-order
        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

        // BitmapProbe: SA supplier
        auto saProbe = std::make_unique<MetalBitmapProbe>(
            std::move(fOrderProbe), "d_q21_sa_supp", "l_suppkey[" + idx + "]");

        // Selection: late (receipt > commit)
        auto lateFilter = std::make_unique<MetalSelection>(
            std::move(saProbe),
            "l_receiptdate[" + idx + "] > l_commitdate[" + idx + "]");

        // BitmapProbe: multi-supplier order
        auto multiSuppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(lateFilter), "d_q21_multi_supp", "l_orderkey[" + idx + "]");

        // AntiBitmapProbe: NOT multi-late order
        auto antiLateProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(multiSuppProbe), "d_q21_multi_late", "l_orderkey[" + idx + "]");

        // AtomicCount: count per supplier
        auto countAgg = std::make_unique<MetalAtomicCount>(
            std::move(antiLateProbe),
            "d_q21_supp_count", "l_suppkey[" + idx + "]");

        appendPhase(plan, "Q21_count_qualifying", std::move(countAgg));
    }

    return plan;
}

} // namespace codegen
