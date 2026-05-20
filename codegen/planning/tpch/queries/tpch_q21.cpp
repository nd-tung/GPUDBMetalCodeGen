#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q21: Suppliers Who Kept Orders Waiting.
std::optional<MetalQueryPlan> buildQ21Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Build SAUDI-supplier bitmap.
    {
        auto scan = makeAutoScan("supplier", idx);
        auto filter = std::make_unique<MetalSelection>(
            std::move(scan), "s_nationkey[" + idx + "] == sa_nk");
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q21_sa_supp", "s_suppkey[" + idx + "]",
            "(maxSuppkey + 31) / 32");
        auto& phase = appendPhase(plan, "Q21_build_sa_supp", std::move(bitmapBuild));
        phase.scalarParams.push_back({"sa_nk", "int"});
    }

    // Track multi-supplier and multi-late orders.
    plan.helpers.push_back(R"(
static void q21_track_supplier(device atomic_int* first_supp,
                                device atomic_uint* multi_supp_bmp,
                                device atomic_int* first_late,
                                device atomic_uint* multi_late_bmp,
                                int ok, int sk, bool is_late) {
    int expected = -1;
    bool was_first = atomic_compare_exchange_weak_explicit(
        &first_supp[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
    if (!was_first && expected != sk) {
        atomic_fetch_or_explicit(&multi_supp_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
    }
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

    // Build final-orders bitmap.
    {
        auto scan = makeAutoScan("orders", idx);

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan), "o_orderstatus[" + idx + "] == 'F'");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q21_f_orders", "o_orderkey[" + idx + "]", "");

        appendPhase(plan, "Q21_build_f_orders", std::move(bitmapBuild));
    }

    // Build multi-supplier and multi-late bitmaps.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

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

    // Count qualifying suppliers.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

        auto saProbe = std::make_unique<MetalBitmapProbe>(
            std::move(fOrderProbe), "d_q21_sa_supp", "l_suppkey[" + idx + "]");

        auto lateFilter = std::make_unique<MetalSelection>(
            std::move(saProbe),
            "l_receiptdate[" + idx + "] > l_commitdate[" + idx + "]");

        auto multiSuppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(lateFilter), "d_q21_multi_supp", "l_orderkey[" + idx + "]");

        auto antiLateProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(multiSuppProbe), "d_q21_multi_late", "l_orderkey[" + idx + "]");

        auto countAgg = std::make_unique<MetalAtomicCount>(
            std::move(antiLateProbe),
            "d_q21_supp_count", "l_suppkey[" + idx + "]");

        appendPhase(plan, "Q21_count_qualifying", std::move(countAgg));
    }

    // Compact suppliers with nonzero wait counts for final ordering.
    plan.helpers.push_back(R"(
static void q21_emit_result(device atomic_uint* counter,
                            device char* out_name,
                            device uint* out_numwait,
                            const device uint* supp_count,
                            const device char* s_name,
                            uint cap, uint sk, uint row) {
    uint cnt = supp_count[sk];
    if (cnt == 0u) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        out_numwait[slot] = cnt;
        for (uint c = 0; c < 25u; ++c)
            out_name[slot * 25u + c] = s_name[row * 25u + c];
    }
}
)");

    const std::string resultRows = "q21_result_rows";
    {
        auto scan = makeScan("supplier", idx, {
            {"s_suppkey", "int"},
            {"s_name", "char"},
        });
        struct Q21CompactTerminal : MetalUnaryOperator {
            std::string idx_;
            Q21CompactTerminal(std::unique_ptr<MetalOperator> child, std::string idx)
                : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                cg.addColumnParam("s_suppkey", "int", "supplier");
                cg.addColumnParam("s_name", "char", "supplier");
                child_->produce(cg, [&]() {
                    cg.addLine("q21_emit_result(d_q21_result_count, "
                               "d_q21_result_name, d_q21_result_numwait, "
                               "d_q21_supp_count, s_name, q21_result_cap, "
                               "(uint)s_suppkey[" + idx_ + "], (uint)" + idx_ + ");");
                });
                consume();
            }
            std::string describe() const override { return "Q21CompactResult"; }
        };
        auto compact = std::make_unique<Q21CompactTerminal>(std::move(scan), idx);
        auto& phase = appendPhase(plan, "Q21_compact_results", std::move(compact));
        phase.extraBuffers.push_back({"d_q21_result_count",   "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q21_result_name",    "char",        false, false});
        phase.extraBuffers.push_back({"d_q21_result_numwait", "uint",        false, false});
        phase.extraBuffers.push_back({"d_q21_supp_count",     "uint",        true,  false});
        phase.scalarParams.push_back({"q21_result_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q21_result_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("s_name", "d_q21_result_name", "char", 25),
            GenericMatColumnDesc("numwait", "d_q21_result_numwait", "uint"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"numwait", true});
        sortSpec.keys.push_back({"s_name", false});
        sortSpec.limit = 100;
        std::string orderError;
        appendBestGenericGpuOrder(plan, "q21_result", resultRows,
                                  "q21_result_cap", columns, sortSpec,
                                  &orderError);
    }

    return plan;
}

} // namespace codegen
