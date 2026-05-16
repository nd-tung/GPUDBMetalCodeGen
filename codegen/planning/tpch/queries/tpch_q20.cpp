#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q20: Potential Part Promotion.
std::optional<MetalQueryPlan> buildQ20Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Hash Helpers ---
    // Hash probe used by the lineitem aggregation phase.
    plan.helpers.push_back(R"(
static void q20_ht_add(const device ulong* ht_keys, device atomic_float* ht_vals,
                        uint ht_mask, ulong key, float qty) {
    uint h = ((uint)(key ^ (key >> 32)) * 2654435769u) & ht_mask;
    for (uint step = 0; step < 64; step++) {
        uint slot = (h + step) & ht_mask;
        ulong k = ht_keys[slot];
        if (k == key) {
            atomic_fetch_add_explicit(&ht_vals[slot], qty, memory_order_relaxed);
            return;
        }
        if (k == 0xFFFFFFFFFFFFFFFFul) return; // not qualifying partsupp
    }
}
)");

    // Build-phase insert: atomic ps_idx claims ownership, then key is written.
    plan.helpers.push_back(R"(
static void q20_ht_insert(device atomic_int* ht_psidx, device ulong* ht_keys,
                           uint ht_mask, ulong key, int ps_idx) {
    uint h = ((uint)(key ^ (key >> 32)) * 2654435769u) & ht_mask;
    for (uint step = 0; step <= ht_mask; step++) {
        uint slot = (h + step) & ht_mask;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &ht_psidx[slot], &expected, ps_idx,
                memory_order_relaxed, memory_order_relaxed)) {
            ht_keys[slot] = key;
            return;
        }
        // Unique (pk, sk) rows mean collisions only need linear probing.
    }
}
)");

    // --- Part Filter ---
    // Build forest% part bitmap.
    {
        auto scan = makeAutoScan("part", idx);
        // p_name is fixed-width; "forest" starts at byte 0.
        std::string base = "p_name[" + idx + "*55";
        std::string pred =
            "(" + base + "+0]=='f' && " + base + "+1]=='o' && " +
            base + "+2]=='r' && " + base + "+3]=='e' && " +
            base + "+4]=='s' && " + base + "+5]=='t')";
        auto filter = std::make_unique<MetalSelection>(std::move(scan), pred);
        auto bmp = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q20_part_bitmap",
            "p_partkey[" + idx + "]", "(maxPartkey + 31) / 32");
        appendPhase(plan, "Q20_build_part_bitmap", std::move(bmp));
    }

    // --- Partsupp Hash Table ---
    // Build (pk, sk) to partsupp-index hash table.
    {
        auto scan = makeAutoScan("partsupp", idx);
        auto gated = std::make_unique<MetalSelection>(std::move(scan),
            "bitmap_test_atomic(d_q20_part_bitmap, ps_partkey[" + idx + "])");
        auto computeKey = std::make_unique<MetalComputeExpr>(
            std::move(gated), "_psk", "ulong",
            "(ulong)ps_partkey[" + idx + "] * (ulong)supp_mul + "
            "(ulong)ps_suppkey[" + idx + "]");

        struct HtInsertTerminal : MetalUnaryOperator {
            std::string idx_;
            HtInsertTerminal(std::unique_ptr<MetalOperator> c, std::string i)
                : MetalUnaryOperator(std::move(c)), idx_(std::move(i)) {}
            void produce(MetalCodegen& cg, ConsumerFn) override {
                cg.addBufferParam("d_q20_ht_keys", "ulong", "q20HtSize",
                                  /*zeroInit=*/true, /*fillByte=*/0xFF);
                cg.addBufferParam("d_q20_ht_psidx", "atomic_int", "q20HtSize",
                                  /*zeroInit=*/true, /*fillByte=*/0xFF);
                // Allocate ht_vals for the later read-only extra buffer.
                cg.addBufferParam("d_q20_ht_vals", "float", "q20HtSize",
                                  /*zeroInit=*/true, /*fillByte=*/0);
                child_->produce(cg, [&]() {
                    cg.addLine(
                        "q20_ht_insert(d_q20_ht_psidx, d_q20_ht_keys, "
                        "d_q20_ht_mask, _psk, (int)" + idx_ + ");");
                });
            }
            std::string describe() const override { return "Q20HtInsert"; }
        };
        auto term = std::make_unique<HtInsertTerminal>(std::move(computeKey), idx);
        auto& phase = appendPhase(plan, "Q20_build_ht", std::move(term));
        phase.bitmapReads.push_back({"d_q20_part_bitmap", ""});
        phase.scalarParams.push_back({"d_q20_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    // --- Quantity Aggregate ---
    // Aggregate 1994 lineitem quantity into hash-table values.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto dateFilter = std::make_unique<MetalSelection>(
            std::move(scan),
            "l_shipdate[" + idx + "] >= 19940101 && l_shipdate[" + idx + "] < 19950101");

        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(dateFilter), "d_q20_part_bitmap", "l_partkey[" + idx + "]");

        auto hashAgg = std::make_unique<MetalComputeExpr>(
            std::move(bmpProbe), "_unused", "int",
            "(q20_ht_add(d_q20_ht_keys, d_q20_ht_vals, d_q20_ht_mask, "
            "(ulong)l_partkey[" + idx + "] * (ulong)supp_mul + (ulong)l_suppkey[" + idx + "], "
            "l_quantity[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q20_lineitem_agg", std::move(hashAgg));
        // Reuse HT keys read-only; write quantities through atomic_float values.
        phase.extraBuffers.push_back({"d_q20_ht_keys", "ulong", true, false});
        phase.extraBuffers.push_back({"d_q20_ht_vals", "atomic_float", false, false});
        phase.scalarParams.push_back({"d_q20_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    // --- Qualifying Supplier Bitmap ---
    // Build qualifying-supplier bitmap from hash-table slots.
    plan.helpers.push_back(R"(
static void q20_qual_emit(const device ulong* ht_keys,
                           const device float* ht_vals,
                           const device int*   ht_psidx,
                           const device int*   ps_availqty,
                           const device int*   ps_suppkey,
                           device atomic_uint* qual_supp_bitmap,
                           uint slot) {
    ulong k = ht_keys[slot];
    if (k == 0xFFFFFFFFFFFFFFFFul) return;
    int psIdx = ht_psidx[slot];
    if (psIdx < 0) return;
    float sumQty = ht_vals[slot];
    if (!(sumQty > 0.0f)) return;
    if ((float)ps_availqty[psIdx] <= 0.5f * sumQty) return;
    int sk = ps_suppkey[psIdx];
    if (sk < 0) return;
    uint w = (uint)sk >> 5;
    uint b = 1u << ((uint)sk & 31u);
    atomic_fetch_or_explicit(&qual_supp_bitmap[w], b, memory_order_relaxed);
}
)");
    {
        auto rscan = std::make_unique<MetalRangeScan>("q20_ht_slots", idx);
        rscan->addSideColumn("partsupp", "ps_availqty", "int");
        rscan->addSideColumn("partsupp", "ps_suppkey",  "int");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q20_unused", "int",
            "(q20_qual_emit(d_q20_ht_keys, d_q20_ht_vals, d_q20_ht_psidx, "
            "ps_availqty, ps_suppkey, d_q20_qual_supp_bitmap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q20_filter_ht_to_bitmap", std::move(sideEffect));
        // Re-bind HT buffers as read-only views after the phase barrier.
        phase.extraBuffers.push_back({"d_q20_ht_keys",          "ulong", true,  false});
        phase.extraBuffers.push_back({"d_q20_ht_vals",          "float", true,  false});
        phase.extraBuffers.push_back({"d_q20_ht_psidx",         "int",   true,  false});
        phase.extraBuffers.push_back({"d_q20_qual_supp_bitmap", "atomic_uint", false, true});
    }

    // --- Compact Results ---
    // Materialize qualifying CANADA suppliers on GPU.
    plan.helpers.push_back(R"(
static void q20_result_emit(device atomic_uint* counter,
                            device char* out_name,
                            device char* out_address,
                            const device atomic_uint* qual_supp_bitmap,
                            const device int* s_suppkey,
                            const device char* s_name,
                            const device char* s_address,
                            const device int* s_nationkey,
                            uint q20_result_cap,
                            int canada_nk,
                            uint i) {
    if (s_nationkey[i] != canada_nk) return;
    int sk = s_suppkey[i];
    if (sk < 0) return;
    if (!bitmap_test_atomic(qual_supp_bitmap, sk)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= q20_result_cap) return;
    for (uint c = 0; c < 25u; ++c) out_name[slot * 25u + c] = s_name[i * 25u + c];
    for (uint c = 0; c < 40u; ++c) out_address[slot * 40u + c] = s_address[i * 40u + c];
}
)");
    const std::string resultRows = "q20_result_rows";
    {
        auto scan = makeAutoScan("supplier", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_name", "char");
        scan->addColumn("s_address", "char");
        scan->addColumn("s_nationkey", "int");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_q20_result_unused", "int",
            "(q20_result_emit(d_q20_result_count, d_q20_result_name, "
            "d_q20_result_address, d_q20_qual_supp_bitmap, s_suppkey, "
            "s_name, s_address, s_nationkey, q20_result_cap, canada_nk, " +
            idx + "), 0)");
        auto& phase = appendPhase(plan, "Q20_materialize_suppliers", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q20_result_count",   "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q20_result_name",    "char",        false, false});
        phase.extraBuffers.push_back({"d_q20_result_address", "char",        false, false});
        phase.extraBuffers.push_back({"d_q20_qual_supp_bitmap", "atomic_uint", true, false});
        phase.scalarParams.push_back({"q20_result_cap", "uint"});
        phase.scalarParams.push_back({"canada_nk", "int"});
        attachMaterializedCountHook(phase, "d_q20_result_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("s_name", "d_q20_result_name", "char", 25),
            GenericMatColumnDesc("s_address", "d_q20_result_address", "char", 40),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"s_name", false});
        std::string sortError;
        appendGenericGpuSort(plan, "q20_result", resultRows,
                             "n_supplier", columns, sortSpec, &sortError);
    }

    return plan;
}

} // namespace codegen
