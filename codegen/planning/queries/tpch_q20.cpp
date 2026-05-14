#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q20: Potential Part Promotion
// Phase 0 (GPU): Scan part → build forest% bitmap on `d_q20_part_bitmap`.
// Phase 1 (GPU): Scan partsupp gated by bitmap → CAS-insert
//                (key=pk*supp_mul+sk, ps_idx) into 64-bit HT and zero-init
//                d_q20_ht_vals; the row's partsupp index is recorded so
//                CPU post can read availqty.
// Phase 2 (GPU): Scan lineitem (date filter 1994) → bitmap probe →
//                hash probe (q20_ht_add) → atomic_float add into ht_vals.
// CPU pre: tiny CANADA-nation lookup + supplier/partsupp mirrors for post;
//          q20HtSize/d_q20_ht_mask/supp_mul scalars.
// CPU post: walk HT slots reading d_q20_ht_keys/_vals/_psidx (all GPU
//           shared buffers) and check availqty > 0.5 * sum_qty against
//           ps_availqty mirror; filter CANADA suppliers.
// ===================================================================
std::optional<MetalQueryPlan> buildQ20Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: hash probe + atomic add (read-only key view, used in agg phase).
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
        if (k == 0xFFFFFFFFFFFFFFFFul) return; // not in HT = not qualifying partsupp
    }
}
)");

    // Helper: CAS-based insert for the build phase. Metal M1 does not
    // support 64-bit atomic CAS, so we use atomic_int CAS on the ps_idx
    // slot (-1 sentinel) for ownership and write the 64-bit key
    // non-atomically afterwards. The agg phase that probes ht_keys runs
    // in a separate command buffer (waitUntilCompleted barrier), so
    // build-phase writes are fully visible at probe time.
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
        // Slot taken by another insert. partsupp rows have unique (pk,sk)
        // so duplicates cannot occur; just probe the next slot.
    }
}
)");

    // Phase 0: forest% bitmap from `part`.
    {
        auto scan = makeAutoScan("part", idx);
        // p_name is fixed-width 55 chars; "forest" check at offset 0 of each row.
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

    // Phase 1: build (pk,sk)→ps_idx HT from partsupp gated by bitmap.
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
                // Allocate ht_vals here so the agg phase's extraBuffer
                // (empty sizeExpr) finds an existing buffer. zero-init.
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

    // Phase 2: lineitem aggregation (existing logic).
    {
        auto scan = makeAutoScan("lineitem", idx);

        // Date filter: 1994-01-01 to 1994-12-31
        auto dateFilter = std::make_unique<MetalSelection>(
            std::move(scan),
            "l_shipdate[" + idx + "] >= 19940101 && l_shipdate[" + idx + "] < 19950101");

        // BitmapProbe: forest% parts
        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(dateFilter), "d_q20_part_bitmap", "l_partkey[" + idx + "]");

        // ComputeExpr: hash probe + atomic add (side-effect only)
        auto hashAgg = std::make_unique<MetalComputeExpr>(
            std::move(bmpProbe), "_unused", "int",
            "(q20_ht_add(d_q20_ht_keys, d_q20_ht_vals, d_q20_ht_mask, "
            "(ulong)l_partkey[" + idx + "] * (ulong)supp_mul + (ulong)l_suppkey[" + idx + "], "
            "l_quantity[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q20_lineitem_agg", std::move(hashAgg));
        // Extra buffers: HT keys read-only; vals as atomic_float for adds.
        phase.extraBuffers.push_back({"d_q20_ht_keys", "ulong", true, false});
        phase.extraBuffers.push_back({"d_q20_ht_vals", "atomic_float", false, false});
        phase.scalarParams.push_back({"d_q20_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    // Phase 3: GPU-build the qualifying-supplier bitmap. Range scan
    // over [0, n_q20_ht_slots) reads ht_keys/ht_vals/ht_psidx and
    // ps_availqty/ps_suppkey, then atomic-OR sets a bit for each
    // qualifying suppkey. Replaces the q20HtSize-row CPU scan + std::set
    // build that previously dominated Q20 post.
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
        // Re-bind the same MTL::Buffer for ht_keys/ht_vals/ht_psidx
        // under their original names but as non-atomic read-only types
        // (atomic-backed storage is bit-identical to its plain twin, and
        // a queue barrier between phases makes the build-phase writes
        // visible).
        phase.extraBuffers.push_back({"d_q20_ht_keys",          "ulong", true,  false});
        phase.extraBuffers.push_back({"d_q20_ht_vals",          "float", true,  false});
        phase.extraBuffers.push_back({"d_q20_ht_psidx",         "int",   true,  false});
        phase.extraBuffers.push_back({"d_q20_qual_supp_bitmap", "atomic_uint", false, true});
    }

    return plan;
}

// ===================================================================
// Dispatch: try all known patterns

} // namespace codegen
