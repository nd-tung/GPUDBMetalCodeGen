#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q9: Product Type Profit Measure
// GPU preprocessing (4 phases): green-parts bitmap from `part`,
// s_nationkey lookup from `supplier`, o_year lookup from `orders`,
// (partkey,suppkey)→supplycost open-addressing HT from `partsupp`
// gated by the bitmap. Final phase scans `lineitem` and aggregates
// profit per (nation, year). Replaces ~110 lines of CPU preprocess.
// ===================================================================
std::optional<MetalQueryPlan> buildQ9Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Probe helper used by the final reduce phase (unchanged).
    plan.helpers.push_back(R"(
static float q9_ht_probe(const device uint* ht_keys, const device float* ht_vals,
                          uint ht_mask, uint key) {
    uint h = (key * 2654435769u) & ht_mask;
    for (uint step = 0; step < 64; step++) {
        uint slot = (h + step) & ht_mask;
        uint k = ht_keys[slot];
        if (k == key) return ht_vals[slot];
        if (k == 0xFFFFFFFFu) break;
    }
    return -1.0f;
}
)");

    // CAS-based insert helper for the partsupp HT build phase. Uses Knuth
    // multiplicative hash (kKnuthHashMul) — must match the host-side mask
    // computation and the q9_ht_probe constant above.
    plan.helpers.push_back(R"(
static void q9_ht_insert(device atomic_uint* ht_keys, device float* ht_vals,
                          uint ht_mask, uint key, float val) {
    uint h = (key * 2654435769u) & ht_mask;
    for (uint step = 0; step <= ht_mask; step++) {
        uint slot = (h + step) & ht_mask;
        uint expected = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(
                &ht_keys[slot], &expected, key,
                memory_order_relaxed, memory_order_relaxed)) {
            ht_vals[slot] = val;
            return;
        }
        // expected now holds the slot's current key; duplicate (pk,sk) pairs
        // do not occur for partsupp, but bail out defensively.
        if (expected == key) return;
    }
}
)");

    // Phase 0: build d_q9_part_bitmap from `part` (p_name contains "green").
    // Mirrors the CPU substring scan byte-for-byte over the 55-byte field.
    {
        auto scan = makeScan("part", idx, {{"p_partkey", "int"}, {"p_name", "char"}});
        std::string pred;
        for (int c = 0; c <= 50; c++) {
            std::string base = "p_name[" + idx + "*55+" + std::to_string(c);
            if (c > 0) pred += " || ";
            pred += "(" + base + "]=='g' && " +
                    base + "+1]=='r' && " +
                    base + "+2]=='e' && " +
                    base + "+3]=='e' && " +
                    base + "+4]=='n')";
        }
        auto filtered = std::make_unique<MetalSelection>(std::move(scan), pred);
        auto bmp = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_q9_part_bitmap",
            "p_partkey[" + idx + "]", "(maxPartkey + 31) / 32");
        appendPhase(plan, "Q9_build_part_bitmap", std::move(bmp));
    }

    // Phase 1: build d_q9_s_nationkey lookup from supplier (sentinel -1).
    {
        auto scan = makeScan("supplier", idx, {{"s_suppkey", "int"}, {"s_nationkey", "int"}});
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q9_s_nationkey",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey", /*fillByte=*/0xFF);
        appendPhase(plan, "Q9_build_s_nationkey", std::move(store));
    }

    // Phase 2: build d_q9_o_year lookup from orders (sentinel 0).
    // Reduce phase later guards against 0 via MetalArrayLookup(sentinel=0).
    {
        auto scan = makeScan("orders", idx, {{"o_orderkey", "int"}, {"o_orderdate", "int"}});
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q9_o_year",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "] / 10000",
            "int", "maxOrderkey", /*fillByte=*/0);
        appendPhase(plan, "Q9_build_o_year", std::move(store));
    }

    // Phase 3: build (pk,sk)→supplycost HT from partsupp gated by green bitmap.
    // Hand-emitted: needs CAS write semantics not provided by MetalArrayStore.
    {
        auto scan = makeScan("partsupp", idx, {
            {"ps_partkey", "int"}, {"ps_suppkey", "int"}, {"ps_supplycost", "float"}
        });
        auto gated = std::make_unique<MetalSelection>(std::move(scan),
            "bitmap_test(d_q9_part_bitmap, ps_partkey[" + idx + "])");
        auto computeKey = std::make_unique<MetalComputeExpr>(
            std::move(gated), "_psk", "uint",
            "(uint)ps_partkey[" + idx + "] * supp_mul + (uint)ps_suppkey[" + idx + "]");

        // Custom terminal: emits q9_ht_insert(...). We piggy-back on
        // MetalComputeExpr's child production to register bindings, then
        // append the call line ourselves via a passthrough terminal.
        // Implemented via a tiny inline operator below.
        struct HtInsertTerminal : MetalUnaryOperator {
            std::string idx_;
            HtInsertTerminal(std::unique_ptr<MetalOperator> c, std::string i)
                : MetalUnaryOperator(std::move(c)), idx_(std::move(i)) {}
            void produce(MetalCodegen& cg, ConsumerFn) override {
                // Build phase outputs.
                cg.addBufferParam("d_ps_ht_keys", "atomic_uint", "q9HtSize",
                                  /*zeroInit=*/true, /*fillByte=*/0xFF);
                cg.addBufferParam("d_ps_ht_vals", "float", "q9HtSize",
                                  /*zeroInit=*/true, /*fillByte=*/0);
                child_->produce(cg, [&]() {
                    cg.addLine(
                        "q9_ht_insert(d_ps_ht_keys, d_ps_ht_vals, d_ps_ht_mask, "
                        "_psk, ps_supplycost[" + idx_ + "]);");
                });
            }
            std::string describe() const override { return "Q9HtInsert"; }
        };
        auto term = std::make_unique<HtInsertTerminal>(std::move(computeKey), idx);

        auto& phase = appendPhase(plan, "Q9_build_ps_ht", std::move(term));
        phase.bitmapReads.push_back({"d_q9_part_bitmap", ""});
        phase.scalarParams.push_back({"d_ps_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    // Phase 4 (existing reduce): scan lineitem, probe HT, accumulate profit
    // per (nation, year) bucket.
    {
        auto scan = makeScan("lineitem", idx, {
            {"l_partkey", "int"}, {"l_suppkey", "int"}, {"l_orderkey", "int"},
            {"l_quantity", "float"}, {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        // BitmapProbe: filter to green parts
        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q9_part_bitmap", "l_partkey[" + idx + "]");

        // ArrayLookup: s_nationkey[suppkey]
        auto natLookup = std::make_unique<MetalArrayLookup>(
            std::move(bmpProbe), "d_q9_s_nationkey",
            "l_suppkey[" + idx + "]", "_nationkey", "int", -1);

        // ArrayLookup: o_year[orderkey]
        auto yearLookup = std::make_unique<MetalArrayLookup>(
            std::move(natLookup), "d_q9_o_year",
            "l_orderkey[" + idx + "]", "_year", "int", 0);

        // ComputeExpr: hash probe for supplycost
        std::string htProbeExpr =
            "q9_ht_probe(d_ps_ht_keys, d_ps_ht_vals, d_ps_ht_mask, "
            "(uint)l_partkey[" + idx + "] * supp_mul + (uint)l_suppkey[" + idx + "])";
        auto computeSC = std::make_unique<MetalComputeExpr>(
            std::move(yearLookup), "_sc", "float", htProbeExpr);

        // Selection: supplycost found (>= 0)
        auto scFilter = std::make_unique<MetalSelection>(
            std::move(computeSC), "_sc >= 0.0f");

        // ComputeExpr: profit and bin
        std::string profitExpr =
            "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "]) - _sc * l_quantity[" + idx + "]";
        auto computeProfit = std::make_unique<MetalComputeExpr>(
            std::move(scFilter), "_profit", "float", profitExpr);

        // Bin = nationkey * 8 + (year - 1992)
        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(computeProfit), "_bin", "int", "_nationkey * 8 + (_year - 1992)");

        // AtomicAgg: accumulate profit per bin
        // 25 nations × 8 year slots = 200 bins
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(computeBin), "d_q9_profit",
            "_bin", "_profit", "200",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q9_profit_reduce", std::move(agg));
        // Extra buffers: hash table keys, values, and mask scalar
        phase.extraBuffers.push_back({"d_ps_ht_keys", "uint", true});
        phase.extraBuffers.push_back({"d_ps_ht_vals", "float", true});
        phase.scalarParams.push_back({"d_ps_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    return plan;
}

} // namespace codegen
