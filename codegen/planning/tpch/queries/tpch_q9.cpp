#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q9: Product Type Profit Measure.
std::optional<MetalQueryPlan> buildQ9Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Hash Helpers ---
    // Probe helper used by the final reduce phase.
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

    // Hash constants must match q9_ht_probe and host sizing.
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
        // Duplicate (pk, sk) pairs should not occur, but exit defensively.
        if (expected == key) return;
    }
}
)");

    // --- Part Filter ---
    // Build green-part bitmap.
    {
        auto scan = makeAutoScan("part", idx);
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

    // --- Lookup Maps ---
    // Build supplier nation lookup.
    {
        auto scan = makeAutoScan("supplier", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q9_s_nationkey",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey", /*fillByte=*/0xFF);
        appendPhase(plan, "Q9_build_s_nationkey", std::move(store));
    }

    // Build nation key -> row lookup for final GPU name materialization.
    {
        auto scan = makeAutoScan("nation", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q9_nation_idx",
            "n_nationkey[" + idx + "]", "(int)" + idx,
            "int", "25", /*fillByte=*/0xFF);
        appendPhase(plan, "Q9_build_nation_idx", std::move(store), 32);
    }

    // Build order year lookup.
    {
        auto scan = makeAutoScan("orders", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q9_o_year",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "] / 10000",
            "int", "maxOrderkey", /*fillByte=*/0);
        appendPhase(plan, "Q9_build_o_year", std::move(store));
    }

    // --- Partsupp Hash Table ---
    // Build (pk, sk) to supply-cost hash table.
    {
        auto scan = makeAutoScan("partsupp", idx);
        auto gated = std::make_unique<MetalSelection>(std::move(scan),
            "bitmap_test_atomic(d_q9_part_bitmap, ps_partkey[" + idx + "])");
        auto computeKey = std::make_unique<MetalComputeExpr>(
            std::move(gated), "_psk", "uint",
            "(uint)ps_partkey[" + idx + "] * supp_mul + (uint)ps_suppkey[" + idx + "]");

        // Custom terminal emits q9_ht_insert after child bindings are registered.
        struct HtInsertTerminal : MetalUnaryOperator {
            std::string idx_;
            HtInsertTerminal(std::unique_ptr<MetalOperator> c, std::string i)
                : MetalUnaryOperator(std::move(c)), idx_(std::move(i)) {}
            void iusUsed(std::vector<IU>& out) const override {
                MetalOperator::appendIUsFromExpr("ps_supplycost[" + idx_ + "]", out);
            }
            void produce(MetalCodegen& cg, ConsumerFn) override {
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

    // --- Profit Aggregate ---
    // Aggregate profit per nation/year bucket.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q9_part_bitmap", "l_partkey[" + idx + "]");

        auto natLookup = std::make_unique<MetalArrayLookup>(
            std::move(bmpProbe), "d_q9_s_nationkey",
            "l_suppkey[" + idx + "]", "_nationkey", "int", -1);

        auto yearLookup = std::make_unique<MetalArrayLookup>(
            std::move(natLookup), "d_q9_o_year",
            "l_orderkey[" + idx + "]", "_year", "int", 0);

        std::string htProbeExpr =
            "q9_ht_probe(d_ps_ht_keys, d_ps_ht_vals, d_ps_ht_mask, "
            "(uint)l_partkey[" + idx + "] * supp_mul + (uint)l_suppkey[" + idx + "])";
        auto computeSC = std::make_unique<MetalComputeExpr>(
            std::move(yearLookup), "_sc", "float", htProbeExpr);

        auto scFilter = std::make_unique<MetalSelection>(
            std::move(computeSC), "_sc >= 0.0f");

        std::string profitExpr =
            "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "]) - _sc * l_quantity[" + idx + "]";
        auto computeProfit = std::make_unique<MetalComputeExpr>(
            std::move(scFilter), "_profit", "float", profitExpr);

        // 200 bins cover 25 nations across eight order years.
        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(computeProfit), "_bin", "int", "_nationkey * 8 + (_year - 1992)");

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(computeBin), "d_q9_profit",
            "_bin", "_profit", "200",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q9_profit_reduce", std::move(agg));
        phase.extraBuffers.push_back({"d_ps_ht_keys", "uint", true});
        phase.extraBuffers.push_back({"d_ps_ht_vals", "float", true});
        phase.scalarParams.push_back({"d_ps_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
    }

    plan.helpers.push_back(R"(
static void q9_emit_profit_result(device atomic_uint* counter,
                                  device char* out_nation,
                                  device int* out_year,
                                  device float* out_profit,
                                  const device float* profit_bins,
                                  const device int* nation_idx,
                                  const device char* n_name,
                                  uint cap, uint bin) {
    uint yr_off = bin & 7u;
    if (yr_off >= 7u) return;
    uint nk = bin >> 3u;
    float profit = profit_bins[bin];
    if (profit == 0.0f) return;
    int nr = nation_idx[nk];
    if (nr < 0) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        out_year[slot] = (int)(1992u + yr_off);
        out_profit[slot] = profit;
        for (uint c = 0; c < 25u; ++c)
            out_nation[slot * 25u + c] = n_name[(uint)nr * 25u + c];
    }
}
)");

    // Sort the finite nation/year domain; the materialized counter still controls emitted rows.
    const std::string resultRows = "q9_result_cap";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q9_profit_bins", idx);
        struct Q9CompactTerminal : MetalUnaryOperator {
            std::string idx_;
            Q9CompactTerminal(std::unique_ptr<MetalOperator> child, std::string idx)
                : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                cg.addColumnParam("n_name", "char", "nation");
                child_->produce(cg, [&]() {
                    cg.addLine("q9_emit_profit_result(d_q9_result_count, "
                               "d_q9_result_nation, d_q9_result_year, "
                               "d_q9_result_profit, d_q9_profit, "
                               "d_q9_nation_idx, n_name, q9_result_cap, " +
                               idx_ + ");");
                });
                consume();
            }
            std::string describe() const override { return "Q9CompactResult"; }
        };
        auto compact = std::make_unique<Q9CompactTerminal>(std::move(rscan), idx);
        auto& phase = appendPhase(plan, "Q9_compact_results", std::move(compact), 64);
        phase.extraBuffers.push_back({"d_q9_result_count",  "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q9_result_nation", "char",        false, true});
        phase.extraBuffers.push_back({"d_q9_result_year",   "int",         false, true});
        phase.extraBuffers.push_back({"d_q9_result_profit", "float",       false, true});
        phase.extraBuffers.push_back({"d_q9_profit",        "float",       true,  false});
        phase.extraBuffers.push_back({"d_q9_nation_idx",    "int",         true,  false});
        phase.scalarParams.push_back({"q9_result_cap", "uint"});
    }

    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("nation", "d_q9_result_nation", "char", 25),
            GenericMatColumnDesc("o_year", "d_q9_result_year", "int"),
            GenericMatColumnDesc("sum_profit", "d_q9_result_profit", "float"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"nation", false});
        sortSpec.keys.push_back({"o_year", true});
        std::string sortError;
        if (!appendGenericGpuSmallSort(plan, "q9_result", resultRows,
                                       256, columns, sortSpec, &sortError)) {
            appendGenericGpuSort(plan, "q9_result", resultRows,
                                 "q9_result_cap", columns, sortSpec, &sortError);
        }
    }

    return plan;
}

} // namespace codegen
