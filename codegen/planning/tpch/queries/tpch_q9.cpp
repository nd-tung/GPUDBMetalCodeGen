#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q9: Product Type Profit Measure.
std::optional<MetalQueryPlan> buildQ9Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Partsupp Lookup Helpers ---
    // TPC-H partsupp has four supplier rows per part.  Keep only green-part
    // rows in a compact direct-address side table instead of a global hash.
    plan.helpers.push_back(R"(
static int q9_ps_lookup(const device uint* ps_counts,
                        const device uint* ps_supps,
                        const device uint* ps_costs,
                        uint partkey,
                        uint suppkey) {
    uint n = ps_counts[partkey];
    if (n > 4u) n = 4u;
    ulong base = (ulong)partkey * 4ul;
    for (uint j = 0u; j < n; ++j) {
        ulong off = base + (ulong)j;
        if (ps_supps[off] == suppkey) return (int)ps_costs[off];
    }
    return -1;
}

static long q9_profit_scaled(float extendedprice, float discount,
                             float quantity, int supplycost_cents) {
    long extended_cents = (long)round(extendedprice * 100.0f);
    long discount_basis = (long)round(discount * 100.0f);
    long quantity_cents = (long)round(quantity * 100.0f);
    return extended_cents * (100L - discount_basis) -
           (long)supplycost_cents * quantity_cents;
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

    // --- Partsupp Direct Map ---
    // Build partkey -> up to four (suppkey, supply-cost-cents) pairs.
    {
        auto scan = makeAutoScan("partsupp", idx);
        auto gated = std::make_unique<MetalSelection>(std::move(scan),
            "bitmap_test_atomic(d_q9_part_bitmap, ps_partkey[" + idx + "])");

        struct PsStoreTerminal : MetalUnaryOperator {
            std::string idx_;
            PsStoreTerminal(std::unique_ptr<MetalOperator> c, std::string i)
                : MetalUnaryOperator(std::move(c)), idx_(std::move(i)) {}
            void iusUsed(std::vector<IU>& out) const override {
                MetalOperator::appendIUsFromExpr("ps_partkey[" + idx_ + "]", out);
                MetalOperator::appendIUsFromExpr("ps_suppkey[" + idx_ + "]", out);
                MetalOperator::appendIUsFromExpr("ps_supplycost[" + idx_ + "]", out);
            }
            void produce(MetalCodegen& cg, ConsumerFn) override {
                cg.addBufferParam("d_q9_ps_count", "atomic_uint", "maxPartkey",
                                  /*zeroInit=*/true, /*fillByte=*/0);
                cg.addBufferParam("d_q9_ps_supp", "uint", "q9_ps_slots",
                                  /*zeroInit=*/false, /*fillByte=*/0);
                cg.addBufferParam("d_q9_ps_cost", "uint", "q9_ps_slots",
                                  /*zeroInit=*/false, /*fillByte=*/0);
                child_->produce(cg, [&]() {
                    cg.addLine("uint _q9_pk = (uint)ps_partkey[" + idx_ + "];");
                    cg.addLine("uint _q9_slot = atomic_fetch_add_explicit("
                               "&d_q9_ps_count[_q9_pk], 1u, memory_order_relaxed);");
                    cg.addIf("_q9_slot < 4u", [&]() {
                        cg.addLine("ulong _q9_off = (ulong)_q9_pk * 4ul + (ulong)_q9_slot;");
                        cg.addLine("d_q9_ps_supp[_q9_off] = (uint)ps_suppkey[" + idx_ + "];");
                        cg.addLine("d_q9_ps_cost[_q9_off] = "
                                   "(uint)round(ps_supplycost[" + idx_ + "] * 100.0f);");
                    });
                });
            }
            std::string describe() const override { return "Q9PartsuppStore"; }
        };
        auto term = std::make_unique<PsStoreTerminal>(std::move(gated), idx);

        auto& phase = appendPhase(plan, "Q9_build_ps_map", std::move(term));
        phase.bitmapReads.push_back({"d_q9_part_bitmap", ""});
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

        std::string psProbeExpr =
            "q9_ps_lookup(d_q9_ps_count, d_q9_ps_supp, d_q9_ps_cost, "
            "(uint)l_partkey[" + idx + "], "
            "(uint)l_suppkey[" + idx + "])";
        auto computeSC = std::make_unique<MetalComputeExpr>(
            std::move(yearLookup), "_sc", "int", psProbeExpr);

        auto scFilter = std::make_unique<MetalSelection>(
            std::move(computeSC), "_sc >= 0");

        std::string profitExpr =
            "q9_profit_scaled(l_extendedprice[" + idx + "], l_discount[" +
            idx + "], l_quantity[" + idx + "], _sc)";
        auto computeProfit = std::make_unique<MetalComputeExpr>(
            std::move(scFilter), "_profit", "long", profitExpr);

        // 200 bins cover 25 nations across eight order years.
        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(computeProfit), "_bin", "int", "_nationkey * 8 + (_year - 1992)");

        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(computeBin), "d_q9_profit", "_bin",
            /*numBuckets=*/200, /*valuesPerBucket=*/2, "400");
        agg->addAggregate("sum_profit", 0, "_profit", "add", true, 10000);

        auto& phase = appendPhase(plan, "Q9_profit_reduce", std::move(agg));
        phase.extraBuffers.push_back({"d_q9_ps_count", "uint", true});
        phase.extraBuffers.push_back({"d_q9_ps_supp",  "uint", true});
        phase.extraBuffers.push_back({"d_q9_ps_cost",  "uint", true});
    }

    plan.helpers.push_back(R"(
static void q9_emit_profit_result(device atomic_uint* counter,
                                  device char* out_nation,
                                  device int* out_year,
                                  device float* out_profit,
                                  const device uint* profit_bins,
                                  const device int* nation_idx,
                                  const device char* n_name,
                                  uint cap, uint bin) {
    uint yr_off = bin & 7u;
    if (yr_off >= 7u) return;
    uint nk = bin >> 3u;
    long profit_cents = load_long_pair(&profit_bins[bin * 2u], &profit_bins[bin * 2u + 1u]);
    if (profit_cents == 0) return;
    float profit = (float)profit_cents / 10000.0f;
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
        phase.extraBuffers.push_back({"d_q9_profit",        "uint",        true,  false});
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
            if (!appendGenericGpuSort(plan, "q9_result", resultRows,
                                      "q9_result_cap", columns, sortSpec,
                                      &sortError)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}

} // namespace codegen
