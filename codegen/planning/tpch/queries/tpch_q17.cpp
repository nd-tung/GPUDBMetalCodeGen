#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q17: Small-Quantity-Order Revenue.
std::optional<MetalQueryPlan> buildQ17Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Part Filter ---
    // Build qualifying-parts bitmap.
    {
        auto scan = makeAutoScan("part", idx);

        std::string pred =
            "p_brand[" + idx + "*10]=='B' && p_brand[" + idx + "*10+5]=='#' && "
            "p_brand[" + idx + "*10+6]=='2' && p_brand[" + idx + "*10+7]=='3' && "
            "p_container[" + idx + "*10]=='M' && p_container[" + idx + "*10+1]=='E' && "
            "p_container[" + idx + "*10+2]=='D' && p_container[" + idx + "*10+3]==' ' && "
            "p_container[" + idx + "*10+4]=='B' && p_container[" + idx + "*10+5]=='O' && "
            "p_container[" + idx + "*10+6]=='X'";

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), pred);
        auto bmp = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_q17_bitmap",
            "p_partkey[" + idx + "]", "(maxPartkey + 31) / 32");
        appendPhase(plan, "Q17_build_bitmap", std::move(bmp));
    }

    // --- Average Quantity Inputs ---
    // Accumulate per-part quantity sum and count.
    {
        auto scan = makeAutoScan("lineitem", idx);
        auto gated = std::make_unique<MetalSelection>(std::move(scan),
            "bitmap_test_atomic(d_q17_bitmap, l_partkey[" + idx + "])");
        auto cnt = std::make_unique<MetalAtomicCount>(
            std::move(gated), "d_q17_cntQty",
            "l_partkey[" + idx + "]", "maxPartkey");
        auto sumOp = std::make_unique<MetalAtomicAgg>(
            std::move(cnt), "d_q17_sumQty",
            "l_partkey[" + idx + "]", "l_quantity[" + idx + "]",
            "maxPartkey", "atomic_float", "float");
        auto& phase = appendPhase(plan, "Q17_build_avg_qty", std::move(sumOp));
        phase.bitmapReads.push_back({"d_q17_bitmap", ""});
    }

    // --- Revenue Reduction ---
    // Reduce revenue using the inline average-quantity threshold.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto bitmapFilter = std::make_unique<MetalSelection>(
            std::move(scan),
            "bitmap_test_atomic(d_q17_bitmap, l_partkey[" + idx + "])");

        auto loadCnt = std::make_unique<MetalComputeExpr>(
            std::move(bitmapFilter), "_cnt", "uint",
            "d_q17_cntQty[l_partkey[" + idx + "]]");

        auto cntFilt = std::make_unique<MetalSelection>(std::move(loadCnt), "_cnt > 0u");

        auto loadSum = std::make_unique<MetalComputeExpr>(
            std::move(cntFilt), "_sum", "float",
            "d_q17_sumQty[l_partkey[" + idx + "]]");

        auto thrFilt = std::make_unique<MetalSelection>(
            std::move(loadSum),
            "l_quantity[" + idx + "] * (float)_cnt < 0.2f * _sum");

        auto reduce = std::make_unique<MetalTGReduce>(std::move(thrFilt), "d_q17");
        reduce->addAccumulator("revenue", "l_extendedprice[" + idx + "]", "float");
        reduce->setResultAlias("avg_yearly", 7);

        auto& phase = appendPhase(plan, "Q17_reduce", std::move(reduce));
        phase.bitmapReads.push_back({"d_q17_bitmap", ""});
        phase.extraBuffers.push_back({"d_q17_sumQty", "float", true, false});
        phase.extraBuffers.push_back({"d_q17_cntQty", "uint",  true, false});
    }

    return plan;
}

} // namespace codegen
