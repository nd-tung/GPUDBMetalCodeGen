#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"
#include "execution/metal_generic_executor.h"

#include <cstdint>

namespace codegen {

// Q22: Global Sales Opportunity.
std::optional<MetalQueryPlan> buildQ22Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Average Balance ---
    // Compute count and sum in separate kernels. Keeping the global float sum
    // alone avoids mixing it with threadgroup count atomics in the hot pass.
    {
        auto scan = makeAutoScan("customer", idx);

        auto computePrefix = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_prefix", "int",
            "(c_phone[" + idx + " * 15] - '0') * 10 + (c_phone[" + idx + " * 15 + 1] - '0')");

        std::string validPrefixCond =
            "(_prefix == 13 || _prefix == 17 || _prefix == 18 || "
            "_prefix == 23 || _prefix == 29 || _prefix == 30 || _prefix == 31) && "
            "c_acctbal[" + idx + "] > 0.0f";
        auto filtered = std::make_unique<MetalSelection>(std::move(computePrefix), validPrefixCond);

        auto countOp = std::make_unique<MetalAtomicCount>(
            std::move(filtered), "d_q22_avgbal_count", "0", "1");

        appendPhase(plan, "Q22_compute_avg_bal_count", std::move(countOp), 256);
    }
    {
        auto scan = makeAutoScan("customer", idx);

        auto computePrefix = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_prefix", "int",
            "(c_phone[" + idx + " * 15] - '0') * 10 + (c_phone[" + idx + " * 15 + 1] - '0')");

        std::string validPrefixCond =
            "(_prefix == 13 || _prefix == 17 || _prefix == 18 || "
            "_prefix == 23 || _prefix == 29 || _prefix == 30 || _prefix == 31) && "
            "c_acctbal[" + idx + "] > 0.0f";
        auto filtered = std::make_unique<MetalSelection>(std::move(computePrefix), validPrefixCond);

        auto sumOp = std::make_unique<MetalAtomicAgg>(
            std::move(filtered), "d_q22_avgbal_sum",
            "0", "c_acctbal[" + idx + "]", "1",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q22_compute_avg_bal_sum", std::move(sumOp), 256);
        phase.postDispatchHook = [](MetalGenericExecutor& ex) {
            auto* sumBuf   = ex.getAllocatedBuffer("d_q22_avgbal_sum");
            auto* countBuf = ex.getAllocatedBuffer("d_q22_avgbal_count");
            float avg = 0.0f;
            if (sumBuf && countBuf) {
                float sum = *static_cast<float*>(sumBuf->contents());
                uint32_t cnt = *static_cast<uint32_t*>(countBuf->contents());
                if (cnt > 0) avg = sum / static_cast<float>(cnt);
            }
            ex.registerScalarFloat("avg_bal", avg);
            return 0.0;
        };
    }

    // --- Order Existence Bitmap ---
    // Build customer-with-orders bitmap.
    {
        auto scan = makeAutoScan("orders", idx);

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(scan), "d_cust_order_bitmap",
            "o_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        appendPhase(plan, "Q22_build_bitmap", std::move(bitmap));
    }

    // --- Country-Code Aggregate ---
    // Aggregate customers above avg_bal with no orders.
    {
        auto scan = makeAutoScan("customer", idx);

        auto computePrefix = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_prefix", "int",
            "(c_phone[" + idx + " * 15] - '0') * 10 + (c_phone[" + idx + " * 15 + 1] - '0')");

        std::string validPrefixCond =
            "(_prefix == 13 || _prefix == 17 || _prefix == 18 || "
            "_prefix == 23 || _prefix == 29 || _prefix == 30 || _prefix == 31) && "
            "c_acctbal[" + idx + "] > avg_bal";
        auto filtered = std::make_unique<MetalSelection>(std::move(computePrefix), validPrefixCond);

        auto antiProbed = std::make_unique<MetalAntiBitmapProbe>(
            std::move(filtered), "d_cust_order_bitmap",
            "c_custkey[" + idx + "]");

        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(antiProbed), "_bin", "int",
            "(_prefix == 13 ? 0 : _prefix == 17 ? 1 : _prefix == 18 ? 2 : "
            "_prefix == 23 ? 3 : _prefix == 29 ? 4 : _prefix == 30 ? 5 : 6)");

        // Seven bins correspond to the accepted country-code prefixes. Use a
        // keyed aggregate so both count and balance sum are reduced within
        // each threadgroup before the global atomics.
        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(computeBin), "d_q22_aggs", "_bin",
            /*numBuckets=*/7, /*valuesPerBucket=*/2, "14");
        agg->addAggregate("numcust", 0, "1u", "add", false, 0);
        agg->addAggregateWithMeta("totacctbal", 1, "c_acctbal[" + idx + "]",
                                  "add", false, 0,
                                  true, false);

        auto& phase = appendPhase(plan, "Q22_final_aggregate", std::move(agg), 256);
        phase.scalarParams = {{"avg_bal", "float"}};
    }

    return plan;
}

} // namespace codegen
