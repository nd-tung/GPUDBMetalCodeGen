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
    // Compute avg_bal, then register it for the final aggregate phase.
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

        auto sumOp = std::make_unique<MetalAtomicAgg>(
            std::move(countOp), "d_q22_avgbal_sum",
            "0", "c_acctbal[" + idx + "]", "1",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q22_compute_avg_bal", std::move(sumOp), 256);
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

        // Seven bins correspond to the accepted country-code prefixes.
        auto count = std::make_unique<MetalAtomicCount>(
            std::move(computeBin), "d_q22_count", "_bin", "7");

        auto sum = std::make_unique<MetalAtomicAgg>(
            std::move(count), "d_q22_sum",
            "_bin", "c_acctbal[" + idx + "]", "7",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q22_final_aggregate", std::move(sum), 256);
        phase.scalarParams = {{"avg_bal", "float"}};
    }

    return plan;
}

} // namespace codegen
