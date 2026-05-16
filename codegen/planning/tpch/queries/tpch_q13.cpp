#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q13: Customer Distribution.
std::optional<MetalQueryPlan> buildQ13Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Match comments containing "special" before "requests".
    plan.helpers.push_back(R"(
static bool q13_comment_match(const device char* comment, uint idx) {
    const device char* c = comment + idx * 79;
    for (int p = 0; p <= 72 && c[p] != '\0'; p++) {
        if (c[p]=='s' && c[p+1]=='p' && c[p+2]=='e' && c[p+3]=='c' &&
            c[p+4]=='i' && c[p+5]=='a' && c[p+6]=='l') {
            for (int q = p + 7; q <= 71 && c[q] != '\0'; q++) {
                if (c[q]=='r' && c[q+1]=='e' && c[q+2]=='q' && c[q+3]=='u' &&
                    c[q+4]=='e' && c[q+5]=='s' && c[q+6]=='t' && c[q+7]=='s') {
                    return true;
                }
            }
            break;
        }
    }
    return false;
}
)");

    // Count qualifying orders per customer.
    {
        auto scan = makeAutoScan("orders", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "!q13_comment_match(o_comment, " + idx + ")");

        auto count = std::make_unique<MetalAtomicCount>(
            std::move(filtered), "d_order_counts",
            "o_custkey[" + idx + "]", "maxCustkey");

        appendPhase(plan, "Q13_count_orders", std::move(count));
    }

    // Build customer-count histogram.
    {
        auto scan = makeAutoScan("customer", idx);

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_order_counts",
            "c_custkey[" + idx + "]",
            "_cnt", "int", 0x7FFFFFFF);

        auto hist = std::make_unique<MetalAtomicCount>(
            std::move(lookup), "d_histogram",
            "_cnt", "256");

        appendPhase(plan, "Q13_build_histogram", std::move(hist), 256);
    }

    // Materialize nonzero bins for GPU sort.
    plan.helpers.push_back(R"(
static void q13_hist_emit(device atomic_uint* counter,
                          device uint* out_c_count,
                          device uint* out_custdist,
                          const device uint* d_histogram,
                          uint q13_hist_cap,
                          uint bin) {
    uint dist = d_histogram[bin];
    if (dist == 0u) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= q13_hist_cap) return;
    out_c_count[slot] = bin;
    out_custdist[slot] = dist;
}
)");
    const std::string resultRows = "q13_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q13_hist_bins", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q13_hist_unused", "int",
            "(q13_hist_emit(d_q13_result_count, d_q13_result_c_count, "
            "d_q13_result_custdist, d_histogram, q13_hist_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q13_materialize_histogram", std::move(sideEffect), 256);
        phase.extraBuffers.push_back({"d_q13_result_count",    "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q13_result_c_count",  "uint",        false, false});
        phase.extraBuffers.push_back({"d_q13_result_custdist", "uint",        false, false});
        phase.extraBuffers.push_back({"d_histogram",           "uint",        true,  false});
        phase.scalarParams.push_back({"q13_hist_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q13_result_count", resultRows);
    }

    // --- Result Order ---
    // Sort compact histogram bins by distribution, then count.
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("c_count", "d_q13_result_c_count", "uint"),
            GenericMatColumnDesc("custdist", "d_q13_result_custdist", "uint"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"custdist", true});
        sortSpec.keys.push_back({"c_count", true});
        std::string sortError;
        if (!appendGenericGpuSmallSort(plan, "q13_result", resultRows,
                                       256, columns, sortSpec, &sortError)) {
            appendGenericGpuSort(plan, "q13_result", resultRows,
                                 "256", columns, sortSpec, &sortError);
        }
    }

    return plan;
}

} // namespace codegen
