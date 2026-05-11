#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q13: Customer Distribution — 2 phases
// ===================================================================
std::optional<MetalQueryPlan> buildQ13Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: two-segment LIKE match for 'special...requests' in comment
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

    // Phase 1: Scan orders, filter NOT LIKE, count per custkey
    {
        auto scan = makeAutoScan("orders", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "!q13_comment_match(o_comment, " + idx + ")");

        auto count = std::make_unique<MetalAtomicCount>(
            std::move(filtered), "d_order_counts",
            "o_custkey[" + idx + "]", "maxCustkey");

        appendPhase(plan, "Q13_count_orders", std::move(count));
    }

    // Phase 2: Scan customers, read order count, build histogram
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

    return plan;
}

} // namespace codegen
