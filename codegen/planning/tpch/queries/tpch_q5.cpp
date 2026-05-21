#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q5: Local Supplier Volume.
std::optional<MetalQueryPlan> buildQ5Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Region Filter ---
    // Build ASIA nation bitmap.
    {
        auto scan = makeAutoScan("nation", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "n_regionkey[" + idx + "] == asia_rk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_nation_bitmap",
            "n_nationkey[" + idx + "]", "(25 + 31) / 32");

        auto& phase = appendPhase(plan, "Q5_build_nation_bitmap", std::move(bitmap), 32);
        phase.scalarParams = {{"asia_rk", "int"}};
    }

    // --- Entity Maps ---
    // Build customer nation map for ASIA customers.
    {
        auto scan = makeAutoScan("customer", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "c_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        appendPhase(plan, "Q5_build_cust_map", std::move(store), 256);
    }

    // Build supplier nation map for ASIA suppliers.
    {
        auto scan = makeAutoScan("supplier", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "s_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        appendPhase(plan, "Q5_build_supp_map", std::move(store), 256);
    }

    // --- Orders Map ---
    // Build date-filtered orders nation map.
    {
        auto scan = makeAutoScan("orders", idx);

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] >= 19940101 && o_orderdate[" + idx + "] < 19950101");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_cust_nation_map",
            "o_custkey[" + idx + "]",
            "_cust_nk", "int", -1);

        auto store = std::make_unique<MetalArrayStore>(
            std::move(lookup), "d_orders_nation_map",
            "o_orderkey[" + idx + "]", "_cust_nk",
            "int", "maxOrderkey");

        appendPhase(plan, "Q5_build_orders_map", std::move(store));
    }

    // --- Revenue Aggregate ---
    // Aggregate same-nation revenue.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_orders_nation_map",
            "l_orderkey[" + idx + "]",
            "_cust_nk", "int", -1);

        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        auto sameNation = std::make_unique<MetalSelection>(std::move(lookupSupp),
            "_cust_nk == _supp_nk");

        std::string valueExpr = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(sameNation), "d_nation_revenue",
            "_cust_nk", valueExpr, "25",
            "atomic_float", "float");

        appendPhase(plan, "Q5_probe_aggregate", std::move(agg));
    }

    plan.helpers.push_back(R"(
static void q5_emit_nation_result(device atomic_uint* counter,
                                  device char* out_name,
                                  device float* out_revenue,
                                  const device float* nation_revenue,
                                  const device char* n_name,
                                  uint cap, uint nk, uint row) {
    float revenue = nation_revenue[nk];
    if (revenue <= 0.0f) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        out_revenue[slot] = revenue;
        for (uint c = 0; c < 25u; ++c)
            out_name[slot * 25u + c] = n_name[row * 25u + c];
    }
}
)");

    // Sort the finite nation domain; the materialized counter still controls emitted rows.
    const std::string resultRows = "q5_result_cap";
    {
        auto scan = makeScan("nation", idx, {
            {"n_nationkey", "int"},
            {"n_name", "char"},
        });
        struct Q5CompactTerminal : MetalUnaryOperator {
            std::string idx_;
            Q5CompactTerminal(std::unique_ptr<MetalOperator> child, std::string idx)
                : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                child_->produce(cg, [&]() {
                    cg.addLine("q5_emit_nation_result(d_q5_result_count, "
                               "d_q5_result_name, d_q5_result_revenue, "
                               "d_nation_revenue, n_name, q5_result_cap, "
                               "(uint)n_nationkey[" + idx_ + "], (uint)" + idx_ + ");");
                });
                consume();
            }
            std::string describe() const override { return "Q5CompactResult"; }
        };
        auto compact = std::make_unique<Q5CompactTerminal>(std::move(scan), idx);
        auto& phase = appendPhase(plan, "Q5_compact_results", std::move(compact), 32);
        phase.extraBuffers.push_back({"d_q5_result_count",   "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q5_result_name",    "char",        false, true});
        phase.extraBuffers.push_back({"d_q5_result_revenue", "float",       false, true});
        phase.extraBuffers.push_back({"d_nation_revenue",    "float",       true,  false});
        phase.scalarParams.push_back({"q5_result_cap", "uint"});
    }

    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("n_name", "d_q5_result_name", "char", 25),
            GenericMatColumnDesc("revenue", "d_q5_result_revenue", "float"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"revenue", true});
        std::string sortError;
        if (!appendGenericGpuSmallSort(plan, "q5_result", resultRows,
                                       32, columns, sortSpec, &sortError)) {
            if (!appendGenericGpuSort(plan, "q5_result", resultRows,
                                      "q5_result_cap", columns, sortSpec,
                                      &sortError)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}

} // namespace codegen
