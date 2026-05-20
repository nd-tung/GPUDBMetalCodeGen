#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ10PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q10";
    std::string idxVar = "i";
    const std::string filterCond =
        "o_orderdate[i] >= 19931001 && o_orderdate[i] < 19940101";

    // --- Orders Map ---
    // Map selected orders to customers before probing from lineitem.
    {
        auto scan = makeAutoScan("orders", idxVar);

        auto filtered = maybeSelect(std::move(scan), filterCond);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_orders_map",
            "o_orderkey[" + idxVar + "]",
            "o_custkey[" + idxVar + "]",
            "int", "maxOrderkey");

        appendPhase(plan, "Q10_build_orders_map", std::move(store));
    }

    // --- Revenue Aggregate ---
    // Sum returned-line revenue by customer key.
    {
        auto scan = makeAutoScan("lineitem", idxVar);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_returnflag[" + idxVar + "] == 'R'");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_orders_map",
            "l_orderkey[" + idxVar + "]",
            "_custkey", "int", -1);

        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_cust_revenue",
            "_custkey", revenue, "maxCustkey",
            "atomic_float", "float");

        appendPhase(plan, "Q10_probe_aggregate", std::move(agg));
    }

    // --- Compact Results ---
    // Keep only customers with positive revenue for the final GPU order.
    plan.helpers.push_back(R"(
static void q10_compact_emit(device atomic_uint* counter,
                              device uint* out_ck,
                              device float* out_rev,
                              device uint* out_customer_idx,
                              device uint* out_nation_idx,
                              const device float* d_cust_revenue,
                              const device int* d_q10_customer_idx,
                              const device int* d_q10_nation_idx,
                              const device int* c_nationkey,
                              uint q10_compact_cap,
                              uint ck) {
    float r = d_cust_revenue[ck];
    if (!(r > 0.0f)) return;
    int ci = d_q10_customer_idx[ck];
    if (ci < 0) return;
    int nk = c_nationkey[ci];
    if (nk < 0 || nk >= 25) return;
    int ni = d_q10_nation_idx[nk];
    if (ni < 0) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q10_compact_cap) {
        out_ck[slot] = ck;
        out_rev[slot] = r;
        out_customer_idx[slot] = (uint)ci;
        out_nation_idx[slot] = (uint)ni;
    }
}
)");

    // --- Row Index Maps ---
    // Build row-index maps for the final top-k payload gather.
    {
        auto scan = makeAutoScan("customer", idxVar);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q10_customer_idx",
            "c_custkey[" + idxVar + "]", "(int)" + idxVar,
            "int", "maxCustkey", 0xFF);
        appendPhase(plan, "Q10_build_customer_idx", std::move(store), 256);
    }
    {
        auto scan = makeAutoScan("nation", idxVar);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q10_nation_idx",
            "n_nationkey[" + idxVar + "]", "(int)" + idxVar,
            "int", "25", 0xFF);
        appendPhase(plan, "Q10_build_nation_idx", std::move(store), 64);
    }

    const std::string resultRows = "q10_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q10_cks", idxVar);
        rscan->addSideColumn("customer", "c_nationkey", "int");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q10_unused", "int",
            "(q10_compact_emit(d_q10_compact_count, d_q10_compact_ck, "
            "d_q10_compact_rev, d_q10_key_customer_idx, d_q10_key_nation_idx, "
            "d_cust_revenue, d_q10_customer_idx, d_q10_nation_idx, c_nationkey, "
            "q10_compact_cap, " + idxVar + "), 0)");
        auto& phase = appendPhase(plan, "Q10_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q10_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q10_compact_ck",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q10_compact_rev",   "float",       false, false});
        phase.extraBuffers.push_back({"d_q10_key_customer_idx", "uint",      false, false});
        phase.extraBuffers.push_back({"d_q10_key_nation_idx", "uint",        false, false});
        phase.extraBuffers.push_back({"d_cust_revenue",      "float",       true,  false});
        phase.extraBuffers.push_back({"d_q10_customer_idx",  "int",         true,  false});
        phase.extraBuffers.push_back({"d_q10_nation_idx",    "int",         true,  false});
        phase.scalarParams.push_back({"q10_compact_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q10_compact_count", resultRows);
    }

    // --- Result Order ---
    // Prefer TopK; fall back to full sort when the shape is unsupported.
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("c_custkey", "d_q10_compact_ck", "int"),
            GenericMatColumnDesc("revenue", "d_q10_compact_rev", "float"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"revenue", true});
        sortSpec.keys.push_back({"c_custkey", false});
        sortSpec.limit = 20;
        std::string orderError;
        appendBestGenericGpuOrder(plan, "q10_result", resultRows,
                                  "maxCustkey", columns, sortSpec,
                                  &orderError);
        if (plan.gpuSort) {
            struct Q10LateMaterializeTerminal : MetalOperator {
                std::string sortedIndexBuffer_;
                std::string rowsSymbol_;
                Q10LateMaterializeTerminal(std::string sortedIndexBuffer,
                                           std::string rowsSymbol)
                    : sortedIndexBuffer_(std::move(sortedIndexBuffer)),
                      rowsSymbol_(std::move(rowsSymbol)) {}
                void produce(MetalCodegen& cg, ConsumerFn consume) override {
                    cg.addScalarParam(rowsSymbol_, "uint");
                    cg.addScalarParam("q10_late_limit", "uint");
                    cg.addBufferParam(sortedIndexBuffer_, "int", "", false);
                    cg.addBufferParam("d_q10_compact_ck", "uint", "", false);
                    cg.addBufferParam("d_q10_compact_rev", "float", "", false);
                    cg.addBufferParam("d_q10_key_customer_idx", "uint", "", false);
                    cg.addBufferParam("d_q10_key_nation_idx", "uint", "", false);
                    cg.addColumnParam("c_name", "char", "customer");
                    cg.addColumnParam("c_acctbal", "float", "customer");
                    cg.addColumnParam("c_address", "char", "customer");
                    cg.addColumnParam("c_phone", "char", "customer");
                    cg.addColumnParam("c_comment", "char", "customer");
                    cg.addColumnParam("n_name", "char", "nation");
                    cg.addAtomicBufferParam("d_q10_late_count", "atomic_uint", "1");
                    cg.addBufferParam("d_q10_result_ck", "uint", "q10_late_limit", false);
                    cg.addBufferParam("d_q10_result_name", "char", "q10_late_limit * 25", false);
                    cg.addBufferParam("d_q10_result_rev", "float", "q10_late_limit", false);
                    cg.addBufferParam("d_q10_result_acctbal", "float", "q10_late_limit", false);
                    cg.addBufferParam("d_q10_result_n_name", "char", "q10_late_limit * 25", false);
                    cg.addBufferParam("d_q10_result_address", "char", "q10_late_limit * 40", false);
                    cg.addBufferParam("d_q10_result_phone", "char", "q10_late_limit * 15", false);
                    cg.addBufferParam("d_q10_result_comment", "char", "q10_late_limit * 117", false);

                    cg.registerMaterializeOutput("d_q10_late_count");
                    cg.registerOutputColumn("c_custkey", "d_q10_result_ck", "uint");
                    cg.registerOutputColumn("c_name", "d_q10_result_name", "char", 25);
                    cg.registerOutputColumn("revenue", "d_q10_result_rev", "float");
                    cg.registerOutputColumn("c_acctbal", "d_q10_result_acctbal", "float");
                    cg.registerOutputColumn("n_name", "d_q10_result_n_name", "char", 25);
                    cg.registerOutputColumn("c_address", "d_q10_result_address", "char", 40);
                    cg.registerOutputColumn("c_phone", "d_q10_result_phone", "char", 15);
                    cg.registerOutputColumn("c_comment", "d_q10_result_comment", "char", 117);

                    cg.addIf("tid == 0", [&]() {
                        cg.addLine("uint _late_n = min((uint)" + rowsSymbol_ + ", q10_late_limit);");
                        cg.addLine("atomic_store_explicit(d_q10_late_count, _late_n, memory_order_relaxed);");
                    });
                    cg.addBlock("for (uint rank = tid; rank < q10_late_limit && rank < (uint)" +
                                rowsSymbol_ + "; rank += tpg)", [&]() {
                        cg.addLine("int src_i = " + sortedIndexBuffer_ + "[rank];");
                        cg.addIf("src_i < 0 || (uint)src_i >= (uint)" + rowsSymbol_, [&]() {
                            cg.addLine("continue;");
                        });
                        cg.addLine("uint src = (uint)src_i;");
                        cg.addLine("uint ci = d_q10_key_customer_idx[src];");
                        cg.addLine("uint ni = d_q10_key_nation_idx[src];");
                        cg.addLine("d_q10_result_ck[rank] = d_q10_compact_ck[src];");
                        cg.addLine("d_q10_result_rev[rank] = d_q10_compact_rev[src];");
                        cg.addLine("d_q10_result_acctbal[rank] = c_acctbal[ci];");
                        cg.addBlock("for (uint c = 0; c < 25u; ++c)", [&]() {
                            cg.addLine("d_q10_result_name[rank * 25u + c] = c_name[ci * 25u + c];");
                            cg.addLine("d_q10_result_n_name[rank * 25u + c] = n_name[ni * 25u + c];");
                        });
                        cg.addBlock("for (uint c = 0; c < 40u; ++c)", [&]() {
                            cg.addLine("d_q10_result_address[rank * 40u + c] = c_address[ci * 40u + c];");
                        });
                        cg.addBlock("for (uint c = 0; c < 15u; ++c)", [&]() {
                            cg.addLine("d_q10_result_phone[rank * 15u + c] = c_phone[ci * 15u + c];");
                        });
                        cg.addBlock("for (uint c = 0; c < 117u; ++c)", [&]() {
                            cg.addLine("d_q10_result_comment[rank * 117u + c] = c_comment[ci * 117u + c];");
                        });
                    });
                    consume();
                }
                std::string describe() const override { return "Q10LateMaterializeResult"; }
            };
            const auto sortInfo = *plan.gpuSort;
            appendPhase(plan, "Q10_late_materialize",
                        std::make_unique<Q10LateMaterializeTerminal>(
                            sortInfo.sortedIndexBuffer, resultRows),
                        256);
            plan.gpuSort.reset();
        }
    }

    return plan;
}

} // namespace

std::optional<MetalQueryPlan> buildQ10Plan_byName() {
    return buildQ10PredefinedPlan();
}

} // namespace codegen
