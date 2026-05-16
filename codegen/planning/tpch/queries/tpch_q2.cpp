#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q2: Minimum Cost Supplier.
std::optional<MetalQueryPlan> buildQ2Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Helpers ---
    // Match fixed-width p_type values ending in BRASS.
    plan.helpers.push_back(R"(
static bool q2_type_ends_brass(const device char* p_type, uint idx) {
    const device char* tp = p_type + (uint)idx * 25u;
    int len = 25;
    while (len > 0 && (tp[len-1] == ' ' || tp[len-1] == '\0')) len--;
    return len >= 5 && tp[len-5]=='B' && tp[len-4]=='R' &&
           tp[len-3]=='A' && tp[len-2]=='S' && tp[len-1]=='S';
}
)");

    // Positive floats preserve ordering when reinterpreted as uint.
    plan.helpers.push_back(R"(
static void q2_atomic_min(device atomic_uint* min_cost, uint partkey, float cost) {
    uint cost_uint = as_type<uint>(cost);
    atomic_fetch_min_explicit(&min_cost[partkey], cost_uint, memory_order_relaxed);
}
)");

    // --- Row Index Maps ---
    // Build row-index maps used by final materialization.
    {
        auto scan = makeAutoScan("part", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q2_part_idx",
            "p_partkey[" + idx + "]", "(int)" + idx,
            "int", "maxPartkey", 0xFF);
        appendPhase(plan, "Q2_build_part_idx", std::move(store), 256);
    }
    {
        auto scan = makeAutoScan("supplier", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q2_supp_idx",
            "s_suppkey[" + idx + "]", "(int)" + idx,
            "int", "maxSuppkey", 0xFF);
        appendPhase(plan, "Q2_build_supp_idx", std::move(store), 256);
    }
    {
        auto scan = makeAutoScan("nation", idx);
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_q2_nation_idx",
            "n_nationkey[" + idx + "]", "(int)" + idx,
            "int", "25", 0xFF);
        appendPhase(plan, "Q2_build_nation_idx", std::move(store), 64);
    }

    // --- Part Filter ---
    // Build qualifying part bitmap.
    {
        auto scan = makeAutoScan("part", idx);

        auto sizeFilter = std::make_unique<MetalSelection>(
            std::move(scan), "p_size[" + idx + "] == 15");

        auto typeFilter = std::make_unique<MetalSelection>(
            std::move(sizeFilter),
            "q2_type_ends_brass(p_type, " + idx + ")");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(typeFilter), "d_q2_part_bitmap", "p_partkey[" + idx + "]", "");

        appendPhase(plan, "Q2_build_part_bitmap", std::move(bitmapBuild));
    }

    // --- Minimum Cost Aggregate ---
    // Find minimum supply cost per part.
    {
        auto scan = makeAutoScan("partsupp", idx);

        auto partProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q2_part_bitmap", "ps_partkey[" + idx + "]");

        auto suppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(partProbe), "d_q2_supp_bitmap", "ps_suppkey[" + idx + "]");

        auto atomicMin = std::make_unique<MetalComputeExpr>(
            std::move(suppProbe), "_unused", "int",
            "(q2_atomic_min(d_q2_min_cost, (uint)ps_partkey[" + idx + "], "
            "ps_supplycost[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q2_find_min_cost", std::move(atomicMin));
        phase.extraBuffers.push_back({"d_q2_min_cost", "atomic_uint", false});
    }

    // --- Compact Results ---
    // Compact fully decorated result rows on GPU.
    plan.helpers.push_back(R"(
static void q2_compact_emit(device atomic_uint* counter,
                             device float* out_acctbal,
                             device char* out_s_name,
                             device char* out_n_name,
                             device uint* out_p_partkey,
                             device char* out_p_mfgr,
                             device char* out_s_address,
                             device char* out_s_phone,
                             device char* out_s_comment,
                             const device uint* d_q2_min_cost,
                             const device int* d_q2_part_idx,
                             const device int* d_q2_supp_idx,
                             const device int* d_q2_nation_idx,
                             const device float* s_acctbal,
                             const device char* s_name,
                             const device int* s_nationkey,
                             const device char* s_address,
                             const device char* s_phone,
                             const device char* s_comment,
                             const device char* p_mfgr,
                             const device char* n_name,
                             uint cap, uint pk, uint sk, float supplycost, uint i) {
    (void)i;
    uint minU = d_q2_min_cost[pk];
    if (minU == 0xFFFFFFFFu) return;
    float minCost = as_type<float>(minU);
    if (supplycost != minCost) return;
    int si = d_q2_supp_idx[sk];
    int pi = d_q2_part_idx[pk];
    if (si < 0 || pi < 0) return;
    int nk = s_nationkey[si];
    if (nk < 0 || nk >= 25) return;
    int ni = d_q2_nation_idx[nk];
    if (ni < 0) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        out_acctbal[slot] = s_acctbal[si];
        out_p_partkey[slot] = pk;
        for (uint c = 0; c < 25u; ++c) out_s_name[slot * 25u + c] = s_name[si * 25u + c];
        for (uint c = 0; c < 25u; ++c) out_n_name[slot * 25u + c] = n_name[ni * 25u + c];
        for (uint c = 0; c < 25u; ++c) out_p_mfgr[slot * 25u + c] = p_mfgr[pi * 25u + c];
        for (uint c = 0; c < 40u; ++c) out_s_address[slot * 40u + c] = s_address[si * 40u + c];
        for (uint c = 0; c < 15u; ++c) out_s_phone[slot * 15u + c] = s_phone[si * 15u + c];
        for (uint c = 0; c < 101u; ++c) out_s_comment[slot * 101u + c] = s_comment[si * 101u + c];
    }
}
)");
    const std::string resultRows = "q2_result_rows";
    {
        auto scan = makeAutoScan("partsupp", idx);
        auto partProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q2_part_bitmap", "ps_partkey[" + idx + "]");
        auto suppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(partProbe), "d_q2_supp_bitmap", "ps_suppkey[" + idx + "]");
        struct Q2CompactTerminal : MetalUnaryOperator {
            std::string idx_;
            Q2CompactTerminal(std::unique_ptr<MetalOperator> child, std::string idx)
                : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                cg.addColumnParam("s_acctbal", "float", "supplier");
                cg.addColumnParam("s_name", "char", "supplier");
                cg.addColumnParam("s_nationkey", "int", "supplier");
                cg.addColumnParam("s_address", "char", "supplier");
                cg.addColumnParam("s_phone", "char", "supplier");
                cg.addColumnParam("s_comment", "char", "supplier");
                cg.addColumnParam("p_mfgr", "char", "part");
                cg.addColumnParam("n_name", "char", "nation");
                cg.addColumnParam("ps_supplycost", "float", "partsupp");
                child_->produce(cg, [&]() {
                    cg.addLine("q2_compact_emit(d_q2_compact_count, d_q2_result_acctbal, "
                               "d_q2_result_s_name, d_q2_result_n_name, "
                               "d_q2_result_p_partkey, d_q2_result_p_mfgr, "
                               "d_q2_result_s_address, d_q2_result_s_phone, "
                               "d_q2_result_s_comment, d_q2_min_cost, "
                               "d_q2_part_idx, d_q2_supp_idx, d_q2_nation_idx, "
                               "s_acctbal, s_name, s_nationkey, s_address, "
                               "s_phone, s_comment, p_mfgr, n_name, q2_compact_cap, "
                               "(uint)ps_partkey[" + idx_ + "], (uint)ps_suppkey[" +
                               idx_ + "], ps_supplycost[" + idx_ + "], " + idx_ + ");");
                });
                consume();
            }
            std::string describe() const override { return "Q2CompactResult"; }
        };
        auto sideEffect = std::make_unique<Q2CompactTerminal>(std::move(suppProbe), idx);
        auto& phase = appendPhase(plan, "Q2_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q2_compact_count",       "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q2_result_acctbal",      "float",       false, false});
        phase.extraBuffers.push_back({"d_q2_result_s_name",       "char",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_n_name",       "char",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_p_partkey",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_p_mfgr",       "char",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_s_address",    "char",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_s_phone",      "char",        false, false});
        phase.extraBuffers.push_back({"d_q2_result_s_comment",    "char",        false, false});
        // Read the atomic min-cost buffer through a plain uint view.
        phase.extraBuffers.push_back({"d_q2_min_cost",           "uint", true, false});
        phase.extraBuffers.push_back({"d_q2_part_idx",           "int",  true, false});
        phase.extraBuffers.push_back({"d_q2_supp_idx",           "int",  true, false});
        phase.extraBuffers.push_back({"d_q2_nation_idx",         "int",  true, false});
        phase.scalarParams.push_back({"q2_compact_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q2_compact_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("s_acctbal", "d_q2_result_acctbal", "float"),
            GenericMatColumnDesc("s_name", "d_q2_result_s_name", "char", 25),
            GenericMatColumnDesc("n_name", "d_q2_result_n_name", "char", 25),
            GenericMatColumnDesc("p_partkey", "d_q2_result_p_partkey", "uint"),
            GenericMatColumnDesc("p_mfgr", "d_q2_result_p_mfgr", "char", 25),
            GenericMatColumnDesc("s_address", "d_q2_result_s_address", "char", 40),
            GenericMatColumnDesc("s_phone", "d_q2_result_s_phone", "char", 15),
            GenericMatColumnDesc("s_comment", "d_q2_result_s_comment", "char", 101),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"s_acctbal", true});
        sortSpec.keys.push_back({"n_name", false});
        sortSpec.keys.push_back({"s_name", false});
        sortSpec.keys.push_back({"p_partkey", false});
        sortSpec.limit = 100;
        std::string sortError;
        if (!appendGenericGpuTopKSelection(plan, "q2_result", resultRows,
                                           "q2_compact_cap", columns, sortSpec,
                                           &sortError)) {
            appendGenericGpuSort(plan, "q2_result", resultRows,
                                 "q2_compact_cap", columns, sortSpec, &sortError);
        }
    }

    return plan;
}

} // namespace codegen
