#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q11: Important Stock Identification.
std::optional<MetalQueryPlan> buildQ11Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Build supplier bitmap for GERMANY.
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == germany_nk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_supp_bitmap",
            "s_suppkey[" + idx + "]", "(maxSuppkey + 31) / 32");

        auto& phase = appendPhase(plan, "Q11_build_supp_bitmap", std::move(bitmap), 256);
        phase.scalarParams = {{"germany_nk", "int"}};
    }

    // Aggregate per-part value through the supplier bitmap.
    {
        auto scan = makeAutoScan("partsupp", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_supp_bitmap", "ps_suppkey[" + idx + "]");

        std::string valueExpr = "ps_supplycost[" + idx + "] * (float)ps_availqty[" + idx + "]";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(probed), "d_part_value",
            "ps_partkey[" + idx + "]", valueExpr, "maxPartkey",
            "atomic_float", "float");

        appendPhase(plan, "Q11_aggregate", std::move(agg), 256);
    }

    // --- Threshold Baseline ---
    // Total value feeds the 0.01% materialization threshold.
    {
        auto scan = std::make_unique<MetalRangeScan>("maxPartkey", idx);
        auto reduce = std::make_unique<MetalTGReduce>(std::move(scan), "d_q11_total");
        reduce->addAccumulator("value", "d_part_value[" + idx + "]", "float");
        auto& phase = appendPhase(plan, "Q11_total_value", std::move(reduce), 256);
        phase.extraBuffers.push_back({"d_part_value", "float", true, false});
    }

    const std::string resultRows = "q11_result_rows";
    // --- Threshold Materialization ---
    // The TG reduce float is read through its raw uint storage.
    {
        auto scan = std::make_unique<MetalRangeScan>("maxPartkey", idx);
        auto filtered = std::make_unique<MetalSelection>(
            std::move(scan),
            "d_part_value[" + idx + "] > as_type<float>(d_q11_total_value_lo[0]) * 0.0001f");
        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(filtered), "d_q11_result_count");
        materialize->addColumn("d_q11_result_partkey", "int",
                               "(int)" + idx, "ps_partkey", "maxPartkey");
        materialize->addColumn("d_q11_result_value", "float",
                               "d_part_value[" + idx + "]", "value", "maxPartkey");
        auto& phase = appendPhase(plan, "Q11_materialize_threshold",
                                  std::move(materialize), 256);
        phase.extraBuffers.push_back({"d_part_value", "float", true, false});
        phase.extraBuffers.push_back({"d_q11_total_value_lo", "uint", true, false});
        attachMaterializedCountHook(phase, "d_q11_result_count", resultRows);
    }

    // Q11 typically emits a small result set. The runtime host post step sorts
    // the materialized rows directly, avoiding a command-buffer boundary and
    // generic sort hook for small-SF runs.

    return plan;
}

} // namespace codegen
