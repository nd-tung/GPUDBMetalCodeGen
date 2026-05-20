#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ6PredefinedPlan() {
    MetalQueryPlan plan;
    plan.name = "Q6";

    // --- Revenue Reduction ---
    // Scale revenue to fixed-point before the long accumulator.
    std::string idxVar = "i";
    const std::string filterCond =
        "l_shipdate[i] >= 19940101 && l_shipdate[i] < 19950101 && "
        "(l_discount[i] >= 0.050000f && l_discount[i] <= 0.070000f) && "
        "l_quantity[i] < 24.000000f";
    const std::string aggExpr = "(l_extendedprice[i] * l_discount[i])";
    const std::string alias = "revenue";
    auto filtered = maybeSelect(makeAutoScan("lineitem", idxVar), filterCond);

    auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), tableDataName(alias));
    reduce->addAccumulator(alias, "(long)(" + aggExpr + " * 100.0f)", "long");
    reduce->setResultAlias(alias, 100);

    appendPhase(plan, "Q6_reduce", std::move(reduce));
    return plan;
}

} // namespace

// Q6: Forecasting Revenue Change.
std::optional<MetalQueryPlan> buildQ6Plan_byName() {
    return buildQ6PredefinedPlan();
}

} // namespace codegen
