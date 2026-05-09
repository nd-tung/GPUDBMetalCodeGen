#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

namespace {

std::optional<MetalQueryPlan> buildQ14PlanForDateFilter(const std::string& filterCond) {
    MetalQueryPlan plan;
    plan.name = "Q14";
    std::string idxVar = "i";

    {
        auto scan = makeScan("part", idxVar, {{"p_partkey", "int"}, {"p_type", "char"}});
        std::string promoFilter =
            "p_type[" + idxVar + " * 25] == 'P' && "
            "p_type[" + idxVar + " * 25 + 1] == 'R' && "
            "p_type[" + idxVar + " * 25 + 2] == 'O' && "
            "p_type[" + idxVar + " * 25 + 3] == 'M' && "
            "p_type[" + idxVar + " * 25 + 4] == 'O'";

        auto filter = std::make_unique<MetalSelection>(std::move(scan), promoFilter);
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_promo_bitmap",
            "p_partkey[" + idxVar + "]", "(maxPartkey + 31) / 32 + 1");

        appendPhase(plan, "Q14_build_bitmap", std::move(bitmapBuild));
    }

    {
        auto scan = makeScan("lineitem", idxVar, {
            {"l_partkey", "int"}, {"l_shipdate", "int"},
            {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        auto filtered = maybeSelect(std::move(scan), filterCond);
        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "] * 0.01f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_q14");
        reduce->addAccumulator("promo",
            "bitmap_test(d_promo_bitmap, l_partkey[" + idxVar + "]) ? " + revenue + " : 0.0f", "float");
        reduce->addAccumulator("total", revenue, "float");
        reduce->setResultAlias("promo_revenue", 0);
        reduce->setResultAlias("total_revenue", 0);

        auto& phase = appendPhase(plan, "Q14_reduce", std::move(reduce));
        phase.bitmapReads.push_back({"d_promo_bitmap", ""});
    }

    return plan;
}

} // namespace

// ===================================================================
// Q14 Plan Builder — Promotion Effect
// Pattern: BitmapBuild(part, PROMO) → Filter+TGReduce(lineitem, date)
// ===================================================================

std::optional<MetalQueryPlan> buildQ14Plan(const AnalyzedQuery& aq) {
    // Q14: two tables (lineitem, part), scalar aggregate, no GROUP BY
    if (aq.tables.size() != 2) return std::nullopt;
    bool hasLineitem = false, hasPart = false;
    for (auto& t : aq.tables) {
        if (t == "lineitem") hasLineitem = true;
        if (t == "part") hasPart = true;
    }
    if (!hasLineitem || !hasPart) return std::nullopt;
    // Q14 has a complex expression 100*SUM(CASE...)/SUM(...) as a single target.
    // The analyzer may not flag isAgg since the top-level is BinaryExpr, not FuncCall.
    // Just require: not GROUP BY, has exactly 1 target.
    if (aq.hasGroupBy()) return std::nullopt;
    if (aq.targets.size() != 1) return std::nullopt;

    bool hasPartJoin = false;
    bool hasShipdateFilter = false;
    for (const auto& j : aq.joins) {
        bool left = j.leftCol == "l_partkey" && j.rightCol == "p_partkey";
        bool right = j.leftCol == "p_partkey" && j.rightCol == "l_partkey";
        if (left || right) hasPartJoin = true;
    }
    for (const auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("l_partkey") && cols.count("p_partkey")) hasPartJoin = true;
        if (cols.count("l_shipdate")) hasShipdateFilter = true;
    }
    if (!hasPartJoin || !hasShipdateFilter) return std::nullopt;

    std::string idxVar = "i";

    std::vector<PredPtr> dateFilters;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("l_shipdate") || cols.count("l_receiptdate")) {
            dateFilters.push_back(f);
        }
    }
    return buildQ14PlanForDateFilter(combineFilters(dateFilters, idxVar));
}

std::optional<MetalQueryPlan> buildQ14Plan_byName() {
    return buildQ14PlanForDateFilter("l_shipdate[i] >= 19950901 && l_shipdate[i] < 19951001");
}

} // namespace codegen
