#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

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

    // lineitem+part, 1 target, no GROUP BY is unique to Q14 in TPC-H

    MetalQueryPlan plan;
    plan.name = "Q14";
    std::string idxVar = "i";

    // Separate filters into lineitem-only and join conditions
    // For Q14: lineitem date filters + join on l_partkey = p_partkey
    // We build a bitmap on p_partkey where p_type starts with "PROMO"
    // Then probe it in the lineitem scan

    // Phase 1: Build promo bitmap from part table
    {
        auto scan = makeScan("part", idxVar, {{"p_partkey", "int"}, {"p_type", "char"}});

        // Filter: p_type starts with "PROMO" → check first 5 chars
        // p_type is CHAR_FIXED(25), accessed as p_type[i * 25 + offset]
        std::string promoFilter =
            "p_type[" + idxVar + " * 25] == 'P' && "
            "p_type[" + idxVar + " * 25 + 1] == 'R' && "
            "p_type[" + idxVar + " * 25 + 2] == 'O' && "
            "p_type[" + idxVar + " * 25 + 3] == 'M' && "
            "p_type[" + idxVar + " * 25 + 4] == 'O'";

        auto filter = std::make_unique<MetalSelection>(std::move(scan), promoFilter);

        // Build bitmap keyed on p_partkey
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_promo_bitmap",
            "p_partkey[" + idxVar + "]", "(maxPartkey + 31) / 32 + 1");

        appendPhase(plan, "Q14_build_bitmap", std::move(bitmapBuild));
    }

    // Phase 2: Scan lineitem, filter by date, reduce with promo/total
    {
        auto scan = makeScan("lineitem", idxVar, {
            {"l_partkey", "int"}, {"l_shipdate", "int"},
            {"l_extendedprice", "float"}, {"l_discount", "float"}
        });

        // Filter: date range from analyzed query (l_shipdate predicates)
        // Extract date filters that reference l_shipdate
        std::vector<PredPtr> dateFilters;
        PredPtr joinPred;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("l_partkey") && cols.count("p_partkey")) {
                joinPred = f; // join condition, handled via bitmap
            } else if (cols.count("l_shipdate") || cols.count("l_receiptdate")) {
                dateFilters.push_back(f);
            }
        }

        auto filtered = maybeSelect(std::move(scan), combineFilters(dateFilters, idxVar));

        // TGReduce with 2 accumulators:
        // total_sum: l_extendedprice * (1 - l_discount) for all qualifying rows
        // promo_sum: same, but only when bitmap_test passes for l_partkey
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

} // namespace codegen
