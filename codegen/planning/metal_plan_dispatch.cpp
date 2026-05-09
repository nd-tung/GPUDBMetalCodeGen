#include "metal_plan_builder.h"
#include "metal_tpch_query_builders.h"

#include <unordered_set>

namespace codegen {

std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName) {
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        // Name-based dispatch first for queries that clash with analysis-based detectors
        if (queryName == "Q19") return buildQ19Plan_byName();
        if (queryName == "Q13") return buildQ13Plan_byName();
        if (queryName == "Q22") return buildQ22Plan_byName();
        if (queryName == "Q11") return buildQ11Plan_byName();
        if (queryName == "Q15") return buildQ15Plan_byName();
        if (queryName == "Q18") return buildQ18Plan_byName();
        if (queryName == "Q17") return buildQ17Plan_byName();
        if (queryName == "Q9")  return buildQ9Plan_byName();
        if (queryName == "Q20") return buildQ20Plan_byName();
        if (queryName == "Q2")  return buildQ2Plan_byName();
        if (queryName == "Q16") return buildQ16Plan_byName();
        if (queryName == "Q21") return buildQ21Plan_byName();
        if (queryName == "Q5")  return buildQ5Plan_byName();
        if (queryName == "Q3")  return buildQ3Plan_byName();
        if (queryName == "Q8")  return buildQ8Plan_byName();
        if (queryName == "Q7")  return buildQ7Plan_byName();

        // Analysis-based dispatch
        if (auto p = buildQ6Plan(aq)) return p;
        if (auto p = buildQ1Plan(aq)) return p;
        if (auto p = buildQ14Plan(aq)) return p;
        if (auto p = buildQ4Plan(aq)) return p;
        if (auto p = buildQ12Plan(aq)) return p;
        if (auto p = buildQ10Plan(aq)) return p;
        if (auto p = buildQ7Plan(aq)) return p;

        return std::nullopt;
    };

    auto plan = dispatch();
    if (!plan) return plan;

    static const std::unordered_set<std::string> kChunkableNames = {
        "Q1", "Q6", "Q12", "Q14", "Q19",
        "Q4", "Q13",
        "Q3", "Q5", "Q7", "Q8", "Q10",
        "Q15", "Q18",
        "Q11", "Q22",
    };
    const bool isMicrobench = queryName.rfind("MB", 0) == 0;
    if (kChunkableNames.count(queryName) || isMicrobench) {
        plan->chunkable = true;
    }
    return plan;
}

} // namespace codegen