#include "metal_tpch_plan_api.h"
#include "metal_tpch_query_builders.h"

#include <unordered_set>

namespace codegen {

namespace {

const std::unordered_set<std::string>& chunkableTPCHNames() {
    static const std::unordered_set<std::string> kNames = {
        "Q1", "Q6", "Q12", "Q14", "Q19",
        "Q4", "Q13",
        "Q3", "Q5", "Q7", "Q8", "Q10",
        "Q15", "Q18",
        "Q11", "Q22",
    };
    return kNames;
}

void applyTPCHMetadata(MetalQueryPlan& plan, const std::string& queryName) {
    plan.name = queryName;
    if (chunkableTPCHNames().count(queryName)) {
        plan.chunkable = true;
    }
}

} // namespace

bool isPredefinedTPCHQueryName(const std::string& queryName) {
    if (queryName.size() < 2 || queryName[0] != 'Q') return false;
    int q = 0;
    for (size_t i = 1; i < queryName.size(); ++i) {
        char c = queryName[i];
        if (c < '0' || c > '9') return false;
        q = q * 10 + (c - '0');
    }
    return q >= 1 && q <= 22;
}

std::optional<MetalQueryPlan> buildPredefinedTPCHPlan(const std::string& queryName) {
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
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
        if (queryName == "Q6")  return buildQ6Plan_byName();
        if (queryName == "Q1")  return buildQ1Plan_byName();
        if (queryName == "Q14") return buildQ14Plan_byName();
        if (queryName == "Q4")  return buildQ4Plan_byName();
        if (queryName == "Q12") return buildQ12Plan_byName();
        if (queryName == "Q10") return buildQ10Plan_byName();

        return std::nullopt;
    };

    auto plan = dispatch();
    if (plan) applyTPCHMetadata(*plan, queryName);
    return plan;
}

} // namespace codegen