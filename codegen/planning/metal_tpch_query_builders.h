#pragma once

#include "metal_plan_builder.h"

#include <optional>

namespace codegen {

std::optional<MetalQueryPlan> buildQ1Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ2Plan_byName();
std::optional<MetalQueryPlan> buildQ3Plan_byName();
std::optional<MetalQueryPlan> buildQ4Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ5Plan_byName();
std::optional<MetalQueryPlan> buildQ6Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ7Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ7Plan_byName();
std::optional<MetalQueryPlan> buildQ8Plan_byName();
std::optional<MetalQueryPlan> buildQ9Plan_byName();
std::optional<MetalQueryPlan> buildQ10Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ11Plan_byName();
std::optional<MetalQueryPlan> buildQ12Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ13Plan_byName();
std::optional<MetalQueryPlan> buildQ14Plan(const AnalyzedQuery& aq);
std::optional<MetalQueryPlan> buildQ15Plan_byName();
std::optional<MetalQueryPlan> buildQ16Plan_byName();
std::optional<MetalQueryPlan> buildQ17Plan_byName();
std::optional<MetalQueryPlan> buildQ18Plan_byName();
std::optional<MetalQueryPlan> buildQ19Plan_byName();
std::optional<MetalQueryPlan> buildQ20Plan_byName();
std::optional<MetalQueryPlan> buildQ21Plan_byName();
std::optional<MetalQueryPlan> buildQ22Plan_byName();

} // namespace codegen