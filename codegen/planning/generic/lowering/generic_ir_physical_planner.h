#pragma once

#include "generic/ir/generic_relational_ir.h"
#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

struct AnalyzedQuery;

// Phase 2 migration entry point. Lowers the narrow single-table
// materialize/order/limit IR shape to the existing GPU operator stack.
// Returns nullopt when the IR shape is not handled by this lowerer.
std::optional<MetalQueryPlan> lowerSingleTableMaterializeIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerSingleTableScalarAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerSingleTableGroupedAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
    const GenericRelPlan& ir,
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
    const GenericRelPlan& ir,
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
    const GenericRelPlan& ir,
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerFromSubqueryAggregateIRToMetal(
    const AnalyzedQuery& aq,
    std::string* error = nullptr);

} // namespace codegen
