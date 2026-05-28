#pragma once

#include "generic/ir/generic_relational_ir.h"
#include "metal_plan_builder.h"

#include <optional>
#include <string>

namespace codegen {

// Lower supported single-table materialize/order/limit IR shapes to Metal.
// Returns nullopt when this route does not handle the shape.
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

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

std::optional<MetalQueryPlan> lowerFromSubqueryAggregateIRToMetal(
    const GenericRelPlan& ir,
    std::string* error = nullptr);

} // namespace codegen
