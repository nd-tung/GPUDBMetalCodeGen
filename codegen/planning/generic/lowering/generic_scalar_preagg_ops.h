#pragma once

#include "metal_plan_builder.h"

#include <memory>
#include <string>

namespace codegen {

std::unique_ptr<MetalOperator> makeScalarGlobalFloatAgg(
    std::unique_ptr<MetalOperator> child,
    std::string op,
    std::string buffer,
    std::string state,
    std::string value);

std::unique_ptr<MetalOperator> makeScalarDirectFloatAgg(
    std::unique_ptr<MetalOperator> child,
    std::string op,
    std::string buffer,
    std::string state,
    std::string key,
    std::string value,
    std::string size);

std::unique_ptr<MetalOperator> makeScalarFillFloatBuffer(
    std::string buffer,
    std::string size,
    std::string fill);

std::unique_ptr<MetalOperator> makeScalarCompositeHashAgg(
    std::unique_ptr<MetalOperator> child,
    std::string map,
    std::string key1,
    std::string key2,
    std::string value,
    std::string capacity,
    bool valueIsFloat);

void ensureScalarCompositeHashHelpers(MetalQueryPlan& plan);

} // namespace codegen
