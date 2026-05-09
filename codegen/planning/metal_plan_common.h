#pragma once

#include "metal_plan_builder.h"

#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

using ColumnList = std::initializer_list<std::pair<const char*, const char*>>;

void collectColumns(const ExprPtr& expr, std::set<std::string>& cols);
void collectColumns(const PredPtr& pred, std::set<std::string>& cols);

std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar);
std::string predToMetal(const PredPtr& pred, const std::string& idxVar);
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar);

std::unique_ptr<MetalGridStrideScan> makeScan(const std::string& table,
                                              const std::string& idxVar,
                                              ColumnList columns);
std::unique_ptr<MetalGridStrideScan> makeScanForCols(const std::string& table,
                                                     const std::string& idxVar,
                                                     const std::set<std::string>& cols);

std::unique_ptr<MetalOperator> maybeSelect(std::unique_ptr<MetalOperator> input,
                                           const std::string& filterCond);

MetalQueryPlan::Phase& appendPhase(MetalQueryPlan& plan,
                                   const std::string& name,
                                   std::unique_ptr<MetalOperator> root,
                                   int threadgroupSize = 1024);

} // namespace codegen