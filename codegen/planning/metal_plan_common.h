#pragma once

#include "metal_plan_builder.h"
#include "../core/schema_provider.h"

#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

class SchemaProvider;  // fwd (already included)

using ColumnList = std::initializer_list<std::pair<const char*, const char*>>;

void collectColumns(const ExprPtr& expr, std::set<std::string>& cols);
void collectColumns(const PredPtr& pred, std::set<std::string>& cols);
// Collect column→table mapping directly from AST ColRef nodes.
void collectColumnTables(const ExprPtr& expr, std::map<std::string, std::string>& colToTable);
void collectColumnTables(const PredPtr& pred, std::map<std::string, std::string>& colToTable);

// `schema` is optional; TPC-H singleton used as fallback when nullptr.
std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar,
                        const SchemaProvider* schema = nullptr);
std::string predToMetal(const PredPtr& pred, const std::string& idxVar,
                        const SchemaProvider* schema = nullptr);
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar);

struct MetalKeyedAggSlotForHaving {
    std::string name;       // aggregate slot display name (e.g. "sum(l_quantity)")
    bool isFloatSum = false;
    bool isLongPair = false;
    std::string funcName;   // aggregate function name ("SUM", "COUNT", "AVG", "MIN", "MAX")
    std::string innerColumn; // referenced column (empty for COUNT(*))
};

std::string exprToMetalForHaving(const ExprPtr& expr,
                                 const std::vector<MetalKeyedAggSlotForHaving>& slots);
std::string predToMetalForHaving(const PredPtr& pred,
                                 const std::vector<MetalKeyedAggSlotForHaving>& slots);

std::unique_ptr<MetalGridStrideScan> makeScan(const std::string& table,
                                               const std::string& idxVar,
                                               ColumnList columns);
std::unique_ptr<MetalGridStrideScan> makeScanForCols(const std::string& table,
                                                      const std::string& idxVar,
                                                      const std::set<std::string>& cols);
// Create a scan with no explicit columns — columns are auto-discovered
// at produce time via the IU chain (requires ColumnTypeResolver on codegen).
std::unique_ptr<MetalGridStrideScan> makeAutoScan(const std::string& table,
                                                   const std::string& idxVar = "i");

std::unique_ptr<MetalOperator> maybeSelect(std::unique_ptr<MetalOperator> input,
                                           const std::string& filterCond);

MetalQueryPlan::Phase& appendPhase(MetalQueryPlan& plan,
                                   const std::string& name,
                                   std::unique_ptr<MetalOperator> root,
                                   int threadgroupSize = 1024);

} // namespace codegen