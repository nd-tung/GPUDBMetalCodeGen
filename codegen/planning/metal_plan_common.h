#pragma once

#include "metal_plan_builder.h"
#include "../core/schema_provider.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

class SchemaProvider;

using ColumnList = std::initializer_list<std::pair<const char*, const char*>>;

void collectColumns(const ExprPtr& expr, std::set<std::string>& cols);
void collectColumns(const PredPtr& pred, std::set<std::string>& cols);
// Collect column -> table mapping directly from ColRef nodes.
void collectColumnTables(const ExprPtr& expr, std::map<std::string, std::string>& colToTable);
void collectColumnTables(const PredPtr& pred, std::map<std::string, std::string>& colToTable);

// Uses the default schema when schema is nullptr.
std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar,
                        const SchemaProvider* schema = nullptr);
std::string predToMetal(const PredPtr& pred, const std::string& idxVar,
                        const SchemaProvider* schema = nullptr);
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar,
                            const SchemaProvider* schema = nullptr);
std::optional<std::string> fixedStringLikeDataMetal(
    const std::string& dataExpr,
    const std::string& rowIndexExpr,
    int width,
    const std::string& pattern,
    bool negated);

struct MetalKeyedAggSlotForHaving {
    std::string name;       // Aggregate slot display name.
    bool isFloatSum = false;
    bool isLongPair = false;
    std::string funcName;   // SUM, COUNT, AVG, MIN, or MAX.
    std::string innerColumn; // Referenced column; empty for COUNT(*).
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
                                                      const std::set<std::string>& cols,
                                                      const SchemaProvider* schema = nullptr);
// Create a scan whose columns are discovered from the IU chain at produce time.
std::unique_ptr<MetalGridStrideScan> makeAutoScan(const std::string& table,
                                                   const std::string& idxVar = "i");

std::unique_ptr<MetalOperator> maybeSelect(std::unique_ptr<MetalOperator> input,
                                           const std::string& filterCond);

MetalQueryPlan::Phase& appendPhase(MetalQueryPlan& plan,
                                    const std::string& name,
                                    std::unique_ptr<MetalOperator> root,
                                    int threadgroupSize = 1024);

// Add GPU bitonic sort phases and record sorted-index result remapping.
void addGpuSortToPlan(MetalQueryPlan& plan,
                      const std::string& sortColBuffer,
                      const std::string& sortColType,
                      const std::string& nResultsExpr,
                      bool sortDesc = false,
                      const std::string& sortKeyBuf = "d_sortKey",
                      const std::string& sortIdxBuf = "d_sortIdx");

} // namespace codegen
