#pragma once

#include "../core/infra.h"
#include "../core/schema_provider.h"
#include "metal_generic_executor.h"
#include "tpch_schema.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

ColSpec colSpecFor(const ColumnDef& cdef);
ColSpec colSpecFor(int columnIndex, DataType type, int fixedWidth);
ColSpec colSpecFor(const SchemaProvider& schema,
                   const std::string& table,
                   const std::string& column);

void registerMaxKeySymbols(
    MetalGenericExecutor& executor,
    const std::vector<std::pair<std::string, QueryColumns>>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const SchemaProvider& schema);

void extendMaxKeysFromStreamColbin(
    MetalGenericExecutor& executor,
    const std::string& streamTblPath,
    const std::set<std::string>& streamCols,
    const SchemaProvider& schema,
    const std::string& streamTable);

} // namespace codegen
