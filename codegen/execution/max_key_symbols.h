#pragma once

#include "../../src/infra.h"
#include "metal_generic_executor.h"
#include "tpch_schema.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

ColSpec colSpecFor(const ColumnDef& cdef);

void registerMaxKeySymbols(
    MetalGenericExecutor& executor,
    const std::vector<std::pair<std::string, QueryColumns>>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const TPCHSchema& schema);

void extendMaxKeysFromStreamColbin(
    MetalGenericExecutor& executor,
    const std::string& streamTblPath,
    const std::set<std::string>& streamCols,
    const TPCHSchema& schema,
    const std::string& streamTable);

} // namespace codegen