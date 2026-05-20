#pragma once

#include "../core/infra.h"
#include "metal_generic_executor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

using LoadedQueryTable = std::pair<std::string, QueryColumns>;

void resetQueryPreprocessingState();

QueryColumns loadPreprocessColumns(MTL::Device* device,
                                   const std::string& tableName,
                                   const std::vector<ColSpec>& specs);
std::vector<int> copyIntColumn(const QueryColumns& columns, int columnIndex);
std::vector<float> copyFloatColumn(const QueryColumns& columns, int columnIndex);
std::vector<char> copyCharColumn(const QueryColumns& columns, int columnIndex, size_t byteCount);

bool prepareQueryPreprocessing(const std::string& queryName,
                               MTL::Device* device,
                               MetalGenericExecutor& executor,
                               const std::vector<LoadedQueryTable>& loadedTables);

} // namespace codegen
