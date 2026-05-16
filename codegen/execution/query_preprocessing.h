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

struct Q16PostData {
    const int* p_partkey = nullptr;
    const char* p_brand = nullptr; // width 10
    const char* p_type = nullptr;  // width 25
    size_t nPart = 0;
    int maxPartkey = 0;
    int maxSk = 0;
    QueryColumns ownedPart;
};

extern Q16PostData g_q16Post;

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
