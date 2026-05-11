#pragma once
// ===================================================================
// IU — Information Unit (projection descriptor)
// ===================================================================
//
// Lightweight triple describing a column reference encountered in a
// kernel-side expression. Scans auto-deduce required columns by
// walking the parent chain, collecting IUs from downstream operators,
// filtering by matching idxVar, and resolving element types via an
// injected ColumnTypeResolver.
//
// Metal expressions use the colName[idxVar] pattern (e.g.
// "l_shipdate[i]"), not CUDA's rowVar.colName[idxVar].
// ===================================================================

#include <functional>
#include <string>
#include <vector>

namespace codegen {

struct IU {
    std::string colName;   ///< column name in shader (e.g. "l_shipdate")
    std::string idxVar;    ///< index variable (e.g. "i", "i0")
    std::string tableName; ///< resolved by scan via ColumnTypeResolver
    std::string metalType; ///< Metal element type (e.g. "int", "float", "char") — resolved later

    IU() = default;
    IU(std::string cn, std::string iv, std::string tn = {},
       std::string mt = {})
        : colName(std::move(cn)), idxVar(std::move(iv)),
          tableName(std::move(tn)), metalType(std::move(mt)) {}

    bool empty() const { return colName.empty() && idxVar.empty(); }
};

using ColumnTypeResolver =
    std::function<std::string(const std::string& /*table*/,
                              const std::string& /*column*/)>;

} // namespace codegen
