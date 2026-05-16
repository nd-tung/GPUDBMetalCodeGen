#pragma once
// --- Information Unit ---
// Describes a column reference found in a kernel-side expression.
// Scans use IUs to auto-deduce required columns through the parent chain.
// Metal expressions use colName[idxVar], for example l_shipdate[i].

#include <functional>
#include <string>
#include <vector>

namespace codegen {

struct IU {
    std::string colName;   ///< Column name in shader.
    std::string idxVar;    ///< Index variable.
    std::string tableName; ///< Resolved by scan via ColumnTypeResolver.
    std::string metalType; ///< Metal element type, resolved later.

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
