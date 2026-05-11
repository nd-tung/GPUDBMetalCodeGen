#pragma once
// ===================================================================
// Schema Provider — abstract interface for schema-agnostic codegen.
//
// Replaces direct TPCHSchema::instance() calls with an injected
// provider, enabling non-TPC-H schemas without code changes.
//
// The TPCHSchemaProvider (in tpch_schema.h) wraps the existing
// TPC-H singleton.  User-supplied schemas implement this interface
// and register it with the query analyzer.
// ===================================================================

#include "../planning/query_plan.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

// Domain info for GROUP BY on integer/date columns.
struct GroupDomain {
    int minValue = 0;
    int maxValue = 0;
};

class SchemaProvider {
public:
    virtual ~SchemaProvider() = default;

    // Column metadata.
    virtual DataType columnType(const std::string& table,
                                const std::string& col) const = 0;
    virtual int columnFixedWidth(const std::string& table,
                                 const std::string& col) const = 0;
    virtual bool hasColumn(const std::string& table,
                           const std::string& col) const = 0;

    // Table metadata for codegen.
    virtual std::string maxKeySymbol(const std::string& table) const = 0;
    virtual std::vector<std::string> tableNames() const = 0;

    // GROUP BY domain info.
    virtual std::optional<GroupDomain> groupDomain(
        const std::string& table, const std::string& col) const = 0;
    virtual std::vector<char> charDomain(
        const std::string& table, const std::string& col) const = 0;

    // Multi-table join planning.
    virtual int tableProbePriority(const std::string& table) const = 0;
    virtual std::optional<std::pair<std::string, std::string>> pkInfo(
        const std::string& table) const = 0;

    // Runtime sizing.
    virtual size_t tableRowCount(const std::string& table) const = 0;
};

} // namespace codegen
