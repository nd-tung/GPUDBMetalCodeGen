#pragma once
// --- Schema Provider ---
// Abstract schema interface used by analyzer, planning, and codegen.
// TPCHSchemaProvider implements this interface for the built-in TPC-H schema.

#include "../planning/query_plan.h"
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace codegen {

// GROUP BY domain for integer/date columns.
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
    virtual int columnIndex(const std::string& table,
                            const std::string& col) const = 0;
    virtual bool hasColumn(const std::string& table,
                           const std::string& col) const = 0;
    virtual std::vector<std::string> columnNames(
        const std::string& table) const = 0;

    // Table metadata for codegen.
    virtual std::string maxKeySymbol(const std::string& table) const = 0;
    virtual std::string keyDomainSymbol(const std::string& table,
                                        const std::string& col) const {
        (void)table;
        (void)col;
        return "";
    }
    virtual std::string distinctDomainSymbol(const std::string& table,
                                             const std::string& col) const {
        return keyDomainSymbol(table, col);
    }
    virtual std::vector<std::string> tableNames() const = 0;

    // GROUP BY domain info.
    virtual std::optional<GroupDomain> groupDomain(
        const std::string& table, const std::string& col) const = 0;
    virtual std::vector<char> charDomain(
        const std::string& table, const std::string& col) const = 0;

    // Multi-table join planning.
    virtual int tableProbePriority(const std::string& table) const {
        const size_t rows = tableRowCount(table);
        const size_t cap = static_cast<size_t>(std::numeric_limits<int>::max());
        return static_cast<int>(rows > cap ? cap : rows);
    }
    virtual std::optional<std::pair<std::string, std::string>> pkInfo(
        const std::string& table) const = 0;

    // Positive scale enables deterministic integer SUM/AVG for decimal-like floats.
    virtual int numericScale(const std::string& table,
                             const std::string& col) const {
        (void)table;
        (void)col;
        return 0;
    }

    // Runtime sizing.
    virtual std::string tableDataPath(const std::string& table) const {
        (void)table;
        return "";
    }
    virtual size_t tableRowCount(const std::string& table) const = 0;
};

} // namespace codegen
