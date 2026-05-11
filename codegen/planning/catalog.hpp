#pragma once
// ===================================================================
// Catalog — generic table/column registry
// ===================================================================
//
// Ported from CUDACodeGeneral/sql/catalog.hpp. Schema layers populate
// the Catalog via addTable(). The catalog itself has zero built-in
// knowledge of any particular schema.
//
// Provides findTable(), findColumn(), resolveColumn() for the query
// analyst and planner.  Copperates with SchemaProvider for deeper
// metadata (domains, max key symbols, row counts).
// ===================================================================

#include "../core/schema_provider.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace codegen {

struct CatColumn {
    std::string name;       // "l_shipdate"
    DataType    type;       // DataType::INT, etc.
    int         fixedWidth = 0;  // for CHAR_FIXED
    int         domainMin  = -1; // for INT/DATE GROUP BY
    int         domainMax  = -1; // for INT/DATE GROUP BY
    std::vector<char> charDomain; // for CHAR1 GROUP BY
    bool        isKey = false;   // primary/foreign key
};

struct CatTable {
    std::string name;
    std::string primaryKey;
    std::string maxKeySymbol;
    std::vector<CatColumn> columns;
    std::unordered_map<std::string, int> nameToIdx;

    const CatColumn* findColumn(const std::string& n) const {
        auto it = nameToIdx.find(n);
        return it != nameToIdx.end() ? &columns[it->second] : nullptr;
    }
};

class Catalog {
    std::unordered_map<std::string, CatTable> tables_;

public:
    Catalog() = default;

    void addTable(CatTable t) {
        auto& ref = tables_[t.name];
        for (size_t i = 0; i < t.columns.size(); ++i)
            t.nameToIdx[t.columns[i].name] = (int)i;
        ref = std::move(t);
    }

    const CatTable* findTable(const std::string& name) const {
        auto it = tables_.find(name);
        return it != tables_.end() ? &it->second : nullptr;
    }

    bool hasTable(const std::string& name) const {
        return tables_.count(name) > 0;
    }

    bool hasColumn(const std::string& table, const std::string& col) const {
        auto* t = findTable(table);
        return t && t->nameToIdx.count(col) > 0;
    }

    // Resolve a possibly-qualified column reference to (table, column).
    // aliasToTable maps query aliases → real table names.
    // Returns empty strings if not found.
    struct Resolved {
        std::string table;
        std::string column;
        DataType type = DataType::INT;
        int fixedWidth = 0;
    };
    Resolved resolve(const std::string& colName,
                     const std::string& qualifier,
                     const std::unordered_map<std::string, std::string>& aliasToTable) const {
        if (!qualifier.empty()) {
            std::string tname;
            auto ait = aliasToTable.find(qualifier);
            tname = (ait != aliasToTable.end()) ? ait->second : qualifier;
            auto* t = findTable(tname);
            if (!t) return {};
            auto* c = t->findColumn(colName);
            if (!c) return {};
            return {t->name, colName, c->type, c->fixedWidth};
        }
        // Unqualified: search all tables in aliasToTable
        Resolved result;
        for (auto& [alias, tname] : aliasToTable) {
            auto* t = findTable(tname);
            if (!t) continue;
            auto* c = t->findColumn(colName);
            if (c) {
                if (!result.table.empty())
                    throw std::runtime_error("Ambiguous column: " + colName);
                result = {t->name, colName, c->type, c->fixedWidth};
            }
        }
        return result;
    }
};

} // namespace codegen
