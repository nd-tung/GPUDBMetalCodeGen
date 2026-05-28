#pragma once
// --- Catalog ---
// Generic table/column registry used by the analyzer and planner.
// Schema-specific metadata is supplied by SchemaProvider.

#include "../core/schema_provider.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace codegen {

struct CatColumn {
    std::string name;
    DataType    type;
    int         fixedWidth = 0;  // CHAR_FIXED width.
    int         domainMin  = -1; // INT/DATE GROUP BY domain.
    int         domainMax  = -1; // INT/DATE GROUP BY domain.
    std::vector<char> charDomain; // CHAR(1) GROUP BY domain.
    bool        isKey = false;
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

    // Build a complete Catalog view from a SchemaProvider.
    static Catalog fromSchemaProvider(const SchemaProvider& sp) {
        Catalog cat;
        for (const auto& tn : sp.tableNames()) {
            CatTable ct;
            ct.name = tn;
            auto pk = sp.pkInfo(tn);
            if (pk) ct.primaryKey = pk->first;
            ct.maxKeySymbol = sp.maxKeySymbol(tn);

            for (const auto& colName : sp.columnNames(tn)) {
                CatColumn cc;
                cc.name = colName;
                cc.type = sp.columnType(tn, colName);
                cc.fixedWidth = sp.columnFixedWidth(tn, colName);
                if (auto domain = sp.groupDomain(tn, colName)) {
                    cc.domainMin = domain->minValue;
                    cc.domainMax = domain->maxValue;
                }
                cc.charDomain = sp.charDomain(tn, colName);
                cc.isKey = pk && pk->first == colName;
                ct.columns.push_back(std::move(cc));
            }
            cat.addTable(std::move(ct));
        }
        return cat;
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

    // Resolve a possibly qualified column; returns empty strings when unknown.
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
        // Unqualified lookup searches all query table aliases.
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
