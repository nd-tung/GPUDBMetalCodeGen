#pragma once
#include "query_plan.h"
#include "../core/schema_provider.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <utility>

namespace codegen {

// ===================================================================
// DATA TYPE ENUM
// ===================================================================
// TPC-H SCHEMA CATALOG
// ===================================================================
// Hard-coded TPC-H table definitions. Column indices match .tbl file layout.

struct ColumnDef {
    std::string name;
    int         index;
    DataType    type;
    int         fixedWidth = 0;    // for CHAR_FIXED
    int         domainMin = -1;    // for INT/DATE GROUP BY: -1 means unknown
    int         domainMax = -1;    // for INT/DATE GROUP BY: -1 means unknown
    std::vector<char> charDomain;  // for CHAR1 GROUP BY: ordered known values

    ColumnDef(std::string n, int idx, DataType dt,
              int width = 0, int minDomain = -1, int maxDomain = -1,
              std::vector<char> chars = {})
        : name(std::move(n)),
          index(idx),
          type(dt),
          fixedWidth(width),
          domainMin(minDomain),
          domainMax(maxDomain),
          charDomain(std::move(chars)) {}
};

struct TableDef {
    std::string maxKeySymbol;
    std::string name;
    std::vector<ColumnDef> columns;
    std::unordered_map<std::string, int> nameToIdx; // column name → index in columns[]

    const ColumnDef& col(const std::string& n) const {
        auto it = nameToIdx.find(n);
        if (it == nameToIdx.end()) throw std::runtime_error("Unknown column: " + name + "." + n);
        return columns[it->second];
    }
};

inline TableDef makeTable(const std::string& name, std::vector<ColumnDef> cols) {
    TableDef t;
    t.name = name;
    t.columns = std::move(cols);
    for (size_t i = 0; i < t.columns.size(); i++)
        t.nameToIdx[t.columns[i].name] = (int)i;
    return t;
}

// ===================================================================
// SCHEMA SINGLETON
// ===================================================================

struct TPCHSchema {
    std::unordered_map<std::string, TableDef> tables;

    const TableDef& table(const std::string& name) const {
        auto it = tables.find(name);
        if (it == tables.end()) throw std::runtime_error("Unknown table: " + name);
        return it->second;
    }

    ColumnBinding binding(const std::string& tableName, const std::string& colName) const {
        auto& t = table(tableName);
        auto& c = t.col(colName);
        return {tableName, colName, c.index, c.type, c.fixedWidth};
    }

    static const TPCHSchema& instance() {
        static TPCHSchema s = build();
        return s;
    }

private:
    static TPCHSchema build() {
        TPCHSchema s;

        // lineitem: 16 columns (0-15)
        {
            s.tables["lineitem"] = makeTable("lineitem", {
                {"l_orderkey",      0,  DataType::INT},
                {"l_partkey",       1,  DataType::INT},
                {"l_suppkey",       2,  DataType::INT},
                {"l_linenumber",    3,  DataType::INT},
                {"l_quantity",      4,  DataType::FLOAT},
                {"l_extendedprice", 5,  DataType::FLOAT},
                {"l_discount",      6,  DataType::FLOAT},
                {"l_tax",           7,  DataType::FLOAT},
                {"l_returnflag",    8,  DataType::CHAR1},
                {"l_linestatus",    9,  DataType::CHAR1},
                {"l_shipdate",      10, DataType::DATE},
                {"l_commitdate",    11, DataType::DATE},
                {"l_receiptdate",   12, DataType::DATE},
                {"l_shipinstruct",  13, DataType::CHAR_FIXED, 25},
                {"l_shipmode",      14, DataType::CHAR_FIXED, 2},
                {"l_comment",       15, DataType::CHAR_FIXED, 44},
            });
            s.tables["lineitem"].columns[0].domainMin = 0;  s.tables["lineitem"].columns[0].domainMax = 6000000; // l_orderkey
            s.tables["lineitem"].columns[3].domainMin = 1;  s.tables["lineitem"].columns[3].domainMax = 7;   // l_linenumber
            s.tables["lineitem"].columns[8].charDomain = {'A', 'N', 'R'};                                    // l_returnflag
            s.tables["lineitem"].columns[9].charDomain = {'F', 'O'};                                          // l_linestatus
            s.tables["lineitem"].maxKeySymbol = "maxOrderkey";  // l_orderkey max ≈ maxOrderkey
        }

        // orders: 9 columns (0-8)
        {
            s.tables["orders"] = makeTable("orders", {
                {"o_orderkey",      0, DataType::INT},
                {"o_custkey",       1, DataType::INT},
                {"o_orderstatus",   2, DataType::CHAR1},
                {"o_totalprice",    3, DataType::FLOAT},
                {"o_orderdate",     4, DataType::DATE},
                {"o_orderpriority", 5, DataType::CHAR1},
                {"o_clerk",         6, DataType::CHAR_FIXED, 15},
                {"o_shippriority",  7, DataType::INT},
                {"o_comment",       8, DataType::CHAR_FIXED, 79},
            });
            s.tables["orders"].columns[7].domainMin = 0;  s.tables["orders"].columns[7].domainMax = 0;   // o_shippriority
        }

        // customer: 8 columns (0-7)
        {
            s.tables["customer"] = makeTable("customer", {
                {"c_custkey",    0, DataType::INT},
                {"c_name",       1, DataType::CHAR_FIXED, 25},
                {"c_address",    2, DataType::CHAR_FIXED, 40},
                {"c_nationkey",  3, DataType::INT},
                {"c_phone",      4, DataType::CHAR_FIXED, 15},
                {"c_acctbal",    5, DataType::FLOAT},
                {"c_mktsegment", 6, DataType::CHAR1},
                {"c_comment",    7, DataType::CHAR_FIXED, 117},
            });
            s.tables["customer"].columns[3].domainMin = 0;  s.tables["customer"].columns[3].domainMax = 24;  // c_nationkey
        }

        // supplier: 7 columns (0-6)
        {
            s.tables["supplier"] = makeTable("supplier", {
                {"s_suppkey",   0, DataType::INT},
                {"s_name",      1, DataType::CHAR_FIXED, 25},
                {"s_address",   2, DataType::CHAR_FIXED, 40},
                {"s_nationkey", 3, DataType::INT},
                {"s_phone",     4, DataType::CHAR_FIXED, 15},
                {"s_acctbal",   5, DataType::FLOAT},
                {"s_comment",   6, DataType::CHAR_FIXED, 101},
            });
            s.tables["supplier"].columns[3].domainMin = 0;  s.tables["supplier"].columns[3].domainMax = 24;  // s_nationkey
        }

        // part: 9 columns (0-8)
        {
            s.tables["part"] = makeTable("part", {
                {"p_partkey",    0, DataType::INT},
                {"p_name",       1, DataType::CHAR_FIXED, 55},
                {"p_mfgr",       2, DataType::CHAR_FIXED, 25},
                {"p_brand",      3, DataType::CHAR_FIXED, 10},
                {"p_type",       4, DataType::CHAR_FIXED, 25},
                {"p_size",       5, DataType::INT},
                {"p_container",  6, DataType::CHAR_FIXED, 10},
                {"p_retailprice",7, DataType::FLOAT},
                {"p_comment",    8, DataType::CHAR_FIXED, 23},
            });
            s.tables["part"].columns[5].domainMin = 1;  s.tables["part"].columns[5].domainMax = 50;  // p_size
        }

        // partsupp: 5 columns (0-4)
        s.tables["partsupp"] = makeTable("partsupp", {
            {"ps_partkey",    0, DataType::INT},
            {"ps_suppkey",    1, DataType::INT},
            {"ps_availqty",   2, DataType::INT},
            {"ps_supplycost", 3, DataType::FLOAT},
            {"ps_comment",    4, DataType::CHAR_FIXED, 199},
        });

        // nation: 4 columns (0-3)
        {
            s.tables["nation"] = makeTable("nation", {
                {"n_nationkey",  0, DataType::INT},
                {"n_name",       1, DataType::CHAR_FIXED, 25},
                {"n_regionkey",  2, DataType::INT},
                {"n_comment",    3, DataType::CHAR_FIXED, 152},
            });
            s.tables["nation"].columns[0].domainMin = 0;  s.tables["nation"].columns[0].domainMax = 24;  // n_nationkey
            s.tables["nation"].columns[2].domainMin = 0;  s.tables["nation"].columns[2].domainMax = 4;   // n_regionkey
        }

        // region: 3 columns (0-2)
        {
            s.tables["region"] = makeTable("region", {
                {"r_regionkey", 0, DataType::INT},
                {"r_name",      1, DataType::CHAR_FIXED, 25},
                {"r_comment",   2, DataType::CHAR_FIXED, 152},
            });
            s.tables["region"].columns[0].domainMin = 0;  s.tables["region"].columns[0].domainMax = 4;   // r_regionkey
        }

        return s;
    }
};

// ===================================================================
// TPCH SCHEMA PROVIDER (SchemaProvider implementation)
// ===================================================================

class TPCHSchemaProvider : public SchemaProvider {
public:
    DataType columnType(const std::string& table, const std::string& col) const override {
        auto it = TPCHSchema::instance().tables.find(table);
        if (it == TPCHSchema::instance().tables.end()) return DataType::INT;
        auto jt = it->second.nameToIdx.find(col);
        if (jt == it->second.nameToIdx.end()) return DataType::INT;
        return it->second.columns[jt->second].type;
    }
    int columnFixedWidth(const std::string& table, const std::string& col) const override {
        return TPCHSchema::instance().table(table).col(col).fixedWidth;
    }
    bool hasColumn(const std::string& table, const std::string& col) const override {
        auto it = TPCHSchema::instance().tables.find(table);
        if (it == TPCHSchema::instance().tables.end()) return false;
        return it->second.nameToIdx.count(col) > 0;
    }
    std::string maxKeySymbol(const std::string& table) const override {
        return TPCHSchema::instance().table(table).maxKeySymbol;
    }
    std::string keyDomainSymbol(const std::string& table,
                                const std::string& col) const override {
        if (auto gd = groupDomain(table, col))
            return std::to_string(gd->maxValue + 1);
        if (auto pk = pkInfo(table); pk && pk->first == col)
            return pk->second;
        if (table == "lineitem" && col == "l_partkey") return "maxPartkey";
        if (table == "lineitem" && col == "l_suppkey") return "maxSuppkey";
        if (table == "orders" && col == "o_custkey") return "maxCustkey";
        if (table == "partsupp" && col == "ps_partkey") return "maxPartkey";
        if (table == "partsupp" && col == "ps_suppkey") return "maxSuppkey";
        if (table == "customer" && col == "c_nationkey") return "25";
        if (table == "supplier" && col == "s_nationkey") return "25";
        if (table == "nation" && col == "n_regionkey") return "5";
        return "";
    }
    std::string distinctDomainSymbol(const std::string& table,
                                     const std::string& col) const override {
        return keyDomainSymbol(table, col);
    }
    std::vector<std::string> tableNames() const override {
        std::vector<std::string> names;
        for (auto& [name, _] : TPCHSchema::instance().tables) names.push_back(name);
        return names;
    }
    std::optional<GroupDomain> groupDomain(const std::string& table,
                                           const std::string& col) const override {
        auto& colDef = TPCHSchema::instance().table(table).col(col);
        if (colDef.domainMin >= 0 && colDef.domainMax >= colDef.domainMin)
            return GroupDomain{colDef.domainMin, colDef.domainMax};
        return std::nullopt;
    }
    std::vector<char> charDomain(const std::string& table,
                                 const std::string& col) const override {
        return TPCHSchema::instance().table(table).col(col).charDomain;
    }
    int tableProbePriority(const std::string& table) const override {
        // TPC-H size ordering: lineitem > orders > partsupp > customer > part > supplier > nation > region
        if (table == "lineitem") return 100;
        if (table == "orders")   return 80;
        if (table == "partsupp") return 70;
        if (table == "customer") return 50;
        if (table == "part")     return 40;
        if (table == "supplier") return 30;
        if (table == "nation")   return 10;
        if (table == "region")   return 5;
        return 0;
    }
    std::optional<std::pair<std::string, std::string>> pkInfo(const std::string& table) const override {
        if (table == "customer") return std::make_pair("c_custkey",   "maxCustkey");
        if (table == "orders")   return std::make_pair("o_orderkey",  "maxOrderkey");
        if (table == "lineitem") return std::make_pair("l_orderkey",  "maxOrderkey");
        if (table == "supplier") return std::make_pair("s_suppkey",   "maxSuppkey");
        if (table == "part")     return std::make_pair("p_partkey",   "maxPartkey");
        if (table == "nation")   return std::make_pair("n_nationkey", "25");
        if (table == "region")   return std::make_pair("r_regionkey", "5");
        return std::nullopt;
    }
    int numericScale(const std::string& table,
                     const std::string& col) const override {
        auto it = TPCHSchema::instance().tables.find(table);
        if (it == TPCHSchema::instance().tables.end()) return 0;
        auto jt = it->second.nameToIdx.find(col);
        if (jt == it->second.nameToIdx.end()) return 0;
        return it->second.columns[jt->second].type == DataType::FLOAT ? 100 : 0;
    }
    size_t tableRowCount(const std::string& /*table*/) const override { return 0; }
};

// ===================================================================
// FILE PATH RESOLUTION
// ===================================================================

inline std::string tblPath(const std::string& dataDir, const std::string& tableName) {
    return dataDir + tableName + ".tbl";
}

} // namespace codegen
