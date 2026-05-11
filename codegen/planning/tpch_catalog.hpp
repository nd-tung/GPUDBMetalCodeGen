#pragma once
// ===================================================================
// TPCH Catalog — populates a Catalog from TPCHSchema
// ===================================================================

#include "catalog.hpp"
#include "tpch_schema.h"
#include <string>

namespace codegen {

inline Catalog makeTPCHCatalog() {
    Catalog cat;
    const auto& s = TPCHSchema::instance();

    for (const auto& [tname, tdef] : s.tables) {
        CatTable ct;
        ct.name = tname;
        ct.primaryKey = tdef.columns.empty() ? "" : tdef.columns.front().name;
        ct.maxKeySymbol = tdef.maxKeySymbol;

        for (const auto& col : tdef.columns) {
            CatColumn cc;
            cc.name = col.name;
            cc.type = col.type;
            cc.fixedWidth = col.fixedWidth;
            cc.domainMin = col.domainMin;
            cc.domainMax = col.domainMax;
            cc.charDomain = col.charDomain;
            cc.isKey = false; // TPCHSchema doesn't explicitly mark keys
            ct.columns.push_back(cc);
        }
        cat.addTable(std::move(ct));
    }
    return cat;
}

} // namespace codegen
