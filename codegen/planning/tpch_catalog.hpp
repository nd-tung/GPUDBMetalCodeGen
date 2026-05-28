#pragma once
// --- TPC-H Catalog ---
// Populate a Catalog from TPCHSchema metadata.

#include "catalog.hpp"
#include "tpch_schema.h"
#include <string>

namespace codegen {

inline Catalog makeTPCHCatalog() {
    TPCHSchemaProvider schema;
    return Catalog::fromSchemaProvider(schema);
}

} // namespace codegen
