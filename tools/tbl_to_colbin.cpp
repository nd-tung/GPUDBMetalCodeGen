// tbl_to_colbin: build .colbin binary column files next to TPC-H .tbl files.
//
// Usage: tbl_to_colbin <data_dir> [table1 table2 ...]
//   <data_dir>   e.g. data/SF-1
//   optional table list (default: all TPC-H tables present)
//
// Writes <data_dir>/<table>.colbin for each table whose .tbl exists.
// Skips tables whose .colbin already matches source size + mtime.
// Use GPUDB_FORCE_REBUILD=1 to regenerate unconditionally.

#include "../src/infra.h"
#include "tpch_schema.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using codegen::TPCHSchema;
using codegen::TableDef;
using codegen::DataType;

static ColType mapType(DataType t) {
    switch (t) {
        case DataType::INT:        return ColType::INT;
        case DataType::FLOAT:      return ColType::FLOAT;
        case DataType::DATE:       return ColType::DATE;
        case DataType::CHAR1:      return ColType::CHAR1;
        case DataType::CHAR_FIXED: return ColType::CHAR_FIXED;
    }
    return ColType::INT;
}

static std::vector<ColSpec> specsForTable(const TableDef& t) {
    std::vector<ColSpec> v;
    v.reserve(t.columns.size());
    for (const auto& c : t.columns) {
        v.push_back(ColSpec{c.index, mapType(c.type), c.fixedWidth});
    }
    return v;
}

static bool fileExists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

static bool binaryUpToDate(const std::string& tblPath) {
    size_t tblSz = 0; int64_t tblMt = 0;
    if (!colbin::statFile(tblPath, tblSz, tblMt)) return false;
    MappedFile mf;
    if (!mf.open(colbin::binaryPath(tblPath))) return false;
    if (mf.size < sizeof(colbin::FileHeader)) return false;
    colbin::FileHeader hdr;
    memcpy(&hdr, mf.data, sizeof(hdr));
    return memcmp(hdr.magic, colbin::MAGIC, 8) == 0
        && hdr.version == colbin::VERSION
        && hdr.source_size == tblSz
        && hdr.source_mtime_ns == tblMt;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: tbl_to_colbin <data_dir> [table...]\n";
        return 2;
    }
    std::string dir = argv[1];
    if (!dir.empty() && dir.back() == '/') dir.pop_back();

    // Force loadColumnsMultiAuto to use the .tbl parser (avoid reading a stale .colbin).
    ::setenv("GPUDB_NO_BINARY", "1", 1);
    const bool force = [] {
        const char* e = ::getenv("GPUDB_FORCE_REBUILD");
        return e && e[0] == '1';
    }();

    std::vector<std::string> tableNames;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) tableNames.emplace_back(argv[i]);
    } else {
        for (const auto& [name, _] : TPCHSchema::instance().tables) {
            tableNames.push_back(name);
        }
    }

    size_t totalOk = 0, totalSkipped = 0, totalFailed = 0;
    for (const auto& name : tableNames) {
        const TableDef& td = TPCHSchema::instance().table(name);
        std::string tbl = dir + "/" + name + ".tbl";
        if (!fileExists(tbl)) {
            std::cerr << "[skip] " << tbl << " not found\n";
            continue;
        }
        if (!force && binaryUpToDate(tbl)) {
            std::cout << "[ok]   " << name << ": binary up-to-date\n";
            totalSkipped++;
            continue;
        }
        auto specs = specsForTable(td);
        auto t0 = std::chrono::steady_clock::now();
        LoadedColumns parsed;
        try {
            parsed = loadColumnsMultiAuto(tbl, specs);
        } catch (const std::exception& e) {
            std::cerr << "[fail] " << name << ": parse error: " << e.what() << "\n";
            totalFailed++;
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        bool w = colbin::writeColbin(tbl, specs, parsed);
        auto t2 = std::chrono::steady_clock::now();
        auto parseMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto writeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        if (!w) {
            std::cerr << "[fail] " << name << ": writeColbin failed\n";
            totalFailed++;
            continue;
        }
        std::cout << "[ok]   " << name
                  << "  parse=" << parseMs << "ms  write=" << writeMs << "ms"
                  << "  binary=" << colbin::binaryPath(tbl) << "\n";
        totalOk++;
    }
    std::cout << "summary: built=" << totalOk
              << " up-to-date=" << totalSkipped
              << " failed=" << totalFailed << "\n";
    return totalFailed == 0 ? 0 : 1;
}
