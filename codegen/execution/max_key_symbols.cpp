#include "max_key_symbols.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#include "../../third_party/nlohmann/json.hpp"

namespace codegen {

ColSpec colSpecFor(int columnIndex, DataType dataType, int fixedWidth) {
    ColType type = ColType::INT;
    switch (dataType) {
        case DataType::INT:        type = ColType::INT; break;
        case DataType::FLOAT:      type = ColType::FLOAT; break;
        case DataType::DATE:       type = ColType::DATE; break;
        case DataType::CHAR1:      type = ColType::CHAR1; break;
        case DataType::CHAR_FIXED: type = ColType::CHAR_FIXED; break;
    }
    return ColSpec(columnIndex, type, fixedWidth);
}

ColSpec colSpecFor(const ColumnDef& cdef) {
    return colSpecFor(cdef.index, cdef.type, cdef.fixedWidth);
}

ColSpec colSpecFor(const SchemaProvider& schema,
                   const std::string& table,
                   const std::string& column) {
    return colSpecFor(schema.columnIndex(table, column),
                      schema.columnType(table, column),
                      schema.columnFixedWidth(table, column));
}

namespace {

std::string tableDataPath(const SchemaProvider& schema, const std::string& table) {
    std::string path = schema.tableDataPath(table);
    if (!path.empty()) return path;
    return g_dataset_path + table + ".colbin";
}

enum class MaxKeyMode { Serial, Parallel, Cache };

MaxKeyMode currentMaxKeyMode() {
    static MaxKeyMode mode = []() {
        const char* env = std::getenv("GPUDB_MAXKEY_MODE");
        if (!env) return MaxKeyMode::Cache;
        std::string value(env);
        if (value == "serial") return MaxKeyMode::Serial;
        if (value == "parallel") return MaxKeyMode::Parallel;
        if (value == "cache") return MaxKeyMode::Cache;
        return MaxKeyMode::Cache;
    }();
    return mode;
}

int parallelMaxInt(const int* data, size_t count) {
    if (count == 0) return 0;
    const unsigned hardwareThreads = std::max(2u, std::thread::hardware_concurrency());
    const size_t threadCount = std::min<size_t>(hardwareThreads, std::max<size_t>(1, count / 65536));
    if (threadCount <= 1) {
        int maxValue = 0;
        for (size_t i = 0; i < count; i++) if (data[i] > maxValue) maxValue = data[i];
        return maxValue;
    }

    std::vector<std::future<int>> futures;
    futures.reserve(threadCount);
    const size_t chunkSize = (count + threadCount - 1) / threadCount;
    for (size_t thread = 0; thread < threadCount; thread++) {
        const size_t begin = thread * chunkSize;
        const size_t end = std::min(count, begin + chunkSize);
        if (begin >= end) break;
        futures.push_back(std::async(std::launch::async, [data, begin, end]() {
            int maxValue = 0;
            for (size_t i = begin; i < end; i++) if (data[i] > maxValue) maxValue = data[i];
            return maxValue;
        }));
    }

    int maxValue = 0;
    for (auto& future : futures) maxValue = std::max(maxValue, future.get());
    return maxValue;
}

struct MaxKeyCacheEntry {
    std::string file;
    uint64_t size;
    int64_t mtimeNs;
    int columnIndex;
    int maxValue;
};

std::string cachePathForDataPath(const std::string& dataPath) {
    if (dataPath.empty()) return g_dataset_path + ".maxkeys.json";
    const std::string colbinPath = colbin::binaryPath(dataPath);
    const size_t slash = colbinPath.find_last_of('/');
    if (slash == std::string::npos) return ".maxkeys.json";
    if (slash == 0) return "/.maxkeys.json";
    return colbinPath.substr(0, slash + 1) + ".maxkeys.json";
}

std::string cachePathForSchema(
    const SchemaProvider& schema,
    const std::map<std::string, std::set<std::string>>& tableCols) {
    for (const auto& [table, _] : tableCols) {
        const std::string path = tableDataPath(schema, table);
        if (!path.empty()) return cachePathForDataPath(path);
    }
    return g_dataset_path + ".maxkeys.json";
}

bool loadMaxKeyCache(const std::string& cachePath,
                     std::vector<MaxKeyCacheEntry>& out) {
    out.clear();
    std::ifstream file(cachePath);
    if (!file) return false;
    try {
        nlohmann::json json;
        file >> json;
        if (!json.is_array()) return false;
        for (const auto& entry : json) {
            MaxKeyCacheEntry cacheEntry;
            cacheEntry.file = entry.at("file").get<std::string>();
            cacheEntry.size = entry.at("size").get<uint64_t>();
            cacheEntry.mtimeNs = entry.at("mtime_ns").get<int64_t>();
            cacheEntry.columnIndex = entry.at("col").get<int>();
            cacheEntry.maxValue = entry.at("max").get<int>();
            out.push_back(std::move(cacheEntry));
        }
        return true;
    } catch (...) {
        return false;
    }
}

void saveMaxKeyCache(const std::string& cachePath,
                     const std::vector<MaxKeyCacheEntry>& entries) {
    nlohmann::json json = nlohmann::json::array();
    for (const auto& entry : entries) {
        json.push_back({
            {"file", entry.file},
            {"size", entry.size},
            {"mtime_ns", entry.mtimeNs},
            {"col", entry.columnIndex},
            {"max", entry.maxValue},
        });
    }
    std::ofstream file(cachePath);
    if (!file) return;
    file << json.dump(2);
}

bool cacheLookup(const std::vector<MaxKeyCacheEntry>& cache,
                 const std::string& file,
                 uint64_t size,
                 int64_t mtimeNs,
                 int columnIndex,
                 int& out) {
    for (const auto& entry : cache) {
        if (entry.columnIndex == columnIndex && entry.file == file &&
            entry.size == size && entry.mtimeNs == mtimeNs) {
            out = entry.maxValue;
            return true;
        }
    }
    return false;
}

bool isNumericLiteral(const std::string& value) {
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isdigit(ch) != 0;
           });
}

int computeColMax(const int* data,
                  size_t count,
                  const std::string& tblPath,
                  int columnIndex,
                  std::vector<MaxKeyCacheEntry>& cacheRead,
                  std::vector<MaxKeyCacheEntry>& cacheWrite,
                  bool& cacheDirty) {
    const MaxKeyMode mode = currentMaxKeyMode();
    if (mode == MaxKeyMode::Serial) {
        int maxValue = 0;
        for (size_t i = 0; i < count; i++) if (data[i] > maxValue) maxValue = data[i];
        return maxValue;
    }
    if (mode == MaxKeyMode::Parallel) return parallelMaxInt(data, count);

    const std::string colbinPath = colbin::binaryPath(tblPath);
    size_t fileSize = 0;
    int64_t fileMtime = 0;
    if (colbin::statFile(colbinPath, fileSize, fileMtime)) {
        const std::string base = colbinPath.substr(colbinPath.find_last_of('/') + 1);
        int hit = 0;
        if (cacheLookup(cacheRead, base, (uint64_t)fileSize, fileMtime, columnIndex, hit)) {
            return hit;
        }
        int maxValue = parallelMaxInt(data, count);
        cacheWrite.push_back({base, (uint64_t)fileSize, fileMtime, columnIndex, maxValue});
        cacheDirty = true;
        return maxValue;
    }
    return parallelMaxInt(data, count);
}

void mergeCacheWrites(std::vector<MaxKeyCacheEntry>& cacheRead,
                      std::vector<MaxKeyCacheEntry>& cacheWrite,
                      bool cacheDirty,
                      const std::string& cachePath) {
    if (!cacheDirty) return;
    for (auto& entry : cacheWrite) cacheRead.push_back(std::move(entry));
    saveMaxKeyCache(cachePath, cacheRead);
}

} // namespace

void registerMaxKeySymbols(
    MetalGenericExecutor& executor,
    const std::vector<std::pair<std::string, QueryColumns>>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const SchemaProvider& schema) {
    std::map<std::string, int> maxBySymbol;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    const std::string cachePath = cachePathForSchema(schema, tableCols);
    if (currentMaxKeyMode() == MaxKeyMode::Cache)
        loadMaxKeyCache(cachePath, cacheRead);

    for (const auto& [tblName, columns] : loadedTables) {
        const auto tableIt = tableCols.find(tblName);
        if (tableIt == tableCols.end()) continue;
        size_t rowCount = columns.rows();
        for (const auto& colName : tableIt->second) {
            const std::string symbol = schema.keyDomainSymbol(tblName, colName);
            if (symbol.empty() || isNumericLiteral(symbol)) continue;

            const DataType type = schema.columnType(tblName, colName);
            if (type != DataType::INT && type != DataType::DATE) continue;

            const int columnIndex = schema.columnIndex(tblName, colName);
            const int* data = columns.ints(columnIndex);
            if (!data) continue;

            const std::string colbinPath = tableDataPath(schema, tblName);
            const int colMax = computeColMax(data, rowCount, colbinPath, columnIndex,
                                             cacheRead, cacheWrite, cacheDirty);
            maxBySymbol[symbol] = std::max(maxBySymbol[symbol], colMax);
        }
    }

    mergeCacheWrites(cacheRead, cacheWrite, cacheDirty, cachePath);
    for (const auto& [symbol, maxValue] : maxBySymbol)
        executor.registerSymbol(symbol, (size_t)maxValue + 1);
}

void extendMaxKeysFromStreamColbin(
    MetalGenericExecutor& executor,
    const std::string& streamTblPath,
    const std::set<std::string>& streamCols,
    const SchemaProvider& schema,
    const std::string& streamTable) {
    if (streamTable.empty()) return;
    std::vector<std::pair<std::string, ColSpec>> intSpecs;
    for (const auto& colName : streamCols) {
        const std::string symbol = schema.keyDomainSymbol(streamTable, colName);
        if (symbol.empty() || isNumericLiteral(symbol)) continue;

        const DataType type = schema.columnType(streamTable, colName);
        if (type != DataType::INT && type != DataType::DATE) continue;

        intSpecs.emplace_back(symbol, colSpecFor(schema, streamTable, colName));
    }
    if (intSpecs.empty()) return;

    std::vector<ColSpec> specs;
    specs.reserve(intSpecs.size());
    for (const auto& [_, spec] : intSpecs) specs.push_back(spec);

    LoadedColumns parsed;
    if (!colbin::loadColumnsFromBinary(streamTblPath, specs, parsed)) {
        std::cerr << "extendMaxKeysFromStreamColbin: failed to read colbin for "
                  << streamTable << " at " << streamTblPath
                  << " (max-key symbols may be wrong)\n";
        return;
    }

    std::map<std::string, int> maxBySymbol;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    const std::string cachePath = cachePathForDataPath(streamTblPath);
    if (currentMaxKeyMode() == MaxKeyMode::Cache)
        loadMaxKeyCache(cachePath, cacheRead);

    for (const auto& [symbol, spec] : intSpecs) {
        const auto& values = parsed.ints(spec.columnIndex);
        if (values.empty()) continue;
        int colMax = computeColMax(values.data(), values.size(), streamTblPath, spec.columnIndex,
                                   cacheRead, cacheWrite, cacheDirty);
        maxBySymbol[symbol] = std::max(maxBySymbol[symbol], colMax);
    }

    mergeCacheWrites(cacheRead, cacheWrite, cacheDirty, cachePath);

    auto bump = [&](const char* name, int streamMax) {
        if (streamMax <= 0) return;
        size_t current = 0;
        if (executor.tryGetSymbol(name, current)) {
            executor.registerSymbol(name, std::max(current, (size_t)streamMax + 1));
        } else {
            executor.registerSymbol(name, (size_t)streamMax + 1);
        }
    };
    for (const auto& [symbol, maxValue] : maxBySymbol)
        bump(symbol.c_str(), maxValue);
}

} // namespace codegen
