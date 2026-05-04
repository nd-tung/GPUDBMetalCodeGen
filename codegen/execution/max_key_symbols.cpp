#include "max_key_symbols.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#include "../../third_party/nlohmann/json.hpp"

namespace codegen {

ColSpec colSpecFor(const ColumnDef& cdef) {
    ColType type = ColType::INT;
    switch (cdef.type) {
        case DataType::INT:        type = ColType::INT; break;
        case DataType::FLOAT:      type = ColType::FLOAT; break;
        case DataType::DATE:       type = ColType::DATE; break;
        case DataType::CHAR1:      type = ColType::CHAR1; break;
        case DataType::CHAR_FIXED: type = ColType::CHAR_FIXED; break;
    }
    return ColSpec(cdef.index, type, cdef.fixedWidth);
}

namespace {

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

std::string maxKeyCachePath() {
    return g_dataset_path + ".maxkeys.json";
}

bool loadMaxKeyCache(std::vector<MaxKeyCacheEntry>& out) {
    out.clear();
    std::ifstream file(maxKeyCachePath());
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

void saveMaxKeyCache(const std::vector<MaxKeyCacheEntry>& entries) {
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
    std::ofstream file(maxKeyCachePath());
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

bool isMaxKeyColumn(const std::string& colName) {
    return colName == "c_custkey" || colName == "o_custkey" ||
           colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey" ||
           colName == "o_orderkey" || colName == "l_orderkey" ||
           colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey";
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
                      bool cacheDirty) {
    if (!cacheDirty) return;
    for (auto& entry : cacheWrite) cacheRead.push_back(std::move(entry));
    saveMaxKeyCache(cacheRead);
}

} // namespace

void registerMaxKeySymbols(
    MetalGenericExecutor& executor,
    const std::vector<std::pair<std::string, QueryColumns>>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const TPCHSchema& schema) {
    int maxCustkey = 0;
    int maxSuppkey = 0;
    int maxOrderkey = 0;
    int maxPartkey = 0;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    if (currentMaxKeyMode() == MaxKeyMode::Cache) loadMaxKeyCache(cacheRead);

    for (const auto& [tblName, columns] : loadedTables) {
        const auto tableIt = tableCols.find(tblName);
        if (tableIt == tableCols.end()) continue;
        const auto& tableDef = schema.table(tblName);
        size_t rowCount = columns.rows();
        for (const auto& colName : tableIt->second) {
            const auto& columnDef = tableDef.col(colName);
            if (columnDef.type != DataType::INT && columnDef.type != DataType::DATE) continue;
            const int* data = columns.ints(columnDef.index);
            if (!data) continue;

            const std::string tblPath = g_dataset_path + tblName + ".tbl";
            int colMax = 0;
            if (isMaxKeyColumn(colName)) {
                colMax = computeColMax(data, rowCount, tblPath, columnDef.index,
                                       cacheRead, cacheWrite, cacheDirty);
            }

            if (colName == "c_custkey" || colName == "o_custkey")
                maxCustkey = std::max(maxCustkey, colMax);
            else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
                maxSuppkey = std::max(maxSuppkey, colMax);
            else if (colName == "o_orderkey" || colName == "l_orderkey")
                maxOrderkey = std::max(maxOrderkey, colMax);
            else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
                maxPartkey = std::max(maxPartkey, colMax);
        }
    }

    mergeCacheWrites(cacheRead, cacheWrite, cacheDirty);
    executor.registerSymbol("maxCustkey", maxCustkey + 1);
    executor.registerSymbol("maxSuppkey", maxSuppkey + 1);
    executor.registerSymbol("maxOrderkey", maxOrderkey + 1);
    executor.registerSymbol("maxPartkey", maxPartkey + 1);
}

void extendMaxKeysFromStreamColbin(
    MetalGenericExecutor& executor,
    const std::string& streamTblPath,
    const std::set<std::string>& streamCols,
    const TPCHSchema& schema,
    const std::string& streamTable) {
    if (streamTable.empty()) return;
    const auto& tableDef = schema.table(streamTable);
    std::vector<std::pair<std::string, ColSpec>> intSpecs;
    for (const auto& colName : streamCols) {
        const auto& columnDef = tableDef.col(colName);
        if (columnDef.type != DataType::INT && columnDef.type != DataType::DATE) continue;
        if (!isMaxKeyColumn(colName)) continue;
        intSpecs.emplace_back(colName, colSpecFor(columnDef));
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

    int maxCustkey = 0;
    int maxSuppkey = 0;
    int maxOrderkey = 0;
    int maxPartkey = 0;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    if (currentMaxKeyMode() == MaxKeyMode::Cache) loadMaxKeyCache(cacheRead);

    for (const auto& [colName, spec] : intSpecs) {
        const auto& values = parsed.ints(spec.columnIndex);
        if (values.empty()) continue;
        int colMax = computeColMax(values.data(), values.size(), streamTblPath, spec.columnIndex,
                                   cacheRead, cacheWrite, cacheDirty);
        if (colName == "c_custkey" || colName == "o_custkey")
            maxCustkey = std::max(maxCustkey, colMax);
        else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
            maxSuppkey = std::max(maxSuppkey, colMax);
        else if (colName == "o_orderkey" || colName == "l_orderkey")
            maxOrderkey = std::max(maxOrderkey, colMax);
        else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
            maxPartkey = std::max(maxPartkey, colMax);
    }

    mergeCacheWrites(cacheRead, cacheWrite, cacheDirty);

    auto bump = [&](const char* name, int streamMax) {
        if (streamMax <= 0) return;
        size_t current = 0;
        if (executor.tryGetSymbol(name, current)) {
            executor.registerSymbol(name, std::max(current, (size_t)streamMax + 1));
        } else {
            executor.registerSymbol(name, (size_t)streamMax + 1);
        }
    };
    bump("maxCustkey", maxCustkey);
    bump("maxSuppkey", maxSuppkey);
    bump("maxOrderkey", maxOrderkey);
    bump("maxPartkey", maxPartkey);
}

} // namespace codegen