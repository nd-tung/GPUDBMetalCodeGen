#pragma once

// Metal-cpp headers (no PRIVATE_IMPLEMENTATION here — that lives in infra.cpp)
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>
#include <atomic>
#include <sys/sysctl.h>

// mmap for binary loaders and large text-file fallback
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ===================================================================
// SYSTEM INFO (chip name + memory)
// ===================================================================
struct SystemInfo {
    std::string chip;
    std::string os;
    std::string gpu;
    size_t      ramBytes      = 0;
    size_t      gpuWorkingSet = 0;
};

inline std::string sysctlString(const char* key) {
    size_t len = 0;
    if (sysctlbyname(key, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string out(len, '\0');
    if (sysctlbyname(key, out.data(), &len, nullptr, 0) != 0) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

inline SystemInfo getSystemInfo(MTL::Device* device = nullptr) {
    SystemInfo info;
    info.chip = sysctlString("machdep.cpu.brand_string");
    info.os   = sysctlString("kern.osproductversion");
    if (!info.os.empty()) info.os = "macOS " + info.os;

    size_t len = sizeof(info.ramBytes);
    sysctlbyname("hw.memsize", &info.ramBytes, &len, nullptr, 0);

    if (device) {
        auto* name = device->name();
        if (name) info.gpu = name->utf8String();
        info.gpuWorkingSet = static_cast<size_t>(device->recommendedMaxWorkingSetSize());
    }
    return info;
}

inline void printSystemInfo(const SystemInfo& s) {
    auto gib = [](size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); };
    printf("System:\n");
    if (!s.chip.empty()) printf("  CPU:        %s\n", s.chip.c_str());
    if (!s.os.empty())   printf("  OS:         %s\n", s.os.c_str());
    if (!s.gpu.empty())  printf("  GPU:        %s\n", s.gpu.c_str());
    if (s.ramBytes)      printf("  RAM:        %.1f GiB\n", gib(s.ramBytes));
    if (s.gpuWorkingSet) printf("  GPU Budget: %.1f GiB (recommendedMaxWorkingSetSize)\n",
                                 gib(s.gpuWorkingSet));
    // Machine-readable single line for CSV harvesting.
    printf("SYSINFO_CSV,%s,%s,%s,%zu,%zu\n",
           s.chip.c_str(), s.os.c_str(), s.gpu.c_str(),
           s.ramBytes, s.gpuWorkingSet);
}

// Round up to next power of 2 (host-side, must match GPU-side next_pow2)
inline uint nextPow2(uint v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Global dataset configuration
extern std::string g_dataset_path;

// --- Memory-mapped file helper ---
struct MappedFile {
    int fd = -1;
    void* data = nullptr;
    size_t size = 0;
    
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::cerr << "Cannot open: " << path << std::endl; return false; }
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { ::close(fd); fd = -1; data = nullptr; return false; }
        madvise(data, size, MADV_SEQUENTIAL);
        return true;
    }
    
    void close() {
        if (data && data != MAP_FAILED) { munmap(data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    
    ~MappedFile() { close(); }
};

// ===================================================================
// REUSABLE HELPERS
// ===================================================================

// --- String Trimming ---
inline std::string trimFixed(const char* chars, size_t index, int width) {
    std::string s(chars + index * width, width);
    s.erase(s.find_last_not_of(std::string("\0 ", 2)) + 1);
    return s;
}

// --- Nation/Region Utilities ---
inline std::map<int, std::string> buildNationNames(const std::vector<int>& nationkeys,
                                                    const char* name_chars, int width) {
    std::map<int, std::string> names;
    for (size_t i = 0; i < nationkeys.size(); i++)
        names[nationkeys[i]] = trimFixed(name_chars, i, width);
    return names;
}

// ===================================================================
// MULTI-COLUMN SINGLE-PASS LOADER (for SF1/SF10)
// ===================================================================

enum class ColType { INT, FLOAT, DATE, CHAR1, CHAR_FIXED };

struct ColSpec {
    int          columnIndex;
    ColType      type;
    int          fixedWidth;  // only for CHAR_FIXED
    
    ColSpec(int idx, ColType t, int fw = 0) : columnIndex(idx), type(t), fixedWidth(fw) {}
};

struct LoadedColumns {
    std::vector<std::vector<int>>   intCols;
    std::vector<std::vector<float>> floatCols;
    std::vector<std::vector<char>>  charCols;
    
    // Maps from (columnIndex, type) → index into intCols/floatCols/charCols
    std::unordered_map<int, size_t> intMap, floatMap, charMap;
    
    const std::vector<int>&   ints(int col)   const { return intCols[intMap.at(col)]; }
    const std::vector<float>& floats(int col) const { return floatCols[floatMap.at(col)]; }
    const std::vector<char>&  chars(int col)  const { return charCols[charMap.at(col)]; }
    
    std::vector<int>&   ints(int col)   { return intCols[intMap.at(col)]; }
    std::vector<float>& floats(int col) { return floatCols[floatMap.at(col)]; }
    std::vector<char>&  chars(int col)  { return charCols[charMap.at(col)]; }
};

inline LoadedColumns loadColumnsMulti(const std::string& filePath, const std::vector<ColSpec>& specs) {
    LoadedColumns result;
    
    // Pre-allocate storage slots and build maps
    struct ColHandler {
        int    columnIndex;
        ColType type;
        int    fixedWidth;
        size_t storageIdx;
    };
    std::vector<ColHandler> handlers;
    int maxCol = 0;
    
    for (const auto& s : specs) {
        ColHandler h;
        h.columnIndex = s.columnIndex;
        h.type = s.type;
        h.fixedWidth = s.fixedWidth;
        maxCol = std::max(maxCol, s.columnIndex);
        
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                h.storageIdx = result.intCols.size();
                result.intMap[s.columnIndex] = h.storageIdx;
                result.intCols.emplace_back();
                break;
            case ColType::FLOAT:
                h.storageIdx = result.floatCols.size();
                result.floatMap[s.columnIndex] = h.storageIdx;
                result.floatCols.emplace_back();
                break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED:
                h.storageIdx = result.charCols.size();
                result.charMap[s.columnIndex] = h.storageIdx;
                result.charCols.emplace_back();
                break;
        }
        handlers.push_back(h);
    }
    
    // Build fast lookup: columnIndex → handler index
    std::vector<int> colToHandler(maxCol + 1, -1);
    for (size_t i = 0; i < handlers.size(); i++)
        colToHandler[handlers[i].columnIndex] = (int)i;
    
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return result; }
    
    std::string line;
    while (std::getline(file, line)) {
        int col = 0;
        size_t start = 0;
        size_t end = line.find('|');
        while (end != std::string::npos && col <= maxCol) {
            if (colToHandler[col] >= 0) {
                auto& h = handlers[colToHandler[col]];
                const char* tok = line.c_str() + start;
                size_t tokLen = end - start;
                
                switch (h.type) {
                    case ColType::INT:
                        result.intCols[h.storageIdx].push_back(atoi(tok));
                        break;
                    case ColType::FLOAT:
                        result.floatCols[h.storageIdx].push_back(strtof(tok, nullptr));
                        break;
                    case ColType::DATE: {
                        // Parse YYYY-MM-DD without string copies
                        int y = 0, m = 0, d = 0;
                        const char* p = tok;
                        const char* tokEnd = tok + tokLen;
                        while (p < tokEnd && *p >= '0' && *p <= '9') { y = y * 10 + (*p - '0'); p++; }
                        if (p < tokEnd) p++; // skip '-'
                        while (p < tokEnd && *p >= '0' && *p <= '9') { m = m * 10 + (*p - '0'); p++; }
                        if (p < tokEnd) p++; // skip '-'
                        while (p < tokEnd && *p >= '0' && *p <= '9') { d = d * 10 + (*p - '0'); p++; }
                        result.intCols[h.storageIdx].push_back(y * 10000 + m * 100 + d);
                        break;
                    }
                    case ColType::CHAR1:
                        result.charCols[h.storageIdx].push_back(tokLen > 0 ? tok[0] : '\0');
                        break;
                    case ColType::CHAR_FIXED: {
                        auto& v = result.charCols[h.storageIdx];
                        int cp = (int)tokLen < h.fixedWidth ? (int)tokLen : h.fixedWidth;
                        v.insert(v.end(), tok, tok + cp);
                        v.insert(v.end(), h.fixedWidth - cp, '\0');
                        break;
                    }
                }
            }
            start = end + 1;
            end = line.find('|', start);
            col++;
        }
    }
    return result;
}

// ===================================================================
// OUT-OF-CORE LOADER (mmap + parallel-parse, exact pre-size)
// -------------------------------------------------------------------
// Drop-in replacement for loadColumnsMulti() when file size is large.
// - mmap'd file: avoids loading the whole .tbl into the heap (page
//   cache is recyclable by the OS; no anonymous RSS for raw text).
// - Pre-sized vectors: no geometric-growth 2x spike during parse.
// - Parallel parse: N threads carve up row ranges independently.
// This keeps peak RSS ~= (final columns only) instead of
// (raw .tbl + final columns + growth overhead).
// ===================================================================
inline size_t countLinesParallel(const char* base, size_t size, int nThreads) {
    if (nThreads < 1) nThreads = 1;
    std::vector<size_t> parts(nThreads, 0);
    std::vector<std::thread> ts;
    ts.reserve(nThreads);
    for (int t = 0; t < nThreads; t++) {
        size_t lo = (size * t) / nThreads;
        size_t hi = (size * (t + 1)) / nThreads;
        ts.emplace_back([base, lo, hi, t, &parts]() {
            size_t c = 0;
            for (size_t i = lo; i < hi; i++) if (base[i] == '\n') c++;
            parts[t] = c;
        });
    }
    for (auto& th : ts) th.join();
    size_t total = 0;
    for (size_t c : parts) total += c;
    return total;
}

// Build line start index in parallel (offsets[i] = file offset of line i, offsets.size() == nLines)
inline std::vector<size_t> buildLineIndexParallel(const char* base, size_t size,
                                                   size_t nLines, int nThreads) {
    std::vector<size_t> offsets(nLines);
    if (nThreads < 1) nThreads = 1;
    // First pass: per-thread count to compute start index
    std::vector<size_t> threadCounts(nThreads, 0);
    std::vector<size_t> threadStarts(nThreads, 0);
    std::vector<size_t> threadLo(nThreads, 0), threadHi(nThreads, 0);
    {
        std::vector<std::thread> ts; ts.reserve(nThreads);
        for (int t = 0; t < nThreads; t++) {
            threadLo[t] = (size * t) / nThreads;
            threadHi[t] = (size * (t + 1)) / nThreads;
            ts.emplace_back([base, t, &threadCounts, &threadLo, &threadHi]() {
                size_t c = 0;
                for (size_t i = threadLo[t]; i < threadHi[t]; i++) if (base[i] == '\n') c++;
                threadCounts[t] = c;
            });
        }
        for (auto& th : ts) th.join();
    }
    // Account for the implicit "line 0 start at offset 0" written by thread 0.
    // Without this offset shift, thread 1 would overwrite thread 0's last
    // newline-position write at slot threadCounts[0], and the very last entry
    // would store the past-EOF position written by the last thread's final
    // newline — producing a phantom all-zero "row" at the end of the table.
    threadStarts[0] = 0;
    {
        size_t acc = 1;  // thread 0's implicit line-0 entry
        for (int t = 1; t < nThreads; t++) {
            threadStarts[t] = acc;
            acc += threadCounts[t - 1];
        }
    }

    // Second pass: write offsets
    // Line 0 starts at offset 0; every subsequent line starts at (pos of '\n') + 1
    // We record: offsets[k] = start of line k
    // The bounds check `writeIdx < nLines` drops the past-EOF write that would
    // come from the file's final '\n'.
    std::vector<std::thread> ts; ts.reserve(nThreads);
    for (int t = 0; t < nThreads; t++) {
        ts.emplace_back([base, t, nLines, &offsets, &threadLo, &threadHi, &threadStarts]() {
            size_t writeIdx = threadStarts[t];
            // Thread 0 owns line 0 start.
            if (t == 0) {
                if (writeIdx < nLines) offsets[writeIdx++] = 0;
            }
            for (size_t i = threadLo[t]; i < threadHi[t]; i++) {
                if (base[i] == '\n') {
                    size_t next = i + 1;
                    // Skip the terminal newline (would record past-EOF as a line start).
                    if (writeIdx < nLines) offsets[writeIdx++] = next;
                }
            }
        });
    }
    for (auto& th : ts) th.join();
    // offsets may contain one past-end entry if the file ends with \n; trim if needed.
    // By construction writeIdx == threadStarts[t+1] for each t except the last, so fine.
    return offsets;
}

// mmap+parallel variant of loadColumnsMulti (drop-in result shape).
inline LoadedColumns loadColumnsMultiMmap(const std::string& filePath,
                                           const std::vector<ColSpec>& specs,
                                           int nThreads = 0) {
    LoadedColumns result;

    // Open+mmap
    MappedFile mf;
    if (!mf.open(filePath)) return result;
    const char* base = (const char*)mf.data;

    if (nThreads <= 0) {
        nThreads = (int)std::thread::hardware_concurrency();
        if (nThreads <= 0) nThreads = 4;
    }

    // Count lines & build offset index (parallel)
    size_t nLines = countLinesParallel(base, mf.size, nThreads);
    std::vector<size_t> lineIndex = buildLineIndexParallel(base, mf.size, nLines, nThreads);

    // Pre-allocate exact-size storage. Fill handlers + maps.
    struct ColHandler {
        int columnIndex;
        ColType type;
        int fixedWidth;
        size_t storageIdx;
    };
    std::vector<ColHandler> handlers;
    int maxCol = 0;
    for (const auto& s : specs) {
        ColHandler h{s.columnIndex, s.type, s.fixedWidth, 0};
        maxCol = std::max(maxCol, s.columnIndex);
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                h.storageIdx = result.intCols.size();
                result.intMap[s.columnIndex] = h.storageIdx;
                result.intCols.emplace_back(nLines);
                break;
            case ColType::FLOAT:
                h.storageIdx = result.floatCols.size();
                result.floatMap[s.columnIndex] = h.storageIdx;
                result.floatCols.emplace_back(nLines);
                break;
            case ColType::CHAR1:
                h.storageIdx = result.charCols.size();
                result.charMap[s.columnIndex] = h.storageIdx;
                result.charCols.emplace_back(nLines);
                break;
            case ColType::CHAR_FIXED:
                h.storageIdx = result.charCols.size();
                result.charMap[s.columnIndex] = h.storageIdx;
                result.charCols.emplace_back(nLines * s.fixedWidth, '\0');
                break;
        }
        handlers.push_back(h);
    }

    // Parallel parse: each thread owns a disjoint row range.
    std::vector<std::thread> workers;
    workers.reserve(nThreads);
    for (int t = 0; t < nThreads; t++) {
        size_t lo = (nLines * t) / nThreads;
        size_t hi = (nLines * (t + 1)) / nThreads;
        workers.emplace_back([&, lo, hi]() {
            const char* fileEnd = base + mf.size;
            for (size_t r = lo; r < hi; r++) {
                const char* line = base + lineIndex[r];
                // For each handler, scan to its column from line start.
                for (const auto& h : handlers) {
                    const char* p = line;
                    int col = 0;
                    while (col < h.columnIndex) {
                        while (p < fileEnd && *p != '|' && *p != '\n') p++;
                        if (p >= fileEnd || *p == '\n') { p = nullptr; break; }
                        p++; col++;
                    }
                    if (!p) continue;
                    const char* s = p;
                    const char* e = s;
                    while (e < fileEnd && *e != '|' && *e != '\n') e++;
                    switch (h.type) {
                        case ColType::INT:
                            result.intCols[h.storageIdx][r] = atoi(s);
                            break;
                        case ColType::FLOAT:
                            result.floatCols[h.storageIdx][r] = strtof(s, nullptr);
                            break;
                        case ColType::DATE: {
                            int y = 0, m = 0, d = 0;
                            const char* q = s;
                            while (q < e && *q >= '0' && *q <= '9') { y = y * 10 + (*q - '0'); q++; }
                            if (q < e) q++;
                            while (q < e && *q >= '0' && *q <= '9') { m = m * 10 + (*q - '0'); q++; }
                            if (q < e) q++;
                            while (q < e && *q >= '0' && *q <= '9') { d = d * 10 + (*q - '0'); q++; }
                            result.intCols[h.storageIdx][r] = y * 10000 + m * 100 + d;
                            break;
                        }
                        case ColType::CHAR1:
                            result.charCols[h.storageIdx][r] = (s < e) ? *s : '\0';
                            break;
                        case ColType::CHAR_FIXED: {
                            int len = (int)(e - s);
                            int cp = len < h.fixedWidth ? len : h.fixedWidth;
                            char* dst = result.charCols[h.storageIdx].data() + r * h.fixedWidth;
                            memcpy(dst, s, cp);
                            // tail already '\0' (vector was zero-init)
                            break;
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    // MappedFile closed by destructor.
    return result;
}

// Dispatcher: use mmap+parallel path for large files; keep ifstream path
// for small files (fast startup, no mmap setup overhead).
// Threshold: 1 GB.
// Per-thread load-source tracker. Populated by loadColumnsMultiAuto.
// Reset with loadStats().reset() at the start of each query.
struct LoadStats {
    size_t bytes       = 0;
    int    tblCalls    = 0;
    int    colbinCalls = 0;
    double excludedMs  = 0.0;   // one-time .tbl->column ingest, subtracted from e2e
    void reset() { bytes = 0; tblCalls = 0; colbinCalls = 0; excludedMs = 0.0; }
    void recordBinary(size_t b) { bytes += b; colbinCalls++; }
    void recordText(size_t b)   { bytes += b; tblCalls++; }
    void recordExcluded(double ms) { excludedMs += ms; }
    std::string source() const {
        if (tblCalls && colbinCalls) return "mixed";
        if (colbinCalls)             return "colbin";
        if (tblCalls)                return "tbl";
        return "none";
    }
};
inline LoadStats& loadStats() { thread_local LoadStats s; return s; }

// Forward decl so binary-format utilities below can be inlined before the definition.
inline LoadedColumns loadColumnsMultiAuto(const std::string& filePath,
                                           const std::vector<ColSpec>& specs);

// ===================================================================
// COLUMNAR BINARY FORMAT (.colbin) — primary on-disk column store.
// -------------------------------------------------------------------
// Layout:
//   [magic 8B "TPCHCB01"]
//   [u32 version][u32 n_cols][u64 n_rows]
//   [u64 source_size][i64 source_mtime_ns][u64 pad]  -> 48 B header
//   n_cols * ColDesc (each 32 B)
//   [payloads, each aligned to 16 B]
// ===================================================================

namespace colbin {
static constexpr char     MAGIC[8]   = {'T','P','C','H','C','B','0','1'};
static constexpr uint32_t VERSION    = 2;
static constexpr size_t   ALIGN      = 16384;

struct FileHeader {
    char     magic[8];
    uint32_t version;
    uint32_t n_cols;
    uint64_t n_rows;
    uint64_t source_size;
    int64_t  source_mtime_ns;
    uint64_t _pad;
};
static_assert(sizeof(FileHeader) == 48, "FileHeader size drift");

struct ColDesc {
    int32_t  columnIndex;
    uint8_t  dtype;
    uint8_t  _pad0;
    uint16_t fixedWidth;
    uint64_t offset;
    uint64_t size_bytes;
    uint64_t _pad1;
};
static_assert(sizeof(ColDesc) == 32, "ColDesc size drift");

inline uint8_t encodeType(ColType t) {
    switch (t) {
        case ColType::INT:        return 0;
        case ColType::FLOAT:      return 1;
        case ColType::DATE:       return 2;
        case ColType::CHAR1:      return 3;
        case ColType::CHAR_FIXED: return 4;
    }
    return 0;
}
inline ColType decodeType(uint8_t b) {
    switch (b) {
        case 0: return ColType::INT;
        case 1: return ColType::FLOAT;
        case 2: return ColType::DATE;
        case 3: return ColType::CHAR1;
        default: return ColType::CHAR_FIXED;
    }
}

inline std::string binaryPath(const std::string& tblPath) {
    auto pos = tblPath.rfind('.');
    std::string base = (pos == std::string::npos) ? tblPath : tblPath.substr(0, pos);
    return base + ".colbin";
}

inline bool statFile(const std::string& p, size_t& sz, int64_t& mtimeNs) {
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0) return false;
    sz = (size_t)st.st_size;
#ifdef __APPLE__
    mtimeNs = (int64_t)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    mtimeNs = (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
#endif
    return true;
}

inline bool loadColumnsFromBinary(const std::string& tblPath,
                                  const std::vector<ColSpec>& specs,
                                  LoadedColumns& out)
{
    const std::string cp = binaryPath(tblPath);
    size_t tblSize = 0; int64_t tblMtime = 0;
    const bool tblPresent = statFile(tblPath, tblSize, tblMtime);
    // .tbl is optional: if missing, trust the .colbin's internal header and
    // skip source-size/mtime cross-check (tbl files may be deleted to save
    // disk once .colbin has been generated).

    MappedFile mf;
    if (!mf.open(cp)) return false;
    if (mf.size < sizeof(FileHeader)) return false;

    const char* base = (const char*)mf.data;
    FileHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, MAGIC, 8) != 0) return false;
    if (hdr.version != VERSION) return false;
    if (tblPresent) {
        if (hdr.source_size != tblSize) return false;
        if (hdr.source_mtime_ns != tblMtime) return false;
    }
    if (sizeof(FileHeader) + hdr.n_cols * sizeof(ColDesc) > mf.size) return false;

    const ColDesc* descs = (const ColDesc*)(base + sizeof(FileHeader));
    std::unordered_map<int, const ColDesc*> byIdx;
    byIdx.reserve(hdr.n_cols);
    for (uint32_t i = 0; i < hdr.n_cols; i++) byIdx[descs[i].columnIndex] = &descs[i];

    LoadedColumns result;
    const size_t n = hdr.n_rows;
    for (const auto& s : specs) {
        auto it = byIdx.find(s.columnIndex);
        if (it == byIdx.end()) return false;
        const ColDesc* d = it->second;
        if (d->offset + d->size_bytes > mf.size) return false;
        if (decodeType(d->dtype) != s.type) return false;
        if (s.type == ColType::CHAR_FIXED && d->fixedWidth != s.fixedWidth) return false;
        const char* src = base + d->offset;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE: {
                if (d->size_bytes != n * sizeof(int32_t)) return false;
                size_t idx = result.intCols.size();
                result.intMap[s.columnIndex] = idx;
                result.intCols.emplace_back(n);
                memcpy(result.intCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::FLOAT: {
                if (d->size_bytes != n * sizeof(float)) return false;
                size_t idx = result.floatCols.size();
                result.floatMap[s.columnIndex] = idx;
                result.floatCols.emplace_back(n);
                memcpy(result.floatCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::CHAR1: {
                if (d->size_bytes != n) return false;
                size_t idx = result.charCols.size();
                result.charMap[s.columnIndex] = idx;
                result.charCols.emplace_back(n);
                memcpy(result.charCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::CHAR_FIXED: {
                if (d->size_bytes != n * (size_t)s.fixedWidth) return false;
                size_t idx = result.charCols.size();
                result.charMap[s.columnIndex] = idx;
                result.charCols.emplace_back(d->size_bytes);
                memcpy(result.charCols[idx].data(), src, d->size_bytes);
                break;
            }
        }
    }
    out = std::move(result);
    return true;
}

inline bool writeColbin(const std::string& tblPath,
                         const std::vector<ColSpec>& specs,
                         const LoadedColumns& parsed)
{
    size_t tblSize = 0; int64_t tblMtime = 0;
    if (!statFile(tblPath, tblSize, tblMtime)) return false;
    const std::string cp  = binaryPath(tblPath);
    const std::string tmp = cp + ".tmp";
    if (specs.empty()) return false;

    size_t nRows = 0;
    {
        const auto& s = specs.front();
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                nRows = parsed.ints(s.columnIndex).size(); break;
            case ColType::FLOAT:
                nRows = parsed.floats(s.columnIndex).size(); break;
            case ColType::CHAR1:
                nRows = parsed.chars(s.columnIndex).size(); break;
            case ColType::CHAR_FIXED: {
                const auto& v = parsed.chars(s.columnIndex);
                nRows = s.fixedWidth > 0 ? v.size() / (size_t)s.fixedWidth : v.size();
                break;
            }
        }
    }

    std::vector<ColDesc> descs(specs.size());
    size_t cursor = sizeof(FileHeader) + specs.size() * sizeof(ColDesc);
    cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1);
    for (size_t i = 0; i < specs.size(); i++) {
        const auto& s = specs[i];
        ColDesc& d = descs[i];
        d.columnIndex = s.columnIndex;
        d.dtype       = encodeType(s.type);
        d._pad0       = 0;
        d.fixedWidth  = (uint16_t)s.fixedWidth;
        d._pad1       = 0;
        size_t bytes = 0;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:       bytes = nRows * sizeof(int32_t); break;
            case ColType::FLOAT:      bytes = nRows * sizeof(float);   break;
            case ColType::CHAR1:      bytes = nRows;                   break;
            case ColType::CHAR_FIXED: bytes = nRows * (size_t)s.fixedWidth; break;
        }
        d.size_bytes = bytes;
        d.offset     = cursor;
        cursor += bytes;
        cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1);
    }
    size_t total = cursor;

    int fd = ::open(tmp.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) { std::cerr << "colbin: cannot open " << tmp << "\n"; return false; }
    if (::ftruncate(fd, (off_t)total) != 0) {
        std::cerr << "colbin: ftruncate failed on " << tmp << "\n";
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }

    FileHeader hdr{};
    memcpy(hdr.magic, MAGIC, 8);
    hdr.version         = VERSION;
    hdr.n_cols          = (uint32_t)specs.size();
    hdr.n_rows          = nRows;
    hdr.source_size     = tblSize;
    hdr.source_mtime_ns = tblMtime;
    hdr._pad            = 0;
    if (::pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }
    if (::pwrite(fd, descs.data(), descs.size() * sizeof(ColDesc), sizeof(FileHeader))
         != (ssize_t)(descs.size() * sizeof(ColDesc))) {
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }

    for (size_t i = 0; i < specs.size(); i++) {
        const auto& s = specs[i];
        const ColDesc& d = descs[i];
        const void* src = nullptr;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                src = parsed.ints(s.columnIndex).data(); break;
            case ColType::FLOAT:
                src = parsed.floats(s.columnIndex).data(); break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED:
                src = parsed.chars(s.columnIndex).data(); break;
        }
        size_t remaining = d.size_bytes;
        off_t off = (off_t)d.offset;
        const char* p = (const char*)src;
        while (remaining > 0) {
            size_t chunk = remaining > (size_t)(1u << 30) ? (size_t)(1u << 30) : remaining;
            ssize_t w = ::pwrite(fd, p, chunk, off);
            if (w <= 0) {
                std::cerr << "colbin: payload write failed at col " << s.columnIndex << "\n";
                ::close(fd); ::unlink(tmp.c_str()); return false;
            }
            p += w; off += w; remaining -= (size_t)w;
        }
    }

    ::fsync(fd);
    ::close(fd);
    if (::rename(tmp.c_str(), cp.c_str()) != 0) {
        ::unlink(tmp.c_str()); return false;
    }
    return true;
}

} // namespace colbin

// ===================================================================
// ZERO-COPY COLUMN BUFFERS (requires .colbin v2 with page-aligned payloads)
// -------------------------------------------------------------------
// Instead of memcpy'ing column data through std::vector and then a second
// memcpy into an MTLBuffer, we mmap the .colbin file once and ask Metal to
// wrap each page-aligned payload directly as an MTL::Buffer (via
// newBufferWithBytesNoCopy). On Apple Silicon the GPU shares page tables
// with the CPU, so these buffers are immediately GPU-readable without any
// copy or coherence pass. This eliminates both the tbl-parse cost and the
// host->buffer staging cost on the warm path.
// ===================================================================
struct MappedColumns {
    std::unordered_map<int, MTL::Buffer*> buffers;
    size_t nRows = 0;
    void*  mapBase = nullptr;
    size_t mapSize = 0;

    MappedColumns() = default;
    MappedColumns(const MappedColumns&) = delete;
    MappedColumns& operator=(const MappedColumns&) = delete;
    MappedColumns(MappedColumns&& o) noexcept
      : buffers(std::move(o.buffers)), nRows(o.nRows),
        mapBase(o.mapBase), mapSize(o.mapSize) {
        o.mapBase = nullptr; o.mapSize = 0;
    }
    MappedColumns& operator=(MappedColumns&& o) noexcept {
        if (this != &o) {
            reset();
            buffers  = std::move(o.buffers);
            nRows    = o.nRows;
            mapBase  = o.mapBase;
            mapSize  = o.mapSize;
            o.mapBase = nullptr; o.mapSize = 0;
        }
        return *this;
    }
    ~MappedColumns() { reset(); }

    MTL::Buffer* get(int col) const {
        auto it = buffers.find(col);
        return it == buffers.end() ? nullptr : it->second;
    }
    bool valid() const { return mapBase != nullptr && !buffers.empty(); }

    void reset() {
        for (auto& [k, b] : buffers) if (b) b->release();
        buffers.clear();
        if (mapBase && mapSize) ::munmap(mapBase, mapSize);
        mapBase = nullptr; mapSize = 0;
    }
};

namespace colbin {

inline MappedColumns loadColumnsAsBuffers(MTL::Device* device,
                                           const std::string& tblPath,
                                           const std::vector<ColSpec>& specs) {
    MappedColumns out;
    if (!device) return out;

    const std::string cp = binaryPath(tblPath);
    size_t tblSize = 0; int64_t tblMtime = 0;
    const bool tblPresent = statFile(tblPath, tblSize, tblMtime);
    // .tbl is optional once .colbin has been generated.

    int fd = ::open(cp.c_str(), O_RDONLY);
    if (fd < 0) return out;

    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); return out; }
    size_t fileSize = (size_t)st.st_size;
    if (fileSize < sizeof(FileHeader)) { ::close(fd); return out; }

    void* base = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return out;

    FileHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, MAGIC, 8) != 0 ||
        hdr.version != VERSION ||
        (tblPresent && (hdr.source_size != tblSize ||
                        hdr.source_mtime_ns != tblMtime))) {
        ::munmap(base, fileSize);
        return out;
    }
    if (sizeof(FileHeader) + hdr.n_cols * sizeof(ColDesc) > fileSize) {
        ::munmap(base, fileSize);
        return out;
    }

    const ColDesc* descs = (const ColDesc*)((const char*)base + sizeof(FileHeader));
    std::unordered_map<int, const ColDesc*> byIdx;
    byIdx.reserve(hdr.n_cols);
    for (uint32_t i = 0; i < hdr.n_cols; i++) byIdx[descs[i].columnIndex] = &descs[i];

    for (const auto& s : specs) {
        auto it = byIdx.find(s.columnIndex);
        if (it == byIdx.end())                 { ::munmap(base, fileSize); return MappedColumns{}; }
        const ColDesc* d = it->second;
        if (decodeType(d->dtype) != s.type)    { ::munmap(base, fileSize); return MappedColumns{}; }
        if (s.type == ColType::CHAR_FIXED && d->fixedWidth != s.fixedWidth) {
            ::munmap(base, fileSize); return MappedColumns{};
        }
        if (d->offset % ALIGN != 0) {
            ::munmap(base, fileSize); return MappedColumns{};
        }
        if (d->offset + d->size_bytes > fileSize) {
            ::munmap(base, fileSize); return MappedColumns{};
        }
        void* colPtr = (char*)base + d->offset;
        MTL::Buffer* buf = device->newBuffer(colPtr, d->size_bytes,
                                              MTL::ResourceStorageModeShared,
                                              nullptr /* no deallocator: we own the mmap */);
        if (!buf) {
            ::munmap(base, fileSize);
            return MappedColumns{};
        }
        out.buffers[s.columnIndex] = buf;
    }
    out.nRows   = hdr.n_rows;
    out.mapBase = base;
    out.mapSize = fileSize;
    return out;
}

} // namespace colbin (zero-copy API re-opened)

inline bool zeroCopyEnabled() {
    static const bool v = []() {
        const char* e = ::getenv("GPUDB_NO_ZEROCOPY");
        return !(e && e[0] == '1');
    }();
    return v;
}

inline bool binaryEnabled() {
    const char* e = ::getenv("GPUDB_NO_BINARY");
    return !(e && e[0] == '1');
}

// ===================================================================
// QueryColumns — unified column source for GPU queries.
// Backed by either zero-copy mmap (default) or copy-path fallback.
// GPUDB_NO_ZEROCOPY=1 forces the legacy copy path for A/B testing.
// ===================================================================
class QueryColumns {
public:
    QueryColumns() = default;
    QueryColumns(const QueryColumns&) = delete;
    QueryColumns& operator=(const QueryColumns&) = delete;
    QueryColumns(QueryColumns&& o) noexcept { moveFrom(std::move(o)); }
    QueryColumns& operator=(QueryColumns&& o) noexcept {
        if (this != &o) { releaseOwned(); moveFrom(std::move(o)); }
        return *this;
    }
    ~QueryColumns() { releaseOwned(); }

    size_t       rows()                   const { return rows_; }
    MTL::Buffer* buffer(int col)          const { auto it = buffers_.find(col); return it==buffers_.end()?nullptr:it->second; }
    const int*   ints  (int col)          const { auto it = intPtrs_.find(col);   return it==intPtrs_.end()?nullptr:it->second; }
    const float* floats(int col)          const { auto it = floatPtrs_.find(col); return it==floatPtrs_.end()?nullptr:it->second; }
    const char*  chars (int col)          const { auto it = charPtrs_.find(col);  return it==charPtrs_.end()?nullptr:it->second; }
    bool         zeroCopy()               const { return mapped_.valid(); }

    void _adoptMapped(MappedColumns&& mc, const std::vector<ColSpec>& specs);
    void _adoptCopied(MTL::Device* dev, LoadedColumns&& lc, const std::vector<ColSpec>& specs);

private:
    MappedColumns                              mapped_;
    LoadedColumns                              copied_;
    std::unordered_map<int, MTL::Buffer*>      buffers_;
    std::unordered_map<int, const int*>        intPtrs_;
    std::unordered_map<int, const float*>      floatPtrs_;
    std::unordered_map<int, const char*>       charPtrs_;
    std::vector<MTL::Buffer*>                  ownedBuffers_;
    size_t                                     rows_ = 0;

    void releaseOwned() {
        for (auto* b : ownedBuffers_) if (b) b->release();
        ownedBuffers_.clear();
        buffers_.clear(); intPtrs_.clear(); floatPtrs_.clear(); charPtrs_.clear();
        rows_ = 0;
    }
    void moveFrom(QueryColumns&& o) {
        mapped_       = std::move(o.mapped_);
        copied_       = std::move(o.copied_);
        buffers_      = std::move(o.buffers_);
        intPtrs_      = std::move(o.intPtrs_);
        floatPtrs_    = std::move(o.floatPtrs_);
        charPtrs_     = std::move(o.charPtrs_);
        ownedBuffers_ = std::move(o.ownedBuffers_);
        rows_         = o.rows_;
        o.rows_ = 0;
    }
};

inline void QueryColumns::_adoptMapped(MappedColumns&& mc, const std::vector<ColSpec>& specs) {
    rows_ = mc.nRows;
    for (const auto& s : specs) {
        MTL::Buffer* buf = mc.get(s.columnIndex);
        if (!buf) continue;
        buffers_[s.columnIndex] = buf;
        const char* base = (const char*)buf->contents();
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:       intPtrs_  [s.columnIndex] = (const int*)base;   break;
            case ColType::FLOAT:      floatPtrs_[s.columnIndex] = (const float*)base; break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED: charPtrs_ [s.columnIndex] = (const char*)base;  break;
        }
    }
    mapped_ = std::move(mc);
}

inline void QueryColumns::_adoptCopied(MTL::Device* dev, LoadedColumns&& lc, const std::vector<ColSpec>& specs) {
    copied_ = std::move(lc);
    if (!specs.empty()) {
        const auto& s0 = specs.front();
        switch (s0.type) {
            case ColType::INT:
            case ColType::DATE:       rows_ = copied_.ints(s0.columnIndex).size(); break;
            case ColType::FLOAT:      rows_ = copied_.floats(s0.columnIndex).size(); break;
            case ColType::CHAR1:      rows_ = copied_.chars(s0.columnIndex).size(); break;
            case ColType::CHAR_FIXED: rows_ = s0.fixedWidth > 0
                                              ? copied_.chars(s0.columnIndex).size() / (size_t)s0.fixedWidth
                                              : copied_.chars(s0.columnIndex).size();
                                      break;
        }
    }
    for (const auto& s : specs) {
        size_t bytes = 0;
        const void* src = nullptr;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE: {
                const auto& v = copied_.ints(s.columnIndex);
                bytes = v.size() * sizeof(int); src = v.data();
                intPtrs_[s.columnIndex] = v.data();
                break;
            }
            case ColType::FLOAT: {
                const auto& v = copied_.floats(s.columnIndex);
                bytes = v.size() * sizeof(float); src = v.data();
                floatPtrs_[s.columnIndex] = v.data();
                break;
            }
            case ColType::CHAR1:
            case ColType::CHAR_FIXED: {
                const auto& v = copied_.chars(s.columnIndex);
                bytes = v.size(); src = v.data();
                charPtrs_[s.columnIndex] = v.data();
                break;
            }
        }
        MTL::Buffer* buf = dev->newBuffer(src, bytes, MTL::ResourceStorageModeShared);
        buffers_[s.columnIndex] = buf;
        ownedBuffers_.push_back(buf);
    }
}

inline QueryColumns loadQueryColumns(MTL::Device* device,
                                      const std::string& tblPath,
                                      const std::vector<ColSpec>& specs) {
    QueryColumns qc;
    if (zeroCopyEnabled() && binaryEnabled()) {
        MappedColumns mc = colbin::loadColumnsAsBuffers(device, tblPath, specs);
        if (mc.valid()) {
            size_t bytes = 0;
            for (auto& [k, b] : mc.buffers) if (b) bytes += b->length();
            loadStats().recordBinary(bytes);
            qc._adoptMapped(std::move(mc), specs);
            return qc;
        }
    }
    LoadedColumns lc = loadColumnsMultiAuto(tblPath, specs);
    qc._adoptCopied(device, std::move(lc), specs);
    return qc;
}


inline LoadedColumns loadColumnsMultiAuto(const std::string& filePath,
                                           const std::vector<ColSpec>& specs) {
    // .colbin is the preferred load path; .tbl parsing is the fallback.
    // Set GPUDB_NO_BINARY=1 to force the .tbl parser (useful for diagnostics).
    if (binaryEnabled()) {
        LoadedColumns fromBinary;
        if (colbin::loadColumnsFromBinary(filePath, specs, fromBinary)) {
            size_t bsz = 0; int64_t bmt = 0;
            (void)colbin::statFile(colbin::binaryPath(filePath), bsz, bmt);
            loadStats().recordBinary(bsz);
            return fromBinary;
        }
        static std::unordered_map<std::string, bool> warned;
        if (!warned[filePath]) {
            warned[filePath] = true;
            std::cerr << "[infra] .colbin missing/stale for " << filePath
                      << " — using .tbl parser (run `make colbin-sfN` to accelerate).\n";
        }
    }
    struct stat st{};
    if (::stat(filePath.c_str(), &st) != 0) {
        loadStats().recordText(0);
        return loadColumnsMulti(filePath, specs);
    }
    loadStats().recordText((size_t)st.st_size);
    constexpr off_t THRESHOLD = 1LL * 1024 * 1024 * 1024; // 1 GiB
    // .tbl parse is a one-time ingest cost: time it and report it to
    // loadStats so callers can subtract it from end-to-end timing.
    auto _ingest0 = std::chrono::high_resolution_clock::now();
    LoadedColumns _out = (st.st_size >= THRESHOLD)
        ? loadColumnsMultiMmap(filePath, specs)
        : loadColumnsMulti(filePath, specs);
    auto _ingest1 = std::chrono::high_resolution_clock::now();
    double _ingestMs = std::chrono::duration<double, std::milli>(_ingest1 - _ingest0).count();
    loadStats().recordExcluded(_ingestMs);
    std::cerr << "[tbl-ingest] " << filePath << ": "
              << std::fixed << std::setprecision(1) << _ingestMs
              << " ms (one-time, excluded from e2e)\n";
    return _out;
}
// ===================================================================
// SHARED TABLE LOADERS
// ===================================================================

// --- Nation Table ---
struct NationData {
    std::vector<int>  nationkey;
    std::vector<char> name;      // fixed-width 25 chars per row
    std::vector<int>  regionkey;
    static constexpr int NAME_WIDTH = 25;
};

inline NationData loadNation(const std::string& sf_path, bool with_regionkey = false) {
    NationData d;
    std::vector<ColSpec> specs = {{0, ColType::INT}, {1, ColType::CHAR_FIXED, NationData::NAME_WIDTH}};
    if (with_regionkey) specs.push_back({2, ColType::INT});
    auto cols = loadColumnsMultiAuto(sf_path + "nation.tbl", specs);
    d.nationkey = std::move(cols.ints(0));
    d.name      = std::move(cols.chars(1));
    if (with_regionkey) d.regionkey = std::move(cols.ints(2));
    return d;
}

// --- Region Table ---
struct RegionData {
    std::vector<int>  regionkey;
    std::vector<char> name;
    static constexpr int NAME_WIDTH = 25;
};

inline RegionData loadRegion(const std::string& sf_path) {
    RegionData d;
    auto cols = loadColumnsMultiAuto(sf_path + "region.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, RegionData::NAME_WIDTH}});
    d.regionkey = std::move(cols.ints(0));
    d.name      = std::move(cols.chars(1));
    return d;
}

// --- Supplier Table (basic: suppkey + nationkey) ---
struct SupplierBasic {
    std::vector<int> suppkey;
    std::vector<int> nationkey;
};

inline SupplierBasic loadSupplierBasic(const std::string& sf_path) {
    SupplierBasic d;
    auto cols = loadColumnsMultiAuto(sf_path + "supplier.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    d.suppkey   = std::move(cols.ints(0));
    d.nationkey = std::move(cols.ints(3));
    return d;
}

// --- Single nation key lookup by name ---
inline int findNationKey(const NationData& nat, const std::string& target) {
    for (size_t i = 0; i < nat.nationkey.size(); i++)
        if (trimFixed(nat.name.data(), i, NationData::NAME_WIDTH) == target) return nat.nationkey[i];
    return -1;
}

// ===================================================================
// TIMING & BUFFER HELPERS
// ===================================================================

// --- Detailed Timing Summary (codegen pipeline breakdown) ---
struct DetailedTiming {
    std::string queryName;
    std::string scaleFactor;
    double analyzeMs     = 0.0;
    double planMs        = 0.0;
    double codegenMs     = 0.0;
    double compileMs     = 0.0;
    double psoMs         = 0.0;
    double dataLoadMs    = 0.0;  // ioMs + preprocessMs (umbrella, back-compat)
    double ioMs          = 0.0;  // pure file I/O: mmap/read of .colbin/.tbl into host buffers
    double preprocessMs  = 0.0;  // CPU prep: max-key scans, per-query preprocessing kernels
    double bufferAllocMs = 0.0;
    double gpuTotalMs    = 0.0;
    double postMs        = 0.0;
    std::string loadSource;
    size_t loadBytes     = 0;
    double ingestMs      = 0.0;  // one-time .tbl->column ingest (excluded from e2e)
    std::vector<std::pair<std::string, double>> phaseKernelMs; // per-kernel GPU time
    // C1: per-trial GPU-time distribution (set when --repeat N>1).
    // Reported as p10/p50/p90 + MAD (median absolute deviation).
    // gpuTotalMs is the p50 (median) for back-compat.
    int    gpuTrialsN  = 0;
    double gpuMsP10    = 0.0;
    double gpuMsP90    = 0.0;
    double gpuMsMad    = 0.0;
};

inline void printDetailedTimingSummary(const DetailedTiming& t, bool quiet = false) {
    // Terminology (used in the table and the CSV):
    //   Compile Overhead = SQL analyze + plan + metal codegen/compile + PSO
    //                      (one-time per query; independent of input size)
    //   Data Load        = Pure I/O + CPU Preprocess
    //     Pure I/O       = file read/mmap of column data into host memory
    //     CPU Preprocess = max-key scans, per-query preprocessing kernels
    //   Buffer Setup     = Metal buffer allocation (pointers only, no copy)
    //   GPU Compute      = GPU kernel execution time
    //   CPU Compute      = CPU post-processing (sort/merge/format)
    //   Query Compute    = GPU Compute + CPU Compute  (kernel-only query work)
    //   Query Execution  = CPU Preprocess + Buffer Setup + Query Compute
    //                      (everything actually executing the query, EXCLUDING pure I/O)
    //   End-to-End       = Compile Overhead + Data Load + Buffer Setup + Query Compute
    const double compileOverheadMs = t.analyzeMs + t.planMs + t.codegenMs +
                                     t.compileMs + t.psoMs;
    const double cpuComputeMs      = t.postMs;
    const double gpuComputeMs      = t.gpuTotalMs;
    const double queryComputeMs    = cpuComputeMs + gpuComputeMs;
    // Back-compat: if ioMs/preprocessMs weren't populated, attribute the
    // whole dataLoadMs window to I/O (matches legacy behavior).
    const double ioMs              = (t.ioMs > 0.0 || t.preprocessMs > 0.0)
                                     ? t.ioMs : t.dataLoadMs;
    const double preprocessMs      = (t.ioMs > 0.0 || t.preprocessMs > 0.0)
                                     ? t.preprocessMs : 0.0;
    const double queryExecutionMs  = preprocessMs + t.bufferAllocMs + queryComputeMs;
    const double end2end           = compileOverheadMs + t.dataLoadMs +
                                     t.bufferAllocMs + queryComputeMs;

    if (!quiet) {
        auto bar = []() {
            printf("  +------------------------+-----------------+\n");
        };
        auto head = [](const char* title) {
            printf("  | %-38s |\n", title);
        };
        auto rowMs = [](const char* label, double ms) {
            printf("  | %-22s | %12.3f ms |\n", label, ms);
        };
        auto rowMsHi = [](const char* label, double ms) {
            // Trailing "<<" marker draws the eye to the headline metric.
            printf("  | %-22s | %12.3f ms | <<\n", label, ms);
        };
        auto rowStr = [](const char* label, const char* value) {
            printf("  | %-22s | %15s |\n", label, value);
        };

        printf("\n  Timing Breakdown");
        if (!t.queryName.empty()) printf(" — %s", t.queryName.c_str());
        if (!t.scaleFactor.empty()) printf(" @ %s", t.scaleFactor.c_str());
        printf("\n");

        // --- 1. Compile Overhead (one-time per query) -------------------
        bar();
        head("Compile Overhead  (one-time)");
        bar();
        rowMs("SQL Analyze",    t.analyzeMs);
        rowMs("Plan Build",     t.planMs);
        rowMs("Metal Codegen",  t.codegenMs);
        rowMs("Metal Compile",  t.compileMs);
        rowMs("PSO Creation",   t.psoMs);
        rowMs("  subtotal",     compileOverheadMs);
        bar();

        // --- 2. Data Load (host I/O) ------------------------------------
        {
            char title[64];
            const char* src = t.loadSource.empty() ? "colbin" : t.loadSource.c_str();
            snprintf(title, sizeof(title), "Data Load  (I/O, %s)", src);
            head(title);
        }
        bar();
        rowMs("Load Time (total)", t.dataLoadMs);
        rowMs("  Pure I/O",        ioMs);
        rowMs("  CPU Preprocess",  preprocessMs);
        if (t.loadBytes > 0) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f MiB", (double)t.loadBytes / (1024.0 * 1024.0));
            rowStr("Bytes", buf);
        }
        if (t.loadBytes > 0 && ioMs > 0.0) {
            const double mibps = ((double)t.loadBytes / (1024.0 * 1024.0)) / (ioMs / 1000.0);
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f MiB/s", mibps);
            rowStr("I/O Throughput", buf);
        }
        if (t.ingestMs > 0.0) {
            const char* show = ::getenv("GPUDB_SHOW_INGEST");
            if (show && show[0] == '1') {
                rowMs("tbl->col ingest (1x)", t.ingestMs);
            }
        }
        bar();

        // --- 3. Query Execution (the actual compute) --------------------
        head("Query Execution  (actual compute)");
        bar();
        rowMs("Buffer Setup",   t.bufferAllocMs);
        for (const auto& [name, ms] : t.phaseKernelMs) {
            char label[64];
            snprintf(label, sizeof(label), "  GPU kernel %s", name.c_str());
            rowMs(label, ms);
        }
        rowMs("GPU Compute",    gpuComputeMs);
        if (t.gpuTrialsN > 1) {
            char buf[64];
            snprintf(buf, sizeof(buf),
                     "p10=%.2f p50=%.2f p90=%.2f mad=%.2f (n=%d)",
                     t.gpuMsP10, gpuComputeMs, t.gpuMsP90, t.gpuMsMad,
                     t.gpuTrialsN);
            rowStr("  GPU dist", buf);
        }
        rowMs("CPU Compute",    cpuComputeMs);
        rowMs("Query Compute",     queryComputeMs);
        rowMsHi("Query Execution",   queryExecutionMs);
        bar();

        // --- 4. Totals --------------------------------------------------
        head("Totals");
        bar();
        rowMs("Compile Overhead",  compileOverheadMs);
        rowMs("Pure I/O",          ioMs);
        rowMs("Query Execution",   queryExecutionMs);
        rowMsHi("End-to-End",      end2end);
        bar();
    }

    // Machine-readable single line for CSV harvesting (always emitted).
    // Kept back-compatible with prior field order. Renamed fields (same slot):
    //   gpu_ms      -> gpu_compute_ms
    //   post_ms     -> cpu_compute_ms
    //   cpu_codegen -> compile_overhead_ms
    //   cpu_total   -> compile_overhead + data_load + buffer_setup + cpu_compute
    //                  (retained for legacy scripts; excludes GPU)
    // Appended: query_compute_ms = cpu_compute_ms + gpu_compute_ms
    double loadMibps = (ioMs > 0.0 && t.loadBytes > 0)
        ? ((double)t.loadBytes / (1024.0*1024.0)) / (ioMs / 1000.0)
        : 0.0;
    const double cpuTotalLegacy = compileOverheadMs + t.dataLoadMs +
                                  t.bufferAllocMs + cpuComputeMs;
    // CSV trailer (appended fields, back-compat preserved): io_ms,
    // preprocess_ms, query_execution_ms.
    printf("TIMING_CSV,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%zu,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
           t.scaleFactor.c_str(), t.queryName.c_str(),
           t.analyzeMs, t.planMs, t.codegenMs, t.compileMs, t.psoMs,
           t.dataLoadMs, t.bufferAllocMs, gpuComputeMs, cpuComputeMs,
           compileOverheadMs, cpuTotalLegacy, end2end,
           t.loadSource.empty() ? "none" : t.loadSource.c_str(),
           t.loadBytes, loadMibps, t.ingestMs, queryComputeMs,
           // C1: per-trial GPU-time distribution (zeros if --repeat 1)
           t.gpuTrialsN, t.gpuMsP10, t.gpuMsP90, t.gpuMsMad,
           // I/O vs preprocess split + query-execution metric
           ioMs, preprocessMs, queryExecutionMs);
}

