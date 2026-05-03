// Codegen standalone entry point
// Usage: GPUDBCodegen [sf1|sf10|sf100] q<N>
//
// This binary performs SQL → AnalyzedQuery → MetalQueryPlan → operators → Metal → GPU
// for all 22 TPC-H queries via runtime code generation.

#include "../src/infra.h"
#include "query_analyzer.h"
#include "runtime_compiler.h"
#include "metal_plan_builder.h"
#include "metal_generic_executor.h"
#include "query_preprocessing.h"
#include "chunked_colbin_loader.h"
#include "tpch_schema.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <optional>
#include <future>
#include <thread>
#include "../third_party/nlohmann/json.hpp"

// ===================================================================
// Experiment flags (set in main, read by runCodegenQuery)
// ===================================================================
static int  g_warmup            = 3;     // --warmup N
static int  g_repeat            = 1;     // --repeat N
static bool g_csv               = false; // --csv  (suppress human-readable breakdown)
static int  g_tgSizeOverride    = 0;     // --threadgroup-size N (0 = use plan default)
static bool g_autotuneTg        = false; // --autotune-tg  (per-query global TG sweep)
static bool g_autotuneTgPerPhase= false; // --autotune-tg-per-phase (per-kernel TG)
static bool g_noPipelineCache   = false; // --no-pipeline-cache
static bool g_fastMath          = false; // --fastmath
static bool g_printPlan         = false; // --print-plan
static std::string g_dumpMslDir;         // --dump-msl PATH (directory or file template)
static std::string g_checkDir;           // --check DIR  (compare result vs DIR/<query>_<sf>.csv)
static std::string g_saveGoldenDir;      // --save-golden DIR
static double g_checkAbsTol = 1e-2;      // --check-abs-tol N
static double g_checkRelTol = 1e-4;      // --check-rel-tol N
static int    g_checkExitCode = 0;       // accumulated: nonzero if any --check failed
static size_t g_chunkRows = 0;           // --chunk N[K|M|G], 0 = full-table mode
static size_t g_chunkRowsExplicit = 0;    // user-set --chunk value (0 = unset);
                                          // g_chunkRows above is the *effective*
                                          // value, possibly raised by the per-query
                                          // auto-chunk trigger and reset between
                                          // queries to g_chunkRowsExplicit.
static bool   g_chunkDoubleBuffer = true;// --no-db uses one reusable chunk slot

// Compare two canonical CSV blobs with float tolerance.
// Column matching is done by name (header row), so queries whose GPU output
// uses different column names (e.g. "bucket") or has extra/fewer columns than
// the DuckDB golden are still validated for the columns they share.
// Returns empty string on full match; a short diff message otherwise.
static std::string compareCanonical(const std::string& got, const std::string& expected,
                                    double absTol, double relTol) {
    auto splitLines = [](const std::string& s) {
        std::vector<std::string> lines;
        std::istringstream is(s);
        std::string ln;
        while (std::getline(is, ln)) {
            // strip trailing CR so Windows-style golden files compare cleanly
            if (!ln.empty() && ln.back() == '\r') ln.pop_back();
            lines.push_back(ln);
        }
        // drop trailing empty line produced by trailing newline
        while (!lines.empty() && lines.back().empty()) lines.pop_back();
        return lines;
    };
    auto splitCsv = [](const std::string& ln) {
        std::vector<std::string> out;
        std::string cur;
        bool inQuote = false;
        for (size_t i = 0; i < ln.size(); i++) {
            char c = ln[i];
            if (inQuote) {
                if (c == '"') {
                    if (i + 1 < ln.size() && ln[i+1] == '"') { cur += '"'; i++; } // escaped ""
                    else inQuote = false;
                } else {
                    cur += c;
                }
            } else {
                if (c == '"') {
                    inQuote = true;
                } else if (c == ',') {
                    out.push_back(cur); cur.clear();
                } else {
                    cur += c;
                }
            }
        }
        out.push_back(cur);
        return out;
    };
    auto isNumber = [](const std::string& s, double& out) {
        if (s.empty()) return false;
        char* end = nullptr;
        out = std::strtod(s.c_str(), &end);
        return end != s.c_str() && *end == '\0';
    };

    auto aLines = splitLines(got);
    auto bLines = splitLines(expected);
    if (aLines.empty() && bLines.empty()) return "";
    if (aLines.empty()) {
        return "got 0 rows, expected " + std::to_string(bLines.size() - 1) + " data rows";
    }
    if (bLines.empty()) {
        return "got " + std::to_string(aLines.size() - 1) + " data rows, expected 0";
    }

    // ---------------------------------------------------------------
    // Parse headers and build column-name → index maps
    // ---------------------------------------------------------------
    auto aHdr = splitCsv(aLines[0]);
    auto bHdr = splitCsv(bLines[0]);

    // Map: column name → (index-in-got, index-in-golden)
    std::vector<std::pair<size_t,size_t>> sharedCols;
    for (size_t ai = 0; ai < aHdr.size(); ai++) {
        for (size_t bi = 0; bi < bHdr.size(); bi++) {
            if (aHdr[ai] == bHdr[bi]) {
                sharedCols.push_back({ai, bi});
                break;
            }
        }
    }

    // If there are no columns in common, report schema mismatch.
    if (sharedCols.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "schema mismatch: got cols=[%s] expected cols=[%s]",
                 aLines[0].c_str(), bLines[0].c_str());
        return buf;
    }

    // ---------------------------------------------------------------
    // Row-count check (header excluded)
    // ---------------------------------------------------------------
    size_t aData = aLines.size() - 1;
    size_t bData = bLines.size() - 1;
    if (aData != bData) {
        char buf[128];
        snprintf(buf, sizeof(buf), "row count mismatch: got=%zu expected=%zu", aData, bData);
        return buf;
    }

    // ---------------------------------------------------------------
    // Per-row comparison over shared columns
    // ---------------------------------------------------------------
    for (size_t row = 0; row < aData; row++) {
        auto aRow = splitCsv(aLines[row + 1]);
        auto bRow = splitCsv(bLines[row + 1]);
        for (auto [ai, bi] : sharedCols) {
            if (ai >= aRow.size() || bi >= bRow.size()) continue;
            const std::string& av = aRow[ai];
            const std::string& bv = bRow[bi];
            if (av == bv) continue;
            double va, vb;
            if (isNumber(av, va) && isNumber(bv, vb)) {
                double diff = std::fabs(va - vb);
                double tol = absTol + relTol * std::max(std::fabs(va), std::fabs(vb));
                if (diff <= tol) continue;
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "row %zu col '%s': %s vs %s (diff=%.6g tol=%.6g)",
                         row + 1, aHdr[ai].c_str(), av.c_str(), bv.c_str(), diff, tol);
                return buf;
            }
            // Both non-numeric strings — must match exactly
            char buf[256];
            snprintf(buf, sizeof(buf), "row %zu col '%s': '%s' vs '%s'",
                     row + 1, aHdr[ai].c_str(), av.c_str(), bv.c_str());
            return buf;
        }
    }
    return "";
}

// Convert YYYYMMDD integer to "YYYY-MM-DD" string (TPC-H date column format)
static std::string intDateToStr(int d) {
    char buf[12];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", d / 10000, (d / 100) % 100, d % 100);
    return buf;
}

using codegen::g_q2Post;
using codegen::g_q16Post;
using codegen::g_q18Post;
using codegen::g_q20Post;
using codegen::g_q21Post;

static ColSpec colSpecFor(const codegen::ColumnDef& cdef) {
    ColType type = ColType::INT;
    switch (cdef.type) {
        case codegen::DataType::INT:        type = ColType::INT; break;
        case codegen::DataType::FLOAT:      type = ColType::FLOAT; break;
        case codegen::DataType::DATE:       type = ColType::DATE; break;
        case codegen::DataType::CHAR1:      type = ColType::CHAR1; break;
        case codegen::DataType::CHAR_FIXED: type = ColType::CHAR_FIXED; break;
    }
    return ColSpec(cdef.index, type, cdef.fixedWidth);
}

static bool parseRowCountWithSuffix(const std::string& text, size_t& out) {
    if (text.empty()) return false;
    char suffix = text.back();
    size_t multiplier = 1;
    std::string digits = text;
    if (suffix == 'k' || suffix == 'K' || suffix == 'm' || suffix == 'M' ||
        suffix == 'g' || suffix == 'G') {
        digits.pop_back();
        if (suffix == 'k' || suffix == 'K') multiplier = 1000ULL;
        if (suffix == 'm' || suffix == 'M') multiplier = 1000ULL * 1000ULL;
        if (suffix == 'g' || suffix == 'G') multiplier = 1000ULL * 1000ULL * 1000ULL;
    }
    if (digits.empty()) return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(digits.c_str(), &end, 10);
    if (errno != 0 || end == digits.c_str() || *end != '\0' || value == 0) return false;
    out = (size_t)value * multiplier;
    return out > 0;
}

// ===================================================================
// Max-key scan strategies (experiment).
//   GPUDB_MAXKEY_MODE = serial (default) | parallel | cache
// ===================================================================
enum class MaxKeyMode { Serial, Parallel, Cache };
static MaxKeyMode currentMaxKeyMode() {
    static MaxKeyMode m = []() {
        const char* e = ::getenv("GPUDB_MAXKEY_MODE");
        if (!e) return MaxKeyMode::Cache;  // default: cached parallel max
        std::string v(e);
        if (v == "serial")   return MaxKeyMode::Serial;
        if (v == "parallel") return MaxKeyMode::Parallel;
        if (v == "cache")    return MaxKeyMode::Cache;
        return MaxKeyMode::Cache;
    }();
    return m;
}

// Approach B: parallel max via std::async. Splits the column into
// nThreads ranges; each thread does a local std::max. Page-fault work
// is parallelized across cores, which is the actual bottleneck on
// cold-mapped colbins (zero-copy mmap means first-touch faults).
static int parallelMaxInt(const int* data, size_t n) {
    if (n == 0) return 0;
    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    const size_t nThreads = std::min<size_t>(hw, std::max<size_t>(1, n / 65536));
    if (nThreads <= 1) {
        int m = 0;
        for (size_t i = 0; i < n; i++) if (data[i] > m) m = data[i];
        return m;
    }
    std::vector<std::future<int>> futs;
    futs.reserve(nThreads);
    const size_t chunk = (n + nThreads - 1) / nThreads;
    for (size_t t = 0; t < nThreads; t++) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        futs.push_back(std::async(std::launch::async, [data, lo, hi]() {
            int m = 0;
            for (size_t i = lo; i < hi; i++) if (data[i] > m) m = data[i];
            return m;
        }));
    }
    int m = 0;
    for (auto& f : futs) m = std::max(m, f.get());
    return m;
}

// Approach C: persistent on-disk cache of per-column max values.
// Sidecar file: <dataset_path>/.maxkeys.json. Keyed by colbin file
// (path basename, size, mtime) + column index. On hit we return the
// cached max with zero scan; on miss we fall through to the parallel
// scan and write the result back.
struct MaxKeyCacheEntry {
    std::string file;     // basename of .colbin
    uint64_t size;
    int64_t  mtime_ns;
    int      columnIndex;
    int      maxValue;
};
static std::string maxKeyCachePath() {
    return g_dataset_path + ".maxkeys.json";
}
static bool loadMaxKeyCache(std::vector<MaxKeyCacheEntry>& out) {
    out.clear();
    std::ifstream f(maxKeyCachePath());
    if (!f) return false;
    try {
        nlohmann::json j; f >> j;
        if (!j.is_array()) return false;
        for (const auto& e : j) {
            MaxKeyCacheEntry x;
            x.file        = e.at("file").get<std::string>();
            x.size        = e.at("size").get<uint64_t>();
            x.mtime_ns    = e.at("mtime_ns").get<int64_t>();
            x.columnIndex = e.at("col").get<int>();
            x.maxValue    = e.at("max").get<int>();
            out.push_back(std::move(x));
        }
        return true;
    } catch (...) { return false; }
}
static void saveMaxKeyCache(const std::vector<MaxKeyCacheEntry>& entries) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& e : entries) {
        j.push_back({
            {"file", e.file},
            {"size", e.size},
            {"mtime_ns", e.mtime_ns},
            {"col", e.columnIndex},
            {"max", e.maxValue},
        });
    }
    std::ofstream f(maxKeyCachePath());
    if (!f) return;
    f << j.dump(2);
}
static bool cacheLookup(const std::vector<MaxKeyCacheEntry>& cache,
                        const std::string& file, uint64_t size,
                        int64_t mtime_ns, int columnIndex, int& out) {
    for (const auto& e : cache) {
        if (e.columnIndex == columnIndex && e.file == file &&
            e.size == size && e.mtime_ns == mtime_ns) {
            out = e.maxValue;
            return true;
        }
    }
    return false;
}
// Compute one column's max under the active mode. For Cache mode we
// look up by colbin metadata; on miss we run the parallel scan and
// queue an entry for write-back. The lookup metadata is the colbin
// file (the source of truth for the underlying mmap'd bytes).
static int computeColMax(const int* data, size_t n,
                         const std::string& tblPath, int columnIndex,
                         std::vector<MaxKeyCacheEntry>& cacheRead,
                         std::vector<MaxKeyCacheEntry>& cacheWrite,
                         bool& cacheDirty) {
    const MaxKeyMode mode = currentMaxKeyMode();
    if (mode == MaxKeyMode::Serial) {
        int m = 0;
        for (size_t i = 0; i < n; i++) if (data[i] > m) m = data[i];
        return m;
    }
    if (mode == MaxKeyMode::Parallel) {
        return parallelMaxInt(data, n);
    }
    // Cache mode: check sidecar by colbin file metadata.
    const std::string cp = colbin::binaryPath(tblPath);
    size_t fsz = 0; int64_t fmt = 0;
    if (colbin::statFile(cp, fsz, fmt)) {
        const std::string base = cp.substr(cp.find_last_of('/') + 1);
        int hit = 0;
        if (cacheLookup(cacheRead, base, (uint64_t)fsz, fmt, columnIndex, hit)) {
            return hit;
        }
        int m = parallelMaxInt(data, n);
        cacheWrite.push_back({base, (uint64_t)fsz, fmt, columnIndex, m});
        cacheDirty = true;
        return m;
    }
    return parallelMaxInt(data, n);
}

static void registerMaxKeySymbols(
    codegen::MetalGenericExecutor& executor,
    const std::vector<codegen::LoadedQueryTable>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const codegen::TPCHSchema& schema) {
    int maxCk = 0, maxSk = 0, maxOk = 0, maxPk = 0;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    if (currentMaxKeyMode() == MaxKeyMode::Cache) {
        loadMaxKeyCache(cacheRead);
    }
    for (auto& [tblName, cols] : loadedTables) {
        const auto tableIt = tableCols.find(tblName);
        if (tableIt == tableCols.end()) continue;
        const auto& tdef = schema.table(tblName);
        size_t nRows = cols.rows();
        for (const auto& colName : tableIt->second) {
            auto& cdef = tdef.col(colName);
            if (cdef.type != codegen::DataType::INT && cdef.type != codegen::DataType::DATE)
                continue;
            const int* data = cols.ints(cdef.index);
            if (!data) continue;
            const std::string tblPath = g_dataset_path + tblName + ".tbl";
            int colMax = 0;
            if (colName == "c_custkey" || colName == "o_custkey" ||
                colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey" ||
                colName == "o_orderkey" || colName == "l_orderkey" ||
                colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey") {
                colMax = computeColMax(data, nRows, tblPath, cdef.index,
                                       cacheRead, cacheWrite, cacheDirty);
            }
            if (colName == "c_custkey" || colName == "o_custkey")
                maxCk = std::max(maxCk, colMax);
            else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
                maxSk = std::max(maxSk, colMax);
            else if (colName == "o_orderkey" || colName == "l_orderkey")
                maxOk = std::max(maxOk, colMax);
            else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
                maxPk = std::max(maxPk, colMax);
        }
    }
    if (cacheDirty) {
        // Merge new entries with existing cache (so we don't lose other columns).
        for (auto& e : cacheWrite) cacheRead.push_back(std::move(e));
        saveMaxKeyCache(cacheRead);
    }
    executor.registerSymbol("maxCustkey", maxCk + 1);
    executor.registerSymbol("maxSuppkey", maxSk + 1);
    executor.registerSymbol("maxOrderkey", maxOk + 1);
    executor.registerSymbol("maxPartkey", maxPk + 1);
}

// In chunked execution mode the stream table is intentionally absent from
// `loadedTables`, so its key columns never contribute to the max-key scan
// performed by registerMaxKeySymbols(). Without this extension Q15/Q18
// (single-table aggregates keyed by l_suppkey / l_orderkey) collapse to
// maxSuppkey == 1 / maxOrderkey == 1 and write OOB. We perform a one-time
// in-memory load of just the int key columns of the stream table from its
// .colbin and merge the per-column max into the already-registered symbols.
static void extendMaxKeysFromStreamColbin(
    codegen::MetalGenericExecutor& executor,
    const std::string& streamTblPath,
    const std::set<std::string>& streamCols,
    const codegen::TPCHSchema& schema,
    const std::string& streamTable) {
    if (streamTable.empty()) return;
    const auto& tdef = schema.table(streamTable);
    std::vector<std::pair<std::string, ColSpec>> intSpecs;
    for (const auto& colName : streamCols) {
        const auto& cdef = tdef.col(colName);
        if (cdef.type != codegen::DataType::INT && cdef.type != codegen::DataType::DATE)
            continue;
        if (colName != "c_custkey" && colName != "o_custkey" &&
            colName != "s_suppkey" && colName != "l_suppkey" && colName != "ps_suppkey" &&
            colName != "o_orderkey" && colName != "l_orderkey" &&
            colName != "p_partkey"  && colName != "l_partkey"  && colName != "ps_partkey")
            continue;
        intSpecs.emplace_back(colName, colSpecFor(cdef));
    }
    if (intSpecs.empty()) return;

    std::vector<ColSpec> specs;
    specs.reserve(intSpecs.size());
    for (const auto& [_, s] : intSpecs) specs.push_back(s);

    LoadedColumns parsed;
    if (!colbin::loadColumnsFromBinary(streamTblPath, specs, parsed)) {
        std::cerr << "extendMaxKeysFromStreamColbin: failed to read colbin for "
                  << streamTable << " at " << streamTblPath
                  << " (max-key symbols may be wrong)\n";
        return;
    }

    int maxCk = 0, maxSk = 0, maxOk = 0, maxPk = 0;
    std::vector<MaxKeyCacheEntry> cacheRead, cacheWrite;
    bool cacheDirty = false;
    if (currentMaxKeyMode() == MaxKeyMode::Cache) {
        loadMaxKeyCache(cacheRead);
    }
    for (const auto& [colName, spec] : intSpecs) {
        const auto& v = parsed.ints(spec.columnIndex);
        if (v.empty()) continue;
        int colMax = computeColMax(v.data(), v.size(), streamTblPath, spec.columnIndex,
                                   cacheRead, cacheWrite, cacheDirty);
        if (colName == "c_custkey" || colName == "o_custkey") maxCk = std::max(maxCk, colMax);
        else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
            maxSk = std::max(maxSk, colMax);
        else if (colName == "o_orderkey" || colName == "l_orderkey")
            maxOk = std::max(maxOk, colMax);
        else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
            maxPk = std::max(maxPk, colMax);
    }
    if (cacheDirty) {
        for (auto& e : cacheWrite) cacheRead.push_back(std::move(e));
        saveMaxKeyCache(cacheRead);
    }

    // sizeResolver_::registerSymbol overwrites; query each previously-registered
    // value and re-register the max with our stream-table contribution.
    auto bump = [&](const char* name, int streamMax) {
        if (streamMax <= 0) return;
        size_t cur = 0;
        if (executor.tryGetSymbol(name, cur)) {
            executor.registerSymbol(name, std::max(cur, (size_t)streamMax + 1));
        } else {
            executor.registerSymbol(name, (size_t)streamMax + 1);
        }
    };
    bump("maxCustkey",  maxCk);
    bump("maxSuppkey",  maxSk);
    bump("maxOrderkey", maxOk);
    bump("maxPartkey",  maxPk);
}


// ===================================================================
// Peek at a .colbin file header to read row count + file size without
// mapping the full file.  Returns false if file is absent / invalid.
// ===================================================================

static bool peekColbinHeader(const std::string& path,
                              uint64_t& out_n_rows, uint64_t& out_file_size) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return false;
    out_file_size = (uint64_t)st.st_size;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    colbin::FileHeader hdr{};
    bool ok = (fread(&hdr, sizeof(hdr), 1, f) == 1);
    fclose(f);
    if (!ok) return false;
    if (memcmp(hdr.magic, colbin::MAGIC, 8) != 0) return false;
    if (hdr.version != colbin::VERSION) return false;
    out_n_rows = hdr.n_rows;
    return true;
}

// Largest .colbin table in tableCols (by file size) becomes the stream table
// for chunked execution. Returns empty string if no .colbin files found.
static std::string autoDetectStreamTable(
        const std::map<std::string, std::set<std::string>>& tableCols) {
    std::string best;
    uint64_t bestSize = 0;
    for (const auto& [tName, _cols] : tableCols) {
        uint64_t nr = 0, fsz = 0;
        if (peekColbinHeader(g_dataset_path + tName + ".colbin", nr, fsz) &&
                fsz > bestSize) {
            bestSize = fsz;
            best = tName;
        }
    }
    // If no .colbin found, fall back to common heuristic.
    if (best.empty()) {
        if (tableCols.count("lineitem")) return "lineitem";
        if (tableCols.count("orders"))   return "orders";
        return tableCols.empty() ? std::string{} : tableCols.begin()->first;
    }
    return best;
}

// ===================================================================
static bool runCodegenQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                            const std::string& sql, const std::string& queryName) {
    if (!g_csv) printf("\n=== Codegen: %s ===\n", queryName.c_str());
    // Reset effective chunk size to whatever the user explicitly asked for.
    // Without this, an earlier query's auto-chunk decision would leak into
    // the next query when the binary is invoked with `all` or `mball`.
    g_chunkRows = g_chunkRowsExplicit;
    try {
        using clk = std::chrono::high_resolution_clock;
        auto elapsedMs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        DetailedTiming timing{};
        timing.queryName = queryName;
        {
            // Derive short SF label from g_dataset_path (e.g. "data/SF-1/" → "SF1")
            const std::string& p = g_dataset_path;
            auto s = p.find("SF-");
            if (s != std::string::npos) {
                std::string digits;
                for (size_t i = s + 3; i < p.size() && isdigit((unsigned char)p[i]); i++)
                    digits += p[i];
                timing.scaleFactor = "SF" + digits;
            }
        }

        // 1. Analyze SQL
        auto tAnalyze0 = clk::now();
        codegen::AnalyzedQuery analyzed;
        try {
            analyzed = codegen::analyzeSQL(sql);
        } catch (...) {
            // Name-based builders don't need analyzed query
        }
        timing.analyzeMs = elapsedMs(tAnalyze0, clk::now());

        // 2. Build operator-based plan
        auto tPlan0 = clk::now();
        auto maybePlan = codegen::buildMetalPlan(analyzed, queryName);
        if (!maybePlan) {
            std::cerr << "Codegen: query pattern not yet supported for " << queryName << std::endl;
            return false;
        }
        auto& plan = *maybePlan;
        plan.name = queryName;
        timing.planMs = elapsedMs(tPlan0, clk::now());

        // ----------------------------------------------------------------
        // Data-larger-than-memory (DLM) safety gate.
        //   * Hard error if the user explicitly passed --chunk for a query
        //     whose plan is not yet certified chunkable (see DOCUMENTATION
        //     §9.4 — only Q1/Q4/Q6/Q12/Q13/Q14/Q19 + microbenchmarks today).
        //   * Otherwise force g_chunkRows back to 0 so the auto-chunk
        //     trigger below cannot silently engage and produce wrong
        //     output for joins / sorts / non-associative aggregates.
        // ----------------------------------------------------------------
        if (!plan.chunkable) {
            if (g_chunkRowsExplicit > 0) {
                std::cerr << "Codegen: " << queryName
                          << " does not support chunked execution yet "
                             "(--chunk is unsafe for this query — see "
                             "DOCUMENTATION.md §9.4).\n";
                return false;
            }
            g_chunkRows = 0;  // suppress auto-chunk below
        }

        // Experiment override: --threadgroup-size N replaces every phase's TG size.
        if (g_tgSizeOverride > 0) {
            for (auto& ph : plan.phases) ph.threadgroupSize = g_tgSizeOverride;
        }

        // Experiment introspection: --print-plan dumps phase summary.
        if (g_printPlan) {
            printf("\n--- MetalQueryPlan: %s ---\n", plan.name.c_str());
            printf("  helpers           : %zu\n", plan.helpers.size());
            printf("  phases            : %zu\n", plan.phases.size());
            for (size_t i = 0; i < plan.phases.size(); i++) {
                const auto& ph = plan.phases[i];
                printf("    [%zu] kernel=%s  tg=%d  singleThread=%s  bitmapReads=%zu  scalarParams=%zu  extraBuffers=%zu\n",
                       i, ph.name.c_str(), ph.threadgroupSize,
                       ph.singleThread ? "true" : "false",
                       ph.bitmapReads.size(), ph.scalarParams.size(), ph.extraBuffers.size());
            }
            if (plan.cpuSort) {
                printf("  cpuSort.keys      : %zu  limit=%d\n",
                       plan.cpuSort->keys.size(), plan.cpuSort->limit);
            }
            printf("---\n");
        }

        // 3. Generate Metal source via producer-consumer operators
        auto tCodegen0 = clk::now();
        auto cg = codegen::generateFromPlan(plan);
        std::string metalSource = cg.print();
        timing.codegenMs = elapsedMs(tCodegen0, clk::now());

        if (!g_csv) {
            printf("Generated Metal source (%zu bytes, %d phase(s))\n",
                   metalSource.size(), cg.phaseCount());
        }

        // Debug: dump generated source to file
        {
            std::string dumpDir = g_dumpMslDir.empty() ? "debug" : g_dumpMslDir;
            std::string path = dumpDir + "/codegen_debug_" + queryName + ".metal";
            std::ofstream dbg(path);
            dbg << metalSource;
            if (!g_csv) printf("  (written to %s)\n", path.c_str());
        }

        // 4. Compile Metal source → MTLLibrary
        auto tCompile0 = clk::now();
        codegen::RuntimeCompiler compiler(device);
        auto* library = compiler.compile(metalSource);
        timing.compileMs = elapsedMs(tCompile0, clk::now());
        if (!library) {
            std::cerr << "Codegen: Metal compilation failed" << std::endl;
            return false;
        }

        // Build CompiledQuery with PSOs for each phase
        auto tPso0 = clk::now();
        codegen::RuntimeCompiler::CompiledQuery compiled;
        compiled.library = library;
        for (const auto& phase : cg.getPhases()) {
            auto* pso = compiler.getPipeline(library, phase.name);
            if (!pso) {
                std::cerr << "Codegen: PSO creation failed for " << phase.name << std::endl;
                return false;
            }
            compiled.pipelines.push_back(pso);
            compiled.kernelNames.push_back(phase.name);
        }
        timing.psoMs = elapsedMs(tPso0, clk::now());

        const auto& schema = codegen::TPCHSchema::instance();
        // Collect columns needed per table from all phases
        std::map<std::string, std::set<std::string>> tableCols;
        for (const auto& phase : cg.getPhases()) {
            for (const auto& b : phase.bindings) {
                if (b.kind == codegen::MetalParamKind::TableData && !b.tableName.empty()) {
                    tableCols[b.tableName].insert(b.name);
                }
            }
        }

        // 5. Load data: full-table zero-copy/copy or bounded chunked colbin.
        loadStats().reset();  // track per-query load source + byte count
        // Auto-enable chunked execution when dataset exceeds available RAM
        // or the GPU working-set budget. Skip entirely for non-chunkable
        // plans (see DLM gate above) so we never auto-engage chunking on a
        // query that would produce wrong results.
        //
        // Adaptive sizing strategy:
        //   * streamBytesPerRow = sum(elemBytes) for *projected* columns of
        //     the stream table only — narrower projections yield wider
        //     chunks and amortise launch overhead better.
        //   * residentBytes     = sum(nrows * sum(elemBytes(projected)))
        //     across non-stream tables that stay fully resident in shared
        //     buffers + GPU side hash maps.
        //   * budget            = min(physRAM, gpuWorkingSet) * fraction
        //                         - residentBytes
        //   * chunkRows         = floor(budget / (streamBytesPerRow * slots))
        //   * Clamped to [256K, totalStreamRows].
        //
        // Note: the trigger uses physMemBytes (not vm_statistics64's
        // free+inactive+speculative). On Apple Silicon the GPU shares
        // system RAM and the kernel evicts file-backed/cached pages on
        // demand, so reclaimable-only accounting is far too pessimistic
        // (e.g. reports ~5 GiB on a 16 GiB Mac with normal browser/IDE
        // usage). Likewise, the working-set estimate compares *projected*
        // bytes (residentBytes + streamRows * streamBytesPerRow), not
        // full .colbin file sizes, since unprojected columns are never
        // loaded into GPU buffers.
        // Enter the budget block whenever the plan is chunkable. We use
        // the computed working-set both to *engage* chunking when it does
        // not fit (auto path, g_chunkRows starts at 0) and to *downgrade*
        // an explicit `--chunk N` request to direct load when it clearly
        // does fit (explicit path, g_chunkRows starts > 0). The downgrade
        // can be disabled by setting GPUDB_FORCE_CHUNK=1 — useful for
        // benchmarking the chunked path on small datasets.
        if (plan.chunkable) {
            const std::string autoStreamTable = autoDetectStreamTable(tableCols);
            if (!autoStreamTable.empty()) {
                // --- Memory budgets ---------------------------------------
                uint64_t physMemBytes = 0;
                {
                    size_t len = sizeof(physMemBytes);
                    sysctlbyname("hw.memsize", &physMemBytes, &len, nullptr, 0);
                }
                uint64_t availMemBytes = physMemBytes;
                {
                    vm_statistics64_data_t vmstat{};
                    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
                    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                         (host_info64_t)&vmstat, &count) == KERN_SUCCESS) {
                        vm_size_t pageSize = vm_kernel_page_size;
                        uint64_t reclaimable =
                            ((uint64_t)vmstat.free_count +
                             (uint64_t)vmstat.inactive_count +
                             (uint64_t)vmstat.speculative_count) * (uint64_t)pageSize;
                        availMemBytes = std::min(reclaimable, physMemBytes);
                    }
                }
                uint64_t gpuBudgetBytes = (uint64_t)device->recommendedMaxWorkingSetSize();
                if (gpuBudgetBytes == 0) gpuBudgetBytes = physMemBytes;
                // Use physMemBytes (not availMemBytes) for the trigger:
                // on UMA, file-backed pages are evicted on demand, so the
                // VM "free+inactive+speculative" estimate is too tight.
                // availMemBytes is kept only for the diagnostic printout.
                uint64_t totalBudget = std::min(physMemBytes, gpuBudgetBytes);

                // --- Per-column elem-byte helper --------------------------
                auto elemBytes = [&](const codegen::ColumnDef& c) -> uint64_t {
                    switch (c.type) {
                        case codegen::DataType::INT:
                        case codegen::DataType::DATE:
                        case codegen::DataType::FLOAT:      return 4;
                        case codegen::DataType::CHAR1:      return 1;
                        case codegen::DataType::CHAR_FIXED: return (uint64_t)c.fixedWidth;
                    }
                    return 0;
                };
                auto projectedBytesPerRow = [&](const std::string& tName,
                                                 const std::set<std::string>& cols) {
                    uint64_t bpr = 0;
                    const auto& tdef = schema.table(tName);
                    for (const auto& cn : cols) bpr += elemBytes(tdef.col(cn));
                    return bpr;
                };

                // --- Total dataset size (full file sizes, diagnostic) ----
                uint64_t totalDataBytes = 0;
                for (const auto& [tName, _cols] : tableCols) {
                    uint64_t nr = 0, fsz = 0;
                    if (peekColbinHeader(g_dataset_path + tName + ".colbin", nr, fsz))
                        totalDataBytes += fsz;
                }

                // --- Resident bytes (non-stream tables, projected only) ---
                uint64_t residentBytes = 0;
                for (const auto& [tName, cols] : tableCols) {
                    if (tName == autoStreamTable) continue;
                    uint64_t nr = 0, fsz = 0;
                    if (!peekColbinHeader(g_dataset_path + tName + ".colbin", nr, fsz))
                        continue;
                    residentBytes += nr * projectedBytesPerRow(tName, cols);
                }

                // --- Projected stream bytes (stream table, projected only)
                uint64_t streamProjectedBytes = 0;
                {
                    uint64_t nr = 0, fsz = 0;
                    if (peekColbinHeader(g_dataset_path + autoStreamTable + ".colbin",
                                         nr, fsz)) {
                        streamProjectedBytes =
                            nr * projectedBytesPerRow(autoStreamTable,
                                                      tableCols.at(autoStreamTable));
                    }
                }
                // Working-set we'd allocate if we loaded the whole stream
                // resident — this is what actually competes with the GPU
                // budget, not the on-disk file footprint.
                uint64_t projectedWorkingSet = residentBytes + streamProjectedBytes;

                // --- Trigger: chunk only when projected working-set would
                //              dominate the budget.
                constexpr double kThreshold    = 0.75;
                constexpr double kBudgetFraction = 0.50; // headroom for hash maps, output, kernels
                const bool fitsInBudget =
                    (totalBudget > 0) &&
                    (projectedWorkingSet <= (uint64_t)(totalBudget * kThreshold));

                // --- Explicit-chunk downgrade ----------------------------
                // If the user passed --chunk N but the projected working
                // set fits comfortably, chunking would only add the
                // O(N) host-copy tax with no memory benefit. Downgrade
                // to direct load unless GPUDB_FORCE_CHUNK=1.
                if (g_chunkRows > 0 && fitsInBudget) {
                    const char* force = std::getenv("GPUDB_FORCE_CHUNK");
                    bool forceChunk = (force && force[0] && force[0] != '0');
                    if (!forceChunk) {
                        if (!g_csv) {
                            printf("[auto-chunk] %s: --chunk %zu downgraded to "
                                   "direct load (working-set=%.2f GiB fits in "
                                   "budget=%.2f GiB; set GPUDB_FORCE_CHUNK=1 "
                                   "to override)\n",
                                   plan.name.c_str(),
                                   g_chunkRows,
                                   projectedWorkingSet / 1e9,
                                   totalBudget * kThreshold / 1e9);
                        }
                        g_chunkRows = 0;
                    }
                }

                if (g_chunkRows == 0 && !fitsInBudget) {
                    uint64_t streamRows = 0, streamFsz = 0;
                    if (peekColbinHeader(g_dataset_path + autoStreamTable + ".colbin",
                                        streamRows, streamFsz) && streamRows > 0) {
                        const std::set<std::string>& streamCols = tableCols.at(autoStreamTable);
                        uint64_t streamBytesPerRow =
                            projectedBytesPerRow(autoStreamTable, streamCols);
                        if (streamBytesPerRow == 0) streamBytesPerRow = 1;
                        const int slots = g_chunkDoubleBuffer ? 2 : 1;

                        // Budget left for the streaming buffers after the
                        // resident tables claim their share. Reserve at
                        // least 1/8 of total budget for chunks even when
                        // residents are large (probe maps are typically
                        // much smaller than full-resident sizes).
                        int64_t streamBudget =
                            (int64_t)((double)totalBudget * kBudgetFraction)
                            - (int64_t)residentBytes;
                        int64_t minStreamBudget =
                            (int64_t)(totalBudget / 8);
                        if (streamBudget < minStreamBudget)
                            streamBudget = minStreamBudget;
                        if (streamBudget < (int64_t)(64ull << 20))  // 64 MiB floor
                            streamBudget = (int64_t)(64ull << 20);

                        size_t autoChunkRows = (size_t)
                            ((uint64_t)streamBudget /
                             (streamBytesPerRow * (uint64_t)slots));
                        // Clamp: floor at 256K rows (launch-overhead amortise),
                        // ceiling at total stream rows (no need to chunk).
                        autoChunkRows = std::max<size_t>(autoChunkRows, 256u * 1024);
                        if (autoChunkRows > streamRows)
                            autoChunkRows = (size_t)streamRows;
                        g_chunkRows = autoChunkRows;
                        if (!g_csv) {
                            printf("[auto-chunk] %s: disk=%.1f GiB working-set=%.1f GiB"
                                   " (resident=%.1f + stream=%.1f)"
                                   " budget=%.1f GiB (avail=%.1f phys=%.1f GPU=%.1f)"
                                   " stream=%s bytes/row=%llu slots=%d"
                                   " — chunk=%zu rows (%.0f MiB/slot)\n",
                                   plan.name.c_str(),
                                   totalDataBytes / 1e9,
                                   projectedWorkingSet / 1e9,
                                   residentBytes / 1e9,
                                   streamProjectedBytes / 1e9,
                                   totalBudget * kBudgetFraction / 1e9,
                                   availMemBytes / 1e9, physMemBytes / 1e9,
                                   gpuBudgetBytes / 1e9,
                                   autoStreamTable.c_str(),
                                   (unsigned long long)streamBytesPerRow,
                                   slots, g_chunkRows,
                                   (double)(g_chunkRows * streamBytesPerRow) / (1ull << 20));
                        }
                    }
                }
            }
        }


        auto parseStart = std::chrono::high_resolution_clock::now();
        codegen::MetalGenericExecutor executor(device, cmdQueue);

        // For chunked execution, auto-detect stream table (largest .colbin).
        const std::string streamTable = (g_chunkRows > 0)
            ? autoDetectStreamTable(tableCols) : std::string{};
        bool didChunk = false;

        // Load each table's columns. In chunked mode, skip the stream table
        // here — it is loaded per-chunk below.
        // Track pure I/O (file read/mmap into host buffers) separately
        // from CPU preprocessing (max-key scans, per-query prep).
        double ioMs = 0.0;
        double preprocessMs = 0.0;
        std::vector<std::pair<std::string, QueryColumns>> loadedTables;
        for (auto& [tableName, colNames] : tableCols) {
            if (!streamTable.empty() && tableName == streamTable) continue;
            const auto& tdef = schema.table(tableName);
            std::vector<ColSpec> specs;
            for (const auto& colName : colNames)
                specs.push_back(colSpecFor(tdef.col(colName)));
            auto _ioStart = clk::now();
            auto cols = loadQueryColumns(device, g_dataset_path + tableName + ".tbl", specs);
            ioMs += elapsedMs(_ioStart, clk::now());
            size_t rowCount = cols.rows();
            for (const auto& colName : colNames) {
                auto& cdef = tdef.col(colName);
                MTL::Buffer* buf = cols.buffer(cdef.index);
                if (!buf) continue;
                executor.registerTableBuffer(colName, buf, rowCount);
            }
            executor.registerTableRowCount(tableName, rowCount);
            loadedTables.emplace_back(tableName, std::move(cols));
        }

        {
            auto _ppStart = clk::now();
            registerMaxKeySymbols(executor, loadedTables, tableCols, schema);
            if (!streamTable.empty() && tableCols.count(streamTable)) {
                extendMaxKeysFromStreamColbin(
                    executor,
                    g_dataset_path + streamTable + ".tbl",
                    tableCols.at(streamTable),
                    schema,
                    streamTable);
            }
            if (!codegen::prepareQueryPreprocessing(plan.name, device, executor, loadedTables)) {
                return false;
            }
            preprocessMs += elapsedMs(_ppStart, clk::now());
        }

        codegen::MetalExecutionResult result;

        // ---------------------------------------------------------------
        // Chunked streaming path — GPU atomic accumulation across chunks.
        // All Metal kernels use atomic_fetch_add_explicit, so output buffers
        // accumulate additively when zero-init is suppressed after chunk 0.
        // ---------------------------------------------------------------
        if (!streamTable.empty()) {
            std::vector<ColSpec> streamSpecs;
            for (const auto& colName : tableCols.at(streamTable))
                streamSpecs.push_back(colSpecFor(schema.table(streamTable).col(colName)));
            const int streamSlots = g_chunkDoubleBuffer ? 2 : 1;
            codegen::ChunkedColbinTable stream;
            std::string streamError;
            if (!stream.open(device, g_dataset_path + streamTable + ".tbl",
                             streamSpecs, g_chunkRows, streamSlots, streamError)) {
                std::cerr << "Codegen: chunk open failed for " << streamTable
                          << ": " << streamError << std::endl;
                return false;
            }
            const auto& streamTdef = schema.table(streamTable);
            const size_t totalRows = stream.rows(), chunkRows = stream.chunkRows();
            size_t chunkCount = 0;
            double chunkCopyMs = 0.0, gpuMs = 0.0, bufAllocMs = 0.0;
            std::map<std::string, double> chunkPhaseSums;

            // Determine stream phase range once.
            // Pre-stream phases (scan non-stream tables before first stream phase)
            // run ONCE before the loop. Stream phases (scan streamTable) run per
            // chunk. Post-stream phases run ONCE after the loop.
            const auto& cgPhases = cg.getPhases();
            const int totalPhases = (int)cgPhases.size();
            int firstStreamPhase = totalPhases, lastStreamPhase = 0;
            for (int _pi = 0; _pi < totalPhases; _pi++) {
                if (cgPhases[_pi].scannedTable == streamTable) {
                    if (firstStreamPhase == totalPhases) firstStreamPhase = _pi;
                    lastStreamPhase = _pi + 1;
                }
            }
            // If no phase scans streamTable explicitly, treat all phases as stream.
            if (firstStreamPhase == totalPhases) { firstStreamPhase = 0; lastStreamPhase = totalPhases; }

            // Run pre-stream phases once (e.g. build bitmap from part/orders).
            if (firstStreamPhase > 0) {
                auto preResult = executor.execute(compiled, cg, 0, 1, 0, firstStreamPhase);
                gpuMs      += preResult.totalKernelTimeMs;
                bufAllocMs += preResult.bufferAllocTimeMs;
                for (size_t i = 0; i < preResult.phaseTimesMs.size(); ++i) {
                    const std::string nm = (i < preResult.phaseNames.size())
                        ? preResult.phaseNames[i] : ("pre_phase" + std::to_string(i));
                    chunkPhaseSums[nm] += preResult.phaseTimesMs[i];
                }
            }

            // Chunk loop: stream phases run per chunk with GPU atomic accumulation.
            for (size_t startRow = 0; startRow < totalRows; startRow += chunkRows) {
                const size_t rowsThisChunk = std::min(chunkRows, totalRows - startRow);
                const int slot = (int)(chunkCount % (size_t)streamSlots);
                auto chunkLoadStart = clk::now();
                if (!stream.loadChunk(slot, startRow, rowsThisChunk, streamError)) {
                    std::cerr << "Codegen: chunk load failed: " << streamError << std::endl;
                    return false;
                }
                chunkCopyMs += elapsedMs(chunkLoadStart, clk::now());
                for (const auto& colName : tableCols.at(streamTable)) {
                    const auto& cdef = streamTdef.col(colName);
                    MTL::Buffer* buf = stream.buffer(slot, cdef.index);
                    if (!buf) {
                        std::cerr << "Codegen: missing chunk buffer for "
                                  << streamTable << "." << colName << std::endl;
                        return false;
                    }
                    executor.registerTableBuffer(colName, buf, rowsThisChunk);
                }
                executor.registerTableRowCount(streamTable, rowsThisChunk);
                if (chunkCount == 1) executor.setSkipZeroInit(true);
                auto chunkResult = executor.execute(compiled, cg, 0, 1,
                                                    firstStreamPhase, lastStreamPhase);
                gpuMs      += chunkResult.totalKernelTimeMs;
                bufAllocMs += chunkResult.bufferAllocTimeMs;
                for (size_t i = 0; i < chunkResult.phaseTimesMs.size(); ++i) {
                    const std::string nm = (i < chunkResult.phaseNames.size())
                        ? chunkResult.phaseNames[i] : ("phase" + std::to_string(i));
                    chunkPhaseSums[nm] += chunkResult.phaseTimesMs[i];
                }
                chunkCount++;
            }
            executor.setSkipZeroInit(false);

            // Run post-stream phases once (e.g. Q4's orders count).
            if (lastStreamPhase < totalPhases) {
                auto postResult = executor.execute(compiled, cg, 0, 1,
                                                   lastStreamPhase, totalPhases);
                gpuMs += postResult.totalKernelTimeMs;
                for (size_t i = 0; i < postResult.phaseTimesMs.size(); ++i) {
                    const std::string nm = (i < postResult.phaseNames.size())
                        ? postResult.phaseNames[i] : ("post_phase" + std::to_string(i));
                    chunkPhaseSums[nm] += postResult.phaseTimesMs[i];
                }
            }

            result.result = executor.collectResult(cg);
            double wallMs = elapsedMs(parseStart, clk::now());
            timing.dataLoadMs    = std::max(0.0, wallMs - gpuMs);
            timing.ingestMs      = loadStats().excludedMs;
            timing.loadSource    = "chunked-colbin";
            timing.loadBytes     = loadStats().bytes + stream.bytesLoaded();
            timing.bufferAllocMs = bufAllocMs;
            // Chunked I/O: pre-stream load(ioMs) + per-chunk loadChunk(chunkCopyMs).
            timing.ioMs          = ioMs + chunkCopyMs;
            timing.preprocessMs  = preprocessMs;
            timing.gpuTotalMs    = gpuMs;
            timing.gpuTrialsN    = 1;
            timing.gpuMsP10      = gpuMs;
            timing.gpuMsP90      = gpuMs;
            timing.gpuMsMad      = 0.0;
            timing.phaseKernelMs.clear();
            for (const auto& [nm, ms] : chunkPhaseSums)
                timing.phaseKernelMs.emplace_back(nm, ms);
            if (!g_csv)
                printf("[chunk] %s: %zu chunks, stream=%s, chunk_rows=%zu, slots=%d, "
                       "GPU=%.3fms, copy=%.3fms\n",
                       queryName.c_str(), chunkCount, streamTable.c_str(),
                       chunkRows, streamSlots, gpuMs, chunkCopyMs);
            printf("STREAMING_CSV,%s,%s,%s,%zu,%zu,%d,%.3f,%.3f,%.3f,%zu\n",
                   timing.scaleFactor.c_str(), queryName.c_str(), streamTable.c_str(),
                   chunkRows, chunkCount, streamSlots,
                   timing.dataLoadMs, gpuMs, 0.0, timing.loadBytes);
            didChunk = true;
        }
        if (!didChunk) {
        // 6. Execute (with experiment harness: warmup + repeat + optional pipeline-cache bypass)
        auto parseEnd = std::chrono::high_resolution_clock::now();
        double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
        // One-time .tbl->column ingest (only when .colbin is missing) is
        // reported separately via timing.ingestMs and excluded from e2e.
        const double ingestMs = loadStats().excludedMs;
        timing.dataLoadMs = parseMs - ingestMs;
        if (timing.dataLoadMs < 0.0) timing.dataLoadMs = 0.0;
        timing.ingestMs   = ingestMs;
        timing.loadSource = loadStats().source();
        timing.loadBytes  = loadStats().bytes;
        // Split data-load window into pure I/O vs CPU preprocess. Anything
        // unaccounted for in the window is attributed to preprocess.
        timing.ioMs         = ioMs;
        timing.preprocessMs = (timing.dataLoadMs > ioMs)
                              ? (timing.dataLoadMs - ioMs) : preprocessMs;

        // ----- B1: --autotune-tg [--autotune-tg-per-phase] --------------
        // Per-query TG sweep over a fixed candidate set. For each
        // candidate, override every phase's threadgroupSize, run a quick
        // calibration (1 untimed + 5 timed iters, drop max), and record
        // the median GPU time -- both totals (for global autotune) and
        // per-phase timings (for per-phase autotune). Finally apply the
        // best candidate(s) and continue with the regular --warmup/--repeat
        // measurement loop. The PSO is shared across candidates: tg_size
        // is a kernel parameter ([[threads_per_threadgroup]]), so
        // changing dispatch TG does NOT require recompiling.
        if (g_autotuneTg || g_autotuneTgPerPhase) {
            const std::vector<int> candidates = {32, 64, 128, 256, 512, 1024};
            auto& phs = cg.getPhasesMutable();
            const size_t nPhases = phs.size();

            // perPhaseP50[c][i] = median (of 4) GPU time of phase i when
            // every phase is dispatched at candidates[c].
            std::vector<std::vector<double>> perPhaseP50(
                candidates.size(), std::vector<double>(nPhases, 0.0));
            std::vector<double> totalP50(candidates.size(), 0.0);

            for (size_t c = 0; c < candidates.size(); c++) {
                int candTg = candidates[c];
                for (auto& p : phs) p.threadgroupSize = candTg;
                // 1 untimed warmup
                (void) executor.execute(compiled, cg, 0, 1);
                // 5 timed trials; drop slowest as outlier.
                std::vector<double> totals; totals.reserve(5);
                std::vector<std::vector<double>> phaseSamples(nPhases);
                for (auto& v : phaseSamples) v.reserve(5);
                for (int t = 0; t < 5; t++) {
                    auto rr = executor.execute(compiled, cg, 0, 1);
                    totals.push_back((double)rr.totalKernelTimeMs);
                    for (size_t i = 0; i < nPhases && i < rr.phaseTimesMs.size(); i++) {
                        phaseSamples[i].push_back((double)rr.phaseTimesMs[i]);
                    }
                }
                auto p50DropMax = [](std::vector<double> v) -> double {
                    if (v.empty()) return 0.0;
                    std::sort(v.begin(), v.end());
                    if (v.size() > 1) v.pop_back();
                    return v[v.size() / 2];
                };
                totalP50[c] = p50DropMax(totals);
                for (size_t i = 0; i < nPhases; i++) {
                    perPhaseP50[c][i] = p50DropMax(phaseSamples[i]);
                }
                if (g_csv) {
                    std::sort(totals.begin(), totals.end());
                    printf("AUTOTUNE_CSV,%s,%s,%d,%.3f,%.3f,%.3f\n",
                           timing.scaleFactor.c_str(),
                           timing.queryName.c_str(),
                           candTg, totals.front(), totalP50[c], totals.back());
                    for (size_t i = 0; i < nPhases; i++) {
                        printf("AUTOTUNE_PHASE_CSV,%s,%s,%s,%d,%.3f\n",
                               timing.scaleFactor.c_str(),
                               timing.queryName.c_str(),
                               phs[i].name.c_str(), candTg, perPhaseP50[c][i]);
                    }
                }
            }

            if (g_autotuneTgPerPhase) {
                // Pick best TG per phase by minimum p50 phase time.
                std::vector<int> chosen(nPhases, candidates.back());
                double sumChosenMs = 0.0;
                for (size_t i = 0; i < nPhases; i++) {
                    double bestMs = std::numeric_limits<double>::infinity();
                    int bestC = candidates.back();
                    for (size_t c = 0; c < candidates.size(); c++) {
                        if (perPhaseP50[c][i] < bestMs) {
                            bestMs = perPhaseP50[c][i];
                            bestC = candidates[c];
                        }
                    }
                    chosen[i] = bestC;
                    sumChosenMs += bestMs;
                    phs[i].threadgroupSize = bestC;
                }
                if (!g_csv) {
                    printf("[autotune-tg-per-phase] picks:");
                    for (size_t i = 0; i < nPhases; i++) {
                        printf(" %s=%d", phs[i].name.c_str(), chosen[i]);
                    }
                    printf("  (sum p50 = %.3f ms)\n", sumChosenMs);
                }
            } else {
                // Global: pick TG that minimises total GPU time.
                int bestTg = candidates.back();
                double bestP50 = std::numeric_limits<double>::infinity();
                for (size_t c = 0; c < candidates.size(); c++) {
                    if (totalP50[c] < bestP50) {
                        bestP50 = totalP50[c];
                        bestTg = candidates[c];
                    }
                }
                for (auto& p : phs) p.threadgroupSize = bestTg;
                if (!g_csv) {
                    printf("[autotune-tg] best TG = %d (p50 GPU = %.3f ms across %zu candidates)\n",
                           bestTg, bestP50, candidates.size());
                }
            }
        }

        // External warmup loop (untimed). Replaces the executor's internal
        // warmup so we control the iteration count via --warmup N.
        for (int w = 0; w < g_warmup; w++) {
            (void) executor.execute(compiled, cg, 0, 1);
        }

        // Measured loop. Each trial captures GPU time + (optionally) JIT compile time.
        std::vector<double> gpuTrials;     gpuTrials.reserve(g_repeat);
        std::vector<double> compileTrials; compileTrials.reserve(g_repeat);
        std::vector<double> e2eTrials;     e2eTrials.reserve(g_repeat);

        for (int r = 0; r < g_repeat; r++) {
            // --no-pipeline-cache: rebuild library + PSOs every measured trial
            // to expose the JIT cost amortization curve. The compiler must
            // outlive execute() because ~RuntimeCompiler releases the PSOs.
            codegen::RuntimeCompiler::CompiledQuery compiledTrial = compiled;
            std::unique_ptr<codegen::RuntimeCompiler> compilerR;
            double trialCompileMs = 0.0;
            if (g_noPipelineCache) {
                auto tcr0 = clk::now();
                compilerR = std::make_unique<codegen::RuntimeCompiler>(device);
                auto* libR = compilerR->compile(metalSource);
                if (!libR) {
                    std::cerr << "Codegen: Metal recompile failed in --no-pipeline-cache trial\n";
                    return false;
                }
                codegen::RuntimeCompiler::CompiledQuery cR;
                cR.library = libR;
                for (const auto& phase : cg.getPhases()) {
                    auto* pso = compilerR->getPipeline(libR, phase.name);
                    if (!pso) {
                        std::cerr << "Codegen: PSO recreation failed for " << phase.name << "\n";
                        return false;
                    }
                    cR.pipelines.push_back(pso);
                    cR.kernelNames.push_back(phase.name);
                }
                compiledTrial = cR;
                trialCompileMs = elapsedMs(tcr0, clk::now());
            }

            auto tr0 = clk::now();
            result = executor.execute(compiledTrial, cg, 0, 1);
            double e2eTrialMs = elapsedMs(tr0, clk::now());

            gpuTrials.push_back((double)result.totalKernelTimeMs);
            compileTrials.push_back(trialCompileMs);
            e2eTrials.push_back(e2eTrialMs);

            if (g_csv && g_repeat > 1) {
                printf("TRIAL_CSV,%s,%s,%d,%.3f,%.3f,%.3f\n",
                       timing.queryName.c_str(),
                       timing.scaleFactor.c_str(),
                       r,
                       (double)result.totalKernelTimeMs,
                       trialCompileMs,
                       e2eTrialMs);
                // C2: per-phase GPU breakdown for this trial. Emitting one
                // row per phase per trial lets analysis spot per-kernel
                // variance and bottlenecks without parsing the text report.
                for (size_t pi = 0; pi < result.phaseTimesMs.size(); pi++) {
                    const std::string& nm = (pi < result.phaseNames.size())
                        ? result.phaseNames[pi] : "phase";
                    int tgUsed = (pi < cg.getPhases().size())
                        ? cg.getPhases()[pi].threadgroupSize : 0;
                    printf("PHASE_CSV,%s,%s,%d,%s,%d,%.3f\n",
                           timing.queryName.c_str(),
                           timing.scaleFactor.c_str(),
                           r, nm.c_str(), tgUsed,
                           (double)result.phaseTimesMs[pi]);
                }
            }
        }

        // Median across measured trials (lower-median for even N).
        auto median = [](std::vector<double> v) -> double {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };

        // C1: percentile + MAD on the GPU-time trial distribution.
        auto pct = [](std::vector<double> v, double p) -> double {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            // Nearest-rank percentile: ceil(p * N) - 1, clamped.
            double idx = p * (double)v.size();
            size_t i = (size_t)std::min<double>(std::max<double>(idx - 1, 0),
                                                (double)(v.size() - 1));
            return v[i];
        };
        auto mad = [&median](std::vector<double> v) -> double {
            if (v.size() < 2) return 0.0;
            double m = median(v);
            std::vector<double> dev;
            dev.reserve(v.size());
            for (double x : v) dev.push_back(std::fabs(x - m));
            return median(std::move(dev));
        };

        result.parseTimeMs = static_cast<float>(parseMs);
        timing.bufferAllocMs = result.bufferAllocTimeMs;
        timing.gpuTotalMs    = median(gpuTrials);
        timing.gpuTrialsN    = (int)gpuTrials.size();
        timing.gpuMsP10      = pct(gpuTrials, 0.10);
        timing.gpuMsP90      = pct(gpuTrials, 0.90);
        timing.gpuMsMad      = mad(gpuTrials);
        if (g_noPipelineCache) {
            // Override the single-shot compile time with the per-trial median
            // so the headline number reflects the cost we're studying.
            timing.compileMs = median(compileTrials);
        }
        timing.phaseKernelMs.clear();
        for (size_t i = 0; i < result.phaseTimesMs.size(); i++) {
            const std::string name = (i < result.phaseNames.size())
                ? result.phaseNames[i] : ("phase" + std::to_string(i));
            timing.phaseKernelMs.emplace_back(name, (double)result.phaseTimesMs[i]);
        }
        } // end if (!didChunk)

        auto postStart = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------------
        // Per-query pre-print normalisation
        // ---------------------------------------------------------------

        // Q14: GPU accumulates raw promo/total sums for precision; convert
        // to the final ratio (100 * promo / total) so the result matches
        // the standard TPC-H single-column output and DuckDB golden format.
        if (plan.name == "Q14" && result.result.numRows() == 1 &&
            result.result.columns.size() == 2) {
            double promo = std::get<double>(result.result.rows[0][0]);
            double total = std::get<double>(result.result.rows[0][1]);
            double ratio = (total > 0) ? (100.0 * promo / total) : 0.0;
            result.result.columns = {{"promo_revenue", "float"}};
            result.result.rows[0] = {ratio};
        }

        // Q12: GPU emits 4 buckets [(MAIL,high), (MAIL,low), (SHIP,high),
        // (SHIP,low)]; pivot to two output rows with columns matching
        // golden CSV (l_shipmode, high_line_count, low_line_count).
        if (plan.name == "Q12" && result.result.numRows() == 4 &&
            result.result.columns.size() == 2) {
            auto getCount = [&](size_t r) -> int64_t {
                const auto& v = result.result.rows[r][1];
                if (std::holds_alternative<int64_t>(v)) return std::get<int64_t>(v);
                if (std::holds_alternative<double>(v))  return (int64_t)std::get<double>(v);
                return 0;
            };
            int64_t mailHigh = getCount(0), mailLow = getCount(1);
            int64_t shipHigh = getCount(2), shipLow = getCount(3);
            result.result.columns = {
                {"l_shipmode", "string"},
                {"high_line_count", "int"},
                {"low_line_count",  "int"}
            };
            result.result.rows.clear();
            result.result.rows.push_back({std::string("MAIL"), mailHigh, mailLow});
            result.result.rows.push_back({std::string("SHIP"), shipHigh, shipLow});
        }

        // 7. Print generic results.
        // Queries whose final output is assembled by CPU post-processing below
        // leave result.result.columns empty here; those queries do their own printf.
        if (!g_csv && !result.result.columns.empty()) {
            printf("\n%s Results:\n", queryName.c_str());
            result.result.print();
        }

        // ---------------------------------------------------------------
        // Per-query post-processing — populates result.result then prints.
        // The correctness oracle (golden check) runs AFTER all blocks below.
        // ---------------------------------------------------------------

        // Q10: top-20 from cust_revenue array
        if (plan.name == "Q10") {
            auto* revBuf = executor.getAllocatedBuffer("d_cust_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                size_t numCust = revBuf->length() / sizeof(float);
                std::vector<std::pair<float, int>> entries;
                for (size_t ck = 0; ck < numCust; ck++) {
                    if (rev[ck] > 0.0f) entries.push_back({rev[ck], (int)ck});
                }
                std::sort(entries.begin(), entries.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                int show = std::min((int)entries.size(), 20);
                result.result.columns = {{"c_custkey","int"},{"revenue","float"}};
                result.result.rows.clear();
                for (int j = 0; j < show; j++)
                    result.result.rows.push_back({(int64_t)entries[j].second, (double)entries[j].first});
                printf("  Top-%d customers by returned-item revenue:\n", show);
                printf("  +----------+--------------+\n");
                printf("  | c_custkey|      revenue |\n");
                printf("  +----------+--------------+\n");
                for (int j = 0; j < show; j++) {
                    printf("  | %8d | %12.2f |\n", entries[j].second, entries[j].first);
                }
                printf("  +----------+--------------+\n");
            }
        }

        // Q7: print 4 revenue bins
        if (plan.name == "Q7") {
            auto* binsBuf = executor.getAllocatedBuffer("d_revenue_bins");
            if (binsBuf) {
                float* bins = (float*)binsBuf->contents();
                const char* pair_supp[] = {"FRANCE", "GERMANY"};
                const char* pair_cust[] = {"GERMANY", "FRANCE"};
                result.result.columns = {{"supp_nation","string"},{"cust_nation","string"},{"l_year","int"},{"revenue","float"}};
                result.result.rows.clear();
                for (int p = 0; p < 2; p++)
                    for (int y = 0; y < 2; y++)
                        result.result.rows.push_back({std::string(pair_supp[p]), std::string(pair_cust[p]), (int64_t)(1995+y), (double)bins[p*2+y]});
                printf("  +----------+----------+--------+-----------------+\n");
                printf("  | supp_nat | cust_nat | l_year |         revenue |\n");
                printf("  +----------+----------+--------+-----------------+\n");
                for (int p = 0; p < 2; p++) {
                    for (int y = 0; y < 2; y++) {
                        printf("  | %-8s | %-8s | %6d | $%14.2f |\n",
                               pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
                    }
                }
                printf("  +----------+----------+--------+-----------------+\n");
            }
        }

        // Q5: nation revenue sorted desc
        if (plan.name == "Q5") {
            auto* revBuf = executor.getAllocatedBuffer("d_nation_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                auto nat = loadNation(g_dataset_path);
                auto nationNames = buildNationNames(nat.nationkey, nat.name.data(),
                                                     NationData::NAME_WIDTH);
                std::vector<std::pair<float, int>> entries;
                for (int nk = 0; nk < 25; nk++) {
                    if (rev[nk] > 0.0f) entries.push_back({rev[nk], nk});
                }
                std::sort(entries.begin(), entries.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                result.result.columns = {{"n_name","string"},{"revenue","float"}};
                result.result.rows.clear();
                for (auto& [r, nk] : entries)
                    result.result.rows.push_back({nationNames[nk], (double)r});
                printf("  +------------------+-----------------+\n");
                printf("  | n_name           |         revenue |\n");
                printf("  +------------------+-----------------+\n");
                for (auto& [r, nk] : entries) {
                    printf("  | %-16s | $%14.2f |\n", nationNames[nk].c_str(), r);
                }
                printf("  +------------------+-----------------+\n");
            }
        }

        // Q8: market share = brazil / total per year
        if (plan.name == "Q8") {
            auto* binsBuf = executor.getAllocatedBuffer("d_result_bins");
            if (binsBuf) {
                float* bins = (float*)binsBuf->contents();
                result.result.columns = {{"o_year","int"},{"mkt_share","float"}};
                result.result.rows.clear();
                for (int y = 0; y < 2; y++) {
                    float brazil = bins[y], total = bins[2+y];
                    float share = (total > 0.0f) ? (brazil / total) : 0.0f;
                    result.result.rows.push_back({(int64_t)(1995+y), (double)share});
                }
                printf("  +--------+------------+\n");
                printf("  | o_year |  mkt_share |\n");
                printf("  +--------+------------+\n");
                for (int y = 0; y < 2; y++) {
                    float brazil = bins[y];
                    float total = bins[2 + y];
                    float share = (total > 0.0f) ? (brazil / total) : 0.0f;
                    printf("  | %6d | %10.4f |\n", 1995 + y, share);
                }
                printf("  +--------+------------+\n");
            }
        }

        // Q3: top-10 by revenue desc with date/prio
        if (plan.name == "Q3") {
            auto* revBuf = executor.getAllocatedBuffer("d_order_revenue");
            auto* dateBuf = executor.getAllocatedBuffer("d_orders_date_map");
            auto* prioBuf = executor.getAllocatedBuffer("d_orders_prio_map");
            if (revBuf && dateBuf && prioBuf) {
                float* rev = (float*)revBuf->contents();
                int* dates = (int*)dateBuf->contents();
                int* prios = (int*)prioBuf->contents();
                size_t n = revBuf->length() / sizeof(float);
                std::vector<std::tuple<float, int, int, int>> entries;
                for (size_t ok = 0; ok < n; ok++) {
                    if (rev[ok] > 0.0f) {
                        entries.push_back({rev[ok], dates[ok], (int)ok, prios[ok]});
                    }
                }
                std::sort(entries.begin(), entries.end(),
                    [](auto& a, auto& b) {
                        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
                        return std::get<1>(a) < std::get<1>(b);
                    });
                int show = std::min((int)entries.size(), 10);
                result.result.columns = {{"l_orderkey","int"},{"revenue","float"},{"o_orderdate","string"},{"o_shippriority","int"}};
                result.result.rows.clear();
                for (int j = 0; j < show; j++) {
                    auto& [r, d, ok, p] = entries[j];
                    result.result.rows.push_back({(int64_t)ok, (double)r, intDateToStr(d), (int64_t)p});
                }
                printf("  +----------+--------------+------------+---------------+\n");
                printf("  |l_orderkey|      revenue | o_orderdate|o_shippriority |\n");
                printf("  +----------+--------------+------------+---------------+\n");
                for (int j = 0; j < show; j++) {
                    auto& [r, d, ok, p] = entries[j];
                    printf("  | %8d | %12.2f | %10d | %13d |\n", ok, r, d, p);
                }
                printf("  +----------+--------------+------------+---------------+\n");
            }
        }

        // Q13: histogram of order counts
        if (plan.name == "Q13") {
            auto* histBuf = executor.getAllocatedBuffer("d_histogram");
            if (histBuf) {
                uint32_t* hist = (uint32_t*)histBuf->contents();
                size_t maxBin = histBuf->length() / sizeof(uint32_t);
                std::vector<std::pair<uint32_t, int>> entries;
                for (size_t c = 0; c < maxBin; c++) {
                    if (hist[c] > 0) entries.push_back({hist[c], (int)c});
                }
                std::sort(entries.begin(), entries.end(),
                    [](auto& a, auto& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second > b.second;
                    });
                result.result.columns = {{"c_count","int"},{"custdist","int"}};
                result.result.rows.clear();
                for (auto& [dist, cnt] : entries)
                    result.result.rows.push_back({(int64_t)cnt, (int64_t)dist});
                printf("  +--------+----------+\n");
                printf("  | c_count|  custdist|\n");
                printf("  +--------+----------+\n");
                for (auto& [dist, cnt] : entries) {
                    printf("  | %6d | %8u |\n", cnt, dist);
                }
                printf("  +--------+----------+\n");
            }
        }

        // Q22: 7 country-code bins
        if (plan.name == "Q22") {
            auto* cntBuf = executor.getAllocatedBuffer("d_q22_count");
            auto* sumBuf = executor.getAllocatedBuffer("d_q22_sum");
            if (cntBuf && sumBuf) {
                uint32_t* counts = (uint32_t*)cntBuf->contents();
                float* sums = (float*)sumBuf->contents();
                const int valid_prefixes[] = {13, 17, 18, 23, 29, 30, 31};
                result.result.columns = {{"cntrycode","int"},{"numcust","int"},{"totacctbal","float"}};
                result.result.rows.clear();
                for (int b = 0; b < 7; b++) {
                    if (counts[b] > 0)
                        result.result.rows.push_back({(int64_t)valid_prefixes[b], (int64_t)counts[b], (double)sums[b]});
                }
                printf("  +----------+----------+---------------+\n");
                printf("  | cntrycode|  numcust |    totacctbal |\n");
                printf("  +----------+----------+---------------+\n");
                for (int b = 0; b < 7; b++) {
                    if (counts[b] > 0) {
                        printf("  | %8d | %8u | %13.2f |\n",
                               valid_prefixes[b], counts[b], sums[b]);
                    }
                }
                printf("  +----------+----------+---------------+\n");
            }
        }

        // Q11: total sum → threshold → filter → sort
        if (plan.name == "Q11") {
            auto* valBuf = executor.getAllocatedBuffer("d_part_value");
            if (valBuf) {
                float* values = (float*)valBuf->contents();
                size_t n = valBuf->length() / sizeof(float);
                double globalSum = 0.0;
                for (size_t k = 0; k < n; k++) globalSum += values[k];
                double threshold = globalSum * 0.0001;
                struct Q11Entry { int partkey; double value; };
                std::vector<Q11Entry> results;
                for (size_t k = 0; k < n; k++) {
                    if (values[k] > threshold) results.push_back({(int)k, (double)values[k]});
                }
                std::sort(results.begin(), results.end(),
                    [](auto& a, auto& b) { return a.value > b.value; });
                result.result.columns = {{"ps_partkey","int"},{"value","float"}};
                result.result.rows.clear();
                for (auto& e : results)
                    result.result.rows.push_back({(int64_t)e.partkey, e.value});
                int show = std::min((int)results.size(), 20);
                printf("  Top-%d of %zu qualifying parts (threshold %.2f):\n",
                       show, results.size(), threshold);
                printf("  +-----------+------------------+\n");
                printf("  | ps_partkey|            value |\n");
                printf("  +-----------+------------------+\n");
                for (int j = 0; j < show; j++) {
                    printf("  | %9d | %16.2f |\n", results[j].partkey, results[j].value);
                }
                printf("  +-----------+------------------+\n");
            }
        }

        // Q15: find max revenue supplier
        if (plan.name == "Q15") {
            auto* revBuf = executor.getAllocatedBuffer("d_supp_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                size_t n = revBuf->length() / sizeof(float);
                float maxRev = 0.0f;
                for (size_t k = 0; k < n; k++) maxRev = std::max(maxRev, rev[k]);
                result.result.columns = {{"s_suppkey","int"},{"total_revenue","float"}};
                result.result.rows.clear();
                for (size_t k = 0; k < n; k++) {
                    if (rev[k] >= maxRev - 0.01f)
                        result.result.rows.push_back({(int64_t)k, (double)rev[k]});
                }
                printf("  Top supplier(s) with max revenue %.2f:\n", maxRev);
                printf("  +----------+------------------+\n");
                printf("  |  s_suppkey|    total_revenue |\n");
                printf("  +----------+------------------+\n");
                for (size_t k = 0; k < n; k++) {
                    if (rev[k] >= maxRev - 0.01f) {
                        printf("  | %9zu | %16.2f |\n", k, rev[k]);
                    }
                }
                printf("  +----------+------------------+\n");
            }
        }

        // Q18: filter qty > 300, join with preloaded orders, sort, top 100
        if (plan.name == "Q18") {
            auto* qtyBuf = executor.getAllocatedBuffer("d_order_qty");
            if (qtyBuf) {
                float* qtys = (float*)qtyBuf->contents();
                size_t n = qtyBuf->length() / sizeof(float);
                auto& o_custkey = g_q18Post.o_custkey;
                auto& o_totalprice = g_q18Post.o_totalprice;
                auto& o_orderdate = g_q18Post.o_orderdate;
                auto& okLookup = g_q18Post.okLookup;

                struct Q18Entry { int orderkey; int custkey; float totalprice; int orderdate; float qty; };
                std::vector<Q18Entry> results;
                for (size_t ok = 0; ok < n; ok++) {
                    if (qtys[ok] > 300.0f) {
                        results.push_back({(int)ok, 0, 0.0f, 0, qtys[ok]});
                    }
                }
                for (auto& r : results) {
                    if (r.orderkey >= 0 && (size_t)r.orderkey < okLookup.size()) {
                        int idx = okLookup[r.orderkey];
                        if (idx >= 0) {
                            r.custkey = o_custkey[idx];
                            r.totalprice = o_totalprice[idx];
                            r.orderdate = o_orderdate[idx];
                        }
                    }
                }
                std::sort(results.begin(), results.end(),
                    [](auto& a, auto& b) {
                        if (a.totalprice != b.totalprice) return a.totalprice > b.totalprice;
                        return a.orderdate < b.orderdate;
                    });
                int show = std::min((int)results.size(), 100);
                result.result.columns = {{"c_custkey","int"},{"o_orderkey","int"},{"o_orderdate","string"},{"o_totalprice","float"},{"sum(l_quantity)","float"}};
                result.result.rows.clear();
                for (int j = 0; j < show; j++) {
                    auto& r = results[j];
                    result.result.rows.push_back({(int64_t)r.custkey, (int64_t)r.orderkey, intDateToStr(r.orderdate), (double)r.totalprice, (double)r.qty});
                }
                printf("  Top-%d large volume orders (qty > 300):\n", show);
                printf("  +----------+----------+---------------+------------+----------+\n");
                printf("  | c_custkey| o_orderkey| o_totalprice |  o_orderdate| o_qty   |\n");
                printf("  +----------+----------+---------------+------------+----------+\n");
                for (int j = 0; j < show; j++) {
                    auto& r = results[j];
                    printf("  | %8d | %8d | %13.2f | %10d | %8.2f |\n",
                           r.custkey, r.orderkey, r.totalprice, r.orderdate, r.qty);
                }
                printf("  +----------+----------+---------------+------------+----------+\n");
            }
        }

        // Q17: read revenue and divide by 7.0
        if (plan.name == "Q17") {
            auto* revLoBuf = executor.getAllocatedBuffer("d_q17_revenue_lo");
            if (revLoBuf) {
                uint32_t raw = *(const uint32_t*)revLoBuf->contents();
                float fval;
                memcpy(&fval, &raw, sizeof(float));
                double revenue = (double)fval;
                double avgYearly = revenue / 7.0;
                result.result.columns = {{"avg_yearly","float"}};
                result.result.rows = {{avgYearly}};
                printf("  +------------------+\n");
                printf("  |      avg_yearly  |\n");
                printf("  +------------------+\n");
                printf("  | %16.2f |\n", avgYearly);
                printf("  +------------------+\n");
            }
        }

        // Q9: read profit bins, sort by nation ASC, year DESC
        if (plan.name == "Q9") {
            auto* profitBuf = executor.getAllocatedBuffer("d_q9_profit");
            if (profitBuf) {
                const float* profits = (const float*)profitBuf->contents();
                auto nCols = codegen::loadPreprocessColumns(device, "nation",
                    {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
                auto n_nk = codegen::copyIntColumn(nCols, 0);
                auto n_nm = codegen::copyCharColumn(nCols, 1, nCols.rows() * 25);
                std::vector<std::string> nationNames(25);
                for (size_t i = 0; i < n_nk.size(); i++) {
                    // Nation names are fixed-width 25-char fields; trim trailing spaces/nulls
                    const char* base = n_nm.data() + i * 25;
                    int len = 25;
                    while (len > 0 && (base[len-1] == ' ' || base[len-1] == '\0')) len--;
                    nationNames[n_nk[i]] = std::string(base, len);
                }

                struct Q9Row { std::string nation; int year; float profit; };
                std::vector<Q9Row> rows;
                for (int nk = 0; nk < 25; nk++) {
                    for (int yr = 1992; yr <= 1998; yr++) {
                        int bin = nk * 8 + (yr - 1992);
                        float p = profits[bin];
                        if (p != 0.0f) {
                            rows.push_back({nationNames[nk], yr, p});
                        }
                    }
                }
                std::sort(rows.begin(), rows.end(), [](const Q9Row& a, const Q9Row& b) {
                    if (a.nation != b.nation) return a.nation < b.nation;
                    return a.year > b.year;
                });
                result.result.columns = {{"nation","string"},{"o_year","int"},{"sum_profit","float"}};
                result.result.rows.clear();
                for (auto& r : rows)
                    result.result.rows.push_back({r.nation, (int64_t)r.year, (double)r.profit});
                printf("  +------------+------+---------------+\n");
                printf("  | Nation     | Year |        Profit |\n");
                printf("  +------------+------+---------------+\n");
                int show = std::min((int)rows.size(), 15);
                for (int j = 0; j < show; j++) {
                    printf("  | %-10s | %4d | $%13.2f |\n",
                           rows[j].nation.c_str(), rows[j].year, rows[j].profit);
                }
                printf("  +------------+------+---------------+\n");
                printf("  Total results: %d\n", (int)rows.size());
            }
        }

        // Q20: check HT values against availqty, filter CANADA suppliers
        if (plan.name == "Q20") {
            auto& pd = g_q20Post;
            if (pd.htValsBuf) {
                const float* htVals = (const float*)pd.htValsBuf->contents();
                std::set<int> qualSuppkeys;
                for (uint32_t slot = 0; slot < pd.htSlots; slot++) {
                    if (pd.htKeys[slot] == ~uint64_t(0)) continue;
                    int psIdx = pd.htPsIdx[slot];
                    if (psIdx < 0) continue;
                    float sumQty = htVals[slot];
                    if (sumQty > 0.0f && (float)pd.ps_availqty[psIdx] > 0.5f * sumQty) {
                        qualSuppkeys.insert(pd.ps_suppkey[psIdx]);
                    }
                }

                struct Q20Row { std::string name; std::string address; };
                std::vector<Q20Row> rows;
                auto extractFixedStr = [](const std::vector<char>& data, size_t idx, int width) {
                    const char* base = data.data() + idx * width;
                    int len = 0;
                    while (len < width && base[len] != '\0') len++;
                    return std::string(base, len);
                };
                for (size_t i = 0; i < pd.s_suppkey.size(); i++) {
                    if (pd.s_nationkey[i] != pd.canada_nk) continue;
                    if (!qualSuppkeys.count(pd.s_suppkey[i])) continue;
                    rows.push_back({extractFixedStr(pd.s_name, i, 25), extractFixedStr(pd.s_address, i, 40)});
                }
                std::sort(rows.begin(), rows.end(), [](const Q20Row& a, const Q20Row& b) {
                    return a.name < b.name;
                });
                result.result.columns = {{"s_name","string"},{"s_address","string"}};
                result.result.rows.clear();
                for (auto& r : rows)
                    result.result.rows.push_back({r.name, r.address});
                printf("  +---------------------------+------------------------------------------+\n");
                printf("  | s_name                    | s_address                                |\n");
                printf("  +---------------------------+------------------------------------------+\n");
                int show = std::min((int)rows.size(), 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-25s | %-40s |\n", rows[j].name.c_str(), rows[j].address.c_str());
                }
                printf("  +---------------------------+------------------------------------------+\n");
                printf("  Total qualifying suppliers: %d\n", (int)rows.size());
            }
        }

        // Q2: read min_cost, match suppliers, join strings, sort, top 100
        if (plan.name == "Q2") {
            auto& pd = g_q2Post;
            auto* partBmpBuf = executor.getAllocatedBuffer("d_q2_part_bitmap");
            if (pd.minCostBuf && partBmpBuf) {
                const uint32_t* minCostU = (const uint32_t*)pd.minCostBuf->contents();
                const uint32_t* partBitmap = (const uint32_t*)partBmpBuf->contents();

                std::unordered_map<int, int> suppIdx;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++)
                    suppIdx[pd.s_suppkey[i]] = (int)i;

                std::unordered_map<int, int> partIdx;
                for (size_t i = 0; i < pd.p_partkey.size(); i++)
                    partIdx[pd.p_partkey[i]] = (int)i;

                struct Q2Row {
                    float s_acctbal;
                    std::string s_name, n_name, s_address, s_phone, s_comment, p_mfgr;
                    int p_partkey;
                };
                std::vector<Q2Row> rows;
                for (size_t i = 0; i < pd.ps_partkey.size(); i++) {
                    int pk = pd.ps_partkey[i];
                    int sk = pd.ps_suppkey[i];

                    if (pk < 0 || pk > pd.maxPartkey) continue;
                    if (!((partBitmap[pk / 32] >> (pk % 32)) & 1)) continue;

                    if (sk < 0 || (size_t)(sk / 32) >= pd.eurSuppBitmap.size()) continue;
                    if (!((pd.eurSuppBitmap[sk / 32] >> (sk % 32)) & 1)) continue;

                    uint32_t minU = minCostU[pk];
                    if (minU == 0xFFFFFFFFu) continue;
                    float minCost;
                    memcpy(&minCost, &minU, sizeof(float));
                    if (pd.ps_supplycost[i] != minCost) continue;

                    auto sit = suppIdx.find(sk);
                    auto pit = partIdx.find(pk);
                    if (sit == suppIdx.end() || pit == partIdx.end()) continue;
                    int si = sit->second;
                    int pi = pit->second;

                    Q2Row row;
                    row.s_acctbal = pd.s_acctbal[si];
                    row.p_partkey = pk;

                    auto extractStr = [](const std::vector<char>& data, int idx, int width) {
                        const char* base = data.data() + idx * width;
                        int len = 0;
                        while (len < width && base[len] != '\0') len++;
                        return std::string(base, len);
                    };

                    row.s_name = extractStr(pd.s_name, si, 25);
                    row.s_address = extractStr(pd.s_address, si, 40);
                    row.s_phone = extractStr(pd.s_phone, si, 15);
                    row.s_comment = extractStr(pd.s_comment, si, 101);
                    row.p_mfgr = extractStr(pd.p_mfgr, pi, 25);
                    row.n_name = (pd.s_nationkey[si] >= 0 && pd.s_nationkey[si] < (int)pd.nationNames.size())
                        ? pd.nationNames[pd.s_nationkey[si]] : "?";

                    rows.push_back(std::move(row));
                }

                std::sort(rows.begin(), rows.end(), [](const Q2Row& a, const Q2Row& b) {
                    if (a.s_acctbal != b.s_acctbal) return a.s_acctbal > b.s_acctbal;
                    if (a.n_name != b.n_name) return a.n_name < b.n_name;
                    if (a.s_name != b.s_name) return a.s_name < b.s_name;
                    return a.p_partkey < b.p_partkey;
                });

                int limit = std::min((int)rows.size(), 100);
                result.result.columns = {{"s_acctbal","float"},{"s_name","string"},{"n_name","string"},{"p_partkey","int"},{"p_mfgr","string"},{"s_address","string"},{"s_phone","string"},{"s_comment","string"}};
                result.result.rows.clear();
                for (int j = 0; j < limit; j++)
                    result.result.rows.push_back({(double)rows[j].s_acctbal, rows[j].s_name, rows[j].n_name, (int64_t)rows[j].p_partkey, rows[j].p_mfgr, rows[j].s_address, rows[j].s_phone, rows[j].s_comment});
                printf("\nQ2 Results:\n");
                printf("  %-10s | %-25s | %-15s | %-8s | %-25s\n",
                       "s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr");
                printf("  ----------+---------------------------+-----------------+----------+---------------------------\n");
                int show = std::min(limit, 10);
                for (int j = 0; j < show; j++) {
                    printf("  %9.2f | %-25s | %-15s | %8d | %-25s\n",
                           rows[j].s_acctbal, rows[j].s_name.c_str(), rows[j].n_name.c_str(),
                           rows[j].p_partkey, rows[j].p_mfgr.c_str());
                }
                if (limit > 10) printf("  ... (%d more rows)\n", limit - 10);
                printf("  Total rows: %d\n", limit);
            }
        }

        // Q16: popcount per-group bitmaps, sort, print
        if (plan.name == "Q16") {
            auto& pd = g_q16Post;
            if (pd.groupBitmapsBuf) {
                const uint32_t* gbm = (const uint32_t*)pd.groupBitmapsBuf->contents();

                struct Q16Result { std::string brand; std::string type; int size; int supplier_cnt; };
                std::vector<Q16Result> results;
                for (uint32_t g = 0; g < pd.numGroups; g++) {
                    int cnt = 0;
                    for (uint32_t w = 0; w < pd.bvInts; w++) {
                        cnt += __builtin_popcount(gbm[g * pd.bvInts + w]);
                    }
                    if (cnt > 0) {
                        results.push_back({pd.groups[g].brand, pd.groups[g].type, pd.groups[g].size, cnt});
                    }
                }

                std::sort(results.begin(), results.end(), [](const Q16Result& a, const Q16Result& b) {
                    if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
                    if (a.brand != b.brand) return a.brand < b.brand;
                    if (a.type != b.type) return a.type < b.type;
                    return a.size < b.size;
                });

                result.result.columns = {{"p_brand","string"},{"p_type","string"},{"p_size","int"},{"supplier_cnt","int"}};
                result.result.rows.clear();
                for (auto& r : results)
                    result.result.rows.push_back({r.brand, r.type, (int64_t)r.size, (int64_t)r.supplier_cnt});
                printf("\nQ16 Results:\n");
                printf("  +-----------+---------------------------+------+--------------+\n");
                printf("  | p_brand   | p_type                    |p_size| supplier_cnt |\n");
                printf("  +-----------+---------------------------+------+--------------+\n");
                int show = std::min((int)results.size(), 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-9s | %-25s | %4d | %12d |\n",
                           results[j].brand.c_str(), results[j].type.c_str(),
                           results[j].size, results[j].supplier_cnt);
                }
                printf("  +-----------+---------------------------+------+--------------+\n");
                printf("  Total groups: %d\n", (int)results.size());
            }
        }

        // Q21: read per-supp counts, join names, sort, top 100
        if (plan.name == "Q21") {
            auto& pd = g_q21Post;
            auto* buf = executor.getAllocatedBuffer("d_q21_supp_count");
            if (buf) {
                const uint32_t* suppCounts = (const uint32_t*)buf->contents();

                std::unordered_map<int, int> suppIdx;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++)
                    suppIdx[pd.s_suppkey[i]] = (int)i;

                struct Q21Row { std::string s_name; int numwait; };
                std::vector<Q21Row> rows;
                for (int sk = 0; sk <= pd.maxSuppkey; sk++) {
                    if (suppCounts[sk] > 0) {
                        auto it = suppIdx.find(sk);
                        if (it != suppIdx.end()) {
                            int si = it->second;
                            int len = 25;
                            while (len > 0 && (pd.s_name[si * 25 + len - 1] == ' ' ||
                                               pd.s_name[si * 25 + len - 1] == '\0')) len--;
                            rows.push_back({std::string(pd.s_name.data() + si * 25, len),
                                           (int)suppCounts[sk]});
                        }
                    }
                }

                std::sort(rows.begin(), rows.end(), [](const Q21Row& a, const Q21Row& b) {
                    if (a.numwait != b.numwait) return a.numwait > b.numwait;
                    return a.s_name < b.s_name;
                });

                int limit = std::min((int)rows.size(), 100);
                result.result.columns = {{"s_name","string"},{"numwait","int"}};
                result.result.rows.clear();
                for (int j = 0; j < limit; j++)
                    result.result.rows.push_back({rows[j].s_name, (int64_t)rows[j].numwait});
                printf("\nQ21 Results:\n");
                printf("  +---------------------------+----------+\n");
                printf("  | s_name                    | numwait  |\n");
                printf("  +---------------------------+----------+\n");
                int show = std::min(limit, 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-25s | %8d |\n", rows[j].s_name.c_str(), rows[j].numwait);
                }
                printf("  +---------------------------+----------+\n");
                printf("  Total qualifying suppliers: %d\n", (int)rows.size());
            }
        }

        // 7b. Correctness oracle — runs after CPU post-processing so all
        // per-query result.result populations above are complete.
        // Column matching is by name; extra/missing columns are skipped.
        if (!g_saveGoldenDir.empty() || !g_checkDir.empty()) {
            std::string canonical = result.result.toCanonical();
            std::string fname = queryName + "_" + timing.scaleFactor + ".csv";

            if (!g_saveGoldenDir.empty()) {
                ::mkdir(g_saveGoldenDir.c_str(), 0755); // ok if exists
                std::string path = g_saveGoldenDir + "/" + fname;
                std::ofstream of(path);
                of << canonical;
                if (!g_csv) printf("[GOLDEN] saved %s (%zu rows)\n",
                                   path.c_str(), result.result.numRows());
            }
            if (!g_checkDir.empty()) {
                std::string path = g_checkDir + "/" + fname;
                std::ifstream ifs(path);
                if (!ifs) {
                    fprintf(stderr, "[CHECK] %s: golden file missing: %s\n",
                            queryName.c_str(), path.c_str());
                    g_checkExitCode = 2;
                } else {
                    std::ostringstream buf;
                    buf << ifs.rdbuf();
                    std::string diff = compareCanonical(canonical, buf.str(),
                                                        g_checkAbsTol, g_checkRelTol);
                    if (diff.empty()) {
                        printf("[CHECK] %s @ %s: OK (%zu rows)\n",
                               queryName.c_str(), timing.scaleFactor.c_str(),
                               result.result.numRows());
                    } else {
                        fprintf(stderr, "[CHECK] %s @ %s: FAIL — %s\n",
                                queryName.c_str(), timing.scaleFactor.c_str(),
                                diff.c_str());
                        g_checkExitCode = 1;
                    }
                }
            }
        }

        auto postEnd = std::chrono::high_resolution_clock::now();
        double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();
        timing.postMs = postMs;

        printDetailedTimingSummary(timing, g_csv);

        executor.releaseAllocatedBuffers();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Codegen error (" << queryName << "): " << e.what() << std::endl;
        return false;
    }
}

// ===================================================================
// main
// ===================================================================

int main(int argc, const char* argv[]) {
    std::string query;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "help" || arg == "--help" || arg == "-h") {
            printf("GPU Database Codegen\n");
            printf("Usage: GPUDBCodegen [flags] [sf1|sf10|sf50|sf100] q<N>|mb<N>\n");
            printf("  q1..q22       - Run TPC-H query via codegen pipeline\n");
            printf("  all           - Run all 22 queries\n");
            printf("  mb1..mb7      - Run microbenchmark (sql/mb<N>.sql)\n");
            printf("  mball         - Run all microbenchmarks\n");
            printf("Loader flags:\n");
            printf("  --no-zerocopy        Disable zero-copy mmap path (force buffer copies)\n");
            printf("  --no-binary          Disable .colbin binary loader (force .tbl parser)\n");
            printf("  --chunk N[K|M|G]     Stream supported queries (Q1,Q6,Q12,Q14,Q19) from .colbin\n");
            printf("                       Auto-enabled when dataset exceeds 80%% of system RAM\n");
            printf("  --no-db              With --chunk, use one reusable chunk slot instead of two\n");
            printf("Experiment flags:\n");
            printf("  --warmup N           Run N untimed warmup iterations (default 3)\n");
            printf("  --repeat N           Run N timed iterations, report median (default 1)\n");
            printf("  --csv                Suppress text breakdown; emit one TIMING_CSV per trial\n");
            printf("  --threadgroup-size N Override default threadgroup size (default = plan-specified)\n");
            printf("  --autotune-tg        Per-query global TG sweep over {32,64,128,256,512,1024};\n");
            printf("                       picks the size with min p50 GPU time (logs AUTOTUNE_CSV)\n");
            printf("  --autotune-tg-per-phase  Per-phase TG sweep; picks min-p50 TG independently\n");
            printf("                       for each kernel (logs AUTOTUNE_PHASE_CSV)\n");
            printf("  --no-pipeline-cache  Recompile Metal source on every measured iteration\n");
            printf("  --fastmath           Enable Metal -ffast-math (default: off)\n");
            printf("  --no-fastmath        Disable Metal -ffast-math (default behavior)\n");
            printf("  --print-plan         Print the MetalQueryPlan structure before codegen\n");
            printf("  --dump-msl DIR       Write generated MSL to DIR/<query>.metal (default: debug/)\n");
            printf("  --check DIR          Compare GPU result against DIR/<query>_<sf>.csv (golden)\n");
            printf("  --save-golden DIR    Write current GPU result to DIR/<query>_<sf>.csv (overwrites)\n");
            printf("  --check-abs-tol N    Absolute float tolerance (default 1e-2)\n");
            printf("  --check-rel-tol N    Relative float tolerance (default 1e-4)\n");
            printf("  --scalar-atomic      Reduction ablation: every thread issues a global atomic\n");
            printf("                       (disables SIMD+TG reduce; for B2 ablation)\n");
            return 0;
        }
        if (arg == "--no-zerocopy")       { ::setenv("GPUDB_NO_ZEROCOPY", "1", 1); continue; }
        if (arg == "--no-binary")         { ::setenv("GPUDB_NO_BINARY",   "1", 1); continue; }
        if (arg == "--no-db")             { g_chunkDoubleBuffer = false; continue; }
        if (arg.rfind("--chunk=", 0) == 0) {
            if (!parseRowCountWithSuffix(arg.substr(8), g_chunkRowsExplicit)) {
                std::cerr << "Invalid value for --chunk: " << arg.substr(8) << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--chunk") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --chunk\n"; return 1; }
            std::string value = argv[++i];
            if (!parseRowCountWithSuffix(value, g_chunkRowsExplicit)) {
                std::cerr << "Invalid value for --chunk: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--scalar-atomic")     { ::setenv("GPUDB_SCALAR_ATOMIC", "1", 1); continue; }
        if (arg == "--csv")               { g_csv = true; continue; }
        if (arg == "--no-pipeline-cache") { g_noPipelineCache = true; continue; }
        if (arg == "--fastmath")          { g_fastMath = true; continue; }
        if (arg == "--no-fastmath")       { g_fastMath = false; continue; }
        if (arg == "--print-plan")        { g_printPlan = true; continue; }
        if (arg == "--dump-msl") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --dump-msl\n"; return 1; }
            g_dumpMslDir = argv[++i]; continue;
        }
        if (arg == "--check") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --check\n"; return 1; }
            g_checkDir = argv[++i]; continue;
        }
        if (arg == "--save-golden") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --save-golden\n"; return 1; }
            g_saveGoldenDir = argv[++i]; continue;
        }
        if (arg == "--check-abs-tol") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --check-abs-tol\n"; return 1; }
            g_checkAbsTol = std::atof(argv[++i]); continue;
        }
        if (arg == "--check-rel-tol") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --check-rel-tol\n"; return 1; }
            g_checkRelTol = std::atof(argv[++i]); continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --warmup\n"; return 1; }
            g_warmup = std::max(0, std::atoi(argv[++i])); continue;
        }
        if (arg == "--repeat") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --repeat\n"; return 1; }
            g_repeat = std::max(1, std::atoi(argv[++i])); continue;
        }
        if (arg == "--threadgroup-size") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --threadgroup-size\n"; return 1; }
            g_tgSizeOverride = std::max(0, std::atoi(argv[++i])); continue;
        }
        if (arg == "--autotune-tg")       { g_autotuneTg = true; continue; }
        if (arg == "--autotune-tg-per-phase") { g_autotuneTgPerPhase = true; continue; }
        if (arg == "sf1")  { g_dataset_path = "data/SF-1/"; continue; }
        if (arg == "sf10") { g_dataset_path = "data/SF-10/"; continue; }
        if (arg == "sf20") { g_dataset_path = "data/SF-20/"; continue; }
        if (arg == "sf50") { g_dataset_path = "data/SF-50/"; continue; }
        if (arg == "sf100") { g_dataset_path = "data/SF-100/"; continue; }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown flag: " << arg << std::endl;
            return 1;
        }
        if (!query.empty()) {
            std::cerr << "Unexpected extra query argument: " << arg << std::endl;
            return 1;
        }
        query = arg;
    }

    if (query.empty()) {
        std::cerr << "Usage: GPUDBCodegen [sf1|sf10|sf100] q<N>" << std::endl;
        return 1;
    }

    // Apply explicit fast-math selection globally before any compile() runs.
    codegen::RuntimeCompiler::setFastMathEnabled(g_fastMath);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "No Metal device found" << std::endl;
        return 1;
    }
    device->setShouldMaximizeConcurrentCompilation(true);
    MTL::CommandQueue* cmdQueue = device->newCommandQueue();

    printSystemInfo(getSystemInfo(device));

    auto runQuery = [&](int qNum) -> bool {
        std::string path = "sql/q" + std::to_string(qNum) + ".sql";
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open SQL file: " << path << std::endl;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        std::string sql = ss.str();
        std::string name = "Q" + std::to_string(qNum);
        return runCodegenQuery(device, cmdQueue, sql, name);
    };

    auto runMicrobench = [&](int mbNum) -> bool {
        std::string path = "sql/mb" + std::to_string(mbNum) + ".sql";
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open SQL file: " << path << std::endl;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        std::string sql = ss.str();
        std::string name = "MB" + std::to_string(mbNum);
        return runCodegenQuery(device, cmdQueue, sql, name);
    };

    bool ok = true;
    if (query == "all") {
        for (int q = 1; q <= 22; q++) ok = runQuery(q) && ok;
    } else if (query == "mball") {
        for (int m = 1; m <= 7; m++) ok = runMicrobench(m) && ok;
    } else if (query.size() >= 3 && query[0] == 'm' && query[1] == 'b') {
        int mbNum = std::stoi(query.substr(2));
        if (mbNum >= 1 && mbNum <= 99) {
            ok = runMicrobench(mbNum);
        } else {
            std::cerr << "Unknown microbench: " << query << std::endl;
            return 1;
        }
    } else if (query.size() >= 2 && query[0] == 'q') {
        int qNum = std::stoi(query.substr(1));
        if (qNum >= 1 && qNum <= 22) {
            ok = runQuery(qNum);
        } else {
            std::cerr << "Unknown query: " << query << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown query: " << query << std::endl;
        return 1;
    }

    pool->release();
    if (!ok && g_checkExitCode == 0) return 1;
    return g_checkExitCode;
}
