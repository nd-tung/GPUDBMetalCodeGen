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
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>

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
static bool g_noFastMath        = false; // --no-fastmath
static bool g_printPlan         = false; // --print-plan
static std::string g_dumpMslDir;         // --dump-msl PATH (directory or file template)
static std::string g_checkDir;           // --check DIR  (compare result vs DIR/<query>_<sf>.csv)
static std::string g_saveGoldenDir;      // --save-golden DIR
static double g_checkAbsTol = 1e-2;      // --check-abs-tol N
static double g_checkRelTol = 1e-4;      // --check-rel-tol N
static int    g_checkExitCode = 0;       // accumulated: nonzero if any --check failed
static size_t g_chunkRows = 0;           // --chunk N[K|M|G], 0 = full-table mode
static bool   g_chunkDoubleBuffer = true;// --no-db uses one reusable chunk slot

// Compare two canonical CSV blobs with float tolerance.
// Returns empty string on match; otherwise a short human-readable diff message.
static std::string compareCanonical(const std::string& got, const std::string& expected,
                                    double absTol, double relTol) {
    auto split = [](const std::string& s) {
        std::vector<std::string> lines;
        std::istringstream is(s);
        std::string ln;
        while (std::getline(is, ln)) lines.push_back(ln);
        return lines;
    };
    auto splitCsv = [](const std::string& ln) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : ln) {
            if (c == ',') { out.push_back(cur); cur.clear(); }
            else cur.push_back(c);
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

    auto a = split(got), b = split(expected);
    if (a.size() != b.size()) {
        char buf[128];
        snprintf(buf, sizeof(buf), "row count mismatch: got=%zu expected=%zu",
                 a.size(), b.size());
        return buf;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i]) continue;
        auto ca = splitCsv(a[i]);
        auto cb = splitCsv(b[i]);
        if (ca.size() != cb.size()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "line %zu: column count mismatch (%zu vs %zu)",
                     i, ca.size(), cb.size());
            return buf;
        }
        for (size_t c = 0; c < ca.size(); c++) {
            if (ca[c] == cb[c]) continue;
            double va, vb;
            if (isNumber(ca[c], va) && isNumber(cb[c], vb)) {
                double diff = std::fabs(va - vb);
                double tol = absTol + relTol * std::max(std::fabs(va), std::fabs(vb));
                if (diff <= tol) continue;
                char buf[256];
                snprintf(buf, sizeof(buf), "line %zu col %zu: %s vs %s (diff=%.6g tol=%.6g)",
                         i, c, ca[c].c_str(), cb[c].c_str(), diff, tol);
                return buf;
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "line %zu col %zu: '%s' vs '%s'",
                     i, c, ca[c].c_str(), cb[c].c_str());
            return buf;
        }
    }
    return "";
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

static double resultValueAsDouble(const codegen::GenericResult::Value& value, bool& wasDouble) {
    if (auto v = std::get_if<double>(&value)) {
        wasDouble = true;
        return *v;
    }
    if (auto v = std::get_if<int64_t>(&value)) {
        return (double)*v;
    }
    wasDouble = true;
    return 0.0;
}

struct ChunkResultAccumulator {
    codegen::MetalResultSchema::Kind kind = codegen::MetalResultSchema::NONE;
    std::vector<codegen::GenericResult::Column> columns;
    std::vector<double> scalarSums;
    std::vector<bool> scalarIsDouble;
    std::map<int64_t, std::vector<double>> keyedSums;
    std::vector<bool> keyedIsDouble;

    void add(const codegen::GenericResult& chunk, codegen::MetalResultSchema::Kind schemaKind) {
        if (chunk.columns.empty()) return;
        if (columns.empty()) columns = chunk.columns;
        kind = schemaKind;

        if (schemaKind == codegen::MetalResultSchema::SCALAR_AGG) {
            if (chunk.rows.empty()) return;
            if (scalarSums.empty()) {
                scalarSums.assign(chunk.rows[0].size(), 0.0);
                scalarIsDouble.assign(chunk.rows[0].size(), false);
            }
            for (size_t i = 0; i < chunk.rows[0].size(); ++i) {
                bool wasDouble = false;
                scalarSums[i] += resultValueAsDouble(chunk.rows[0][i], wasDouble);
                scalarIsDouble[i] = scalarIsDouble[i] || wasDouble;
            }
            return;
        }

        if (schemaKind == codegen::MetalResultSchema::KEYED_AGG) {
            if (keyedIsDouble.empty() && chunk.columns.size() > 1) {
                keyedIsDouble.assign(chunk.columns.size() - 1, false);
            }
            for (const auto& row : chunk.rows) {
                if (row.empty()) continue;
                int64_t bucket = 0;
                if (auto b = std::get_if<int64_t>(&row[0])) bucket = *b;
                else if (auto b = std::get_if<double>(&row[0])) bucket = (int64_t)*b;
                auto& sums = keyedSums[bucket];
                if (sums.empty()) sums.assign(row.size() - 1, 0.0);
                for (size_t i = 1; i < row.size(); ++i) {
                    bool wasDouble = false;
                    sums[i - 1] += resultValueAsDouble(row[i], wasDouble);
                    keyedIsDouble[i - 1] = keyedIsDouble[i - 1] || wasDouble;
                }
            }
        }
    }

    codegen::GenericResult finish() const {
        codegen::GenericResult out;
        out.columns = columns;
        if (kind == codegen::MetalResultSchema::SCALAR_AGG) {
            codegen::GenericResult::Row row;
            for (size_t i = 0; i < scalarSums.size(); ++i) {
                if (i < scalarIsDouble.size() && scalarIsDouble[i]) row.push_back(scalarSums[i]);
                else row.push_back((int64_t)std::llround(scalarSums[i]));
            }
            if (!row.empty()) out.rows.push_back(std::move(row));
            return out;
        }
        if (kind == codegen::MetalResultSchema::KEYED_AGG) {
            for (const auto& [bucket, sums] : keyedSums) {
                codegen::GenericResult::Row row;
                row.push_back(bucket);
                for (size_t i = 0; i < sums.size(); ++i) {
                    if (i < keyedIsDouble.size() && keyedIsDouble[i]) row.push_back(sums[i]);
                    else row.push_back((int64_t)std::llround(sums[i]));
                }
                out.rows.push_back(std::move(row));
            }
        }
        return out;
    }
};

static bool chunkedQuerySupported(const std::string& queryName, std::string& reason) {
    if (queryName == "Q1" || queryName == "Q6" || queryName == "Q14" || queryName == "Q19") {
        return true;
    }
    reason = "--chunk currently supports Q1, Q6, Q14, and Q19. Other queries need query-specific state merging.";
    return false;
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

static std::string chooseStreamTable(const std::map<std::string, std::set<std::string>>& tableCols) {
    if (tableCols.count("lineitem")) return "lineitem";
    if (tableCols.count("orders")) return "orders";
    return tableCols.empty() ? std::string{} : tableCols.begin()->first;
}

static void registerMaxKeySymbols(
    codegen::MetalGenericExecutor& executor,
    const std::vector<codegen::LoadedQueryTable>& loadedTables,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const codegen::TPCHSchema& schema) {
    int maxCk = 0, maxSk = 0, maxOk = 0, maxPk = 0;
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
            if (colName == "c_custkey" || colName == "o_custkey")
                for (size_t i = 0; i < nRows; ++i) maxCk = std::max(maxCk, data[i]);
            else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
                for (size_t i = 0; i < nRows; ++i) maxSk = std::max(maxSk, data[i]);
            else if (colName == "o_orderkey" || colName == "l_orderkey")
                for (size_t i = 0; i < nRows; ++i) maxOk = std::max(maxOk, data[i]);
            else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
                for (size_t i = 0; i < nRows; ++i) maxPk = std::max(maxPk, data[i]);
        }
    }
    executor.registerSymbol("maxCustkey", maxCk + 1);
    executor.registerSymbol("maxSuppkey", maxSk + 1);
    executor.registerSymbol("maxOrderkey", maxOk + 1);
    executor.registerSymbol("maxPartkey", maxPk + 1);
}

static bool handleGoldenResult(const std::string& queryName,
                               const std::string& scaleFactor,
                               const codegen::GenericResult& result) {
    if (g_saveGoldenDir.empty() && g_checkDir.empty()) return true;

    std::string canonical = result.toCanonical();
    std::string fname = queryName + "_" + scaleFactor + ".csv";

    if (!g_saveGoldenDir.empty()) {
        ::mkdir(g_saveGoldenDir.c_str(), 0755);
        std::string path = g_saveGoldenDir + "/" + fname;
        std::ofstream of(path);
        of << canonical;
        if (!g_csv) printf("[GOLDEN] saved %s (%zu rows)\n", path.c_str(), result.numRows());
    }
    if (!g_checkDir.empty()) {
        std::string path = g_checkDir + "/" + fname;
        std::ifstream ifs(path);
        if (!ifs) {
            fprintf(stderr, "[CHECK] %s: golden file missing: %s\n", queryName.c_str(), path.c_str());
            g_checkExitCode = 2;
            return false;
        }
        std::ostringstream buf;
        buf << ifs.rdbuf();
        std::string diff = compareCanonical(canonical, buf.str(), g_checkAbsTol, g_checkRelTol);
        if (diff.empty()) {
            printf("[CHECK] %s @ %s: OK (%zu rows)\n",
                   queryName.c_str(), scaleFactor.c_str(), result.numRows());
        } else {
            fprintf(stderr, "[CHECK] %s @ %s: FAIL - %s\n",
                    queryName.c_str(), scaleFactor.c_str(), diff.c_str());
            g_checkExitCode = 1;
            return false;
        }
    }
    return true;
}

static bool runChunkedCodegenQuery(
    MTL::Device* device,
    MTL::CommandQueue* cmdQueue,
    const codegen::RuntimeCompiler::CompiledQuery& compiled,
    const codegen::MetalCodegen& cg,
    const codegen::MetalQueryPlan& plan,
    const std::map<std::string, std::set<std::string>>& tableCols,
    const codegen::TPCHSchema& schema,
    DetailedTiming& timing) {

    std::string reason;
    if (!chunkedQuerySupported(plan.name, reason)) {
        std::cerr << "Codegen: " << reason << std::endl;
        return false;
    }
    if (cg.getResultSchema().kind != codegen::MetalResultSchema::SCALAR_AGG &&
        cg.getResultSchema().kind != codegen::MetalResultSchema::KEYED_AGG) {
        std::cerr << "Codegen: --chunk only supports scalar/keyed aggregate result schemas" << std::endl;
        return false;
    }

    using clk = std::chrono::high_resolution_clock;
    auto elapsedMs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const std::string streamTable = chooseStreamTable(tableCols);
    if (streamTable.empty()) {
        std::cerr << "Codegen: --chunk could not identify a streamed table" << std::endl;
        return false;
    }

    std::map<std::string, std::vector<ColSpec>> tableSpecs;
    for (const auto& [tableName, colNames] : tableCols) {
        const auto& tdef = schema.table(tableName);
        auto& specs = tableSpecs[tableName];
        for (const auto& colName : colNames) {
            specs.push_back(colSpecFor(tdef.col(colName)));
        }
    }

    const int streamSlots = g_chunkDoubleBuffer ? 2 : 1;
    auto loadStart = clk::now();

    codegen::ChunkedColbinTable stream;
    std::string streamError;
    if (!stream.open(device, g_dataset_path + streamTable + ".tbl",
                     tableSpecs[streamTable], g_chunkRows, streamSlots, streamError)) {
        std::cerr << "Codegen: --chunk failed for " << streamTable << ": "
                  << streamError << std::endl;
        return false;
    }

    codegen::MetalGenericExecutor executor(device, cmdQueue);
    std::vector<codegen::LoadedQueryTable> loadedTables;
    for (const auto& [tableName, specs] : tableSpecs) {
        if (tableName == streamTable) continue;
        auto cols = loadQueryColumns(device, g_dataset_path + tableName + ".tbl", specs);
        size_t rowCount = cols.rows();
        const auto& tdef = schema.table(tableName);
        for (const auto& colName : tableCols.at(tableName)) {
            const auto& cdef = tdef.col(colName);
            if (MTL::Buffer* buf = cols.buffer(cdef.index)) {
                executor.registerTableBuffer(colName, buf, rowCount);
            }
        }
        executor.registerTableRowCount(tableName, rowCount);
        loadedTables.emplace_back(tableName, std::move(cols));
    }

    registerMaxKeySymbols(executor, loadedTables, tableCols, schema);
    if (!codegen::prepareQueryPreprocessing(plan.name, device, executor, loadedTables)) {
        return false;
    }

    double loadSetupMs = elapsedMs(loadStart, clk::now());
    double chunkCopyMs = 0.0;
    double bufferAllocMs = 0.0;
    double gpuMs = 0.0;
    double mergeMs = 0.0;
    std::map<std::string, double> phaseSums;
    ChunkResultAccumulator merged;

    const auto& streamTdef = schema.table(streamTable);
    const size_t totalRows = stream.rows();
    const size_t chunkRows = stream.chunkRows();
    size_t chunkCount = 0;

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
                std::cerr << "Codegen: missing chunk buffer for " << streamTable
                          << "." << colName << std::endl;
                return false;
            }
            executor.registerTableBuffer(colName, buf, rowsThisChunk);
        }
        executor.registerTableRowCount(streamTable, rowsThisChunk);

        auto result = executor.execute(compiled, cg, 0, 1);
        bufferAllocMs += result.bufferAllocTimeMs;
        gpuMs += result.totalKernelTimeMs;
        for (size_t i = 0; i < result.phaseTimesMs.size(); ++i) {
            const std::string name = (i < result.phaseNames.size())
                ? result.phaseNames[i] : ("phase" + std::to_string(i));
            phaseSums[name] += result.phaseTimesMs[i];
        }

        auto mergeStart = clk::now();
        merged.add(result.result, cg.getResultSchema().kind);
        mergeMs += elapsedMs(mergeStart, clk::now());
        chunkCount++;
    }

    codegen::GenericResult finalResult = merged.finish();
    timing.dataLoadMs = loadSetupMs + chunkCopyMs;
    timing.ingestMs = 0.0;
    timing.loadSource = "chunked-colbin";
    timing.loadBytes = loadStats().bytes + stream.bytesLoaded();
    timing.bufferAllocMs = bufferAllocMs;
    timing.gpuTotalMs = gpuMs;
    timing.gpuTrialsN = 1;
    timing.gpuMsP10 = gpuMs;
    timing.gpuMsP90 = gpuMs;
    timing.gpuMsMad = 0.0;
    timing.postMs = mergeMs;
    timing.phaseKernelMs.clear();
    for (const auto& [name, ms] : phaseSums) {
        timing.phaseKernelMs.emplace_back(name, ms);
    }

    if (!g_csv) {
        printf("\n%s Results:\n", plan.name.c_str());
        finalResult.print();
    }

    handleGoldenResult(plan.name, timing.scaleFactor, finalResult);

    if (plan.name == "Q14" && finalResult.numRows() == 1 && finalResult.columns.size() == 2) {
        double promo = std::get<double>(finalResult.rows[0][0]);
        double total = std::get<double>(finalResult.rows[0][1]);
        if (total > 0 && !g_csv) {
            printf("  -> promo_revenue = %.2f%%\n", 100.0 * promo / total);
        }
    }

    printf("SF100 streaming: %zu chunks, parse=%.3fms, GPU=%.3fms, post=%.3fms, chunk_rows=%zu, slots=%d\n",
           chunkCount, timing.dataLoadMs, timing.gpuTotalMs, timing.postMs,
           chunkRows, streamSlots);
    printf("STREAMING_CSV,%s,%s,%s,%zu,%zu,%d,%.3f,%.3f,%.3f,%zu\n",
           timing.scaleFactor.c_str(), timing.queryName.c_str(), streamTable.c_str(),
           chunkRows, chunkCount, streamSlots, timing.dataLoadMs,
           timing.gpuTotalMs, timing.postMs, timing.loadBytes);

    printDetailedTimingSummary(timing, g_csv);
    executor.releaseAllocatedBuffers();
    return true;
}

// ===================================================================
// Run a query through the codegen pipeline
// ===================================================================

static bool runCodegenQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                            const std::string& sql, const std::string& queryName) {
    if (!g_csv) printf("\n=== Codegen: %s ===\n", queryName.c_str());
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
        if (g_chunkRows > 0) {
            return runChunkedCodegenQuery(device, cmdQueue, compiled, cg, plan,
                                          tableCols, schema, timing);
        }

        auto parseStart = std::chrono::high_resolution_clock::now();
        codegen::MetalGenericExecutor executor(device, cmdQueue);

        // Load each table's columns and register with executor (zero-copy via QueryColumns).
        std::vector<std::pair<std::string, QueryColumns>> loadedTables;
        for (auto& [tableName, colNames] : tableCols) {
            const auto& tdef = schema.table(tableName);
            std::vector<ColSpec> specs;
            for (const auto& colName : colNames) {
                specs.push_back(colSpecFor(tdef.col(colName)));
            }

            auto cols = loadQueryColumns(device, g_dataset_path + tableName + ".tbl", specs);

            size_t rowCount = cols.rows();
            for (const auto& colName : colNames) {
                auto& cdef = tdef.col(colName);
                MTL::Buffer* buf = cols.buffer(cdef.index);
                if (!buf) continue;
                size_t count = rowCount;
                if (cdef.type == codegen::DataType::CHAR_FIXED) {
                    // rowCount is already rows; registerTableBuffer expects row count.
                } else if (cdef.type == codegen::DataType::CHAR1) {
                    // one char per row; rowCount == bytes.
                }
                executor.registerTableBuffer(colName, buf, count);
            }

            executor.registerTableRowCount(tableName, rowCount);
            loadedTables.emplace_back(tableName, std::move(cols));
        }

        // ---------------------------------------------------------------
        // Compute max key values for dynamic buffer sizing
        // ---------------------------------------------------------------
        registerMaxKeySymbols(executor, loadedTables, tableCols, schema);

        // ---------------------------------------------------------------
        // Per-query pre-processing: resolve small lookup tables/scalars
        // ---------------------------------------------------------------
        if (!codegen::prepareQueryPreprocessing(plan.name, device, executor, loadedTables)) {
            return false;
        }

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
        codegen::MetalExecutionResult result;

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

        auto postStart = std::chrono::high_resolution_clock::now();

        // 7. Print results (suppressed under --csv to keep stdout machine-readable)
        if (!g_csv) {
            printf("\n%s Results:\n", queryName.c_str());
            result.result.print();
        }

        // 7b. Correctness oracle (golden-result compare).
        // Operates on the GPU result struct only (pre-CPU-postprocess), so
        // queries that rely on CPU sort/format (Q2, Q16, Q21) are validated
        // up to GPU output; the post-processed output is not (yet) checked.
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

        // ---------------------------------------------------------------
        // Per-query post-processing
        // ---------------------------------------------------------------

        // Q14: compute 100 * promo / total
        if (plan.name == "Q14" && result.result.numRows() == 1 &&
            result.result.columns.size() == 2) {
            double promo = std::get<double>(result.result.rows[0][0]);
            double total = std::get<double>(result.result.rows[0][1]);
            if (total > 0)
                printf("  → promo_revenue = %.2f%%\n", 100.0 * promo / total);
        }

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
                    std::string nm;
                    for (int c = 0; c < 25; c++) {
                        char ch = n_nm[i * 25 + c];
                        if (ch == ' ' || ch == '\0') break;
                        nm += ch;
                    }
                    nationNames[n_nk[i]] = nm;
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
                    if (pd.htKeys[slot] == 0xFFFFFFFFu) continue;
                    int psIdx = pd.htPsIdx[slot];
                    if (psIdx < 0) continue;
                    float sumQty = htVals[slot];
                    if (sumQty > 0.0f && (float)pd.ps_availqty[psIdx] > 0.5f * sumQty) {
                        qualSuppkeys.insert(pd.ps_suppkey[psIdx]);
                    }
                }

                struct Q20Row { std::string name; std::string address; };
                std::vector<Q20Row> rows;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++) {
                    if (pd.s_nationkey[i] != pd.canada_nk) continue;
                    if (!qualSuppkeys.count(pd.s_suppkey[i])) continue;
                    std::string nm, addr;
                    for (int c = 0; c < 25; c++) {
                        char ch = pd.s_name[i * 25 + c];
                        if (ch == ' ' || ch == '\0') break;
                        nm += ch;
                    }
                    for (int c = 0; c < 40; c++) {
                        char ch = pd.s_address[i * 40 + c];
                        if (ch == '\0') break;
                        addr += ch;
                    }
                    rows.push_back({nm, addr});
                }
                std::sort(rows.begin(), rows.end(), [](const Q20Row& a, const Q20Row& b) {
                    return a.name < b.name;
                });

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
                        std::string s;
                        for (int c = 0; c < width; c++) {
                            char ch = data[idx * width + c];
                            if (ch == '\0') break;
                            if (ch == ' ' && c > 0) {
                                bool allSpace = true;
                                for (int d = c; d < width; d++) {
                                    if (data[idx * width + d] != ' ' && data[idx * width + d] != '\0') {
                                        allSpace = false; break;
                                    }
                                }
                                if (allSpace) break;
                            }
                            s += ch;
                        }
                        return s;
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
            printf("  --chunk N[K|M|G]     Stream supported queries from .colbin in N-row chunks\n");
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
            printf("  --no-fastmath        Disable Metal -ffast-math (numerics study)\n");
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
            if (!parseRowCountWithSuffix(arg.substr(8), g_chunkRows)) {
                std::cerr << "Invalid value for --chunk: " << arg.substr(8) << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--chunk") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --chunk\n"; return 1; }
            std::string value = argv[++i];
            if (!parseRowCountWithSuffix(value, g_chunkRows)) {
                std::cerr << "Invalid value for --chunk: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--scalar-atomic")     { ::setenv("GPUDB_SCALAR_ATOMIC", "1", 1); continue; }
        if (arg == "--csv")               { g_csv = true; continue; }
        if (arg == "--no-pipeline-cache") { g_noPipelineCache = true; continue; }
        if (arg == "--no-fastmath")       { g_noFastMath = true; continue; }
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

    // Apply --no-fastmath globally before any compile() runs.
    if (g_noFastMath) codegen::RuntimeCompiler::setFastMathEnabled(false);

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
