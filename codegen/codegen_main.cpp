// Standalone SQL-to-Metal entry point.

#include "core/infra.h"
#include "query_analyzer.h"
#include "runtime_compiler.h"
#include "metal_plan_builder.h"
#include "api/metal_adhoc_plan_api.h"
#include "api/metal_tpch_plan_api.h"
#include "metal_generic_executor.h"
#include "max_key_symbols.h"
#include "query_preprocessing.h"
#include "chunked_colbin_loader.h"
#include "tpch_schema.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <set>
#include <map>
#include <algorithm>
#include <memory>
#include <optional>
#include <cstring>

// Runtime flags shared by main and runCodegenQuery.
static int  g_warmup            = 3;     // --warmup N
static int  g_repeat            = 1;     // --repeat N
static bool g_csv               = false; // --csv  (suppress human-readable breakdown)
static int  g_tgSizeOverride    = 0;     // --threadgroup-size N (0 = use plan default)
static bool g_autotuneTg        = false; // --autotune-tg  (per-query global TG sweep)
static bool g_autotuneTgPerPhase= false; // --autotune-tg-per-phase (per-kernel TG)
static bool g_noPipelineCache   = false; // --no-pipeline-cache
static bool g_profilePhases     = false; // --profile-phases
static bool g_fastMath          = false; // --fastmath
static bool g_printPlan         = false; // --print-plan
static std::string g_dumpMslDir;         // --dump-msl PATH (directory or file template)
static std::string g_checkDir;           // --check DIR  (compare result vs DIR/<query>_<sf>.csv)
static std::string g_saveGoldenDir;      // --save-golden DIR
static double g_checkAbsTol = 1e-2;      // --check-abs-tol N
static double g_checkRelTol = 1e-4;      // --check-rel-tol N
static int    g_checkExitCode = 0;       // accumulated: nonzero if any --check failed
static size_t g_chunkRows = 0;           // --chunk N[K|M|G], 0 = full-table mode
static size_t g_chunkRowsExplicit = 0;    // user-set --chunk rows; resets g_chunkRows per query
static bool   g_chunkDoubleBuffer = true;// --no-db uses one reusable chunk slot
static bool   g_autoChunk = true;         // --no-auto-chunk disables budget trigger
static bool   g_forceChunk = false;       // --force-chunk disables explicit downgrade

enum class QueryApiKind {
    PredefinedTPCH,
    AdhocSQL,
};

struct HostPostOpTracker {
    std::vector<std::string> ops;
    std::set<std::string> seen;

    void mark(const std::string& op) {
        if (seen.insert(op).second) ops.push_back(op);
    }

    bool empty() const {
        return ops.empty();
    }

    std::string joined() const {
        std::string out;
        for (size_t i = 0; i < ops.size(); i++) {
            if (i) out += ";";
            out += ops[i];
        }
        return out;
    }
};

static double medianValue(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t mid = v.size() / 2;
    if (v.size() % 2 == 1) return v[mid];
    return 0.5 * (v[mid - 1] + v[mid]);
}

static double percentileValue(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double rank = std::ceil(p * (double)v.size());
    size_t i = (size_t)std::min<double>(std::max<double>(rank - 1.0, 0.0),
                                        (double)(v.size() - 1));
    return v[i];
}

static double medianAbsoluteDeviation(std::vector<double> v) {
    if (v.size() < 2) return 0.0;
    double m = medianValue(v);
    std::vector<double> dev;
    dev.reserve(v.size());
    for (double x : v) dev.push_back(std::fabs(x - m));
    return medianValue(std::move(dev));
}

// Compare two canonical CSV blobs with float tolerance.
// Golden schemas must match exactly; numeric values are compared with tolerance.
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
    auto rtrimSpaces = [](std::string s) {
        while (!s.empty() && s.back() == ' ') s.pop_back();
        return s;
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

    auto aHdr = splitCsv(aLines[0]);
    auto bHdr = splitCsv(bLines[0]);

    auto joinHdr = [](const std::vector<std::string>& hdr) {
        std::string out;
        for (size_t i = 0; i < hdr.size(); i++) {
            if (i) out += ",";
            out += hdr[i];
        }
        return out;
    };

    if (aHdr != bHdr) {
        return "schema mismatch: got cols=[" + joinHdr(aHdr) +
               "] expected cols=[" + joinHdr(bHdr) + "]";
    }

    std::vector<std::pair<size_t,size_t>> sharedCols;
    for (size_t ai = 0; ai < aHdr.size(); ai++) {
        for (size_t bi = 0; bi < bHdr.size(); bi++) {
            if (aHdr[ai] == bHdr[bi]) {
                sharedCols.push_back({ai, bi});
                break;
            }
        }
    }

    if (sharedCols.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "schema mismatch: got cols=[%s] expected cols=[%s]",
                 aLines[0].c_str(), bLines[0].c_str());
        return buf;
    }

    size_t aData = aLines.size() - 1;
    size_t bData = bLines.size() - 1;
    if (aData != bData) {
        char buf[128];
        snprintf(buf, sizeof(buf), "row count mismatch: got=%zu expected=%zu", aData, bData);
        return buf;
    }

    for (size_t row = 0; row < aData; row++) {
        auto aRow = splitCsv(aLines[row + 1]);
        auto bRow = splitCsv(bLines[row + 1]);
        for (auto [ai, bi] : sharedCols) {
            if (ai >= aRow.size() || bi >= bRow.size()) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "row %zu column count mismatch: got=%zu expected=%zu",
                         row + 1, aRow.size(), bRow.size());
                return buf;
            }
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
            if (rtrimSpaces(av) == rtrimSpaces(bv)) continue;
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

// Reorder materialized rows by the GPU sort index.
static void applyGpuSortRemap(codegen::GenericResult& result,
                               const codegen::MetalQueryPlan::GpuSort& gpuSort,
                               codegen::MetalGenericExecutor& executor) {
    auto* idxBuf = executor.getAllocatedBuffer(gpuSort.sortedIndexBuffer);
    if (!idxBuf) return;

    size_t n_results = 0;
    executor.tryGetSymbol(gpuSort.nResults, n_results);
    if (n_results == 0 || n_results > result.rows.size())
        n_results = result.rows.size();

    const int* indices = static_cast<const int*>(idxBuf->contents());
    if (!indices) return;

    const size_t outRows = (gpuSort.limit >= 0)
        ? std::min(n_results, static_cast<size_t>(gpuSort.limit))
        : n_results;

    // Remap only the visible prefix; remapping all rows would undo GPU top-k.
    std::vector<size_t> remap(outRows);
    for (size_t i = 0; i < outRows; ++i) {
        int src = indices[i];
        remap[i] = (src >= 0 && static_cast<size_t>(src) < result.rows.size())
                       ? static_cast<size_t>(src) : i;
    }

    std::vector<codegen::GenericResult::Row> sortedRows;
    sortedRows.reserve(outRows);
    for (size_t i = 0; i < outRows; ++i) {
        sortedRows.push_back(std::move(result.rows[remap[i]]));
    }
    result.rows = std::move(sortedRows);
}

// Read .colbin row count and file size without mapping the payload.
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

// Use the largest referenced .colbin as the streaming table.
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
    // Use a deterministic stream table when .colbin metadata is unavailable.
    if (best.empty()) {
        if (tableCols.count("lineitem")) return "lineitem";
        if (tableCols.count("orders"))   return "orders";
        return tableCols.empty() ? std::string{} : tableCols.begin()->first;
    }
    return best;
}

static uint64_t saturatingAdd(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        return std::numeric_limits<uint64_t>::max();
    return a + b;
}

static uint64_t saturatingMul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return std::numeric_limits<uint64_t>::max();
    return a * b;
}

struct DeviceBufferEstimate {
    uint64_t totalBytes = 0;
    uint64_t largestBytes = 0;
    std::string largestName;
    size_t resolvedBuffers = 0;
    size_t unresolvedBuffers = 0;
};

static DeviceBufferEstimate estimateDeviceBufferBytes(
        const codegen::MetalCodegen& cg,
        const std::map<std::string, uint64_t>& tableRows) {
    codegen::MetalSizeResolver resolver;
    for (const auto& [table, rows] : tableRows) {
        resolver.registerSymbol(codegen::tableSizeName(table), (size_t)rows);
        resolver.registerSymbol("num" + table, (size_t)rows);
    }

    DeviceBufferEstimate out;
    for (const auto& b : cg.getAllBindings()) {
        if (b.kind != codegen::MetalParamKind::DeviceBuffer ||
            b.readOnly || b.sizeExpr.empty()) {
            continue;
        }
        size_t count = 0;
        try {
            count = resolver.resolve(b.sizeExpr);
        } catch (...) {
            out.unresolvedBuffers++;
            continue;
        }
        uint64_t bytes = saturatingMul((uint64_t)count,
                                       (uint64_t)b.elemSizeBytes());
        if (bytes == 0) bytes = (uint64_t)b.elemSizeBytes();
        out.totalBytes = saturatingAdd(out.totalBytes, bytes);
        out.resolvedBuffers++;
        if (bytes > out.largestBytes) {
            out.largestBytes = bytes;
            out.largestName = b.name;
        }
    }
    return out;
}

static bool runCodegenQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                            const std::string& sql, const std::string& queryName,
                            QueryApiKind apiKind) {
    if (!g_csv) printf("\n=== Codegen: %s ===\n", queryName.c_str());
    const bool isPredefinedTpchRoute = apiKind == QueryApiKind::PredefinedTPCH;
    // Prevent auto-chunk decisions from leaking across `all` / `mball`.
    g_chunkRows = g_chunkRowsExplicit;
    try {
        using clk = std::chrono::high_resolution_clock;
        auto elapsedMs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        DetailedTiming timing{};
        timing.queryName = queryName;
        timing.route = isPredefinedTpchRoute ? "predefined" : "generic";
        {
            // Derive short SF label from g_dataset_path.
            const std::string& p = g_dataset_path;
            auto s = p.find("SF-");
            if (s != std::string::npos) {
                std::string digits;
                for (size_t i = s + 3; i < p.size() && isdigit((unsigned char)p[i]); i++)
                    digits += p[i];
                timing.scaleFactor = "SF" + digits;
            }
        }

        // Analyze SQL only for the ad-hoc route.
        codegen::AnalyzedQuery analyzed;
        bool analyzedOk = false;
        if (apiKind == QueryApiKind::AdhocSQL) {
            auto tAnalyze0 = clk::now();
            try {
                analyzed = codegen::analyzeSQL(sql);
                analyzedOk = true;
            } catch (const std::exception& e) {
                std::cerr << "Codegen: SQL analysis failed for " << queryName
                          << ": " << e.what() << std::endl;
                return false;
            }
            timing.analyzeMs = elapsedMs(tAnalyze0, clk::now());
        }

        // Build operator-based plan.
        auto tPlan0 = clk::now();
        std::optional<codegen::MetalQueryPlan> maybePlan;
        if (isPredefinedTpchRoute) {
            maybePlan = codegen::buildPredefinedTPCHPlan(queryName);
            if (!maybePlan) {
                std::cerr << "Codegen: predefined TPC-H plan not available for "
                          << queryName << std::endl;
                return false;
            }
        } else {
            if (!analyzedOk) {
                std::cerr << "Codegen: ad-hoc SQL requires successful analysis for "
                          << queryName << std::endl;
                return false;
            }
            std::string planError;
            maybePlan = codegen::buildAdhocSQLPlan(analyzed, queryName, &planError);
            if (!maybePlan) {
                std::cerr << "Codegen: ad-hoc SQL pattern not supported for "
                          << queryName << std::endl;
                if (!planError.empty()) {
                    std::cerr << "  Reason: " << planError << std::endl;
                }
                return false;
            }
        }
        auto& plan = *maybePlan;
        plan.name = queryName;
        timing.planMs = elapsedMs(tPlan0, clk::now());

        // Chunking is allowed only for plans marked chunkable.
        if (!plan.chunkable) {
            if (g_chunkRowsExplicit > 0) {
                std::cerr << "Codegen: " << queryName
                          << " does not support chunked execution yet "
                             "(--chunk is unsafe for this query — see "
                             "DOCUMENTATION.md §9.4).\n";
                return false;
            }
            g_chunkRows = 0;
        }

        // Experiment override.
        if (g_tgSizeOverride > 0) {
            for (auto& ph : plan.phases) ph.threadgroupSize = g_tgSizeOverride;
        }

        // Print phase summary and operator-tree JSON.
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
            if (plan.gpuSort) {
                printf("  gpuSort.index     : %s  limit=%d\n",
                       plan.gpuSort->sortedIndexBuffer.c_str(), plan.gpuSort->limit);
            }
            printf("---\n");

            try {
                std::string planFile = "debug/codegen_debug_" + plan.name + "_plan.json";
                std::ofstream ofs(planFile);
                ofs << plan.toTreeJSON().dump(2) << std::endl;
            } catch (...) {}
        }

        // Generate Metal source.
        auto tCodegen0 = clk::now();
        // Predefined TPC-H plans need the default schema for auto-projection.
        auto cg = codegen::generateFromPlan(plan, analyzed.schema ? analyzed.schema : &codegen::defaultSchemaProvider());
        std::string metalSource = cg.print();
        timing.codegenMs = elapsedMs(tCodegen0, clk::now());

        if (!g_csv) {
            printf("Generated Metal source (%zu bytes, %d phase(s))\n",
                   metalSource.size(), cg.phaseCount());
        }

        // Dump generated source.
        {
            std::string dumpDir = g_dumpMslDir.empty() ? "debug" : g_dumpMslDir;
            std::string path = dumpDir + "/codegen_debug_" + queryName + ".metal";
            std::ofstream dbg(path);
            dbg << metalSource;
            if (!g_csv) printf("  (written to %s)\n", path.c_str());
        }

        // Compile Metal source.
        auto tCompile0 = clk::now();
        codegen::RuntimeCompiler compiler(device);
        auto* library = compiler.compile(metalSource);
        timing.compileMs = elapsedMs(tCompile0, clk::now());
        if (!library) {
            std::cerr << "Codegen: Metal compilation failed" << std::endl;
            return false;
        }

        // Build one PSO per phase.
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
        // Collect columns referenced by all phases.
        std::map<std::string, std::set<std::string>> tableCols;
        for (const auto& phase : cg.getPhases()) {
            for (const auto& b : phase.bindings) {
                if (b.kind == codegen::MetalParamKind::TableData && !b.tableName.empty()) {
                    tableCols[b.tableName].insert(b.name);
                }
            }
        }

        // Load data via full-table buffers or chunked .colbin streaming.
        loadStats().reset();
        // Chunk sizing uses projected bytes, not full file size. Physmem is
        // the trigger on UMA because reclaimable-page accounting is too tight.
        if (plan.chunkable && (g_autoChunk || g_chunkRows > 0)) {
            const std::string autoStreamTable = autoDetectStreamTable(tableCols);
            if (!autoStreamTable.empty()) {
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
                // availMemBytes is diagnostic only.
                uint64_t totalBudget = std::min(physMemBytes, gpuBudgetBytes);

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

                // Full file size is diagnostic; projected size drives chunking.
                uint64_t totalDataBytes = 0;
                std::map<std::string, uint64_t> fullTableRows;
                for (const auto& [tName, _cols] : tableCols) {
                    uint64_t nr = 0, fsz = 0;
                    if (peekColbinHeader(g_dataset_path + tName + ".colbin", nr, fsz)) {
                        totalDataBytes += fsz;
                        fullTableRows[tName] = nr;
                    }
                }

                uint64_t residentBytes = 0;
                for (const auto& [tName, cols] : tableCols) {
                    if (tName == autoStreamTable) continue;
                    uint64_t nr = 0, fsz = 0;
                    if (!peekColbinHeader(g_dataset_path + tName + ".colbin", nr, fsz))
                        continue;
                    residentBytes += nr * projectedBytesPerRow(tName, cols);
                }

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
                auto estimateForStreamRows = [&](uint64_t streamRows) {
                    auto rows = fullTableRows;
                    rows[autoStreamTable] = streamRows;
                    return estimateDeviceBufferBytes(cg, rows);
                };

                DeviceBufferEstimate fullDeviceBuffers =
                    estimateDeviceBufferBytes(cg, fullTableRows);

                // Resident buffers plus projected stream buffers plus
                // generated operator buffers. The latter catches hash/group/sort
                // allocations that can dwarf table input size.
                uint64_t projectedWorkingSet = saturatingAdd(
                    saturatingAdd(residentBytes, streamProjectedBytes),
                    fullDeviceBuffers.totalBytes);

                constexpr double kThreshold    = 0.75;
                constexpr double kBudgetFraction = 0.50; // headroom for hash maps, output, kernels
                const bool fitsInBudget =
                    (totalBudget > 0) &&
                    (projectedWorkingSet <= (uint64_t)(totalBudget * kThreshold));

                // Downgrade explicit chunking when direct load fits.
                if (g_chunkRows > 0 && fitsInBudget) {
                    const char* force = std::getenv("GPUDB_FORCE_CHUNK");
                    bool forceChunk = g_forceChunk ||
                        (force && force[0] && force[0] != '0');
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

                if (g_chunkRows == 0 && g_autoChunk && !fitsInBudget) {
                    uint64_t streamRows = 0, streamFsz = 0;
                    if (peekColbinHeader(g_dataset_path + autoStreamTable + ".colbin",
                                        streamRows, streamFsz) && streamRows > 0) {
                        const std::set<std::string>& streamCols = tableCols.at(autoStreamTable);
                        uint64_t streamBytesPerRow =
                            projectedBytesPerRow(autoStreamTable, streamCols);
                        if (streamBytesPerRow == 0) streamBytesPerRow = 1;
                        const int slots = g_chunkDoubleBuffer ? 2 : 1;

                        const uint64_t chunkBudget =
                            (uint64_t)((double)totalBudget * kBudgetFraction);
                        auto workingSetForChunkRows = [&](uint64_t rows) {
                            const uint64_t streamInputBytes = saturatingMul(
                                saturatingMul(rows, streamBytesPerRow),
                                (uint64_t)slots);
                            DeviceBufferEstimate deviceBuffers =
                                estimateForStreamRows(rows);
                            return saturatingAdd(
                                saturatingAdd(residentBytes, streamInputBytes),
                                deviceBuffers.totalBytes);
                        };

                        uint64_t lo = 1, hi = streamRows, best = 0;
                        while (lo <= hi) {
                            uint64_t mid = lo + (hi - lo) / 2;
                            if (workingSetForChunkRows(mid) <= chunkBudget) {
                                best = mid;
                                lo = mid + 1;
                            } else {
                                if (mid == 0) break;
                                hi = mid - 1;
                            }
                        }
                        size_t autoChunkRows = best > 0 ? (size_t)best : 1;
                        g_chunkRows = autoChunkRows;
                        if (!g_csv) {
                            DeviceBufferEstimate chunkDeviceBuffers =
                                estimateForStreamRows(g_chunkRows);
                            printf("[auto-chunk] %s: disk=%.1f GiB working-set=%.1f GiB"
                                   " (resident=%.1f + stream=%.1f + device=%.1f)"
                                   " budget=%.1f GiB (avail=%.1f phys=%.1f GPU=%.1f)"
                                   " stream=%s bytes/row=%llu slots=%d"
                                   " — chunk=%zu rows (%.0f MiB/slot,"
                                   " chunk-working-set=%.1f GiB,"
                                   " chunk-device=%.1f GiB",
                                   plan.name.c_str(),
                                   totalDataBytes / 1e9,
                                   projectedWorkingSet / 1e9,
                                   residentBytes / 1e9,
                                   streamProjectedBytes / 1e9,
                                   fullDeviceBuffers.totalBytes / 1e9,
                                   totalBudget * kBudgetFraction / 1e9,
                                   availMemBytes / 1e9, physMemBytes / 1e9,
                                   gpuBudgetBytes / 1e9,
                                   autoStreamTable.c_str(),
                                   (unsigned long long)streamBytesPerRow,
                                   slots, g_chunkRows,
                                   (double)(g_chunkRows * streamBytesPerRow) / (1ull << 20),
                                   workingSetForChunkRows(g_chunkRows) / 1e9,
                                   chunkDeviceBuffers.totalBytes / 1e9);
                            if (!fullDeviceBuffers.largestName.empty()) {
                                printf(", largest-device=%s %.1f GiB",
                                       fullDeviceBuffers.largestName.c_str(),
                                       fullDeviceBuffers.largestBytes / 1e9);
                            }
                            if (fullDeviceBuffers.unresolvedBuffers > 0 ||
                                chunkDeviceBuffers.unresolvedBuffers > 0) {
                                printf(", unresolved-device-buffers=%zu",
                                       std::max(fullDeviceBuffers.unresolvedBuffers,
                                                chunkDeviceBuffers.unresolvedBuffers));
                            }
                            printf(")\n");
                        }
                    }
                }
            }
        }


        auto parseStart = std::chrono::high_resolution_clock::now();
        codegen::MetalGenericExecutor executor(device, cmdQueue);
        executor.setDetailedPhaseTiming(g_profilePhases || g_autotuneTgPerPhase);

        const std::string streamTable = (g_chunkRows > 0)
            ? autoDetectStreamTable(tableCols) : std::string{};
        bool didChunk = false;

        // Stream table columns are loaded per chunk below.
        double ioMs = 0.0;
        double preprocessMs = 0.0;
        std::vector<std::pair<std::string, QueryColumns>> loadedTables;
        for (auto& [tableName, colNames] : tableCols) {
            if (!streamTable.empty() && tableName == streamTable) continue;
            const auto& tdef = schema.table(tableName);
            std::vector<ColSpec> specs;
            for (const auto& colName : colNames)
                specs.push_back(codegen::colSpecFor(tdef.col(colName)));
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
            codegen::registerMaxKeySymbols(executor, loadedTables, tableCols, schema);
            if (!streamTable.empty() && tableCols.count(streamTable)) {
                codegen::extendMaxKeysFromStreamColbin(
                    executor,
                    g_dataset_path + streamTable + ".tbl",
                    tableCols.at(streamTable),
                    schema,
                    streamTable);
            }
            codegen::resetQueryPreprocessingState();
            if (isPredefinedTpchRoute &&
                !codegen::prepareQueryPreprocessing(plan.name, device, executor, loadedTables)) {
                return false;
            }
            preprocessMs += elapsedMs(_ppStart, clk::now());
        }

        codegen::MetalExecutionResult result;

        // Chunked path: kernels accumulate into output buffers across chunks.
        if (!streamTable.empty()) {
            std::vector<ColSpec> streamSpecs;
            for (const auto& colName : tableCols.at(streamTable))
                streamSpecs.push_back(codegen::colSpecFor(schema.table(streamTable).col(colName)));
            const int streamSlots = g_chunkDoubleBuffer ? 2 : 1;
            codegen::ChunkedColbinTable stream;
            std::string streamError;
            auto streamOpenStart = clk::now();
            if (!stream.open(device, g_dataset_path + streamTable + ".tbl",
                             streamSpecs, g_chunkRows, streamSlots, streamError)) {
                std::cerr << "Codegen: chunk open failed for " << streamTable
                          << ": " << streamError << std::endl;
                return false;
            }
            double streamOpenMs = elapsedMs(streamOpenStart, clk::now());
            const auto& streamTdef = schema.table(streamTable);
            const size_t totalRows = stream.rows(), chunkRows = stream.chunkRows();
            size_t chunkCount = 0;
            double chunkCopyMs = 0.0, gpuMs = 0.0, bufAllocMs = 0.0;
            double hookCpuMs = 0.0, hookGpuMs = 0.0;
            double resultCollectMs = 0.0, executeWallMs = 0.0;
            std::map<std::string, double> chunkPhaseSums;
            auto addExecutionTiming = [&](const codegen::MetalExecutionResult& rr) {
                gpuMs += rr.totalKernelTimeMs;
                bufAllocMs += rr.bufferAllocTimeMs;
                hookCpuMs += rr.hookCpuTimeMs;
                hookGpuMs += rr.hookGpuTimeMs;
                resultCollectMs += rr.resultCollectTimeMs;
                executeWallMs += rr.executeWallTimeMs;
                for (size_t i = 0; i < rr.phaseTimesMs.size(); ++i) {
                    const std::string nm = (i < rr.phaseNames.size())
                        ? rr.phaseNames[i] : ("phase" + std::to_string(i));
                    chunkPhaseSums[nm] += rr.phaseTimesMs[i];
                }
            };

            // Split phases into pre-stream, stream, and post-stream ranges.
            const auto& cgPhases = cg.getPhases();
            const int totalPhases = (int)cgPhases.size();
            int firstStreamPhase = totalPhases, lastStreamPhase = 0;
            for (int _pi = 0; _pi < totalPhases; _pi++) {
                if (cgPhases[_pi].scannedTable == streamTable) {
                    if (firstStreamPhase == totalPhases) firstStreamPhase = _pi;
                    lastStreamPhase = _pi + 1;
                }
            }
            // If no phase scans streamTable, run all phases per chunk.
            if (firstStreamPhase == totalPhases) { firstStreamPhase = 0; lastStreamPhase = totalPhases; }

            // Run pre-stream phases once.
            if (firstStreamPhase > 0) {
                auto preResult = executor.execute(compiled, cg, 0, 1, 0, firstStreamPhase);
                addExecutionTiming(preResult);
            }

            // Stream phases run once per chunk.
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
                addExecutionTiming(chunkResult);
                chunkCount++;
            }
            executor.setSkipZeroInit(false);

            // Run post-stream phases once.
            if (lastStreamPhase < totalPhases) {
                auto postResult = executor.execute(compiled, cg, 0, 1,
                                                   lastStreamPhase, totalPhases);
                addExecutionTiming(postResult);
            }

            auto finalCollectStart = clk::now();
            result.result = executor.collectResult(cg);
            resultCollectMs += elapsedMs(finalCollectStart, clk::now());
            timing.dataLoadMs    = ioMs + streamOpenMs + chunkCopyMs + preprocessMs;
            timing.ingestMs      = loadStats().excludedMs;
            timing.loadSource    = "chunked-colbin";
            timing.loadBytes     = loadStats().bytes + stream.bytesLoaded();
            timing.bufferAllocMs = bufAllocMs;
            // Chunked I/O: pre-stream load(ioMs) + per-chunk loadChunk(chunkCopyMs).
            timing.ioMs          = ioMs + streamOpenMs + chunkCopyMs;
            timing.preprocessMs  = preprocessMs;
            timing.gpuTotalMs    = gpuMs;
            timing.hookCpuMs     = hookCpuMs;
            timing.hookGpuMs     = hookGpuMs;
            timing.resultCollectMs = resultCollectMs;
            timing.executeWallMs = executeWallMs;
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
        // Execute with warmup/repeat and optional pipeline-cache bypass.
        auto parseEnd = std::chrono::high_resolution_clock::now();
        double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
        // One-time .tbl->column ingest (only when .colbin is missing) is
        // reported separately via timing.ingestMs and excluded from end-to-end timing.
        const double ingestMs = loadStats().excludedMs;
        timing.dataLoadMs = parseMs - ingestMs;
        if (timing.dataLoadMs < 0.0) timing.dataLoadMs = 0.0;
        timing.ingestMs   = ingestMs;
        timing.loadSource = loadStats().source();
        timing.loadBytes  = loadStats().bytes;
        // Split data-load window into pure I/O vs CPU preprocess. Anything
        // unaccounted for in the window is attributed to preprocess.
        const double pureIoMs = std::max(0.0, ioMs - ingestMs);
        timing.ioMs         = pureIoMs;
        timing.preprocessMs = (timing.dataLoadMs > pureIoMs)
                              ? (timing.dataLoadMs - pureIoMs) : preprocessMs;

        // Sweep threadgroup sizes, pick the best median, then continue timing.
        // The PSO is shared across candidates: tg_size
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
                    size_t mid = v.size() / 2;
                    if (v.size() % 2 == 1) return v[mid];
                    return 0.5 * (v[mid - 1] + v[mid]);
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

        // Measured loop. Each trial captures GPU time, execute wall time,
        // host-side hook/result work, and optionally JIT compile time.
        std::vector<double> gpuTrials;     gpuTrials.reserve(g_repeat);
        std::vector<double> compileTrials; compileTrials.reserve(g_repeat);
        std::vector<double> executeWallTrials; executeWallTrials.reserve(g_repeat);
        std::vector<double> bufferAllocTrials; bufferAllocTrials.reserve(g_repeat);
        std::vector<double> hookCpuTrials; hookCpuTrials.reserve(g_repeat);
        std::vector<double> hookGpuTrials; hookGpuTrials.reserve(g_repeat);
        std::vector<double> resultCollectTrials; resultCollectTrials.reserve(g_repeat);
        std::vector<std::string> phaseNamesForSummary;
        std::vector<std::vector<double>> phaseTrialSamples;
        std::vector<std::vector<double>> phaseWallTrialSamples;
        std::vector<std::vector<double>> phaseOverheadTrialSamples;
        std::vector<std::vector<double>> phaseHookCpuTrialSamples;
        std::vector<std::vector<double>> phaseHookGpuTrialSamples;

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
            double executeWallTrialMs = elapsedMs(tr0, clk::now());

            gpuTrials.push_back((double)result.totalKernelTimeMs);
            compileTrials.push_back(trialCompileMs);
            executeWallTrials.push_back(executeWallTrialMs);
            bufferAllocTrials.push_back((double)result.bufferAllocTimeMs);
            hookCpuTrials.push_back((double)result.hookCpuTimeMs);
            hookGpuTrials.push_back((double)result.hookGpuTimeMs);
            resultCollectTrials.push_back((double)result.resultCollectTimeMs);

            if (phaseTrialSamples.size() < result.phaseTimesMs.size()) {
                size_t oldSize = phaseTrialSamples.size();
                phaseTrialSamples.resize(result.phaseTimesMs.size());
                phaseWallTrialSamples.resize(result.phaseTimesMs.size());
                phaseOverheadTrialSamples.resize(result.phaseTimesMs.size());
                phaseHookCpuTrialSamples.resize(result.phaseTimesMs.size());
                phaseHookGpuTrialSamples.resize(result.phaseTimesMs.size());
                phaseNamesForSummary.resize(result.phaseTimesMs.size());
                for (size_t i = oldSize; i < result.phaseTimesMs.size(); i++) {
                    phaseNamesForSummary[i] = (i < result.phaseNames.size())
                        ? result.phaseNames[i] : ("phase" + std::to_string(i));
                }
            }
            for (size_t i = 0; i < result.phaseTimesMs.size(); i++) {
                if (i < phaseNamesForSummary.size() && phaseNamesForSummary[i].empty()) {
                    phaseNamesForSummary[i] = (i < result.phaseNames.size())
                        ? result.phaseNames[i] : ("phase" + std::to_string(i));
                }
                phaseTrialSamples[i].push_back((double)result.phaseTimesMs[i]);
                phaseWallTrialSamples[i].push_back(
                    i < result.phaseWallTimesMs.size()
                        ? (double)result.phaseWallTimesMs[i] : 0.0);
                phaseOverheadTrialSamples[i].push_back(
                    i < result.phaseOverheadTimesMs.size()
                        ? (double)result.phaseOverheadTimesMs[i] : 0.0);
                phaseHookCpuTrialSamples[i].push_back(
                    i < result.phaseHookCpuTimesMs.size()
                        ? (double)result.phaseHookCpuTimesMs[i] : 0.0);
                phaseHookGpuTrialSamples[i].push_back(
                    i < result.phaseHookGpuTimesMs.size()
                        ? (double)result.phaseHookGpuTimesMs[i] : 0.0);
            }

            if (g_csv && g_repeat > 1) {
                printf("TRIAL_CSV,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                       timing.scaleFactor.c_str(),
                       timing.queryName.c_str(),
                       timing.route.c_str(),
                       r,
                       (double)result.totalKernelTimeMs,
                       trialCompileMs,
                       executeWallTrialMs,
                       (double)result.bufferAllocTimeMs,
                       (double)result.hookCpuTimeMs,
                       (double)result.hookGpuTimeMs,
                       (double)result.resultCollectTimeMs);
                // Per-phase profiling rows are emitted when detailed phase
                // timing is enabled; normal e2e runs may batch phases.
                for (size_t pi = 0; pi < result.phaseTimesMs.size(); pi++) {
                    const std::string& nm = (pi < result.phaseNames.size())
                        ? result.phaseNames[pi] : "phase";
                    int tgUsed = (pi < cg.getPhases().size())
                        ? cg.getPhases()[pi].threadgroupSize : 0;
                    double phaseWall = (pi < result.phaseWallTimesMs.size())
                        ? (double)result.phaseWallTimesMs[pi] : 0.0;
                    double phaseOverhead = (pi < result.phaseOverheadTimesMs.size())
                        ? (double)result.phaseOverheadTimesMs[pi] : 0.0;
                    double phaseHookCpu = (pi < result.phaseHookCpuTimesMs.size())
                        ? (double)result.phaseHookCpuTimesMs[pi] : 0.0;
                    double phaseHookGpu = (pi < result.phaseHookGpuTimesMs.size())
                        ? (double)result.phaseHookGpuTimesMs[pi] : 0.0;
                    printf("PHASE_CSV,%s,%s,%s,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                           timing.scaleFactor.c_str(),
                           timing.queryName.c_str(),
                           timing.route.c_str(),
                           r, nm.c_str(), tgUsed,
                           (double)result.phaseTimesMs[pi],
                           phaseWall, phaseOverhead,
                           phaseHookCpu, phaseHookGpu);
                }
            }
        }

        timing.bufferAllocMs = medianValue(bufferAllocTrials);
        timing.gpuTotalMs    = medianValue(gpuTrials);
        timing.hookCpuMs     = medianValue(hookCpuTrials);
        timing.hookGpuMs     = medianValue(hookGpuTrials);
        timing.resultCollectMs = medianValue(resultCollectTrials);
        timing.executeWallMs = medianValue(executeWallTrials);
        timing.gpuTrialsN    = (int)gpuTrials.size();
        timing.gpuMsP10      = percentileValue(gpuTrials, 0.10);
        timing.gpuMsP90      = percentileValue(gpuTrials, 0.90);
        timing.gpuMsMad      = medianAbsoluteDeviation(gpuTrials);
        if (g_noPipelineCache) {
            // Override the single-shot compile time with the per-trial median
            // so the headline number reflects the cost we're studying.
            timing.compileMs = medianValue(compileTrials);
        }
        timing.phaseKernelMs.clear();
        for (size_t i = 0; i < phaseTrialSamples.size(); i++) {
            const std::string name = (i < phaseNamesForSummary.size() &&
                                      !phaseNamesForSummary[i].empty())
                ? phaseNamesForSummary[i] : ("phase" + std::to_string(i));
            timing.phaseKernelMs.emplace_back(name, medianValue(phaseTrialSamples[i]));
        }
        } // end if (!didChunk)

        HostPostOpTracker hostPostOps;
        auto runHostPost = [&](codegen::MetalExecutionResult& result,
                               bool emitOutput,
                               HostPostOpTracker* hostPostOpsForRun) -> double {
        auto markHostPost = [&](const std::string& op) {
            if (isPredefinedTpchRoute && hostPostOpsForRun) {
                hostPostOpsForRun->mark(op);
            }
        };
        auto isPredefinedPlan = [&](const char* name) {
            return isPredefinedTpchRoute && plan.name == name;
        };

        auto postStart = std::chrono::high_resolution_clock::now();

        // Per-query output normalization.

        // Q14: convert raw promo/total sums to the final ratio column.
        if (isPredefinedPlan("Q14") && result.result.numRows() == 1 &&
            result.result.columns.size() == 2) {
            double promo = std::get<double>(result.result.rows[0][0]);
            double total = std::get<double>(result.result.rows[0][1]);
            double ratio = (total > 0) ? (100.0 * promo / total) : 0.0;
            result.result.columns = {{"promo_revenue", "float"}};
            result.result.rows[0] = {ratio};
        }

        // Q12: pivot four GPU buckets into the two TPC-H output rows.
        if (isPredefinedPlan("Q12") && result.result.numRows() == 4 &&
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

        // Apply GPU sort order to materialized rows.
        if (plan.gpuSort && !result.result.columns.empty()) {
            applyGpuSortRemap(result.result, *plan.gpuSort, executor);
        }

        // Empty columns mean a query-specific block below will assemble output.
        if (emitOutput && !result.result.columns.empty()) {
            printf("\n%s Results:\n", queryName.c_str());
            result.result.print();
        }

        // Per-query post-processing runs before the golden check.

        // Q10: gather GPU-sorted top-k when available.
        if (isPredefinedPlan("Q10") && result.result.columns.empty()) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q10_compact_count");
            auto* ckBuf  = executor.getAllocatedBuffer("d_q10_compact_ck");
            auto* revBuf = executor.getAllocatedBuffer("d_q10_compact_rev");
            if (cntBuf && ckBuf && revBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = ckBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const uint32_t* cks  = (const uint32_t*)ckBuf->contents();
                const float*    revs = (const float*)revBuf->contents();

                std::vector<std::pair<float, int>> entries;
                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int limit = plan.gpuSort->limit >= 0 ? plan.gpuSort->limit : (int)n;
                        int show = std::min((int)n, limit);
                        entries.reserve(show);
                        for (int j = 0; j < show; j++) {
                            int src = order[j];
                            if (src < 0 || (uint32_t)src >= n) continue;
                            entries.push_back({revs[src], (int)cks[src]});
                        }
                    }
                }

                if (entries.empty()) {
                    markHostPost("hostTopKSort");
                    entries.reserve(n);
                    for (uint32_t k = 0; k < n; k++)
                        entries.push_back({revs[k], (int)cks[k]});
                    int show = std::min((int)entries.size(), 20);
                    if (show < (int)entries.size()) {
                        std::partial_sort(entries.begin(), entries.begin() + show,
                                          entries.end(),
                                          [](auto& a, auto& b) { return a.first > b.first; });
                    } else {
                        std::sort(entries.begin(), entries.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
                    }
                    entries.resize(show);
                } else {
                    // Already gathered by GPU order.
                }
                result.result.columns = {{"c_custkey","int"},{"revenue","float"}};
                result.result.rows.clear();
                int show = (int)entries.size();
                for (int j = 0; j < show; j++)
                    result.result.rows.push_back({(int64_t)entries[j].second, (double)entries[j].first});
                if (emitOutput) {
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
        }

        // Q7: print 4 revenue bins
        if (isPredefinedPlan("Q7")) {
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
                if (emitOutput) {
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
        }

        // Q5: nation revenue sorted desc
        if (isPredefinedPlan("Q5")) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q5_result_count");
            auto* nameBuf = executor.getAllocatedBuffer("d_q5_result_name");
            auto* revBuf = executor.getAllocatedBuffer("d_q5_result_revenue");
            if (cntBuf && nameBuf && revBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = revBuf->length() / sizeof(float);
                if (n > cap) n = (uint32_t)cap;
                const char* names = (const char*)nameBuf->contents();
                const float* revenues = (const float*)revBuf->contents();
                result.result.columns = {{"n_name","string"},{"revenue","float"}};
                result.result.rows.clear();
                auto extractName = [](const char* base) {
                    int len = 25;
                    while (len > 0 && (base[len - 1] == ' ' || base[len - 1] == '\0')) len--;
                    return std::string(base, len);
                };
                auto appendRow = [&](uint32_t src) {
                    if (src >= n) return;
                    result.result.rows.push_back({
                        extractName(names + (size_t)src * 25),
                        (double)revenues[src]
                    });
                };

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        for (uint32_t j = 0; j < n; j++) {
                            int src = order[j];
                            if (src >= 0) appendRow((uint32_t)src);
                        }
                    }
                }

                if (result.result.rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    std::vector<uint32_t> order(n);
                    for (uint32_t i = 0; i < n; i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        return revenues[a] > revenues[b];
                    });
                    for (uint32_t src : order) appendRow(src);
                }

                if (emitOutput) {
                    printf("  +------------------+-----------------+\n");
                    printf("  | n_name           |         revenue |\n");
                    printf("  +------------------+-----------------+\n");
                    for (const auto& row : result.result.rows) {
                        printf("  | %-16s | $%14.2f |\n",
                               std::get<std::string>(row[0]).c_str(),
                               std::get<double>(row[1]));
                    }
                    printf("  +------------------+-----------------+\n");
                }
            }
        }

        // Q8: market share = brazil / total per year
        if (isPredefinedPlan("Q8")) {
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
                if (emitOutput) {
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
        }

        // Q3: gather compacted rows, using GPU order when available.
        if (isPredefinedPlan("Q3")) {
            auto* cntBuf  = executor.getAllocatedBuffer("d_q3_compact_count");
            auto* okBuf   = executor.getAllocatedBuffer("d_q3_compact_ok");
            auto* revBuf  = executor.getAllocatedBuffer("d_q3_compact_rev");
            auto* dateBuf = executor.getAllocatedBuffer("d_q3_compact_date");
            auto* prioBuf = executor.getAllocatedBuffer("d_q3_compact_prio");
            if (cntBuf && okBuf && revBuf && dateBuf && prioBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = okBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const uint32_t* oks  = (const uint32_t*)okBuf->contents();
                const float*    revs = (const float*)revBuf->contents();
                const int* dates = (const int*)dateBuf->contents();
                const int* prios = (const int*)prioBuf->contents();
                std::vector<std::tuple<float, int, int, int>> entries;

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int limit = plan.gpuSort->limit >= 0 ? plan.gpuSort->limit : (int)n;
                        int show = std::min((int)n, limit);
                        entries.reserve(show);
                        for (int j = 0; j < show; j++) {
                            int src = order[j];
                            if (src < 0 || (uint32_t)src >= n) continue;
                            entries.push_back({revs[src], dates[src], (int)oks[src], prios[src]});
                        }
                    }
                }

                if (entries.empty()) {
                    markHostPost("hostTopKSort");
                    entries.reserve(n);
                    for (uint32_t k = 0; k < n; k++) {
                        entries.push_back({revs[k], dates[k], (int)oks[k], prios[k]});
                    }
                    int show = std::min((int)entries.size(), 10);
                    auto cmp = [](auto& a, auto& b) {
                        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
                        return std::get<1>(a) < std::get<1>(b);
                    };
                    if (show < (int)entries.size()) {
                        std::partial_sort(entries.begin(), entries.begin() + show,
                                          entries.end(), cmp);
                    } else {
                        std::sort(entries.begin(), entries.end(), cmp);
                    }
                    entries.resize(show);
                }
                result.result.columns = {{"l_orderkey","int"},{"revenue","float"},{"o_orderdate","string"},{"o_shippriority","int"}};
                result.result.rows.clear();
                int show = (int)entries.size();
                for (int j = 0; j < show; j++) {
                    auto& [r, d, ok, p] = entries[j];
                    result.result.rows.push_back({(int64_t)ok, (double)r, intDateToStr(d), (int64_t)p});
                }
                if (emitOutput) {
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
        }

        // Q13: histogram of order counts
        if (isPredefinedPlan("Q13")) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q13_result_count");
            auto* cCountBuf = executor.getAllocatedBuffer("d_q13_result_c_count");
            auto* custDistBuf = executor.getAllocatedBuffer("d_q13_result_custdist");
            if (cntBuf && cCountBuf && custDistBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = cCountBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const uint32_t* cCounts = (const uint32_t*)cCountBuf->contents();
                const uint32_t* custDists = (const uint32_t*)custDistBuf->contents();
                std::vector<std::pair<uint32_t, int>> entries;

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        entries.reserve(n);
                        for (uint32_t j = 0; j < n; j++) {
                            int src = order[j];
                            if (src < 0 || (uint32_t)src >= n) continue;
                            entries.push_back({custDists[src], (int)cCounts[src]});
                        }
                    }
                }

                if (entries.empty() && n > 0) {
                    markHostPost("hostSort");
                    entries.reserve(n);
                    for (uint32_t k = 0; k < n; k++) {
                        entries.push_back({custDists[k], (int)cCounts[k]});
                    }
                    std::sort(entries.begin(), entries.end(),
                        [](auto& a, auto& b) {
                            if (a.first != b.first) return a.first > b.first;
                            return a.second > b.second;
                        });
                }

                result.result.columns = {{"c_count","int"},{"custdist","int"}};
                result.result.rows.clear();
                for (auto& [dist, cnt] : entries)
                    result.result.rows.push_back({(int64_t)cnt, (int64_t)dist});
                if (emitOutput) {
                    printf("  +--------+----------+\n");
                    printf("  | c_count|  custdist|\n");
                    printf("  +--------+----------+\n");
                    for (auto& [dist, cnt] : entries) {
                        printf("  | %6d | %8u |\n", cnt, dist);
                    }
                    printf("  +--------+----------+\n");
                }
            }
        }

        // Q22: 7 country-code bins
        if (isPredefinedPlan("Q22")) {
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
                if (emitOutput) {
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
        }

        // Q11 host path: threshold, filter, and sort.
        if (isPredefinedPlan("Q11") && result.result.columns.empty()) {
            auto* valBuf = executor.getAllocatedBuffer("d_part_value");
            if (valBuf) {
                markHostPost("hostScalarScan");
                markHostPost("hostFilter");
                markHostPost("hostSort");
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
                if (emitOutput) {
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
        }

        // Q15: find max revenue supplier
        if (isPredefinedPlan("Q15") && result.result.columns.empty()) {
            auto* revBuf = executor.getAllocatedBuffer("d_supp_revenue");
            if (revBuf) {
                markHostPost("hostScalarScan");
                markHostPost("hostFilter");
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
                if (emitOutput) {
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
        }

        // Q18: gather compacted rows, using GPU top-k order when available.
        if (isPredefinedPlan("Q18")) {
            auto* cntBuf     = executor.getAllocatedBuffer("d_q18_compact_count");
            auto* okBuf      = executor.getAllocatedBuffer("d_q18_compact_ok");
            auto* custBuf    = executor.getAllocatedBuffer("d_q18_compact_custkey");
            auto* totalBuf   = executor.getAllocatedBuffer("d_q18_compact_totalprice");
            auto* dateBuf    = executor.getAllocatedBuffer("d_q18_compact_orderdate");
            auto* qtyBuf     = executor.getAllocatedBuffer("d_q18_compact_qty");
            if (cntBuf && okBuf && custBuf && totalBuf && dateBuf && qtyBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                // Match the kernel's compact-buffer cap.
                size_t cap = okBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const uint32_t* oks       = (const uint32_t*)okBuf->contents();
                const int*      custkeys  = (const int*)custBuf->contents();
                const float*    totals    = (const float*)totalBuf->contents();
                const int*      dates     = (const int*)dateBuf->contents();
                const float*    qtys      = (const float*)qtyBuf->contents();

                struct Q18Entry { int orderkey; int custkey; float totalprice; int orderdate; float qty; };
                std::vector<Q18Entry> results;
                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int limit = plan.gpuSort->limit >= 0 ? plan.gpuSort->limit : (int)n;
                        int show = std::min((int)n, limit);
                        results.reserve(show);
                        for (int j = 0; j < show; j++) {
                            int src = order[j];
                            if (src < 0 || (uint32_t)src >= n) continue;
                            results.push_back({(int)oks[src], custkeys[src], totals[src],
                                               dates[src], qtys[src]});
                        }
                    }
                }

                if (results.empty() && n > 0) {
                    markHostPost("hostTopKSort");
                    results.reserve(n);
                    for (uint32_t k = 0; k < n; k++) {
                        results.push_back({(int)oks[k], custkeys[k], totals[k], dates[k], qtys[k]});
                    }
                    int show = std::min((int)results.size(), 100);
                    auto cmp = [](const Q18Entry& a, const Q18Entry& b) {
                        if (a.totalprice != b.totalprice) return a.totalprice > b.totalprice;
                        return a.orderdate < b.orderdate;
                    };
                    if (show < (int)results.size()) {
                        std::partial_sort(results.begin(), results.begin() + show,
                                          results.end(), cmp);
                    } else {
                        std::sort(results.begin(), results.end(), cmp);
                    }
                    results.resize(show);
                }

                int show = (int)results.size();
                result.result.columns = {{"c_custkey","int"},{"o_orderkey","int"},{"o_orderdate","string"},{"o_totalprice","float"},{"sum(l_quantity)","float"}};
                result.result.rows.clear();
                for (int j = 0; j < show; j++) {
                    auto& r = results[j];
                    result.result.rows.push_back({(int64_t)r.custkey, (int64_t)r.orderkey, intDateToStr(r.orderdate), (double)r.totalprice, (double)r.qty});
                }
                if (emitOutput) {
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
        }

        // Q9: sort profit bins by nation ASC, year DESC.
        if (isPredefinedPlan("Q9")) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q9_result_count");
            auto* nationBuf = executor.getAllocatedBuffer("d_q9_result_nation");
            auto* yearBuf = executor.getAllocatedBuffer("d_q9_result_year");
            auto* profitBuf = executor.getAllocatedBuffer("d_q9_result_profit");
            if (cntBuf && nationBuf && yearBuf && profitBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = yearBuf->length() / sizeof(int32_t);
                if (n > cap) n = (uint32_t)cap;
                const char* nations = (const char*)nationBuf->contents();
                const int32_t* years = (const int32_t*)yearBuf->contents();
                const float* profits = (const float*)profitBuf->contents();
                result.result.columns = {{"nation","string"},{"o_year","int"},{"sum_profit","float"}};
                result.result.rows.clear();
                auto extractNation = [](const char* base) {
                    int len = 25;
                    while (len > 0 && (base[len - 1] == ' ' || base[len - 1] == '\0')) len--;
                    return std::string(base, len);
                };
                auto appendRow = [&](uint32_t src) {
                    if (src >= n) return;
                    result.result.rows.push_back({
                        extractNation(nations + (size_t)src * 25),
                        (int64_t)years[src],
                        (double)profits[src]
                    });
                };

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        for (uint32_t j = 0; j < n; j++) {
                            int src = order[j];
                            if (src >= 0) appendRow((uint32_t)src);
                        }
                    }
                }

                if (result.result.rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    std::vector<uint32_t> order(n);
                    for (uint32_t i = 0; i < n; i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        int cmp = strncmp(nations + (size_t)a * 25,
                                          nations + (size_t)b * 25, 25);
                        if (cmp != 0) return cmp < 0;
                        return years[a] > years[b];
                    });
                    for (uint32_t src : order) appendRow(src);
                }

                if (emitOutput) {
                    printf("  +------------+------+---------------+\n");
                    printf("  | Nation     | Year |        Profit |\n");
                    printf("  +------------+------+---------------+\n");
                    int show = std::min((int)result.result.rows.size(), 15);
                    for (int j = 0; j < show; j++) {
                        const auto& row = result.result.rows[j];
                        printf("  | %-10s | %4lld | $%13.2f |\n",
                               std::get<std::string>(row[0]).c_str(),
                               (long long)std::get<int64_t>(row[1]),
                               std::get<double>(row[2]));
                    }
                    printf("  +------------+------+---------------+\n");
                    printf("  Total results: %zu\n", result.result.rows.size());
                }
            }
        }

        // Q20: gather GPU-materialized supplier rows in GPU sort order.
        if (isPredefinedPlan("Q20")) {
            auto* cntBuf  = executor.getAllocatedBuffer("d_q20_result_count");
            auto* nameBuf = executor.getAllocatedBuffer("d_q20_result_name");
            auto* addrBuf = executor.getAllocatedBuffer("d_q20_result_address");
            if (cntBuf && nameBuf && addrBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = nameBuf->length() / 25;
                if (n > cap) n = (uint32_t)cap;
                const char* names = (const char*)nameBuf->contents();
                const char* addresses = (const char*)addrBuf->contents();
                struct Q20Row { std::string name; std::string address; };
                std::vector<Q20Row> rows;
                auto extractFixedStr = [](const char* base, int width, bool trimSpaces) {
                    int len = 0;
                    while (len < width && base[len] != '\0') len++;
                    if (trimSpaces) {
                        while (len > 0 && base[len - 1] == ' ') len--;
                    }
                    return std::string(base, len);
                };

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int show = (int)n;
                        rows.reserve(show);
                        for (int j = 0; j < show; j++) {
                            int src = order[j];
                            if (src < 0 || (uint32_t)src >= n) continue;
                            rows.push_back({
                                extractFixedStr(names + (size_t)src * 25, 25, true),
                                extractFixedStr(addresses + (size_t)src * 40, 40, true)
                            });
                        }
                    }
                }

                if (rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    rows.reserve(n);
                    for (uint32_t k = 0; k < n; k++) {
                        rows.push_back({
                            extractFixedStr(names + (size_t)k * 25, 25, true),
                            extractFixedStr(addresses + (size_t)k * 40, 40, true)
                        });
                    }
                    std::sort(rows.begin(), rows.end(), [](const Q20Row& a, const Q20Row& b) {
                        return a.name < b.name;
                    });
                }

                result.result.columns = {{"s_name","string"},{"s_address","string"}};
                result.result.rows.clear();
                for (auto& r : rows)
                    result.result.rows.push_back({r.name, r.address});
                if (emitOutput) {
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
        }

        // Q2: gather GPU-decorated and GPU-sorted compact rows.
        if (isPredefinedPlan("Q2")) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q2_compact_count");
            auto* acctBuf = executor.getAllocatedBuffer("d_q2_result_acctbal");
            auto* nameBuf = executor.getAllocatedBuffer("d_q2_result_s_name");
            auto* nationBuf = executor.getAllocatedBuffer("d_q2_result_n_name");
            auto* partkeyBuf = executor.getAllocatedBuffer("d_q2_result_p_partkey");
            auto* mfgrBuf = executor.getAllocatedBuffer("d_q2_result_p_mfgr");
            auto* addrBuf = executor.getAllocatedBuffer("d_q2_result_s_address");
            auto* phoneBuf = executor.getAllocatedBuffer("d_q2_result_s_phone");
            auto* commentBuf = executor.getAllocatedBuffer("d_q2_result_s_comment");
            auto* lateCntBuf = executor.getAllocatedBuffer("d_q2_late_count");
            if (cntBuf && acctBuf && nameBuf && nationBuf && partkeyBuf &&
                mfgrBuf && addrBuf && phoneBuf && commentBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = partkeyBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const float* acct = (const float*)acctBuf->contents();
                const char* names = (const char*)nameBuf->contents();
                const char* nations = (const char*)nationBuf->contents();
                const uint32_t* partkeys = (const uint32_t*)partkeyBuf->contents();
                const char* mfgrs = (const char*)mfgrBuf->contents();
                const char* addresses = (const char*)addrBuf->contents();
                const char* phones = (const char*)phoneBuf->contents();
                const char* comments = (const char*)commentBuf->contents();
                result.result.columns = {{"s_acctbal","float"},{"s_name","string"},{"n_name","string"},{"p_partkey","int"},{"p_mfgr","string"},{"s_address","string"},{"s_phone","string"},{"s_comment","string"}};
                result.result.rows.clear();
                auto extractStr = [](const char* base, int width, bool trimSpaces) {
                    int len = 0;
                    while (len < width && base[len] != '\0') len++;
                    if (trimSpaces) {
                        while (len > 0 && base[len - 1] == ' ') len--;
                    }
                    return std::string(base, len);
                };

                auto appendRow = [&](uint32_t src) {
                    if (src >= n) return;
                    result.result.rows.push_back({
                        (double)acct[src],
                        extractStr(names + (size_t)src * 25, 25, true),
                        extractStr(nations + (size_t)src * 25, 25, true),
                        (int64_t)partkeys[src],
                        extractStr(mfgrs + (size_t)src * 25, 25, true),
                        extractStr(addresses + (size_t)src * 40, 40, true),
                        extractStr(phones + (size_t)src * 15, 15, false),
                        extractStr(comments + (size_t)src * 101, 101, false)
                    });
                };

                int limit = std::min((int)n, 100);
                if (lateCntBuf && !plan.gpuSort) {
                    uint32_t lateN = *static_cast<uint32_t*>(lateCntBuf->contents());
                    limit = std::min((int)lateN, 100);
                    for (int j = 0; j < limit; j++) appendRow((uint32_t)j);
                } else if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int gpuLimit = plan.gpuSort->limit >= 0 ? plan.gpuSort->limit : limit;
                        limit = std::min((int)n, gpuLimit);
                        for (int j = 0; j < limit; j++) {
                            int src = order[j];
                            if (src >= 0) appendRow((uint32_t)src);
                        }
                    }
                }

                if (result.result.rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    std::vector<uint32_t> order(n);
                    for (uint32_t i = 0; i < n; i++) order[i] = i;
                    auto fixedCmp = [&](const char* base, int width, uint32_t a, uint32_t b) {
                        return strncmp(base + (size_t)a * width,
                                       base + (size_t)b * width,
                                       (size_t)width);
                    };
                    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        if (acct[a] != acct[b]) return acct[a] > acct[b];
                        int cmp = fixedCmp(nations, 25, a, b);
                        if (cmp != 0) return cmp < 0;
                        cmp = fixedCmp(names, 25, a, b);
                        if (cmp != 0) return cmp < 0;
                        return partkeys[a] < partkeys[b];
                    });
                    limit = std::min((int)order.size(), 100);
                    for (int j = 0; j < limit; j++) appendRow(order[(size_t)j]);
                }

                if (emitOutput) {
                    printf("\nQ2 Results:\n");
                    printf("  %-10s | %-25s | %-15s | %-8s | %-25s\n",
                           "s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr");
                    printf("  ----------+---------------------------+-----------------+----------+---------------------------\n");
                    int show = std::min((int)result.result.rows.size(), 10);
                    for (int j = 0; j < show; j++) {
                        auto& r = result.result.rows[j];
                        printf("  %9.2f | %-25s | %-15s | %8lld | %-25s\n",
                               std::get<double>(r[0]), std::get<std::string>(r[1]).c_str(),
                               std::get<std::string>(r[2]).c_str(), (long long)std::get<int64_t>(r[3]),
                               std::get<std::string>(r[4]).c_str());
                    }
                    if ((int)result.result.rows.size() > 10)
                        printf("  ... (%zu more rows)\n", result.result.rows.size() - 10);
                    printf("  Total rows: %zu\n", result.result.rows.size());
                }
            }
        }

        // Q16: gather GPU-decorated and GPU-sorted group rows.
        if (isPredefinedPlan("Q16")) {
            auto* rowCntBuf = executor.getAllocatedBuffer("d_q16_result_count");
            auto* brandBuf = executor.getAllocatedBuffer("d_q16_result_brand");
            auto* typeBuf = executor.getAllocatedBuffer("d_q16_result_type");
            auto* sizeBuf = executor.getAllocatedBuffer("d_q16_result_size");
            auto* suppCntBuf = executor.getAllocatedBuffer("d_q16_result_supplier_cnt");
            if (rowCntBuf && brandBuf && typeBuf && sizeBuf && suppCntBuf) {
                uint32_t n = *static_cast<uint32_t*>(rowCntBuf->contents());
                size_t cap = sizeBuf->length() / sizeof(int32_t);
                if (n > cap) n = (uint32_t)cap;
                const char* brands = (const char*)brandBuf->contents();
                const char* types = (const char*)typeBuf->contents();
                const int32_t* sizes = (const int32_t*)sizeBuf->contents();
                const uint32_t* supplierCnts = (const uint32_t*)suppCntBuf->contents();
                result.result.columns = {{"p_brand","string"},{"p_type","string"},{"p_size","int"},{"supplier_cnt","int"}};
                result.result.rows.clear();
                auto fixedStr = [](const char* base, int width) {
                    int len = width;
                    while (len > 0 && (base[len - 1] == ' ' || base[len - 1] == '\0')) len--;
                    return std::string(base, len);
                };
                auto appendRow = [&](uint32_t src) {
                    if (src >= n) return;
                    result.result.rows.push_back({
                        fixedStr(brands + (size_t)src * 10, 10),
                        fixedStr(types + (size_t)src * 25, 25),
                        (int64_t)sizes[src],
                        (int64_t)supplierCnts[src]
                    });
                };

                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        for (uint32_t j = 0; j < n; j++) {
                            int src = order[j];
                            if (src >= 0) appendRow((uint32_t)src);
                        }
                    }
                }

                if (result.result.rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    std::vector<uint32_t> order(n);
                    for (uint32_t i = 0; i < n; i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        if (supplierCnts[a] != supplierCnts[b]) return supplierCnts[a] > supplierCnts[b];
                        int cmp = strncmp(brands + (size_t)a * 10,
                                          brands + (size_t)b * 10, 10);
                        if (cmp != 0) return cmp < 0;
                        cmp = strncmp(types + (size_t)a * 25,
                                      types + (size_t)b * 25, 25);
                        if (cmp != 0) return cmp < 0;
                        return sizes[a] < sizes[b];
                    });
                    for (uint32_t src : order) appendRow(src);
                }

                if (emitOutput) {
                    printf("\nQ16 Results:\n");
                    printf("  +-----------+---------------------------+------+--------------+\n");
                    printf("  | p_brand   | p_type                    |p_size| supplier_cnt |\n");
                    printf("  +-----------+---------------------------+------+--------------+\n");
                    int show = std::min((int)result.result.rows.size(), 10);
                    for (int j = 0; j < show; j++) {
                        const auto& row = result.result.rows[j];
                        printf("  | %-9s | %-25s | %4lld | %12lld |\n",
                               std::get<std::string>(row[0]).c_str(),
                               std::get<std::string>(row[1]).c_str(),
                               (long long)std::get<int64_t>(row[2]),
                               (long long)std::get<int64_t>(row[3]));
                    }
                    printf("  +-----------+---------------------------+------+--------------+\n");
                    printf("  Total groups: %zu\n", result.result.rows.size());
                }
            }
        }

        // Q21: gather GPU-decorated and GPU-sorted compact rows.
        if (isPredefinedPlan("Q21")) {
            auto* cntBuf = executor.getAllocatedBuffer("d_q21_result_count");
            auto* nameBuf = executor.getAllocatedBuffer("d_q21_result_name");
            auto* numwaitBuf = executor.getAllocatedBuffer("d_q21_result_numwait");
            if (cntBuf && nameBuf && numwaitBuf) {
                uint32_t n = *static_cast<uint32_t*>(cntBuf->contents());
                size_t cap = numwaitBuf->length() / sizeof(uint32_t);
                if (n > cap) n = (uint32_t)cap;
                const char* names = (const char*)nameBuf->contents();
                const uint32_t* numwait = (const uint32_t*)numwaitBuf->contents();

                result.result.columns = {{"s_name","string"},{"numwait","int"}};
                result.result.rows.clear();
                auto extractName = [](const char* base) {
                    int len = 25;
                    while (len > 0 && (base[len - 1] == ' ' || base[len - 1] == '\0')) len--;
                    return std::string(base, len);
                };
                auto appendRow = [&](uint32_t src) {
                    if (src >= n) return;
                    result.result.rows.push_back({
                        extractName(names + (size_t)src * 25),
                        (int64_t)numwait[src]
                    });
                };

                int limit = std::min((int)n, 100);
                if (plan.gpuSort) {
                    auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
                    if (idxBuf) {
                        const int* order = static_cast<const int*>(idxBuf->contents());
                        int gpuLimit = plan.gpuSort->limit >= 0 ? plan.gpuSort->limit : limit;
                        limit = std::min((int)n, gpuLimit);
                        for (int j = 0; j < limit; j++) {
                            int src = order[j];
                            if (src >= 0) appendRow((uint32_t)src);
                        }
                    }
                }

                if (result.result.rows.empty() && n > 0) {
                    markHostPost("hostSort");
                    std::vector<uint32_t> order(n);
                    for (uint32_t i = 0; i < n; i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        if (numwait[a] != numwait[b]) return numwait[a] > numwait[b];
                        return strncmp(names + (size_t)a * 25,
                                       names + (size_t)b * 25, 25) < 0;
                    });
                    limit = std::min((int)order.size(), 100);
                    for (int j = 0; j < limit; j++) appendRow(order[(size_t)j]);
                }

                if (emitOutput) {
                    printf("\nQ21 Results:\n");
                    printf("  +---------------------------+----------+\n");
                    printf("  | s_name                    | numwait  |\n");
                    printf("  +---------------------------+----------+\n");
                    int show = std::min((int)result.result.rows.size(), 10);
                    for (int j = 0; j < show; j++) {
                        const auto& row = result.result.rows[j];
                        printf("  | %-25s | %8lld |\n",
                               std::get<std::string>(row[0]).c_str(),
                               (long long)std::get<int64_t>(row[1]));
                    }
                    printf("  +---------------------------+----------+\n");
                    printf("  Total qualifying suppliers: %u\n", n);
                }
            }
        }

        return elapsedMs(postStart, clk::now());
        };

        std::vector<double> postTrials;
        postTrials.reserve((size_t)g_repeat);
        const codegen::MetalExecutionResult rawResult = result;
        for (int pr = 0; pr < g_repeat; pr++) {
            codegen::MetalExecutionResult postResult = rawResult;
            const bool emitOutput = !g_csv && pr == g_repeat - 1;
            HostPostOpTracker* tracker = (pr == g_repeat - 1) ? &hostPostOps : nullptr;
            postTrials.push_back(runHostPost(postResult, emitOutput, tracker));
            if (pr == g_repeat - 1) {
                result = std::move(postResult);
            }
        }
        timing.postMs = medianValue(postTrials);

        double validationMs = 0.0;

        // Golden check runs after query-specific output assembly, but it is
        // validation work rather than query execution time.
        if (!g_saveGoldenDir.empty() || !g_checkDir.empty()) {
            auto validationStart = clk::now();
            std::string canonical = result.result.toCanonical();
            std::string checkName = queryName;
            std::string fname = checkName + "_" + timing.scaleFactor + ".csv";

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
                            checkName.c_str(), path.c_str());
                    g_checkExitCode = 2;
                } else {
                    std::ostringstream buf;
                    buf << ifs.rdbuf();
                    std::string diff = compareCanonical(canonical, buf.str(),
                                                        g_checkAbsTol, g_checkRelTol);
                    if (diff.empty()) {
                        printf("[CHECK] %s @ %s: OK (%zu rows)\n",
                               checkName.c_str(), timing.scaleFactor.c_str(),
                               result.result.numRows());
                    } else {
                        fprintf(stderr, "[CHECK] %s @ %s: FAIL — %s\n",
                                checkName.c_str(), timing.scaleFactor.c_str(),
                                diff.c_str());
                        g_checkExitCode = 1;
                    }
                }
            }
            validationMs = elapsedMs(validationStart, clk::now());
        }

        timing.validationMs = validationMs;

        if (!hostPostOps.empty()) {
            printf("HOST_POST_CSV,%s,%s,%s\n",
                   timing.scaleFactor.c_str(), queryName.c_str(),
                   hostPostOps.joined().c_str());
        }

        printDetailedTimingSummary(timing, g_csv);

        executor.releaseAllocatedBuffers();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Codegen error (" << queryName << "): " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, const char* argv[]) {
    std::string query;
    std::string inlineSql;
    std::string sqlFile;
    bool hasInlineSql = false;
    bool hasSqlFile = false;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "help" || arg == "--help" || arg == "-h") {
            printf("GPU Database Codegen\n");
            printf("Usage: GPUDBCodegen [flags] [sf1|sf10|sf20|sf50|sf100] q<N>|mb<N>|--sql SQL|--sql-file FILE\n");
            printf("Predefined TPC-H API:\n");
            printf("  q1..q22       - Run predefined TPC-H query\n");
            printf("  all           - Run all 22 predefined TPC-H queries\n");
            printf("Ad-hoc SQL API:\n");
            printf("  --sql SQL     - Run supported-pattern SQL text through the analyzer route\n");
            printf("  --sql-file F  - Run supported-pattern SQL file through the analyzer route\n");
            printf("  mb1..mb7      - Run microbenchmark SQL through the analyzer route\n");
            printf("  mball         - Run all microbenchmarks\n");
            printf("Loader flags:\n");
            printf("  --no-zerocopy        Disable zero-copy mmap path (force buffer copies)\n");
            printf("  --no-binary          Disable .colbin binary loader (force .tbl parser)\n");
            printf("  --chunk N[K|M|G]     Stream certified chunkable plans from .colbin\n");
            printf("  --auto-chunk         Auto-enable chunking for certified chunkable plans (default)\n");
            printf("  --no-auto-chunk      Disable budget-triggered chunking; explicit --chunk still works\n");
            printf("  --force-chunk        Keep explicit --chunk even when the direct load fits budget\n");
            printf("  --no-db              With --chunk, use one reusable chunk slot instead of two\n");
            printf("Experiment flags:\n");
            printf("  --warmup N           Run N untimed warmup iterations (default 3)\n");
            printf("  --repeat N           Run N timed iterations, report median (default 1)\n");
            printf("  --csv                Suppress text breakdown; emit CSV timing rows\n");
            printf("  --threadgroup-size N Override default threadgroup size (default = plan-specified)\n");
            printf("  --autotune-tg        Per-query global TG sweep over {32,64,128,256,512,1024};\n");
            printf("                       picks the size with min p50 GPU time (logs AUTOTUNE_CSV)\n");
            printf("  --autotune-tg-per-phase  Per-phase TG sweep; picks min-p50 TG independently\n");
            printf("                       for each kernel (logs AUTOTUNE_PHASE_CSV)\n");
            printf("  --no-pipeline-cache  Recompile Metal source on every measured iteration\n");
            printf("  --profile-phases     Emit per-phase GPU, wall, residual, and hook timings\n");
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
        if (arg == "--auto-chunk")        { g_autoChunk = true; continue; }
        if (arg == "--no-auto-chunk")     { g_autoChunk = false; continue; }
        if (arg == "--force-chunk")       { g_forceChunk = true; continue; }
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
        if (arg == "--sql") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --sql\n"; return 1; }
            inlineSql = argv[++i]; hasInlineSql = true; continue;
        }
        if (arg.rfind("--sql=", 0) == 0) {
            inlineSql = arg.substr(6); hasInlineSql = true; continue;
        }
        if (arg == "--sql-file") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --sql-file\n"; return 1; }
            sqlFile = argv[++i]; hasSqlFile = true; continue;
        }
        if (arg.rfind("--sql-file=", 0) == 0) {
            sqlFile = arg.substr(11); hasSqlFile = true; continue;
        }
        if (arg == "--csv")               { g_csv = true; continue; }
        if (arg == "--no-pipeline-cache") { g_noPipelineCache = true; continue; }
        if (arg == "--profile-phases")    { g_profilePhases = true; continue; }
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

    if (hasInlineSql && hasSqlFile) {
        std::cerr << "Use either --sql or --sql-file, not both" << std::endl;
        return 1;
    }
    const bool hasSqlRequest = hasInlineSql || hasSqlFile;
    if (hasSqlRequest && !query.empty()) {
        std::cerr << "Ad-hoc SQL options cannot be combined with q<N>, all, mb<N>, or mball" << std::endl;
        return 1;
    }

    if (query.empty() && !hasSqlRequest) {
        std::cerr << "Usage: GPUDBCodegen [sf1|sf10|sf20|sf50|sf100] q<N>|--sql SQL|--sql-file FILE" << std::endl;
        return 1;
    }

    // Apply explicit fast-math selection globally before any compile() runs.
    codegen::RuntimeCompiler::setFastMathEnabled(g_fastMath);
    codegen::RuntimeCompiler::setGlobalCacheEnabled(!g_noPipelineCache);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "No Metal device found" << std::endl;
        return 1;
    }
    device->setShouldMaximizeConcurrentCompilation(true);
    MTL::CommandQueue* cmdQueue = device->newCommandQueue();

    printSystemInfo(getSystemInfo(device));

    auto readSqlFile = [](const std::string& path, std::string& sql) -> bool {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open SQL file: " << path << std::endl;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        sql = ss.str();
        return true;
    };

    auto runQuery = [&](int qNum) -> bool {
        std::string name = "Q" + std::to_string(qNum);
        return runCodegenQuery(device, cmdQueue, "", name, QueryApiKind::PredefinedTPCH);
    };

    auto runMicrobench = [&](int mbNum) -> bool {
        std::string path = "sql/mb" + std::to_string(mbNum) + ".sql";
        std::string sql;
        if (!readSqlFile(path, sql)) return false;
        std::string name = "MB" + std::to_string(mbNum);
        return runCodegenQuery(device, cmdQueue, sql, name, QueryApiKind::AdhocSQL);
    };

    bool ok = true;
    if (hasSqlRequest) {
        std::string sql;
        if (hasInlineSql) {
            sql = inlineSql;
        } else if (!readSqlFile(sqlFile, sql)) {
            return 1;
        }
        ok = runCodegenQuery(device, cmdQueue, sql, "SQL", QueryApiKind::AdhocSQL);
    } else if (query == "all") {
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
