// Standalone SQL-to-Metal entry point.

#include "core/infra.h"
#include "runtime_compiler.h"
#include "metal_plan_builder.h"
#include "api/metal_adhoc_plan_api.h"
#include "api/metal_tpch_plan_api.h"
#include "generic/ir/generic_ir_builder.h"
#include "metal_generic_executor.h"
#include "max_key_symbols.h"
#include "query_preprocessing.h"
#include "predefined_result_finalizer.h"
#include "chunked_colbin_loader.h"
#include "tpch_schema.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <set>
#include <map>
#include <algorithm>
#include <memory>
#include <optional>
#include <filesystem>
#include <vector>
#include <unistd.h>

// Runtime flags shared by main and runCodegenQuery.
static int  g_warmup            = 3;     // --warmup N
static int  g_repeat            = 1;     // --repeat N
static bool g_csv               = false; // --csv  (suppress human-readable breakdown)
static int  g_tgSizeOverride    = 0;     // --threadgroup-size N (0 = use plan default)
static bool g_autotuneTg        = false; // --autotune-tg  (per-query global TG sweep)
static bool g_autotuneTgPerPhase= false; // --autotune-tg-per-phase (per-kernel TG)
static bool g_noPipelineCache   = false; // --no-pipeline-cache
static bool g_coldStart         = false; // --cold-start
static bool g_clearMetalCache   = false; // --clear-metal-cache
static bool g_profilePhases     = false; // --profile-phases
static bool g_fastMath          = false; // --fastmath
static bool g_printPlan         = false; // --print-plan
static bool g_fullResult        = false; // --full-result
static bool g_microPrivateStorage = false; // --micro-input-storage private
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
static constexpr int kMaxMicrobench = 10;

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

static std::optional<std::string> darwinUserCacheDir() {
    const char* override = std::getenv("GPUDB_DARWIN_USER_CACHE_DIR");
    if (override && override[0]) return std::string(override);

#ifdef _CS_DARWIN_USER_CACHE_DIR
    size_t len = confstr(_CS_DARWIN_USER_CACHE_DIR, nullptr, 0);
    if (len > 0) {
        std::string out(len, '\0');
        size_t written = confstr(_CS_DARWIN_USER_CACHE_DIR, out.data(), out.size());
        if (written > 0) {
            while (!out.empty() && out.back() == '\0') out.pop_back();
            if (!out.empty()) return out;
        }
    }
#endif
    return std::nullopt;
}

static bool clearMetalUserCaches(std::string& summary) {
    auto rootOpt = darwinUserCacheDir();
    if (!rootOpt) {
        summary = "DARWIN_USER_CACHE_DIR unavailable";
        return false;
    }

    const std::filesystem::path root(*rootOpt);
    const std::vector<std::string> dirs = {
        "com.apple.metal",
        "com.apple.metalfe",
    };

    bool ok = true;
    std::ostringstream ss;
    ss << "root=" << root.string();
    for (const auto& dir : dirs) {
        const auto path = root / dir;
        std::error_code ec;
        uintmax_t removed = std::filesystem::remove_all(path, ec);
        ss << ";" << dir << "=" << removed;
        if (ec) {
            ok = false;
            ss << "(" << ec.message() << ")";
        }
    }
    summary = ss.str();
    return ok;
}

// Read .colbin row count and file size without mapping the payload.
static std::string tableColbinPath(const codegen::SchemaProvider& schema,
                                   const std::string& tableName) {
    std::string path = schema.tableDataPath(tableName);
    if (!path.empty()) return path;
    return g_dataset_path + tableName + ".colbin";
}

static bool peekColbinHeader(const std::string& path,
                              uint64_t& out_n_rows, uint64_t& out_file_size) {
    return colbin::peekRowCount(path, out_n_rows, &out_file_size);
}

// Use the largest referenced .colbin as the streaming table.
static std::string autoDetectStreamTable(
        const codegen::SchemaProvider& schema,
        const std::map<std::string, std::set<std::string>>& tableCols) {
    std::string best;
    uint64_t bestSize = 0;
    for (const auto& [tName, _cols] : tableCols) {
        uint64_t nr = 0, fsz = 0;
        if (peekColbinHeader(tableColbinPath(schema, tName), nr, fsz) &&
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
    const bool isMicrobenchRoute = queryName.rfind("MB", 0) == 0;
    if (g_microPrivateStorage && !isMicrobenchRoute) {
        std::cerr << "Codegen: --micro-input-storage private is only supported for mb<N> microbenchmarks"
                  << std::endl;
        return false;
    }
    const bool usePrivateStorage = g_microPrivateStorage && isMicrobenchRoute;
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

        // Build Generic IR only for the ad-hoc route.
        std::optional<codegen::GenericRelPlan> genericIr;
        codegen::TPCHSchemaProvider tpchSchema(g_dataset_path);
        const codegen::SchemaProvider* activeSchema =
            static_cast<const codegen::SchemaProvider*>(&tpchSchema);
        if (apiKind == QueryApiKind::AdhocSQL) {
            auto tAnalyze0 = clk::now();
            std::string irError;
            genericIr = codegen::buildGenericRelationalIRFromSQL(
                sql, *activeSchema, &irError);
            if (!genericIr) {
                std::cerr << "Codegen: generic relational IR build failed for "
                          << queryName << std::endl;
                if (!irError.empty()) {
                    std::cerr << "  Reason: " << irError << std::endl;
                }
                return false;
            }
            if (genericIr->schema) activeSchema = genericIr->schema;
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
            if (!genericIr) {
                std::cerr << "Codegen: ad-hoc SQL requires successful Generic IR build for "
                          << queryName << std::endl;
                return false;
            }
            std::string planError;
            maybePlan = codegen::buildAdhocGenericPlan(*genericIr, queryName, &planError);
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
        if (usePrivateStorage && g_chunkRows > 0) {
            std::cerr << "Codegen: --micro-input-storage private does not support --chunk; "
                         "private storage is a full-table microbenchmark mode"
                      << std::endl;
            return false;
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
        auto cg = codegen::generateFromPlan(plan, activeSchema);
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

        if (g_clearMetalCache) {
            std::string cacheSummary;
            if (!clearMetalUserCaches(cacheSummary)) {
                std::cerr << "Codegen: failed to clear Metal user cache: "
                          << cacheSummary << std::endl;
                return false;
            }
            if (g_csv) {
                printf("METAL_CACHE_CSV,%s,%s,%s\n",
                       timing.scaleFactor.c_str(),
                       timing.queryName.c_str(),
                       cacheSummary.c_str());
            } else {
                printf("[metal-cache] cleared %s\n", cacheSummary.c_str());
            }
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
        if (plan.chunkable && !usePrivateStorage && (g_autoChunk || g_chunkRows > 0)) {
            const std::string autoStreamTable = autoDetectStreamTable(*activeSchema, tableCols);
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

                auto elemBytes = [&](codegen::DataType type, int fixedWidth) -> uint64_t {
                    switch (type) {
                        case codegen::DataType::INT:
                        case codegen::DataType::DATE:
                        case codegen::DataType::FLOAT:      return 4;
                        case codegen::DataType::CHAR1:      return 1;
                        case codegen::DataType::CHAR_FIXED: return (uint64_t)fixedWidth;
                    }
                    return 0;
                };
                auto projectedBytesPerRow = [&](const std::string& tName,
                                                 const std::set<std::string>& cols) {
                    uint64_t bpr = 0;
                    for (const auto& cn : cols) {
                        bpr += elemBytes(activeSchema->columnType(tName, cn),
                                         activeSchema->columnFixedWidth(tName, cn));
                    }
                    return bpr;
                };

                // Full file size is diagnostic; projected size drives chunking.
                uint64_t totalDataBytes = 0;
                std::map<std::string, uint64_t> fullTableRows;
                for (const auto& [tName, _cols] : tableCols) {
                    uint64_t nr = 0, fsz = 0;
                    if (peekColbinHeader(tableColbinPath(*activeSchema, tName), nr, fsz)) {
                        totalDataBytes += fsz;
                        fullTableRows[tName] = nr;
                    }
                }

                uint64_t residentBytes = 0;
                for (const auto& [tName, cols] : tableCols) {
                    if (tName == autoStreamTable) continue;
                    uint64_t nr = 0, fsz = 0;
                    if (!peekColbinHeader(tableColbinPath(*activeSchema, tName), nr, fsz))
                        continue;
                    residentBytes += nr * projectedBytesPerRow(tName, cols);
                }

                uint64_t streamProjectedBytes = 0;
                {
                    uint64_t nr = 0, fsz = 0;
                    if (peekColbinHeader(tableColbinPath(*activeSchema, autoStreamTable),
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
                    if (peekColbinHeader(tableColbinPath(*activeSchema, autoStreamTable),
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
        executor.setPrivateDeviceBuffers(usePrivateStorage);

        const std::string streamTable = (g_chunkRows > 0)
            ? autoDetectStreamTable(*activeSchema, tableCols) : std::string{};
        bool didChunk = false;

        // Stream table columns are loaded per chunk below.
        double ioMs = 0.0;
        double preprocessMs = 0.0;
        double privateInputUploadMs = 0.0;
        size_t privateInputBytes = 0;
        std::vector<std::pair<std::string, QueryColumns>> loadedTables;
        for (auto& [tableName, colNames] : tableCols) {
            if (!streamTable.empty() && tableName == streamTable) continue;
            std::vector<ColSpec> specs;
            for (const auto& colName : colNames)
                specs.push_back(codegen::colSpecFor(*activeSchema, tableName, colName));
            auto _ioStart = clk::now();
            auto cols = loadQueryColumns(device, tableColbinPath(*activeSchema, tableName), specs);
            if (usePrivateStorage) {
                size_t uploadedBytes = 0;
                std::string uploadError;
                auto uploadStart = clk::now();
                if (!cols.promoteBuffersToPrivate(device, cmdQueue,
                                                  &uploadedBytes, &uploadError)) {
                    std::cerr << "Codegen: private input upload failed for "
                              << tableName;
                    if (!uploadError.empty())
                        std::cerr << ": " << uploadError;
                    std::cerr << std::endl;
                    return false;
                }
                privateInputUploadMs += elapsedMs(uploadStart, clk::now());
                privateInputBytes += uploadedBytes;
            }
            ioMs += elapsedMs(_ioStart, clk::now());
            size_t rowCount = cols.rows();
            for (const auto& colName : colNames) {
                const int columnIndex = activeSchema->columnIndex(tableName, colName);
                MTL::Buffer* buf = cols.buffer(columnIndex);
                if (!buf) continue;
                executor.registerTableBuffer(tableName, colName, buf, rowCount);
            }
            executor.registerTableRowCount(tableName, rowCount);
            loadedTables.emplace_back(tableName, std::move(cols));
        }
        if (usePrivateStorage) {
            if (g_csv) {
                printf("MICRO_STORAGE_CSV,%s,%s,private,%zu,%.3f\n",
                       timing.scaleFactor.c_str(), queryName.c_str(),
                       privateInputBytes, privateInputUploadMs);
            } else {
                printf("[micro-storage] %s: promoted %.1f MiB of table inputs to private buffers in %.3f ms; device/output buffers private\n",
                       queryName.c_str(),
                       (double)privateInputBytes / (1024.0 * 1024.0),
                       privateInputUploadMs);
            }
        }

        {
            auto _ppStart = clk::now();
            codegen::registerMaxKeySymbols(executor, loadedTables, tableCols, *activeSchema);
            if (!streamTable.empty() && tableCols.count(streamTable)) {
                codegen::extendMaxKeysFromStreamColbin(
                    executor,
                    tableColbinPath(*activeSchema, streamTable),
                    tableCols.at(streamTable),
                    *activeSchema,
                    streamTable);
            }
            codegen::resetQueryPreprocessingState();
            if (isPredefinedTpchRoute &&
                !codegen::prepareQueryPreprocessing(plan.name, device, executor,
                                                    *activeSchema, loadedTables)) {
                return false;
            }
            preprocessMs += elapsedMs(_ppStart, clk::now());
        }

        codegen::MetalExecutionResult result;

        // Chunked path: kernels accumulate into output buffers across chunks.
        if (!streamTable.empty()) {
            std::vector<ColSpec> streamSpecs;
            for (const auto& colName : tableCols.at(streamTable))
                streamSpecs.push_back(codegen::colSpecFor(*activeSchema, streamTable, colName));
            const int streamSlots = g_chunkDoubleBuffer ? 2 : 1;
            codegen::ChunkedColbinTable stream;
            std::string streamError;
            auto streamOpenStart = clk::now();
            if (!stream.open(device, tableColbinPath(*activeSchema, streamTable),
                             streamSpecs, g_chunkRows, streamSlots, streamError)) {
                std::cerr << "Codegen: chunk open failed for " << streamTable
                          << ": " << streamError << std::endl;
                return false;
            }
            double streamOpenMs = elapsedMs(streamOpenStart, clk::now());
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
                    const int columnIndex = activeSchema->columnIndex(streamTable, colName);
                    MTL::Buffer* buf = stream.buffer(slot, columnIndex);
                    if (!buf) {
                        std::cerr << "Codegen: missing chunk buffer for "
                                  << streamTable << "." << colName << std::endl;
                        return false;
                    }
                    executor.registerTableBuffer(streamTable, colName, buf, rowsThisChunk);
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
            double finalCollectMs = elapsedMs(finalCollectStart, clk::now());
            resultCollectMs += finalCollectMs;
            executeWallMs += finalCollectMs;
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
        if (usePrivateStorage && !timing.loadSource.empty())
            timing.loadSource += "+private-storage";
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
        std::vector<double> psoTrials;     psoTrials.reserve(g_repeat);
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
            double trialPsoMs = 0.0;
            if (g_noPipelineCache) {
                compilerR = std::make_unique<codegen::RuntimeCompiler>(device);
                auto tcr0 = clk::now();
                auto* libR = compilerR->compile(metalSource);
                trialCompileMs = elapsedMs(tcr0, clk::now());
                if (!libR) {
                    std::cerr << "Codegen: Metal recompile failed in --no-pipeline-cache trial\n";
                    return false;
                }
                codegen::RuntimeCompiler::CompiledQuery cR;
                cR.library = libR;
                auto tPsoR0 = clk::now();
                for (const auto& phase : cg.getPhases()) {
                    auto* pso = compilerR->getPipeline(libR, phase.name);
                    if (!pso) {
                        std::cerr << "Codegen: PSO recreation failed for " << phase.name << "\n";
                        return false;
                    }
                    cR.pipelines.push_back(pso);
                    cR.kernelNames.push_back(phase.name);
                }
                trialPsoMs = elapsedMs(tPsoR0, clk::now());
                compiledTrial = cR;
            }

            auto tr0 = clk::now();
            result = executor.execute(compiledTrial, cg, 0, 1);
            double executeWallTrialMs = elapsedMs(tr0, clk::now());

            gpuTrials.push_back((double)result.totalKernelTimeMs);
            compileTrials.push_back(trialCompileMs);
            psoTrials.push_back(trialPsoMs);
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
            // Override one-shot compile/PSO timings with per-trial medians so
            // the headline numbers reflect the cost we're studying without
            // folding PSO creation into Metal Compile.
            timing.compileMs = medianValue(compileTrials);
            timing.psoMs = medianValue(psoTrials);
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
                               HostPostOpTracker* hostPostOpsForRun) -> double {
            auto postStart = std::chrono::high_resolution_clock::now();
            std::vector<std::string> hostOps;
            codegen::finalizeHostResult(plan, executor, result.result,
                                        isPredefinedTpchRoute ? &hostOps : nullptr);
            if (isPredefinedTpchRoute && hostPostOpsForRun) {
                for (const auto& op : hostOps) hostPostOpsForRun->mark(op);
            }
            return elapsedMs(postStart, clk::now());
        };

        std::vector<double> postTrials;
        postTrials.reserve((size_t)g_repeat);
        const codegen::MetalExecutionResult rawResult = result;
        for (int pr = 0; pr < g_repeat; pr++) {
            codegen::MetalExecutionResult postResult = rawResult;
            HostPostOpTracker* tracker = (pr == g_repeat - 1) ? &hostPostOps : nullptr;
            postTrials.push_back(runHostPost(postResult, tracker));
            if (pr == g_repeat - 1) {
                result = std::move(postResult);
            }
        }
        timing.postMs = medianValue(postTrials);

        if (!g_csv && !result.result.columns.empty()) {
            printf("\n%s Results:\n", queryName.c_str());
            int displayLimit = g_fullResult
                ? -1
                : (plan.hostResult ? plan.hostResult->displayLimit : -1);
            result.result.print(displayLimit);
        }

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
            printf("  mb1..mb%d      - Run microbenchmark SQL through the analyzer route\n",
                   kMaxMicrobench);
            printf("  mball         - Run all microbenchmarks\n");
            printf("Loader flags:\n");
            printf("  --no-zerocopy        Disable zero-copy mmap path (copy into shared buffers)\n");
            printf("  --no-binary          Disable .colbin binary loader (force .tbl parser)\n");
            printf("  --chunk N[K|M|G]     Stream certified chunkable plans from .colbin\n");
            printf("  --auto-chunk         Auto-enable chunking for certified chunkable plans (default)\n");
            printf("  --no-auto-chunk      Disable budget-triggered chunking; explicit --chunk still works\n");
            printf("  --force-chunk        Keep explicit --chunk even when the direct load fits budget\n");
            printf("  --no-db              With --chunk, use one reusable chunk slot instead of two\n");
            printf("  --micro-input-storage MODE\n");
            printf("                       For mb<N> only, use shared or all-private kernel buffers\n");
            printf("  --micro-private-storage Alias for --micro-input-storage private\n");
            printf("  --micro-private-inputs  Deprecated alias for --micro-input-storage private\n");
            printf("Experiment flags:\n");
            printf("  --warmup N           Run N untimed warmup iterations (default 3)\n");
            printf("  --repeat N           Run N timed iterations, report median (default 1)\n");
            printf("  --csv                Suppress text breakdown; emit CSV timing rows\n");
            printf("  --threadgroup-size N Override default threadgroup size (default = plan-specified)\n");
            printf("  --autotune-tg        Per-query global TG sweep over {32,64,128,256,512,1024};\n");
            printf("                       picks the size with min p50 GPU time (logs AUTOTUNE_CSV)\n");
            printf("  --autotune-tg-per-phase  Per-phase TG sweep; picks min-p50 TG independently\n");
            printf("                       for each kernel (logs AUTOTUNE_PHASE_CSV)\n");
            printf("  --cold-start         Single-query cold JIT mode: clear Metal user cache,\n");
            printf("                       force --warmup 0 --repeat 1, and measure first compile/PSO\n");
            printf("  --clear-metal-cache  Remove user-level Metal cache dirs before compiling each query\n");
            printf("  --no-pipeline-cache  Recompile Metal source on every measured iteration\n");
            printf("                       (JIT ablation only; not a cold-start measurement)\n");
            printf("  --profile-phases     Emit per-phase GPU, wall, residual, and hook timings\n");
            printf("  --fastmath           Enable Metal -ffast-math (default: off)\n");
            printf("  --no-fastmath        Disable Metal -ffast-math (default behavior)\n");
            printf("  --print-plan         Print the MetalQueryPlan structure before codegen\n");
            printf("  --full-result        Print every result row instead of the plan display limit\n");
            printf("  --dump-msl DIR       Write generated MSL to DIR/<query>.metal (default: debug/)\n");
            printf("  --check DIR          Compare GPU result against DIR/<query>_<sf>.csv (golden)\n");
            printf("  --save-golden DIR    Write current GPU result to DIR/<query>_<sf>.csv (overwrites)\n");
            printf("  --check-abs-tol N    Absolute float tolerance (default 1e-2)\n");
            printf("  --check-rel-tol N    Relative float tolerance (default 1e-4)\n");
            printf("  --scalar-atomic      Reduction ablation: every thread issues a global atomic\n");
            printf("                       (disables SIMD+TG reduce; for B2 ablation)\n");
            printf("  --keyed-agg-backend MODE\n");
            printf("                       Force keyed aggregation backend: auto, private,\n");
            printf("                       shared/tg_atomic/histogram, or global\n");
            return 0;
        }
        if (arg == "--no-zerocopy")       { ::setenv("GPUDB_NO_ZEROCOPY", "1", 1); continue; }
        if (arg == "--no-binary")         { ::setenv("GPUDB_NO_BINARY",   "1", 1); continue; }
        if (arg == "--no-db")             { g_chunkDoubleBuffer = false; continue; }
        if (arg == "--auto-chunk")        { g_autoChunk = true; continue; }
        if (arg == "--no-auto-chunk")     { g_autoChunk = false; continue; }
        if (arg == "--force-chunk")       { g_forceChunk = true; continue; }
        auto setMicroInputStorage = [](const std::string& mode) -> bool {
            if (mode == "shared") {
                g_microPrivateStorage = false;
                return true;
            }
            if (mode == "private") {
                g_microPrivateStorage = true;
                return true;
            }
            return false;
        };
        if (arg == "--micro-private-inputs") {
            g_microPrivateStorage = true;
            continue;
        }
        if (arg == "--micro-private-storage") {
            g_microPrivateStorage = true;
            continue;
        }
        if (arg == "--micro-input-storage") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --micro-input-storage\n"; return 1; }
            std::string value = argv[++i];
            if (!setMicroInputStorage(value)) {
                std::cerr << "Invalid --micro-input-storage: " << value
                          << " (expected shared or private)\n";
                return 1;
            }
            continue;
        }
        if (arg.rfind("--micro-input-storage=", 0) == 0) {
            std::string value = arg.substr(22);
            if (!setMicroInputStorage(value)) {
                std::cerr << "Invalid --micro-input-storage: " << value
                          << " (expected shared or private)\n";
                return 1;
            }
            continue;
        }
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
        auto setKeyedAggBackend = [](const std::string& mode) -> bool {
            if (mode == "auto") {
                ::unsetenv("GPUDB_KEYED_AGG_BACKEND");
                return true;
            }
            if (mode == "private" || mode == "shared" || mode == "tg_atomic" ||
                mode == "histogram" || mode == "global") {
                ::setenv("GPUDB_KEYED_AGG_BACKEND", mode.c_str(), 1);
                return true;
            }
            return false;
        };
        if (arg == "--keyed-agg-backend") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --keyed-agg-backend\n"; return 1; }
            std::string value = argv[++i];
            if (!setKeyedAggBackend(value)) {
                std::cerr << "Invalid --keyed-agg-backend: " << value
                          << " (expected auto, private, shared, tg_atomic, histogram, or global)\n";
                return 1;
            }
            continue;
        }
        if (arg.rfind("--keyed-agg-backend=", 0) == 0) {
            std::string value = arg.substr(20);
            if (!setKeyedAggBackend(value)) {
                std::cerr << "Invalid --keyed-agg-backend: " << value
                          << " (expected auto, private, shared, tg_atomic, histogram, or global)\n";
                return 1;
            }
            continue;
        }
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
        if (arg == "--cold-start")        { g_coldStart = true; continue; }
        if (arg == "--clear-metal-cache") { g_clearMetalCache = true; continue; }
        if (arg == "--no-pipeline-cache") { g_noPipelineCache = true; continue; }
        if (arg == "--profile-phases")    { g_profilePhases = true; continue; }
        if (arg == "--fastmath")          { g_fastMath = true; continue; }
        if (arg == "--no-fastmath")       { g_fastMath = false; continue; }
        if (arg == "--print-plan")        { g_printPlan = true; continue; }
        if (arg == "--full-result")       { g_fullResult = true; continue; }
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

    if (g_coldStart) {
        if (g_noPipelineCache) {
            std::cerr << "Codegen: --cold-start cannot be combined with "
                         "--no-pipeline-cache. Cold-start measures the first "
                         "compile/PSO in a fresh process; --no-pipeline-cache "
                         "measures per-trial recompilation after an initial compile."
                      << std::endl;
            return 1;
        }
        if (g_autotuneTg || g_autotuneTgPerPhase) {
            std::cerr << "Codegen: --cold-start cannot be combined with autotuning, "
                         "which runs extra warmup measurements."
                      << std::endl;
            return 1;
        }
        if (query == "all" || query == "mball") {
            std::cerr << "Codegen: --cold-start is single-query only. Run one "
                         "fresh GPUDBCodegen process per query, e.g. loop over "
                         "sf10 q1..q22, so process-level Metal state cannot leak "
                         "between queries."
                      << std::endl;
            return 1;
        }
        g_clearMetalCache = true;
        g_warmup = 0;
        g_repeat = 1;
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
        for (int m = 1; m <= kMaxMicrobench; m++) ok = runMicrobench(m) && ok;
    } else if (query.size() >= 3 && query[0] == 'm' && query[1] == 'b') {
        int mbNum = std::stoi(query.substr(2));
        if (mbNum >= 1 && mbNum <= kMaxMicrobench) {
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
