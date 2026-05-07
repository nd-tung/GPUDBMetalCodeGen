#include "query_preprocessing.h"
#include "tpch_schema.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace codegen {

Q20PostData g_q20Post;
Q2PostData g_q2Post;
Q16PostData g_q16Post;
Q21PostData g_q21Post;
Q18PostData g_q18Post;

QueryColumns loadPreprocessColumns(MTL::Device* device,
                                   const std::string& tableName,
                                   const std::vector<ColSpec>& specs) {
    return loadQueryColumns(device, g_dataset_path + tableName + ".tbl", specs);
}

std::vector<int> copyIntColumn(const QueryColumns& columns, int columnIndex) {
    const int* values = columns.ints(columnIndex);
    return values ? std::vector<int>(values, values + columns.rows()) : std::vector<int>{};
}

std::vector<float> copyFloatColumn(const QueryColumns& columns, int columnIndex) {
    const float* values = columns.floats(columnIndex);
    return values ? std::vector<float>(values, values + columns.rows()) : std::vector<float>{};
}

std::vector<char> copyCharColumn(const QueryColumns& columns, int columnIndex, size_t byteCount) {
    const char* values = columns.chars(columnIndex);
    return values ? std::vector<char>(values, values + byteCount) : std::vector<char>{};
}

namespace {

bool hasColumn(const QueryColumns& columns, const ColSpec& spec) {
    switch (spec.type) {
        case ColType::INT:
        case ColType::DATE:
            return columns.ints(spec.columnIndex) != nullptr;
        case ColType::FLOAT:
            return columns.floats(spec.columnIndex) != nullptr;
        case ColType::CHAR1:
        case ColType::CHAR_FIXED:
            return columns.chars(spec.columnIndex) != nullptr;
    }
    return false;
}

const QueryColumns* findLoadedColumns(const std::vector<LoadedQueryTable>& loadedTables,
                                      const std::string& tableName,
                                      const std::vector<ColSpec>& specs) {
    for (const auto& [loadedName, columns] : loadedTables) {
        if (loadedName != tableName) continue;
        for (const auto& spec : specs) {
            if (!hasColumn(columns, spec)) return nullptr;
        }
        return &columns;
    }
    return nullptr;
}

struct PreprocessColumns {
    QueryColumns owned;
    const QueryColumns* borrowed = nullptr;

    const QueryColumns& get() const {
        return borrowed ? *borrowed : owned;
    }
};

PreprocessColumns resolvePreprocessColumns(MTL::Device* device,
                                           const std::string& tableName,
                                           const std::vector<ColSpec>& specs,
                                           const std::vector<LoadedQueryTable>& loadedTables) {
    PreprocessColumns result;
    if (const QueryColumns* loaded = findLoadedColumns(loadedTables, tableName, specs)) {
        result.borrowed = loaded;
        return result;
    }
    result.owned = loadPreprocessColumns(device, tableName, specs);
    return result;
}

int findFixedNameKey(const QueryColumns& columns, int keyColumn, int nameColumn,
                     int width, const std::string& target) {
    const int* keys = columns.ints(keyColumn);
    const char* names = columns.chars(nameColumn);
    if (!keys || !names) return -1;
    for (size_t i = 0; i < columns.rows(); i++) {
        if (trimFixed(names, i, width) == target) return keys[i];
    }
    return -1;
}

// Resolve a fixed-width NAME -> id from a small lookup table (e.g. nation/region
// where col 0 = id, col 1 = CHAR(25) name) and register it as a scalar int param.
// Returns false (and logs to stderr) if the name is missing.
bool registerNameKey(MTL::Device* device,
                     MetalGenericExecutor& executor,
                     const std::vector<LoadedQueryTable>& loadedTables,
                     const std::string& tableName,
                     const std::string& target,
                     const std::string& paramName) {
    auto view = resolvePreprocessColumns(device, tableName,
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
    int key = findFixedNameKey(view.get(), 0, 1, 25, target);
    if (key == -1) {
        std::cerr << "Error: " << target << " not found in " << tableName << " table\n";
        return false;
    }
    executor.registerScalarInt(paramName, key);
    return true;
}

MTL::Buffer* registerFilledBuffer(MTL::Device* device,
                                  MetalGenericExecutor& executor,
                                  const std::string& name,
                                  size_t bytes,
                                  int fillByte = 0) {
    size_t allocBytes = std::max(bytes, (size_t)4);
    auto* buffer = device->newBuffer(allocBytes, MTL::ResourceStorageModeShared);
    // Large fills are dominated by the CPU memset (>= 1 MB shows up in
    // perf traces). Use a Metal blit fillBuffer for these — the GPU
    // memsets at full bandwidth and overlaps with subsequent CPU prep.
    // Small fills stay on CPU because the command-buffer dispatch is
    // ~50 us of fixed overhead.
    constexpr size_t kBlitThreshold = 256 * 1024;
    if (allocBytes >= kBlitThreshold && executor.commandQueue()) {
        auto* cmdBuf = executor.commandQueue()->commandBuffer();
        auto* blit = cmdBuf->blitCommandEncoder();
        blit->fillBuffer(buffer, NS::Range(0, allocBytes), (uint8_t)fillByte);
        blit->endEncoding();
        cmdBuf->commit();
        // Don't wait — subsequent GPU phases naturally serialize behind
        // this command buffer on the same queue. The buffer is safe to
        // hand back immediately because no CPU reads it before phase 1.
    } else {
        memset(buffer->contents(), fillByte, allocBytes);
    }
    executor.registerAllocatedBuffer(name, buffer);
    return buffer;
}

template<typename T>
MTL::Buffer* uploadAndRegister(MTL::Device* device,
                               MetalGenericExecutor& executor,
                               const std::string& name,
                               const std::vector<T>& values) {
    const size_t bytes = values.size() * sizeof(T);
    if (bytes == 0) return registerFilledBuffer(device, executor, name, 0);
    auto* buffer = device->newBuffer(values.data(), bytes, MTL::ResourceStorageModeShared);
    executor.registerAllocatedBuffer(name, buffer);
    return buffer;
}

} // namespace

bool prepareQueryPreprocessing(const std::string& queryName,
                               MTL::Device* device,
                               MetalGenericExecutor& executor,
                               const std::vector<LoadedQueryTable>& loadedTables) {
    // Reset all per-query post-processing globals so a second in-process
    // call (e.g. a sweep) does not see stale vectors / GPU buffer pointers
    // from a previous query. The buffers themselves are owned by the
    // executor's allocatedBuffers_ and freed on its destruction.
    g_q2Post  = {};
    g_q16Post = {};
    g_q18Post = {};
    g_q20Post = {};
    g_q21Post = {};

    // Q7: resolve nation keys
    if (queryName == "Q7") {
        if (!registerNameKey(device, executor, loadedTables, "nation", "FRANCE",  "france_nk"))  return false;
        if (!registerNameKey(device, executor, loadedTables, "nation", "GERMANY", "germany_nk")) return false;
    }

    // Q5: resolve ASIA regionkey
    if (queryName == "Q5") {
        if (!registerNameKey(device, executor, loadedTables, "region", "ASIA", "asia_rk")) return false;
    }

    // Q8: resolve AMERICA regionkey and BRAZIL nationkey
    if (queryName == "Q8") {
        if (!registerNameKey(device, executor, loadedTables, "region", "AMERICA", "america_rk")) return false;
        if (!registerNameKey(device, executor, loadedTables, "nation", "BRAZIL",  "brazil_nk"))  return false;
    }

    // Q22: avg_bal computed on GPU via Q22_compute_avg_bal phase; the
    // postDispatchHook registers the avg_bal scalar before the next phase.

    // Q11: resolve GERMANY nationkey
    if (queryName == "Q11") {
        if (!registerNameKey(device, executor, loadedTables, "nation", "GERMANY", "germany_nk")) return false;
    }

    // Q17: bitmap, per-partkey sum/count, and threshold test all run on
    // GPU via Q17_build_bitmap and Q17_build_avg_qty phases.

    // Q9: green-parts bitmap, lookup arrays, and partsupp HT all run on
    // GPU. Host work: size the (pk,sk) HT and register its scalars
    // (`d_ps_ht_mask`, `supp_mul`). HT is sized to next_pow2(2 * n_partsupp)
    // so load factor stays ≤ 0.5.
    if (queryName == "Q9") {
        size_t nPartSupp = 0;
        if (!executor.tryGetSymbol("n_partsupp", nPartSupp) || nPartSupp == 0) {
            std::cerr << "Q9 preprocessing: n_partsupp symbol unavailable\n";
            return false;
        }
        size_t maxSk = 0;
        if (!executor.tryGetSymbol("maxSuppkey", maxSk) || maxSk == 0) {
            std::cerr << "Q9 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        size_t htSlots = 1;
        while (htSlots < nPartSupp * 2) htSlots <<= 1;
        executor.registerSymbol("q9HtSize", htSlots);
        executor.registerScalarInt("d_ps_ht_mask", (int)(htSlots - 1));
        executor.registerScalarInt("supp_mul", (int)maxSk);
    }

    // Q20: forest bitmap, partsupp HT, and lineitem agg all run on GPU.
    // Host work: tiny CANADA-nation lookup, supplier/partsupp mirrors for
    // post-processing, HT sizing scalars.
    if (queryName == "Q20") {
        size_t nPartSupp = 0, maxSk = 0;
        if (!executor.tryGetSymbol("n_partsupp", nPartSupp) || nPartSupp == 0) {
            std::cerr << "Q20 preprocessing: n_partsupp symbol unavailable\n";
            return false;
        }
        if (!executor.tryGetSymbol("maxSuppkey", maxSk) || maxSk == 0) {
            std::cerr << "Q20 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        size_t htSlots = 1;
        while (htSlots < nPartSupp * 2) htSlots <<= 1;
        executor.registerSymbol("q20HtSize", htSlots);
        executor.registerScalarInt("d_q20_ht_mask", (int)(htSlots - 1));
        executor.registerScalarInt("supp_mul", (int)maxSk);

        // Mirror borrows for post-processing.
        auto psView = resolvePreprocessColumns(device, "partsupp",
            {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}}, loadedTables);
        if (!psView.borrowed) g_q20Post.ownedPartsupp = std::move(psView.owned);
        const QueryColumns& psCols = psView.borrowed ? *psView.borrowed : g_q20Post.ownedPartsupp;
        g_q20Post.ps_partkey  = psCols.ints(0);
        g_q20Post.ps_suppkey  = psCols.ints(1);
        g_q20Post.ps_availqty = psCols.ints(2);
        g_q20Post.nPS = psCols.rows();

        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT}},
            loadedTables);
        if (!sView.borrowed) g_q20Post.ownedSupplier = std::move(sView.owned);
        const QueryColumns& sCols = sView.borrowed ? *sView.borrowed : g_q20Post.ownedSupplier;
        g_q20Post.s_suppkey   = sCols.ints(0);
        g_q20Post.s_name      = sCols.chars(1);
        g_q20Post.s_address   = sCols.chars(2);
        g_q20Post.s_nationkey = sCols.ints(3);
        g_q20Post.nS = sCols.rows();

        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& nCols = nView.get();
        const int*  nk_p = nCols.ints(0);
        const char* nm_p = nCols.chars(1);
        for (size_t i = 0; i < nCols.rows(); i++) {
            if (nm_p[i*25]=='C' && nm_p[i*25+1]=='A' && nm_p[i*25+2]=='N') {
                g_q20Post.canada_nk = nk_p[i];
                break;
            }
        }

        g_q20Post.htMask = (uint32_t)(htSlots - 1);
        g_q20Post.htSlots = (uint32_t)htSlots;

        // Q20_filter_ht_to_bitmap GPU phase: range scan htSlots and
        // atomic-OR set qualifying suppkey bits into d_q20_qual_supp_bitmap.
        // Replaces the q20HtSize-row CPU loop building std::set qualSuppkeys.
        executor.registerSymbol("n_q20_ht_slots", htSlots);
        executor.registerScalarInt("n_q20_ht_slots", (int)htSlots);
        size_t qualBmpInts = (maxSk + 32) / 32;
        registerFilledBuffer(device, executor, "d_q20_qual_supp_bitmap",
                             qualBmpInts * sizeof(uint32_t));
    }

    // Q2: build EUROPE supplier bitmap, allocate part bitmap (filled by GPU Phase 1)
    if (queryName == "Q2") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {2, ColType::CHAR_FIXED, 25}}, loadedTables);
        auto sView = resolvePreprocessColumns(device, "supplier", {
            {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
            {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT},
            {6, ColType::CHAR_FIXED, 101}
        }, loadedTables);
        auto psView = resolvePreprocessColumns(device, "partsupp",
            {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}}, loadedTables);
        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::INT}}, loadedTables);
        auto rView = resolvePreprocessColumns(device, "region",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);

        if (!pView.borrowed)  g_q2Post.ownedPart     = std::move(pView.owned);
        if (!sView.borrowed)  g_q2Post.ownedSupplier = std::move(sView.owned);
        if (!psView.borrowed) g_q2Post.ownedPartsupp = std::move(psView.owned);
        const QueryColumns& pCols  = pView.borrowed  ? *pView.borrowed  : g_q2Post.ownedPart;
        const QueryColumns& sCols  = sView.borrowed  ? *sView.borrowed  : g_q2Post.ownedSupplier;
        const QueryColumns& psCols = psView.borrowed ? *psView.borrowed : g_q2Post.ownedPartsupp;
        const auto& nCols = nView.get();
        const auto& rCols = rView.get();

        const size_t nP  = pCols.rows();
        const size_t nS  = sCols.rows();
        const size_t nPS = psCols.rows();
        const size_t nN  = nCols.rows();
        const size_t nR  = rCols.rows();

        g_q2Post.ps_partkey    = psCols.ints(0);
        g_q2Post.ps_suppkey    = psCols.ints(1);
        g_q2Post.ps_supplycost = psCols.floats(3);
        g_q2Post.nPS = nPS;

        g_q2Post.s_suppkey   = sCols.ints(0);
        g_q2Post.s_name      = sCols.chars(1);
        g_q2Post.s_address   = sCols.chars(2);
        g_q2Post.s_nationkey = sCols.ints(3);
        g_q2Post.s_phone     = sCols.chars(4);
        g_q2Post.s_acctbal   = sCols.floats(5);
        g_q2Post.s_comment   = sCols.chars(6);
        g_q2Post.nS = nS;

        g_q2Post.p_partkey = pCols.ints(0);
        g_q2Post.p_mfgr    = pCols.chars(2);
        g_q2Post.nP = nP;

        // Small serial work: nation/region scans + bitmap derivation.
        const int*  n_nationkey = nCols.ints(0);
        const char* n_name      = nCols.chars(1);
        const int*  n_regionkey = nCols.ints(2);
        const int*  r_regionkey = rCols.ints(0);
        const char* r_name      = rCols.chars(1);

        int europe_rk = -1;
        for (size_t i = 0; i < nR; i++) {
            if (r_name[i*25]=='E' && r_name[i*25+1]=='U' &&
                r_name[i*25+2]=='R' && r_name[i*25+3]=='O') {
                europe_rk = r_regionkey[i];
                break;
            }
        }

        std::set<int> europeNks;
        for (size_t i = 0; i < nN; i++) {
            if (n_regionkey[i] == europe_rk) europeNks.insert(n_nationkey[i]);
        }

        std::vector<std::string> nationNames(25);
        for (size_t i = 0; i < nN; i++) {
            int len = 25;
            while (len > 0 && (n_name[i*25 + len - 1] == ' ' || n_name[i*25 + len - 1] == '\0')) len--;
            nationNames[n_nationkey[i]] = std::string(n_name + i*25, len);
        }

        // Use canonical maxSuppkey/maxPartkey symbols (registered as count = max + 1).
        size_t maxSkSym = 0, maxPkSym = 0;
        int maxSk, maxPk;
        if (executor.tryGetSymbol("maxSuppkey", maxSkSym) && maxSkSym > 0) {
            maxSk = (int)(maxSkSym - 1);
        } else {
            maxSk = 0;
            for (size_t i = 0; i < nS; i++) maxSk = std::max(maxSk, g_q2Post.s_suppkey[i]);
        }
        if (executor.tryGetSymbol("maxPartkey", maxPkSym) && maxPkSym > 0) {
            maxPk = (int)(maxPkSym - 1);
        } else {
            maxPk = 0;
            for (size_t i = 0; i < nP; i++) maxPk = std::max(maxPk, g_q2Post.p_partkey[i]);
        }

        size_t suppBmpInts = ((size_t)maxSk + 32) / 32;
        std::vector<uint32_t> eurSuppBitmap(suppBmpInts, 0);
        for (size_t i = 0; i < nS; i++) {
            if (europeNks.count(g_q2Post.s_nationkey[i])) {
                int sk = g_q2Post.s_suppkey[i];
                eurSuppBitmap[sk / 32] |= (1u << (sk % 32));
            }
        }

        size_t partBmpInts = ((size_t)maxPk + 32) / 32;
        registerFilledBuffer(device, executor, "d_q2_part_bitmap",
                     partBmpInts * sizeof(uint32_t));

        size_t minCostSize = (size_t)maxPk + 1;
        registerFilledBuffer(device, executor, "d_q2_min_cost",
                            minCostSize * sizeof(uint32_t), 0xFF);
        uploadAndRegister(device, executor, "d_q2_supp_bitmap", eurSuppBitmap);

        // Direct-address part/supp index arrays for the Q2 CPU post.
        // Replaces unordered_map<int,int> builds (200K supp + 4M parts at SF20).
        std::vector<int> partIdxArr((size_t)maxPk + 1, -1);
        for (size_t i = 0; i < nP; i++) {
            int pk = g_q2Post.p_partkey[i];
            if (pk >= 0 && pk <= maxPk) partIdxArr[pk] = (int)i;
        }
        std::vector<int> suppIdxArr((size_t)maxSk + 1, -1);
        for (size_t i = 0; i < nS; i++) {
            int sk = g_q2Post.s_suppkey[i];
            if (sk >= 0 && sk <= maxSk) suppIdxArr[sk] = (int)i;
        }

        g_q2Post.nationNames = std::move(nationNames);
        g_q2Post.maxPartkey = maxPk;
        g_q2Post.maxSuppkey = maxSk;
        g_q2Post.partIdxArr = std::move(partIdxArr);
        g_q2Post.suppIdxArr = std::move(suppIdxArr);

        // Q2_compact GPU phase: re-scans partsupp with both bitmaps +
        // cost==min equality, atomic-appends a compact (pk, sk, psi)
        // list. Replaces the 80M-row CPU loop in Q2 post.
        constexpr uint32_t kQ2CompactCap = 1u << 18;
        executor.registerScalarInt("q2_compact_cap", (int)kQ2CompactCap);
        registerFilledBuffer(device, executor, "d_q2_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_compact_pk",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_compact_sk",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_compact_psi",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
    }

    // Q16: GPU-side filter+key compaction. The CPU dictionary build
    // (formerly ~22-470 ms across SF1\u2013SF20) is replaced by a GPU phase
    // (Q16_filter_compact in the plan) that emits a compact list of
    // qualifying part rows + 64-bit (brand_idx, size, fnv32(type)) keys.
    // The post-dispatch hook then builds the dict from ~12 % of the
    // input rows.
    if (queryName == "Q16") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}},
            loadedTables);
        if (!pView.borrowed) g_q16Post.ownedPart = std::move(pView.owned);
        const QueryColumns& pCols = pView.borrowed ? *pView.borrowed : g_q16Post.ownedPart;
        const int*  p_partkey = pCols.ints(0);
        size_t nPart = pCols.rows();
        g_q16Post.p_partkey = p_partkey;
        g_q16Post.p_brand = pCols.chars(3);
        g_q16Post.p_type  = pCols.chars(4);
        g_q16Post.nPart   = nPart;

        // maxSk for complaint bitmap from canonical maxSuppkey symbol
        // (registered as count = max + 1) \u2014 avoids re-scanning supplier.tbl.
        size_t maxSkSym = 0;
        if (!executor.tryGetSymbol("maxSuppkey", maxSkSym) || maxSkSym == 0) {
            std::cerr << "Q16 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        int maxSk = (int)(maxSkSym - 1);
        g_q16Post.maxSk = maxSk;
        size_t complaintBmpInts = ((size_t)maxSk + 32) / 32;

        registerFilledBuffer(device, executor, "d_q16_complaint_bitmap",
                     complaintBmpInts * sizeof(uint32_t));

        // maxPartkey from canonical symbol (set during table load).
        size_t maxPkSym = 0;
        if (executor.tryGetSymbol("maxPartkey", maxPkSym) && maxPkSym > 0) {
            g_q16Post.maxPartkey = (int)(maxPkSym - 1);
        } else {
            int maxPk = 0;
            for (size_t i = 0; i < nPart; i++) maxPk = std::max(maxPk, p_partkey[i]);
            g_q16Post.maxPartkey = maxPk;
        }

        // Pre-allocate compaction buffers + the part_group_map.  The
        // Q16_filter_compact GPU phase fills filt_idx/filt_key/count;
        // its post-dispatch hook builds groups[] and writes into
        // d_q16_part_group_map (pre-filled with -1) directly.
        registerFilledBuffer(device, executor, "d_q16_filt_count", sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q16_filt_idx",  nPart * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q16_filt_key",  nPart * sizeof(uint64_t));
        registerFilledBuffer(device, executor, "d_q16_part_group_map",
                             ((size_t)g_q16Post.maxPartkey + 1) * sizeof(int32_t), 0xFF);
        // d_q16_group_bitmaps is allocated by the post-dispatch hook
        // once numGroups is known.
    }

    // Q3: GPU compact-emit needs range-scan size + compact buffers.
    if (queryName == "Q3") {
        size_t maxOk = 0;
        executor.tryGetSymbol("maxOrderkey", maxOk);
        executor.registerSymbol("n_q3_oks", maxOk);
        executor.registerScalarInt("n_q3_oks", (int)maxOk);
        // Q3 selectivity is ~5% of orders at SF1 (BUILDING segment * date<1995-03-15
        // intersected with shipdate>1995-03-15); cap at maxOk to be safe.
        size_t cap = std::max<size_t>(maxOk, (size_t)(1u << 12));
        executor.registerScalarInt("q3_compact_cap", (int)cap);
        registerFilledBuffer(device, executor, "d_q3_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q3_compact_ok",
                             cap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q3_compact_rev",
                             cap * sizeof(float));
    }

    // Q10: GPU compact-emit phase needs a range-scan size and capped
    // output buffers. Falls back to no-op when symbols unavailable.
    if (queryName == "Q10") {
        size_t maxCk = 0;
        executor.tryGetSymbol("maxCustkey", maxCk);
        executor.registerSymbol("n_q10_cks", maxCk);
        executor.registerScalarInt("n_q10_cks", (int)maxCk);
        // Q10 selectivity is high: most customers have at least one
        // returned-item lineitem at SF>=1. Cap at maxCk to be safe.
        size_t cap = std::max<size_t>(maxCk, (size_t)(1u << 12));
        executor.registerScalarInt("q10_compact_cap", (int)cap);
        registerFilledBuffer(device, executor, "d_q10_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q10_compact_ck",
                             cap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q10_compact_rev",
                             cap * sizeof(float));
    }

    if (queryName == "Q18") {
        auto oView = resolvePreprocessColumns(device, "orders",
            {{1, ColType::INT}, {3, ColType::FLOAT}, {4, ColType::DATE}},
            loadedTables);
        // Take ownership when the view had to be loaded fresh — otherwise
        // the borrowed pointers below would dangle as soon as oView dies.
        if (!oView.borrowed) g_q18Post.ownedOrders = std::move(oView.owned);
        const QueryColumns& oCols = oView.borrowed ? *oView.borrowed : g_q18Post.ownedOrders;
        g_q18Post.o_custkey    = oCols.ints(1);
        g_q18Post.o_totalprice = oCols.floats(3);
        g_q18Post.o_orderdate  = oCols.ints(4);
        // okLookup is now built by the Q18_build_ok_lookup GPU phase
        // and read directly from d_q18_ok_lookup in post.

        // Q18_compact phase: range scan over [0, maxOrderkey+1) of
        // d_order_qty, atomic-append qualifying (ok, qty) pairs into a
        // small compact list. Replaces the maxOrderkey-sized CPU loop
        // (~150M iterations at SF100) with ~few-hundred CPU iterations.
        size_t maxOk = 0;
        executor.tryGetSymbol("maxOrderkey", maxOk);
        // n_q18_oks doubles as both the dispatch sizing symbol (used by
        // the executor when scannedTable=="q18_oks") and the kernel's
        // loop bound scalar.
        executor.registerSymbol("n_q18_oks", maxOk);
        executor.registerScalarInt("n_q18_oks", (int)maxOk);
        // Cap matches the d_q18_compact_* buffer slot count. At all
        // realistic SF the qualifying-order count is in the hundreds;
        // 1<<20 leaves multiple orders of safety margin.
        constexpr uint32_t kQ18CompactCap = 1u << 20;
        executor.registerScalarInt("q18_compact_cap", (int)kQ18CompactCap);
        registerFilledBuffer(device, executor, "d_q18_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_ok",
                             (size_t)kQ18CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_qty",
                             (size_t)kQ18CompactCap * sizeof(float));
    }

    // Q21: GPU builds the SAUDI-supplier bitmap (Q21_build_sa_supp).
    // Host: nation lookup for sa_nk, small s_suppkey/s_name mirror for
    // post-processing, scratch-array allocations sized via the canonical
    // maxOrderkey/maxSuppkey symbols.
    if (queryName == "Q21") {
        if (!registerNameKey(device, executor, loadedTables, "nation",
                             "SAUDI ARABIA", "sa_nk")) {
            return false;
        }

        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        if (!sView.borrowed) g_q21Post.ownedSupplier = std::move(sView.owned);
        const QueryColumns& sCols = sView.borrowed ? *sView.borrowed : g_q21Post.ownedSupplier;
        g_q21Post.s_suppkey = sCols.ints(0);
        g_q21Post.s_name    = sCols.chars(1);
        g_q21Post.nS        = sCols.rows();

        size_t maxOkSym = 0, maxSkSym = 0;
        if (!executor.tryGetSymbol("maxOrderkey", maxOkSym) ||
            !executor.tryGetSymbol("maxSuppkey", maxSkSym)) {
            std::cerr << "Q21 preprocessing: maxOrderkey/maxSuppkey symbols unavailable\n";
            return false;
        }
        // Symbols are registered as actualMax + 1, i.e. element counts.
        size_t fBmpInts = (maxOkSym + 31) / 32;
        size_t okMapSize = maxOkSym;
        size_t suppCountSize = maxSkSym;

        registerFilledBuffer(device, executor, "d_q21_f_orders",
                     fBmpInts * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_first_supp",
                     okMapSize * sizeof(int), 0xFF);
        registerFilledBuffer(device, executor, "d_q21_first_late",
                     okMapSize * sizeof(int), 0xFF);
        registerFilledBuffer(device, executor, "d_q21_multi_supp",
                     fBmpInts * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_multi_late",
                     fBmpInts * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_supp_count",
                     suppCountSize * sizeof(uint32_t));
    }

    return true;
}

} // namespace codegen
