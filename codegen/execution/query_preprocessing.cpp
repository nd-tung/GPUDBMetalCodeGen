#include "query_preprocessing.h"
#include "tpch_schema.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace codegen {

Q16PostData g_q16Post;

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

// Resolve a fixed-width name key and register it as a scalar.
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
    // Use GPU blit for large fills; CPU memset wins below command-buffer overhead.
    constexpr size_t kBlitThreshold = 256 * 1024;
    if (allocBytes >= kBlitThreshold && executor.commandQueue()) {
        auto* cmdBuf = executor.commandQueue()->commandBuffer();
        auto* blit = cmdBuf->blitCommandEncoder();
        blit->fillBuffer(buffer, NS::Range(0, allocBytes), (uint8_t)fillByte);
        blit->endEncoding();
        cmdBuf->commit();
        // Same command queue preserves ordering for later phases.
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

void resetQueryPreprocessingState() {
    g_q16Post = {};
}

bool prepareQueryPreprocessing(const std::string& queryName,
                               MTL::Device* device,
                               MetalGenericExecutor& executor,
                               const std::vector<LoadedQueryTable>& loadedTables) {
    resetQueryPreprocessingState();

    if (queryName == "Q7") {
        if (!registerNameKey(device, executor, loadedTables, "nation", "FRANCE",  "france_nk"))  return false;
        if (!registerNameKey(device, executor, loadedTables, "nation", "GERMANY", "germany_nk")) return false;
    }

    if (queryName == "Q5") {
        if (!registerNameKey(device, executor, loadedTables, "region", "ASIA", "asia_rk")) return false;
        constexpr uint32_t kQ5ResultCap = 25;
        executor.registerScalarInt("q5_result_cap", (int)kQ5ResultCap);
        executor.registerSymbol("q5_result_cap", kQ5ResultCap);
        registerFilledBuffer(device, executor, "d_q5_result_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q5_result_name",
                             (size_t)kQ5ResultCap * 25);
        registerFilledBuffer(device, executor, "d_q5_result_revenue",
                             (size_t)kQ5ResultCap * sizeof(float));
    }

    if (queryName == "Q8") {
        if (!registerNameKey(device, executor, loadedTables, "region", "AMERICA", "america_rk")) return false;
        if (!registerNameKey(device, executor, loadedTables, "nation", "BRAZIL",  "brazil_nk"))  return false;
    }

    // Q22 avg_bal is registered by the GPU phase hook.

    if (queryName == "Q11") {
        if (!registerNameKey(device, executor, loadedTables, "nation", "GERMANY", "germany_nk")) return false;
    }

    // Q17 preprocessing runs in GPU phases.

    if (queryName == "Q13") {
        constexpr uint32_t kQ13HistBins = 256;
        executor.registerSymbol("n_q13_hist_bins", kQ13HistBins);
        executor.registerScalarInt("n_q13_hist_bins", (int)kQ13HistBins);
        executor.registerScalarInt("q13_hist_cap", (int)kQ13HistBins);
        registerFilledBuffer(device, executor, "d_q13_result_count", sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q13_result_c_count",
                             kQ13HistBins * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q13_result_custdist",
                             kQ13HistBins * sizeof(uint32_t));
    }

    // Q9 host work: size the GPU partsupp hash table.
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
        constexpr uint32_t kQ9ProfitBins = 25u * 8u;
        executor.registerSymbol("q9_profit_bins", kQ9ProfitBins);
        executor.registerScalarInt("q9_profit_bins", (int)kQ9ProfitBins);
        executor.registerSymbol("q9_result_cap", kQ9ProfitBins);
        executor.registerScalarInt("q9_result_cap", (int)kQ9ProfitBins);
        registerFilledBuffer(device, executor, "d_q9_result_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q9_result_nation",
                             (size_t)kQ9ProfitBins * 25);
        registerFilledBuffer(device, executor, "d_q9_result_year",
                             (size_t)kQ9ProfitBins * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q9_result_profit",
                             (size_t)kQ9ProfitBins * sizeof(float));
    }

    // Q20 host work: nation key and GPU buffer sizing.
    if (queryName == "Q20") {
        size_t nPartSupp = 0, maxSk = 0, nSupplier = 0;
        if (!executor.tryGetSymbol("n_partsupp", nPartSupp) || nPartSupp == 0) {
            std::cerr << "Q20 preprocessing: n_partsupp symbol unavailable\n";
            return false;
        }
        if (!executor.tryGetSymbol("maxSuppkey", maxSk) || maxSk == 0) {
            std::cerr << "Q20 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        if (!executor.tryGetSymbol("n_supplier", nSupplier) || nSupplier == 0) {
            std::cerr << "Q20 preprocessing: n_supplier symbol unavailable\n";
            return false;
        }
        size_t htSlots = 1;
        while (htSlots < nPartSupp * 2) htSlots <<= 1;
        executor.registerSymbol("q20HtSize", htSlots);
        executor.registerScalarInt("d_q20_ht_mask", (int)(htSlots - 1));
        executor.registerScalarInt("supp_mul", (int)maxSk);

        if (!registerNameKey(device, executor, loadedTables, "nation",
                             "CANADA", "canada_nk")) {
            return false;
        }

        // GPU phase scans HT slots into the qualifying-supplier bitmap.
        executor.registerSymbol("n_q20_ht_slots", htSlots);
        executor.registerScalarInt("n_q20_ht_slots", (int)htSlots);
        size_t qualBmpInts = (maxSk + 32) / 32;
        registerFilledBuffer(device, executor, "d_q20_qual_supp_bitmap",
                             qualBmpInts * sizeof(uint32_t));

        executor.registerScalarInt("q20_result_cap", (int)nSupplier);
        executor.registerSymbol("q20_result_cap", nSupplier);
        registerFilledBuffer(device, executor, "d_q20_result_count", sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q20_result_name", nSupplier * 25);
        registerFilledBuffer(device, executor, "d_q20_result_address", nSupplier * 40);
    }

    // Q2: CPU builds EUROPE supplier bitmap; GPU fills part/min-cost buffers.
    if (queryName == "Q2") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}}, loadedTables);
        auto sView = resolvePreprocessColumns(device, "supplier", {
            {0, ColType::INT}, {3, ColType::INT}
        }, loadedTables);
        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {2, ColType::INT}}, loadedTables);
        auto rView = resolvePreprocessColumns(device, "region",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);

        const QueryColumns& pCols  = pView.get();
        const QueryColumns& sCols  = sView.get();
        const auto& nCols = nView.get();
        const auto& rCols = rView.get();

        const size_t nP  = pCols.rows();
        const size_t nS  = sCols.rows();
        const size_t nN  = nCols.rows();
        const size_t nR  = rCols.rows();
        const int* s_suppkey = sCols.ints(0);
        const int* s_nationkey = sCols.ints(3);
        const int* p_partkey = pCols.ints(0);

        // Small table scans for the EUROPE supplier bitmap.
        const int*  n_nationkey = nCols.ints(0);
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

        // Symbols store counts, so subtract one for max key.
        size_t maxSkSym = 0, maxPkSym = 0;
        int maxSk, maxPk;
        if (executor.tryGetSymbol("maxSuppkey", maxSkSym) && maxSkSym > 0) {
            maxSk = (int)(maxSkSym - 1);
        } else {
            maxSk = 0;
            for (size_t i = 0; i < nS; i++) maxSk = std::max(maxSk, s_suppkey[i]);
        }
        if (executor.tryGetSymbol("maxPartkey", maxPkSym) && maxPkSym > 0) {
            maxPk = (int)(maxPkSym - 1);
        } else {
            maxPk = 0;
            for (size_t i = 0; i < nP; i++) maxPk = std::max(maxPk, p_partkey[i]);
        }

        size_t suppBmpInts = ((size_t)maxSk + 32) / 32;
        std::vector<uint32_t> eurSuppBitmap(suppBmpInts, 0);
        for (size_t i = 0; i < nS; i++) {
            if (europeNks.count(s_nationkey[i])) {
                int sk = s_suppkey[i];
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

        // GPU compaction emits sort keys and row ids; late materialization emits
        // only the visible top-k payload.
        constexpr uint32_t kQ2CompactCap = 1u << 18;
        constexpr uint32_t kQ2LateLimit = 100;
        executor.registerScalarInt("q2_compact_cap", (int)kQ2CompactCap);
        executor.registerSymbol("q2_compact_cap", kQ2CompactCap);
        executor.registerScalarInt("q2_late_limit", (int)kQ2LateLimit);
        executor.registerSymbol("q2_late_limit", kQ2LateLimit);
        registerFilledBuffer(device, executor, "d_q2_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_key_acctbal",
                             (size_t)kQ2CompactCap * sizeof(float));
        registerFilledBuffer(device, executor, "d_q2_key_s_name",
                             (size_t)kQ2CompactCap * 25);
        registerFilledBuffer(device, executor, "d_q2_key_n_name",
                             (size_t)kQ2CompactCap * 25);
        registerFilledBuffer(device, executor, "d_q2_key_p_partkey",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_key_supp_idx",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_key_part_idx",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_key_nation_idx",
                             (size_t)kQ2CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_result_acctbal",
                             (size_t)kQ2LateLimit * sizeof(float));
        registerFilledBuffer(device, executor, "d_q2_result_s_name",
                             (size_t)kQ2LateLimit * 25);
        registerFilledBuffer(device, executor, "d_q2_result_n_name",
                             (size_t)kQ2LateLimit * 25);
        registerFilledBuffer(device, executor, "d_q2_result_p_partkey",
                             (size_t)kQ2LateLimit * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q2_result_p_mfgr",
                             (size_t)kQ2LateLimit * 25);
        registerFilledBuffer(device, executor, "d_q2_result_s_address",
                             (size_t)kQ2LateLimit * 40);
        registerFilledBuffer(device, executor, "d_q2_result_s_phone",
                             (size_t)kQ2LateLimit * 15);
        registerFilledBuffer(device, executor, "d_q2_result_s_comment",
                             (size_t)kQ2LateLimit * 101);
    }

    // Q16: GPU filters qualifying parts; a post-dispatch CPU hook builds the
    // low-cardinality part-group dictionary and fills GPU label buffers.
    if (queryName == "Q16") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10},
             {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}},
            loadedTables);
        if (!pView.borrowed) g_q16Post.ownedPart = std::move(pView.owned);
        const QueryColumns& pCols = pView.borrowed ? *pView.borrowed : g_q16Post.ownedPart;
        const int* p_partkey = pCols.ints(0);
        const size_t nPart = pCols.rows();
        g_q16Post.p_partkey = p_partkey;
        g_q16Post.p_brand = pCols.chars(3);
        g_q16Post.p_type = pCols.chars(4);
        g_q16Post.nPart = nPart;

        size_t maxSkSym = 0;
        if (!executor.tryGetSymbol("maxSuppkey", maxSkSym) || maxSkSym == 0) {
            std::cerr << "Q16 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        g_q16Post.maxSk = (int)(maxSkSym - 1);
        size_t maxPkSym = 0;
        if (!executor.tryGetSymbol("maxPartkey", maxPkSym) || maxPkSym == 0) {
            std::cerr << "Q16 preprocessing: maxPartkey symbol unavailable\n";
            return false;
        }
        g_q16Post.maxPartkey = (int)(maxPkSym - 1);

        size_t complaintBmpInts = (maxSkSym + 31) / 32;
        registerFilledBuffer(device, executor, "d_q16_complaint_bitmap",
                     complaintBmpInts * sizeof(uint32_t));

        constexpr uint32_t kQ16ResultCap = 1u << 16;
        executor.registerScalarInt("q16_result_cap", (int)kQ16ResultCap);
        executor.registerSymbol("q16_result_cap", kQ16ResultCap);
        registerFilledBuffer(device, executor, "d_q16_filt_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q16_filt_idx",
                             nPart * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q16_filt_key",
                             nPart * sizeof(uint64_t));
        registerFilledBuffer(device, executor, "d_q16_part_group_map",
                             ((size_t)g_q16Post.maxPartkey + 1) * sizeof(int32_t), 0xFF);
        registerFilledBuffer(device, executor, "d_q16_group_brand",
                             (size_t)kQ16ResultCap * 10);
        registerFilledBuffer(device, executor, "d_q16_group_type",
                             (size_t)kQ16ResultCap * 25);
        registerFilledBuffer(device, executor, "d_q16_group_size",
                             (size_t)kQ16ResultCap * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q16_result_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q16_result_brand",
                             (size_t)kQ16ResultCap * 10);
        registerFilledBuffer(device, executor, "d_q16_result_type",
                             (size_t)kQ16ResultCap * 25);
        registerFilledBuffer(device, executor, "d_q16_result_size",
                             (size_t)kQ16ResultCap * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q16_result_supplier_cnt",
                             (size_t)kQ16ResultCap * sizeof(uint32_t));
        // d_q16_group_bitmaps and d_q16_group_counts are allocated after
        // the CPU dictionary hook reports q16_num_groups.
    }

    // Q3 compact output is sized by orderkey range.
    if (queryName == "Q3") {
        size_t maxOk = 0;
        executor.tryGetSymbol("maxOrderkey", maxOk);
        executor.registerSymbol("n_q3_oks", maxOk);
        executor.registerScalarInt("n_q3_oks", (int)maxOk);
        // Keep a small floor for tiny datasets.
        size_t cap = std::max<size_t>(maxOk, (size_t)(1u << 12));
        executor.registerScalarInt("q3_compact_cap", (int)cap);
        registerFilledBuffer(device, executor, "d_q3_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q3_compact_ok",
                             cap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q3_compact_rev",
                             cap * sizeof(float));
        registerFilledBuffer(device, executor, "d_q3_compact_date",
                             cap * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q3_compact_prio",
                             cap * sizeof(int32_t));
    }

    // Q10 compact output is sized by customer key range.
    if (queryName == "Q10") {
        size_t maxCk = 0;
        executor.tryGetSymbol("maxCustkey", maxCk);
        executor.registerSymbol("n_q10_cks", maxCk);
        executor.registerScalarInt("n_q10_cks", (int)maxCk);
        // Keep a small floor for tiny datasets.
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
        // Q18 GPU compaction replaces host order lookup/top-k prep.
        size_t maxOk = 0;
        executor.tryGetSymbol("maxOrderkey", maxOk);
        // n_q18_oks is both dispatch range and kernel loop bound.
        executor.registerSymbol("n_q18_oks", maxOk);
        executor.registerScalarInt("n_q18_oks", (int)maxOk);
        // Compact output is small; this is the buffer safety ceiling.
        constexpr uint32_t kQ18CompactCap = 1u << 20;
        executor.registerScalarInt("q18_compact_cap", (int)kQ18CompactCap);
        registerFilledBuffer(device, executor, "d_q18_compact_count",
                             sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_ok",
                             (size_t)kQ18CompactCap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_custkey",
                             (size_t)kQ18CompactCap * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_totalprice",
                             (size_t)kQ18CompactCap * sizeof(float));
        registerFilledBuffer(device, executor, "d_q18_compact_orderdate",
                             (size_t)kQ18CompactCap * sizeof(int32_t));
        registerFilledBuffer(device, executor, "d_q18_compact_qty",
                             (size_t)kQ18CompactCap * sizeof(float));
    }

    // Q21 setup: SAUDI nation key, scratch buffers, and compact result buffers.
    if (queryName == "Q21") {
        if (!registerNameKey(device, executor, loadedTables, "nation",
                             "SAUDI ARABIA", "sa_nk")) {
            return false;
        }

        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}}, loadedTables);
        const QueryColumns& sCols = sView.get();
        const size_t nSupplier = sCols.rows();

        size_t maxOkSym = 0, maxSkSym = 0;
        if (!executor.tryGetSymbol("maxOrderkey", maxOkSym) ||
            !executor.tryGetSymbol("maxSuppkey", maxSkSym)) {
            std::cerr << "Q21 preprocessing: maxOrderkey/maxSuppkey symbols unavailable\n";
            return false;
        }
        // Symbols are counts, not inclusive max keys.
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
        executor.registerScalarInt("q21_result_cap", (int)nSupplier);
        executor.registerSymbol("q21_result_cap", nSupplier);
        registerFilledBuffer(device, executor, "d_q21_result_count",
                     sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_result_name",
                     nSupplier * 25);
        registerFilledBuffer(device, executor, "d_q21_result_numwait",
                     nSupplier * sizeof(uint32_t));
    }

    return true;
}

} // namespace codegen
