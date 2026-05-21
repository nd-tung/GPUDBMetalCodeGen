#include "query_preprocessing.h"
#include "tpch_schema.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace codegen {

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

size_t nextPow2AtLeast(size_t value) {
    size_t out = 1;
    while (out < value) out <<= 1;
    return out;
}

size_t estimateFixedPrefixHashSlots(size_t inputRows,
                                    size_t fixedPrefixBytes,
                                    size_t probeRows,
                                    size_t minSlots = 1024) {
    // Fixed string prefixes are selective, but probe-heavy hash tables still
    // need low load factors because misses pay linear-probe cost.
    const size_t prefixDenom = fixedPrefixBytes >= 4 ? 64 : 16;
    const size_t probeHeavyGuard = probeRows >= inputRows * 4 ? 16 : 8;
    const size_t estimatedRows = std::max<size_t>(
        1, (inputRows + prefixDenom - 1) / prefixDenom);
    const size_t guardedRows = std::max(minSlots, estimatedRows * probeHeavyGuard);
    return nextPow2AtLeast(std::min(inputRows * 2, guardedRows));
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
        executor.registerSymbol("q13_hist_cap", kQ13HistBins);
        executor.registerScalarInt("q13_hist_cap", (int)kQ13HistBins);
        registerFilledBuffer(device, executor, "d_q13_result_count", sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q13_result_c_count",
                             kQ13HistBins * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q13_result_custdist",
                             kQ13HistBins * sizeof(uint32_t));
    }

    // Q9 host work: size the GPU partsupp direct map.
    if (queryName == "Q9") {
        size_t maxPartkey = 0;
        if (!executor.tryGetSymbol("maxPartkey", maxPartkey) || maxPartkey == 0) {
            std::cerr << "Q9 preprocessing: maxPartkey symbol unavailable\n";
            return false;
        }
        executor.registerSymbol("q9_ps_slots", maxPartkey * 4);
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
        size_t nPartSupp = 0, maxSk = 0, nSupplier = 0, nLineitem = 0;
        if (!executor.tryGetSymbol("n_partsupp", nPartSupp) || nPartSupp == 0) {
            std::cerr << "Q20 preprocessing: n_partsupp symbol unavailable\n";
            return false;
        }
        if (!executor.tryGetSymbol("n_lineitem", nLineitem) || nLineitem == 0) {
            nLineitem = nPartSupp;
        }
        if (!executor.tryGetSymbol("maxSuppkey", maxSk) || maxSk == 0) {
            std::cerr << "Q20 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        if (!executor.tryGetSymbol("n_supplier", nSupplier) || nSupplier == 0) {
            std::cerr << "Q20 preprocessing: n_supplier symbol unavailable\n";
            return false;
        }
        const size_t htSlots = estimateFixedPrefixHashSlots(nPartSupp, 6, nLineitem);
        executor.registerSymbol("q20HtSize", htSlots);
        executor.registerScalarInt("d_q20_ht_mask", (int)(htSlots - 1));
        executor.registerScalarInt("supp_mul", (int)maxSk);

        if (!registerNameKey(device, executor, loadedTables, "nation",
                             "CANADA", "canada_nk")) {
            return false;
        }

        // Size hash-table materialization by a fixed-prefix filter estimate.
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

    // Q2: GPU phases build the EUROPE supplier bitmap; preprocessing only
    // allocates reusable scratch/result buffers.
    if (queryName == "Q2") {
        size_t maxSkSym = 0, maxPkSym = 0;
        if (!executor.tryGetSymbol("maxSuppkey", maxSkSym) || maxSkSym == 0 ||
            !executor.tryGetSymbol("maxPartkey", maxPkSym) || maxPkSym == 0) {
            std::cerr << "Q2 preprocessing: key-domain symbols unavailable\n";
            return false;
        }

        const size_t maxPk = maxPkSym - 1;
        size_t partBmpInts = (maxPk + 32) / 32;
        registerFilledBuffer(device, executor, "d_q2_part_bitmap",
                     partBmpInts * sizeof(uint32_t));

        size_t minCostSize = maxPk + 1;
        registerFilledBuffer(device, executor, "d_q2_min_cost",
                            minCostSize * sizeof(uint32_t), 0xFF);

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

    // Q16: finite group-domain metadata. The plan builds partkey->group and
    // group labels on GPU from the part table.
    if (queryName == "Q16") {
        size_t maxSkSym = 0;
        if (!executor.tryGetSymbol("maxSuppkey", maxSkSym) || maxSkSym == 0) {
            std::cerr << "Q16 preprocessing: maxSuppkey symbol unavailable\n";
            return false;
        }
        size_t maxPkSym = 0;
        if (!executor.tryGetSymbol("maxPartkey", maxPkSym) || maxPkSym == 0) {
            std::cerr << "Q16 preprocessing: maxPartkey symbol unavailable\n";
            return false;
        }

        size_t complaintBmpInts = (maxSkSym + 31) / 32;
        registerFilledBuffer(device, executor, "d_q16_complaint_bitmap",
                     complaintBmpInts * sizeof(uint32_t));

        constexpr uint32_t kQ16ResultCap = 1u << 16;
        constexpr uint32_t kQ16GroupCap = 24u * 145u * 8u;
        uint32_t bvInts = static_cast<uint32_t>((maxSkSym + 31) / 32);
        size_t popWords = static_cast<size_t>(kQ16GroupCap) * bvInts;
        executor.registerScalarInt("q16_result_cap", (int)kQ16ResultCap);
        executor.registerSymbol("q16_result_cap", kQ16ResultCap);
        executor.registerScalarInt("q16_group_cap", (int)kQ16GroupCap);
        executor.registerSymbol("q16_group_cap", kQ16GroupCap);
        executor.registerScalarInt("d_q16_bv_ints", (int)bvInts);
        executor.registerSymbol("d_q16_bv_ints", bvInts);
        executor.registerSymbol("q16_num_groups", kQ16GroupCap);
        executor.registerSymbol("n_q16_num_groups", kQ16GroupCap);
        executor.registerScalarInt("n_q16_num_groups", (int)kQ16GroupCap);
        executor.registerSymbol("q16_pop_words", popWords);
        executor.registerSymbol("n_q16_pop_words", popWords);
        executor.registerScalarInt("n_q16_pop_words", (int)popWords);
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
        registerFilledBuffer(device, executor, "d_q10_key_customer_idx",
                             cap * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q10_key_nation_idx",
                             cap * sizeof(uint32_t));

        constexpr uint32_t kQ10LateLimit = 20;
        executor.registerScalarInt("q10_late_limit", (int)kQ10LateLimit);
        executor.registerSymbol("q10_late_limit", kQ10LateLimit);
        registerFilledBuffer(device, executor, "d_q10_result_ck",
                             (size_t)kQ10LateLimit * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q10_result_name",
                             (size_t)kQ10LateLimit * 25);
        registerFilledBuffer(device, executor, "d_q10_result_rev",
                             (size_t)kQ10LateLimit * sizeof(float));
        registerFilledBuffer(device, executor, "d_q10_result_acctbal",
                             (size_t)kQ10LateLimit * sizeof(float));
        registerFilledBuffer(device, executor, "d_q10_result_n_name",
                             (size_t)kQ10LateLimit * 25);
        registerFilledBuffer(device, executor, "d_q10_result_address",
                             (size_t)kQ10LateLimit * 40);
        registerFilledBuffer(device, executor, "d_q10_result_phone",
                             (size_t)kQ10LateLimit * 15);
        registerFilledBuffer(device, executor, "d_q10_result_comment",
                             (size_t)kQ10LateLimit * 117);
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
        registerFilledBuffer(device, executor, "d_q18_compact_name",
                             (size_t)kQ18CompactCap * 25);
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
