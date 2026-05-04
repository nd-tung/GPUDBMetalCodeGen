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
    memset(buffer->contents(), fillByte, allocBytes);
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

    // Q22: compute avg balance for valid-prefix customers
    if (queryName == "Q22") {
        for (auto& [tblName, cols] : loadedTables) {
            if (tblName == "customer") {
                const auto& tdef = TPCHSchema::instance().table("customer");
                const char*  phoneCols = cols.chars(tdef.col("c_phone").index);
                const float* balCols   = cols.floats(tdef.col("c_acctbal").index);
                size_t custCount = cols.rows();
                double sumBal = 0.0;
                int countBal = 0;
                for (size_t j = 0; j < custCount; j++) {
                    int prefix = (phoneCols[j * 15] - '0') * 10 + (phoneCols[j * 15 + 1] - '0');
                    if (balCols[j] > 0.0f &&
                        (prefix == 13 || prefix == 17 || prefix == 18 ||
                         prefix == 23 || prefix == 29 || prefix == 30 || prefix == 31)) {
                        sumBal += balCols[j];
                        countBal++;
                    }
                }
                float avgBal = (countBal > 0) ? (float)(sumBal / countBal) : 0.0f;
                executor.registerScalarFloat("avg_bal", avgBal);
                break;
            }
        }
    }

    // Q11: resolve GERMANY nationkey
    if (queryName == "Q11") {
        if (!registerNameKey(device, executor, loadedTables, "nation", "GERMANY", "germany_nk")) return false;
    }

    // Q17: build per-partkey threshold from loaded data
    if (queryName == "Q17") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {6, ColType::CHAR_FIXED, 10}},
            loadedTables);
        const auto& pCols = pView.get();
        auto p_partkey = copyIntColumn(pCols, 0);
        auto p_brand = copyCharColumn(pCols, 3, pCols.rows() * 10);
        auto p_container = copyCharColumn(pCols, 6, pCols.rows() * 10);
        size_t partRows = p_partkey.size();

        // lineitem may be the stream table in chunked mode and therefore
        // absent from loadedTables. resolvePreprocessColumns falls back to
        // a one-time disk load (l_partkey + l_quantity = ~960 MB at SF20)
        // which is freed when pView_li goes out of scope at end of block.
        const auto& tpchSchema = TPCHSchema::instance();
        const int liPartkeyIdx  = tpchSchema.table("lineitem").col("l_partkey").index;
        const int liQuantityIdx = tpchSchema.table("lineitem").col("l_quantity").index;
        auto liView = resolvePreprocessColumns(device, "lineitem",
            {{liPartkeyIdx, ColType::INT}, {liQuantityIdx, ColType::FLOAT}},
            loadedTables);
        const auto& liCols = liView.get();
        const int*   pl_partkey  = liCols.ints  (liPartkeyIdx);
        const float* pl_quantity = liCols.floats(liQuantityIdx);
        size_t liRows = liCols.rows();
        if (!pl_partkey || !pl_quantity) {
            std::cerr << "Q17 preprocessing: failed to obtain l_partkey/l_quantity\n";
            return false;
        }

        int maxPk = 0;
        for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
        size_t mapSize = (size_t)(maxPk + 1);

        std::vector<uint32_t> bitmap((mapSize + 31) / 32, 0);
        for (size_t i = 0; i < partRows; i++) {
            const char* brand = p_brand.data() + i * 10;
            const char* cont = p_container.data() + i * 10;
            bool isBrand23 = (brand[0]=='B' && brand[5]=='#' && brand[6]=='2' && brand[7]=='3');
            bool isMedBox = (cont[0]=='M' && cont[1]=='E' && cont[2]=='D' && cont[3]==' ' &&
                             cont[4]=='B' && cont[5]=='O' && cont[6]=='X');
            if (isBrand23 && isMedBox) {
                int pk = p_partkey[i];
                bitmap[pk / 32] |= (1u << (pk % 32));
            }
        }

        std::vector<double> sumQty(mapSize, 0.0);
        std::vector<int> cntQty(mapSize, 0);
        for (size_t i = 0; i < liRows; i++) {
            int pk = pl_partkey[i];
            if (pk >= 0 && (size_t)pk < mapSize && (bitmap[pk / 32] >> (pk % 32)) & 1) {
                sumQty[pk] += pl_quantity[i];
                cntQty[pk]++;
            }
        }

        std::vector<float> threshold(mapSize, 0.0f);
        for (size_t pk = 0; pk < mapSize; pk++) {
            if (cntQty[pk] > 0) {
                threshold[pk] = (float)(0.2 * sumQty[pk] / cntQty[pk]);
            }
        }

        uploadAndRegister(device, executor, "d_q17_threshold", threshold);
        uploadAndRegister(device, executor, "d_q17_bitmap", bitmap);
    }

    // Q9: build green-parts bitmap, lookup arrays, and partsupp HT
    if (queryName == "Q9") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}}, loadedTables);
        const auto& pCols = pView.get();
        auto p_partkey = copyIntColumn(pCols, 0);
        auto p_name = copyCharColumn(pCols, 1, pCols.rows() * 55);

        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}, {3, ColType::INT}}, loadedTables);
        const auto& sCols = sView.get();
        auto s_suppkey = copyIntColumn(sCols, 0);
        auto s_nationkey = copyIntColumn(sCols, 3);

        auto oView = resolvePreprocessColumns(device, "orders",
            {{0, ColType::INT}, {4, ColType::DATE}}, loadedTables);
        const auto& oCols = oView.get();
        auto o_orderkey = copyIntColumn(oCols, 0);
        auto o_orderdate = copyIntColumn(oCols, 4);

        auto psView = resolvePreprocessColumns(device, "partsupp",
            {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}}, loadedTables);
        const auto& psCols = psView.get();
        auto ps_partkey = copyIntColumn(psCols, 0);
        auto ps_suppkey = copyIntColumn(psCols, 1);
        auto ps_supplycost = copyFloatColumn(psCols, 3);

        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& nCols = nView.get();
        auto n_nationkey = copyIntColumn(nCols, 0);
        auto n_name = copyCharColumn(nCols, 1, nCols.rows() * 25);

        std::vector<std::string> nationNames(25);
        for (size_t i = 0; i < n_nationkey.size(); i++) {
            std::string nm;
            for (int c = 0; c < 25; c++) {
                char ch = n_name[i * 25 + c];
                if (ch == ' ' || ch == '\0') break;
                nm += ch;
            }
            nationNames[n_nationkey[i]] = nm;
        }

        int maxPk = 0;
        for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
        size_t bmpInts = ((size_t)maxPk + 32) / 32;
        std::vector<uint32_t> partBitmap(bmpInts, 0);
        for (size_t i = 0; i < p_partkey.size(); i++) {
            bool found = false;
            for (int c = 0; c <= 50; c++) {
                if (p_name[i * 55 + c] == 'g' && p_name[i * 55 + c + 1] == 'r' &&
                    p_name[i * 55 + c + 2] == 'e' && p_name[i * 55 + c + 3] == 'e' &&
                    p_name[i * 55 + c + 4] == 'n') { found = true; break; }
            }
            if (found) {
                int pk = p_partkey[i];
                partBitmap[pk / 32] |= (1u << (pk % 32));
            }
        }

        int maxSk = 0;
        for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
        std::vector<int> sNatArray((size_t)maxSk + 1, -1);
        for (size_t i = 0; i < s_suppkey.size(); i++)
            sNatArray[s_suppkey[i]] = s_nationkey[i];

        int maxOk = 0;
        for (int ok : o_orderkey) maxOk = std::max(maxOk, ok);
        std::vector<int> oYearArray((size_t)maxOk + 1, 0);
        for (size_t i = 0; i < o_orderkey.size(); i++)
            oYearArray[o_orderkey[i]] = o_orderdate[i] / 10000;

        size_t psEntries = 0;
        for (size_t i = 0; i < ps_partkey.size(); i++) {
            int pk = ps_partkey[i];
            if (pk >= 0 && (size_t)pk / 32 < bmpInts && (partBitmap[pk / 32] >> (pk % 32)) & 1)
                psEntries++;
        }
        uint32_t htSlots = 1;
        while (htSlots < psEntries * 2) htSlots <<= 1;
        uint32_t htMask = htSlots - 1;
        uint32_t suppMul = (uint32_t)(maxSk + 1);
        std::vector<uint32_t> htKeys(htSlots, 0xFFFFFFFFu);
        std::vector<float> htVals(htSlots, 0.0f);
        for (size_t i = 0; i < ps_partkey.size(); i++) {
            int pk = ps_partkey[i];
            if (pk < 0 || (size_t)pk / 32 >= bmpInts || !((partBitmap[pk / 32] >> (pk % 32)) & 1))
                continue;
            uint32_t key = (uint32_t)pk * suppMul + (uint32_t)ps_suppkey[i];
            uint32_t h = (key * kKnuthHashMul) & htMask;
            for (uint32_t s = 0; s <= htMask; s++) {
                uint32_t slot = (h + s) & htMask;
                if (htKeys[slot] == 0xFFFFFFFFu) {
                    htKeys[slot] = key;
                    htVals[slot] = ps_supplycost[i];
                    break;
                }
            }
        }

        uploadAndRegister(device, executor, "d_q9_part_bitmap", partBitmap);
        uploadAndRegister(device, executor, "d_q9_s_nationkey", sNatArray);
        uploadAndRegister(device, executor, "d_q9_o_year", oYearArray);
        uploadAndRegister(device, executor, "d_ps_ht_keys", htKeys);
        uploadAndRegister(device, executor, "d_ps_ht_vals", htVals);
        executor.registerScalarInt("d_ps_ht_mask", (int)htMask);
        executor.registerScalarInt("supp_mul", (int)suppMul);

        static std::vector<std::string> q9NationNames;
        q9NationNames = std::move(nationNames);
    }

    // Q20: build forest% bitmap, partsupp HT, CANADA filter
    if (queryName == "Q20") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}}, loadedTables);
        const auto& pCols = pView.get();
        auto p_partkey = copyIntColumn(pCols, 0);
        auto p_name = copyCharColumn(pCols, 1, pCols.rows() * 55);

        auto psView = resolvePreprocessColumns(device, "partsupp",
            {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}}, loadedTables);
        const auto& psCols = psView.get();
        auto ps_partkey = copyIntColumn(psCols, 0);
        auto ps_suppkey = copyIntColumn(psCols, 1);
        auto ps_availqty = copyIntColumn(psCols, 2);

        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT}},
            loadedTables);
        const auto& sCols = sView.get();
        auto s_suppkey = copyIntColumn(sCols, 0);
        auto s_name = copyCharColumn(sCols, 1, sCols.rows() * 25);
        auto s_address = copyCharColumn(sCols, 2, sCols.rows() * 40);
        auto s_nationkey = copyIntColumn(sCols, 3);

        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& nCols = nView.get();
        auto n_nationkey = copyIntColumn(nCols, 0);
        auto n_name = copyCharColumn(nCols, 1, nCols.rows() * 25);

        int canada_nk = -1;
        for (size_t i = 0; i < n_nationkey.size(); i++) {
            if (n_name[i * 25] == 'C' && n_name[i * 25 + 1] == 'A' &&
                n_name[i * 25 + 2] == 'N') {
                canada_nk = n_nationkey[i];
                break;
            }
        }

        int maxPk = 0;
        for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
        size_t bmpInts = ((size_t)maxPk + 32) / 32;
        std::vector<uint32_t> partBitmap(bmpInts, 0);
        for (size_t i = 0; i < p_partkey.size(); i++) {
            const char* nm = p_name.data() + i * 55;
            if (nm[0]=='f' && nm[1]=='o' && nm[2]=='r' && nm[3]=='e' &&
                nm[4]=='s' && nm[5]=='t') {
                partBitmap[p_partkey[i] / 32] |= (1u << (p_partkey[i] % 32));
            }
        }

        size_t psEntries = 0;
        for (size_t i = 0; i < ps_partkey.size(); i++) {
            int pk = ps_partkey[i];
            if (pk >= 0 && (size_t)pk / 32 < bmpInts && (partBitmap[pk / 32] >> (pk % 32)) & 1)
                psEntries++;
        }
        uint32_t htSlots = 1;
        while (htSlots < psEntries * 2) htSlots <<= 1;
        uint32_t htMask = htSlots - 1;
        int maxSk = 0;
        for (int sk : ps_suppkey) maxSk = std::max(maxSk, sk);
        uint32_t suppMul = (uint32_t)(maxSk + 1);
        // 64-bit packed key: at SF20 maxPk*suppMul ~ 8e11 overflows uint32.
        std::vector<uint64_t> htKeys(htSlots, ~uint64_t(0));
        std::vector<int> htPsIdx(htSlots, -1);
        for (size_t i = 0; i < ps_partkey.size(); i++) {
            int pk = ps_partkey[i];
            if (pk < 0 || (size_t)pk / 32 >= bmpInts || !((partBitmap[pk / 32] >> (pk % 32)) & 1))
                continue;
            uint64_t key = (uint64_t)pk * (uint64_t)suppMul + (uint64_t)ps_suppkey[i];
            uint32_t h = ((uint32_t)(key ^ (key >> 32)) * kKnuthHashMul) & htMask;
            for (uint32_t s = 0; s <= htMask; s++) {
                uint32_t slot = (h + s) & htMask;
                if (htKeys[slot] == ~uint64_t(0)) {
                    htKeys[slot] = key;
                    htPsIdx[slot] = (int)i;
                    break;
                }
            }
        }

        uploadAndRegister(device, executor, "d_q20_part_bitmap", partBitmap);
        uploadAndRegister(device, executor, "d_q20_ht_keys", htKeys);
        auto* htValsBuf = registerFilledBuffer(device, executor, "d_q20_ht_vals",
                               htSlots * sizeof(float));
        executor.registerScalarInt("d_q20_ht_mask", (int)htMask);
        executor.registerScalarInt("supp_mul", (int)suppMul);

        g_q20Post.ps_partkey = std::move(ps_partkey);
        g_q20Post.ps_suppkey = std::move(ps_suppkey);
        g_q20Post.ps_availqty = std::move(ps_availqty);
        g_q20Post.htKeys = std::move(htKeys);
        g_q20Post.htPsIdx = std::move(htPsIdx);
        g_q20Post.htMask = htMask;
        g_q20Post.htSlots = htSlots;
        g_q20Post.s_suppkey = std::move(s_suppkey);
        g_q20Post.s_nationkey = std::move(s_nationkey);
        g_q20Post.s_name = std::move(s_name);
        g_q20Post.s_address = std::move(s_address);
        g_q20Post.canada_nk = canada_nk;
        g_q20Post.htValsBuf = htValsBuf;
    }

    // Q2: build EUROPE supplier bitmap, allocate part bitmap (filled by GPU Phase 1)
    if (queryName == "Q2") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {2, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& pCols = pView.get();
        auto p_partkey = copyIntColumn(pCols, 0);
        auto p_mfgr = copyCharColumn(pCols, 2, pCols.rows() * 25);

        auto sView = resolvePreprocessColumns(device, "supplier", {
            {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
            {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT},
            {6, ColType::CHAR_FIXED, 101}
        }, loadedTables);
        const auto& sCols = sView.get();
        auto s_suppkey = copyIntColumn(sCols, 0);
        auto s_name = copyCharColumn(sCols, 1, sCols.rows() * 25);
        auto s_address = copyCharColumn(sCols, 2, sCols.rows() * 40);
        auto s_nationkey = copyIntColumn(sCols, 3);
        auto s_phone = copyCharColumn(sCols, 4, sCols.rows() * 15);
        auto s_acctbal = copyFloatColumn(sCols, 5);
        auto s_comment = copyCharColumn(sCols, 6, sCols.rows() * 101);

        auto psView = resolvePreprocessColumns(device, "partsupp",
            {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}}, loadedTables);
        const auto& psCols = psView.get();
        auto ps_partkey = copyIntColumn(psCols, 0);
        auto ps_suppkey = copyIntColumn(psCols, 1);
        auto ps_supplycost = copyFloatColumn(psCols, 3);

        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::INT}}, loadedTables);
        const auto& nCols = nView.get();
        auto n_nationkey = copyIntColumn(nCols, 0);
        auto n_name = copyCharColumn(nCols, 1, nCols.rows() * 25);
        auto n_regionkey = copyIntColumn(nCols, 2);

        auto rView = resolvePreprocessColumns(device, "region",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& rCols = rView.get();
        auto r_regionkey = copyIntColumn(rCols, 0);
        auto r_name = copyCharColumn(rCols, 1, rCols.rows() * 25);

        int europe_rk = -1;
        for (size_t i = 0; i < r_regionkey.size(); i++) {
            if (r_name[i * 25] == 'E' && r_name[i * 25 + 1] == 'U' &&
                r_name[i * 25 + 2] == 'R' && r_name[i * 25 + 3] == 'O') {
                europe_rk = r_regionkey[i];
                break;
            }
        }

        std::set<int> europeNks;
        for (size_t i = 0; i < n_nationkey.size(); i++) {
            if (n_regionkey[i] == europe_rk)
                europeNks.insert(n_nationkey[i]);
        }

        std::vector<std::string> nationNames(25);
        for (size_t i = 0; i < n_nationkey.size(); i++) {
            int len = 25;
            while (len > 0 && (n_name[i * 25 + len - 1] == ' ' || n_name[i * 25 + len - 1] == '\0')) len--;
            nationNames[n_nationkey[i]] = std::string(n_name.data() + i * 25, len);
        }

        int maxSk = 0;
        for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
        size_t suppBmpInts = ((size_t)maxSk + 32) / 32;
        std::vector<uint32_t> eurSuppBitmap(suppBmpInts, 0);
        for (size_t i = 0; i < s_suppkey.size(); i++) {
            if (europeNks.count(s_nationkey[i])) {
                int sk = s_suppkey[i];
                eurSuppBitmap[sk / 32] |= (1u << (sk % 32));
            }
        }

        int maxPk = 0;
        for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
        size_t partBmpInts = ((size_t)maxPk + 32) / 32;

        registerFilledBuffer(device, executor, "d_q2_part_bitmap",
                     partBmpInts * sizeof(uint32_t));

        size_t minCostSize = (size_t)maxPk + 1;
        auto* minCostBuf = registerFilledBuffer(device, executor, "d_q2_min_cost",
                            minCostSize * sizeof(uint32_t), 0xFF);
        uploadAndRegister(device, executor, "d_q2_supp_bitmap", eurSuppBitmap);

        g_q2Post.ps_partkey = std::move(ps_partkey);
        g_q2Post.ps_suppkey = std::move(ps_suppkey);
        g_q2Post.ps_supplycost = std::move(ps_supplycost);
        g_q2Post.s_suppkey = std::move(s_suppkey);
        g_q2Post.s_nationkey = std::move(s_nationkey);
        g_q2Post.s_acctbal = std::move(s_acctbal);
        g_q2Post.s_name = std::move(s_name);
        g_q2Post.s_address = std::move(s_address);
        g_q2Post.s_phone = std::move(s_phone);
        g_q2Post.s_comment = std::move(s_comment);
        g_q2Post.p_partkey = std::move(p_partkey);
        g_q2Post.p_mfgr = std::move(p_mfgr);
        g_q2Post.nationNames = std::move(nationNames);
        g_q2Post.eurSuppBitmap = std::move(eurSuppBitmap);
        g_q2Post.maxPartkey = maxPk;
        g_q2Post.minCostBuf = minCostBuf;
    }

    // Q16: build part_group_map, allocate complaint bitmap (filled by GPU Phase 1)
    if (queryName == "Q16") {
        auto pView = resolvePreprocessColumns(device, "part",
            {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}},
            loadedTables);
        const auto& pCols = pView.get();
        auto p_partkey = copyIntColumn(pCols, 0);
        auto p_brand = copyCharColumn(pCols, 3, pCols.rows() * 10);
        auto p_type = copyCharColumn(pCols, 4, pCols.rows() * 25);
        auto p_size = copyIntColumn(pCols, 5);

        // Get maxSk from loaded supplier data (standard mechanism loads it for Phase 1)
        int maxSk = 0;
        const auto& tpchSchema3 = TPCHSchema::instance();
        for (auto& [tblName, cols] : loadedTables) {
            if (tblName == "supplier") {
                const int* skCol = cols.ints(tpchSchema3.table("supplier").col("s_suppkey").index);
                size_t n = cols.rows();
                if (skCol) for (size_t i = 0; i < n; ++i) maxSk = std::max(maxSk, skCol[i]);
            }
        }
        size_t complaintBmpInts = ((size_t)maxSk + 32) / 32;

        registerFilledBuffer(device, executor, "d_q16_complaint_bitmap",
                     complaintBmpInts * sizeof(uint32_t));

        std::set<int> validSizes = {49, 14, 23, 45, 19, 3, 36, 9};
        struct GroupKey { std::string brand; std::string type; int size;
            bool operator<(const GroupKey& o) const {
                if (brand != o.brand) return brand < o.brand;
                if (type != o.type) return type < o.type;
                return size < o.size;
            }
        };
        std::map<GroupKey, int> groupMap;
        std::vector<Q16PostData::GroupKey> groups;

        int maxPk = 0;
        for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
        std::vector<int> partGroupMap((size_t)maxPk + 1, -1);

        for (size_t i = 0; i < p_partkey.size(); i++) {
            const char* br = p_brand.data() + i * 10;
            int brLen = 10;
            while (brLen > 0 && (br[brLen-1] == ' ' || br[brLen-1] == '\0')) brLen--;
            std::string brand(br, brLen);

            const char* tp = p_type.data() + i * 25;
            int tpLen = 25;
            while (tpLen > 0 && (tp[tpLen-1] == ' ' || tp[tpLen-1] == '\0')) tpLen--;
            std::string type(tp, tpLen);

            int size = p_size[i];

            if (brand == "Brand#45") continue;
            if (tpLen >= 15 && tp[0]=='M' && tp[1]=='E' && tp[2]=='D' && tp[3]=='I' &&
                tp[4]=='U' && tp[5]=='M' && tp[6]==' ' && tp[7]=='P' && tp[8]=='O' &&
                tp[9]=='L' && tp[10]=='I' && tp[11]=='S' && tp[12]=='H' && tp[13]=='E' &&
                tp[14]=='D') continue;
            if (!validSizes.count(size)) continue;

            GroupKey gk{brand, type, size};
            auto it = groupMap.find(gk);
            int gid;
            if (it == groupMap.end()) {
                gid = (int)groups.size();
                groupMap[gk] = gid;
                groups.push_back({brand, type, size});
            } else {
                gid = it->second;
            }
            partGroupMap[p_partkey[i]] = gid;
        }

        uint32_t numGroups = (uint32_t)groups.size();
        uint32_t bvInts = ((uint32_t)maxSk + 32) / 32;

        uploadAndRegister(device, executor, "d_q16_part_group_map", partGroupMap);
        size_t gbmBytes = (size_t)numGroups * bvInts * sizeof(uint32_t);
        auto* gbmBuf = registerFilledBuffer(device, executor, "d_q16_group_bitmaps", gbmBytes);
        executor.registerScalarInt("d_q16_bv_ints", (int)bvInts);

        g_q16Post.groups = std::move(groups);
        g_q16Post.bvInts = bvInts;
        g_q16Post.numGroups = numGroups;
        g_q16Post.groupBitmapsBuf = gbmBuf;
    }

    // Q18: preload orders.tbl for post-processing (avoids reloading during post)
    if (queryName == "Q18") {
        auto oView = resolvePreprocessColumns(device, "orders",
            {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}, {4, ColType::DATE}},
            loadedTables);
        const auto& oCols = oView.get();
        auto okeys = copyIntColumn(oCols, 0);
        g_q18Post.o_custkey = copyIntColumn(oCols, 1);
        g_q18Post.o_totalprice = copyFloatColumn(oCols, 3);
        g_q18Post.o_orderdate = copyIntColumn(oCols, 4);
        // Build direct-mapped orderkey -> row index
        int maxOk = 0;
        for (int k : okeys) if (k > maxOk) maxOk = k;
        g_q18Post.okLookup.assign(maxOk + 1, -1);
        for (size_t j = 0; j < okeys.size(); j++)
            g_q18Post.okLookup[okeys[j]] = (int)j;
    }

    // Q21: allocate GPU buffers for phases, build SA-supp bitmap (tiny)
    if (queryName == "Q21") {
        // Load supplier/nation for SA bitmap + post-processing (small tables)
        auto sView = resolvePreprocessColumns(device, "supplier",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {3, ColType::INT}}, loadedTables);
        const auto& sCols = sView.get();
        auto s_suppkey = copyIntColumn(sCols, 0);
        auto s_name = copyCharColumn(sCols, 1, sCols.rows() * 25);
        auto s_nationkey = copyIntColumn(sCols, 3);

        auto nView = resolvePreprocessColumns(device, "nation",
            {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}}, loadedTables);
        const auto& nCols = nView.get();
        auto n_nationkey = copyIntColumn(nCols, 0);
        auto n_name = copyCharColumn(nCols, 1, nCols.rows() * 25);

        int sa_nk = -1;
        for (size_t i = 0; i < n_nationkey.size(); i++) {
            if (n_name[i * 25] == 'S' && n_name[i * 25 + 1] == 'A' &&
                n_name[i * 25 + 2] == 'U' && n_name[i * 25 + 3] == 'D') {
                sa_nk = n_nationkey[i];
                break;
            }
        }

        int maxSk = 0;
        for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
        size_t saBmpInts = ((size_t)maxSk + 32) / 32;
        std::vector<uint32_t> saBitmap(saBmpInts, 0);
        for (size_t i = 0; i < s_suppkey.size(); i++) {
            if (s_nationkey[i] == sa_nk) {
                int sk = s_suppkey[i];
                saBitmap[sk / 32] |= (1u << (sk % 32));
            }
        }

        uploadAndRegister(device, executor, "d_q21_sa_supp", saBitmap);

        // Find max orderkey from loaded orders data for buffer sizing
        int maxOk = 0;
        const auto& tpchSchema2 = TPCHSchema::instance();
        for (auto& [tblName, cols] : loadedTables) {
            if (tblName == "orders") {
                const int* okCol = cols.ints(tpchSchema2.table("orders").col("o_orderkey").index);
                size_t n = cols.rows();
                if (okCol) for (size_t i = 0; i < n; ++i) maxOk = std::max(maxOk, okCol[i]);
            }
        }

        size_t fBmpInts = ((size_t)maxOk + 32) / 32;
        size_t okMapSize = (size_t)maxOk + 1;

        registerFilledBuffer(device, executor, "d_q21_f_orders", fBmpInts * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_first_supp", okMapSize * sizeof(int), 0xFF);
        registerFilledBuffer(device, executor, "d_q21_first_late", okMapSize * sizeof(int), 0xFF);
        registerFilledBuffer(device, executor, "d_q21_multi_supp", fBmpInts * sizeof(uint32_t));
        registerFilledBuffer(device, executor, "d_q21_multi_late", fBmpInts * sizeof(uint32_t));

        size_t suppCountSize = (size_t)maxSk + 1;
        registerFilledBuffer(device, executor, "d_q21_supp_count",
                     suppCountSize * sizeof(uint32_t));

        g_q21Post.s_suppkey = std::move(s_suppkey);
        g_q21Post.s_name = std::move(s_name);
        g_q21Post.maxSuppkey = maxSk;
    }

    return true;
}

} // namespace codegen
