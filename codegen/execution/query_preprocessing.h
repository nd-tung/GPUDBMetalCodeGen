#pragma once

#include "../core/infra.h"
#include "metal_generic_executor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

using LoadedQueryTable = std::pair<std::string, QueryColumns>;

struct Q20PostData {
    // Borrowed pointers (mmap views).
    const int* ps_partkey = nullptr;
    const int* ps_suppkey = nullptr;
    const int* ps_availqty = nullptr;
    size_t nPS = 0;

    const int*  s_suppkey = nullptr;
    const int*  s_nationkey = nullptr;
    const char* s_name = nullptr;     // width 25
    const char* s_address = nullptr;  // width 40
    size_t nS = 0;

    uint32_t htMask = 0, htSlots = 0;
    int canada_nk = -1;

    QueryColumns ownedPartsupp, ownedSupplier;
};

struct Q2PostData {
    // Borrowed pointers (mmap views). See Q18 note on lifetime.
    const int*   ps_partkey = nullptr;
    const int*   ps_suppkey = nullptr;
    const float* ps_supplycost = nullptr;
    size_t nPS = 0;

    const int*   s_suppkey = nullptr;
    const int*   s_nationkey = nullptr;
    const float* s_acctbal = nullptr;
    const char*  s_name = nullptr;
    const char*  s_address = nullptr;
    const char*  s_phone = nullptr;
    const char*  s_comment = nullptr;
    size_t nS = 0;

    const int*  p_partkey = nullptr;
    const char* p_mfgr = nullptr;
    size_t nP = 0;

    std::vector<std::string> nationNames;
    int maxPartkey = 0;
    int maxSuppkey = 0;
    // Direct-address lookups: index by key, -1 if absent. Built once
    // during preprocessing to replace per-query unordered_map<int,int>
    // builds (a 4M-row hash insert at SF20 dominated CPU time).
    std::vector<int> partIdxArr;
    std::vector<int> suppIdxArr;

    // Backing storage when the table wasn't already in loadedTables.
    QueryColumns ownedPart, ownedSupplier, ownedPartsupp;
};

struct Q16PostData {
    struct GroupKey { std::string brand; std::string type; int size; };
    std::vector<GroupKey> groups;
    uint32_t numGroups = 0;
    // Inputs cached for the Q16_filter_compact post-dispatch hook,
    // which reads the GPU-emitted compact (idx, key) list and builds
    // groups[] + d_q16_part_group_map on host.
    const int*  p_partkey = nullptr;
    const char* p_brand = nullptr;   // width 10
    const char* p_type  = nullptr;   // width 25
    size_t nPart = 0;
    int    maxPartkey = 0;
    int    maxSk = 0;
    QueryColumns ownedPart;          // empty if borrowed
};

struct Q21PostData {
    const int*  s_suppkey = nullptr;
    const char* s_name = nullptr;  // width 25
    size_t nS = 0;
    QueryColumns ownedSupplier;
};

struct Q18PostData {
    // Borrowed pointers into orders columns (mmap view). Backing storage
    // is either loadedTables (in main) or `ownedOrders` below when the
    // view had to be loaded fresh by preprocessing.
    const int*   o_custkey = nullptr;
    const int*   o_orderdate = nullptr;
    const float* o_totalprice = nullptr;
    QueryColumns ownedOrders;  // empty if borrowed
    // The orderkey -> row-index lookup is built by the
    // Q18_build_ok_lookup GPU phase into d_q18_ok_lookup.
};

extern Q20PostData g_q20Post;
extern Q2PostData g_q2Post;
extern Q16PostData g_q16Post;
extern Q21PostData g_q21Post;
extern Q18PostData g_q18Post;

QueryColumns loadPreprocessColumns(MTL::Device* device,
                                   const std::string& tableName,
                                   const std::vector<ColSpec>& specs);
std::vector<int> copyIntColumn(const QueryColumns& columns, int columnIndex);
std::vector<float> copyFloatColumn(const QueryColumns& columns, int columnIndex);
std::vector<char> copyCharColumn(const QueryColumns& columns, int columnIndex, size_t byteCount);

bool prepareQueryPreprocessing(const std::string& queryName,
                               MTL::Device* device,
                               MetalGenericExecutor& executor,
                               const std::vector<LoadedQueryTable>& loadedTables);

} // namespace codegen
