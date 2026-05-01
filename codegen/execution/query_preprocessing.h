#pragma once

#include "../../src/infra.h"
#include "metal_generic_executor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

using LoadedQueryTable = std::pair<std::string, QueryColumns>;

struct Q20PostData {
    std::vector<int> ps_partkey, ps_suppkey, ps_availqty;
    std::vector<uint64_t> htKeys;
    std::vector<int> htPsIdx;
    uint32_t htMask = 0, htSlots = 0;
    std::vector<int> s_suppkey, s_nationkey;
    std::vector<char> s_name, s_address;
    int canada_nk = -1;
    MTL::Buffer* htValsBuf = nullptr;
};

struct Q2PostData {
    std::vector<int> ps_partkey, ps_suppkey;
    std::vector<float> ps_supplycost;
    std::vector<int> s_suppkey, s_nationkey;
    std::vector<float> s_acctbal;
    std::vector<char> s_name, s_address, s_phone, s_comment;
    std::vector<int> p_partkey;
    std::vector<char> p_mfgr;
    std::vector<std::string> nationNames;
    std::vector<uint32_t> eurSuppBitmap;
    int maxPartkey = 0;
    MTL::Buffer* minCostBuf = nullptr;
};

struct Q16PostData {
    struct GroupKey { std::string brand; std::string type; int size; };
    std::vector<GroupKey> groups;
    uint32_t bvInts = 0;
    uint32_t numGroups = 0;
    MTL::Buffer* groupBitmapsBuf = nullptr;
};

struct Q21PostData {
    std::vector<int> s_suppkey;
    std::vector<char> s_name;
    int maxSuppkey = 0;
};

struct Q18PostData {
    std::vector<int> o_custkey, o_orderdate;
    std::vector<float> o_totalprice;
    std::vector<int> okLookup;
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
