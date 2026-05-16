#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"
#include "execution/query_preprocessing.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace codegen {

// Q16: Parts/Supplier Relationship.
std::optional<MetalQueryPlan> buildQ16Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Part Groups ---
    // GPU filters qualifying parts to a compact list; a small host hook builds
    // the low-cardinality group dictionary and label buffers. This is the
    // measured fast path until the GPU group-build becomes genuinely parallel.
    plan.helpers.push_back(R"(
static uint q16_fnv32_25(const device char* tp, uint base) {
    uint h = 2166136261u;
    for (uint k = 0; k < 25; k++) {
        h ^= (uint)(uchar)tp[base + k];
        h *= 16777619u;
    }
    return h;
}
static void q16_filter_emit(device atomic_uint* counter,
                            device uint* out_idx,
                            device ulong* out_key,
                            const device char* p_brand,
                            const device char* p_type,
                            const device int*  p_size,
                            uint i) {
    int sz = p_size[i];
    if (!(sz==49 || sz==14 || sz==23 || sz==45 ||
          sz==19 || sz== 3 || sz==36 || sz== 9)) return;
    uint bb = i * 10u;
    if (p_brand[bb+0]=='B' && p_brand[bb+1]=='r' && p_brand[bb+2]=='a' &&
        p_brand[bb+3]=='n' && p_brand[bb+4]=='d' && p_brand[bb+5]=='#' &&
        p_brand[bb+6]=='4' && p_brand[bb+7]=='5') return;
    uint tb = i * 25u;
    if (p_type[tb+ 0]=='M' && p_type[tb+ 1]=='E' && p_type[tb+ 2]=='D' &&
        p_type[tb+ 3]=='I' && p_type[tb+ 4]=='U' && p_type[tb+ 5]=='M' &&
        p_type[tb+ 6]==' ' && p_type[tb+ 7]=='P' && p_type[tb+ 8]=='O' &&
        p_type[tb+ 9]=='L' && p_type[tb+10]=='I' && p_type[tb+11]=='S' &&
        p_type[tb+12]=='H' && p_type[tb+13]=='E' && p_type[tb+14]=='D') return;
    uint bidx = (uint)(p_brand[bb+6] - '1') * 5u + (uint)(p_brand[bb+7] - '1');
    uint h = q16_fnv32_25(p_type, tb);
    ulong key = ((ulong)bidx << 56) | ((ulong)(uint)sz << 48) | (ulong)h;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    out_idx[slot] = i;
    out_key[slot] = key;
}
)");

    // Build the compact part list, then use the host hook for group ids.
    {
        auto scan = makeAutoScan("part", idx);
        scan->addColumn("p_brand", "char");
        scan->addColumn("p_type", "char");
        scan->addColumn("p_size", "int");
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_q16_unused", "int",
            "(q16_filter_emit(d_q16_filt_count, d_q16_filt_idx, d_q16_filt_key, "
            "p_brand, p_type, p_size, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q16_filter_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q16_filt_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q16_filt_idx",   "uint",        false, false});
        phase.extraBuffers.push_back({"d_q16_filt_key",   "ulong",       false, false});

        // Host dictionary build. It consumes the GPU-emitted compact list and
        // fills both the direct partkey->group lookup and the GPU label buffers.
        phase.postDispatchHook = [](MetalGenericExecutor& ex) {
            auto& pd = g_q16Post;
            auto* cntBuf = ex.getAllocatedBuffer("d_q16_filt_count");
            auto* idxBuf = ex.getAllocatedBuffer("d_q16_filt_idx");
            auto* keyBuf = ex.getAllocatedBuffer("d_q16_filt_key");
            auto* mapBuf = ex.getAllocatedBuffer("d_q16_part_group_map");
            auto* brandBuf = ex.getAllocatedBuffer("d_q16_group_brand");
            auto* typeBuf = ex.getAllocatedBuffer("d_q16_group_type");
            auto* sizeBuf = ex.getAllocatedBuffer("d_q16_group_size");
            if (!cntBuf || !idxBuf || !keyBuf || !mapBuf ||
                !brandBuf || !typeBuf || !sizeBuf ||
                !pd.p_partkey || !pd.p_brand || !pd.p_type) {
                return 0.0;
            }

            uint32_t cnt = *static_cast<uint32_t*>(cntBuf->contents());
            const auto* fidx = static_cast<const uint32_t*>(idxBuf->contents());
            const auto* fkey = static_cast<const uint64_t*>(keyBuf->contents());
            auto* partGroupMap = static_cast<int32_t*>(mapBuf->contents());
            auto* groupBrand = static_cast<char*>(brandBuf->contents());
            auto* groupType = static_cast<char*>(typeBuf->contents());
            auto* groupSize = static_cast<int32_t*>(sizeBuf->contents());

            size_t capSym = 0;
            ex.tryGetSymbol("q16_result_cap", capSym);
            uint32_t groupCap = capSym > 0 ? (uint32_t)capSym : (1u << 16);

            std::unordered_map<uint64_t, uint32_t> groupMap;
            groupMap.reserve(2048);
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t partRow = fidx[k];
                uint64_t key = fkey[k];
                auto it = groupMap.find(key);
                uint32_t gid;
                if (it == groupMap.end()) {
                    gid = (uint32_t)groupMap.size();
                    if (gid >= groupCap) {
                        continue;
                    }
                    groupMap.emplace(key, gid);
                    std::memcpy(groupBrand + (size_t)gid * 10,
                                pd.p_brand + (size_t)partRow * 10, 10);
                    std::memcpy(groupType + (size_t)gid * 25,
                                pd.p_type + (size_t)partRow * 25, 25);
                    groupSize[gid] = (int32_t)((key >> 48) & 0xFF);
                } else {
                    gid = it->second;
                }

                int pk = pd.p_partkey[partRow];
                if (pk >= 0 && pk <= pd.maxPartkey) {
                    partGroupMap[pk] = (int32_t)gid;
                }
            }

            uint32_t numGroups = (uint32_t)groupMap.size();
            if (numGroups > groupCap) {
                numGroups = groupCap;
            }
            mapBuf->didModifyRange(NS::Range::Make(0, ((size_t)pd.maxPartkey + 1) * sizeof(int32_t)));
            brandBuf->didModifyRange(NS::Range::Make(0, (size_t)numGroups * 10));
            typeBuf->didModifyRange(NS::Range::Make(0, (size_t)numGroups * 25));
            sizeBuf->didModifyRange(NS::Range::Make(0, (size_t)numGroups * sizeof(int32_t)));

            uint32_t bvInts = ((uint32_t)pd.maxSk + 32) / 32;
            size_t gbmBytes = (size_t)numGroups * bvInts * sizeof(uint32_t);
            auto* dev = ex.device();
            size_t allocBytes = std::max<size_t>(gbmBytes, 4);
            auto* gbmBuf = dev->newBuffer(allocBytes, MTL::ResourceStorageModeShared);
            std::memset(gbmBuf->contents(), 0, allocBytes);
            gbmBuf->didModifyRange(NS::Range::Make(0, allocBytes));
            ex.registerAllocatedBuffer("d_q16_group_bitmaps", gbmBuf);
            ex.registerScalarInt("d_q16_bv_ints", (int)bvInts);

            size_t cntBytes = std::max<size_t>((size_t)numGroups * sizeof(uint32_t), 4);
            auto* cntBuf2 = dev->newBuffer(cntBytes, MTL::ResourceStorageModeShared);
            std::memset(cntBuf2->contents(), 0, cntBytes);
            cntBuf2->didModifyRange(NS::Range::Make(0, cntBytes));
            ex.registerAllocatedBuffer("d_q16_group_counts", cntBuf2);
            size_t popWords = (size_t)numGroups * (size_t)bvInts;
            ex.registerSymbol("n_q16_pop_words", popWords);
            ex.registerScalarInt("n_q16_pop_words", (int)popWords);
            ex.registerSymbol("q16_num_groups", numGroups);
            ex.registerScalarInt("q16_num_groups", (int)numGroups);
            return 0.0;
        };
    }

    // --- Complaint Suppliers ---
    // Match supplier comments containing "Customer" before "Complaints".
    plan.helpers.push_back(R"(
static bool q16_has_complaint(const device char* s_comment, uint idx, int width) {
    const device char* cmt = s_comment + (uint)idx * (uint)width;
    int len = width;
    while (len > 0 && (cmt[len-1] == ' ' || cmt[len-1] == '\0')) len--;
    for (int c = 0; c <= len - 8; c++) {
        if (cmt[c]=='C' && cmt[c+1]=='u' && cmt[c+2]=='s' && cmt[c+3]=='t' &&
            cmt[c+4]=='o' && cmt[c+5]=='m' && cmt[c+6]=='e' && cmt[c+7]=='r') {
            for (int d = c + 8; d <= len - 10; d++) {
                if (cmt[d]=='C' && cmt[d+1]=='o' && cmt[d+2]=='m' && cmt[d+3]=='p' &&
                    cmt[d+4]=='l' && cmt[d+5]=='a' && cmt[d+6]=='i' && cmt[d+7]=='n' &&
                    cmt[d+8]=='t' && cmt[d+9]=='s') {
                    return true;
                }
            }
            return false;
        }
    }
    return false;
}
)");

    // Set the supplier bit for a group.
    plan.helpers.push_back(R"(
static void q16_bitmap_set(device atomic_uint* group_bitmaps, uint bv_ints,
                            int group_id, int suppkey) {
    uint offset = (uint)group_id * bv_ints + ((uint)suppkey >> 5u);
    atomic_fetch_or_explicit(&group_bitmaps[offset], 1u << ((uint)suppkey & 31u), memory_order_relaxed);
}
)");

    // Build complaint-supplier bitmap.
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan),
            "q16_has_complaint(s_comment, " + idx + ", 101)");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q16_complaint_bitmap", "s_suppkey[" + idx + "]", "");

        appendPhase(plan, "Q16_build_complaint", std::move(bitmapBuild));
    }

    // --- Supplier Bitmaps ---
    // Populate per-group supplier bitmaps.
    {
        auto scan = makeAutoScan("partsupp", idx);

        auto groupLookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_q16_part_group_map", "ps_partkey[" + idx + "]",
            "q16_group_id", "int");

        auto filter = std::make_unique<MetalSelection>(
            std::move(groupLookup), "q16_group_id >= 0");

        auto antiProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(filter), "d_q16_complaint_bitmap", "ps_suppkey[" + idx + "]");

        auto bitmapSet = std::make_unique<MetalComputeExpr>(
            std::move(antiProbe), "_unused", "int",
            "(q16_bitmap_set(d_q16_group_bitmaps, d_q16_bv_ints, "
            "q16_group_id, ps_suppkey[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q16_scan_bitmap", std::move(bitmapSet));
        phase.extraBuffers.push_back({"d_q16_group_bitmaps", "atomic_uint", false});
        phase.scalarParams.push_back({"d_q16_bv_ints", "uint"});
    }

    // --- Supplier Counts ---
    // Popcount each per-group supplier bitmap.
    plan.helpers.push_back(R"(
static void q16_popcount_word(const device uint* group_bitmaps,
                               device atomic_uint* group_counts,
                               uint bv_ints,
                               uint i) {
    uint gid = i / bv_ints;
    uint w   = i - gid * bv_ints;
    uint p = popcount(group_bitmaps[i]);
    if (p) atomic_fetch_add_explicit(&group_counts[gid], p, memory_order_relaxed);
    (void)w;
}
)");
    {
        auto rscan = std::make_unique<MetalRangeScan>("q16_pop_words", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q16_pc_unused", "int",
            "(q16_popcount_word(d_q16_group_bitmaps, d_q16_group_counts, "
            "d_q16_bv_ints, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q16_popcount_groups", std::move(sideEffect));
        // Re-bind bitmaps as read-only uint after the phase barrier.
        phase.extraBuffers.push_back({"d_q16_group_bitmaps", "uint",        true,  false});
        phase.extraBuffers.push_back({"d_q16_group_counts",  "atomic_uint", false, true});
        phase.scalarParams.push_back({"d_q16_bv_ints", "uint"});
    }

    // --- Compact Results ---
    plan.helpers.push_back(R"(
static void q16_emit_group_result(device atomic_uint* counter,
                                  device char* out_brand,
                                  device char* out_type,
                                  device int* out_size,
                                  device uint* out_supplier_cnt,
                                  const device char* group_brand,
                                  const device char* group_type,
                                  const device int* group_size,
                                  const device uint* group_counts,
                                  uint cap, uint gid) {
    if (gid >= cap) return;
    uint cnt = group_counts[gid];
    if (cnt == 0u) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < cap) {
        for (uint c = 0; c < 10u; ++c)
            out_brand[slot * 10u + c] = group_brand[gid * 10u + c];
        for (uint c = 0; c < 25u; ++c)
            out_type[slot * 25u + c] = group_type[gid * 25u + c];
        out_size[slot] = group_size[gid];
        out_supplier_cnt[slot] = cnt;
    }
}
)");

    const std::string resultRows = "q16_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q16_num_groups", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q16_emit_unused", "int",
            "(q16_emit_group_result(d_q16_result_count, d_q16_result_brand, "
            "d_q16_result_type, d_q16_result_size, d_q16_result_supplier_cnt, "
            "d_q16_group_brand, d_q16_group_type, d_q16_group_size, "
            "d_q16_group_counts, q16_result_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q16_compact_results", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q16_result_count",        "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q16_result_brand",        "char",        false, false});
        phase.extraBuffers.push_back({"d_q16_result_type",         "char",        false, false});
        phase.extraBuffers.push_back({"d_q16_result_size",         "int",         false, false});
        phase.extraBuffers.push_back({"d_q16_result_supplier_cnt", "uint",        false, false});
        phase.extraBuffers.push_back({"d_q16_group_brand",         "char",        true,  false});
        phase.extraBuffers.push_back({"d_q16_group_type",          "char",        true,  false});
        phase.extraBuffers.push_back({"d_q16_group_size",          "int",         true,  false});
        phase.extraBuffers.push_back({"d_q16_group_counts",        "uint",        true,  false});
        phase.scalarParams.push_back({"q16_result_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q16_result_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("p_brand", "d_q16_result_brand", "char", 10),
            GenericMatColumnDesc("p_type", "d_q16_result_type", "char", 25),
            GenericMatColumnDesc("p_size", "d_q16_result_size", "int"),
            GenericMatColumnDesc("supplier_cnt", "d_q16_result_supplier_cnt", "uint"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"supplier_cnt", true});
        sortSpec.keys.push_back({"p_brand", false});
        sortSpec.keys.push_back({"p_type", false});
        sortSpec.keys.push_back({"p_size", false});
        std::string sortError;
        appendGenericGpuSort(plan, "q16_result", resultRows,
                             "q16_result_cap", columns, sortSpec, &sortError);
    }

    return plan;
}

} // namespace codegen
