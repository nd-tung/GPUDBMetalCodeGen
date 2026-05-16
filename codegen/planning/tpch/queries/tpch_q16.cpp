#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"
#include "execution/query_preprocessing.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace codegen {

// ===================================================================
// Q16: Parts/Supplier Relationship (COUNT DISTINCT)
// Phase 0 (GPU): Scan part → filter (size in valid set, brand != #45,
//      type doesn't start with "MEDIUM POLISHED") → atomically append
//      (idx, key64) to compact compaction buffers. Post-dispatch hook
//      builds the dict + d_q16_part_group_map on host from the ~12 %
//      of qualifying rows (was a 4M-row CPU scan at SF20: ~470 ms).
// Phase 1 (GPU): Scan supplier s_comment → build complaint bitmap
// Phase 2 (GPU): scan partsupp → ArrayLookup(group_id) → Selection(>=0) →
//      AntiBitmapProbe(complaint) → helper(per-group bitmap set).
// CPU post: popcount each group's bitmap for supplier_cnt.
// ===================================================================
std::optional<MetalQueryPlan> buildQ16Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: FNV-1a 32-bit over a 25-byte type field, plus the
    // filter+key+atomic-append routine emitted per qualifying row.
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

    // Phase 0: GPU filter + compact. The ComputeExpr's value isn't used;
    // the helper performs the atomic-append side-effect.
    {
        auto scan = makeAutoScan("part", idx);
        // Bare-pointer columns used by q16_filter_emit helper
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

        // Post-dispatch hook: read compact (idx, key) list, build dict
        // and d_q16_part_group_map on host. Allocates d_q16_group_bitmaps
        // sized exactly to numGroups * bvInts.
        phase.postDispatchHook = [](MetalGenericExecutor& ex) {
            auto& pd = g_q16Post;
            auto* cntBuf = ex.getAllocatedBuffer("d_q16_filt_count");
            auto* idxBuf = ex.getAllocatedBuffer("d_q16_filt_idx");
            auto* keyBuf = ex.getAllocatedBuffer("d_q16_filt_key");
            auto* mapBuf = ex.getAllocatedBuffer("d_q16_part_group_map");
            if (!cntBuf || !idxBuf || !keyBuf || !mapBuf || !pd.p_brand) return;
            uint32_t cnt = *static_cast<uint32_t*>(cntBuf->contents());
            const uint32_t* fidx = static_cast<const uint32_t*>(idxBuf->contents());
            const uint64_t* fkey = static_cast<const uint64_t*>(keyBuf->contents());
            int* partGroupMap = static_cast<int*>(mapBuf->contents());
            // map already pre-filled with -1 (0xFF) by preprocess.

            std::unordered_map<uint64_t, int> groupMap;
            groupMap.reserve(2048);
            std::vector<Q16PostData::GroupKey> groups;
            groups.reserve(2048);

            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t i = fidx[k];
                uint64_t key = fkey[k];
                auto it = groupMap.find(key);
                int gid;
                if (it == groupMap.end()) {
                    gid = (int)groups.size();
                    groupMap.emplace(key, gid);
                    const char* br = pd.p_brand + (size_t)i * 10;
                    const char* tp = pd.p_type  + (size_t)i * 25;
                    int brLen = 10;
                    while (brLen > 0 && (br[brLen-1] == ' ' || br[brLen-1] == '\0')) brLen--;
                    int tpLen = 25;
                    while (tpLen > 0 && (tp[tpLen-1] == ' ' || tp[tpLen-1] == '\0')) tpLen--;
                    int sz = (int)((key >> 48) & 0xFF);
                    groups.push_back({std::string(br, brLen), std::string(tp, tpLen), sz});
                } else {
                    gid = it->second;
                }
                // partkey lookup cached on pd.p_partkey (set in preprocess).
                int pk = pd.p_partkey ? pd.p_partkey[i] : 0;
                if (pk >= 0 && pk <= pd.maxPartkey) partGroupMap[pk] = gid;
            }

            uint32_t numGroups = (uint32_t)groups.size();
            uint32_t bvInts = ((uint32_t)pd.maxSk + 32) / 32;
            size_t gbmBytes = (size_t)numGroups * bvInts * sizeof(uint32_t);
            // Allocate d_q16_group_bitmaps now that numGroups is known.
            auto* dev = ex.device();
            size_t allocBytes = std::max<size_t>(gbmBytes, 4);
            auto* gbmBuf = dev->newBuffer(allocBytes, MTL::ResourceStorageModeShared);
            std::memset(gbmBuf->contents(), 0, allocBytes);
            ex.registerAllocatedBuffer("d_q16_group_bitmaps", gbmBuf);
            ex.registerScalarInt("d_q16_bv_ints", (int)bvInts);

            // Phase 4 (Q16_popcount_groups) reduces each group's bitmap on
            // GPU. Allocate the count buffer and register the dispatch
            // size symbol/scalar (one thread per word).
            size_t cntBytes = std::max<size_t>((size_t)numGroups * sizeof(uint32_t), 4);
            auto* cntBuf2 = dev->newBuffer(cntBytes, MTL::ResourceStorageModeShared);
            std::memset(cntBuf2->contents(), 0, cntBytes);
            ex.registerAllocatedBuffer("d_q16_group_counts", cntBuf2);
            size_t popWords = (size_t)numGroups * (size_t)bvInts;
            ex.registerSymbol("n_q16_pop_words", popWords);
            ex.registerScalarInt("n_q16_pop_words", (int)popWords);

            pd.groups = std::move(groups);
            pd.numGroups = numGroups;
        };
    }

    // Helper: substring search for "Customer" ... "Complaints" in s_comment
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

    // Helper: set bit in per-group bitmap
    plan.helpers.push_back(R"(
static void q16_bitmap_set(device atomic_uint* group_bitmaps, uint bv_ints,
                            int group_id, int suppkey) {
    uint offset = (uint)group_id * bv_ints + ((uint)suppkey >> 5u);
    atomic_fetch_or_explicit(&group_bitmaps[offset], 1u << ((uint)suppkey & 31u), memory_order_relaxed);
}
)");

    // Phase 1: Build complaint bitmap on GPU
    {
        auto scan = makeAutoScan("supplier", idx);

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan),
            "q16_has_complaint(s_comment, " + idx + ", 101)");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q16_complaint_bitmap", "s_suppkey[" + idx + "]", "");

        appendPhase(plan, "Q16_build_complaint", std::move(bitmapBuild));
    }

    // Phase 2: partsupp scan with bitmap ops
    {
        auto scan = makeAutoScan("partsupp", idx);

        // ArrayLookup: part_group_map[ps_partkey] → group_id
        auto groupLookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_q16_part_group_map", "ps_partkey[" + idx + "]",
            "q16_group_id", "int");

        // Selection: group_id >= 0 (qualifying part)
        auto filter = std::make_unique<MetalSelection>(
            std::move(groupLookup), "q16_group_id >= 0");

        // AntiBitmapProbe: supplier not complained-about
        auto antiProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(filter), "d_q16_complaint_bitmap", "ps_suppkey[" + idx + "]");

        // ComputeExpr: set bit in per-group bitmap (side-effect only)
        auto bitmapSet = std::make_unique<MetalComputeExpr>(
            std::move(antiProbe), "_unused", "int",
            "(q16_bitmap_set(d_q16_group_bitmaps, d_q16_bv_ints, "
            "q16_group_id, ps_suppkey[" + idx + "]), 0)");

        auto& phase = appendPhase(plan, "Q16_scan_bitmap", std::move(bitmapSet));
        phase.extraBuffers.push_back({"d_q16_group_bitmaps", "atomic_uint", false});
        phase.scalarParams.push_back({"d_q16_bv_ints", "uint"});
    }

    // Phase 4: GPU popcount per group → d_q16_group_counts. Replaces a
    // CPU popcount loop over numGroups * bvInts uint32 entries (~70 ms
    // at SF20). Dispatch is one thread per (group, word) — each thread
    // reads one uint32, popcounts it, and atomically adds to its group
    // counter. This keeps SIMD-group memory accesses sequential.
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
        // Re-bind d_q16_group_bitmaps as plain uint readonly (same MTL::Buffer
        // as the prior phase's atomic_uint binding — bit-identical storage,
        // queue barrier makes prior atomic-OR writes visible).
        phase.extraBuffers.push_back({"d_q16_group_bitmaps", "uint",        true,  false});
        phase.extraBuffers.push_back({"d_q16_group_counts",  "atomic_uint", false, true});
        phase.scalarParams.push_back({"d_q16_bv_ints", "uint"});
    }

    return plan;
}

} // namespace codegen
