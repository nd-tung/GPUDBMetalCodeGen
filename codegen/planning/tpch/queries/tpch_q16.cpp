#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

#include <vector>

namespace codegen {

// Q16: Parts/Supplier Relationship.
std::optional<MetalQueryPlan> buildQ16Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Part Groups ---
    // Q16 groups have a finite TPC-H domain after the brand, type-prefix, and
    // size filters: 24 brands * 145 types * 8 sizes.
    plan.helpers.push_back(R"(
static bool q16_match_standard(const device char* s, uint p) {
    return s[p+0]=='S' && s[p+1]=='T' && s[p+2]=='A' && s[p+3]=='N' &&
           s[p+4]=='D' && s[p+5]=='A' && s[p+6]=='R' && s[p+7]=='D';
}
static bool q16_match_small(const device char* s, uint p) {
    return s[p+0]=='S' && s[p+1]=='M' && s[p+2]=='A' && s[p+3]=='L' &&
           s[p+4]=='L';
}
static bool q16_match_medium(const device char* s, uint p) {
    return s[p+0]=='M' && s[p+1]=='E' && s[p+2]=='D' && s[p+3]=='I' &&
           s[p+4]=='U' && s[p+5]=='M';
}
static bool q16_match_large(const device char* s, uint p) {
    return s[p+0]=='L' && s[p+1]=='A' && s[p+2]=='R' && s[p+3]=='G' &&
           s[p+4]=='E';
}
static bool q16_match_economy(const device char* s, uint p) {
    return s[p+0]=='E' && s[p+1]=='C' && s[p+2]=='O' && s[p+3]=='N' &&
           s[p+4]=='O' && s[p+5]=='M' && s[p+6]=='Y';
}
static bool q16_match_promo(const device char* s, uint p) {
    return s[p+0]=='P' && s[p+1]=='R' && s[p+2]=='O' && s[p+3]=='M' &&
           s[p+4]=='O';
}
static bool q16_match_anodized(const device char* s, uint p) {
    return s[p+0]=='A' && s[p+1]=='N' && s[p+2]=='O' && s[p+3]=='D' &&
           s[p+4]=='I' && s[p+5]=='Z' && s[p+6]=='E' && s[p+7]=='D';
}
static bool q16_match_burnished(const device char* s, uint p) {
    return s[p+0]=='B' && s[p+1]=='U' && s[p+2]=='R' && s[p+3]=='N' &&
           s[p+4]=='I' && s[p+5]=='S' && s[p+6]=='H' && s[p+7]=='E' &&
           s[p+8]=='D';
}
static bool q16_match_plated(const device char* s, uint p) {
    return s[p+0]=='P' && s[p+1]=='L' && s[p+2]=='A' && s[p+3]=='T' &&
           s[p+4]=='E' && s[p+5]=='D';
}
static bool q16_match_polished(const device char* s, uint p) {
    return s[p+0]=='P' && s[p+1]=='O' && s[p+2]=='L' && s[p+3]=='I' &&
           s[p+4]=='S' && s[p+5]=='H' && s[p+6]=='E' && s[p+7]=='D';
}
static bool q16_match_brushed(const device char* s, uint p) {
    return s[p+0]=='B' && s[p+1]=='R' && s[p+2]=='U' && s[p+3]=='S' &&
           s[p+4]=='H' && s[p+5]=='E' && s[p+6]=='D';
}
static bool q16_match_tin(const device char* s, uint p) {
    return s[p+0]=='T' && s[p+1]=='I' && s[p+2]=='N';
}
static bool q16_match_nickel(const device char* s, uint p) {
    return s[p+0]=='N' && s[p+1]=='I' && s[p+2]=='C' && s[p+3]=='K' &&
           s[p+4]=='E' && s[p+5]=='L';
}
static bool q16_match_brass(const device char* s, uint p) {
    return s[p+0]=='B' && s[p+1]=='R' && s[p+2]=='A' && s[p+3]=='S' &&
           s[p+4]=='S';
}
static bool q16_match_steel(const device char* s, uint p) {
    return s[p+0]=='S' && s[p+1]=='T' && s[p+2]=='E' && s[p+3]=='E' &&
           s[p+4]=='L';
}
static bool q16_match_copper(const device char* s, uint p) {
    return s[p+0]=='C' && s[p+1]=='O' && s[p+2]=='P' && s[p+3]=='P' &&
           s[p+4]=='E' && s[p+5]=='R';
}
static int q16_size_slot(int sz) {
    if (sz == 49) return 0;
    if (sz == 14) return 1;
    if (sz == 23) return 2;
    if (sz == 45) return 3;
    if (sz == 19) return 4;
    if (sz ==  3) return 5;
    if (sz == 36) return 6;
    if (sz ==  9) return 7;
    return -1;
}
static int q16_brand_slot(const device char* p_brand, uint base) {
    if (!(p_brand[base+0]=='B' && p_brand[base+1]=='r' &&
          p_brand[base+2]=='a' && p_brand[base+3]=='n' &&
          p_brand[base+4]=='d' && p_brand[base+5]=='#')) return -1;
    int major = (int)p_brand[base+6] - (int)'1';
    int minor = (int)p_brand[base+7] - (int)'1';
    if (major < 0 || major >= 5 || minor < 0 || minor >= 5) return -1;
    int raw = major * 5 + minor;
    if (raw == 19) return -1;
    return raw > 19 ? raw - 1 : raw;
}
static int q16_type_slot(const device char* p_type, uint base) {
    int first = -1;
    uint firstLen = 0;
    if (q16_match_standard(p_type, base)) { first = 0; firstLen = 8; }
    else if (q16_match_small(p_type, base)) { first = 1; firstLen = 5; }
    else if (q16_match_medium(p_type, base)) { first = 2; firstLen = 6; }
    else if (q16_match_large(p_type, base)) { first = 3; firstLen = 5; }
    else if (q16_match_economy(p_type, base)) { first = 4; firstLen = 7; }
    else if (q16_match_promo(p_type, base)) { first = 5; firstLen = 5; }
    if (first < 0 || p_type[base + firstLen] != ' ') return -1;

    uint secondBase = base + firstLen + 1u;
    int second = -1;
    uint secondLen = 0;
    if (q16_match_anodized(p_type, secondBase)) { second = 0; secondLen = 8; }
    else if (q16_match_burnished(p_type, secondBase)) { second = 1; secondLen = 9; }
    else if (q16_match_plated(p_type, secondBase)) { second = 2; secondLen = 6; }
    else if (q16_match_polished(p_type, secondBase)) { second = 3; secondLen = 8; }
    else if (q16_match_brushed(p_type, secondBase)) { second = 4; secondLen = 7; }
    if (second < 0 || p_type[secondBase + secondLen] != ' ') return -1;

    uint thirdBase = secondBase + secondLen + 1u;
    int third = -1;
    if (q16_match_tin(p_type, thirdBase)) third = 0;
    else if (q16_match_nickel(p_type, thirdBase)) third = 1;
    else if (q16_match_brass(p_type, thirdBase)) third = 2;
    else if (q16_match_steel(p_type, thirdBase)) third = 3;
    else if (q16_match_copper(p_type, thirdBase)) third = 4;
    if (third < 0) return -1;

    int raw = first * 25 + second * 5 + third;
    if (raw >= 65 && raw <= 69) return -1;
    return raw > 69 ? raw - 5 : raw;
}
static int q16_group_slot(const device char* p_brand,
                          const device char* p_type,
                          const device int* p_size,
                          uint i) {
    int brandSlot = q16_brand_slot(p_brand, i * 10u);
    if (brandSlot < 0) return -1;
    int typeSlot = q16_type_slot(p_type, i * 25u);
    if (typeSlot < 0) return -1;
    int sizeSlot = q16_size_slot(p_size[i]);
    if (sizeSlot < 0) return -1;
    return ((brandSlot * 145 + typeSlot) * 8) + sizeSlot;
}
static void q16_build_part_group(device int* part_group_map,
                                 device atomic_uint* group_seen,
                                 device char* group_brand,
                                 device char* group_type,
                                 device int* group_size,
                                 const device int* p_partkey,
                                 const device char* p_brand,
                                 const device char* p_type,
                                 const device int* p_size,
                                 uint max_partkey,
                                 uint group_cap,
                                 uint i) {
    int gid = q16_group_slot(p_brand, p_type, p_size, i);
    if (gid < 0 || (uint)gid >= group_cap) return;

    int pk = p_partkey[i];
    if (pk >= 0 && (uint)pk < max_partkey) {
        part_group_map[(uint)pk] = gid;
    }

    uint wasSeen = atomic_exchange_explicit(&group_seen[(uint)gid], 1u,
                                            memory_order_relaxed);
    if (wasSeen == 0u) {
        for (uint c = 0; c < 10u; ++c)
            group_brand[(uint)gid * 10u + c] = p_brand[i * 10u + c];
        for (uint c = 0; c < 25u; ++c)
            group_type[(uint)gid * 25u + c] = p_type[i * 25u + c];
        group_size[(uint)gid] = p_size[i];
    }
}
)");

    {
        auto scan = makeAutoScan("part", idx);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_brand", "char");
        scan->addColumn("p_type", "char");
        scan->addColumn("p_size", "int");

        struct Q16BuildGroupsTerminal : MetalUnaryOperator {
            std::string idx_;
            Q16BuildGroupsTerminal(std::unique_ptr<MetalOperator> child,
                                   std::string idx)
                : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}
            void produce(MetalCodegen& cg, ConsumerFn consume) override {
                cg.addScalarParam("maxPartkey", "uint");
                cg.addScalarParam("q16_group_cap", "uint");
                cg.addBufferParam("d_q16_part_group_map", "int",
                                  "maxPartkey", true, 0xFF);
                cg.addBufferParam("d_q16_group_seen", "atomic_uint",
                                  "q16_group_cap", true);
                cg.addBufferParam("d_q16_group_brand", "char",
                                  "q16_group_cap * 10", false);
                cg.addBufferParam("d_q16_group_type", "char",
                                  "q16_group_cap * 25", false);
                cg.addBufferParam("d_q16_group_size", "int",
                                  "q16_group_cap", false);
                cg.addBufferParam("d_q16_group_bitmaps", "atomic_uint",
                                  "q16_pop_words", true);
                cg.addBufferParam("d_q16_group_counts", "atomic_uint",
                                  "q16_group_cap", true);
                child_->produce(cg, [&]() {
                    cg.addLine("q16_build_part_group(d_q16_part_group_map, "
                               "d_q16_group_seen, d_q16_group_brand, "
                               "d_q16_group_type, d_q16_group_size, "
                               "p_partkey, p_brand, p_type, p_size, "
                               "maxPartkey, q16_group_cap, (uint)" + idx_ + ");");
                });
                consume();
            }
            std::string describe() const override { return "Q16BuildGroups"; }
        };

        appendPhase(plan, "Q16_build_part_groups",
                    std::make_unique<Q16BuildGroupsTerminal>(
                        std::move(scan), idx));
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
        std::string orderError;
        appendBestGenericGpuOrder(plan, "q16_result", resultRows,
                                  "q16_result_cap", columns, sortSpec,
                                  &orderError);
    }

    return plan;
}

} // namespace codegen
