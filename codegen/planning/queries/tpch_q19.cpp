#include "../metal_plan_common.h"
#include "../metal_tpch_query_builders.h"

namespace codegen {

// ===================================================================
// Q19: Discounted Revenue — 2 phases
// ===================================================================
std::optional<MetalQueryPlan> buildQ19Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    plan.helpers.push_back(R"(
static bool brand_eq(const device char* brand, uint idx, char d1, char d2) {
    const device char* b = brand + idx * 10;
    return b[0]=='B' && b[1]=='r' && b[2]=='a' && b[3]=='n' && b[4]=='d' && b[5]=='#' && b[6]==d1 && b[7]==d2;
}
static int container_match(const device char* cont, uint idx) {
    const device char* c = cont + idx * 10;
    // SM CASE/BOX/PACK/PKG -> 1
    if (c[0]=='S' && c[1]=='M' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 1;
    }
    // MED BAG/BOX/PKG/PACK -> 2
    if (c[0]=='M' && c[1]=='E' && c[2]=='D' && c[3]==' ') {
        char c4=c[4],c5=c[5],c6=c[6];
        if ((c4=='B'&&c5=='A'&&c6=='G') || (c4=='B'&&c5=='O') ||
            (c4=='P'&&c5=='K') || (c4=='P'&&c5=='A'&&c6=='C')) return 2;
    }
    // LG CASE/BOX/PACK/PKG -> 3
    if (c[0]=='L' && c[1]=='G' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 3;
    }
    return 0;
}
)");

    // Phase 1: Build part condition bitmask map
    {
        auto scan = makeAutoScan("part", idx);

        auto computeCond = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_cond", "int",
            "(brand_eq(p_brand, " + idx + ", '1', '2') && container_match(p_container, " + idx + ") == 1 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 5 ? 1 : 0) | "
            "(brand_eq(p_brand, " + idx + ", '2', '3') && container_match(p_container, " + idx + ") == 2 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 10 ? 2 : 0) | "
            "(brand_eq(p_brand, " + idx + ", '3', '4') && container_match(p_container, " + idx + ") == 3 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 15 ? 4 : 0)");

        auto filtered = std::make_unique<MetalSelection>(std::move(computeCond), "_cond > 0");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_part_cond",
            "p_partkey[" + idx + "]", "_cond", "int", "maxPartkey");

        appendPhase(plan, "Q19_build_part_cond", std::move(store), 256);
    }

    // Phase 2: Scan lineitem, lookup part condition, check quantity, reduce revenue
    {
        auto scan = makeAutoScan("lineitem", idx);

        // l_shipmode IN ('AIR', 'REG AIR') — 'A..' or 'RE..'
        // l_shipinstruct = 'DELIVER IN PERSON' — first char 'D'
        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "(l_shipmode[" + idx + " * 2] == 'A' || (l_shipmode[" + idx + " * 2] == 'R' && l_shipmode[" + idx + " * 2 + 1] == 'E')) && l_shipinstruct[" + idx + " * 25] == 'D'");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_part_cond",
            "l_partkey[" + idx + "]", "_cond", "int", -1);

        std::string qtyCheck =
            "((_cond & 1) && l_quantity[" + idx + "] >= 1.0f && l_quantity[" + idx + "] <= 11.0f) || "
            "((_cond & 2) && l_quantity[" + idx + "] >= 10.0f && l_quantity[" + idx + "] <= 20.0f) || "
            "((_cond & 4) && l_quantity[" + idx + "] >= 20.0f && l_quantity[" + idx + "] <= 30.0f)";
        auto qtyFiltered = std::make_unique<MetalSelection>(std::move(lookup), qtyCheck);

        std::string revenue = "(long)(l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "]) * 100.0f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(qtyFiltered), "d_q19");
        reduce->addAccumulator("revenue", revenue, "long");
        reduce->setResultAlias("revenue", 100);

        appendPhase(plan, "Q19_reduce", std::move(reduce));
    }

    return plan;
}

} // namespace codegen
