#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q19: Discounted Revenue.
std::optional<MetalQueryPlan> buildQ19Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Helpers ---
    // Fixed-width brand and container checks encode the three Q19 part cases.
    plan.helpers.push_back(R"(
static bool q19_brand_eq(const device char* brand, uint idx, char d1, char d2) {
    const device char* b = brand + (ulong)idx * 10ul;
    return b[0] == 'B' && b[1] == 'r' && b[2] == 'a' &&
           b[3] == 'n' && b[4] == 'd' && b[5] == '#' &&
           b[6] == d1 && b[7] == d2 &&
           (b[8] == '\0' || b[8] == ' ') &&
           (b[9] == '\0' || b[9] == ' ');
}
static int q19_container_match(const device char* container, uint idx) {
    const device char* c = container + (ulong)idx * 10ul;
    if (c[0] == 'S' && c[1] == 'M' && c[2] == ' ') {
        if ((c[3] == 'C' && c[4] == 'A' && c[5] == 'S' && c[6] == 'E') ||
            (c[3] == 'B' && c[4] == 'O' && c[5] == 'X') ||
            (c[3] == 'P' && c[4] == 'A' && c[5] == 'C' && c[6] == 'K') ||
            (c[3] == 'P' && c[4] == 'K' && c[5] == 'G')) return 1;
    }
    if (c[0] == 'M' && c[1] == 'E' && c[2] == 'D' && c[3] == ' ') {
        if ((c[4] == 'B' && c[5] == 'A' && c[6] == 'G') ||
            (c[4] == 'B' && c[5] == 'O' && c[6] == 'X') ||
            (c[4] == 'P' && c[5] == 'K' && c[6] == 'G') ||
            (c[4] == 'P' && c[5] == 'A' && c[6] == 'C' && c[7] == 'K')) return 2;
    }
    if (c[0] == 'L' && c[1] == 'G' && c[2] == ' ') {
        if ((c[3] == 'C' && c[4] == 'A' && c[5] == 'S' && c[6] == 'E') ||
            (c[3] == 'B' && c[4] == 'O' && c[5] == 'X') ||
            (c[3] == 'P' && c[4] == 'A' && c[5] == 'C' && c[6] == 'K') ||
            (c[3] == 'P' && c[4] == 'K' && c[5] == 'G')) return 3;
    }
    return 0;
}
static uchar q19_part_condition(const device char* brand,
                                const device char* container,
                                const device int* size,
                                uint idx) {
    int c = q19_container_match(container, idx);
    int s = size[idx];
    uchar mask = 0;
    if (c == 1 && s >= 1 && s <= 5 &&
        q19_brand_eq(brand, idx, '1', '2')) mask |= 1;
    if (c == 2 && s >= 1 && s <= 10 &&
        q19_brand_eq(brand, idx, '2', '3')) mask |= 2;
    if (c == 3 && s >= 1 && s <= 15 &&
        q19_brand_eq(brand, idx, '3', '4')) mask |= 4;
    return mask;
}
)");

    // --- Part Condition Map ---
    // Bitmask values identify which Q19 quantity range applies.
    {
        auto scan = makeAutoScan("part", idx);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_brand", "char");
        scan->addColumn("p_container", "char");
        scan->addColumn("p_size", "int");
        std::string condExpr =
            "q19_part_condition(p_brand, p_container, p_size, " + idx + ")";

        auto computeCond = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_q19_part_cond", "uchar", condExpr);
        auto filtered = std::make_unique<MetalSelection>(
            std::move(computeCond), "_q19_part_cond > 0");
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_q19_part_cond",
            "p_partkey[" + idx + "]", "_q19_part_cond", "uchar",
            "maxPartkey", /*fillByte=*/0);
        appendPhase(plan, "Q19_build_part_cond", std::move(store), 256);
    }

    // --- Revenue Reduction ---
    // Probe candidate branch masks, re-check quantity, then reduce revenue.
    {
        auto scan = makeAutoScan("lineitem", idx);
        scan->addColumn("l_shipmode", "char");
        scan->addColumn("l_shipinstruct", "char");
        scan->addColumn("l_quantity", "float");
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");
        std::string shipModeCond =
            "((l_shipmode[(ulong)" + idx + " * 2ul] == 'A' && l_shipmode[(ulong)" + idx + " * 2ul + 1ul] == 'I') || "
            "(l_shipmode[(ulong)" + idx + " * 2ul] == 'R' && l_shipmode[(ulong)" + idx + " * 2ul + 1ul] == 'E'))";
        std::string instructCond =
            "l_shipinstruct[(ulong)" + idx + " * 25ul] == 'D'";
        std::string qtyCond =
            "((l_quantity[" + idx + "] >= 1.0f && l_quantity[" + idx + "] <= 11.0f) || "
            "(l_quantity[" + idx + "] >= 10.0f && l_quantity[" + idx + "] <= 20.0f) || "
            "(l_quantity[" + idx + "] >= 20.0f && l_quantity[" + idx + "] <= 30.0f))";
        auto lineFiltered = std::make_unique<MetalSelection>(
            std::move(scan), shipModeCond + " && " + instructCond + " && " + qtyCond);

        auto condLookup = std::make_unique<MetalArrayLookup>(
            std::move(lineFiltered), "d_q19_part_cond",
            "l_partkey[" + idx + "]", "_q19_part_cond", "uchar", 0);

        std::string branchQtyCond =
            "(((_q19_part_cond & 1) != 0 && l_quantity[" + idx + "] >= 1.0f && l_quantity[" + idx + "] <= 11.0f) || "
            "((_q19_part_cond & 2) != 0 && l_quantity[" + idx + "] >= 10.0f && l_quantity[" + idx + "] <= 20.0f) || "
            "((_q19_part_cond & 4) != 0 && l_quantity[" + idx + "] >= 20.0f && l_quantity[" + idx + "] <= 30.0f))";
        auto branchFiltered = std::make_unique<MetalSelection>(
            std::move(condLookup), branchQtyCond);

        std::string revenueExpr =
            "(long)round(l_extendedprice[" + idx + "] * (1.0f - l_discount[" +
            idx + "]) * 100.0f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(branchFiltered), "d_q19");
        reduce->addAccumulator("revenue", revenueExpr, "long");
        reduce->setResultAlias("revenue", 100);

        appendPhase(plan, "Q19_reduce", std::move(reduce));
    }

    return plan;
}

} // namespace codegen
