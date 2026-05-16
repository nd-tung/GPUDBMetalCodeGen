#include "metal_plan_common.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

// Q8: National Market Share.
std::optional<MetalQueryPlan> buildQ8Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Region And Part Filters ---
    // Build AMERICA nation bitmap.
    {
        auto scan = makeAutoScan("nation", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "n_regionkey[" + idx + "] == america_rk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_america_bitmap",
            "n_nationkey[" + idx + "]", "(25 + 31) / 32");

        auto& phase = appendPhase(plan, "Q8_build_nation_bitmap", std::move(bitmap), 32);
        phase.scalarParams = {{"america_rk", "int"}};
    }

    // Build target part-type bitmap.
    {
        auto scan = makeAutoScan("part", idx);

        // Compare the fixed-width p_type prefix.
        std::string cond =
            "p_type[" + idx + " * 25] == 'E' && "
            "p_type[" + idx + " * 25 + 1] == 'C' && "
            "p_type[" + idx + " * 25 + 2] == 'O' && "
            "p_type[" + idx + " * 25 + 3] == 'N' && "
            "p_type[" + idx + " * 25 + 4] == 'O' && "
            "p_type[" + idx + " * 25 + 5] == 'M' && "
            "p_type[" + idx + " * 25 + 6] == 'Y' && "
            "p_type[" + idx + " * 25 + 7] == ' ' && "
            "p_type[" + idx + " * 25 + 8] == 'A' && "
            "p_type[" + idx + " * 25 + 9] == 'N' && "
            "p_type[" + idx + " * 25 + 10] == 'O' && "
            "p_type[" + idx + " * 25 + 11] == 'D' && "
            "p_type[" + idx + " * 25 + 12] == 'I' && "
            "p_type[" + idx + " * 25 + 13] == 'Z' && "
            "p_type[" + idx + " * 25 + 14] == 'E' && "
            "p_type[" + idx + " * 25 + 15] == 'D' && "
            "p_type[" + idx + " * 25 + 16] == ' ' && "
            "p_type[" + idx + " * 25 + 17] == 'S' && "
            "p_type[" + idx + " * 25 + 18] == 'T' && "
            "p_type[" + idx + " * 25 + 19] == 'E' && "
            "p_type[" + idx + " * 25 + 20] == 'E' && "
            "p_type[" + idx + " * 25 + 21] == 'L'";

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), cond);

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_part_bitmap",
            "p_partkey[" + idx + "]", "(maxPartkey + 31) / 32");

        appendPhase(plan, "Q8_build_part_bitmap", std::move(bitmap), 256);
    }

    // --- Join Maps ---
    // Build customer nation map for AMERICA customers.
    {
        auto scan = makeAutoScan("customer", idx);

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_america_bitmap", "c_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        appendPhase(plan, "Q8_build_cust_map", std::move(store), 256);
    }

    // Build supplier nation map.
    {
        auto scan = makeAutoScan("supplier", idx);

        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        appendPhase(plan, "Q8_build_supp_map", std::move(store), 256);
    }

    // Build date-filtered order year map.
    {
        auto scan = makeAutoScan("orders", idx);

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] >= 19950101 && o_orderdate[" + idx + "] <= 19961231");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_cust_nation_map",
            "o_custkey[" + idx + "]",
            "_cust_nk", "int", -1);

        auto store = std::make_unique<MetalArrayStore>(
            std::move(lookup), "d_orders_year_map",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "] / 10000",
            "int", "maxOrderkey");

        appendPhase(plan, "Q8_build_orders_map", std::move(store));
    }

    // --- Market Share Aggregate ---
    // Aggregate total and Brazil revenue by year.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto partProbed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_part_bitmap", "l_partkey[" + idx + "]");

        auto lookupYear = std::make_unique<MetalArrayLookup>(
            std::move(partProbed), "d_orders_year_map",
            "l_orderkey[" + idx + "]",
            "_year", "int", -1);

        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupYear), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";

        // Bins 0-1 hold Brazil revenue; bins 2-3 hold total revenue.
        auto totalAgg = std::make_unique<MetalAtomicAgg>(
            std::move(lookupSupp), "d_result_bins",
            "2 + (_year - 1995)", revenue, "4",
            "atomic_float", "float");

        auto brazilFilter = std::make_unique<MetalSelection>(std::move(totalAgg),
            "_supp_nk == brazil_nk");

        auto brazilAgg = std::make_unique<MetalAtomicAgg>(
            std::move(brazilFilter), "d_result_bins",
            "_year - 1995", revenue, "4",
            "atomic_float", "float");

        auto& phase = appendPhase(plan, "Q8_probe_aggregate", std::move(brazilAgg));
        phase.scalarParams = {{"brazil_nk", "int"}};
    }

    return plan;
}

} // namespace codegen
