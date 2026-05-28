#include "metal_tpch_plan_api.h"
#include "tpch/metal_tpch_query_builders.h"

#include <unordered_set>

namespace codegen {

namespace {

const std::unordered_set<std::string>& chunkableTPCHNames() {
    static const std::unordered_set<std::string> kNames = {
        "Q1", "Q6", "Q12", "Q14", "Q19",
        "Q4", "Q13",
        "Q3", "Q5", "Q7", "Q8", "Q10",
        "Q15", "Q18",
        "Q11", "Q22",
    };
    return kNames;
}

void applyTPCHMetadata(MetalQueryPlan& plan, const std::string& queryName) {
    plan.name = queryName;
    if (chunkableTPCHNames().count(queryName)) {
        plan.chunkable = true;
    }

    using Spec = MetalQueryPlan::HostResultSpec;
    using Kind = Spec::Kind;
    using CellKind = Spec::CellKind;
    auto col = [](const char* name, const char* type) {
        return Spec::Column{std::string(name), std::string(type)};
    };
    auto bcol = [](const char* name, const char* buffer, const char* type,
                   int width = 0, bool trim = true, bool date = false) {
        Spec::BufferColumn c;
        c.displayName = name;
        c.bufferName = buffer;
        c.elementType = type;
        c.stringLen = width;
        c.trimSpaces = trim;
        c.asDateString = date;
        return c;
    };
    auto sort = [](int idx, bool desc) {
        return Spec::SortKey{idx, desc};
    };
    auto intCell = [&](int64_t v) {
        Spec::Cell c;
        c.kind = CellKind::IntLiteral;
        c.intValue = v;
        return c;
    };
    auto strCell = [&](const char* v) {
        Spec::Cell c;
        c.kind = CellKind::StringLiteral;
        c.stringValue = v;
        return c;
    };
    auto bufUInt = [&](const char* buffer, int idx) {
        Spec::Cell c;
        c.kind = CellKind::BufferUInt;
        c.bufferName = buffer;
        c.index = idx;
        return c;
    };
    auto bufFloat = [&](const char* buffer, int idx) {
        Spec::Cell c;
        c.kind = CellKind::BufferFloat;
        c.bufferName = buffer;
        c.index = idx;
        return c;
    };
    auto existingCell = [&](int row, int column) {
        Spec::Cell c;
        c.kind = CellKind::ExistingCell;
        c.row = row;
        c.column = column;
        return c;
    };
    auto existingRatio = [&](int nr, int nc, int dr, int dc, double mult) {
        Spec::Cell c;
        c.kind = CellKind::ExistingRatio;
        c.numeratorRow = nr;
        c.numeratorColumn = nc;
        c.denominatorRow = dr;
        c.denominatorColumn = dc;
        c.multiplier = mult;
        return c;
    };
    auto bufferRatio = [&](const char* buffer, int n, int d, double mult = 1.0) {
        Spec::Cell c;
        c.kind = CellKind::BufferRatio;
        c.bufferName = buffer;
        c.numeratorIndex = n;
        c.denominatorIndex = d;
        c.multiplier = mult;
        return c;
    };
    auto staticRow = [&](std::vector<Spec::Cell> values,
                         std::optional<Spec::Cell> includeIf = std::nullopt) {
        Spec::StaticRow r;
        r.values = std::move(values);
        r.includeIf = std::move(includeIf);
        return r;
    };

    auto bufferRows = [&]() {
        Spec s;
        s.kind = Kind::BufferRows;
        return s;
    };
    auto staticRows = [&]() {
        Spec s;
        s.kind = Kind::StaticRows;
        return s;
    };

    if (queryName == "Q2") {
        auto s = bufferRows();
        s.countBuffer = "d_q2_compact_count";
        s.identityCountBuffer = "d_q2_late_count";
        s.limit = 100;
        s.displayLimit = 10;
        s.bufferColumns = {
            bcol("s_acctbal", "d_q2_result_acctbal", "float"),
            bcol("s_name", "d_q2_result_s_name", "char", 25),
            bcol("n_name", "d_q2_result_n_name", "char", 25),
            bcol("p_partkey", "d_q2_result_p_partkey", "uint"),
            bcol("p_mfgr", "d_q2_result_p_mfgr", "char", 25),
            bcol("s_address", "d_q2_result_s_address", "char", 40),
            bcol("s_phone", "d_q2_result_s_phone", "char", 15, false),
            bcol("s_comment", "d_q2_result_s_comment", "char", 101, false),
        };
        s.fallbackSort = {sort(0, true), sort(2, false), sort(1, false), sort(3, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q3") {
        auto s = bufferRows();
        s.countBuffer = "d_q3_compact_count";
        s.limit = 10;
        s.bufferColumns = {
            bcol("l_orderkey", "d_q3_compact_ok", "uint"),
            bcol("revenue", "d_q3_compact_rev", "float"),
            bcol("o_orderdate", "d_q3_compact_date", "uint", 0, true, true),
            bcol("o_shippriority", "d_q3_compact_prio", "uint"),
        };
        s.fallbackSort = {sort(1, true), sort(2, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q5") {
        auto s = bufferRows();
        s.countBuffer = "d_q5_result_count";
        s.bufferColumns = {
            bcol("n_name", "d_q5_result_name", "char", 25),
            bcol("revenue", "d_q5_result_revenue", "float"),
        };
        s.fallbackSort = {sort(1, true)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q7") {
        auto s = staticRows();
        s.columns = {col("supp_nation", "string"), col("cust_nation", "string"),
                     col("l_year", "int"), col("revenue", "float")};
        s.staticRows = {
            staticRow({strCell("FRANCE"), strCell("GERMANY"), intCell(1995), bufFloat("d_revenue_bins", 0)}),
            staticRow({strCell("FRANCE"), strCell("GERMANY"), intCell(1996), bufFloat("d_revenue_bins", 1)}),
            staticRow({strCell("GERMANY"), strCell("FRANCE"), intCell(1995), bufFloat("d_revenue_bins", 2)}),
            staticRow({strCell("GERMANY"), strCell("FRANCE"), intCell(1996), bufFloat("d_revenue_bins", 3)}),
        };
        plan.hostResult = std::move(s);
    } else if (queryName == "Q8") {
        auto s = staticRows();
        s.columns = {col("o_year", "int"), col("mkt_share", "float")};
        s.staticRows = {
            staticRow({intCell(1995), bufferRatio("d_result_bins", 0, 2)}),
            staticRow({intCell(1996), bufferRatio("d_result_bins", 1, 3)}),
        };
        plan.hostResult = std::move(s);
    } else if (queryName == "Q9") {
        auto s = bufferRows();
        s.countBuffer = "d_q9_result_count";
        s.displayLimit = 15;
        s.bufferColumns = {
            bcol("nation", "d_q9_result_nation", "char", 25),
            bcol("o_year", "d_q9_result_year", "uint"),
            bcol("sum_profit", "d_q9_result_profit", "float"),
        };
        s.fallbackSort = {sort(0, false), sort(1, true)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q10") {
        auto s = bufferRows();
        s.countBuffer = "d_q10_compact_count";
        s.identityCountBuffer = "d_q10_late_count";
        s.limit = 20;
        s.bufferColumns = {
            bcol("c_custkey", "d_q10_result_ck", "uint"),
            bcol("c_name", "d_q10_result_name", "char", 25),
            bcol("revenue", "d_q10_result_rev", "float"),
            bcol("c_acctbal", "d_q10_result_acctbal", "float"),
            bcol("n_name", "d_q10_result_n_name", "char", 25),
            bcol("c_address", "d_q10_result_address", "char", 40),
            bcol("c_phone", "d_q10_result_phone", "char", 15),
            bcol("c_comment", "d_q10_result_comment", "char", 117),
        };
        plan.hostResult = std::move(s);
    } else if (queryName == "Q11") {
        Spec s;
        s.kind = Kind::ExistingSort;
        s.existingSort = {sort(1, true), sort(0, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q12") {
        auto s = staticRows();
        s.columns = {col("l_shipmode", "string"), col("high_line_count", "int"),
                     col("low_line_count", "int")};
        s.staticRows = {
            staticRow({strCell("MAIL"), existingCell(0, 1), existingCell(1, 1)}),
            staticRow({strCell("SHIP"), existingCell(2, 1), existingCell(3, 1)}),
        };
        plan.hostResult = std::move(s);
    } else if (queryName == "Q13") {
        auto s = bufferRows();
        s.countBuffer = "d_q13_result_count";
        s.bufferColumns = {
            bcol("c_count", "d_q13_result_c_count", "uint"),
            bcol("custdist", "d_q13_result_custdist", "uint"),
        };
        s.fallbackSort = {sort(1, true), sort(0, true)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q14") {
        auto s = staticRows();
        s.columns = {col("promo_revenue", "float")};
        s.staticRows = {
            staticRow({existingRatio(0, 0, 0, 1, 100.0)}),
        };
        plan.hostResult = std::move(s);
    } else if (queryName == "Q16") {
        auto s = bufferRows();
        s.countBuffer = "d_q16_result_count";
        s.displayLimit = 10;
        s.bufferColumns = {
            bcol("p_brand", "d_q16_result_brand", "char", 10),
            bcol("p_type", "d_q16_result_type", "char", 25),
            bcol("p_size", "d_q16_result_size", "uint"),
            bcol("supplier_cnt", "d_q16_result_supplier_cnt", "uint"),
        };
        s.fallbackSort = {sort(3, true), sort(0, false), sort(1, false), sort(2, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q18") {
        auto s = bufferRows();
        s.countBuffer = "d_q18_compact_count";
        s.limit = 100;
        s.bufferColumns = {
            bcol("c_name", "d_q18_compact_name", "char", 25),
            bcol("c_custkey", "d_q18_compact_custkey", "uint"),
            bcol("o_orderkey", "d_q18_compact_ok", "uint"),
            bcol("o_orderdate", "d_q18_compact_orderdate", "uint", 0, true, true),
            bcol("o_totalprice", "d_q18_compact_totalprice", "float"),
            bcol("sum(l_quantity)", "d_q18_compact_qty", "float"),
        };
        s.fallbackSort = {sort(3, true), sort(2, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q20") {
        auto s = bufferRows();
        s.countBuffer = "d_q20_result_count";
        s.displayLimit = 10;
        s.bufferColumns = {
            bcol("s_name", "d_q20_result_name", "char", 25),
            bcol("s_address", "d_q20_result_address", "char", 40),
        };
        s.fallbackSort = {sort(0, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q21") {
        auto s = bufferRows();
        s.countBuffer = "d_q21_result_count";
        s.limit = 100;
        s.displayLimit = 10;
        s.bufferColumns = {
            bcol("s_name", "d_q21_result_name", "char", 25),
            bcol("numwait", "d_q21_result_numwait", "uint"),
        };
        s.fallbackSort = {sort(1, true), sort(0, false)};
        plan.hostResult = std::move(s);
    } else if (queryName == "Q22") {
        auto s = staticRows();
        s.columns = {col("cntrycode", "int"), col("numcust", "int"),
                     col("totacctbal", "float")};
        const int prefixes[] = {13, 17, 18, 23, 29, 30, 31};
        for (int i = 0; i < 7; ++i) {
            s.staticRows.push_back(staticRow({
                intCell(prefixes[i]),
                bufUInt("d_q22_aggs", i * 2),
                bufFloat("d_q22_aggs", i * 2 + 1),
            }, bufUInt("d_q22_aggs", i * 2)));
        }
        plan.hostResult = std::move(s);
    }
}

} // namespace

bool isPredefinedTPCHQueryName(const std::string& queryName) {
    if (queryName.size() < 2 || queryName[0] != 'Q') return false;
    int q = 0;
    for (size_t i = 1; i < queryName.size(); ++i) {
        char c = queryName[i];
        if (c < '0' || c > '9') return false;
        q = q * 10 + (c - '0');
    }
    return q >= 1 && q <= 22;
}

std::optional<MetalQueryPlan> buildPredefinedTPCHPlan(const std::string& queryName) {
    auto dispatch = [&]() -> std::optional<MetalQueryPlan> {
        if (queryName == "Q19") return buildQ19Plan_byName();
        if (queryName == "Q13") return buildQ13Plan_byName();
        if (queryName == "Q22") return buildQ22Plan_byName();
        if (queryName == "Q11") return buildQ11Plan_byName();
        if (queryName == "Q15") return buildQ15Plan_byName();
        if (queryName == "Q18") return buildQ18Plan_byName();
        if (queryName == "Q17") return buildQ17Plan_byName();
        if (queryName == "Q9")  return buildQ9Plan_byName();
        if (queryName == "Q20") return buildQ20Plan_byName();
        if (queryName == "Q2")  return buildQ2Plan_byName();
        if (queryName == "Q16") return buildQ16Plan_byName();
        if (queryName == "Q21") return buildQ21Plan_byName();
        if (queryName == "Q5")  return buildQ5Plan_byName();
        if (queryName == "Q3")  return buildQ3Plan_byName();
        if (queryName == "Q8")  return buildQ8Plan_byName();
        if (queryName == "Q7")  return buildQ7Plan_byName();
        if (queryName == "Q6")  return buildQ6Plan_byName();
        if (queryName == "Q1")  return buildQ1Plan_byName();
        if (queryName == "Q14") return buildQ14Plan_byName();
        if (queryName == "Q4")  return buildQ4Plan_byName();
        if (queryName == "Q12") return buildQ12Plan_byName();
        if (queryName == "Q10") return buildQ10Plan_byName();

        return std::nullopt;
    };

    auto plan = dispatch();
    if (plan) applyTPCHMetadata(*plan, queryName);
    return plan;
}

} // namespace codegen
