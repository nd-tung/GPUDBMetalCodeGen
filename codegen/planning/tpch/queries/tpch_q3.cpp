#include "metal_plan_common.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "tpch/metal_tpch_query_builders.h"

namespace codegen {

namespace {

class Q3OrderMapBuild : public MetalUnaryOperator {
public:
    explicit Q3OrderMapBuild(std::unique_ptr<MetalOperator> child,
                             std::string idx)
        : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam("d_orders_date_map", "int", "maxOrderkey", false);
        cg.addBufferParam("d_orders_prio_map", "int", "maxOrderkey", false);
        cg.addBufferParam("d_order_revenue", "atomic_float", "maxOrderkey", false);
        cg.addAtomicBufferParam("d_q3_order_bitmap", "atomic_uint",
                                "(maxOrderkey + 31) / 32");

        child_->produce(cg, [&]() {
            cg.addLine("uint _q3_ok = (uint)o_orderkey[" + idx_ + "];");
            cg.addLine("d_orders_date_map[_q3_ok] = o_orderdate[" + idx_ + "];");
            cg.addLine("d_orders_prio_map[_q3_ok] = o_shippriority[" + idx_ + "];");
            cg.addLine("atomic_store_explicit(&d_order_revenue[_q3_ok], 0.0f, memory_order_relaxed);");
            cg.addLine("bitmap_set(d_q3_order_bitmap, (int)_q3_ok);");
            consume();
        });
    }

    std::string describe() const override { return "Q3OrderMapBuild"; }
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr("o_orderkey[" + idx_ + "]", out);
        appendIUsFromExpr("o_orderdate[" + idx_ + "]", out);
        appendIUsFromExpr("o_shippriority[" + idx_ + "]", out);
    }

private:
    std::string idx_;
};

class Q3RevenueAgg : public MetalUnaryOperator {
public:
    explicit Q3RevenueAgg(std::unique_ptr<MetalOperator> child,
                          std::string idx)
        : MetalUnaryOperator(std::move(child)), idx_(std::move(idx)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam("d_q3_order_bitmap", "const atomic_uint", "", false);
        cg.addBufferParam("d_order_revenue", "atomic_float", "", false);

        child_->produce(cg, [&]() {
            cg.addLine("int _q3_ok = (int)l_orderkey[" + idx_ + "];");
            cg.addIf("bitmap_test_atomic(d_q3_order_bitmap, _q3_ok)", [&]() {
                cg.addLine("atomic_fetch_add_explicit(&d_order_revenue[(uint)_q3_ok], "
                           "l_extendedprice[" + idx_ + "] * (1.0f - l_discount[" +
                           idx_ + "]), memory_order_relaxed);");
                consume();
            });
        });
    }

    std::string describe() const override { return "Q3RevenueAgg"; }
    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr("l_orderkey[" + idx_ + "]", out);
        appendIUsFromExpr("l_extendedprice[" + idx_ + "]", out);
        appendIUsFromExpr("l_discount[" + idx_ + "]", out);
    }

private:
    std::string idx_;
};

} // namespace

// Q3: Shipping Priority.
std::optional<MetalQueryPlan> buildQ3Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // --- Customer Filter ---
    // Build customer bitmap for BUILDING segment.
    {
        auto scan = makeAutoScan("customer", idx);

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_mktsegment[" + idx + "] == 'B'");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_cust_bitmap",
            "c_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        appendPhase(plan, "Q3_build_cust_bitmap", std::move(bitmap), 256);
    }

    // --- Order Maps ---
    // Build order date and priority maps.
    {
        auto scan = makeAutoScan("orders", idx);

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] < 19950315");

        auto custProbed = std::make_unique<MetalBitmapProbe>(std::move(dateFiltered),
            "d_cust_bitmap", "o_custkey[" + idx + "]");

        auto buildMaps = std::make_unique<Q3OrderMapBuild>(std::move(custProbed), idx);

        appendPhase(plan, "Q3_build_orders_maps", std::move(buildMaps), 256);
    }

    // --- Revenue Aggregate ---
    // Aggregate lineitem revenue per orderkey.
    {
        auto scan = makeAutoScan("lineitem", idx);

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] > 19950315");

        auto agg = std::make_unique<Q3RevenueAgg>(std::move(dateFiltered), idx);

        appendPhase(plan, "Q3_probe_aggregate", std::move(agg));
    }

    // --- Compact Results ---
    // Compact qualifying orders for GPU top-k.
    plan.helpers.push_back(R"(
static void q3_compact_emit(device atomic_uint* counter,
                             device uint* out_ok,
                             device float* out_rev,
                             device int* out_date,
                             device int* out_prio,
                             const device atomic_uint* d_q3_order_bitmap,
                             const device float* d_order_revenue,
                             const device int* d_orders_date_map,
                             const device int* d_orders_prio_map,
                             uint q3_compact_cap,
                             uint ok) {
    if (!bitmap_test_atomic(d_q3_order_bitmap, (int)ok)) return;
    float r = d_order_revenue[ok];
    if (!(r > 0.0f)) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < q3_compact_cap) {
        out_ok[slot] = ok;
        out_rev[slot] = r;
        out_date[slot] = d_orders_date_map[ok];
        out_prio[slot] = d_orders_prio_map[ok];
    }
}
)");
    const std::string resultRows = "q3_result_rows";
    {
        auto rscan = std::make_unique<MetalRangeScan>("q3_oks", idx);
        auto sideEffect = std::make_unique<MetalComputeExpr>(
            std::move(rscan), "_q3_unused", "int",
            "(q3_compact_emit(d_q3_compact_count, d_q3_compact_ok, "
            "d_q3_compact_rev, d_q3_compact_date, d_q3_compact_prio, "
            "d_q3_order_bitmap, d_order_revenue, d_orders_date_map, d_orders_prio_map, "
            "q3_compact_cap, " + idx + "), 0)");
        auto& phase = appendPhase(plan, "Q3_compact", std::move(sideEffect));
        phase.extraBuffers.push_back({"d_q3_compact_count", "atomic_uint", false, true});
        phase.extraBuffers.push_back({"d_q3_compact_ok",    "uint",        false, false});
        phase.extraBuffers.push_back({"d_q3_compact_rev",   "float",       false, false});
        phase.extraBuffers.push_back({"d_q3_compact_date",  "int",         false, false});
        phase.extraBuffers.push_back({"d_q3_compact_prio",  "int",         false, false});
        phase.extraBuffers.push_back({"d_q3_order_bitmap",  "atomic_uint", true,  false});
        phase.extraBuffers.push_back({"d_order_revenue",    "float",       true,  false});
        phase.extraBuffers.push_back({"d_orders_date_map",  "int",         true,  false});
        phase.extraBuffers.push_back({"d_orders_prio_map",  "int",         true,  false});
        phase.scalarParams.push_back({"q3_compact_cap", "uint"});
        attachMaterializedCountHook(phase, "d_q3_compact_count", resultRows);
    }

    // --- Result Order ---
    {
        std::vector<GenericMatColumnDesc> columns = {
            GenericMatColumnDesc("l_orderkey", "d_q3_compact_ok", "uint"),
            GenericMatColumnDesc("revenue", "d_q3_compact_rev", "float"),
            GenericMatColumnDesc("o_orderdate", "d_q3_compact_date", "int"),
            GenericMatColumnDesc("o_shippriority", "d_q3_compact_prio", "int"),
        };
        GenericSortSpec sortSpec;
        sortSpec.keys.push_back({"revenue", true});
        sortSpec.keys.push_back({"o_orderdate", false});
        sortSpec.limit = 10;
        std::string orderError;
        appendBestGenericGpuOrder(plan, "q3_result", resultRows,
                                  "maxOrderkey", columns, sortSpec,
                                  &orderError);
    }

    return plan;
}

} // namespace codegen
