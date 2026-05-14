#pragma once
// ===================================================================
// Metal Query Plan Builder — converts query planning decisions → operator trees
// ===================================================================
//
// Produces MetalQueryPlan containing per-phase operator trees that
// can be fed to MetalCodegen for Metal shader generation.
//
// Predefined TPC-H planning enters by query name; ad-hoc SQL planning enters
// through AnalyzedQuery. Both produce this MetalQueryPlan representation.
// ===================================================================

#include "metal_operators.h"
#include "metal_codegen_base.h"
#include "query_analyzer.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace codegen {

class SchemaProvider;  // fwd (for generateFromPlan)

struct MetalQueryPlan {
    std::string name;  // "Q1", "Q6", etc.

    // Each phase is one Metal kernel
    struct Phase {
        std::string name;
        std::unique_ptr<MetalOperator> root;
        int threadgroupSize = 1024;
        bool singleThread = false;
        // Bitmap buffers to register as read-only params (name, sizeExpr)
        // Used when an expression references a bitmap from a prior phase
        // without going through a BitmapProbe operator.
        std::vector<std::pair<std::string, std::string>> bitmapReads;
        // Scalar constant params (name, type) — registered before operator production
        std::vector<std::pair<std::string, std::string>> scalarParams;
        // Scalar constants whose host values are resolved from a size expression
        // at dispatch time (name, type, sizeExpr).
        struct ResolvedScalarParam {
            std::string name;
            std::string type;
            std::string sizeExpr;
        };
        std::vector<ResolvedScalarParam> resolvedScalarParams;
        // Extra buffer params not added by operators (e.g., pre-built hash tables)
        struct ExtraBuffer { std::string name; std::string type; bool readOnly = true; bool zeroInit = false; };
        std::vector<ExtraBuffer> extraBuffers;
        // Optional host-side callback after this phase's GPU dispatch (see
        // PostDispatchHook in metal_codegen_base.h). Used by GPU
        // preprocessing migration to read back computed scalars.
        PostDispatchHook postDispatchHook;
    };
    std::vector<Phase> phases;

    // Helper device functions emitted before all kernels
    std::vector<std::string> helpers;

    // Data-larger-than-memory (DLM) opt-in. When true, the driver may run
    // this plan under the chunked-streaming path: the largest scanned
    // .colbin table is split into row-chunks, stream phases run per chunk
    // with GPU output zero-init suppressed across chunks, and pre/post
    // phases run once around the loop. A plan is only safe to mark
    // chunkable when:
    //   - exactly one table appears as the "stream" scan (largest table),
    //   - every output buffer written by a stream phase uses an
    //     associative atomic op (atomic_fetch_add/_or), so partial
    //     results combine correctly across chunks,
    //   - any pre/post phases are bounded in size and read-only against
    //     the streamed table.
    // Default false is the safe choice.
    bool chunkable = false;

    // CPU-side post-processing
    struct CpuSort {
        struct SortKey { std::string column; bool descending; };
        std::vector<SortKey> keys;
        int limit = -1;
    };
    std::optional<CpuSort> cpuSort;

    // Host-side GROUP BY when GPU KeyedAgg cannot handle it.
    // Columns map by display name to aggregate function: "SUM", "COUNT", "AVG", "MIN", "MAX".
    struct CpuGroupBy {
        std::vector<std::string> keyColumns;       // column names in result
        std::vector<std::string> aggColumns;       // aggregate result columns
        std::vector<std::string> aggFuncs;         // "SUM", "COUNT", "AVG", "MIN", "MAX", "RATIO"
        // Pairs of (numIdx, denIdx) into aggColumns for RATIO functions.
        // After grouping, num/den columns are replaced with their ratio.
        std::vector<std::pair<int,int>> ratioPairs;
        // HAVING: if non-empty, the agg column at havingAggIdx is compared
        // against havingMultiplier * global_sum_of_agg.  If havingMultiplier
        // is 0, the having literal (from sentinel) is used directly.
        int havingAggIdx = -1;        // index into aggColumns
        double havingMultiplier = 0;  // e.g. 0.0001 → filter where agg > global_sum * 0.0001
        int havingSentinel = 0;       // the sentinel literal value (INT_MIN+idx)
    };
    std::optional<CpuGroupBy> cpuGroupBy;

    // CPU scalar aggregation (no GROUP BY).  Used as fallback when
    // GPU TGReduce cannot handle the probe pipe (e.g., semi-join probes).
    struct CpuScalarAgg {
        std::vector<std::string> aggColumns;       // aggregate result columns
        std::vector<std::string> aggFuncs;         // "SUM", "COUNT", "AVG", "MIN", "MAX", "RATIO"
    };
    std::optional<CpuScalarAgg> cpuScalarAgg;

    // GPU bitonic sort info.  When set, a GPU sort was attached to the
    // query plan.  The post-processing reads `sortedIndexBuffer` to
    // remap materialized rows into the final ORDER BY sequence.
    struct GpuSort {
        std::string sortedIndexBuffer;   // name of int[] buffer with sorted indices
        std::string nResults;            // symbolic name for the row count (e.g. n_lineitem)
        bool descending = false;         // column sort direction
    };
    std::optional<GpuSort> gpuSort;

    // Serialize the plan to JSON for debugging / CI integration.
    nlohmann::json toTreeJSON() const;

};

// Legacy compatibility wrapper. New callers should prefer the explicit APIs in
// metal_tpch_plan_api.h (predefined TPC-H) or metal_adhoc_plan_api.h
// (ad-hoc supported SQL patterns).
std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName = "");

// Generate Metal source from a MetalQueryPlan using the operator framework.
// If `schema` is provided, a ColumnTypeResolver is injected into the
// MetalCodegen to enable IU auto-projection (deduceRequiredColumns).
// Returns the configured MetalCodegen (with bindings, result schema, etc.)
MetalCodegen generateFromPlan(const MetalQueryPlan& plan,
                               const SchemaProvider* schema = nullptr);

} // namespace codegen
