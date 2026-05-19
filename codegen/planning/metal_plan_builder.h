#pragma once
// --- Metal Query Plan Builder ---
// Converts planner decisions into per-phase operator trees for MetalCodegen.
// Predefined TPC-H and ad-hoc SQL routes both produce MetalQueryPlan.

#include "metal_operators.h"
#include "metal_codegen_base.h"
#include "query_analyzer.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace codegen {

class SchemaProvider;

struct MetalQueryPlan {
    std::string name;

    // Each phase is one Metal kernel.
    struct Phase {
        std::string name;
        std::unique_ptr<MetalOperator> root;
        int threadgroupSize = 1024;
        bool singleThread = false;
        // Read-only bitmap params referenced without a BitmapProbe operator.
        std::vector<std::pair<std::string, std::string>> bitmapReads;
        // Scalar constants registered before operator production.
        std::vector<std::pair<std::string, std::string>> scalarParams;
        // Scalar constants resolved from size expressions at dispatch time.
        struct ResolvedScalarParam {
            std::string name;
            std::string type;
            std::string sizeExpr;
        };
        std::vector<ResolvedScalarParam> resolvedScalarParams;
        // Extra buffers not added directly by operators.
        struct ExtraBuffer { std::string name; std::string type; bool readOnly = true; bool zeroInit = false; };
        std::vector<ExtraBuffer> extraBuffers;
        // Optional host callback after this phase's GPU dispatch.
        PostDispatchHook postDispatchHook;
    };
    std::vector<Phase> phases;

    // Helper device functions emitted before all kernels.
    std::vector<std::string> helpers;

    // Optional planner cost traces for debugging physical-plan choices.
    std::vector<std::string> costTraces;

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

    // GPU bitonic sort info.  When set, a GPU sort was attached to the
    // query plan.  The post-processing reads `sortedIndexBuffer` to
    // remap materialized rows into the final ORDER BY sequence.
    struct GpuSort {
        std::string sortedIndexBuffer;   // name of int[] buffer with sorted indices
        std::string nResults;            // symbolic name for the row count (e.g. n_lineitem)
        bool descending = false;         // column sort direction
        int limit = -1;                  // optional LIMIT after GPU ordering
    };
    std::optional<GpuSort> gpuSort;

    // Serialize the plan to JSON for debugging / CI integration.
    nlohmann::json toTreeJSON() const;

};

// Compatibility wrapper for callers that do not select a route explicitly.
std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName = "");

// Generate Metal source and return the configured codegen state.
MetalCodegen generateFromPlan(const MetalQueryPlan& plan,
                               const SchemaProvider* schema = nullptr);

} // namespace codegen
