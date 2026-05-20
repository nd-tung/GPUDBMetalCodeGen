#pragma once
// --- Metal Query Plan Builder ---
// Converts planner decisions into per-phase operator trees for MetalCodegen.
// Predefined TPC-H and ad-hoc SQL routes both produce MetalQueryPlan.

#include "metal_operators.h"
#include "metal_codegen_base.h"
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
        // Compile the phase kernel for hook use, but do not dispatch it
        // through the normal executor path.
        bool hookOnly = false;
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

    // Optional host-side final-result assembly. This is route metadata, not
    // runtime query dispatch: predefined builders describe how compact GPU
    // buffers become the final TPC-H result shape, and the executor applies the
    // same generic reader/sorter/formatter for every spec.
    struct HostResultSpec {
        enum class Kind {
            BufferRows,
            StaticRows,
            ExistingSort,
        };

        enum class CellKind {
            IntLiteral,
            StringLiteral,
            BufferUInt,
            BufferFloat,
            ExistingCell,
            ExistingRatio,
            BufferRatio,
        };

        struct Column {
            std::string displayName;
            std::string type;
        };

        struct BufferColumn {
            std::string displayName;
            std::string bufferName;
            std::string elementType;
            int stringLen = 0;
            bool trimSpaces = true;
            bool asDateString = false;
        };

        struct SortKey {
            int columnIndex = 0;
            bool descending = false;
        };

        struct Cell {
            CellKind kind = CellKind::IntLiteral;
            int64_t intValue = 0;
            double doubleValue = 0.0;
            std::string stringValue;
            std::string bufferName;
            int index = 0;
            int numeratorIndex = 0;
            int denominatorIndex = 0;
            int row = 0;
            int column = 0;
            int numeratorRow = 0;
            int numeratorColumn = 0;
            int denominatorRow = 0;
            int denominatorColumn = 0;
            double multiplier = 1.0;
        };

        struct StaticRow {
            std::vector<Cell> values;
            std::optional<Cell> includeIf;
        };

        Kind kind = Kind::BufferRows;
        std::vector<Column> columns;
        int displayLimit = -1;

        // BufferRows.
        std::string countBuffer;
        std::string identityCountBuffer;
        std::vector<BufferColumn> bufferColumns;
        bool useGpuSort = true;
        int limit = -1;
        std::vector<SortKey> fallbackSort;

        // StaticRows.
        std::vector<StaticRow> staticRows;

        // ExistingSort.
        std::vector<SortKey> existingSort;
    };
    std::optional<HostResultSpec> hostResult;

    // Serialize the plan to JSON for debugging / CI integration.
    nlohmann::json toTreeJSON() const;

};

// Generate Metal source and return the configured codegen state.
MetalCodegen generateFromPlan(const MetalQueryPlan& plan,
                               const SchemaProvider* schema = nullptr);

} // namespace codegen
