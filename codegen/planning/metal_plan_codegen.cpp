#include "metal_plan_builder.h"
#include "../core/schema_provider.h"

namespace codegen {

MetalCodegen generateFromPlan(const MetalQueryPlan& plan,
                               const SchemaProvider* schema) {
    MetalCodegen cg;

    if (schema) {
        cg.setColumnTypeResolver([schema](const std::string& table,
                                          const std::string& col) -> std::string {
            if (!schema->hasColumn(table, col)) return {};
            DataType dt = schema->columnType(table, col);
            switch (dt) {
                case DataType::INT:  case DataType::DATE:
                    return "int";
                case DataType::FLOAT:
                    return "float";
                case DataType::CHAR1: case DataType::CHAR_FIXED:
                    return "char";
            }
            return {};
        });
    }

    for (const auto& h : plan.helpers) {
        cg.addHelper(h);
    }

    for (const auto& phase : plan.phases) {
        cg.beginPhase(phase.name);
        cg.setPhaseThreadgroupSize(phase.threadgroupSize);
        cg.setPhaseSingleThread(phase.singleThread);

        for (const auto& [bmpName, bmpSize] : phase.bitmapReads) {
            cg.addBitmapReadParam(bmpName, bmpSize);
        }

        for (const auto& [scName, scType] : phase.scalarParams) {
            cg.addScalarParam(scName, scType);
        }

        for (const auto& sp : phase.resolvedScalarParams) {
            cg.addResolvedScalarParam(sp.name, sp.type, sp.sizeExpr);
        }

        for (const auto& eb : phase.extraBuffers) {
            if (eb.readOnly)
                cg.addBufferParam(eb.name, "const " + eb.type, "", false);
            else
                cg.addBufferParam(eb.name, eb.type, "", eb.zeroInit);
        }

        if (phase.root) {
            phase.root->produce(cg, [](){});
        }

        if (phase.postDispatchHook) {
            cg.setPhasePostDispatchHook(phase.postDispatchHook);
        }

        cg.endPhase();
    }

    return cg;
}

nlohmann::json MetalQueryPlan::toTreeJSON() const {
    nlohmann::json j;
    j["name"] = name;
    j["chunkable"] = chunkable;
    j["strictGenericGpuOnly"] = true;
    j["genericOperatorShapes"] = nlohmann::json::array();
    for (const auto& phase : phases) {
        if (phase.root) j["genericOperatorShapes"].push_back(phase.root->describe());
    }
    if (gpuSort) j["genericOperatorShapes"].push_back("GpuSort");
    if (gpuSort) {
        nlohmann::json gs;
        gs["sortedIndexBuffer"] = gpuSort->sortedIndexBuffer;
        gs["nResults"] = gpuSort->nResults;
        gs["descending"] = gpuSort->descending;
        gs["limit"] = gpuSort->limit;
        j["gpuSort"] = gs;
    }
    for (const auto& phase : phases) {
        nlohmann::json pj;
        pj["name"] = phase.name;
        pj["threadgroupSize"] = phase.threadgroupSize;
        if (phase.singleThread) pj["singleThread"] = true;
        if (!phase.bitmapReads.empty()) {
            for (auto& [n, s] : phase.bitmapReads)
                pj["bitmapReads"].push_back({{"name", n}, {"sizeExpr", s}});
        }
        if (!phase.scalarParams.empty()) {
            for (auto& [n, t] : phase.scalarParams)
                pj["scalarParams"].push_back({{"name", n}, {"type", t}});
        }
        if (!phase.resolvedScalarParams.empty()) {
            for (const auto& sp : phase.resolvedScalarParams) {
                pj["resolvedScalarParams"].push_back({
                    {"name", sp.name},
                    {"type", sp.type},
                    {"sizeExpr", sp.sizeExpr}
                });
            }
        }
        if (phase.root) {
            pj["operatorTree"] = phase.root->toJSON();
        }
        j["phases"].push_back(pj);
    }
    return j;
}

} // namespace codegen
