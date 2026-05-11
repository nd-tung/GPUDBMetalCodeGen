#include "metal_plan_builder.h"
#include "../core/schema_provider.h"

namespace codegen {

MetalCodegen generateFromPlan(const MetalQueryPlan& plan,
                               const SchemaProvider* schema) {
    MetalCodegen cg;

    if (schema) {
        cg.setColumnTypeResolver([schema](const std::string& table,
                                          const std::string& col) -> std::string {
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
    if (cpuSort) {
        nlohmann::json s;
        s["limit"] = cpuSort->limit;
        for (auto& k : cpuSort->keys) {
            nlohmann::json sk;
            sk["column"] = k.column;
            sk["descending"] = k.descending;
            s["keys"].push_back(sk);
        }
        j["cpuSort"] = s;
    }
    if (cpuGroupBy) {
        nlohmann::json g;
        g["keyColumns"] = cpuGroupBy->keyColumns;
        g["aggColumns"] = cpuGroupBy->aggColumns;
        g["aggFuncs"] = cpuGroupBy->aggFuncs;
        j["cpuGroupBy"] = g;
    }
    if (gpuSort) {
        nlohmann::json gs;
        gs["sortedIndexBuffer"] = gpuSort->sortedIndexBuffer;
        gs["nResults"] = gpuSort->nResults;
        gs["descending"] = gpuSort->descending;
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
        if (phase.root) {
            pj["operatorTree"] = phase.root->toJSON();
        }
        j["phases"].push_back(pj);
    }
    return j;
}

} // namespace codegen