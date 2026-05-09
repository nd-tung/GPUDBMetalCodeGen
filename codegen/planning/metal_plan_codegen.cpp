#include "metal_plan_builder.h"

namespace codegen {

MetalCodegen generateFromPlan(const MetalQueryPlan& plan) {
    MetalCodegen cg;

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

} // namespace codegen