#include "generic/lowering/generic_scalar_lookup.h"

#include "core/schema_provider.h"
#include "generic/lowering/generic_scalar_hash_names.h"
#include "scalar_subquery_placeholder.h"

#include <sstream>

namespace codegen {

namespace {

using ScalarLookupInfo = GenericScalarLookupInfo;

std::string scalarFloatLiteral(float v) {
    std::ostringstream oss;
    oss << v;
    std::string s = oss.str();
    if (s.find_first_of(".eE") == std::string::npos)
        s += ".0";
    return s + "f";
}

std::string scalarNanExpr() {
    return "as_type<float>(0x7fc00000u)";
}

std::string scalarLookupKeyExpr(const ScalarLookupInfo& info,
                                size_t keyIndex,
                                const std::string& idxVar,
                                const std::string& probeTable,
                                const SchemaProvider* schema) {
    std::string inner = keyIndex < info.keyCols.size() ? info.keyCols[keyIndex] :
                        (keyIndex == 0 ? info.keyCol : info.keyCol2);
    std::string outer = keyIndex < info.outerKeyCols.size() ? info.outerKeyCols[keyIndex] : "";
    if (schema && !probeTable.empty()) {
        if (!outer.empty() && schema->hasColumn(probeTable, outer))
            return outer + "[" + idxVar + "]";
        if (!inner.empty() && schema->hasColumn(probeTable, inner))
            return inner + "[" + idxVar + "]";
    }
    return (outer.empty() ? inner : outer) + "[" + idxVar + "]";
}

std::string scalarHashLookupRaw(const std::string& mapName,
                                const std::string& key1,
                                const std::string& key2) {
    const auto names = scalarCompositeHashNames(mapName);
    return "scalar_hash_lookup_raw64(" + names.states() + ", " +
           names.keys1() + ", " + names.keys2() + ", " +
           names.values() + ", " + names.capacity() + ", " + key1 + ", " +
           key2 + ")";
}

std::string scalarLookupReplacement(const ScalarLookupInfo& info,
                                    const std::string& idxVar,
                                    const std::string& probeTable,
                                    const SchemaProvider* schema) {
    if (!info.scalarName.empty()) return info.scalarName;
    const std::string key0 = scalarLookupKeyExpr(info, 0, idxVar, probeTable, schema);
    const std::string key1 = scalarLookupKeyExpr(info, 1, idxVar, probeTable, schema);
    switch (info.kind) {
        case ScalarLookupInfo::SumByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.sumBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.sumBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::AvgByKey:
            return "((" + info.cntVar + " > 0) ? ("
                 + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" + info.sumVar
                 + ") / (float)" + info.cntVar
                 + ") : " + scalarNanExpr() + ")";
        case ScalarLookupInfo::MinByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.minBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.minBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::MaxByKey:
            if (!info.stateBuffer.empty()) {
                return "((" + info.stateBuffer + "[" + key0 + "] != 0u) ? (" +
                       scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                       info.maxBuffer + "[" + key0 + "])) : " + scalarNanExpr() + ")";
            }
            return "(" + scalarFloatLiteral(info.multiplier) + " * as_type<float>(" +
                   info.maxBuffer + "[" + key0 + "]))";
        case ScalarLookupInfo::CountByKey:
            return "(" + scalarFloatLiteral(info.multiplier) + " * (float)" +
                   info.countBuffer + "[" + key0 + "])";
        case ScalarLookupInfo::SumByCompositeHash: {
            const auto names = scalarCompositeHashNames(info.hashMap);
            std::string val = "scalar_hash_lookup_float_or_nan64(" + names.states() + ", " +
                              names.keys1() + ", " + names.keys2() + ", " +
                              names.values() + ", " + names.capacity() + ", (uint)(" +
                              key0 + "), (uint)(" + key1 + "))";
            return "(" + scalarFloatLiteral(info.multiplier) + " * " + val + ")";
        }
        case ScalarLookupInfo::CountByCompositeHash: {
            std::string raw = scalarHashLookupRaw(info.countHashMap, "(uint)(" + key0 + ")",
                                                  "(uint)(" + key1 + ")");
            return "(" + scalarFloatLiteral(info.multiplier) + " * (float)(" + raw + "))";
        }
        case ScalarLookupInfo::AvgByCompositeHash: {
            std::string sk = "(uint)(" + key0 + ")";
            std::string tk = "(uint)(" + key1 + ")";
            std::string sumRaw = scalarHashLookupRaw(info.hashMap, sk, tk);
            std::string cntRaw = scalarHashLookupRaw(info.countHashMap, sk, tk);
            return "((" + cntRaw + " > 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(" + sumRaw + ") / (float)(" + cntRaw + ")) : " +
                   scalarNanExpr() + ")";
        }
        case ScalarLookupInfo::GlobalSum:
            return "(" + scalarFloatLiteral(info.multiplier) +
                   " * atomic_load_explicit(&" + info.sumBuffer +
                   "[0], memory_order_relaxed))";
        case ScalarLookupInfo::GlobalAvg:
            return "((atomic_load_explicit(&" + info.countBuffer +
                   "[0], memory_order_relaxed) > 0u) ? (" +
                   scalarFloatLiteral(info.multiplier) +
                   " * atomic_load_explicit(&" + info.sumBuffer +
                   "[0], memory_order_relaxed) / (float)atomic_load_explicit(&" +
                   info.countBuffer + "[0], memory_order_relaxed)) : " +
                   scalarNanExpr() + ")";
        case ScalarLookupInfo::GlobalCount:
            return "(" + scalarFloatLiteral(info.multiplier) +
                   " * (float)atomic_load_explicit(&" + info.countBuffer +
                   "[0], memory_order_relaxed))";
        case ScalarLookupInfo::GlobalMin:
            return "((" + info.stateBuffer +
                   "[0] != 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(atomic_load_explicit(&" + info.minBuffer +
                   "[0], memory_order_relaxed))) : " + scalarNanExpr() + ")";
        case ScalarLookupInfo::GlobalMax:
            return "((" + info.stateBuffer +
                   "[0] != 0u) ? (" + scalarFloatLiteral(info.multiplier) +
                   " * as_type<float>(atomic_load_explicit(&" + info.maxBuffer +
                   "[0], memory_order_relaxed))) : " + scalarNanExpr() + ")";
        default: return "0";
    }
}

bool replaceScalarLookupToken(std::string& str,
                              const std::string& token,
                              const std::string& replacement) {
    if (token.empty()) return false;
    bool replaced = false;
    size_t pos = 0;
    while ((pos = str.find(token, pos)) != std::string::npos) {
        str.replace(pos, token.size(), replacement);
        pos += replacement.size();
        replaced = true;
    }
    return replaced;
}

std::string scalarLookupPlaceholderToken(const ScalarLookupInfo& info) {
    if (info.scalarSubqueryIndex < 0) return "";
    return scalarSubqueryPlaceholderToken(info.scalarSubqueryIndex);
}

bool scalarLookupNeedsPhaseBuffers(const ScalarLookupInfo& info) {
    if (!info.scalarName.empty()) return true;
    const bool global = info.kind == ScalarLookupInfo::GlobalSum ||
                        info.kind == ScalarLookupInfo::GlobalAvg ||
                        info.kind == ScalarLookupInfo::GlobalMin ||
                        info.kind == ScalarLookupInfo::GlobalMax ||
                        info.kind == ScalarLookupInfo::GlobalCount;
    if (global) {
        return !info.sumBuffer.empty() || !info.countBuffer.empty() ||
               !info.minBuffer.empty() || !info.maxBuffer.empty() ||
               !info.stateBuffer.empty();
    }
    if (info.kind == ScalarLookupInfo::AvgByKey)
        return false;
    return !info.sumBuffer.empty() || !info.countBuffer.empty() ||
           !info.minBuffer.empty() || !info.maxBuffer.empty() ||
           !info.stateBuffer.empty() || !info.hashMap.empty() ||
           !info.countHashMap.empty() || !info.htFlags.empty() ||
           !info.htKeys.empty() || !info.htVals.empty();
}

} // namespace

std::string genericScalarLookupKeyExpr(
        const GenericScalarLookupInfo& info,
        size_t keyIndex,
        const std::string& idxVar,
        const std::string& probeTable,
        const SchemaProvider* schema) {
    return scalarLookupKeyExpr(info, keyIndex, idxVar, probeTable, schema);
}

std::string rewriteGenericScalarPlaceholders(
        const std::string& cond,
        const std::string& idxVar,
        const std::vector<GenericScalarLookupInfo>& lookups,
        const std::string& probeTable,
        const SchemaProvider* schema) {
    return rewriteGenericScalarPlaceholdersWithDeps(
        cond, idxVar, lookups, probeTable, schema).text;
}

GenericScalarRewriteResult rewriteGenericScalarPlaceholdersWithDeps(
        const std::string& cond,
        const std::string& idxVar,
        const std::vector<GenericScalarLookupInfo>& lookups,
        const std::string& probeTable,
        const SchemaProvider* schema) {
    GenericScalarRewriteResult result;
    result.text = cond;
    for (const auto& info : lookups) {
        if (replaceScalarLookupToken(
                result.text, scalarLookupPlaceholderToken(info),
                scalarLookupReplacement(info, idxVar, probeTable, schema))) {
            result.referencedScalarLookup = true;
            result.needsScalarLookupLoads =
                result.needsScalarLookupLoads ||
                info.kind == ScalarLookupInfo::AvgByKey;
            result.needsScalarLookupBuffers =
                result.needsScalarLookupBuffers ||
                scalarLookupNeedsPhaseBuffers(info);
        }
    }
    return result;
}

void attachGenericScalarLookupBuffers(
        MetalQueryPlan::Phase& phase,
        const std::vector<GenericScalarLookupInfo>& lookups) {
    auto addResolvedScalar = [&](const std::string& name,
                                 const std::string& type,
                                 const std::string& sizeExpr) {
        if (name.empty() || sizeExpr.empty()) return;
        for (const auto& existing : phase.resolvedScalarParams) {
            if (existing.name == name) return;
        }
        phase.resolvedScalarParams.push_back({name, type, sizeExpr});
    };
    auto addScalar = [&](const std::string& name, const std::string& type) {
        if (name.empty()) return;
        for (const auto& existing : phase.scalarParams) {
            if (existing.first == name) return;
        }
        phase.scalarParams.push_back({name, type});
    };

    for (const auto& info : lookups) {
        bool global = info.kind == ScalarLookupInfo::GlobalSum ||
                      info.kind == ScalarLookupInfo::GlobalAvg ||
                      info.kind == ScalarLookupInfo::GlobalMin ||
                      info.kind == ScalarLookupInfo::GlobalMax ||
                      info.kind == ScalarLookupInfo::GlobalCount;
        if (global) {
            if (!info.scalarName.empty()) {
                addScalar(info.scalarName, "float");
                continue;
            }
            if (!info.sumBuffer.empty())
                phase.extraBuffers.push_back({
                    info.sumBuffer,
                    (info.kind == ScalarLookupInfo::GlobalSum ||
                     info.kind == ScalarLookupInfo::GlobalAvg) ? "atomic_float" : "atomic_uint",
                    true,
                    false});
            if (!info.countBuffer.empty())
                phase.extraBuffers.push_back({info.countBuffer, "atomic_uint", true, false});
            if (!info.minBuffer.empty())
                phase.extraBuffers.push_back({info.minBuffer, "atomic_uint", true, false});
            if (!info.maxBuffer.empty())
                phase.extraBuffers.push_back({info.maxBuffer, "atomic_uint", true, false});
            if (!info.stateBuffer.empty())
                phase.extraBuffers.push_back({info.stateBuffer, "uint", true, false});
            continue;
        }
        if (info.kind == ScalarLookupInfo::AvgByKey) continue; // handled by ScalarAtomicLookup
        if (!info.sumBuffer.empty())
            phase.extraBuffers.push_back({info.sumBuffer, "uint", true, false});
        if (!info.countBuffer.empty())
            phase.extraBuffers.push_back({info.countBuffer, "uint", true, false});
        if (!info.minBuffer.empty())
            phase.extraBuffers.push_back({info.minBuffer, "uint", true, false});
        if (!info.maxBuffer.empty())
            phase.extraBuffers.push_back({info.maxBuffer, "uint", true, false});
        if (!info.stateBuffer.empty())
            phase.extraBuffers.push_back({info.stateBuffer, "uint", true, false});
        if (!info.hashMap.empty()) {
            const auto names = scalarCompositeHashNames(info.hashMap);
            phase.extraBuffers.push_back({names.states(), "uint", true, false});
            phase.extraBuffers.push_back({names.keys1(), "uint", true, false});
            phase.extraBuffers.push_back({names.keys2(), "uint", true, false});
            phase.extraBuffers.push_back({names.values(), "uint", true, false});
            addResolvedScalar(names.capacity(), "uint", info.hashCapacityExpr);
        }
        if (!info.countHashMap.empty()) {
            const auto names = scalarCompositeHashNames(info.countHashMap);
            phase.extraBuffers.push_back({names.states(), "uint", true, false});
            phase.extraBuffers.push_back({names.keys1(), "uint", true, false});
            phase.extraBuffers.push_back({names.keys2(), "uint", true, false});
            phase.extraBuffers.push_back({names.values(), "uint", true, false});
            addResolvedScalar(names.capacity(), "uint", info.hashCapacityExpr);
        }
        if (!info.htFlags.empty())
            phase.extraBuffers.push_back({info.htFlags, "uint", true, false});
        if (!info.htKeys.empty())
            phase.extraBuffers.push_back({info.htKeys, "uint", true, false});
        if (!info.htVals.empty())
            phase.extraBuffers.push_back({info.htVals, "uint", true, false});
    }
}

} // namespace codegen
