#include "metal_generic_sql_physical_ops.h"

#include "metal_generic_executor.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <utility>

namespace codegen {
namespace {

struct GenericSortKeySpec {
    GenericMatColumnDesc column;
    bool descending = false;
};

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front()))) {
        name = "c_" + name;
    }
    return name;
}

const GenericMatColumnDesc* findMatColumn(
        const std::vector<GenericMatColumnDesc>& cols,
        const std::string& displayName) {
    for (const auto& c : cols) {
        if (c.displayName == displayName) return &c;
    }
    return nullptr;
}

void addGenericGpuGroupHelpers(MetalQueryPlan& plan) {
    const std::string marker = "gpu_generic_fixed_eq";
    for (const auto& h : plan.helpers) {
        if (h.find(marker) != std::string::npos) return;
    }
    plan.helpers.push_back(R"(
inline bool gpu_generic_fixed_eq(const device char* a, const device char* b, uint len) {
    for (uint i = 0; i < len; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}
)");
}

class MetalGenericHashGroupBuild : public MetalOperator {
public:
    explicit MetalGenericHashGroupBuild(GenericGpuGroupSpec spec)
        : spec_(std::move(spec)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addScalarParam(spec_.inputRowsSymbol, "uint");
        cg.addResolvedScalarParam(spec_.capacitySymbol, "uint", spec_.capacityExpr);

        for (const auto& col : spec_.inputColumns) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
        }

        const std::string state = stateName();
        cg.addAtomicBufferParam(state, "atomic_uint", spec_.capacityExpr);

        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            std::string sizeExpr = spec_.capacityExpr;
            if (col->stringLen > 0)
                sizeExpr += " * " + std::to_string(col->stringLen);
            cg.addBufferParam(keyStoreName(key), col->metalType, sizeExpr, false);
        }

        for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
            const std::string fn = ai < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[ai] : "SUM";
            if (fn == "COUNT_DISTINCT") {
                const auto* col = findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai]);
                if (!col) continue;
                cg.addResolvedScalarParam(distinctDomainParamName(ai), "uint",
                                          distinctDomainExpr(ai));
                cg.addAtomicBufferParam(aggName(ai), "atomic_uint", spec_.capacityExpr);
                cg.addAtomicBufferParam(distinctBitmapName(ai), "atomic_uint",
                                        distinctBitmapSizeExpr(ai));
            } else if (fn == "AVG") {
                cg.addAtomicBufferParam(aggName(ai), "atomic_uint", aggSizeExpr(ai));
                cg.addAtomicBufferParam(avgCountName(ai), "atomic_uint", spec_.capacityExpr);
            } else {
                cg.addAtomicBufferParam(aggName(ai), "atomic_uint", aggSizeExpr(ai));
            }
        }
        if (spec_.groupBy.havingAggIdx >= 0) {
            cg.addAtomicBufferParam(totalName(), "atomic_uint", havingTotalSizeExpr());
        }

        cg.addBlock("for (uint _r = tid; _r < " + spec_.inputRowsSymbol + "; _r += tpg)", [&]() {
            emitHashExpr(cg, "_r", "_hash");
            cg.addLine("uint _slot = _hash & (" + spec_.capacitySymbol + " - 1u);");
            cg.addLine("uint _found = 0xFFFFFFFFu;");
            cg.addBlock("for (uint _probe = 0u; _probe < " + spec_.capacitySymbol + "; ++_probe)", [&]() {
                cg.addLine("uint _st = atomic_load_explicit(&" + state + "[_slot], memory_order_relaxed);");
                cg.addIf("_st == 0u", [&]() {
                    cg.addLine("bool _claimed = false;");
                    cg.addBlock("while (true)", [&]() {
                        cg.addLine("uint _expected = 0u;");
                        cg.addIf("atomic_compare_exchange_weak_explicit(&" + state +
                                 "[_slot], &_expected, 1u, memory_order_relaxed, memory_order_relaxed)", [&]() {
                            cg.addLine("_claimed = true;");
                            cg.addLine("break;");
                        });
                        cg.addIf("_expected != 0u", [&]() {
                            cg.addLine("break;");
                        });
                    });
                    cg.addIf("_claimed", [&]() {
                        emitStoreKeys(cg, "_slot", "_r");
                        cg.addLine("atomic_store_explicit(&" + state + "[_slot], 2u, memory_order_relaxed);");
                        cg.addLine("_found = _slot;");
                    });
                });
                cg.addIf("_found != 0xFFFFFFFFu", [&]() {
                    cg.addLine("break;");
                });
                cg.addBlock("while (atomic_load_explicit(&" + state + "[_slot], memory_order_relaxed) == 1u)", [&]() {});
                cg.addIf("atomic_load_explicit(&" + state + "[_slot], memory_order_relaxed) == 2u && " +
                         keyEqualsExpr("_slot", "_r"), [&]() {
                    cg.addLine("_found = _slot;");
                    cg.addLine("break;");
                });
                cg.addLine("_slot = (_slot + 1u) & (" + spec_.capacitySymbol + " - 1u);");
            });
            cg.addIf("_found != 0xFFFFFFFFu", [&]() {
                emitAggregateUpdates(cg, "_found", "_r");
            });
            consume();
        });
    }

    std::string describe() const override { return "GenericHashGroupBuild"; }

private:
    GenericGpuGroupSpec spec_;

    std::string suffix() const { return sanitizeIdentifier(spec_.tag); }
    std::string stateName() const { return "d_gpu_gb_" + suffix() + "_state"; }
    std::string keyStoreName(const std::string& display) const {
        return "d_gpu_gb_" + suffix() + "_key_" + sanitizeIdentifier(display);
    }
    std::string aggName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai);
    }
    std::string avgCountName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai) + "_count";
    }
    std::string distinctBitmapName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_distinct_" + std::to_string(ai);
    }
    std::string distinctDomainParamName(size_t ai) const {
        return "n_gpu_gb_" + suffix() + "_distinct_domain_" + std::to_string(ai);
    }
    std::string distinctDomainExpr(size_t ai) const {
        const auto* col = ai < spec_.groupBy.aggColumns.size()
            ? findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai])
            : nullptr;
        return (col && !col->distinctDomainSymbol.empty())
            ? col->distinctDomainSymbol
            : "0";
    }
    std::string distinctBitmapSizeExpr(size_t ai) const {
        return spec_.capacityExpr + " * " + distinctDomainExpr(ai) + " / 32 + " +
               spec_.capacityExpr + " * 2";
    }
    std::string totalName() const { return "d_gpu_gb_" + suffix() + "_having_total"; }

    std::string fnAt(size_t ai) const {
        return ai < spec_.groupBy.aggFuncs.size() ? spec_.groupBy.aggFuncs[ai] : "SUM";
    }

    const GenericMatColumnDesc* aggInputColumn(size_t ai) const {
        return ai < spec_.groupBy.aggColumns.size()
            ? findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai])
            : nullptr;
    }

    int aggScale(size_t ai) const {
        const auto* col = aggInputColumn(ai);
        const std::string fn = fnAt(ai);
        if (!col || (fn != "SUM" && fn != "AVG")) return 0;
        return col->scaleDown;
    }

    std::string aggSizeExpr(size_t ai) const {
        return aggScale(ai) > 0 ? (spec_.capacityExpr + " * 2") : spec_.capacityExpr;
    }

    std::string havingTotalSizeExpr() const {
        return spec_.groupBy.havingAggIdx >= 0 &&
               aggScale((size_t)spec_.groupBy.havingAggIdx) > 0 ? "2" : "1";
    }

    std::string distinctStrideExpr(size_t ai) const {
        return "((" + distinctDomainParamName(ai) + " + 32) / 32)";
    }

    std::string valueAt(const GenericMatColumnDesc& col, const std::string& row) const {
        if (col.stringLen > 0) return col.bufferName + " + " + row + " * " + std::to_string(col.stringLen);
        return col.bufferName + "[" + row + "]";
    }

    void emitHashMix(MetalCodegen& cg, const std::string& hashVar, const std::string& valueExpr) const {
        cg.addLine(hashVar + " ^= (uint)(" + valueExpr + ");");
        cg.addLine(hashVar + " *= 16777619u;");
    }

    void emitHashExpr(MetalCodegen& cg, const std::string& row, const std::string& hashVar) const {
        cg.addLine("uint " + hashVar + " = 2166136261u;");
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            if (col->stringLen > 0) {
                cg.addBlock("for (uint _hc = 0; _hc < " + std::to_string(col->stringLen) + "u; ++_hc)", [&]() {
                    emitHashMix(cg, hashVar, col->bufferName + "[" + row + " * " +
                                std::to_string(col->stringLen) + "u + _hc]");
                });
            } else if (col->metalType == "float") {
                emitHashMix(cg, hashVar, "as_type<uint>(" + valueAt(*col, row) + ")");
            } else if (col->metalType == "char") {
                emitHashMix(cg, hashVar, valueAt(*col, row));
            } else {
                emitHashMix(cg, hashVar, valueAt(*col, row));
            }
        }
    }

    void emitStoreKeys(MetalCodegen& cg, const std::string& slot, const std::string& row) const {
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            const std::string dst = keyStoreName(key);
            if (col->stringLen > 0) {
                cg.addBlock("for (uint _kc = 0; _kc < " + std::to_string(col->stringLen) + "u; ++_kc)", [&]() {
                    cg.addLine(dst + "[" + slot + " * " + std::to_string(col->stringLen) +
                               "u + _kc] = " + col->bufferName + "[" + row + " * " +
                               std::to_string(col->stringLen) + "u + _kc];");
                });
            } else {
                cg.addLine(dst + "[" + slot + "] = " + valueAt(*col, row) + ";");
            }
        }
    }

    std::string keyEqualsExpr(const std::string& slot, const std::string& row) const {
        std::string expr = "true";
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            const std::string dst = keyStoreName(key);
            std::string part;
            if (col->stringLen > 0) {
                part = "gpu_generic_fixed_eq(" + dst + " + " + slot + " * " +
                       std::to_string(col->stringLen) + ", " + col->bufferName + " + " +
                       row + " * " + std::to_string(col->stringLen) + ", " +
                       std::to_string(col->stringLen) + ")";
            } else {
                part = "(" + dst + "[" + slot + "] == " + valueAt(*col, row) + ")";
            }
            expr = "(" + expr + " && " + part + ")";
        }
        return expr;
    }

    std::string numericInputExpr(size_t ai, const std::string& row) const {
        const auto* col = ai < spec_.groupBy.aggColumns.size()
            ? findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai])
            : nullptr;
        if (!col) return "0.0f";
        if (col->metalType == "float") return col->bufferName + "[" + row + "]";
        return "(float)(" + col->bufferName + "[" + row + "])";
    }

    std::string scaledInputExpr(size_t ai, const std::string& row) const {
        int scale = aggScale(ai);
        return "(long)round((" + numericInputExpr(ai, row) + ") * " +
               std::to_string(scale) + ".0f)";
    }

    void emitAtomicAddScaled(MetalCodegen& cg,
                             const std::string& buffer,
                             const std::string& slotExpr,
                             const std::string& valueExpr) const {
        cg.addLine("atomic_add_long_pair(&" + buffer + "[" + slotExpr +
                   " * 2u], &" + buffer + "[" + slotExpr +
                   " * 2u + 1u], " + valueExpr + ");");
    }

    void emitAggregateUpdates(MetalCodegen& cg, const std::string& slot, const std::string& row) const {
        for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
            const std::string fn = ai < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[ai] : "SUM";
            if (fn == "COUNT") {
                cg.addLine("atomic_add_float(&" + aggName(ai) + "[" + slot + "], 1.0f);");
            } else if (fn == "COUNT_DISTINCT") {
                const auto* col = findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai]);
                if (!col) continue;
                const std::string v = col->bufferName + "[" + row + "]";
                const std::string stride = distinctStrideExpr(ai);
                cg.addLine("atomic_fetch_or_explicit(&" + distinctBitmapName(ai) + "[" +
                           slot + " * " + stride + " + (((uint)(" + v + ")) >> 5)], "
                           "1u << (((uint)(" + v + ")) & 31u), memory_order_relaxed);");
            } else if (fn == "MIN") {
                cg.addLine("atomic_min_float(&" + aggName(ai) + "[" + slot + "], " +
                           numericInputExpr(ai, row) + ");");
            } else if (fn == "MAX") {
                cg.addLine("atomic_max_float(&" + aggName(ai) + "[" + slot + "], " +
                           numericInputExpr(ai, row) + ");");
            } else if (fn == "AVG") {
                if (aggScale(ai) > 0) {
                    emitAtomicAddScaled(cg, aggName(ai), slot, scaledInputExpr(ai, row));
                } else {
                    cg.addLine("atomic_add_float(&" + aggName(ai) + "[" + slot + "], " +
                               numericInputExpr(ai, row) + ");");
                }
                cg.addLine("atomic_fetch_add_explicit(&" + avgCountName(ai) + "[" + slot +
                           "], 1u, memory_order_relaxed);");
            } else if (aggScale(ai) > 0) {
                emitAtomicAddScaled(cg, aggName(ai), slot, scaledInputExpr(ai, row));
            } else {
                cg.addLine("atomic_add_float(&" + aggName(ai) + "[" + slot + "], " +
                           numericInputExpr(ai, row) + ");");
            }
            if ((int)ai == spec_.groupBy.havingAggIdx && spec_.groupBy.havingMultiplier > 0.0) {
                if (aggScale(ai) > 0) {
                    cg.addLine("atomic_add_long_pair(&" + totalName() + "[0], &" +
                               totalName() + "[1], " + scaledInputExpr(ai, row) + ");");
                } else {
                    cg.addLine("atomic_add_float(&" + totalName() + "[0], " +
                               numericInputExpr(ai, row) + ");");
                }
            }
        }
    }
};

class MetalGenericHashGroupCompact : public MetalOperator {
public:
    explicit MetalGenericHashGroupCompact(GenericGpuGroupSpec spec)
        : spec_(std::move(spec)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addResolvedScalarParam(spec_.capacitySymbol, "uint", spec_.capacityExpr);
        cg.addBufferParam(stateName(), "atomic_uint", spec_.capacityExpr, false);
        cg.addAtomicBufferParam(spec_.outputCounter, "atomic_uint", "1");

        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            std::string sizeExpr = spec_.capacityExpr;
            if (col->stringLen > 0) sizeExpr += " * " + std::to_string(col->stringLen);
            cg.addBufferParam(keyStoreName(key), col->metalType, sizeExpr, false);
        }
        for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
            const std::string fn = ai < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[ai] : "SUM";
            cg.addBufferParam(aggName(ai), "atomic_uint", aggSizeExpr(ai), false);
            if (fn == "AVG") {
                cg.addBufferParam(avgCountName(ai), "atomic_uint", spec_.capacityExpr, false);
            }
            if (fn == "COUNT_DISTINCT") {
                cg.addResolvedScalarParam(distinctDomainParamName(ai), "uint",
                                          distinctDomainExpr(ai));
                cg.addBufferParam(distinctBitmapName(ai), "atomic_uint",
                                  distinctBitmapSizeExpr(ai), false);
            }
        }
        if (spec_.groupBy.havingAggIdx >= 0) {
            cg.addBufferParam(totalName(), "atomic_uint", havingTotalSizeExpr(), false);
        }

        for (const auto& out : outputColumns()) {
            std::string sizeExpr = spec_.capacityExpr;
            if (out.stringLen > 0) sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair) sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
        }

        cg.registerMaterializeOutput(spec_.outputCounter);
        for (const auto& out : outputColumns()) {
            cg.registerOutputColumn(out.displayName, out.bufferName, out.metalType,
                                    out.stringLen, out.scaleDown, out.isLongPair);
        }

        cg.addBlock("for (uint _slot = tid; _slot < " + spec_.capacitySymbol + "; _slot += tpg)", [&]() {
            cg.addIf("atomic_load_explicit(&" + stateName() + "[_slot], memory_order_relaxed) == 2u", [&]() {
                if (spec_.groupBy.havingAggIdx >= 0 && spec_.groupBy.havingMultiplier > 0.0) {
                    const int h = spec_.groupBy.havingAggIdx;
                    if (aggScale((size_t)h) > 0) {
                        cg.addLine("float _hv = " + longPairAsFloatExpr(aggName((size_t)h), "_slot") + ";");
                        cg.addLine("float _tot = " + longPairAsFloatExpr(totalName(), "0u") + ";");
                        cg.addLine("float _threshold = _tot * " +
                                   std::to_string(spec_.groupBy.havingMultiplier) + "f;");
                    } else {
                        cg.addLine("uint _hv_raw = atomic_load_explicit(&" + aggName((size_t)h) +
                                   "[_slot], memory_order_relaxed);");
                        cg.addLine("float _hv = as_type<float>(_hv_raw);");
                        cg.addLine("uint _tot_raw = atomic_load_explicit(&" + totalName() +
                                   "[0], memory_order_relaxed);");
                        cg.addLine("float _threshold = as_type<float>(_tot_raw) * " +
                                   std::to_string(spec_.groupBy.havingMultiplier) + "f;");
                    }
                    const std::string scalarOp = spec_.groupBy.havingScalarCompareOp.empty()
                        ? ">"
                        : spec_.groupBy.havingScalarCompareOp;
                    cg.addIf("!(_hv " + scalarOp + " _threshold)", [&]() {
                        cg.addLine("continue;");
                    });
                }
                if (spec_.groupBy.havingCompareAggIdx >= 0 &&
                    !spec_.groupBy.havingCompareOp.empty()) {
                    const int h = spec_.groupBy.havingCompareAggIdx;
                    cg.addLine("float _having_value = " +
                               aggregateValueExprForComparison((size_t)h, "_slot") + ";");
                    cg.addIf("!(_having_value " + spec_.groupBy.havingCompareOp + " " +
                             std::to_string(spec_.groupBy.havingCompareValue) + "f)", [&]() {
                        cg.addLine("continue;");
                    });
                }
                cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + spec_.outputCounter +
                           "[0], 1u, memory_order_relaxed);");
                emitOutputWrites(cg, "_slot", "_pos");
            });
            consume();
        });
    }

    std::string describe() const override { return "GenericHashGroupCompact"; }

private:
    GenericGpuGroupSpec spec_;

    std::string suffix() const { return sanitizeIdentifier(spec_.tag); }
    std::string stateName() const { return "d_gpu_gb_" + suffix() + "_state"; }
    std::string keyStoreName(const std::string& display) const {
        return "d_gpu_gb_" + suffix() + "_key_" + sanitizeIdentifier(display);
    }
    std::string aggName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai);
    }
    std::string avgCountName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai) + "_count";
    }
    std::string distinctBitmapName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_distinct_" + std::to_string(ai);
    }
    std::string distinctDomainParamName(size_t ai) const {
        return "n_gpu_gb_" + suffix() + "_distinct_domain_" + std::to_string(ai);
    }
    std::string distinctDomainExpr(size_t ai) const {
        const auto* col = ai < spec_.groupBy.aggColumns.size()
            ? findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai])
            : nullptr;
        return (col && !col->distinctDomainSymbol.empty())
            ? col->distinctDomainSymbol
            : "0";
    }
    std::string distinctBitmapSizeExpr(size_t ai) const {
        return spec_.capacityExpr + " * " + distinctDomainExpr(ai) + " / 32 + " +
               spec_.capacityExpr + " * 2";
    }
    std::string totalName() const { return "d_gpu_gb_" + suffix() + "_having_total"; }
    std::string distinctStrideExpr(size_t ai) const {
        return "((" + distinctDomainParamName(ai) + " + 32) / 32)";
    }

    std::string fnAt(size_t ai) const {
        return ai < spec_.groupBy.aggFuncs.size() ? spec_.groupBy.aggFuncs[ai] : "SUM";
    }

    const GenericMatColumnDesc* aggInputColumn(size_t ai) const {
        return ai < spec_.groupBy.aggColumns.size()
            ? findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai])
            : nullptr;
    }

    int aggScale(size_t ai) const {
        const auto* col = aggInputColumn(ai);
        const std::string fn = fnAt(ai);
        if (!col || (fn != "SUM" && fn != "AVG")) return 0;
        return col->scaleDown;
    }

    std::string aggSizeExpr(size_t ai) const {
        return aggScale(ai) > 0 ? (spec_.capacityExpr + " * 2") : spec_.capacityExpr;
    }

    std::string havingTotalSizeExpr() const {
        return spec_.groupBy.havingAggIdx >= 0 &&
               aggScale((size_t)spec_.groupBy.havingAggIdx) > 0 ? "2" : "1";
    }

    std::string longPairAsFloatExpr(const std::string& buffer,
                                    const std::string& slot) const {
        return "((float)atomic_load_explicit(&" + buffer + "[" + slot +
               " * 2u + 1u], memory_order_relaxed) * 4294967296.0f + "
               "(float)atomic_load_explicit(&" + buffer + "[" + slot +
               " * 2u], memory_order_relaxed))";
    }

    std::vector<GenericMatColumnDesc> outputColumns() const {
        return genericGpuGroupOutputColumns(spec_);
    }

    void emitCopyKey(MetalCodegen& cg, const GenericMatColumnDesc& out,
                     const GenericMatColumnDesc& keyCol,
                     const std::string& slot, const std::string& pos) const {
        const std::string src = keyStoreName(keyCol.displayName);
        if (keyCol.stringLen > 0) {
            cg.addBlock("for (uint _oc = 0; _oc < " + std::to_string(keyCol.stringLen) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[" + pos + " * " + std::to_string(keyCol.stringLen) +
                           "u + _oc] = " + src + "[" + slot + " * " +
                           std::to_string(keyCol.stringLen) + "u + _oc];");
            });
        } else {
            cg.addLine(out.bufferName + "[" + pos + "] = " + src + "[" + slot + "];");
        }
    }

    std::string aggregateValueExpr(size_t ai, const std::string& slot) const {
        const std::string fn = ai < spec_.groupBy.aggFuncs.size()
            ? spec_.groupBy.aggFuncs[ai] : "SUM";
        if (fn == "COUNT_DISTINCT") {
            return "_distinct_val_" + std::to_string(ai);
        }
        if (fn == "AVG") {
            std::string sum = aggScale(ai) > 0
                ? ("(" + longPairAsFloatExpr(aggName(ai), slot) + " / " +
                   std::to_string(aggScale(ai)) + ".0f)")
                : ("as_type<float>(atomic_load_explicit(&" + aggName(ai) +
                   "[" + slot + "], memory_order_relaxed))");
            std::string cnt = "(float)atomic_load_explicit(&" + avgCountName(ai) +
                              "[" + slot + "], memory_order_relaxed)";
            return "((" + cnt + ") > 0.0f ? (" + sum + ") / (" + cnt + ") : 0.0f)";
        }
        if (fn == "RATIO") {
            size_t den = ai + 1;
            std::string num = "as_type<float>(atomic_load_explicit(&" + aggName(ai) +
                              "[" + slot + "], memory_order_relaxed))";
            std::string dv = den < spec_.groupBy.aggColumns.size()
                ? "as_type<float>(atomic_load_explicit(&" + aggName(den) +
                  "[" + slot + "], memory_order_relaxed))"
                : "0.0f";
            return "((" + dv + ") != 0.0f ? (" + num + ") / (" + dv + ") : 0.0f)";
        }
        if (aggScale(ai) > 0) {
            return longPairAsFloatExpr(aggName(ai), slot);
        }
        return "as_type<float>(atomic_load_explicit(&" + aggName(ai) +
               "[" + slot + "], memory_order_relaxed))";
    }

    std::string aggregateValueExprForComparison(size_t ai, const std::string& slot) const {
        const std::string fn = fnAt(ai);
        if (fn == "SUM" && aggScale(ai) > 0) {
            return "(" + longPairAsFloatExpr(aggName(ai), slot) + " / " +
                   std::to_string(aggScale(ai)) + ".0f)";
        }
        return aggregateValueExpr(ai, slot);
    }

    void emitDistinctCount(MetalCodegen& cg, size_t ai, const std::string& slot) const {
        const std::string var = "_distinct_val_" + std::to_string(ai);
        const std::string stride = distinctStrideExpr(ai);
        cg.addLine("uint " + var + " = 0u;");
        cg.addBlock("for (uint _dw = 0; _dw < " + stride + "; ++_dw)", [&]() {
            cg.addLine(var + " += popcount(atomic_load_explicit(&" + distinctBitmapName(ai) +
                       "[" + slot + " * " + stride + " + _dw], memory_order_relaxed));");
        });
    }

    void emitOutputWrites(MetalCodegen& cg, const std::string& slot, const std::string& pos) const {
        auto outs = outputColumns();
        std::set<size_t> emittedDistinct;
        for (const auto& out : outs) {
            if (std::find(spec_.groupBy.keyColumns.begin(), spec_.groupBy.keyColumns.end(),
                          out.displayName) != spec_.groupBy.keyColumns.end()) {
                const auto* keyCol = findMatColumn(spec_.inputColumns, out.displayName);
                if (keyCol) emitCopyKey(cg, out, *keyCol, slot, pos);
                continue;
            }
            size_t aggIdx = spec_.groupBy.aggColumns.size();
            for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
                if (spec_.groupBy.aggColumns[ai] == out.displayName) { aggIdx = ai; break; }
            }
            if (aggIdx >= spec_.groupBy.aggColumns.size()) continue;
            const std::string fn = aggIdx < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[aggIdx] : "";
            if (fn == "COUNT_DISTINCT" && !emittedDistinct.count(aggIdx)) {
                emitDistinctCount(cg, aggIdx, slot);
                emittedDistinct.insert(aggIdx);
            }
            if (out.isLongPair) {
                cg.addLine(out.bufferName + "[" + pos + " * 2u] = atomic_load_explicit(&" +
                           aggName(aggIdx) + "[" + slot + " * 2u], memory_order_relaxed);");
                cg.addLine(out.bufferName + "[" + pos + " * 2u + 1u] = atomic_load_explicit(&" +
                           aggName(aggIdx) + "[" + slot + " * 2u + 1u], memory_order_relaxed);");
            } else {
                cg.addLine(out.bufferName + "[" + pos + "] = " + aggregateValueExpr(aggIdx, slot) + ";");
            }
        }
    }
};

std::string metalCharLiteral(char c) {
    if (c == '\\') return "'\\\\'";
    if (c == '\'') return "'\\''";
    if (c == '\0') return "'\\0'";
    return std::string("'") + c + "'";
}

class MetalKeyedAggCompact : public MetalOperator {
public:
    MetalKeyedAggCompact(std::string inputBuffer,
                         std::string outputCounter,
                         int numBuckets,
                         int valuesPerBucket,
                         std::vector<KeyedCompactKeySpec> keys,
                         std::vector<KeyedCompactAggSpec> aggs,
                         std::vector<GenericMatColumnDesc> outputs)
        : inputBuffer_(std::move(inputBuffer)),
          outputCounter_(std::move(outputCounter)),
          numBuckets_(numBuckets),
          valuesPerBucket_(valuesPerBucket),
          keys_(std::move(keys)),
          aggs_(std::move(aggs)),
          outputs_(std::move(outputs)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string bucketCount = std::to_string(numBuckets_);
        cg.addBufferParam(inputBuffer_, "atomic_uint",
                          std::to_string(numBuckets_ * valuesPerBucket_), false);
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");
        for (const auto& out : outputs_) {
            std::string sizeExpr = bucketCount;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
        }

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& out : outputs_) {
            cg.registerOutputColumn(out.displayName, out.bufferName,
                                    out.metalType, out.stringLen, out.scaleDown,
                                    out.isLongPair);
        }

        cg.addBlock("for (uint _bucket = tid; _bucket < " + bucketCount + "u; _bucket += tpg)", [&]() {
            cg.addLine("bool _has_data = false;");
            cg.addBlock("for (uint _v = 0; _v < " + std::to_string(valuesPerBucket_) + "u; ++_v)", [&]() {
                cg.addIf("atomic_load_explicit(&" + inputBuffer_ + "[_bucket * " +
                         std::to_string(valuesPerBucket_) + "u + _v], memory_order_relaxed) != 0u", [&]() {
                    cg.addLine("_has_data = true;");
                    cg.addLine("break;");
                });
            });
            cg.addIf("!_has_data", [&]() { cg.addLine("continue;"); });
            cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + outputCounter_ +
                       "[0], 1u, memory_order_relaxed);");
            for (size_t ki = 0; ki < keys_.size(); ++ki) {
                emitKeyWrite(cg, keys_[ki], ki);
            }
            for (size_t ai = 0; ai < aggs_.size(); ++ai) {
                emitAggWrite(cg, aggs_[ai], keys_.size() + ai);
            }
            consume();
        });
    }

    std::string describe() const override { return "KeyedAggCompact"; }

private:
    std::string inputBuffer_;
    std::string outputCounter_;
    int numBuckets_ = 0;
    int valuesPerBucket_ = 0;
    std::vector<KeyedCompactKeySpec> keys_;
    std::vector<KeyedCompactAggSpec> aggs_;
    std::vector<GenericMatColumnDesc> outputs_;

    std::string valueBase() const {
        return "_bucket * " + std::to_string(valuesPerBucket_) + "u";
    }

    void emitKeyWrite(MetalCodegen& cg, const KeyedCompactKeySpec& key, size_t outIdx) const {
        const auto& out = outputs_[outIdx];
        cg.addLine("uint _encoded_" + std::to_string(outIdx) + " = (_bucket / " +
                   std::to_string(key.stride) + "u) % " +
                   std::to_string(std::max(1, key.numValues)) + "u;");
        if (!key.charMap.empty()) {
            std::string expr = metalCharLiteral(key.charMap.back());
            for (int i = (int)key.charMap.size() - 2; i >= 0; --i) {
                expr = "(_encoded_" + std::to_string(outIdx) + " == " +
                       std::to_string(i) + "u ? " + metalCharLiteral(key.charMap[(size_t)i]) +
                       " : " + expr + ")";
            }
            cg.addLine(out.bufferName + "[_pos] = " + expr + ";");
        } else {
            cg.addLine(out.bufferName + "[_pos] = (int)_encoded_" +
                       std::to_string(outIdx) + " + " + std::to_string(key.keyBase) + ";");
        }
    }

    std::string loadUintAt(int offset) const {
        return "atomic_load_explicit(&" + inputBuffer_ + "[" + valueBase() +
               " + " + std::to_string(offset) + "u], memory_order_relaxed)";
    }

    void emitAggWrite(MetalCodegen& cg, const KeyedCompactAggSpec& agg, size_t outIdx) const {
        const auto& out = outputs_[outIdx];
        if (agg.isAvg) {
            if (agg.avgSumIsLongPair) {
                cg.addLine("float _avg_sum_" + std::to_string(outIdx) +
                           " = (float)(" + loadUintAt(agg.offset + 1) +
                           ") * 4294967296.0f + (float)(" + loadUintAt(agg.offset) + ");");
                if (agg.scaleDown < -1) {
                    cg.addLine("_avg_sum_" + std::to_string(outIdx) +
                               " /= " + std::to_string(-agg.scaleDown) + ".0f;");
                }
            } else if (agg.isFloatSum) {
                cg.addLine("float _avg_sum_" + std::to_string(outIdx) +
                           " = as_type<float>(" + loadUintAt(agg.offset) + ");");
            } else {
                cg.addLine("float _avg_sum_" + std::to_string(outIdx) +
                           " = (float)(" + loadUintAt(agg.offset) + ");");
            }
            if (agg.countIsFloat) {
                cg.addLine("float _avg_cnt_" + std::to_string(outIdx) +
                           " = as_type<float>(" + loadUintAt(agg.countOffset) + ");");
            } else {
                cg.addLine("float _avg_cnt_" + std::to_string(outIdx) +
                           " = (float)(" + loadUintAt(agg.countOffset) + ");");
            }
            cg.addLine(out.bufferName + "[_pos] = (_avg_cnt_" + std::to_string(outIdx) +
                       " > 0.0f ? _avg_sum_" + std::to_string(outIdx) +
                       " / _avg_cnt_" + std::to_string(outIdx) + " : 0.0f);");
            return;
        }

        if (agg.isLongPair) {
            cg.addLine(out.bufferName + "[_pos * 2u] = " + loadUintAt(agg.offset) + ";");
            cg.addLine(out.bufferName + "[_pos * 2u + 1u] = " + loadUintAt(agg.offset + 1) + ";");
        } else if (agg.isFloatSum || agg.isMinMax) {
            cg.addLine(out.bufferName + "[_pos] = as_type<float>(" +
                       loadUintAt(agg.offset) + ");");
        } else {
            cg.addLine(out.bufferName + "[_pos] = " + loadUintAt(agg.offset) + ";");
        }
    }
};

class MetalGenericSortInitIndices : public MetalOperator {
public:
    MetalGenericSortInitIndices(std::string idxBuffer, std::string nRowsSymbol,
                                std::string capacityExpr)
        : idxBuffer_(std::move(idxBuffer)),
          nRowsSymbol_(std::move(nRowsSymbol)),
          capacityExpr_(std::move(capacityExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addScalarParam(nRowsSymbol_, "uint");
        cg.addBufferParam(idxBuffer_, "int", "next_pow2(" + capacityExpr_ + ")", true, 0xFF);
        cg.addBlock("for (uint _i = tid; _i < " + nRowsSymbol_ + "; _i += tpg)", [&]() {
            cg.addLine(idxBuffer_ + "[_i] = (int)_i;");
        });
        consume();
    }

    std::string describe() const override { return "GenericSortInitIndices"; }

private:
    std::string idxBuffer_;
    std::string nRowsSymbol_;
    std::string capacityExpr_;
};

class MetalGenericSortStep : public MetalOperator {
public:
    MetalGenericSortStep(std::string idxBuffer, std::vector<GenericSortKeySpec> keys,
                         std::string capacityExpr)
        : idxBuffer_(std::move(idxBuffer)),
          keys_(std::move(keys)),
          capacityExpr_(std::move(capacityExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(idxBuffer_, "int", "next_pow2(" + capacityExpr_ + ")", false);
        for (const auto& key : keys_) {
            std::string sizeExpr = capacityExpr_;
            if (key.column.stringLen > 0)
                sizeExpr += " * " + std::to_string(key.column.stringLen);
            if (key.column.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(key.column.bufferName, key.column.metalType, sizeExpr, false);
        }
        cg.addScalarParam("sort_k", "uint");
        cg.addScalarParam("sort_j", "uint");
        cg.addScalarParam("n_sort", "uint");

        cg.addLine("uint _i = tid;");
        cg.addLine("uint _ixj = _i ^ sort_j;");
        cg.addLine("if (_ixj > _i && _ixj < n_sort) {");
        cg.increaseIndent();
        cg.addLine("bool _asc = (_i & sort_k) == 0;");
        cg.addLine("int _a = " + idxBuffer_ + "[_i];");
        cg.addLine("int _b = " + idxBuffer_ + "[_ixj];");
        cg.addLine("int _cmp = 0;");
        cg.addIf("_a < 0 && _b >= 0", [&]() { cg.addLine("_cmp = 1;"); });
        cg.addIf("_a >= 0 && _b < 0", [&]() { cg.addLine("_cmp = -1;"); });
        cg.addIf("_a >= 0 && _b >= 0", [&]() {
            for (const auto& key : keys_) {
                emitKeyCompare(cg, key);
            }
        });
        cg.addLine("bool _swap = _asc ? (_cmp > 0) : (_cmp < 0);");
        cg.addIf("_swap", [&]() {
            cg.addLine(idxBuffer_ + "[_i] = _b;");
            cg.addLine(idxBuffer_ + "[_ixj] = _a;");
        });
        cg.decreaseIndent();
        cg.addLine("}");
        consume();
    }

    std::string describe() const override { return "GenericSortStep"; }

private:
    std::string idxBuffer_;
    std::vector<GenericSortKeySpec> keys_;
    std::string capacityExpr_;

    void emitKeyCompare(MetalCodegen& cg, const GenericSortKeySpec& key) const {
        const auto& c = key.column;
        cg.addIf("_cmp == 0", [&]() {
            if (c.stringLen > 0) {
                cg.addBlock("for (uint _sc = 0; _sc < " + std::to_string(c.stringLen) + "u; ++_sc)", [&]() {
                    cg.addLine("char _ca = " + c.bufferName + "[(uint)_a * " +
                               std::to_string(c.stringLen) + "u + _sc];");
                    cg.addLine("char _cb = " + c.bufferName + "[(uint)_b * " +
                               std::to_string(c.stringLen) + "u + _sc];");
                    cg.addIf("_ca < _cb", [&]() {
                        cg.addLine("_cmp = " + std::string(key.descending ? "1" : "-1") + ";");
                        cg.addLine("break;");
                    });
                    cg.addIf("_ca > _cb", [&]() {
                        cg.addLine("_cmp = " + std::string(key.descending ? "-1" : "1") + ";");
                        cg.addLine("break;");
                    });
                });
            } else if (c.metalType == "float") {
                cg.addLine("float _ka = " + c.bufferName + "[(uint)_a];");
                cg.addLine("float _kb = " + c.bufferName + "[(uint)_b];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "-1" : "1") + ";");
                });
            } else if (c.isLongPair) {
                cg.addLine("long _ka = (((long)" + c.bufferName +
                           "[(uint)_a * 2u + 1u]) << 32) | (long)" +
                           c.bufferName + "[(uint)_a * 2u];");
                cg.addLine("long _kb = (((long)" + c.bufferName +
                           "[(uint)_b * 2u + 1u]) << 32) | (long)" +
                           c.bufferName + "[(uint)_b * 2u];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "-1" : "1") + ";");
                });
            } else {
                cg.addLine(c.metalType + " _ka = " + c.bufferName + "[(uint)_a];");
                cg.addLine(c.metalType + " _kb = " + c.bufferName + "[(uint)_b];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine("_cmp = " + std::string(key.descending ? "-1" : "1") + ";");
                });
            }
        });
    }
};

PostDispatchHook makeGenericSortHook(
        const std::string& sortPhaseName,
        const std::string& sortIdxBufName,
        const std::string& nRowsSymbol,
        const std::vector<GenericSortKeySpec>& keys) {
    return [=](MetalGenericExecutor& executor) {
        auto* pso = executor.getPipelineState(sortPhaseName);
        auto* idxBuf = executor.getAllocatedBuffer(sortIdxBufName);
        if (!pso || !idxBuf) return;
        size_t nRows = 0;
        if (!executor.tryGetSymbol(nRowsSymbol, nRows) || nRows == 0) return;
        unsigned int n = (unsigned int)nRows;
        unsigned int np2 = MetalInitSortKeys::nextPow2(n);
        int* idxs = static_cast<int*>(idxBuf->contents());
        for (unsigned int i = n; i < np2; ++i) idxs[i] = -1;
        auto* queue = executor.commandQueue();
        if (!queue) return;
        for (unsigned int k = 2; k <= np2; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                auto* cmdBuf = queue->commandBuffer();
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(pso);
                enc->setBuffer(idxBuf, 0, 0);
                int bindIdx = 1;
                for (const auto& key : keys) {
                    auto* keyBuf = executor.getAllocatedBuffer(key.column.bufferName);
                    if (!keyBuf) { enc->endEncoding(); cmdBuf->commit(); cmdBuf->waitUntilCompleted(); return; }
                    enc->setBuffer(keyBuf, 0, bindIdx++);
                }
                enc->setBytes(&k, sizeof(uint), bindIdx++);
                enc->setBytes(&j, sizeof(uint), bindIdx++);
                enc->setBytes(&np2, sizeof(uint), bindIdx++);
                uint tgSize = pso->maxTotalThreadsPerThreadgroup();
                if (tgSize > 256) tgSize = 256;
                uint numTG = (np2 + tgSize - 1) / tgSize;
                if (numTG < 1) numTG = 1;
                enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1),
                                          MTL::Size::Make(tgSize, 1, 1));
                enc->endEncoding();
                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();
            }
        }
    };
}

} // namespace

std::vector<GenericMatColumnDesc> genericGpuGroupOutputColumns(
        const GenericGpuGroupSpec& spec) {
    std::vector<GenericMatColumnDesc> out;
    std::set<std::string> seen;
    const std::string suffix = sanitizeIdentifier(spec.tag);
    auto isGroupKey = [&](const std::string& display) {
        return std::find(spec.groupBy.keyColumns.begin(), spec.groupBy.keyColumns.end(),
                         display) != spec.groupBy.keyColumns.end();
    };
    auto aggIndex = [&](const std::string& display) -> int {
        for (size_t ai = 0; ai < spec.groupBy.aggColumns.size(); ++ai)
            if (spec.groupBy.aggColumns[ai] == display) return (int)ai;
        return -1;
    };
    auto appendDisplay = [&](const std::string& display) {
        const auto* col = findMatColumn(spec.inputColumns, display);
        if (!col) return;
        int ai = aggIndex(display);
        const std::string fn = ai >= 0 && (size_t)ai < spec.groupBy.aggFuncs.size()
            ? spec.groupBy.aggFuncs[(size_t)ai] : "";
        if (display.rfind("__hidden_", 0) == 0 || fn == "RATIO_DEN") return;
        if (seen.count(display)) return;
        if (isGroupKey(display)) {
            out.push_back({display, "d_gpu_gb_" + suffix + "_out_" + sanitizeIdentifier(display),
                           col->metalType, col->stringLen});
        } else if (ai >= 0) {
            bool longPair = false;
            int scaleDown = 0;
            std::string outType = "float";
            if (fn == "COUNT_DISTINCT") {
                outType = "uint";
            } else if (fn == "SUM" && col->scaleDown > 0) {
                outType = "uint";
                scaleDown = col->scaleDown;
                longPair = true;
            }
            out.push_back({display, "d_gpu_gb_" + suffix + "_out_" + sanitizeIdentifier(display),
                           outType, 0, scaleDown, longPair});
        }
        seen.insert(display);
    };

    for (const auto& display : spec.groupBy.outputColumns) {
        appendDisplay(display);
    }
    for (const auto& col : spec.inputColumns) {
        appendDisplay(col.displayName);
    }
    return out;
}

void attachMaterializedCountHook(MetalQueryPlan::Phase& phase,
                                 std::string counterName,
                                 std::string symbolName) {
    phase.postDispatchHook = [counterName = std::move(counterName),
                              symbolName = std::move(symbolName)](MetalGenericExecutor& executor) {
        auto* buf = executor.getAllocatedBuffer(counterName);
        if (!buf) return;
        uint32_t n = *static_cast<const uint32_t*>(buf->contents());
        executor.registerScalarInt(symbolName, (int)n);
        executor.registerSymbol(symbolName, n);
    };
}

void appendGenericGpuGroupBy(MetalQueryPlan& plan,
                             const GenericGpuGroupSpec& spec) {
    addGenericGpuGroupHelpers(plan);
    appendPhase(plan, "ADHOC_gpu_group_build_" + sanitizeIdentifier(spec.tag),
                std::make_unique<MetalGenericHashGroupBuild>(spec));
    appendPhase(plan, "ADHOC_gpu_group_compact_" + sanitizeIdentifier(spec.tag),
                std::make_unique<MetalGenericHashGroupCompact>(spec));
}

std::unique_ptr<MetalOperator> makeKeyedAggCompactOperator(
    std::string inputBuffer,
    std::string outputCounter,
    int numBuckets,
    int valuesPerBucket,
    std::vector<KeyedCompactKeySpec> keys,
    std::vector<KeyedCompactAggSpec> aggs,
    std::vector<GenericMatColumnDesc> outputs) {
    return std::make_unique<MetalKeyedAggCompact>(
        std::move(inputBuffer), std::move(outputCounter), numBuckets, valuesPerBucket,
        std::move(keys), std::move(aggs), std::move(outputs));
}

bool appendGenericGpuSort(MetalQueryPlan& plan,
                          const std::string& tag,
                          const std::string& nRowsSymbol,
                          const std::string& capacityExpr,
                          const std::vector<GenericMatColumnDesc>& columns,
                          const GenericSortSpec& sortSpec,
                          std::string* error) {
    std::vector<GenericSortKeySpec> keys;
    for (const auto& sk : sortSpec.keys) {
        const auto* col = findMatColumn(columns, sk.column);
        if (!col) {
            if (error) *error = "GPU sort key not present in materialized output: " + sk.column;
            return false;
        }
        keys.push_back({*col, sk.descending});
    }
    const std::string suffix = sanitizeIdentifier(tag);
    const std::string idxBuf = "d_gpu_sort_idx_" + suffix;
    appendPhase(plan, "ADHOC_gpu_sort_init_" + suffix,
                std::make_unique<MetalGenericSortInitIndices>(idxBuf, nRowsSymbol, capacityExpr));
    if (!keys.empty()) {
        const std::string phaseName = "ADHOC_gpu_sort_step_" + suffix;
        auto& sortPhase = appendPhase(plan, phaseName,
            std::make_unique<MetalGenericSortStep>(idxBuf, keys, capacityExpr));
        sortPhase.postDispatchHook = makeGenericSortHook(phaseName, idxBuf, nRowsSymbol, keys);
    }
    plan.gpuSort = MetalQueryPlan::GpuSort{idxBuf, nRowsSymbol, false, sortSpec.limit};
    return true;
}

} // namespace codegen
