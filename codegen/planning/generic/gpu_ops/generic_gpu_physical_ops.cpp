#include "generic_gpu_physical_ops.h"

#include "metal_generic_executor.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <optional>
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
            if (col.stringRowRef && !col.stringSourceColumn.empty()) {
                cg.addColumnParam(col.stringSourceColumn, "char",
                                  col.stringSourceTable);
            }
        }

        const std::string state = stateName();
        cg.addAtomicBufferParam(state, "atomic_uint", spec_.capacityExpr);
        const std::string hash = hashName();
        cg.addAtomicBufferParam(hash, "atomic_uint", spec_.capacityExpr);
        const std::string hash2 = hash2Name();
        cg.addAtomicBufferParam(hash2, "atomic_uint", spec_.capacityExpr);
        if (hasStringGroupKey()) {
            cg.addBufferParam(repRowName(), "uint", spec_.capacityExpr, false);
        }

        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            if (col->stringLen > 0) continue;
            std::string sizeExpr = spec_.capacityExpr;
            cg.addBufferParam(keyStoreName(key), col->metalType, sizeExpr, false);
        }

        for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
            const std::string fn = ai < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[ai] : "SUM";
            if (fn == "COUNT_DISTINCT") {
                const auto* col = findMatColumn(spec_.inputColumns, spec_.groupBy.aggColumns[ai]);
                if (!col) continue;
                cg.addAtomicBufferParam(aggName(ai), "atomic_uint", spec_.capacityExpr);
                cg.addAtomicBufferParam(distinctStateName(ai), "atomic_uint",
                                        spec_.capacityExpr);
                cg.addBufferParam(distinctGroupName(ai), "uint", spec_.capacityExpr,
                                  false);
                cg.addBufferParam(distinctValueName(ai), "uint", spec_.capacityExpr,
                                  false);
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
            emitHashExpr2(cg, "_r", "_hash2");
            cg.addLine("uint _fp = _hash | 1u;");
            cg.addLine("uint _fp2 = _hash2 | 1u;");
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
                        cg.addLine("atomic_store_explicit(&" + hash + "[_slot], _fp, memory_order_relaxed);");
                        cg.addLine("atomic_store_explicit(&" + hash2 + "[_slot], _fp2, memory_order_relaxed);");
                        cg.addLine("atomic_store_explicit(&" + state + "[_slot], 2u, memory_order_relaxed);");
                        cg.addLine("_found = _slot;");
                    });
                });
                cg.addIf("_found != 0xFFFFFFFFu", [&]() {
                    cg.addLine("break;");
                });
                cg.addBlock("while (atomic_load_explicit(&" + state + "[_slot], memory_order_relaxed) == 1u)", [&]() {});
                cg.addIf("atomic_load_explicit(&" + state + "[_slot], memory_order_relaxed) == 2u", [&]() {
                    cg.addLine("uint _slot_fp = atomic_load_explicit(&" + hash + "[_slot], memory_order_relaxed);");
                    cg.addLine("uint _slot_fp2 = atomic_load_explicit(&" + hash2 + "[_slot], memory_order_relaxed);");
                    cg.addBlock("for (uint _hvis = 0u; _slot_fp == 0u && _hvis < 256u; ++_hvis)", [&]() {
                        cg.addLine("_slot_fp = atomic_load_explicit(&" + hash + "[_slot], memory_order_relaxed);");
                        cg.addLine("_slot_fp2 = atomic_load_explicit(&" + hash2 + "[_slot], memory_order_relaxed);");
                    });
                    const std::string keyEq = keyEqualsExpr("_slot", "_r");
                    cg.addLine("bool _key_eq = " + keyEq + ";");
                    cg.addBlock("for (uint _vis = 0u; !_key_eq && _vis < 4096u; ++_vis)", [&]() {
                        cg.addLine("_key_eq = " + keyEq + ";");
                    });
                    cg.addIf("_key_eq || (_slot_fp == _fp && _slot_fp2 == _fp2)", [&]() {
                        cg.addLine("_found = _slot;");
                        cg.addLine("break;");
                    });
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
    std::string hashName() const { return "d_gpu_gb_" + suffix() + "_hash"; }
    std::string hash2Name() const { return "d_gpu_gb_" + suffix() + "_hash2"; }
    std::string repRowName() const { return "d_gpu_gb_" + suffix() + "_rep_row"; }
    std::string keyStoreName(const std::string& display) const {
        return "d_gpu_gb_" + suffix() + "_key_" + sanitizeIdentifier(display);
    }
    std::string aggName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai);
    }
    std::string avgCountName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai) + "_count";
    }
    std::string distinctStateName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_distinct_state_" + std::to_string(ai);
    }
    std::string distinctGroupName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_distinct_group_" + std::to_string(ai);
    }
    std::string distinctValueName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_distinct_value_" + std::to_string(ai);
    }
    std::string totalName() const { return "d_gpu_gb_" + suffix() + "_having_total"; }

    bool hasStringGroupKey() const {
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (col && col->stringLen > 0) return true;
        }
        return false;
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

    std::string valueAt(const GenericMatColumnDesc& col, const std::string& row) const {
        if (col.stringLen > 0) return col.bufferName + " + " + row + " * " + std::to_string(col.stringLen);
        return col.bufferName + "[" + row + "]";
    }

    std::string stringPtrAt(const GenericMatColumnDesc& col,
                            const std::string& row) const {
        if (col.stringRowRef) {
            return col.stringSourceColumn + " + " + col.bufferName + "[" + row +
                   "] * " + std::to_string(col.stringLen);
        }
        return col.bufferName + " + " + row + " * " +
               std::to_string(col.stringLen);
    }

    std::string stringByteAt(const GenericMatColumnDesc& col,
                             const std::string& row,
                             const std::string& offset) const {
        if (col.stringRowRef) {
            return col.stringSourceColumn + "[" + col.bufferName + "[" + row +
                   "] * " + std::to_string(col.stringLen) + "u + " + offset + "]";
        }
        return col.bufferName + "[" + row + " * " +
               std::to_string(col.stringLen) + "u + " + offset + "]";
    }

    void emitHashMix(MetalCodegen& cg, const std::string& hashVar, const std::string& valueExpr) const {
        cg.addLine(hashVar + " ^= (uint)(" + valueExpr + ");");
        cg.addLine(hashVar + " *= 16777619u;");
    }

    void emitHashMix2(MetalCodegen& cg, const std::string& hashVar, const std::string& valueExpr) const {
        cg.addLine(hashVar + " += (uint)(" + valueExpr + ") + 0x9e3779b9u + (" +
                   hashVar + " << 6) + (" + hashVar + " >> 2);");
        cg.addLine(hashVar + " ^= " + hashVar + " >> 15;");
        cg.addLine(hashVar + " *= 2246822519u;");
    }

    void emitHashExpr(MetalCodegen& cg, const std::string& row, const std::string& hashVar) const {
        cg.addLine("uint " + hashVar + " = 2166136261u;");
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            if (col->stringLen > 0) {
                cg.addBlock("for (uint _hc = 0; _hc < " + std::to_string(col->stringLen) + "u; ++_hc)", [&]() {
                    emitHashMix(cg, hashVar, stringByteAt(*col, row, "_hc"));
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

    void emitHashExpr2(MetalCodegen& cg, const std::string& row, const std::string& hashVar) const {
        cg.addLine("uint " + hashVar + " = 0x85ebca6bu;");
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            if (col->stringLen > 0) {
                cg.addBlock("for (uint _hc2 = 0; _hc2 < " + std::to_string(col->stringLen) + "u; ++_hc2)", [&]() {
                    emitHashMix2(cg, hashVar, stringByteAt(*col, row, "_hc2"));
                });
            } else if (col->metalType == "float") {
                emitHashMix2(cg, hashVar, "as_type<uint>(" + valueAt(*col, row) + ")");
            } else {
                emitHashMix2(cg, hashVar, valueAt(*col, row));
            }
        }
    }

    void emitStoreKeys(MetalCodegen& cg, const std::string& slot, const std::string& row) const {
        for (const auto& key : spec_.groupBy.keyColumns) {
            const auto* col = findMatColumn(spec_.inputColumns, key);
            if (!col) continue;
            const std::string dst = keyStoreName(key);
            if (col->stringLen > 0) {
                cg.addLine(repRowName() + "[" + slot + "] = (uint)(" + row + ");");
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
                const std::string repRow = repRowName() + "[" + slot + "]";
                part = "gpu_generic_fixed_eq(" + stringPtrAt(*col, repRow) + ", " +
                       stringPtrAt(*col, row) + ", " +
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

    std::string distinctValueExpr(const GenericMatColumnDesc& col,
                                  const std::string& row) const {
        if (col.metalType == "float")
            return "as_type<uint>(" + valueAt(col, row) + ")";
        return "(uint)(" + valueAt(col, row) + ")";
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
                const std::string suffix = std::to_string(ai);
                const std::string dv = "_distinct_value_" + suffix;
                const std::string dh = "_distinct_hash_" + suffix;
                const std::string ds = "_distinct_slot_" + suffix;
                const std::string st = "_distinct_state_" + suffix;
                cg.addLine("uint " + dv + " = " + distinctValueExpr(*col, row) + ";");
                cg.addLine("uint " + dh + " = ((uint)(" + slot + ") * 16777619u) ^ (" +
                           dv + " * 2166136261u);");
                cg.addLine(dh + " ^= " + dh + " >> 16;");
                cg.addLine(dh + " *= 2246822519u;");
                cg.addLine(dh + " ^= " + dh + " >> 13;");
                cg.addLine("uint " + ds + " = " + dh + " & (" +
                           spec_.capacitySymbol + " - 1u);");
                cg.addBlock("for (uint _dprobe_" + suffix + " = 0u; _dprobe_" +
                            suffix + " < " + spec_.capacitySymbol + "; ++_dprobe_" +
                            suffix + ")", [&]() {
                    cg.addLine("uint " + st + " = atomic_load_explicit(&" +
                               distinctStateName(ai) + "[" + ds +
                               "], memory_order_relaxed);");
                    cg.addIf(st + " == 0u", [&]() {
                        cg.addLine("bool _dclaimed_" + suffix + " = false;");
                        cg.addBlock("while (true)", [&]() {
                            cg.addLine("uint _dexpected_" + suffix + " = 0u;");
                            cg.addIf("atomic_compare_exchange_weak_explicit(&" +
                                     distinctStateName(ai) + "[" + ds + "], &_dexpected_" +
                                     suffix + ", 1u, memory_order_relaxed, memory_order_relaxed)", [&]() {
                                cg.addLine("_dclaimed_" + suffix + " = true;");
                                cg.addLine("break;");
                            });
                            cg.addIf("_dexpected_" + suffix + " != 0u", [&]() {
                                cg.addLine("break;");
                            });
                        });
                        cg.addIf("_dclaimed_" + suffix, [&]() {
                            cg.addLine(distinctGroupName(ai) + "[" + ds + "] = (uint)(" +
                                       slot + ");");
                            cg.addLine(distinctValueName(ai) + "[" + ds + "] = " + dv + ";");
                            cg.addLine("atomic_store_explicit(&" + distinctStateName(ai) +
                                       "[" + ds + "], 2u, memory_order_relaxed);");
                            cg.addLine("atomic_fetch_add_explicit(&" + aggName(ai) + "[" +
                                       slot + "], 1u, memory_order_relaxed);");
                            cg.addLine("break;");
                        });
                    });
                    cg.addBlock("while (atomic_load_explicit(&" + distinctStateName(ai) +
                                "[" + ds + "], memory_order_relaxed) == 1u)", [&]() {});
                    cg.addIf("atomic_load_explicit(&" + distinctStateName(ai) + "[" + ds +
                             "], memory_order_relaxed) == 2u", [&]() {
                        cg.addIf(distinctGroupName(ai) + "[" + ds + "] == (uint)(" +
                                 slot + ") && " + distinctValueName(ai) + "[" + ds +
                                 "] == " + dv, [&]() {
                            cg.addLine("break;");
                        });
                    });
                    cg.addLine(ds + " = (" + ds + " + 1u) & (" +
                               spec_.capacitySymbol + " - 1u);");
                });
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
            if (col->stringLen > 0) {
                cg.addBufferParam(repRowName(), "uint", spec_.capacityExpr, false);
                cg.addBufferParam(col->bufferName, col->metalType, "", false);
                if (col->stringRowRef && !col->stringSourceColumn.empty()) {
                    cg.addColumnParam(col->stringSourceColumn, "char",
                                      col->stringSourceTable);
                }
            } else {
                cg.addBufferParam(keyStoreName(key), col->metalType,
                                  spec_.capacityExpr, false);
            }
        }
        for (size_t ai = 0; ai < spec_.groupBy.aggColumns.size(); ++ai) {
            const std::string fn = ai < spec_.groupBy.aggFuncs.size()
                ? spec_.groupBy.aggFuncs[ai] : "SUM";
            cg.addBufferParam(aggName(ai), "atomic_uint", aggSizeExpr(ai), false);
            if (fn == "AVG") {
                cg.addBufferParam(avgCountName(ai), "atomic_uint", spec_.capacityExpr, false);
            }
        }
        if (spec_.groupBy.havingAggIdx >= 0) {
            cg.addBufferParam(totalName(), "atomic_uint", havingTotalSizeExpr(), false);
        }

        for (const auto& out : outputColumns()) {
            std::string sizeExpr = outputCapacityExpr();
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
    std::string repRowName() const { return "d_gpu_gb_" + suffix() + "_rep_row"; }
    std::string keyStoreName(const std::string& display) const {
        return "d_gpu_gb_" + suffix() + "_key_" + sanitizeIdentifier(display);
    }
    std::string aggName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai);
    }
    std::string avgCountName(size_t ai) const {
        return "d_gpu_gb_" + suffix() + "_agg_" + std::to_string(ai) + "_count";
    }
    std::string totalName() const { return "d_gpu_gb_" + suffix() + "_having_total"; }
    std::string outputCapacityExpr() const {
        return spec_.maxOutputRowsExpr.empty()
            ? spec_.capacityExpr
            : spec_.maxOutputRowsExpr;
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

    std::string stringByteAt(const GenericMatColumnDesc& col,
                             const std::string& row,
                             const std::string& offset) const {
        if (col.stringRowRef) {
            return col.stringSourceColumn + "[" + col.bufferName + "[" + row +
                   "] * " + std::to_string(col.stringLen) + "u + " + offset + "]";
        }
        return col.bufferName + "[" + row + " * " +
               std::to_string(col.stringLen) + "u + " + offset + "]";
    }

    void emitCopyKey(MetalCodegen& cg, const GenericMatColumnDesc& out,
                     const GenericMatColumnDesc& keyCol,
                     const std::string& slot, const std::string& pos) const {
        const std::string src = keyStoreName(keyCol.displayName);
        if (keyCol.stringLen > 0) {
            const std::string row = repRowName() + "[" + slot + "]";
            cg.addBlock("for (uint _oc = 0; _oc < " + std::to_string(keyCol.stringLen) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[" + pos + " * " + std::to_string(keyCol.stringLen) +
                           "u + _oc] = " + stringByteAt(keyCol, row, "_oc") + ";");
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
        cg.addLine("uint " + var + " = atomic_load_explicit(&" + aggName(ai) +
                   "[" + slot + "], memory_order_relaxed);");
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
                         std::vector<GenericMatColumnDesc> outputs,
                         std::string bucketCountExpr,
                         std::string bucketCountSymbol,
                         KeyedCompactHavingSpec having)
        : inputBuffer_(std::move(inputBuffer)),
          outputCounter_(std::move(outputCounter)),
          numBuckets_(numBuckets),
          valuesPerBucket_(valuesPerBucket),
          keys_(std::move(keys)),
          aggs_(std::move(aggs)),
          outputs_(std::move(outputs)),
          bucketCountExpr_(std::move(bucketCountExpr)),
          bucketCountSymbol_(std::move(bucketCountSymbol)),
          having_(std::move(having)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string bucketCountExpr = bucketCountExpr_.empty()
            ? std::to_string(numBuckets_)
            : bucketCountExpr_;
        std::string bucketCount;
        if (!bucketCountExpr_.empty()) {
            bucketCount = bucketCountSymbol_.empty()
                ? "n_keyed_compact_buckets"
                : bucketCountSymbol_;
            cg.addResolvedScalarParam(bucketCount, "uint", bucketCountExpr);
        } else {
            bucketCount = std::to_string(numBuckets_);
        }
        const std::string bucketCountLimit = bucketCountExpr_.empty()
            ? bucketCount + "u"
            : bucketCount;
        cg.addBufferParam(inputBuffer_, "atomic_uint",
                          bucketCountExpr + " * " +
                          std::to_string(valuesPerBucket_), false);
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");
        if (!having_.scalarTotalBuffer.empty()) {
            cg.addBufferParam(having_.scalarTotalBuffer, "atomic_uint",
                              having_.scalarAggIsLongPair ? "2" : "1", false);
        }
        for (const auto& key : keys_) {
            if (key.stringRowRef && !key.stringSourceColumn.empty())
                cg.addColumnParam(key.stringSourceColumn, "char",
                                  key.stringSourceTable);
        }
        for (const auto& out : outputs_) {
            std::string sizeExpr = bucketCountExpr;
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

        cg.addBlock("for (uint _bucket = tid; _bucket < " + bucketCountLimit + "; _bucket += tpg)", [&]() {
            cg.addLine("bool _has_data = false;");
            cg.addBlock("for (uint _v = 0; _v < " + std::to_string(valuesPerBucket_) + "u; ++_v)", [&]() {
                cg.addIf("atomic_load_explicit(&" + inputBuffer_ + "[_bucket * " +
                         std::to_string(valuesPerBucket_) + "u + _v], memory_order_relaxed) != 0u", [&]() {
                    cg.addLine("_has_data = true;");
                    cg.addLine("break;");
                });
            });
            cg.addIf("!_has_data", [&]() { cg.addLine("continue;"); });
            emitHavingFilter(cg);
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
    std::string bucketCountExpr_;
    std::string bucketCountSymbol_;
    KeyedCompactHavingSpec having_;

    std::string valueBase() const {
        return "_bucket * " + std::to_string(valuesPerBucket_) + "u";
    }

    void emitKeyWrite(MetalCodegen& cg, const KeyedCompactKeySpec& key, size_t outIdx) const {
        const auto& out = outputs_[outIdx];
        const std::string numValues = key.numValuesExpr.empty()
            ? std::to_string(std::max(1, key.numValues)) + "u"
            : key.numValuesExpr;
        cg.addLine("uint _encoded_" + std::to_string(outIdx) + " = (_bucket / " +
                   std::to_string(key.stride) + "u) % " + numValues + ";");
        if (key.stringRowRef && !key.stringSourceColumn.empty()) {
            const int width = std::max(1, key.stringLen);
            cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(width) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[_pos * " +
                           std::to_string(width) + "u + _oc] = " +
                           key.stringSourceColumn + "[_encoded_" +
                           std::to_string(outIdx) + " * " +
                           std::to_string(width) + "u + _oc];");
            });
        } else if (!key.stringMap.empty()) {
            const int width = std::max(1, key.stringLen);
            for (int ci = 0; ci < width; ++ci) {
                auto charAt = [&](const std::string& value) {
                    return metalCharLiteral(ci < static_cast<int>(value.size())
                        ? value[(size_t)ci]
                        : '\0');
                };
                std::string expr = charAt(key.stringMap.back());
                for (int i = static_cast<int>(key.stringMap.size()) - 2; i >= 0; --i) {
                    expr = "(_encoded_" + std::to_string(outIdx) + " == " +
                           std::to_string(i) + "u ? " + charAt(key.stringMap[(size_t)i]) +
                           " : " + expr + ")";
                }
                cg.addLine(out.bufferName + "[_pos * " + std::to_string(width) +
                           "u + " + std::to_string(ci) + "u] = " + expr + ";");
            }
        } else if (!key.charMap.empty()) {
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

    std::string longPairAsFloatExpr(int offset) const {
        return "((float)" + loadUintAt(offset + 1) +
               " * 4294967296.0f + (float)" + loadUintAt(offset) + ")";
    }

    std::string havingValueExpr(int offset,
                                bool isLongPair,
                                bool isFloatSum,
                                int scaleDown,
                                bool divideScale) const {
        if (isLongPair) {
            std::string expr = longPairAsFloatExpr(offset);
            if (divideScale && scaleDown > 0)
                expr = "(" + expr + " / " + std::to_string(scaleDown) + ".0f)";
            return expr;
        }
        if (isFloatSum)
            return "as_type<float>(" + loadUintAt(offset) + ")";
        return "(float)(" + loadUintAt(offset) + ")";
    }

    std::string totalHavingValueExpr() const {
        if (having_.scalarAggIsLongPair) {
            return "((float)atomic_load_explicit(&" + having_.scalarTotalBuffer +
                   "[1], memory_order_relaxed) * 4294967296.0f + "
                   "(float)atomic_load_explicit(&" + having_.scalarTotalBuffer +
                   "[0], memory_order_relaxed))";
        }
        if (having_.scalarAggIsFloatSum) {
            return "as_type<float>(atomic_load_explicit(&" +
                   having_.scalarTotalBuffer + "[0], memory_order_relaxed))";
        }
        return "(float)atomic_load_explicit(&" + having_.scalarTotalBuffer +
               "[0], memory_order_relaxed)";
    }

    void emitHavingFilter(MetalCodegen& cg) const {
        if (having_.scalarAggOffset >= 0 && !having_.scalarTotalBuffer.empty() &&
            having_.scalarMultiplier >= 0.0) {
            cg.addLine("float _having_value = " +
                       havingValueExpr(having_.scalarAggOffset,
                                       having_.scalarAggIsLongPair,
                                       having_.scalarAggIsFloatSum,
                                       having_.scalarAggScaleDown, false) + ";");
            cg.addLine("float _having_threshold = " + totalHavingValueExpr() +
                       " * " + std::to_string(having_.scalarMultiplier) + "f;");
            const std::string op = having_.scalarCompareOp.empty()
                ? ">"
                : having_.scalarCompareOp;
            cg.addIf("!(_having_value " + op + " _having_threshold)", [&]() {
                cg.addLine("continue;");
            });
        }
        if (having_.compareAggOffset >= 0 && !having_.compareOp.empty()) {
            cg.addLine("float _having_cmp_value = " +
                       havingValueExpr(having_.compareAggOffset,
                                       having_.compareAggIsLongPair,
                                       having_.compareAggIsFloatSum,
                                       having_.compareAggScaleDown, true) + ";");
            cg.addIf("!(_having_cmp_value " + having_.compareOp + " " +
                     std::to_string(having_.compareValue) + "f)", [&]() {
                cg.addLine("continue;");
            });
        }
    }

    void emitAggWrite(MetalCodegen& cg, const KeyedCompactAggSpec& agg, size_t outIdx) const {
        const auto& out = outputs_[outIdx];
        if (agg.isRatio) {
            std::string num = havingValueExpr(agg.offset, agg.isLongPair,
                                              agg.isFloatSum, agg.scaleDown, true);
            std::string den = havingValueExpr(agg.ratioDenOffset,
                                              agg.ratioDenIsLongPair,
                                              agg.ratioDenIsFloatSum,
                                              agg.ratioDenScaleDown, true);
            cg.addLine(out.bufferName + "[_pos] = ((" + den +
                       ") != 0.0f ? (" + num + ") / (" + den + ") : 0.0f);");
            return;
        }
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

class MetalGenericSmallSort : public MetalOperator {
public:
    MetalGenericSmallSort(std::string idxBuffer,
                          std::vector<GenericSortKeySpec> keys,
                          std::string nRowsSymbol,
                          int maxRows)
        : idxBuffer_(std::move(idxBuffer)),
          keys_(std::move(keys)),
          nRowsSymbol_(std::move(nRowsSymbol)),
          maxRows_(maxRows) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.setPhaseMaxThreadgroups(1);
        cg.addScalarParam(nRowsSymbol_, "uint");
        cg.addBufferParam(idxBuffer_, "int", std::to_string(maxRows_), true, 0xFF);
        for (const auto& key : keys_) {
            std::string sizeExpr = std::to_string(maxRows_);
            if (key.column.stringLen > 0)
                sizeExpr += " * " + std::to_string(key.column.stringLen);
            if (key.column.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(key.column.bufferName, key.column.metalType,
                              sizeExpr, false);
        }

        cg.addLine("threadgroup int _idx[" + std::to_string(maxRows_) + "];");
        cg.addLine("uint _n = min((uint)" + nRowsSymbol_ + ", " +
                   std::to_string(maxRows_) + "u);");
        cg.addLine("uint _np2 = 1u;");
        cg.addBlock("while (_np2 < _n)", [&]() {
            cg.addLine("_np2 <<= 1u;");
        });
        cg.addBlock("for (uint _i = lid; _i < _np2; _i += tg_size)", [&]() {
            cg.addLine("_idx[_i] = (_i < _n) ? (int)_i : -1;");
        });
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        cg.addBlock("for (uint _k = 2u; _k <= _np2; _k <<= 1u)", [&]() {
            cg.addBlock("for (uint _j = _k >> 1u; _j > 0u; _j >>= 1u)", [&]() {
                cg.addBlock("for (uint _i = lid; _i < _np2; _i += tg_size)", [&]() {
                    cg.addLine("uint _ixj = _i ^ _j;");
                    cg.addIf("_ixj > _i && _ixj < _np2", [&]() {
                        cg.addLine("bool _asc = (_i & _k) == 0u;");
                        cg.addLine("int _a = _idx[_i];");
                        cg.addLine("int _b = _idx[_ixj];");
                        cg.addLine("int _cmp = 0;");
                        cg.addIf("_a < 0 && _b >= 0", [&]() { cg.addLine("_cmp = 1;"); });
                        cg.addIf("_a >= 0 && _b < 0", [&]() { cg.addLine("_cmp = -1;"); });
                        cg.addIf("_a >= 0 && _b >= 0", [&]() {
                            for (const auto& key : keys_) emitKeyCompare(cg, key);
                        });
                        cg.addLine("bool _swap = _asc ? (_cmp > 0) : (_cmp < 0);");
                        cg.addIf("_swap", [&]() {
                            cg.addLine("_idx[_i] = _b;");
                            cg.addLine("_idx[_ixj] = _a;");
                        });
                    });
                });
                cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            });
        });
        cg.addBlock("for (uint _i = lid; _i < _n; _i += tg_size)", [&]() {
            cg.addLine(idxBuffer_ + "[_i] = _idx[_i];");
        });
        consume();
    }

    std::string describe() const override { return "GenericSmallSort"; }

private:
    std::string idxBuffer_;
    std::vector<GenericSortKeySpec> keys_;
    std::string nRowsSymbol_;
    int maxRows_ = 0;

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

class MetalGenericTopKSelection : public MetalOperator {
public:
    MetalGenericTopKSelection(std::string idxBuffer,
                              std::vector<GenericSortKeySpec> keys,
                              std::string nRowsSymbol,
                              std::string capacityExpr,
                              int limit)
        : idxBuffer_(std::move(idxBuffer)),
          keys_(std::move(keys)),
          nRowsSymbol_(std::move(nRowsSymbol)),
          capacityExpr_(std::move(capacityExpr)),
          limit_(limit) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.setPhaseMaxThreadgroups(1);
        cg.addScalarParam(nRowsSymbol_, "uint");
        cg.addBufferParam(idxBuffer_, "int", std::to_string(limit_), true, 0xFF);
        for (const auto& key : keys_) {
            std::string sizeExpr = capacityExpr_;
            if (key.column.stringLen > 0)
                sizeExpr += " * " + std::to_string(key.column.stringLen);
            if (key.column.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(key.column.bufferName, key.column.metalType,
                              sizeExpr, false);
        }

        cg.addLine("threadgroup int _topk_indices[256];");
        cg.addLine("threadgroup int _prev_idx_tg;");
        cg.addLine("threadgroup int _done_tg;");
        cg.addIf("lid == 0", [&]() {
            cg.addLine("_prev_idx_tg = -1;");
            cg.addLine("_done_tg = 0;");
        });
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        cg.addBlock("for (uint _out = 0; _out < " + std::to_string(limit_) +
                    "u; ++_out)", [&]() {
            cg.addLine("int _local_idx = -1;");
            cg.addIf("_done_tg == 0", [&]() {
                cg.addBlock("for (uint _i = lid; _i < " + nRowsSymbol_ +
                            "; _i += tg_size)", [&]() {
                    cg.addLine("bool _eligible = true;");
                    cg.addIf("_prev_idx_tg >= 0", [&]() {
                        emitRowCompare(cg, "_prev_idx_tg", "_i", "_cmp_prev");
                        cg.addLine("_eligible = (_cmp_prev < 0);");
                    });
                    cg.addIf("_eligible", [&]() {
                        cg.addIf("_local_idx < 0", [&]() {
                            cg.addLine("_local_idx = (int)_i;");
                        });
                        cg.addIf("_local_idx >= 0", [&]() {
                            emitRowCompare(cg, "_i", "_local_idx", "_cmp_local");
                            cg.addIf("_cmp_local < 0", [&]() {
                                cg.addLine("_local_idx = (int)_i;");
                            });
                        });
                    });
                });
            });
            cg.addLine("_topk_indices[lid] = _local_idx;");
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            for (int stride = 128; stride > 0; stride >>= 1) {
                cg.addIf("lid < " + std::to_string(stride) + "u", [&]() {
                    cg.addLine("int _cand_idx = _topk_indices[lid + " +
                               std::to_string(stride) + "u];");
                    cg.addIf("_cand_idx >= 0", [&]() {
                        cg.addIf("_topk_indices[lid] < 0", [&]() {
                            cg.addLine("_topk_indices[lid] = _cand_idx;");
                        });
                        cg.addIf("_topk_indices[lid] >= 0", [&]() {
                            emitRowCompare(cg, "_cand_idx", "_topk_indices[lid]",
                                           "_cmp_reduce");
                            cg.addIf("_cmp_reduce < 0", [&]() {
                                cg.addLine("_topk_indices[lid] = _cand_idx;");
                            });
                        });
                    });
                });
                cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            cg.addIf("lid == 0", [&]() {
                cg.addLine(idxBuffer_ + "[_out] = _topk_indices[0];");
                cg.addIf("_topk_indices[0] >= 0", [&]() {
                    cg.addLine("_prev_idx_tg = _topk_indices[0];");
                });
                cg.addIf("_topk_indices[0] < 0", [&]() {
                    cg.addLine("_done_tg = 1;");
                });
            });
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        });
        consume();
    }

    std::string describe() const override { return "GenericTopKSelection"; }

private:
    std::string idxBuffer_;
    std::vector<GenericSortKeySpec> keys_;
    std::string nRowsSymbol_;
    std::string capacityExpr_;
    int limit_ = 0;

    void emitRowCompare(MetalCodegen& cg,
                        const std::string& aExpr,
                        const std::string& bExpr,
                        const std::string& cmpVar) const {
        cg.addLine("int " + cmpVar + " = 0;");
        for (const auto& key : keys_) {
            emitKeyCompare(cg, key, aExpr, bExpr, cmpVar);
        }
        cg.addIf(cmpVar + " == 0", [&]() {
            cg.addIf("(uint)(" + aExpr + ") < (uint)(" + bExpr + ")", [&]() {
                cg.addLine(cmpVar + " = -1;");
            });
            cg.addIf("(uint)(" + aExpr + ") > (uint)(" + bExpr + ")", [&]() {
                cg.addLine(cmpVar + " = 1;");
            });
        });
    }

    void emitKeyCompare(MetalCodegen& cg,
                        const GenericSortKeySpec& key,
                        const std::string& aExpr,
                        const std::string& bExpr,
                        const std::string& cmpVar) const {
        const auto& c = key.column;
        const std::string aIdx = "(uint)(" + aExpr + ")";
        const std::string bIdx = "(uint)(" + bExpr + ")";
        cg.addIf(cmpVar + " == 0", [&]() {
            if (c.stringLen > 0) {
                cg.addBlock("for (uint _sc = 0; _sc < " + std::to_string(c.stringLen) + "u; ++_sc)", [&]() {
                    cg.addLine("char _ca = " + c.bufferName + "[" + aIdx + " * " +
                               std::to_string(c.stringLen) + "u + _sc];");
                    cg.addLine("char _cb = " + c.bufferName + "[" + bIdx + " * " +
                               std::to_string(c.stringLen) + "u + _sc];");
                    cg.addIf("_ca < _cb", [&]() {
                        cg.addLine(cmpVar + " = " +
                                   std::string(key.descending ? "1" : "-1") + ";");
                        cg.addLine("break;");
                    });
                    cg.addIf("_ca > _cb", [&]() {
                        cg.addLine(cmpVar + " = " +
                                   std::string(key.descending ? "-1" : "1") + ";");
                        cg.addLine("break;");
                    });
                });
            } else if (c.metalType == "float") {
                cg.addLine("float _ka = " + c.bufferName + "[" + aIdx + "];");
                cg.addLine("float _kb = " + c.bufferName + "[" + bIdx + "];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "-1" : "1") + ";");
                });
            } else if (c.isLongPair) {
                cg.addLine("long _ka = (((long)" + c.bufferName +
                           "[" + aIdx + " * 2u + 1u]) << 32) | (long)" +
                           c.bufferName + "[" + aIdx + " * 2u];");
                cg.addLine("long _kb = (((long)" + c.bufferName +
                           "[" + bIdx + " * 2u + 1u]) << 32) | (long)" +
                           c.bufferName + "[" + bIdx + " * 2u];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "-1" : "1") + ";");
                });
            } else {
                cg.addLine(c.metalType + " _ka = " + c.bufferName + "[" + aIdx + "];");
                cg.addLine(c.metalType + " _kb = " + c.bufferName + "[" + bIdx + "];");
                cg.addIf("_ka < _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "1" : "-1") + ";");
                });
                cg.addIf("_ka > _kb", [&]() {
                    cg.addLine(cmpVar + " = " +
                               std::string(key.descending ? "-1" : "1") + ";");
                });
            }
        });
    }
};

class MetalGenericTopKFloatInt : public MetalOperator {
public:
    MetalGenericTopKFloatInt(std::string idxBuffer,
                             GenericSortKeySpec primary,
                             std::optional<GenericSortKeySpec> tie,
                             std::string nRowsSymbol,
                             std::string capacityExpr,
                             int limit)
        : idxBuffer_(std::move(idxBuffer)),
          primary_(std::move(primary)),
          tie_(std::move(tie)),
          nRowsSymbol_(std::move(nRowsSymbol)),
          capacityExpr_(std::move(capacityExpr)),
          limit_(limit) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.setPhaseMaxThreadgroups(1);
        cg.addScalarParam(nRowsSymbol_, "uint");
        cg.addBufferParam(idxBuffer_, "int", std::to_string(limit_), true, 0xFF);
        cg.addBufferParam(primary_.column.bufferName, primary_.column.metalType,
                          capacityExpr_, false);
        if (tie_) {
            cg.addBufferParam(tie_->column.bufferName, tie_->column.metalType,
                              capacityExpr_, false);
        }

        cg.addLine("threadgroup ulong _topk_keys[256];");
        cg.addLine("threadgroup int _topk_indices[256];");
        cg.addLine("ulong _prev_key = 0xfffffffffffffffful;");
        cg.addBlock("for (uint _out = 0; _out < " + std::to_string(limit_) +
                    "u; ++_out)", [&]() {
            cg.addLine("ulong _local_key = 0ul;");
            cg.addLine("int _local_idx = -1;");
            cg.addBlock("for (uint _i = lid; _i < " + nRowsSymbol_ +
                        "; _i += tg_size)", [&]() {
                emitCandidateKey(cg);
                cg.addIf("_key < _prev_key && (_local_idx < 0 || _key > _local_key)", [&]() {
                    cg.addLine("_local_key = _key;");
                    cg.addLine("_local_idx = (int)_i;");
                });
            });
            cg.addLine("_topk_keys[lid] = (_local_idx >= 0 ? _local_key : 0ul);");
            cg.addLine("_topk_indices[lid] = _local_idx;");
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            for (int stride = 128; stride > 0; stride >>= 1) {
                cg.addIf("lid < " + std::to_string(stride) + "u", [&]() {
                    cg.addLine("ulong _cand_key = _topk_keys[lid + " +
                               std::to_string(stride) + "u];");
                    cg.addLine("int _cand_idx = _topk_indices[lid + " +
                               std::to_string(stride) + "u];");
                    cg.addIf("_cand_idx >= 0 && (_topk_indices[lid] < 0 || "
                             "_cand_key > _topk_keys[lid])", [&]() {
                        cg.addLine("_topk_keys[lid] = _cand_key;");
                        cg.addLine("_topk_indices[lid] = _cand_idx;");
                    });
                });
                cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            cg.addIf("lid == 0", [&]() {
                cg.addLine(idxBuffer_ + "[_out] = _topk_indices[0];");
            });
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
            cg.addLine("_prev_key = _topk_keys[0];");
            cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        });
        consume();
    }

    std::string describe() const override { return "GenericTopKFloatInt"; }

private:
    std::string idxBuffer_;
    GenericSortKeySpec primary_;
    std::optional<GenericSortKeySpec> tie_;
    std::string nRowsSymbol_;
    std::string capacityExpr_;
    int limit_ = 0;

    void emitCandidateKey(MetalCodegen& cg) const {
        cg.addLine("float _primary_v = " + primary_.column.bufferName + "[_i];");
        cg.addLine("uint _primary_bits = as_type<uint>(_primary_v);");
        cg.addLine("uint _primary_rank = ((_primary_bits & 0x80000000u) != 0u) ? "
                   "(~_primary_bits) : (_primary_bits ^ 0x80000000u);");
        if (!primary_.descending) cg.addLine("_primary_rank = ~_primary_rank;");
        if (tie_) {
            cg.addLine("uint _tie_raw = uint(" + tie_->column.bufferName + "[_i]);");
        } else {
            cg.addLine("uint _tie_raw = _i;");
        }
        cg.addLine("uint _tie_rank = (_tie_raw ^ 0x80000000u);");
        const bool tieDescending = tie_ && tie_->descending;
        if (!tieDescending) cg.addLine("_tie_rank = ~_tie_rank;");
        cg.addLine("ulong _key = ((ulong)_primary_rank << 32) | (ulong)_tie_rank;");
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
        if (!pso || !idxBuf) return 0.0;
        size_t nRows = 0;
        if (!executor.tryGetSymbol(nRowsSymbol, nRows) || nRows == 0) return 0.0;
        unsigned int n = (unsigned int)nRows;
        unsigned int np2 = MetalInitSortKeys::nextPow2(n);
        int* idxs = static_cast<int*>(idxBuf->contents());
        for (unsigned int i = n; i < np2; ++i) idxs[i] = -1;
        auto* queue = executor.commandQueue();
        if (!queue) return 0.0;
        double gpuMs = 0.0;
        for (unsigned int k = 2; k <= np2; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                auto* cmdBuf = queue->commandBuffer();
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(pso);
                enc->setBuffer(idxBuf, 0, 0);
                int bindIdx = 1;
                for (const auto& key : keys) {
                    auto* keyBuf = executor.getAllocatedBuffer(key.column.bufferName);
                    if (!keyBuf) {
                        enc->endEncoding();
                        cmdBuf->commit();
                        cmdBuf->waitUntilCompleted();
                        gpuMs += (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
                        return gpuMs;
                    }
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
                gpuMs += (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
            }
        }
        return gpuMs;
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
            const std::string outType = col->stringLen > 0 ? "char" : col->metalType;
            out.push_back({display, "d_gpu_gb_" + suffix + "_out_" + sanitizeIdentifier(display),
                           outType, col->stringLen});
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
        if (!buf) return 0.0;
        uint32_t n = *static_cast<const uint32_t*>(buf->contents());
        executor.registerScalarInt(symbolName, (int)n);
        executor.registerSymbol(symbolName, n);
        executor.registerScalarInt(tableSizeName(symbolName), (int)n);
        executor.registerSymbol(tableSizeName(symbolName), n);
        return 0.0;
    };
}

void appendGenericGpuGroupBy(MetalQueryPlan& plan,
                             const GenericGpuGroupSpec& spec) {
    addGenericGpuGroupHelpers(plan);
    appendPhase(plan, "GENERIC_gpu_group_build_" + sanitizeIdentifier(spec.tag),
                std::make_unique<MetalGenericHashGroupBuild>(spec));
    appendPhase(plan, "GENERIC_gpu_group_compact_" + sanitizeIdentifier(spec.tag),
                std::make_unique<MetalGenericHashGroupCompact>(spec));
}

std::unique_ptr<MetalOperator> makeKeyedAggCompactOperator(
    std::string inputBuffer,
    std::string outputCounter,
    int numBuckets,
    int valuesPerBucket,
    std::vector<KeyedCompactKeySpec> keys,
    std::vector<KeyedCompactAggSpec> aggs,
    std::vector<GenericMatColumnDesc> outputs,
    std::string bucketCountExpr,
    std::string bucketCountSymbol,
    KeyedCompactHavingSpec having) {
    return std::make_unique<MetalKeyedAggCompact>(
        std::move(inputBuffer), std::move(outputCounter), numBuckets, valuesPerBucket,
        std::move(keys), std::move(aggs), std::move(outputs),
        std::move(bucketCountExpr), std::move(bucketCountSymbol), std::move(having));
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
    appendPhase(plan, "GENERIC_gpu_sort_init_" + suffix,
                std::make_unique<MetalGenericSortInitIndices>(idxBuf, nRowsSymbol, capacityExpr));
    if (!keys.empty()) {
        const std::string phaseName = "GENERIC_gpu_sort_step_" + suffix;
        auto& sortPhase = appendPhase(plan, phaseName,
            std::make_unique<MetalGenericSortStep>(idxBuf, keys, capacityExpr));
        sortPhase.postDispatchHook = makeGenericSortHook(phaseName, idxBuf, nRowsSymbol, keys);
    }
    plan.gpuSort = MetalQueryPlan::GpuSort{idxBuf, nRowsSymbol, false, sortSpec.limit};
    return true;
}

bool appendGenericGpuSmallSort(MetalQueryPlan& plan,
                               const std::string& tag,
                               const std::string& nRowsSymbol,
                               int maxRows,
                               const std::vector<GenericMatColumnDesc>& columns,
                               const GenericSortSpec& sortSpec,
                               std::string* error) {
    if (maxRows <= 0 || (maxRows & (maxRows - 1)) != 0) {
        if (error) *error = "GPU small sort requires a positive power-of-two maxRows.";
        return false;
    }
    if (sortSpec.keys.empty()) {
        if (error) *error = "GPU small sort requires at least one ORDER BY key.";
        return false;
    }

    std::vector<GenericSortKeySpec> keys;
    for (const auto& sk : sortSpec.keys) {
        const auto* col = findMatColumn(columns, sk.column);
        if (!col) {
            if (error) *error = "GPU small sort key not present in materialized output: " +
                                sk.column;
            return false;
        }
        if (!col->isLongPair && col->stringLen <= 0 &&
            col->metalType != "uint" && col->metalType != "int" &&
            col->metalType != "float") {
            if (error) *error = "GPU small sort currently supports int/uint/float, long-pair, and fixed string keys.";
            return false;
        }
        keys.push_back({*col, sk.descending});
    }

    const std::string suffix = sanitizeIdentifier(tag);
    const std::string idxBuf = "d_gpu_smallsort_idx_" + suffix;
    appendPhase(plan, "GENERIC_gpu_smallsort_" + suffix,
                std::make_unique<MetalGenericSmallSort>(
                    idxBuf, std::move(keys), nRowsSymbol, maxRows),
                256);
    plan.gpuSort = MetalQueryPlan::GpuSort{idxBuf, nRowsSymbol, false, sortSpec.limit};
    return true;
}

bool appendGenericGpuTopKSelection(MetalQueryPlan& plan,
                                   const std::string& tag,
                                   const std::string& nRowsSymbol,
                                   const std::string& capacityExpr,
                                   const std::vector<GenericMatColumnDesc>& columns,
                                   const GenericSortSpec& sortSpec,
                                   std::string* error) {
    if (sortSpec.limit <= 0) {
        if (error) *error = "GPU selection top-k requires a positive LIMIT.";
        return false;
    }
    if (sortSpec.limit > 256) {
        if (error) *error = "GPU selection top-k currently supports LIMIT <= 256.";
        return false;
    }
    if (sortSpec.keys.empty()) {
        if (error) *error = "GPU selection top-k requires at least one ORDER BY key.";
        return false;
    }

    std::vector<GenericSortKeySpec> keys;
    for (const auto& sk : sortSpec.keys) {
        const auto* col = findMatColumn(columns, sk.column);
        if (!col) {
            if (error) *error = "GPU selection top-k key not present in materialized output: " +
                                sk.column;
            return false;
        }
        if (!col->isLongPair && col->stringLen <= 0 &&
            col->metalType != "uint" && col->metalType != "int" &&
            col->metalType != "float") {
            if (error) *error = "GPU selection top-k currently supports int/uint/float, long-pair, and fixed string keys.";
            return false;
        }
        keys.push_back({*col, sk.descending});
    }

    const std::string suffix = sanitizeIdentifier(tag);
    const std::string idxBuf = "d_gpu_topk_select_idx_" + suffix;
    appendPhase(plan, "GENERIC_gpu_topk_select_" + suffix,
                std::make_unique<MetalGenericTopKSelection>(
                    idxBuf, std::move(keys), nRowsSymbol, capacityExpr, sortSpec.limit),
                256);
    plan.gpuSort = MetalQueryPlan::GpuSort{idxBuf, nRowsSymbol, false, sortSpec.limit};
    return true;
}

bool appendGenericGpuTopK(MetalQueryPlan& plan,
                          const std::string& tag,
                          const std::string& nRowsSymbol,
                          const std::string& capacityExpr,
                          const std::vector<GenericMatColumnDesc>& columns,
                          const GenericSortSpec& sortSpec,
                          std::string* error) {
    if (sortSpec.limit <= 0) {
        if (error) *error = "GPU top-k requires a positive LIMIT.";
        return false;
    }
    if (sortSpec.keys.empty() || sortSpec.keys.size() > 2) {
        if (error) *error = "GPU top-k currently supports one float key plus one optional integer tie key.";
        return false;
    }

    const auto* primaryCol = findMatColumn(columns, sortSpec.keys[0].column);
    if (!primaryCol) {
        if (error) *error = "GPU top-k primary key not present in materialized output: " +
                            sortSpec.keys[0].column;
        return false;
    }
    if (primaryCol->metalType != "float" || primaryCol->stringLen > 0 ||
        primaryCol->isLongPair) {
        if (error) *error = "GPU top-k primary key must be a plain float column.";
        return false;
    }

    std::optional<GenericSortKeySpec> tie;
    if (sortSpec.keys.size() == 2) {
        const auto* tieCol = findMatColumn(columns, sortSpec.keys[1].column);
        if (!tieCol) {
            if (error) *error = "GPU top-k tie key not present in materialized output: " +
                                sortSpec.keys[1].column;
            return false;
        }
        if ((tieCol->metalType != "int" && tieCol->metalType != "uint") ||
            tieCol->stringLen > 0 || tieCol->isLongPair) {
            if (error) *error = "GPU top-k tie key must be a plain integer column.";
            return false;
        }
        tie = GenericSortKeySpec{*tieCol, sortSpec.keys[1].descending};
    }

    const std::string suffix = sanitizeIdentifier(tag);
    const std::string idxBuf = "d_gpu_topk_idx_" + suffix;
    appendPhase(plan, "GENERIC_gpu_topk_" + suffix,
                std::make_unique<MetalGenericTopKFloatInt>(
                    idxBuf,
                    GenericSortKeySpec{*primaryCol, sortSpec.keys[0].descending},
                    std::move(tie), nRowsSymbol, capacityExpr, sortSpec.limit),
                256);
    plan.gpuSort = MetalQueryPlan::GpuSort{idxBuf, nRowsSymbol, false, sortSpec.limit};
    return true;
}

} // namespace codegen
