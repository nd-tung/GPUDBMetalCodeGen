#include "generic/lowering/generic_multi_table_join_lowering.h"

#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_relation_analysis.h"
#include "metal_plan_common.h"
#include "core/schema_provider.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace codegen {

template <typename Shape>
std::optional<Shape> shapeFail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

bool isSimpleIdentifier(const std::string& value) {
    if (value.empty()) return false;
    for (char ch : value) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') return false;
    }
    return true;
}

bool domainKeyListExistsEnabled() {
    const char* value = std::getenv("GPUDB_DOMAIN_KEYLIST_EXISTS");
    return value && value[0] != '\0' && value[0] != '0';
}

CmpOp reverseCmpOp(CmpOp op) {
    switch (op) {
        case CmpOp::LT: return CmpOp::GT;
        case CmpOp::LE: return CmpOp::GE;
        case CmpOp::GT: return CmpOp::LT;
        case CmpOp::GE: return CmpOp::LE;
        case CmpOp::EQ: return CmpOp::EQ;
        case CmpOp::NE: return CmpOp::NE;
    }
    return op;
}

struct IrJoinColumns {
    GenericColumnExpr left;
    GenericColumnExpr right;
};

struct IrJoinEdgeCandidate {
    std::vector<IrJoinColumns> columns;
    std::vector<GenericPredicatePtr> predicates;
    std::set<int> semiInnerRelationInstances;
    bool semiJoinFilter = false;
    bool antiJoinFilter = false;
    size_t index = 0;
};

bool extractEqJoinColumnsFromPredicate(const GenericPredicatePtr& pred,
                                       IrJoinColumns& out) {
    if (!pred) return false;
    if (auto* cmp = std::get_if<GenericComparisonPred>(&pred->node)) {
        if (cmp->op != CmpOp::EQ) return false;
        auto* left = cmp->left ? std::get_if<GenericColumnExpr>(&cmp->left->node) : nullptr;
        auto* right = cmp->right ? std::get_if<GenericColumnExpr>(&cmp->right->node) : nullptr;
        if (!left || !right) return false;
        out.left = *left;
        out.right = *right;
        return true;
    }
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op != GenericLogicalPred::Op::And) return false;
        for (const auto& child : logical->children) {
            if (extractEqJoinColumnsFromPredicate(child, out))
                return true;
        }
    }
    return false;
}

struct IrExistsDistinctInfo {
    GenericColumnExpr childValueCol;
    GenericColumnExpr parentValueCol;
    std::string firstBuffer;
    std::string stateBuffer;
    std::string multiBitmap;
    bool anti = false;
};

struct IrSiblingBitmapFilter {
    std::string bitmapName;
    std::string keyColumn;
    int sourceRelationInstance = -1;
};

struct IrDomainKeyDrive {
    int sourceRelationInstance = -1;
    std::string keyListBuffer;
    std::string keyListCountBuffer;
    std::string dispatchTable;
};

bool typeCanUseArrayCarry(DataType type) {
    return type == DataType::INT || type == DataType::DATE ||
           type == DataType::FLOAT || type == DataType::CHAR1 ||
           type == DataType::CHAR_FIXED;
}

std::string encodeHashCarryValue(const GenericColumnExpr& col,
                                 const std::string& expr) {
    switch (col.type.type) {
        case DataType::FLOAT: return "as_type<uint>(" + expr + ")";
        case DataType::CHAR1: return "(uint)(" + expr + ")";
        case DataType::INT:
        case DataType::DATE:
        default: return "(uint)(" + expr + ")";
    }
}

std::string hashLookupResultType(const GenericColumnExpr& col) {
    return metalTypeForType(col.type);
}

std::string carryVarName(const GenericColumnExpr& col) {
    std::string scope = !col.alias.empty() ? col.alias : col.table;
    return "_ir_carry_" + sanitizeIdentifier(scope) + "_" +
           sanitizeIdentifier(col.column);
}

std::string carryRowVarName(const GenericColumnExpr& col) {
    return carryVarName(col) + "_row";
}

std::string carryBufferName(const GenericColumnExpr& col) {
    std::string scope = !col.alias.empty() ? col.alias : col.table;
    return "d_ir_carry_" + sanitizeIdentifier(scope) + "_" +
           sanitizeIdentifier(col.column);
}

std::string carryKey(const GenericColumnExpr& col) {
    return std::to_string(col.relationInstance.value) + ":" + col.column;
}

std::string carryStorageBufferName(const GenericScanDetail& storage,
                                   const GenericColumnExpr& col) {
    std::string storageScope = !storage.alias.empty() ? storage.alias : storage.table;
    std::string originScope = !col.alias.empty() ? col.alias : col.table;
    return "d_ir_carry_" + sanitizeIdentifier(storageScope) + "_" +
           sanitizeIdentifier(originScope) + "_" + sanitizeIdentifier(col.column);
}

std::string existsDistinctBufferPrefix(const GenericScanDetail& scan,
                                       const GenericColumnExpr& valueCol) {
    std::string scope = !scan.alias.empty() ? scan.alias : scan.table;
    return "d_ir_exists_" + sanitizeIdentifier(scope) + "_" +
           sanitizeIdentifier(valueCol.column);
}

std::string rowRangeFirstBufferName(const std::string& table,
                                    const std::string& keyColumn) {
    return "d_ir_row_first_" + sanitizeIdentifier(table) + "_" +
           sanitizeIdentifier(keyColumn);
}

std::string rowRangeLastBufferName(const std::string& table,
                                   const std::string& keyColumn) {
    return "d_ir_row_last_" + sanitizeIdentifier(table) + "_" +
           sanitizeIdentifier(keyColumn);
}

std::string rowRangeIndexKey(const std::string& table,
                             const std::string& keyColumn) {
    return table + ":" + keyColumn;
}

bool typeCanUseExistsDistinct(DataType type) {
    return type == DataType::INT || type == DataType::DATE ||
           type == DataType::CHAR1;
}

void collectPredicateColumnsForRelation(const GenericPredicatePtr& pred,
                                        GenericRelationInstanceId relationInstance,
                                        std::map<std::string, GenericColumnExpr>& out);

void collectColumnsForRelation(const GenericExprPtr& expr,
                               GenericRelationInstanceId relationInstance,
                               std::map<std::string, GenericColumnExpr>& out) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (node.relationInstance.value == relationInstance.value)
                out[node.column] = node;
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectColumnsForRelation(node.left, relationInstance, out);
            collectColumnsForRelation(node.right, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectPredicateColumnsForRelation(branch.condition, relationInstance, out);
                collectColumnsForRelation(branch.result, relationInstance, out);
            }
            collectColumnsForRelation(node.elseResult, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectColumnsForRelation(arg, relationInstance, out);
        }
    }, expr->node);
}

void collectPredicateColumnsForRelation(const GenericPredicatePtr& pred,
                                        GenericRelationInstanceId relationInstance,
                                        std::map<std::string, GenericColumnExpr>& out) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectColumnsForRelation(node.left, relationInstance, out);
            collectColumnsForRelation(node.right, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectColumnsForRelation(node.expr, relationInstance, out);
            collectColumnsForRelation(node.low, relationInstance, out);
            collectColumnsForRelation(node.high, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectColumnsForRelation(node.expr, relationInstance, out);
            for (const auto& value : node.values)
                collectColumnsForRelation(value, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectColumnsForRelation(node.expr, relationInstance, out);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectPredicateColumnsForRelation(child, relationInstance, out);
        }
    }, pred->node);
}

struct IrBuildSide {
    const GenericRelNode* scanNode = nullptr;
    const GenericScanDetail* scan = nullptr;
    const GenericRelation* relation = nullptr;
    int relationInstance = -1;
    int parentRelationInstance = -1;
    GenericColumnExpr joinCol;
    GenericColumnExpr parentCol;
    GenericColumnExpr joinCol2;
    GenericColumnExpr parentCol2;
    bool useHashJoin = false;
    bool semiJoinFilter = false;
    bool antiJoinFilter = false;
    std::vector<int> children;
    std::vector<GenericPredicatePtr> filters;
    std::map<std::string, IrCarryColumn> localCarries;
    std::vector<IrCarryColumn> subtreeCarries;
    std::vector<IrSiblingBitmapFilter> siblingBitmapFilters;
    std::string keyDomain;
    std::string bitmapName;
    std::optional<IrExistsDistinctInfo> existsDistinct;
    bool emitKeyList = false;
    std::optional<IrDomainKeyDrive> domainKeyDrive;
};

struct IrScanSide {
    const GenericRelNode* node = nullptr;
    const GenericScanDetail* scan = nullptr;
    const GenericRelation* relation = nullptr;
};

std::string buildScopeName(const IrBuildSide& build) {
    return sanitizeIdentifier(build.scan && !build.scan->alias.empty()
        ? build.scan->alias
        : (build.scan ? build.scan->table : std::string("rel")));
}

std::string keyListBufferName(const IrBuildSide& build) {
    return "d_ir_keylist_" + buildScopeName(build);
}

std::string keyListCountBufferName(const IrBuildSide& build) {
    return keyListBufferName(build) + "_count";
}

class MetalIrExistsDistinctBuild : public MetalUnaryOperator {
public:
    MetalIrExistsDistinctBuild(std::unique_ptr<MetalOperator> child,
                               std::string firstBuffer,
                               std::string stateBuffer,
                               std::string multiBitmap,
                               std::string keyExpr,
                               std::string valueExpr,
                               std::string sizeExpr)
        : MetalUnaryOperator(std::move(child)),
          firstBuffer_(std::move(firstBuffer)),
          stateBuffer_(std::move(stateBuffer)),
          multiBitmap_(std::move(multiBitmap)),
          keyExpr_(std::move(keyExpr)),
          valueExpr_(std::move(valueExpr)),
          sizeExpr_(std::move(sizeExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addAtomicBufferParam(firstBuffer_, "atomic_uint", sizeExpr_);
        cg.addAtomicBufferParam(stateBuffer_, "atomic_uint", sizeExpr_);
        cg.addBitmapWriteParam(multiBitmap_, "(" + sizeExpr_ + " + 31) / 32");

        const std::string suffix = sanitizeIdentifier(firstBuffer_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_exists_key_" + suffix + " = (uint)(" + keyExpr_ + ");");
            cg.addLine("uint _ir_exists_val_" + suffix + " = (uint)(" + valueExpr_ + ");");
            cg.addLine("while (true) {");
            cg.addLine("    uint _ir_exists_state_" + suffix + " = atomic_load_explicit(&" +
                       stateBuffer_ + "[_ir_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("    if (_ir_exists_state_" + suffix + " == 0u) {");
            cg.addLine("        uint _ir_exists_expected_" + suffix + " = 0u;");
            cg.addLine("        if (atomic_compare_exchange_weak_explicit(&" + stateBuffer_ +
                       "[_ir_exists_key_" + suffix + "], &_ir_exists_expected_" + suffix +
                       ", 1u, memory_order_relaxed, memory_order_relaxed)) {");
            cg.addLine("            atomic_store_explicit(&" + firstBuffer_ +
                       "[_ir_exists_key_" + suffix + "], _ir_exists_val_" + suffix +
                       ", memory_order_relaxed);");
            cg.addLine("            atomic_store_explicit(&" + stateBuffer_ +
                       "[_ir_exists_key_" + suffix + "], 2u, memory_order_relaxed);");
            cg.addLine("            break;");
            cg.addLine("        }");
            cg.addLine("    } else if (_ir_exists_state_" + suffix + " == 2u) {");
            cg.addLine("        uint _ir_exists_first_" + suffix + " = atomic_load_explicit(&" +
                       firstBuffer_ + "[_ir_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("        if (_ir_exists_first_" + suffix + " != _ir_exists_val_" +
                       suffix + ") bitmap_set(" + multiBitmap_ + ", _ir_exists_key_" +
                       suffix + ");");
            cg.addLine("        break;");
            cg.addLine("    }");
            cg.addLine("}");
            consume();
        });
    }

    std::string describe() const override {
        return "IrExistsDistinctBuild(" + firstBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string firstBuffer_;
    std::string stateBuffer_;
    std::string multiBitmap_;
    std::string keyExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
};

class MetalIrExistsDistinctMultiBuild : public MetalUnaryOperator {
public:
    struct Target {
        std::string firstBuffer;
        std::string stateBuffer;
        std::string multiBitmap;
        std::string valueExpr;
        std::string predicate;
    };

    MetalIrExistsDistinctMultiBuild(std::unique_ptr<MetalOperator> child,
                                    std::string keyExpr,
                                    std::string keyDomain,
                                    std::vector<Target> targets)
        : MetalUnaryOperator(std::move(child)),
          keyExpr_(std::move(keyExpr)),
          keyDomain_(std::move(keyDomain)),
          targets_(std::move(targets)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string keySuffix = sanitizeIdentifier(keyExpr_);
        for (const auto& target : targets_) {
            cg.addAtomicBufferParam(target.firstBuffer, "atomic_uint",
                                    keyDomain_);
            cg.addAtomicBufferParam(target.stateBuffer, "atomic_uint",
                                    keyDomain_);
            cg.addBitmapWriteParam(target.multiBitmap,
                                   "(" + keyDomain_ + " + 31) / 32");
        }

        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_exists_key_" + keySuffix +
                       " = (uint)(" + keyExpr_ + ");");
            for (const auto& target : targets_) {
                auto emitUpdate = [&]() {
                    const std::string targetSuffix =
                        sanitizeIdentifier(target.firstBuffer);
                    cg.addLine("uint _ir_exists_val_" + targetSuffix +
                               " = (uint)(" + target.valueExpr + ");");
                    cg.addLine("while (true) {");
                    cg.addLine("    uint _ir_exists_state_" + targetSuffix +
                               " = atomic_load_explicit(&" +
                               target.stateBuffer + "[_ir_exists_key_" +
                               keySuffix + "], memory_order_relaxed);");
                    cg.addLine("    if (_ir_exists_state_" + targetSuffix +
                               " == 0u) {");
                    cg.addLine("        uint _ir_exists_expected_" +
                               targetSuffix + " = 0u;");
                    cg.addLine("        if (atomic_compare_exchange_weak_explicit(&" +
                               target.stateBuffer + "[_ir_exists_key_" +
                               keySuffix + "], &_ir_exists_expected_" +
                               targetSuffix +
                               ", 1u, memory_order_relaxed, memory_order_relaxed)) {");
                    cg.addLine("            atomic_store_explicit(&" +
                               target.firstBuffer + "[_ir_exists_key_" +
                               keySuffix + "], _ir_exists_val_" +
                               targetSuffix + ", memory_order_relaxed);");
                    cg.addLine("            atomic_store_explicit(&" +
                               target.stateBuffer + "[_ir_exists_key_" +
                               keySuffix + "], 2u, memory_order_relaxed);");
                    cg.addLine("            break;");
                    cg.addLine("        }");
                    cg.addLine("    } else if (_ir_exists_state_" +
                               targetSuffix + " == 2u) {");
                    cg.addLine("        uint _ir_exists_first_" +
                               targetSuffix +
                               " = atomic_load_explicit(&" +
                               target.firstBuffer + "[_ir_exists_key_" +
                               keySuffix + "], memory_order_relaxed);");
                    cg.addLine("        if (_ir_exists_first_" + targetSuffix +
                               " != _ir_exists_val_" + targetSuffix +
                               ") bitmap_set(" + target.multiBitmap +
                               ", _ir_exists_key_" + keySuffix + ");");
                    cg.addLine("        break;");
                    cg.addLine("    }");
                    cg.addLine("}");
                };
                if (target.predicate.empty() || target.predicate == "true") {
                    emitUpdate();
                } else {
                    cg.addIf(target.predicate, emitUpdate);
                }
            }
            consume();
        });
    }

    std::string describe() const override {
        return "IrExistsDistinctMultiBuild(" +
               std::to_string(targets_.size()) + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        for (const auto& target : targets_) {
            appendIUsFromExpr(target.valueExpr, out);
            appendIUsFromExpr(target.predicate, out);
        }
    }

private:
    std::string keyExpr_;
    std::string keyDomain_;
    std::vector<Target> targets_;
};

class MetalBitmapBuildWithKeyList : public MetalUnaryOperator {
public:
    MetalBitmapBuildWithKeyList(std::unique_ptr<MetalOperator> child,
                                std::string bitmapName,
                                std::string keyExpr,
                                std::string bitmapSizeExpr,
                                std::string keyListBuffer,
                                std::string keyListCountBuffer,
                                std::string keyListCapacityExpr)
        : MetalUnaryOperator(std::move(child)),
          bitmapName_(std::move(bitmapName)),
          keyExpr_(std::move(keyExpr)),
          bitmapSizeExpr_(std::move(bitmapSizeExpr)),
          keyListBuffer_(std::move(keyListBuffer)),
          keyListCountBuffer_(std::move(keyListCountBuffer)),
          keyListCapacityExpr_(std::move(keyListCapacityExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBitmapWriteParam(bitmapName_, bitmapSizeExpr_);
        cg.addBufferParam(keyListBuffer_, "uint", keyListCapacityExpr_, false);
        cg.addAtomicBufferParam(keyListCountBuffer_, "atomic_uint", "1");

        const std::string suffix = sanitizeIdentifier(keyListBuffer_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_keylist_key_" + suffix + " = (uint)(" +
                       keyExpr_ + ");");
            cg.addLine("bitmap_set(" + bitmapName_ + ", _ir_keylist_key_" +
                       suffix + ");");
            cg.addLine("uint _ir_keylist_slot_" + suffix +
                       " = atomic_fetch_add_explicit(&" + keyListCountBuffer_ +
                       "[0], 1u, memory_order_relaxed);");
            cg.addIf("_ir_keylist_slot_" + suffix + " < (uint)(" +
                     keyListCapacityExpr_ + ")", [&]() {
                cg.addLine(keyListBuffer_ + "[_ir_keylist_slot_" + suffix +
                           "] = _ir_keylist_key_" + suffix + ";");
            });
            consume();
        });
    }

    std::string describe() const override {
        return "BitmapBuildWithKeyList(" + bitmapName_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string bitmapName_;
    std::string keyExpr_;
    std::string bitmapSizeExpr_;
    std::string keyListBuffer_;
    std::string keyListCountBuffer_;
    std::string keyListCapacityExpr_;
};

class MetalRowRangeIndexBuild : public MetalUnaryOperator {
public:
    MetalRowRangeIndexBuild(std::unique_ptr<MetalOperator> child,
                            std::string table,
                            std::string keyColumn,
                            std::string firstRowBuffer,
                            std::string lastRowBuffer,
                            std::string keyExpr,
                            std::string keyDomain)
        : MetalUnaryOperator(std::move(child)),
          table_(std::move(table)),
          keyColumn_(std::move(keyColumn)),
          firstRowBuffer_(std::move(firstRowBuffer)),
          lastRowBuffer_(std::move(lastRowBuffer)),
          keyExpr_(std::move(keyExpr)),
          keyDomain_(std::move(keyDomain)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string suffix = sanitizeIdentifier(firstRowBuffer_);
        const std::string domainParam =
            "n_row_range_" + suffix + "_domain";
        cg.addResolvedScalarParam(domainParam, "uint", keyDomain_);
        cg.addAtomicBufferParam(firstRowBuffer_, "atomic_uint", keyDomain_,
                                0xFF);
        cg.addAtomicBufferParam(lastRowBuffer_, "atomic_uint", keyDomain_);

        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_row_key_" + suffix + " = (uint)(" +
                       keyExpr_ + ");");
            cg.addIf("_ir_row_key_" + suffix + " < " + domainParam, [&]() {
                cg.addLine("bool _ir_row_start_" + suffix +
                           " = (i == 0u) || ((uint)(" + keyColumn_ +
                           "[i - 1u]) != _ir_row_key_" + suffix + ");");
                cg.addLine("bool _ir_row_end_" + suffix +
                           " = (i + 1u >= " + tableSizeName(table_) +
                           ") || ((uint)(" + keyColumn_ +
                           "[i + 1u]) != _ir_row_key_" + suffix + ");");
                cg.addIf("_ir_row_start_" + suffix, [&]() {
                    cg.addLine("atomic_fetch_min_explicit(&" + firstRowBuffer_ +
                               "[_ir_row_key_" + suffix + "], (uint)i, "
                               "memory_order_relaxed);");
                });
                cg.addIf("_ir_row_end_" + suffix, [&]() {
                    cg.addLine("atomic_fetch_max_explicit(&" + lastRowBuffer_ +
                               "[_ir_row_key_" + suffix + "], (uint)i, "
                               "memory_order_relaxed);");
                });
            });
            consume();
        });
    }

    std::string describe() const override {
        return "RowRangeIndexBuild(" + firstRowBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string table_;
    std::string keyColumn_;
    std::string firstRowBuffer_;
    std::string lastRowBuffer_;
    std::string keyExpr_;
    std::string keyDomain_;
};

class MetalIrExistsDistinctKeyListBuild : public MetalOperator {
public:
    struct Target {
        std::string firstBuffer;
        std::string stateBuffer;
        std::string multiBitmap;
        std::string valueExpr;
        std::string predicate;
    };

    MetalIrExistsDistinctKeyListBuild(std::string dispatchTable,
                                      std::string table,
                                      std::string keyColumn,
                                      std::string keyDomain,
                                      std::string keyListBuffer,
                                      std::string keyListCountBuffer,
                                      std::string firstRowBuffer,
                                      std::string lastRowBuffer,
                                      std::vector<GenericColumnExpr> columns,
                                      std::vector<Target> targets)
        : dispatchTable_(std::move(dispatchTable)),
          table_(std::move(table)),
          keyColumn_(std::move(keyColumn)),
          keyDomain_(std::move(keyDomain)),
          keyListBuffer_(std::move(keyListBuffer)),
          keyListCountBuffer_(std::move(keyListCountBuffer)),
          firstRowBuffer_(std::move(firstRowBuffer)),
          lastRowBuffer_(std::move(lastRowBuffer)),
          columns_(std::move(columns)),
          targets_(std::move(targets)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string suffix = sanitizeIdentifier(keyListBuffer_);
        const std::string domainParam =
            "n_exists_keylist_" + suffix + "_domain";
        cg.setPhaseScannedTable(dispatchTable_);
        cg.addResolvedScalarParam(domainParam, "uint", keyDomain_);
        cg.addBufferParam(keyListBuffer_, "const uint", "", false);
        cg.addBufferParam(keyListCountBuffer_, "const atomic_uint", "", false);
        cg.addBufferParam(firstRowBuffer_, "const atomic_uint", "", false);
        cg.addBufferParam(lastRowBuffer_, "const atomic_uint", "", false);

        std::set<std::string> seenColumns;
        for (const auto& col : columns_) {
            if (!seenColumns.insert(col.column).second) continue;
            cg.addColumnParam(col.column, metalTypeForType(col.type), table_);
        }

        for (const auto& target : targets_) {
            cg.addAtomicBufferParam(target.firstBuffer, "atomic_uint",
                                    keyDomain_);
            cg.addAtomicBufferParam(target.stateBuffer, "atomic_uint",
                                    keyDomain_);
            cg.addBitmapWriteParam(target.multiBitmap,
                                   "(" + keyDomain_ + " + 31) / 32");
        }

        cg.addLine("uint _ir_keylist_count_" + suffix +
                   " = atomic_load_explicit(&" + keyListCountBuffer_ +
                   "[0], memory_order_relaxed);");
        cg.addBlock("for (uint i = tid; i < _ir_keylist_count_" + suffix +
                    "; i += tpg)", [&]() {
            cg.addLine("uint _ir_range_key_" + suffix + " = " +
                       keyListBuffer_ + "[i];");
            cg.addIf("_ir_range_key_" + suffix + " < " + domainParam, [&]() {
                cg.addLine("uint _ir_range_first_" + suffix +
                           " = atomic_load_explicit(&" + firstRowBuffer_ +
                           "[_ir_range_key_" + suffix + "], "
                           "memory_order_relaxed);");
                cg.addLine("uint _ir_range_last_" + suffix +
                           " = atomic_load_explicit(&" + lastRowBuffer_ +
                           "[_ir_range_key_" + suffix + "], "
                           "memory_order_relaxed);");
                cg.addIf("_ir_range_first_" + suffix +
                         " != 0xFFFFFFFFu && _ir_range_first_" + suffix +
                         " <= _ir_range_last_" + suffix, [&]() {
                    cg.addBlock("for (uint j = _ir_range_first_" + suffix +
                                "; j <= _ir_range_last_" + suffix + "; ++j)",
                                [&]() {
                        cg.addIf("(uint)(" + keyColumn_ + "[j]) == " +
                                 "_ir_range_key_" + suffix, [&]() {
                            for (const auto& target : targets_) {
                                auto emitUpdate = [&]() {
                                    const std::string targetSuffix =
                                        sanitizeIdentifier(target.firstBuffer);
                                    cg.addLine("uint _ir_exists_val_" +
                                               targetSuffix + " = (uint)(" +
                                               target.valueExpr + ");");
                                    cg.addLine("while (true) {");
                                    cg.addLine("    uint _ir_exists_state_" +
                                               targetSuffix +
                                               " = atomic_load_explicit(&" +
                                               target.stateBuffer +
                                               "[_ir_range_key_" + suffix +
                                               "], memory_order_relaxed);");
                                    cg.addLine("    if (_ir_exists_state_" +
                                               targetSuffix + " == 0u) {");
                                    cg.addLine("        uint _ir_exists_expected_" +
                                               targetSuffix + " = 0u;");
                                    cg.addLine("        if (atomic_compare_exchange_weak_explicit(&" +
                                               target.stateBuffer +
                                               "[_ir_range_key_" + suffix +
                                               "], &_ir_exists_expected_" +
                                               targetSuffix +
                                               ", 1u, memory_order_relaxed, memory_order_relaxed)) {");
                                    cg.addLine("            atomic_store_explicit(&" +
                                               target.firstBuffer +
                                               "[_ir_range_key_" + suffix +
                                               "], _ir_exists_val_" +
                                               targetSuffix +
                                               ", memory_order_relaxed);");
                                    cg.addLine("            atomic_store_explicit(&" +
                                               target.stateBuffer +
                                               "[_ir_range_key_" + suffix +
                                               "], 2u, memory_order_relaxed);");
                                    cg.addLine("            break;");
                                    cg.addLine("        }");
                                    cg.addLine("    } else if (_ir_exists_state_" +
                                               targetSuffix + " == 2u) {");
                                    cg.addLine("        uint _ir_exists_first_" +
                                               targetSuffix +
                                               " = atomic_load_explicit(&" +
                                               target.firstBuffer +
                                               "[_ir_range_key_" + suffix +
                                               "], memory_order_relaxed);");
                                    cg.addLine("        if (_ir_exists_first_" +
                                               targetSuffix +
                                               " != _ir_exists_val_" +
                                               targetSuffix + ") bitmap_set(" +
                                               target.multiBitmap +
                                               ", _ir_range_key_" + suffix +
                                               ");");
                                    cg.addLine("        break;");
                                    cg.addLine("    }");
                                    cg.addLine("}");
                                };
                                if (target.predicate.empty() ||
                                    target.predicate == "true") {
                                    emitUpdate();
                                } else {
                                    cg.addIf(target.predicate, emitUpdate);
                                }
                            }
                        });
                    });
                });
            });
        });
    }

    std::string describe() const override {
        return "IrExistsDistinctKeyListBuild(" + keyListBuffer_ + ")";
    }

private:
    std::string dispatchTable_;
    std::string table_;
    std::string keyColumn_;
    std::string keyDomain_;
    std::string keyListBuffer_;
    std::string keyListCountBuffer_;
    std::string firstRowBuffer_;
    std::string lastRowBuffer_;
    std::vector<GenericColumnExpr> columns_;
    std::vector<Target> targets_;
};

class MetalIrExistsDistinctProbe : public MetalUnaryOperator {
public:
    MetalIrExistsDistinctProbe(std::unique_ptr<MetalOperator> child,
                               std::string firstBuffer,
                               std::string stateBuffer,
                               std::string multiBitmap,
                               std::string keyExpr,
                               std::string valueExpr,
                               bool anti)
        : MetalUnaryOperator(std::move(child)),
          firstBuffer_(std::move(firstBuffer)),
          stateBuffer_(std::move(stateBuffer)),
          multiBitmap_(std::move(multiBitmap)),
          keyExpr_(std::move(keyExpr)),
          valueExpr_(std::move(valueExpr)),
          anti_(anti) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(firstBuffer_, "const atomic_uint", "", false);
        cg.addBufferParam(stateBuffer_, "const atomic_uint", "", false);
        cg.addBitmapReadParam(multiBitmap_, "");

        const std::string suffix = sanitizeIdentifier(firstBuffer_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_exists_key_" + suffix + " = (uint)(" + keyExpr_ + ");");
            cg.addLine("uint _ir_exists_val_" + suffix + " = (uint)(" + valueExpr_ + ");");
            cg.addLine("uint _ir_exists_state_" + suffix +
                       " = atomic_load_explicit(&" + stateBuffer_ +
                       "[_ir_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("bool _ir_exists_other_" + suffix + " = false;");
            cg.addLine("if (_ir_exists_state_" + suffix + " == 2u) {");
            cg.addLine("    uint _ir_exists_first_" + suffix +
                       " = atomic_load_explicit(&" + firstBuffer_ +
                       "[_ir_exists_key_" + suffix + "], memory_order_relaxed);");
            cg.addLine("    _ir_exists_other_" + suffix + " = bitmap_test_atomic(" +
                       multiBitmap_ + ", _ir_exists_key_" + suffix + ") || " +
                       "(_ir_exists_first_" + suffix + " != _ir_exists_val_" +
                       suffix + ");");
            cg.addLine("}");
            cg.addIf(std::string(anti_ ? "!" : "") + "_ir_exists_other_" + suffix,
                     [&]() { consume(); });
        });
    }

    std::string describe() const override {
        return std::string(anti_ ? "IrNotExistsDistinctProbe(" :
                                   "IrExistsDistinctProbe(") +
               firstBuffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string firstBuffer_;
    std::string stateBuffer_;
    std::string multiBitmap_;
    std::string keyExpr_;
    std::string valueExpr_;
    bool anti_;
};

class MetalRunSegmentedInSubAgg : public MetalOperator {
public:
    MetalRunSegmentedInSubAgg(std::string table,
                              std::string keyColumn,
                              std::string valueColumn,
                              std::string outputBuffer,
                              std::string keyDomain,
                              bool countOnly)
        : table_(std::move(table)),
          keyColumn_(std::move(keyColumn)),
          valueColumn_(std::move(valueColumn)),
          outputBuffer_(std::move(outputBuffer)),
          keyDomain_(std::move(keyDomain)),
          countOnly_(countOnly) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string nRows = tableSizeName(table_);
        const std::string domainParam =
            "n_segmented_in_" + sanitizeIdentifier(outputBuffer_) + "_domain";
        cg.setPhaseScannedTable(table_);
        cg.addScalarParam(nRows, "uint");
        cg.addResolvedScalarParam(domainParam, "uint", keyDomain_);
        cg.addColumnParam(keyColumn_, "int", table_);
        if (!countOnly_)
            cg.addColumnParam(valueColumn_, "float", table_);
        cg.addAtomicBufferParam(outputBuffer_, "atomic_uint", keyDomain_);

        cg.addBlock("for (uint i = tid; i < " + nRows + "; i += tpg)", [&]() {
            cg.addLine("const uint _seg_rows = 1024u;");
            cg.addLine("uint _seg_begin = (i / _seg_rows) * _seg_rows;");
            cg.addLine("uint _seg_end = min(_seg_begin + _seg_rows, " + nRows + ");");
            cg.addLine("int _seg_key = " + keyColumn_ + "[i];");
            cg.addIf("_seg_key >= 0 && (uint)_seg_key < " + domainParam, [&]() {
                cg.addLine("bool _seg_start = (i == _seg_begin) || (" +
                           keyColumn_ + "[i - 1u] != _seg_key);");
                cg.addIf("_seg_start", [&]() {
                    cg.addLine("float _seg_sum = 0.0f;");
                    cg.addLine("uint _seg_j = i;");
                    cg.addBlock("while (_seg_j < _seg_end && " +
                                keyColumn_ + "[_seg_j] == _seg_key)", [&]() {
                        if (countOnly_)
                            cg.addLine("_seg_sum += 1.0f;");
                        else
                            cg.addLine("_seg_sum += " + valueColumn_ +
                                       "[_seg_j];");
                        cg.addLine("_seg_j++;");
                    });
                    cg.addLine("atomic_add_float(&" + outputBuffer_ +
                               "[(uint)_seg_key], _seg_sum);");
                });
            });
        });
    }

    std::string describe() const override {
        return "RunSegmentedInSubAgg(" + table_ + "." + keyColumn_ + ")";
    }

private:
    std::string table_;
    std::string keyColumn_;
    std::string valueColumn_;
    std::string outputBuffer_;
    std::string keyDomain_;
    bool countOnly_ = false;
};

class MetalIrScalarAtomicLookup : public MetalUnaryOperator {
public:
    MetalIrScalarAtomicLookup(std::unique_ptr<MetalOperator> child,
                              std::string buffer,
                              std::string keyExpr,
                              std::string varName)
        : MetalUnaryOperator(std::move(child)),
          buffer_(std::move(buffer)),
          keyExpr_(std::move(keyExpr)),
          varName_(std::move(varName)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(buffer_, "const atomic_uint", "", false);
        child_->produce(cg, [&]() {
            cg.addLine("uint " + varName_ + " = atomic_load_explicit(&" +
                       buffer_ + "[" + keyExpr_ + "], memory_order_relaxed);");
            consume();
        });
    }

    std::string describe() const override {
        return "IrScalarAtomicLookup(" + buffer_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string buffer_;
    std::string keyExpr_;
    std::string varName_;
};

class MetalFixedStringRowCarryLookup : public MetalUnaryOperator {
public:
    MetalFixedStringRowCarryLookup(std::unique_ptr<MetalOperator> child,
                                   std::string rowBuffer,
                                   std::string keyExpr,
                                   std::string rowVar,
                                   std::string ptrVar,
                                   std::string sourceTable,
                                   std::string sourceColumn,
                                   int width)
        : MetalUnaryOperator(std::move(child)),
          rowBuffer_(std::move(rowBuffer)),
          keyExpr_(std::move(keyExpr)),
          rowVar_(std::move(rowVar)),
          ptrVar_(std::move(ptrVar)),
          sourceTable_(std::move(sourceTable)),
          sourceColumn_(std::move(sourceColumn)),
          width_(width) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(rowBuffer_, "uint", "", false);
        cg.addColumnParam(sourceColumn_, "char", sourceTable_);
        child_->produce(cg, [&]() {
            cg.addLine("uint " + rowVar_ + " = " + rowBuffer_ + "[" +
                       keyExpr_ + "];");
            cg.addIf(rowVar_ + " != 0xFFFFFFFFu", [&]() {
                cg.addLine("const device char* " + ptrVar_ + " = " +
                           sourceColumn_ + " + " + rowVar_ + " * " +
                           std::to_string(width_) + "u;");
                consume();
            });
        });
    }

    std::string describe() const override {
        return "FixedStringRowCarryLookup(" + ptrVar_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(keyExpr_, out);
    }

private:
    std::string rowBuffer_;
    std::string keyExpr_;
    std::string rowVar_;
    std::string ptrVar_;
    std::string sourceTable_;
    std::string sourceColumn_;
    int width_ = 0;
};

std::unique_ptr<MetalOperator> appendScalarLookupLoads(
        std::unique_ptr<MetalOperator> pipe,
        const std::vector<GenericScalarLookupInfo>* scalarLookups,
        const std::string& idxVar,
        const std::string& currentTable,
        const SchemaProvider* schema) {
    if (!scalarLookups) return pipe;
    for (const auto& info : *scalarLookups) {
        if (info.kind != GenericScalarLookupInfo::AvgByKey)
            continue;
        std::string keyExpr = genericScalarLookupKeyExpr(info, 0, idxVar,
                                                         currentTable, schema);
        if (!info.countBuffer.empty() && !info.cntVar.empty()) {
            pipe = std::make_unique<MetalIrScalarAtomicLookup>(
                std::move(pipe), info.countBuffer, keyExpr, info.cntVar);
        }
        if (!info.sumBuffer.empty() && !info.sumVar.empty()) {
            pipe = std::make_unique<MetalIrScalarAtomicLookup>(
                std::move(pipe), info.sumBuffer, keyExpr, info.sumVar);
        }
    }
    return pipe;
}

std::string rewriteScalarLookupsInCondition(
        std::string condition,
        const std::vector<GenericScalarLookupInfo>* scalarLookups,
        const std::string& idxVar,
        const std::string& currentTable,
        const SchemaProvider* schema) {
    if (!scalarLookups || scalarLookups->empty()) return condition;
    return rewriteGenericScalarSentinels(condition, idxVar, *scalarLookups,
                                         currentTable, schema);
}

std::string carryValueExpr(const IrCarryColumn& carry,
                           int currentRelationInstance,
                           const std::string& idxVar) {
    if (carry.column.relationInstance.value != currentRelationInstance)
        return carry.varName;
    if (carry.column.type.type == DataType::CHAR_FIXED) {
        int width = carry.column.type.fixedWidth > 0 ? carry.column.type.fixedWidth : 1;
        return carry.column.column + " + " + idxVar + " * " + std::to_string(width);
    }
    return carry.column.column + "[" + idxVar + "]";
}

std::unique_ptr<MetalOperator> appendCarryLookup(
        std::unique_ptr<MetalOperator> pipe,
        const GenericScanDetail& storage,
        const IrCarryColumn& carry,
        const std::string& keyExpr) {
    if (carry.column.type.type == DataType::CHAR_FIXED) {
        int width = carry.column.type.fixedWidth > 0 ? carry.column.type.fixedWidth : 1;
        return std::make_unique<MetalFixedStringRowCarryLookup>(
            std::move(pipe), carryStorageBufferName(storage, carry.column),
            keyExpr, carry.rowVarName, carry.varName,
            carry.column.table, carry.column.column, width);
    }
    return std::make_unique<MetalArrayLookup>(
        std::move(pipe), carryStorageBufferName(storage, carry.column),
        keyExpr, carry.varName, metalTypeForType(carry.column.type));
}

std::unique_ptr<MetalOperator> appendCarryStore(
        std::unique_ptr<MetalOperator> pipe,
        const GenericScanDetail& storage,
        const IrCarryColumn& carry,
        const std::string& keyExpr,
        int currentRelationInstance,
        const std::string& idxVar,
        const std::string& keyDomain) {
    std::string valueExpr = carryValueExpr(carry, currentRelationInstance, idxVar);
    if (carry.column.type.type == DataType::CHAR_FIXED) {
        std::string rowExpr = carry.column.relationInstance.value == currentRelationInstance
            ? "(uint)(" + idxVar + ")"
            : carry.rowVarName;
        return std::make_unique<MetalArrayStore>(
            std::move(pipe), carryStorageBufferName(storage, carry.column),
            keyExpr, rowExpr, "uint", keyDomain, 0xFF);
    }
    return std::make_unique<MetalArrayStore>(
        std::move(pipe), carryStorageBufferName(storage, carry.column),
        keyExpr, valueExpr, metalTypeForType(carry.column.type), keyDomain);
}

const AnalyzedQuery::InSubqueryAggInfo* inSubAggForBuild(
        const AnalyzedQuery* aq,
        const IrBuildSide& build) {
    if (!aq) return nullptr;
    for (const auto& info : aq->inSubAggs) {
        if (info.tableIndex >= 0 && info.tableIndex == build.relationInstance)
            return &info;
        if (!info.alias.empty() && build.scan && build.scan->alias == info.alias)
            return &info;
        if (info.tableIndex < 0 && build.scan &&
            info.baseTable == build.scan->table &&
            (info.alias.empty() || info.alias == build.scan->alias)) {
            return &info;
        }
    }
    return nullptr;
}

std::optional<std::string> analyzedLiteralToFloatMetal(const Literal& lit) {
    return std::visit([](const auto& value) -> std::optional<std::string> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int>) {
            return std::to_string(value) + ".0f";
        } else if constexpr (std::is_same_v<T, float>) {
            return std::to_string(value) + "f";
        } else {
            return std::nullopt;
        }
    }, lit.value);
}

bool analyzedExprIsInSubAggCall(const ExprPtr& expr,
                                const AnalyzedQuery::InSubqueryAggInfo& info) {
    if (!expr) return false;
    auto* call = std::get_if<FuncCall>(&expr->node);
    if (!call || lowerAscii(call->name) != lowerAscii(info.aggFunc))
        return false;
    if (lowerAscii(info.aggFunc) == "count")
        return true;
    if (call->args.empty() || !call->args.front())
        return info.aggExpr.empty();
    auto* col = std::get_if<ColRef>(&call->args.front()->node);
    return col && col->column == info.aggExpr;
}

std::optional<std::string> inSubAggHavingCondition(
        const AnalyzedQuery::InSubqueryAggInfo& info,
        const std::string& aggRef) {
    auto* cmp = info.havingPred ? std::get_if<Comparison>(&info.havingPred->node) : nullptr;
    if (!cmp) return aggRef + " > 0.0f";

    CmpOp op = cmp->op;
    const Literal* literal = nullptr;
    if (analyzedExprIsInSubAggCall(cmp->left, info)) {
        literal = cmp->right ? std::get_if<Literal>(&cmp->right->node) : nullptr;
    } else if (analyzedExprIsInSubAggCall(cmp->right, info)) {
        literal = cmp->left ? std::get_if<Literal>(&cmp->left->node) : nullptr;
        op = reverseCmpOp(cmp->op);
    }
    if (!literal) return std::nullopt;
    auto rhs = analyzedLiteralToFloatMetal(*literal);
    if (!rhs) return std::nullopt;
    return aggRef + " " + cmpOpToMetal(op) + " " + *rhs;
}

GenericPredicatePtr makeLogicalPredicate(
        GenericLogicalPred::Op op,
        std::vector<GenericPredicatePtr> children) {
    std::vector<GenericPredicatePtr> flat;
    for (const auto& child : children) {
        if (!child) continue;
        if (auto* logical = std::get_if<GenericLogicalPred>(&child->node)) {
            if (logical->op == op && op != GenericLogicalPred::Op::Not) {
                flat.insert(flat.end(), logical->children.begin(),
                            logical->children.end());
                continue;
            }
        }
        flat.push_back(child);
    }
    if (flat.empty()) return {};
    if (flat.size() == 1) return flat.front();

    auto out = std::make_shared<GenericPredicate>();
    out->node = GenericLogicalPred{op, std::move(flat)};
    return out;
}

std::optional<GenericPredicatePtr> relationLocalNecessaryPredicate(
        const GenericPredicatePtr& pred,
        int relationInstance) {
    if (!pred) return std::nullopt;

    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op == GenericLogicalPred::Op::And) {
            std::vector<GenericPredicatePtr> localParts;
            for (const auto& child : logical->children) {
                if (auto local =
                        relationLocalNecessaryPredicate(child, relationInstance)) {
                    localParts.push_back(*local);
                }
            }
            auto out = makeLogicalPredicate(GenericLogicalPred::Op::And,
                                            std::move(localParts));
            if (!out) return std::nullopt;
            return out;
        }

        if (logical->op == GenericLogicalPred::Op::Or) {
            std::vector<GenericPredicatePtr> localBranches;
            for (const auto& child : logical->children) {
                auto local =
                    relationLocalNecessaryPredicate(child, relationInstance);
                if (!local) return std::nullopt;
                localBranches.push_back(*local);
            }
            auto out = makeLogicalPredicate(GenericLogicalPred::Op::Or,
                                            std::move(localBranches));
            if (!out) return std::nullopt;
            return out;
        }
    }

    std::set<int> rels;
    collectPredicateRelations(pred, rels);
    if (rels.size() == 1 && rels.count(relationInstance))
        return pred;
    return std::nullopt;
}

void addRelationLocalFilter(
        const GenericPredicatePtr& pred,
        int relationInstance,
        int probeRel,
        std::map<int, IrBuildSide>& buildByRel,
        std::vector<GenericPredicatePtr>& probeFilters) {
    if (!pred) return;
    if (relationInstance == probeRel) {
        probeFilters.push_back(pred);
        return;
    }
    auto it = buildByRel.find(relationInstance);
    if (it != buildByRel.end())
        it->second.filters.push_back(pred);
}

std::vector<int> orderedJoinChildren(
        const std::vector<int>& children,
        const std::map<int, IrBuildSide>& buildByRel) {
    std::vector<int> ordered = children;
    auto rank = [&](int rel) {
        const auto& build = buildByRel.at(rel);
        if (build.useHashJoin) return 5;
        if (build.antiJoinFilter) return 4;
        if (build.semiJoinFilter || build.existsDistinct) return 3;
        if (!build.filters.empty()) return 0;
        if (!build.subtreeCarries.empty()) return 2;
        return 1;
    };
    std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
        return rank(a) < rank(b);
    });
    return ordered;
}

std::optional<std::string> siblingBitmapProbeColumn(
        const IrBuildSide& targetBuild,
        const IrBuildSide& sourceBuild) {
    if (sourceBuild.useHashJoin || sourceBuild.antiJoinFilter ||
        sourceBuild.existsDistinct || sourceBuild.bitmapName.empty()) {
        return std::nullopt;
    }
    if (sourceBuild.filters.empty() && sourceBuild.children.empty())
        return std::nullopt;
    if (targetBuild.parentCol.column == sourceBuild.parentCol.column)
        return targetBuild.joinCol.column;
    if (targetBuild.useHashJoin && !targetBuild.parentCol2.column.empty() &&
        targetBuild.parentCol2.column == sourceBuild.parentCol.column) {
        return targetBuild.joinCol2.column;
    }
    return std::nullopt;
}

void addSiblingBitmapFilter(IrBuildSide& targetBuild,
                            const IrBuildSide& sourceBuild,
                            const std::string& keyColumn) {
    for (const auto& existing : targetBuild.siblingBitmapFilters) {
        if (existing.bitmapName == sourceBuild.bitmapName &&
            existing.keyColumn == keyColumn &&
            existing.sourceRelationInstance == sourceBuild.relationInstance) {
            return;
        }
    }
    targetBuild.siblingBitmapFilters.push_back(
        {sourceBuild.bitmapName, keyColumn, sourceBuild.relationInstance});
}

void addSiblingBitmapFilters(std::map<int, IrBuildSide>& buildByRel) {
    for (auto& [parentRel, parentBuild] : buildByRel) {
        (void)parentRel;
        const auto children = orderedJoinChildren(parentBuild.children, buildByRel);
        for (size_t targetIdx = 0; targetIdx < children.size(); ++targetIdx) {
            auto targetIt = buildByRel.find(children[targetIdx]);
            if (targetIt == buildByRel.end()) continue;
            auto& targetBuild = targetIt->second;
            for (size_t sourceIdx = 0; sourceIdx < targetIdx; ++sourceIdx) {
                auto sourceIt = buildByRel.find(children[sourceIdx]);
                if (sourceIt == buildByRel.end()) continue;
                const auto& sourceBuild = sourceIt->second;
                auto keyColumn = siblingBitmapProbeColumn(targetBuild, sourceBuild);
                if (!keyColumn) continue;
                addSiblingBitmapFilter(targetBuild, sourceBuild, *keyColumn);
            }
        }
    }
}

void configureDomainKeyListDrives(std::map<int, IrBuildSide>& buildByRel) {
    if (!domainKeyListExistsEnabled()) return;
    for (auto& [targetRel, targetBuild] : buildByRel) {
        (void)targetRel;
        if (!targetBuild.existsDistinct ||
            targetBuild.siblingBitmapFilters.size() != 1 ||
            targetBuild.useHashJoin || !targetBuild.children.empty()) {
            continue;
        }
        if (!targetBuild.relation ||
            targetBuild.relation->primaryKeyColumn != targetBuild.joinCol.column) {
            continue;
        }

        const auto& sibling = targetBuild.siblingBitmapFilters.front();
        auto sourceIt = buildByRel.find(sibling.sourceRelationInstance);
        if (sourceIt == buildByRel.end() || !sourceIt->second.scan)
            continue;

        auto& sourceBuild = sourceIt->second;
        sourceBuild.emitKeyList = true;
        targetBuild.domainKeyDrive = IrDomainKeyDrive{
            sourceBuild.relationInstance,
            keyListBufferName(sourceBuild),
            keyListCountBufferName(sourceBuild),
            sourceBuild.scan->table,
        };
    }
}

bool orientJoinTreeEdge(const IrJoinEdgeCandidate& candidate,
                        int parentRel,
                        const std::map<int, IrScanSide>& sideByRel,
                        IrBuildSide& build) {
    if (candidate.columns.empty()) return false;
    auto relationIdOf = [](const GenericColumnExpr& col) {
        return col.relationInstance.value;
    };
    int leftRel = relationIdOf(candidate.columns.front().left);
    int rightRel = relationIdOf(candidate.columns.front().right);
    int childRel = leftRel == parentRel ? rightRel : leftRel;
    if (candidate.semiJoinFilter &&
        !candidate.semiInnerRelationInstances.count(childRel)) {
        return false;
    }
    auto childIt = sideByRel.find(childRel);
    if (childIt == sideByRel.end()) return false;

    const auto& childSide = childIt->second;
    auto orientCols = [&](const IrJoinColumns& columns,
                          GenericColumnExpr& joinCol,
                          GenericColumnExpr& parentCol) -> bool {
        int colLeftRel = relationIdOf(columns.left);
        int colRightRel = relationIdOf(columns.right);
        if (!((colLeftRel == childRel && colRightRel == parentRel) ||
              (colRightRel == childRel && colLeftRel == parentRel))) {
            return false;
        }
        joinCol = colLeftRel == childRel ? columns.left : columns.right;
        parentCol = colLeftRel == childRel ? columns.right : columns.left;
        return true;
    };

    GenericColumnExpr joinCol;
    GenericColumnExpr parentCol;
    if (!orientCols(candidate.columns.front(), joinCol, parentCol))
        return false;

    GenericColumnExpr joinCol2;
    GenericColumnExpr parentCol2;
    if (candidate.columns.size() > 2) return false;
    if (candidate.columns.size() == 2 &&
        !orientCols(candidate.columns[1], joinCol2, parentCol2)) {
        return false;
    }

    const bool composite = candidate.columns.size() == 2;
    std::string keyDomain;
    if (composite && candidate.semiJoinFilter)
        return false;
    if (!composite && !candidate.semiJoinFilter) {
        if (childSide.relation->primaryKeyColumn.empty() ||
            childSide.relation->primaryKeyColumn != joinCol.column) {
            return false;
        }
    }

    if (!composite) {
        keyDomain = joinCol.keyDomainSymbol;
        if (keyDomain.empty())
            keyDomain = childSide.relation->primaryKeyDomainSymbol;
        if (keyDomain.empty())
            keyDomain = childSide.relation->maxKeySymbol;
        if (keyDomain.empty())
            return false;
    }

    build.scanNode = childSide.node;
    build.scan = childSide.scan;
    build.relation = childSide.relation;
    build.relationInstance = childRel;
    build.parentRelationInstance = parentRel;
    build.joinCol = std::move(joinCol);
    build.parentCol = std::move(parentCol);
    build.joinCol2 = std::move(joinCol2);
    build.parentCol2 = std::move(parentCol2);
    build.useHashJoin = composite;
    build.semiJoinFilter = candidate.semiJoinFilter;
    build.antiJoinFilter = candidate.antiJoinFilter;
    build.keyDomain = std::move(keyDomain);
    build.bitmapName = "d_ir_join_bitmap_" +
        sanitizeIdentifier(build.scan->alias.empty() ? build.scan->table
                                                     : build.scan->alias);
    return true;
}

void classifyPredicateForJoinLowering(
        const GenericPredicatePtr& pred,
        int probeRel,
        std::map<int, IrBuildSide>& buildByRel,
        std::vector<GenericPredicatePtr>& probeFilters,
        std::vector<GenericPredicatePtr>& crossFilters) {
    std::set<int> rels;
    collectPredicateRelations(pred, rels);
    if (rels.empty() || (rels.size() == 1 && rels.count(probeRel))) {
        probeFilters.push_back(pred);
    } else if (rels.size() == 1) {
        auto it = buildByRel.find(*rels.begin());
        if (it != buildByRel.end())
            it->second.filters.push_back(pred);
        else
            crossFilters.push_back(pred);
    } else {
        for (int rel : rels) {
            if (auto local = relationLocalNecessaryPredicate(pred, rel)) {
                addRelationLocalFilter(*local, rel, probeRel, buildByRel,
                                       probeFilters);
            }
        }
        crossFilters.push_back(pred);
    }
}

std::optional<IrExistsDistinctInfo> tryMakeExistsDistinctInfo(
        const GenericPredicatePtr& pred,
        const IrBuildSide& build,
        int probeRel) {
    if (!build.semiJoinFilter || build.useHashJoin ||
        build.parentRelationInstance != probeRel || !build.scan) {
        return std::nullopt;
    }

    auto* cmp = pred ? std::get_if<GenericComparisonPred>(&pred->node) : nullptr;
    if (!cmp || cmp->op != CmpOp::NE) return std::nullopt;

    auto* left = cmp->left ? std::get_if<GenericColumnExpr>(&cmp->left->node) : nullptr;
    auto* right = cmp->right ? std::get_if<GenericColumnExpr>(&cmp->right->node) : nullptr;
    if (!left || !right) return std::nullopt;

    const GenericColumnExpr* childCol = nullptr;
    const GenericColumnExpr* parentCol = nullptr;
    if (left->relationInstance.value == build.relationInstance &&
        right->relationInstance.value == build.parentRelationInstance) {
        childCol = left;
        parentCol = right;
    } else if (right->relationInstance.value == build.relationInstance &&
               left->relationInstance.value == build.parentRelationInstance) {
        childCol = right;
        parentCol = left;
    } else {
        return std::nullopt;
    }

    if (childCol->type.type != parentCol->type.type ||
        !typeCanUseExistsDistinct(childCol->type.type)) {
        return std::nullopt;
    }

    IrExistsDistinctInfo info;
    info.childValueCol = *childCol;
    info.parentValueCol = *parentCol;
    const std::string prefix = existsDistinctBufferPrefix(*build.scan, *childCol);
    info.firstBuffer = prefix + "_first";
    info.stateBuffer = prefix + "_state";
    info.multiBitmap = prefix + "_multi";
    info.anti = build.antiJoinFilter;
    return info;
}

std::optional<MultiTableJoinLowering> buildMultiTableJoinLowering(
        const GenericRelPlan& ir,
        const std::vector<const GenericRelNode*>& scans,
        const std::vector<const GenericRelNode*>& joins,
        const GenericRelNode* filterNode,
        const std::vector<GenericExprPtr>& neededExprs,
        const std::string& planName,
        const AnalyzedQuery* aq,
        const std::vector<GenericScalarLookupInfo>* scalarLookups,
        std::string* error) {
    std::vector<IrScanSide> sides;
    for (const auto* scanNode : scans) {
        auto* scan = scanDetail(scanNode);
        auto* relation = relationForScan(ir, scanNode);
        if (!scan || !relation)
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: malformed scan metadata.");
        sides.push_back(IrScanSide{scanNode, scan, relation});
    }

    auto relationIdOf = [](const GenericColumnExpr& col) {
        return col.relationInstance.value;
    };

    std::map<int, IrScanSide> sideByRel;
    for (const auto& side : sides)
        sideByRel[side.scan->relationInstance.value] = side;

    std::vector<IrJoinEdgeCandidate> candidates;
    std::map<std::pair<int, int>, size_t> candidateByPair;
    std::set<int> semiInnerRelationInstances;
    for (const auto* joinNode : joins) {
        auto* join = std::get_if<GenericJoinDetail>(&joinNode->detail);
        if (!join)
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: malformed join detail.");
        if (!join->predicate)
            continue;
        std::set<int> joinSemiInnerRelations;
        if (join->kind == GenericJoinKind::Semi ||
            join->kind == GenericJoinKind::Anti) {
            if (joinNode->inputs.size() < 2)
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: semi/anti join has no inner input.");
            collectScanRelationInstances(ir, ir.findNode(joinNode->inputs[1]),
                                         joinSemiInnerRelations);
            semiInnerRelationInstances.insert(joinSemiInnerRelations.begin(),
                                              joinSemiInnerRelations.end());
        }
        std::vector<GenericPredicatePtr> conjuncts;
        splitConjuncts(join->predicate, conjuncts);
        for (const auto& pred : conjuncts) {
            IrJoinColumns edge;
            if (!extractEqJoinColumnsFromPredicate(pred, edge))
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: join predicate is not an equi-join.");
            if (!sideByRel.count(relationIdOf(edge.left)) ||
                !sideByRel.count(relationIdOf(edge.right))) {
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: join predicate references a relation outside the join tree.");
            }
            int leftRel = relationIdOf(edge.left);
            int rightRel = relationIdOf(edge.right);
            std::pair<int, int> key{std::min(leftRel, rightRel),
                                    std::max(leftRel, rightRel)};
            auto found = candidateByPair.find(key);
            if (found == candidateByPair.end()) {
                candidateByPair[key] = candidates.size();
                IrJoinEdgeCandidate candidate;
                candidate.index = candidates.size();
                candidate.columns.push_back(std::move(edge));
                candidate.predicates.push_back(pred);
                candidate.semiJoinFilter =
                    join->kind == GenericJoinKind::Semi ||
                    join->kind == GenericJoinKind::Anti;
                candidate.antiJoinFilter = join->kind == GenericJoinKind::Anti;
                candidate.semiInnerRelationInstances = joinSemiInnerRelations;
                candidates.push_back(std::move(candidate));
            } else {
                auto& candidate = candidates[found->second];
                if (candidate.columns.size() >= 2) {
                    return shapeFail<MultiTableJoinLowering>(
                        error, "IR multi-table join lowerer: more than two join predicates between relation instances.");
                }
                candidate.columns.push_back(std::move(edge));
                candidate.predicates.push_back(pred);
                candidate.semiJoinFilter =
                    candidate.semiJoinFilter ||
                    join->kind == GenericJoinKind::Semi ||
                    join->kind == GenericJoinKind::Anti;
                candidate.antiJoinFilter =
                    candidate.antiJoinFilter ||
                    join->kind == GenericJoinKind::Anti;
                candidate.semiInnerRelationInstances.insert(
                    joinSemiInnerRelations.begin(), joinSemiInnerRelations.end());
            }
        }
    }

    if (candidates.size() < sides.size() - 1)
        return shapeFail<MultiTableJoinLowering>(
            error, "IR multi-table join lowerer: insufficient equi-join predicates for connected join tree.");

    const IrScanSide* probe = nullptr;
    for (const auto& side : sides) {
        int rel = side.scan->relationInstance.value;
        if (semiInnerRelationInstances.count(rel)) continue;
        if (!probe || side.relation->probePriority > probe->relation->probePriority)
            probe = &side;
    }
    if (!probe)
        return shapeFail<MultiTableJoinLowering>(
            error, "IR multi-table join lowerer: no non-semi-inner probe relation.");

    std::map<int, std::vector<const IrJoinEdgeCandidate*>> adjacency;
    for (const auto& candidate : candidates) {
        adjacency[relationIdOf(candidate.columns.front().left)].push_back(&candidate);
        adjacency[relationIdOf(candidate.columns.front().right)].push_back(&candidate);
    }

    std::map<int, IrBuildSide> buildByRel;
    const int probeRel = probe->scan->relationInstance.value;
    std::set<int> visited{probeRel};
    std::vector<int> bfs{probeRel};
    std::set<size_t> treeEdgeIndexes;
    for (size_t head = 0; head < bfs.size(); ++head) {
        int parentRel = bfs[head];
        for (const auto* candidate : adjacency[parentRel]) {
            int leftRel = relationIdOf(candidate->columns.front().left);
            int rightRel = relationIdOf(candidate->columns.front().right);
            int childRel = leftRel == parentRel ? rightRel : leftRel;
            if (visited.count(childRel)) continue;

            IrBuildSide build;
            if (!orientJoinTreeEdge(*candidate, parentRel, sideByRel, build))
                continue;
            buildByRel[childRel] = std::move(build);
            buildByRel[parentRel].children.push_back(childRel);
            treeEdgeIndexes.insert(candidate->index);
            visited.insert(childRel);
            bfs.push_back(childRel);
        }
    }
    if (visited.size() != sides.size() || buildByRel.empty())
        return shapeFail<MultiTableJoinLowering>(
            error, "IR multi-table join lowerer: no schema-oriented PK/FK spanning tree.");

    std::vector<GenericPredicatePtr> probeFilters;
    std::vector<GenericPredicatePtr> crossFilters;
    for (const auto& candidate : candidates) {
        if (treeEdgeIndexes.count(candidate.index)) continue;
        for (const auto& pred : candidate.predicates) {
            if (!predicateSupported(pred))
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: residual join predicate is not supported.");
            classifyPredicateForJoinLowering(pred, probeRel, buildByRel,
                                             probeFilters, crossFilters);
        }
    }
    if (auto* filter = filterDetail(filterNode)) {
        if (!predicateSupported(filter->predicate))
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: filter predicate is not supported.");
        std::vector<GenericPredicatePtr> conjuncts;
        splitConjuncts(filter->predicate, conjuncts);
        for (const auto& pred : conjuncts) {
            classifyPredicateForJoinLowering(pred, probeRel, buildByRel,
                                             probeFilters, crossFilters);
        }
    }

    std::set<size_t> consumedCrossFilters;
    for (auto& [rel, build] : buildByRel) {
        if (rel == probeRel || !build.semiJoinFilter || build.existsDistinct)
            continue;
        for (size_t i = 0; i < crossFilters.size(); ++i) {
            if (consumedCrossFilters.count(i)) continue;
            auto info = tryMakeExistsDistinctInfo(crossFilters[i], build, probeRel);
            if (!info) continue;
            build.existsDistinct = std::move(info);
            consumedCrossFilters.insert(i);
            break;
        }
    }
    if (!consumedCrossFilters.empty()) {
        std::vector<GenericPredicatePtr> residualCrossFilters;
        residualCrossFilters.reserve(crossFilters.size() -
                                     consumedCrossFilters.size());
        for (size_t i = 0; i < crossFilters.size(); ++i) {
            if (!consumedCrossFilters.count(i))
                residualCrossFilters.push_back(crossFilters[i]);
        }
        crossFilters.swap(residualCrossFilters);
    }

    std::map<int, std::map<std::string, GenericColumnExpr>> neededByRel;
    auto addNeeded = [&](const GenericExprPtr& expr) {
        for (const auto& [rel, side] : sideByRel) {
            if (rel == probeRel) continue;
            collectColumnsForRelation(expr, side.scan->relationInstance,
                                      neededByRel[rel]);
        }
    };
    for (const auto& expr : neededExprs)
        addNeeded(expr);
    for (const auto& pred : crossFilters) {
        for (const auto& [rel, side] : sideByRel) {
            if (rel == probeRel) continue;
            collectPredicateColumnsForRelation(pred, side.scan->relationInstance,
                                               neededByRel[rel]);
        }
    }

    for (auto& [rel, build] : buildByRel) {
        if (rel == probeRel) continue;
        std::map<std::string, GenericColumnExpr> needed;
        auto neededIt = neededByRel.find(rel);
        if (neededIt != neededByRel.end()) needed = neededIt->second;
        for (const auto& [name, col] : needed) {
            if (!typeCanUseArrayCarry(col.type.type))
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: required carried column type is not supported.");
            build.localCarries[name] = IrCarryColumn{
                col, carryVarName(col), carryRowVarName(col), carryBufferName(col)};
        }
    }

    std::function<std::vector<IrCarryColumn>(int)> computeSubtreeCarries =
        [&](int rel) -> std::vector<IrCarryColumn> {
            std::map<std::string, IrCarryColumn> merged;
            auto bit = buildByRel.find(rel);
            if (bit != buildByRel.end() && rel != probeRel) {
                for (const auto& [_, carry] : bit->second.localCarries)
                    merged[carryKey(carry.column)] = carry;
            }
            if (bit != buildByRel.end()) {
                for (int childRel : bit->second.children) {
                    for (const auto& carry : computeSubtreeCarries(childRel))
                        merged[carryKey(carry.column)] = carry;
                }
            }
            std::vector<IrCarryColumn> out;
            for (const auto& [_, carry] : merged)
                out.push_back(carry);
            if (bit != buildByRel.end() && rel != probeRel)
                bit->second.subtreeCarries = out;
            return out;
    };
    computeSubtreeCarries(probeRel);
    addSiblingBitmapFilters(buildByRel);
    configureDomainKeyListDrives(buildByRel);

    for (const auto& [rel, build] : buildByRel) {
        if (rel == probeRel || !build.useHashJoin) continue;
        if (build.parentRelationInstance != probeRel) {
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: composite hash joins must connect directly to the probe relation.");
        }
        if (!build.children.empty()) {
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: composite hash build side must be a leaf.");
        }
        if (build.subtreeCarries.size() > 1) {
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: composite hash join can carry at most one value.");
        }
        if (!build.subtreeCarries.empty() &&
            build.subtreeCarries.front().column.type.type == DataType::CHAR_FIXED) {
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: composite hash join cannot carry fixed-width strings yet.");
        }
    }
    for (const auto& [rel, build] : buildByRel) {
        if (rel == probeRel || !build.semiJoinFilter) continue;
        if (!build.subtreeCarries.empty()) {
            return shapeFail<MultiTableJoinLowering>(
                error, "IR multi-table join lowerer: semi/anti joins cannot carry build-side values.");
        }
    }

    std::vector<int> postorder;
    std::function<void(int)> appendPostorder = [&](int rel) {
        auto it = buildByRel.find(rel);
        if (it == buildByRel.end()) return;
        for (int childRel : orderedJoinChildren(it->second.children, buildByRel))
            appendPostorder(childRel);
        if (rel != probeRel) postorder.push_back(rel);
    };
    appendPostorder(probeRel);

    const std::string idxVar = "i";
    MultiTableJoinLowering lowering;
    lowering.plan.name = planName;
    lowering.probeScan = probe->scan;
    lowering.outputSize = tableSizeName(probe->scan->table);

    auto predicateHasScalarLookup = [&](const GenericPredicatePtr& pred,
                                        const std::string& table) {
        if (!scalarLookups) return false;
        std::string cond = genericPredicateToMetal(pred, idxVar);
        if (referencesGenericScalarSentinel(cond, *scalarLookups)) return true;
        cond = rewriteScalarLookupsInCondition(
            std::move(cond), scalarLookups, idxVar, table,
            aq ? aq->schema : nullptr);
        return referencesGenericScalarLookupBuffer(cond, *scalarLookups);
    };

    auto applyPredicateFilters = [&](std::unique_ptr<MetalOperator> pipe,
                                     const std::vector<GenericPredicatePtr>& filters,
                                     const std::string& table,
                                     bool& scalarLookupsLoaded,
                                     bool& usesScalarLookupBuffer) {
        for (const auto& pred : filters) {
            std::string cond = genericPredicateToMetal(pred, idxVar);
            if (scalarLookups && referencesGenericScalarSentinel(cond, *scalarLookups) &&
                !scalarLookupsLoaded) {
                pipe = appendScalarLookupLoads(
                    std::move(pipe), scalarLookups, idxVar, table,
                    aq ? aq->schema : nullptr);
                scalarLookupsLoaded = true;
            }
            cond = rewriteScalarLookupsInCondition(
                std::move(cond), scalarLookups, idxVar, table,
                aq ? aq->schema : nullptr);
            usesScalarLookupBuffer =
                usesScalarLookupBuffer ||
                (scalarLookups &&
                 referencesGenericScalarLookupBuffer(cond, *scalarLookups));
            pipe = maybeSelect(std::move(pipe), cond);
        }
        return pipe;
    };

    struct BuildFilterSplit {
        std::vector<GenericPredicatePtr> immediate;
        std::vector<GenericPredicatePtr> deferred;
    };

    auto splitBuildFilters = [&](const IrBuildSide& build) {
        BuildFilterSplit split;
        for (const auto& pred : build.filters) {
            if (predicateHasScalarLookup(pred, build.scan->table)) {
                split.deferred.push_back(pred);
                continue;
            }
            split.immediate.push_back(pred);
        }
        return split;
    };

    auto buildPredicateForRow = [&](const BuildFilterSplit& split,
                                    const std::string& rowIdxVar) {
        std::string predicate = "true";
        for (const auto& pred : split.immediate) {
            std::string part = genericPredicateToMetal(pred, rowIdxVar);
            predicate = predicate == "true" ? part
                                            : "(" + predicate + ") && (" +
                                                  part + ")";
        }
        return predicate;
    };

    auto isDomainDrivenExistsCandidate = [&](const IrBuildSide& build,
                                             const BuildFilterSplit& split) {
        return build.existsDistinct &&
               build.domainKeyDrive &&
               !build.useHashJoin &&
               build.children.empty() &&
               split.deferred.empty() &&
               build.siblingBitmapFilters.size() == 1 &&
               build.relation &&
               build.relation->primaryKeyColumn == build.joinCol.column;
    };

    auto sameDomainDrivenGroup = [&](const IrBuildSide& base,
                                     const IrBuildSide& other,
                                     const BuildFilterSplit& otherSplit) {
        if (!isDomainDrivenExistsCandidate(other, otherSplit))
            return false;
        return base.domainKeyDrive &&
               other.domainKeyDrive &&
               base.domainKeyDrive->sourceRelationInstance ==
                   other.domainKeyDrive->sourceRelationInstance &&
               base.scan && other.scan &&
               base.scan->table == other.scan->table &&
               base.joinCol.column == other.joinCol.column &&
               base.keyDomain == other.keyDomain;
    };

    auto sameSiblingBitmapFilters = [](const IrBuildSide& a,
                                       const IrBuildSide& b) {
        if (a.siblingBitmapFilters.size() != b.siblingBitmapFilters.size())
            return false;
        for (size_t i = 0; i < a.siblingBitmapFilters.size(); ++i) {
            const auto& af = a.siblingBitmapFilters[i];
            const auto& bf = b.siblingBitmapFilters[i];
            if (af.bitmapName != bf.bitmapName ||
                af.keyColumn != bf.keyColumn ||
                af.sourceRelationInstance != bf.sourceRelationInstance) {
                return false;
            }
        }
        return true;
    };

    auto isFusedExistsCandidate = [&](const IrBuildSide& build,
                                      const BuildFilterSplit& split) {
        return build.existsDistinct &&
               !build.useHashJoin &&
               build.children.empty() &&
               split.deferred.empty() &&
               !build.siblingBitmapFilters.empty();
    };

    auto sameFusedExistsGroup = [&](const IrBuildSide& base,
                                    const IrBuildSide& other,
                                    const BuildFilterSplit& otherSplit) {
        if (!isFusedExistsCandidate(other, otherSplit))
            return false;
        return base.scan && other.scan &&
               base.scan->table == other.scan->table &&
               base.joinCol.column == other.joinCol.column &&
               base.keyDomain == other.keyDomain &&
               sameSiblingBitmapFilters(base, other);
    };

    std::set<std::string> emittedRowRangeIndexes;
    std::set<int> emittedDomainDrivenExists;
    std::set<int> emittedFusedExists;

    for (int rel : postorder) {
        if (emittedDomainDrivenExists.count(rel) ||
            emittedFusedExists.count(rel)) {
            continue;
        }
        const auto& build = buildByRel.at(rel);
        const std::string buildKeyExpr = build.joinCol.column + "[" + idxVar + "]";
        const std::string buildTag = sanitizeIdentifier(build.scan->alias.empty()
            ? build.scan->table : build.scan->alias);
        const BuildFilterSplit buildFilterSplit = splitBuildFilters(build);

        if (isDomainDrivenExistsCandidate(build, buildFilterSplit)) {
            std::vector<int> groupRels;
            groupRels.push_back(rel);
            for (int otherRel : postorder) {
                if (otherRel == rel ||
                    emittedDomainDrivenExists.count(otherRel)) {
                    continue;
                }
                const auto& otherBuild = buildByRel.at(otherRel);
                BuildFilterSplit otherSplit = splitBuildFilters(otherBuild);
                if (sameDomainDrivenGroup(build, otherBuild, otherSplit))
                    groupRels.push_back(otherRel);
            }

            const std::string rangeKey =
                rowRangeIndexKey(build.scan->table, build.joinCol.column);
            const std::string firstRowBuffer =
                rowRangeFirstBufferName(build.scan->table, build.joinCol.column);
            const std::string lastRowBuffer =
                rowRangeLastBufferName(build.scan->table, build.joinCol.column);
            if (!emittedRowRangeIndexes.count(rangeKey)) {
                auto rangeIndexScan = makeAutoScan(build.scan->table, idxVar);
                auto rangeIndex = std::make_unique<MetalRowRangeIndexBuild>(
                    std::move(rangeIndexScan), build.scan->table,
                    build.joinCol.column, firstRowBuffer, lastRowBuffer,
                    buildKeyExpr, build.keyDomain);
                appendPhase(lowering.plan,
                            "GENERIC_ir_multi_table_row_range_" + buildTag,
                            std::move(rangeIndex));
                emittedRowRangeIndexes.insert(rangeKey);
            }

            std::map<std::string, GenericColumnExpr> neededColumns;
            neededColumns[build.joinCol.column] = build.joinCol;
            std::vector<MetalIrExistsDistinctKeyListBuild::Target> targets;
            for (int groupRel : groupRels) {
                const auto& groupBuild = buildByRel.at(groupRel);
                const auto& info = *groupBuild.existsDistinct;
                BuildFilterSplit groupSplit = splitBuildFilters(groupBuild);
                neededColumns[groupBuild.joinCol.column] = groupBuild.joinCol;
                neededColumns[info.childValueCol.column] = info.childValueCol;
                for (const auto& pred : groupSplit.immediate) {
                    collectPredicateColumnsForRelation(
                        pred, groupBuild.scan->relationInstance,
                        neededColumns);
                }
                targets.push_back({
                    info.firstBuffer,
                    info.stateBuffer,
                    info.multiBitmap,
                    info.childValueCol.column + "[j]",
                    buildPredicateForRow(groupSplit, "j"),
                });
                emittedDomainDrivenExists.insert(groupRel);
            }

            std::vector<GenericColumnExpr> columnList;
            for (const auto& [_, col] : neededColumns)
                columnList.push_back(col);

            const auto& drive = *build.domainKeyDrive;
            auto domainBuild =
                std::make_unique<MetalIrExistsDistinctKeyListBuild>(
                    drive.dispatchTable,
                    build.scan->table,
                    build.joinCol.column,
                    build.keyDomain,
                    drive.keyListBuffer,
                    drive.keyListCountBuffer,
                    firstRowBuffer,
                    lastRowBuffer,
                    std::move(columnList),
                    std::move(targets));
            appendPhase(lowering.plan,
                        "GENERIC_ir_multi_table_exists_keylist_" + buildTag,
                        std::move(domainBuild));
            continue;
        }

        if (isFusedExistsCandidate(build, buildFilterSplit)) {
            std::vector<int> groupRels;
            groupRels.push_back(rel);
            for (int otherRel : postorder) {
                if (otherRel == rel ||
                    emittedDomainDrivenExists.count(otherRel) ||
                    emittedFusedExists.count(otherRel)) {
                    continue;
                }
                const auto& otherBuild = buildByRel.at(otherRel);
                BuildFilterSplit otherSplit = splitBuildFilters(otherBuild);
                if (sameFusedExistsGroup(build, otherBuild, otherSplit))
                    groupRels.push_back(otherRel);
            }

            if (groupRels.size() > 1) {
                std::unique_ptr<MetalOperator> fusedPipe =
                    makeAutoScan(build.scan->table, idxVar);
                for (const auto& siblingFilter : build.siblingBitmapFilters) {
                    fusedPipe = std::make_unique<MetalBitmapProbe>(
                        std::move(fusedPipe), siblingFilter.bitmapName,
                        siblingFilter.keyColumn + "[" + idxVar + "]");
                }

                std::vector<MetalIrExistsDistinctMultiBuild::Target> targets;
                for (int groupRel : groupRels) {
                    const auto& groupBuild = buildByRel.at(groupRel);
                    const auto& info = *groupBuild.existsDistinct;
                    BuildFilterSplit groupSplit = splitBuildFilters(groupBuild);
                    targets.push_back({
                        info.firstBuffer,
                        info.stateBuffer,
                        info.multiBitmap,
                        info.childValueCol.column + "[" + idxVar + "]",
                        buildPredicateForRow(groupSplit, idxVar),
                    });
                    emittedFusedExists.insert(groupRel);
                }

                fusedPipe = std::make_unique<MetalIrExistsDistinctMultiBuild>(
                    std::move(fusedPipe), buildKeyExpr, build.keyDomain,
                    std::move(targets));
                appendPhase(lowering.plan,
                            "GENERIC_ir_multi_table_exists_multi_" + buildTag,
                            std::move(fusedPipe));
                continue;
            }
        }

        std::unique_ptr<MetalOperator> buildPipe =
            makeAutoScan(build.scan->table, idxVar);
        bool buildScalarLookupsLoaded = false;
        bool buildUsesScalarLookupBuffer = false;

        for (const auto& siblingFilter : build.siblingBitmapFilters) {
            buildPipe = std::make_unique<MetalBitmapProbe>(
                std::move(buildPipe), siblingFilter.bitmapName,
                siblingFilter.keyColumn + "[" + idxVar + "]");
        }

        buildPipe = applyPredicateFilters(
            std::move(buildPipe), buildFilterSplit.immediate, build.scan->table,
            buildScalarLookupsLoaded, buildUsesScalarLookupBuffer);

        if (build.useHashJoin) {
            const std::string mapName = "hm_ir_join_" + buildTag;
            const std::string capExpr = "next_pow2((" +
                tableSizeName(build.scan->table) + ") * 4 + 16)";
            const std::string buildKeyExpr2 =
                build.joinCol2.column + "[" + idxVar + "]";
            std::string valueExpr = "0u";
            if (!build.subtreeCarries.empty()) {
                const auto& carry = build.subtreeCarries.front();
                valueExpr = encodeHashCarryValue(
                    carry.column, carry.column.column + "[" + idxVar + "]");
            }
            buildPipe = applyPredicateFilters(
                std::move(buildPipe), buildFilterSplit.deferred, build.scan->table,
                buildScalarLookupsLoaded, buildUsesScalarLookupBuffer);
            buildPipe = std::make_unique<MetalHashMapBuild>(
                std::move(buildPipe), mapName, buildKeyExpr, buildKeyExpr2,
                valueExpr, capExpr);
            auto& phase = appendPhase(
                lowering.plan, "GENERIC_ir_multi_table_build_" + buildTag,
                std::move(buildPipe));
            if (buildUsesScalarLookupBuffer && scalarLookups)
                attachGenericScalarLookupBuffers(phase, *scalarLookups);
            continue;
        }

        for (int childRel : orderedJoinChildren(build.children, buildByRel)) {
            const auto& child = buildByRel.at(childRel);
            const std::string childProbeKeyExpr = child.parentCol.column + "[" + idxVar + "]";
            if (child.antiJoinFilter) {
                buildPipe = std::make_unique<MetalAntiBitmapProbe>(
                    std::move(buildPipe), child.bitmapName, childProbeKeyExpr);
            } else {
                buildPipe = std::make_unique<MetalBitmapProbe>(
                    std::move(buildPipe), child.bitmapName, childProbeKeyExpr);
            }
            for (const auto& carry : child.subtreeCarries) {
                buildPipe = appendCarryLookup(std::move(buildPipe), *child.scan,
                                               carry, childProbeKeyExpr);
            }
        }

        buildPipe = applyPredicateFilters(
            std::move(buildPipe), buildFilterSplit.deferred, build.scan->table,
            buildScalarLookupsLoaded, buildUsesScalarLookupBuffer);

        if (build.existsDistinct) {
            const auto& info = *build.existsDistinct;
            buildPipe = std::make_unique<MetalIrExistsDistinctBuild>(
                std::move(buildPipe),
                info.firstBuffer,
                info.stateBuffer,
                info.multiBitmap,
                buildKeyExpr,
                info.childValueCol.column + "[" + idxVar + "]",
                build.keyDomain);
            auto& phase = appendPhase(
                lowering.plan, "GENERIC_ir_multi_table_build_" + buildTag,
                std::move(buildPipe));
            if (buildUsesScalarLookupBuffer && scalarLookups)
                attachGenericScalarLookupBuffers(phase, *scalarLookups);
            continue;
        }

        if (const auto* subAgg = inSubAggForBuild(aq, build)) {
            const std::string aggFunc = lowerAscii(subAgg->aggFunc);
            if (aggFunc != "sum" && aggFunc != "count") {
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: IN aggregate supports SUM/COUNT only.");
            }
            if (subAgg->groupCol.empty() || subAgg->groupCol != build.joinCol.column) {
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: IN aggregate group key must match the semi-join key.");
            }

            const std::string aggArrayName = "d_ir_in_" + buildTag + "_agg";
            const std::string bucketExpr = subAgg->groupCol + "[" + idxVar + "]";
            const std::string valueExpr = aggFunc == "count"
                ? "1.0f"
                : subAgg->aggExpr + "[" + idxVar + "]";
            const bool canSegmentRuns =
                build.scan && build.filters.empty() && build.children.empty() &&
                !buildUsesScalarLookupBuffer &&
                (aggFunc == "count" || isSimpleIdentifier(subAgg->aggExpr));
            if (canSegmentRuns) {
                appendPhase(
                    lowering.plan, "GENERIC_ir_multi_table_agg_" + buildTag,
                    std::make_unique<MetalRunSegmentedInSubAgg>(
                        build.scan->table, subAgg->groupCol,
                        aggFunc == "count" ? std::string{} : subAgg->aggExpr,
                        aggArrayName, build.keyDomain, aggFunc == "count"));
            } else {
                auto aggPipe = std::make_unique<MetalAtomicAgg>(
                    std::move(buildPipe), aggArrayName, bucketExpr, valueExpr,
                    build.keyDomain, "atomic_uint", "float");
                auto& aggPhase = appendPhase(
                    lowering.plan, "GENERIC_ir_multi_table_agg_" + buildTag,
                    std::move(aggPipe));
                if (buildUsesScalarLookupBuffer && scalarLookups)
                    attachGenericScalarLookupBuffers(aggPhase, *scalarLookups);
            }

            auto rangeScan = std::make_unique<MetalRangeScan>(build.keyDomain, idxVar);
            const std::string aggBits =
                "atomic_load_explicit(&" + aggArrayName + "[" + idxVar +
                "], memory_order_relaxed)";
            const std::string aggRef = "as_type<float>(" + aggBits + ")";
            auto havingCond = inSubAggHavingCondition(*subAgg, aggRef);
            if (!havingCond) {
                return shapeFail<MultiTableJoinLowering>(
                    error, "IR multi-table join lowerer: unsupported IN aggregate HAVING predicate.");
            }
            auto filterPipe = std::make_unique<MetalSelection>(
                std::move(rangeScan), *havingCond);
            auto bitmapPipe = std::make_unique<MetalBitmapBuild>(
                std::move(filterPipe), build.bitmapName, idxVar,
                "(" + build.keyDomain + " + 31) / 32");
            auto& bitmapPhase = appendPhase(
                lowering.plan, "GENERIC_ir_multi_table_build_" + buildTag,
                std::move(bitmapPipe));
            bitmapPhase.extraBuffers.push_back(
                {aggArrayName, "atomic_uint", true, false});
            continue;
        }

        if (build.emitKeyList) {
            buildPipe = std::make_unique<MetalBitmapBuildWithKeyList>(
                std::move(buildPipe), build.bitmapName, buildKeyExpr,
                "(" + build.keyDomain + " + 31) / 32",
                keyListBufferName(build),
                keyListCountBufferName(build),
                tableSizeName(build.scan->table));
        } else {
            buildPipe = std::make_unique<MetalBitmapBuild>(
                std::move(buildPipe), build.bitmapName, buildKeyExpr,
                "(" + build.keyDomain + " + 31) / 32");
        }
        for (const auto& carry : build.subtreeCarries) {
            buildPipe = appendCarryStore(std::move(buildPipe), *build.scan,
                                         carry, buildKeyExpr, rel, idxVar,
                                         build.keyDomain);
        }
        auto& phase = appendPhase(lowering.plan,
                                  "GENERIC_ir_multi_table_build_" + buildTag,
                                  std::move(buildPipe));
        if (buildUsesScalarLookupBuffer && scalarLookups)
            attachGenericScalarLookupBuffers(phase, *scalarLookups);
    }

    lowering.probePipe = makeAutoScan(probe->scan->table, idxVar);
    bool probeScalarLookupsLoaded = false;
    bool probeUsesScalarLookupBuffer = false;
    std::vector<GenericPredicatePtr> deferredProbeFilters;
    for (const auto& pred : probeFilters) {
        if (predicateHasScalarLookup(pred, probe->scan->table)) {
            deferredProbeFilters.push_back(pred);
            continue;
        }
        lowering.probePipe = applyPredicateFilters(
            std::move(lowering.probePipe), {pred}, probe->scan->table,
            probeScalarLookupsLoaded, probeUsesScalarLookupBuffer);
    }

    for (int childRel : orderedJoinChildren(buildByRel[probeRel].children,
                                            buildByRel)) {
        const auto& build = buildByRel.at(childRel);
        const std::string probeKeyExpr = build.parentCol.column + "[" + idxVar + "]";
        if (build.useHashJoin) {
            const std::string buildTag = sanitizeIdentifier(build.scan->alias.empty()
                ? build.scan->table : build.scan->alias);
            const std::string mapName = "hm_ir_join_" + buildTag;
            const std::string capExpr = "next_pow2((" +
                tableSizeName(build.scan->table) + ") * 4 + 16)";
            const std::string probeKeyExpr2 =
                build.parentCol2.column + "[" + idxVar + "]";
            if (build.subtreeCarries.empty()) {
                lowering.probePipe = std::make_unique<MetalHashMapLookup>(
                    std::move(lowering.probePipe), mapName, probeKeyExpr,
                    probeKeyExpr2, capExpr, "_ir_hash_join_hit_" + buildTag,
                    "uint");
            } else {
                const auto& carry = build.subtreeCarries.front();
                lowering.probePipe = std::make_unique<MetalHashMapLookup>(
                    std::move(lowering.probePipe), mapName, probeKeyExpr,
                    probeKeyExpr2, capExpr, carry.varName,
                    hashLookupResultType(carry.column));
                lowering.carryMap[carry.column.relationInstance.value]
                                 [carry.column.column] = carry;
            }
            continue;
        }
        if (build.existsDistinct) {
            const auto& info = *build.existsDistinct;
            lowering.probePipe = std::make_unique<MetalIrExistsDistinctProbe>(
                std::move(lowering.probePipe),
                info.firstBuffer,
                info.stateBuffer,
                info.multiBitmap,
                probeKeyExpr,
                info.parentValueCol.column + "[" + idxVar + "]",
                info.anti);
            continue;
        }
        if (build.antiJoinFilter) {
            lowering.probePipe = std::make_unique<MetalAntiBitmapProbe>(
                std::move(lowering.probePipe), build.bitmapName, probeKeyExpr);
        } else {
            lowering.probePipe = std::make_unique<MetalBitmapProbe>(
                std::move(lowering.probePipe), build.bitmapName, probeKeyExpr);
        }
        for (const auto& carry : build.subtreeCarries) {
            lowering.probePipe = appendCarryLookup(std::move(lowering.probePipe),
                                                   *build.scan, carry,
                                                   probeKeyExpr);
            lowering.carryMap[carry.column.relationInstance.value][carry.column.column] = carry;
        }
    }

    lowering.probePipe = applyPredicateFilters(
        std::move(lowering.probePipe), deferredProbeFilters,
        probe->scan->table, probeScalarLookupsLoaded,
        probeUsesScalarLookupBuffer);

    for (const auto& pred : crossFilters) {
        std::string cond = genericPredicateToMetalWithCarryMap(pred, idxVar,
                                                               lowering.carryMap);
        if (scalarLookups && referencesGenericScalarSentinel(cond, *scalarLookups) &&
            !probeScalarLookupsLoaded) {
            lowering.probePipe = appendScalarLookupLoads(
                std::move(lowering.probePipe), scalarLookups, idxVar,
                probe->scan->table, aq ? aq->schema : nullptr);
            probeScalarLookupsLoaded = true;
        }
        cond = rewriteScalarLookupsInCondition(
            std::move(cond), scalarLookups, idxVar, probe->scan->table,
            aq ? aq->schema : nullptr);
        lowering.probePipe = maybeSelect(
            std::move(lowering.probePipe), cond);
    }

    return lowering;
}

} // namespace codegen
