#include "generic/lowering/generic_multi_table_join_lowering.h"

#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_relation_analysis.h"
#include "metal_plan_common.h"
#include "core/schema_provider.h"

#include <algorithm>
#include <cctype>
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
    std::string keyDomain;
    std::string bitmapName;
    std::optional<IrExistsDistinctInfo> existsDistinct;
};

struct IrScanSide {
    const GenericRelNode* node = nullptr;
    const GenericScanDetail* scan = nullptr;
    const GenericRelation* relation = nullptr;
};

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
        return std::make_unique<MetalArraySliceLookup>(
            std::move(pipe), carryStorageBufferName(storage, carry.column),
            keyExpr, carry.varName, width);
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
        int width = carry.column.type.fixedWidth > 0 ? carry.column.type.fixedWidth : 1;
        return std::make_unique<MetalArraySliceStore>(
            std::move(pipe), carryStorageBufferName(storage, carry.column),
            keyExpr, valueExpr, width, "char",
            "(" + keyDomain + ") * " + std::to_string(width), 0,
            carry.column.relationInstance.value == currentRelationInstance
                ? carry.column.column : std::string{},
            carry.column.relationInstance.value == currentRelationInstance
                ? idxVar : std::string{});
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
            build.localCarries[name] = IrCarryColumn{col, carryVarName(col),
                                                     carryBufferName(col)};
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
        for (int childRel : it->second.children)
            appendPostorder(childRel);
        if (rel != probeRel) postorder.push_back(rel);
    };
    appendPostorder(probeRel);

    const std::string idxVar = "i";
    MultiTableJoinLowering lowering;
    lowering.plan.name = planName;
    lowering.probeScan = probe->scan;
    lowering.outputSize = tableSizeName(probe->scan->table);

    for (int rel : postorder) {
        const auto& build = buildByRel.at(rel);
        const std::string buildKeyExpr = build.joinCol.column + "[" + idxVar + "]";
        const std::string buildTag = sanitizeIdentifier(build.scan->alias.empty()
            ? build.scan->table : build.scan->alias);
        std::unique_ptr<MetalOperator> buildPipe =
            makeAutoScan(build.scan->table, idxVar);
        bool buildScalarLookupsLoaded = false;
        bool buildUsesScalarLookupBuffer = false;
        for (const auto& pred : build.filters) {
            std::string cond = genericPredicateToMetal(pred, idxVar);
            if (scalarLookups && referencesGenericScalarSentinel(cond, *scalarLookups) &&
                !buildScalarLookupsLoaded) {
                buildPipe = appendScalarLookupLoads(
                    std::move(buildPipe), scalarLookups, idxVar,
                    build.scan->table, aq ? aq->schema : nullptr);
                buildScalarLookupsLoaded = true;
            }
            cond = rewriteScalarLookupsInCondition(
                std::move(cond), scalarLookups, idxVar, build.scan->table,
                aq ? aq->schema : nullptr);
            buildUsesScalarLookupBuffer =
                buildUsesScalarLookupBuffer ||
                (scalarLookups &&
                 referencesGenericScalarLookupBuffer(cond, *scalarLookups));
            buildPipe = maybeSelect(std::move(buildPipe), cond);
        }

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

        for (int childRel : build.children) {
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
            auto aggPipe = std::make_unique<MetalAtomicAgg>(
                std::move(buildPipe), aggArrayName, bucketExpr, valueExpr,
                build.keyDomain, "atomic_uint", "float");
            auto& aggPhase = appendPhase(
                lowering.plan, "GENERIC_ir_multi_table_agg_" + buildTag,
                std::move(aggPipe));
            if (buildUsesScalarLookupBuffer && scalarLookups)
                attachGenericScalarLookupBuffers(aggPhase, *scalarLookups);

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

        buildPipe = std::make_unique<MetalBitmapBuild>(
            std::move(buildPipe), build.bitmapName, buildKeyExpr,
            "(" + build.keyDomain + " + 31) / 32");
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
    for (const auto& pred : probeFilters) {
        std::string cond = genericPredicateToMetal(pred, idxVar);
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
        lowering.probePipe = maybeSelect(std::move(lowering.probePipe), cond);
    }

    for (int childRel : buildByRel[probeRel].children) {
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
