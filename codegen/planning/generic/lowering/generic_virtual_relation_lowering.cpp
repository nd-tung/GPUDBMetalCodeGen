#include "generic/lowering/generic_ir_physical_planner.h"

#include "core/schema_provider.h"
#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_expression_metal.h"
#include "metal_plan_common.h"
#include "query_analyzer.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::optional<std::string> jsonStringValueForIr(const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") && node["String"].contains("sval"))
        return node["String"]["sval"].get<std::string>();
    return std::nullopt;
}

std::string jsonFuncNameForIr(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() || fc["funcname"].empty())
        return {};
    auto s = jsonStringValueForIr(fc["funcname"].back());
    return s ? lowerAscii(*s) : "";
}

std::string analyzedColumnNameForExpr(const ExprPtr& expr) {
    if (auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr)
        return col->column;
    return "";
}

std::string analyzedDisplayNameForExpr(const ExprPtr& expr) {
    if (!expr) return "expr";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            return node.column;
        } else if constexpr (std::is_same_v<T, Literal>) {
            if (auto* i = std::get_if<int>(&node.value))
                return std::to_string(*i);
            if (auto* f = std::get_if<float>(&node.value)) {
                std::ostringstream oss;
                oss << *f;
                return oss.str();
            }
            if (auto* s = std::get_if<std::string>(&node.value))
                return "'" + *s + "'";
            return "literal";
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            std::string out = node.name + "(";
            for (size_t i = 0; i < node.args.size(); ++i) {
                if (i) out += ",";
                out += analyzedDisplayNameForExpr(node.args[i]);
            }
            out += ")";
            return out;
        }
        return "expr";
    }, expr->node);
}

std::string analyzedDisplayNameForTarget(const SelectTarget& target,
                                         size_t targetIndex) {
    if (!target.alias.empty()) return target.alias;
    if (target.isAgg && target.agg) {
        std::string base = target.agg->isStar ? "*"
            : analyzedDisplayNameForExpr(target.agg->innerExpr);
        return aggFuncName(target.agg->func) + "(" + base + ")";
    }
    if (target.expr && std::holds_alternative<ColRef>(target.expr->node))
        return analyzedColumnNameForExpr(target.expr);
    return "expr_" + std::to_string(targetIndex);
}

std::optional<std::string> analyzedOrderColumnForExpr(
        const ExprPtr& expr,
        const std::vector<SelectTarget>& targets) {
    if (!expr) return std::nullopt;
    if (auto* lit = std::get_if<Literal>(&expr->node)) {
        if (auto* ordinal = std::get_if<int>(&lit->value)) {
            if (*ordinal >= 1 &&
                static_cast<size_t>(*ordinal) <= targets.size()) {
                return analyzedDisplayNameForTarget(
                    targets[*ordinal - 1],
                    static_cast<size_t>(*ordinal - 1));
            }
        }
        return std::nullopt;
    }
    auto* orderCol = std::get_if<ColRef>(&expr->node);
    if (!orderCol) return std::nullopt;

    std::optional<std::string> unqualifiedMatch;
    const bool orderQualified =
        !orderCol->table.empty() || !orderCol->tableAlias.empty();
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        std::string displayName = analyzedDisplayNameForTarget(target, i);
        if (displayName == orderCol->column) return displayName;
        auto* targetCol = target.expr
            ? std::get_if<ColRef>(&target.expr->node)
            : nullptr;
        if (!targetCol || targetCol->column != orderCol->column) continue;
        bool tableMatches = true;
        if (!orderCol->table.empty() && targetCol->table != orderCol->table)
            tableMatches = false;
        if (!orderCol->tableAlias.empty() &&
            targetCol->tableAlias != orderCol->tableAlias) {
            tableMatches = false;
        }
        if (orderQualified) {
            if (tableMatches) return displayName;
        } else if (!unqualifiedMatch) {
            unqualifiedMatch = displayName;
        }
    }
    return unqualifiedMatch;
}

std::optional<std::string> analyzedResolveOrderColumn(
        const ExprPtr& expr,
        int orderIdx,
        const std::vector<SelectTarget>& targets) {
    auto col = analyzedOrderColumnForExpr(expr, targets);
    if (col) return col;
    if (expr) {
        const std::string orderExprName = analyzedDisplayNameForExpr(expr);
        for (size_t i = 0; i < targets.size(); ++i) {
            if (targets[i].expr &&
                analyzedDisplayNameForExpr(targets[i].expr) == orderExprName) {
                return analyzedDisplayNameForTarget(targets[i], i);
            }
        }
    }
    if (orderIdx >= 0 && orderIdx < static_cast<int>(targets.size()))
        return analyzedDisplayNameForTarget(targets[orderIdx], orderIdx);
    return std::nullopt;
}

void collectAnalyzedPredTables(const PredPtr& pred,
                               std::set<std::string>& tables) {
    std::map<std::string, std::string> colToTable;
    collectColumnTables(pred, colToTable);
    for (const auto& [_, table] : colToTable) {
        if (!table.empty()) tables.insert(table);
    }
}

std::string keyDomainSymbolForAnalyzedColumn(const std::string& table,
                                             const std::string& column,
                                             const SchemaProvider* schema) {
    if (!schema) return "";
    auto keySym = schema->keyDomainSymbol(table, column);
    if (!keySym.empty()) return keySym;
    if (auto gd = schema->groupDomain(table, column))
        return std::to_string(gd->maxValue + 1);
    auto pk = schema->pkInfo(table);
    if (pk && pk->first == column) return pk->second;
    return schema->maxKeySymbol(table);
}

std::string fromSubqueryBaseForKey(const FromSubqueryAggInfo& info,
                                   const std::string& tableKey) {
    for (size_t i = 0; i < info.tables.size(); ++i) {
        if (info.tables[i] == tableKey) return info.tables[i];
        if (i < info.tableAliases.size() && info.tableAliases[i] == tableKey)
            return info.tables[i];
    }
    return tableKey;
}

bool fromSubqueryColMatches(const FromSubqueryAggInfo& info,
                            const ColRef& col,
                            const std::string& tableKey,
                            const std::string& column) {
    if (col.column != column) return false;
    if (col.table == tableKey) return true;
    if (!col.tableAlias.empty() && col.tableAlias == tableKey) return true;
    return fromSubqueryBaseForKey(info, tableKey) == col.table;
}

TypeInfo analyzedTypeInfoForExpr(const ExprPtr& expr,
                                 const SchemaProvider* schema) {
    if (!expr) return {DataType::INT, 0};
    return std::visit([&](const auto& node) -> TypeInfo {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            int width = node.fixedWidth;
            if (node.dataType == DataType::CHAR_FIXED && width <= 0 && schema) {
                try { width = schema->columnFixedWidth(node.table, node.column); }
                catch (...) { width = 0; }
            }
            return {node.dataType, width};
        } else if constexpr (std::is_same_v<T, Literal>) {
            if (std::holds_alternative<float>(node.value))
                return {DataType::FLOAT, 0};
            if (auto* s = std::get_if<std::string>(&node.value))
                return {DataType::CHAR_FIXED, static_cast<int>(s->size())};
            return {DataType::INT, 0};
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            TypeInfo left = analyzedTypeInfoForExpr(node.left, schema);
            TypeInfo right = analyzedTypeInfoForExpr(node.right, schema);
            if (node.op == ExprOp::DIV ||
                left.type == DataType::FLOAT ||
                right.type == DataType::FLOAT) {
                return {DataType::FLOAT, 0};
            }
            return {DataType::INT, 0};
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            if (!node.branches.empty())
                return analyzedTypeInfoForExpr(node.branches.front().result, schema);
            return analyzedTypeInfoForExpr(node.elseResult, schema);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            return {DataType::INT, 0};
        }
        return {DataType::INT, 0};
    }, expr->node);
}

bool analyzedTypeIsNumericLike(DataType type) {
    return type == DataType::INT || type == DataType::FLOAT ||
           type == DataType::DATE;
}

bool analyzedMaterializeExprSupported(const ExprPtr& expr);
bool analyzedPredicateSupportedForMaterialize(const PredPtr& pred);

bool analyzedMaterializeExprSupported(const ExprPtr& expr) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            return node.dataType == DataType::INT ||
                   node.dataType == DataType::FLOAT ||
                   node.dataType == DataType::DATE ||
                   node.dataType == DataType::CHAR1 ||
                   node.dataType == DataType::CHAR_FIXED;
        } else if constexpr (std::is_same_v<T, Literal>) {
            return !std::holds_alternative<std::string>(node.value);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            return analyzedMaterializeExprSupported(node.left) &&
                   analyzedMaterializeExprSupported(node.right) &&
                   analyzedTypeIsNumericLike(
                       analyzedTypeInfoForExpr(node.left, nullptr).type) &&
                   analyzedTypeIsNumericLike(
                       analyzedTypeInfoForExpr(node.right, nullptr).type);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (const auto& branch : node.branches) {
                TypeInfo branchType =
                    analyzedTypeInfoForExpr(branch.result, nullptr);
                if (!analyzedTypeIsNumericLike(branchType.type) ||
                    !analyzedPredicateSupportedForMaterialize(branch.condition) ||
                    !analyzedMaterializeExprSupported(branch.result)) {
                    return false;
                }
            }
            if (!node.elseResult) return true;
            TypeInfo elseType = analyzedTypeInfoForExpr(node.elseResult, nullptr);
            return analyzedTypeIsNumericLike(elseType.type) &&
                   analyzedMaterializeExprSupported(node.elseResult);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            const std::string name = lowerAscii(node.name);
            return name == "date_part" || name == "extract" ||
                   name == "substring" || name == "sum" ||
                   name == "count" || name == "avg" ||
                   name == "min" || name == "max";
        }
        return false;
    }, expr->node);
}

bool analyzedPredicateSupportedForMaterialize(const PredPtr& pred) {
    if (!pred) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            return analyzedMaterializeExprSupported(node.left) &&
                   analyzedMaterializeExprSupported(node.right);
        } else if constexpr (std::is_same_v<T, Between>) {
            return analyzedMaterializeExprSupported(node.expr) &&
                   analyzedMaterializeExprSupported(node.low) &&
                   analyzedMaterializeExprSupported(node.high);
        } else if constexpr (std::is_same_v<T, InList>) {
            if (!analyzedMaterializeExprSupported(node.expr)) return false;
            for (const auto& value : node.values) {
                if (!analyzedMaterializeExprSupported(value)) return false;
            }
            return true;
        } else if constexpr (std::is_same_v<T, Like>) {
            return analyzedMaterializeExprSupported(node.expr);
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children) {
                if (!analyzedPredicateSupportedForMaterialize(child)) return false;
            }
            return true;
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            return analyzedPredicateSupportedForMaterialize(node.child);
        }
        return false;
    }, pred->node);
}

int analyzedFixedStringLenForExpr(const ExprPtr& expr,
                                  const SchemaProvider* schema) {
    if (!expr) return 0;
    auto* col = std::get_if<ColRef>(&expr->node);
    if (!col) return 0;
    if (col->dataType == DataType::CHAR1) return 1;
    if (col->dataType != DataType::CHAR_FIXED) return 0;
    if (col->fixedWidth > 0) return col->fixedWidth;
    if (!schema) return 0;
    try { return schema->columnFixedWidth(col->table, col->column); }
    catch (...) { return 0; }
}

std::string analyzedMaterializeValueExpr(const ExprPtr& expr,
                                         const std::string& idxVar,
                                         const SchemaProvider* schema) {
    if (auto* col = expr ? std::get_if<ColRef>(&expr->node) : nullptr) {
        if (col->dataType == DataType::CHAR1)
            return col->column + " + " + idxVar;
        if (col->dataType == DataType::CHAR_FIXED) {
            int len = analyzedFixedStringLenForExpr(expr, schema);
            if (len <= 0) len = 1;
            std::string aliasPrefix;
            if (!col->tableAlias.empty())
                aliasPrefix = "/*" + col->tableAlias + "*/";
            return aliasPrefix + col->column + " + " + idxVar +
                   " * " + std::to_string(len);
        }
    }
    return exprToMetal(expr, idxVar, schema);
}

void analyzedCollectColumnsForTable(const ExprPtr& expr,
                                    const std::string& table,
                                    std::set<std::string>& cols);

void analyzedCollectPredicateColumnsForTable(const PredPtr& pred,
                                             const std::string& table,
                                             std::set<std::string>& cols) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            analyzedCollectColumnsForTable(node.left, table, cols);
            analyzedCollectColumnsForTable(node.right, table, cols);
        } else if constexpr (std::is_same_v<T, Between>) {
            analyzedCollectColumnsForTable(node.expr, table, cols);
            analyzedCollectColumnsForTable(node.low, table, cols);
            analyzedCollectColumnsForTable(node.high, table, cols);
        } else if constexpr (std::is_same_v<T, InList>) {
            analyzedCollectColumnsForTable(node.expr, table, cols);
            for (const auto& value : node.values)
                analyzedCollectColumnsForTable(value, table, cols);
        } else if constexpr (std::is_same_v<T, Like>) {
            analyzedCollectColumnsForTable(node.expr, table, cols);
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children)
                analyzedCollectPredicateColumnsForTable(child, table, cols);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            analyzedCollectPredicateColumnsForTable(node.child, table, cols);
        }
    }, pred->node);
}

void analyzedCollectColumnsForTable(const ExprPtr& expr,
                                    const std::string& table,
                                    std::set<std::string>& cols) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            if (node.table == table || node.tableAlias == table)
                cols.insert(node.column);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            analyzedCollectColumnsForTable(node.left, table, cols);
            analyzedCollectColumnsForTable(node.right, table, cols);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (const auto& arg : node.args)
                analyzedCollectColumnsForTable(arg, table, cols);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (const auto& branch : node.branches) {
                analyzedCollectPredicateColumnsForTable(
                    branch.condition, table, cols);
                analyzedCollectColumnsForTable(branch.result, table, cols);
            }
            analyzedCollectColumnsForTable(node.elseResult, table, cols);
        }
    }, expr->node);
}

struct JsonColumnRefForIr {
    std::string qualifier;
    std::string column;
};

std::optional<JsonColumnRefForIr> jsonRawColumnRefForIr(
        const nlohmann::json& node) {
    const nlohmann::json* cr = nullptr;
    if (node.is_object() && node.contains("ColumnRef")) cr = &node["ColumnRef"];
    else if (node.is_object() && node.contains("fields")) cr = &node;
    if (!cr || !cr->contains("fields") || !(*cr)["fields"].is_array())
        return std::nullopt;

    std::vector<std::string> fields;
    for (const auto& field : (*cr)["fields"]) {
        if (auto s = jsonStringValueForIr(field)) fields.push_back(*s);
    }
    if (fields.empty()) return std::nullopt;
    JsonColumnRefForIr out;
    out.column = fields.back();
    if (fields.size() >= 2) out.qualifier = fields[fields.size() - 2];
    return out;
}

struct FromSubqueryScalarExtremumForIr {
    int sqIdx = -1;
    AggFunc func = AggFunc::MAX;
    std::string argAlias;
};

bool analyzedExprIsIntLiteral(const ExprPtr& expr, int value) {
    auto* lit = expr ? std::get_if<Literal>(&expr->node) : nullptr;
    if (!lit) return false;
    auto* iv = std::get_if<int>(&lit->value);
    return iv && *iv == value;
}

bool analyzedPredicateReferencesIntLiteral(const PredPtr& pred, int value) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            return analyzedExprIsIntLiteral(node.left, value) ||
                   analyzedExprIsIntLiteral(node.right, value);
        } else if constexpr (std::is_same_v<T, Between>) {
            return analyzedExprIsIntLiteral(node.expr, value) ||
                   analyzedExprIsIntLiteral(node.low, value) ||
                   analyzedExprIsIntLiteral(node.high, value);
        } else if constexpr (std::is_same_v<T, InList>) {
            if (analyzedExprIsIntLiteral(node.expr, value)) return true;
            for (const auto& candidate : node.values) {
                if (analyzedExprIsIntLiteral(candidate, value)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, LogicalAnd> ||
                             std::is_same_v<T, LogicalOr>) {
            for (const auto& child : node.children) {
                if (analyzedPredicateReferencesIntLiteral(child, value))
                    return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            return analyzedPredicateReferencesIntLiteral(node.child, value);
        }
        return false;
    }, pred->node);
}

bool analyzedFiltersReferenceScalarSentinel(const AnalyzedQuery& aq,
                                            int sqIdx) {
    const int sentinel =
        std::numeric_limits<int>::min() + static_cast<int>(sqIdx);
    for (const auto& filter : aq.filters) {
        if (analyzedPredicateReferencesIntLiteral(filter, sentinel))
            return true;
    }
    return false;
}

std::optional<FromSubqueryScalarExtremumForIr>
parseFromSubqueryScalarExtremumForIr(const AnalyzedQuery& aq,
                                     const FromSubqueryAggInfo& fsq,
                                     const std::string& aggregateAlias) {
    if (fsq.alias.empty() || aggregateAlias.empty()) return std::nullopt;
    for (size_t sqIdx = 0; sqIdx < aq.subqueries.size(); ++sqIdx) {
        const auto& sq = aq.subqueries[sqIdx];
        if (sq.type != AnalyzedQuery::Subquery::SCALAR_SUBQUERY) continue;
        if (!analyzedFiltersReferenceScalarSentinel(
                aq, static_cast<int>(sqIdx))) {
            continue;
        }

        nlohmann::json root;
        try { root = nlohmann::json::parse(sq.sql); }
        catch (...) { continue; }
        if (!root.contains("SelectStmt")) continue;
        const auto& ss = root["SelectStmt"];
        if (!ss.contains("fromClause") || !ss["fromClause"].is_array())
            continue;

        bool scansView = false;
        for (const auto& from : ss["fromClause"]) {
            if (!from.contains("RangeVar")) continue;
            const auto& rv = from["RangeVar"];
            if (rv.value("relname", "") == fsq.alias) {
                scansView = true;
                break;
            }
        }
        if (!scansView) continue;
        if (!ss.contains("targetList") || !ss["targetList"].is_array() ||
            ss["targetList"].empty()) {
            continue;
        }

        for (const auto& target : ss["targetList"]) {
            if (!target.contains("ResTarget")) continue;
            const auto& rt = target["ResTarget"];
            if (!rt.contains("val") || !rt["val"].contains("FuncCall"))
                continue;
            const auto& fc = rt["val"]["FuncCall"];
            const std::string func = jsonFuncNameForIr(fc);
            if (func != "max" && func != "min") continue;
            if (!fc.contains("args") || !fc["args"].is_array() ||
                fc["args"].empty()) {
                continue;
            }
            auto arg = jsonRawColumnRefForIr(fc["args"][0]);
            if (!arg || arg->column != aggregateAlias) continue;
            FromSubqueryScalarExtremumForIr out;
            out.sqIdx = static_cast<int>(sqIdx);
            out.func = (func == "max") ? AggFunc::MAX : AggFunc::MIN;
            out.argAlias = aggregateAlias;
            return out;
        }
    }
    return std::nullopt;
}

class MetalIrAtomicFloatSumWithSeen : public MetalUnaryOperator {
public:
    MetalIrAtomicFloatSumWithSeen(std::unique_ptr<MetalOperator> child,
                                    std::string inputArray,
                                    std::string seenArray,
                                    std::string bucketExpr,
                                    std::string valueExpr,
                                    std::string sizeExpr)
        : MetalUnaryOperator(std::move(child)),
          inputArray_(std::move(inputArray)),
          seenArray_(std::move(seenArray)),
          bucketExpr_(std::move(bucketExpr)),
          valueExpr_(std::move(valueExpr)),
          sizeExpr_(std::move(sizeExpr)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addAtomicBufferParam(inputArray_, "atomic_float", sizeExpr_);
        cg.addAtomicBufferParam(seenArray_, "atomic_uint", sizeExpr_);
        child_->produce(cg, [&]() {
            cg.addLine("uint _ir_from_subquery_bucket = (uint)(" +
                       bucketExpr_ + ");");
            cg.addLine("atomic_fetch_add_explicit(&" + inputArray_ +
                       "[_ir_from_subquery_bucket], (float)(" + valueExpr_ +
                       "), memory_order_relaxed);");
            cg.addLine("atomic_store_explicit(&" + seenArray_ +
                       "[_ir_from_subquery_bucket], 1u, memory_order_relaxed);");
            consume();
        });
    }

    std::string describe() const override {
        return "IrAtomicFloatSumWithSeen(" + inputArray_ + "[" +
               bucketExpr_ + "])";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(bucketExpr_, out);
        appendIUsFromExpr(valueExpr_, out);
    }

private:
    std::string inputArray_;
    std::string seenArray_;
    std::string bucketExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
};

class MetalIrAtomicExtremumFloatArray : public MetalUnaryOperator {
public:
    MetalIrAtomicExtremumFloatArray(std::unique_ptr<MetalOperator> child,
                                    std::string inputArray,
                                    std::string seenArray,
                                    std::string outputScalar,
                                    std::string stateScalar,
                                    std::string indexExpr,
                                    bool useMax)
        : MetalUnaryOperator(std::move(child)),
          inputArray_(std::move(inputArray)),
          seenArray_(std::move(seenArray)),
          outputScalar_(std::move(outputScalar)),
          stateScalar_(std::move(stateScalar)),
          indexExpr_(std::move(indexExpr)),
          useMax_(useMax) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addBufferParam(inputArray_, "const atomic_float", "", false, 0);
        cg.addBufferParam(seenArray_, "const atomic_uint", "", false, 0);
        cg.addAtomicBufferParam(outputScalar_, "atomic_uint", "1");
        cg.addAtomicBufferParam(stateScalar_, "atomic_uint", "1");
        child_->produce(cg, [&]() {
            cg.addIf("atomic_load_explicit(&" + seenArray_ + "[" +
                     indexExpr_ + "], memory_order_relaxed) != 0u", [&]() {
                cg.addLine("float _ir_from_subquery_value = atomic_load_explicit(&" +
                           inputArray_ + "[" + indexExpr_ +
                           "], memory_order_relaxed);");
                if (useMax_) {
                    cg.addLine("atomic_max_float(&" + outputScalar_ +
                               "[0], _ir_from_subquery_value);");
                    cg.addLine("atomic_store_explicit(&" + stateScalar_ +
                               "[0], 2u, memory_order_relaxed);");
                } else {
                    cg.addLine("atomic_min_float_seen(&" + outputScalar_ +
                               "[0], &" + stateScalar_ +
                               "[0], _ir_from_subquery_value);");
                }
            });
            consume();
        });
    }

    std::string describe() const override {
        return std::string(useMax_ ? "IrAtomicMax" : "IrAtomicMin") +
               "FloatArray(" + inputArray_ + ")";
    }

private:
    std::string inputArray_;
    std::string seenArray_;
    std::string outputScalar_;
    std::string stateScalar_;
    std::string indexExpr_;
    bool useMax_ = true;
};

std::optional<MetalQueryPlan> lowerFromSubqueryTopScalarIRToMetal(
        const AnalyzedQuery& aq,
        std::string* error) {
    if (aq.fromSubqueryAggs.size() != 1) return std::nullopt;
    if (aq.hasAggregation() || aq.hasGroupBy()) return std::nullopt;

    const auto& fsq = aq.fromSubqueryAggs[0];
    if (fsq.tables.size() != 1 || fsq.groupBy.size() != 1)
        return std::nullopt;
    auto* groupCol = fsq.groupBy[0]
        ? std::get_if<ColRef>(&fsq.groupBy[0]->node)
        : nullptr;
    if (!groupCol) return std::nullopt;

    const SelectTarget* innerAgg = nullptr;
    size_t innerAggIndex = 0;
    FromSubqueryScalarExtremumForIr scalarExtremum;
    for (size_t ti = 0; ti < fsq.targets.size(); ++ti) {
        const auto& target = fsq.targets[ti];
        if (!target.isAgg || !target.agg) continue;
        const std::string alias = analyzedDisplayNameForTarget(target, ti);
        auto parsed = parseFromSubqueryScalarExtremumForIr(aq, fsq, alias);
        if (!parsed) continue;
        innerAgg = &target;
        innerAggIndex = ti;
        scalarExtremum = *parsed;
        break;
    }
    if (!innerAgg || !innerAgg->agg) return std::nullopt;
    if (scalarExtremum.func != AggFunc::MAX) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer currently supports scalar MAX.");
    }
    if (innerAgg->agg->func != AggFunc::SUM) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer currently supports SUM aggregate values.");
    }
    TypeInfo aggValueType =
        analyzedTypeInfoForExpr(innerAgg->agg->innerExpr, aq.schema);
    if (!innerAgg->agg->innerExpr ||
        !analyzedTypeIsNumericLike(aggValueType.type)) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer requires a numeric aggregate expression.");
    }

    std::string viewBase = fromSubqueryBaseForKey(
        fsq, groupCol->tableAlias.empty()
                 ? groupCol->table
                 : groupCol->tableAlias);
    if (viewBase.empty()) viewBase = groupCol->table;
    if (viewBase.empty()) return std::nullopt;

    bool foundOuterJoin = false;
    std::string outerTable;
    std::string outerKeyCol;
    for (const auto& jc : aq.joins) {
        const bool leftIsViewKey =
            fromSubqueryColMatches(fsq, *groupCol, jc.leftTable, jc.leftCol);
        const bool rightIsViewKey =
            fromSubqueryColMatches(fsq, *groupCol, jc.rightTable, jc.rightCol);
        if (leftIsViewKey == rightIsViewKey) continue;
        const std::string candidateTable = leftIsViewKey
            ? fromSubqueryBaseForKey(fsq, jc.rightTable)
            : fromSubqueryBaseForKey(fsq, jc.leftTable);
        const std::string candidateKey =
            leftIsViewKey ? jc.rightCol : jc.leftCol;
        if (std::find(fsq.tables.begin(), fsq.tables.end(),
                      candidateTable) != fsq.tables.end()) {
            continue;
        }
        foundOuterJoin = true;
        outerTable = candidateTable;
        outerKeyCol = candidateKey;
        break;
    }
    if (!foundOuterJoin || outerTable.empty() || outerKeyCol.empty())
        return std::nullopt;

    const std::string sizeSymbol = keyDomainSymbolForAnalyzedColumn(
        viewBase, groupCol->column, aq.schema);
    if (sizeSymbol.empty()) {
        return fail(error,
            "IR grouped FROM-view scalar lowerer: group key has no schema domain.");
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_FROM_SUBQUERY_TOP_SCALAR";
    const std::string idxVar = "i";
    const std::string tag = sanitizeIdentifier(
        fsq.alias.empty() ? "from_subquery" : fsq.alias);
    const std::string aggAlias =
        analyzedDisplayNameForTarget(*innerAgg, innerAggIndex);
    const std::string aggBuffer = "d_ir_from_subquery_" + tag + "_" +
        sanitizeIdentifier(aggAlias);
    const std::string aggSeenBuffer = aggBuffer + "_seen_keys";
    const std::string extremumBuffer = aggBuffer + "_" +
        (scalarExtremum.func == AggFunc::MAX ? "max" : "min");
    const std::string extremumState = extremumBuffer + "_seen";

    {
        std::set<std::string> scanCols{groupCol->column};
        collectColumns(innerAgg->agg->innerExpr, scanCols);
        for (const auto& filter : fsq.filters)
            collectColumns(filter, scanCols);
        auto scan = makeScanForCols(viewBase, idxVar, scanCols, aq.schema);
        auto filtered = maybeSelect(
            std::move(scan), combineFilters(fsq.filters, idxVar, aq.schema));
        const std::string valueExpr =
            exprToMetal(innerAgg->agg->innerExpr, idxVar, aq.schema);
        auto agg = std::make_unique<MetalIrAtomicFloatSumWithSeen>(
            std::move(filtered), aggBuffer, aggSeenBuffer,
            groupCol->column + "[" + idxVar + "]", valueExpr, sizeSymbol);
        appendPhase(plan, "GENERIC_ir_from_subquery_aggregate_" + tag,
                    std::move(agg));
    }

    {
        auto range = std::make_unique<MetalRangeScan>(sizeSymbol, idxVar);
        auto extremum = std::make_unique<MetalIrAtomicExtremumFloatArray>(
            std::move(range), aggBuffer, aggSeenBuffer, extremumBuffer,
            extremumState, idxVar, scalarExtremum.func == AggFunc::MAX);
        appendPhase(plan, "GENERIC_ir_from_subquery_extremum_" + tag,
                    std::move(extremum));
    }

    {
        std::set<std::string> scanCols{outerKeyCol};
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            if (analyzedDisplayNameForTarget(aq.targets[ti], ti) == aggAlias)
                continue;
            analyzedCollectColumnsForTable(
                aq.targets[ti].expr, outerTable, scanCols);
        }
        auto scan = makeScanForCols(outerTable, idxVar, scanCols, aq.schema);
        const std::string outerKeyExpr = outerKeyCol + "[" + idxVar + "]";
        const std::string aggValueExpr = "atomic_load_explicit(&" +
            aggBuffer + "[" + outerKeyExpr + "], memory_order_relaxed)";
        const std::string extremumExpr =
            "as_type<float>(atomic_load_explicit(&" + extremumBuffer +
            "[0], memory_order_relaxed))";
        const std::string stateReady = "(atomic_load_explicit(&" +
            extremumState + "[0], memory_order_relaxed) == 2u)";
        auto filtered = maybeSelect(
            std::move(scan),
            stateReady + " && (" + aggValueExpr + " == " +
                extremumExpr + ")");

        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(filtered), "d_ir_from_subquery_" + tag + "_result_count",
            "1");
        const std::string outputSize = tableSizeName(outerTable);
        std::vector<GenericMatColumnDesc> materializedCols;

        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            const auto& target = aq.targets[ti];
            const std::string displayName =
                analyzedDisplayNameForTarget(target, ti);
            const std::string bufferName = "d_ir_from_subquery_" + tag +
                "_" + std::to_string(ti) + "_" +
                sanitizeIdentifier(displayName);
            if (displayName == aggAlias) {
                materialize->addColumn(bufferName, "float", aggValueExpr,
                                       displayName, outputSize, 0);
                materializedCols.push_back(
                    {displayName, bufferName, "float", 0, 0, false});
                continue;
            }
            if (!target.expr ||
                !analyzedMaterializeExprSupported(target.expr)) {
                return fail(error,
                    "IR grouped FROM-view scalar lowerer target expression is not supported.");
            }
            TypeInfo type = analyzedTypeInfoForExpr(target.expr, aq.schema);
            const int stringLen =
                analyzedFixedStringLenForExpr(target.expr, aq.schema);
            std::string sizeExpr = outputSize;
            if (stringLen > 0)
                sizeExpr += " * " + std::to_string(stringLen);
            const std::string outType = metalTypeForType(type);
            const std::string valueExpr = analyzedMaterializeValueExpr(
                target.expr, idxVar, aq.schema);
            materialize->addColumn(bufferName, outType, valueExpr,
                                   displayName, sizeExpr, stringLen);
            materializedCols.push_back(
                {displayName, bufferName, outType, stringLen, 0, false});
        }

        auto& phase = appendPhase(
            plan, "GENERIC_ir_from_subquery_materialize_" + tag,
            std::move(materialize));
        phase.extraBuffers.push_back(
            {aggBuffer, "atomic_float", true, false});
        phase.extraBuffers.push_back(
            {extremumBuffer, "atomic_uint", true, false});
        phase.extraBuffers.push_back(
            {extremumState, "atomic_uint", true, false});

        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < static_cast<int>(aq.orderBy.size()); ++oi) {
            auto column = analyzedResolveOrderColumn(
                aq.orderBy[oi].expr, oi, aq.targets);
            if (!column) {
                return fail(error,
                    "IR grouped FROM-view scalar lowerer: ORDER BY key is not projected.");
            }
            sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            const std::string sortRowsSym =
                "n_gpu_sort_ir_from_subquery_" + tag + "_rows";
            attachMaterializedCountHook(
                phase, "d_ir_from_subquery_" + tag + "_result_count",
                sortRowsSym);
            if (!appendGenericGpuSort(plan, "ir_from_subquery_" + tag,
                                      sortRowsSym, outputSize,
                                      materializedCols, sortSpec, error)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}

std::optional<MetalQueryPlan> lowerFromSubqueryHistogramIRToMetal(
        const AnalyzedQuery& aq,
        std::string* error) {
    if (aq.fromSubqueryAggs.size() != 1) return std::nullopt;
    if (aq.groupBy.size() != 1) return std::nullopt;

    const auto& fsq = aq.fromSubqueryAggs[0];
    auto* outerGroupCol = aq.groupBy[0]
        ? std::get_if<ColRef>(&aq.groupBy[0]->node)
        : nullptr;
    std::string innerAggAlias = outerGroupCol ? outerGroupCol->column : "";
    if (innerAggAlias.empty()) {
        for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
            if (!aq.targets[ti].isAgg) {
                if (!innerAggAlias.empty()) return std::nullopt;
                innerAggAlias = analyzedDisplayNameForTarget(aq.targets[ti], ti);
            }
        }
    }
    if (innerAggAlias.empty()) return std::nullopt;

    const SelectTarget* outerCount = nullptr;
    size_t outerCountIndex = 0;
    bool hasOuterGroupProjection = false;
    for (size_t ti = 0; ti < aq.targets.size(); ++ti) {
        const auto& target = aq.targets[ti];
        const std::string display = analyzedDisplayNameForTarget(target, ti);
        if (!target.isAgg && display == innerAggAlias) {
            hasOuterGroupProjection = true;
        } else if (target.isAgg && target.agg &&
                   target.agg->func == AggFunc::COUNT) {
            outerCount = &target;
            outerCountIndex = ti;
        }
    }
    if (!hasOuterGroupProjection || !outerCount) return std::nullopt;

    const SelectTarget* innerAgg = nullptr;
    for (size_t ti = 0; ti < fsq.targets.size(); ++ti) {
        const auto& target = fsq.targets[ti];
        if (target.isAgg && target.agg &&
            analyzedDisplayNameForTarget(target, ti) == innerAggAlias) {
            innerAgg = &target;
            break;
        }
    }
    if (!innerAgg || !innerAgg->agg ||
        innerAgg->agg->func != AggFunc::COUNT) {
        return std::nullopt;
    }
    if (fsq.groupBy.size() != 1) return std::nullopt;
    auto* innerGroupCol = fsq.groupBy[0]
        ? std::get_if<ColRef>(&fsq.groupBy[0]->node)
        : nullptr;
    if (!innerGroupCol) return std::nullopt;

    const JoinClause* leftOuterJoin = nullptr;
    for (const auto& jc : fsq.joins) {
        if (!jc.leftOuter) continue;
        if (fromSubqueryColMatches(fsq, *innerGroupCol,
                                   jc.leftTable, jc.leftCol)) {
            leftOuterJoin = &jc;
            break;
        }
    }
    if (!leftOuterJoin) return std::nullopt;

    const std::string groupBase =
        fromSubqueryBaseForKey(fsq, leftOuterJoin->leftTable);
    const std::string aggBase =
        fromSubqueryBaseForKey(fsq, leftOuterJoin->rightTable);
    const std::string groupJoinCol = leftOuterJoin->leftCol;
    const std::string aggJoinCol = leftOuterJoin->rightCol;
    if (groupBase.empty() || aggBase.empty() ||
        groupJoinCol.empty() || aggJoinCol.empty()) {
        return std::nullopt;
    }

    std::vector<PredPtr> aggFilters;
    for (const auto& filter : fsq.filters) {
        std::set<std::string> filterTables;
        collectAnalyzedPredTables(filter, filterTables);
        bool appliesToAgg = filterTables.empty();
        if (!filterTables.empty()) {
            appliesToAgg = std::all_of(
                filterTables.begin(), filterTables.end(),
                [&](const std::string& table) { return table == aggBase; });
        }
        if (!appliesToAgg) return std::nullopt;
        aggFilters.push_back(filter);
    }

    std::string countSize =
        keyDomainSymbolForAnalyzedColumn(groupBase, groupJoinCol, aq.schema);
    if (countSize.empty()) {
        return fail(error, "IR FROM-subquery histogram lowerer: group key has no schema domain.");
    }

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_FROM_SUBQUERY_HISTOGRAM";
    const std::string idxVar = "i";
    const std::string tag = sanitizeIdentifier(fsq.alias.empty()
        ? "from_subquery"
        : fsq.alias);
    const std::string countBuffer = "d_ir_from_subquery_" + tag + "_count";

    {
        std::set<std::string> scanCols{aggJoinCol};
        for (const auto& filter : aggFilters) collectColumns(filter, scanCols);
        auto scan = makeScanForCols(aggBase, idxVar, scanCols, aq.schema);
        auto filtered = maybeSelect(
            std::move(scan), combineFilters(aggFilters, idxVar, aq.schema));
        auto count = std::make_unique<MetalAtomicCount>(
            std::move(filtered), countBuffer, aggJoinCol + "[" + idxVar + "]",
            countSize);
        appendPhase(plan, "GENERIC_ir_from_subquery_count_" + tag,
                    std::move(count));
    }

    {
        std::set<std::string> scanCols{groupJoinCol};
        auto scan = makeScanForCols(groupBase, idxVar, scanCols, aq.schema);
        const std::string groupKeyExpr = groupJoinCol + "[" + idxVar + "]";
        const std::string countExpr =
            "(int)atomic_load_explicit(&" + countBuffer + "[" +
            groupKeyExpr + "], memory_order_relaxed)";
        const std::string outerCountName =
            analyzedDisplayNameForTarget(*outerCount, outerCountIndex);
        constexpr int kHistogramBucketCap = 65536;
        const std::string groupTag = "ir_from_subquery_hist_" + tag;
        const std::string histBuffer = "d_ir_from_subquery_" + tag + "_hist";
        auto hist = std::make_unique<MetalKeyedAgg>(
            std::move(scan), histBuffer,
            "min(" + countExpr + ", " +
                std::to_string(kHistogramBucketCap - 1) + ")",
            kHistogramBucketCap, 1);
        hist->setKeyResult(innerAggAlias, 0);
        hist->addAggregateWithMeta(outerCountName, 0, "1u", "add",
                                   false, 0, false, false, "COUNT", "");
        auto& histPhase = appendPhase(
            plan, "GENERIC_ir_from_subquery_histogram_" + tag,
            std::move(hist));
        histPhase.extraBuffers.push_back({countBuffer, "atomic_uint", true, false});

        const std::string compactCounter =
            "d_ir_from_subquery_" + tag + "_hist_result_count";
        std::vector<KeyedCompactKeySpec> compactKeys = {
            {innerAggAlias, kHistogramBucketCap, 1, {}, 0, {}, 0}
        };
        std::vector<KeyedCompactAggSpec> compactAggs;
        KeyedCompactAggSpec countOut;
        countOut.displayName = outerCountName;
        countOut.offset = 0;
        compactAggs.push_back(countOut);

        std::vector<GenericMatColumnDesc> compactCols;
        const std::string countCol = "d_ir_from_subquery_" + tag + "_0_" +
            sanitizeIdentifier(innerAggAlias);
        const std::string outerCountCol = "d_ir_from_subquery_" + tag + "_1_" +
            sanitizeIdentifier(outerCountName);
        compactCols.push_back({innerAggAlias, countCol, "int", 0, 0, false});
        compactCols.push_back({outerCountName, outerCountCol, "uint", 0, 0, false});

        auto& compactPhase = appendPhase(
            plan, "GENERIC_ir_from_subquery_histogram_compact_" + tag,
            makeKeyedAggCompactOperator(
                histBuffer, compactCounter, kHistogramBucketCap, 1,
                compactKeys, compactAggs, compactCols));
        const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
        attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);

        GenericSortSpec sortSpec;
        sortSpec.limit = aq.limit;
        for (int oi = 0; oi < static_cast<int>(aq.orderBy.size()); ++oi) {
            auto column = analyzedResolveOrderColumn(
                aq.orderBy[oi].expr, oi, aq.targets);
            if (!column) {
                return fail(error, "IR FROM-subquery histogram lowerer: ORDER BY key is not projected.");
            }
            sortSpec.keys.push_back({*column, aq.orderBy[oi].descending});
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            if (!appendGenericGpuSort(plan, "group_" + groupTag,
                                      sortRowsSym,
                                      std::to_string(kHistogramBucketCap),
                                      compactCols,
                                      sortSpec, error)) {
                return std::nullopt;
            }
        }
    }

    return plan;
}



} // namespace

std::optional<MetalQueryPlan> lowerFromSubqueryAggregateIRToMetal(
        const AnalyzedQuery& aq,
        std::string* error) {
    if (auto p = lowerFromSubqueryHistogramIRToMetal(aq, error))
        return p;
    return lowerFromSubqueryTopScalarIRToMetal(aq, error);
}

} // namespace codegen
