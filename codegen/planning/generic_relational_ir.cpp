#include "generic_relational_ir.h"

#include <sstream>

namespace codegen {

namespace {

nlohmann::json typeInfoToJSON(const TypeInfo& type) {
    nlohmann::json j;
    j["type"] = dataTypeName(type.type);
    if (type.type == DataType::CHAR_FIXED)
        j["fixedWidth"] = type.fixedWidth;
    return j;
}

nlohmann::json idJSON(int value) {
    if (value < 0) return nullptr;
    return value;
}

} // namespace

const GenericColumn* GenericOutputSchema::findByDisplayName(const std::string& name) const {
    for (const auto& col : columns) {
        if (col.displayName == name) return &col;
    }
    return nullptr;
}

const GenericRelation* GenericRelPlan::findRelation(GenericRelationId id) const {
    if (!id.valid()) return nullptr;
    for (const auto& rel : relations) {
        if (rel.id.value == id.value) return &rel;
    }
    return nullptr;
}

const GenericRelationInstance* GenericRelPlan::findRelationInstance(GenericRelationInstanceId id) const {
    if (!id.valid()) return nullptr;
    for (const auto& inst : relationInstances) {
        if (inst.id.value == id.value) return &inst;
    }
    return nullptr;
}

const GenericRelNode* GenericRelPlan::findNode(GenericNodeId id) const {
    if (!id.valid()) return nullptr;
    for (const auto& node : nodes) {
        if (node.id.value == id.value) return &node;
    }
    return nullptr;
}

nlohmann::json GenericRelPlan::toJSON() const {
    nlohmann::json j;
    j["root"] = idJSON(root.value);

    j["relations"] = nlohmann::json::array();
    for (const auto& rel : relations) {
        nlohmann::json relJson = {
            {"id", rel.id.value},
            {"name", rel.name},
            {"virtual", rel.virtualRelation}
        };
        if (!rel.maxKeySymbol.empty()) relJson["maxKeySymbol"] = rel.maxKeySymbol;
        if (!rel.primaryKeyColumn.empty()) relJson["primaryKeyColumn"] = rel.primaryKeyColumn;
        if (!rel.primaryKeyDomainSymbol.empty())
            relJson["primaryKeyDomainSymbol"] = rel.primaryKeyDomainSymbol;
        if (rel.probePriority != 0) relJson["probePriority"] = rel.probePriority;
        j["relations"].push_back(std::move(relJson));
    }

    j["relationInstances"] = nlohmann::json::array();
    for (const auto& inst : relationInstances) {
        j["relationInstances"].push_back({
            {"id", inst.id.value},
            {"relation", idJSON(inst.relation.value)},
            {"baseName", inst.baseName},
            {"alias", inst.alias}
        });
    }

    j["nodes"] = nlohmann::json::array();
    for (const auto& node : nodes)
        j["nodes"].push_back(genericNodeToJSON(node));
    return j;
}

GenericRelationId GenericRelPlanBuilder::addRelation(
        const std::string& name,
        bool virtualRelation,
        std::string maxKeySymbol,
        std::string primaryKeyColumn,
        std::string primaryKeyDomainSymbol,
        int probePriority) {
    GenericRelationId id{static_cast<int>(plan_.relations.size())};
    GenericRelation relation;
    relation.id = id;
    relation.name = name;
    relation.virtualRelation = virtualRelation;
    relation.maxKeySymbol = std::move(maxKeySymbol);
    relation.primaryKeyColumn = std::move(primaryKeyColumn);
    relation.primaryKeyDomainSymbol = std::move(primaryKeyDomainSymbol);
    relation.probePriority = probePriority;
    plan_.relations.push_back(std::move(relation));
    return id;
}

GenericRelationInstanceId GenericRelPlanBuilder::addRelationInstance(
        GenericRelationId relation,
        const std::string& baseName,
        const std::string& alias) {
    GenericRelationInstanceId id{static_cast<int>(plan_.relationInstances.size())};
    plan_.relationInstances.push_back(GenericRelationInstance{id, relation, baseName, alias});
    return id;
}

GenericNodeId GenericRelPlanBuilder::addNode(
        GenericRelOp op,
        std::vector<GenericNodeId> inputs,
        GenericOutputSchema output,
        std::variant<GenericScanDetail,
                     GenericFilterDetail,
                     GenericProjectDetail,
                     GenericJoinDetail,
                     GenericAggregateDetail,
                     GenericSortDetail,
                     GenericLimitDetail,
                     GenericMaterializeDetail> detail) {
    GenericNodeId id{static_cast<int>(plan_.nodes.size())};
    plan_.nodes.push_back(GenericRelNode{id, op, std::move(inputs),
                                         std::move(output), std::move(detail)});
    return id;
}

GenericExprId GenericRelPlanBuilder::nextExprId() {
    return GenericExprId{nextExprId_++};
}

GenericColumnId GenericRelPlanBuilder::nextColumnId() {
    return GenericColumnId{nextColumnId_++};
}

GenericRelPlan GenericRelPlanBuilder::finish(GenericNodeId root) {
    plan_.root = root;
    return std::move(plan_);
}

std::string dataTypeName(DataType type) {
    switch (type) {
        case DataType::INT: return "INT";
        case DataType::FLOAT: return "FLOAT";
        case DataType::DATE: return "DATE";
        case DataType::CHAR1: return "CHAR1";
        case DataType::CHAR_FIXED: return "CHAR_FIXED";
    }
    return "UNKNOWN";
}

std::string exprOpName(ExprOp op) {
    switch (op) {
        case ExprOp::ADD: return "ADD";
        case ExprOp::SUB: return "SUB";
        case ExprOp::MUL: return "MUL";
        case ExprOp::DIV: return "DIV";
    }
    return "UNKNOWN";
}

std::string aggFuncName(AggFunc func) {
    switch (func) {
        case AggFunc::SUM: return "SUM";
        case AggFunc::COUNT: return "COUNT";
        case AggFunc::AVG: return "AVG";
        case AggFunc::MIN: return "MIN";
        case AggFunc::MAX: return "MAX";
        case AggFunc::COUNT_DISTINCT: return "COUNT_DISTINCT";
    }
    return "UNKNOWN";
}

std::string cmpOpName(CmpOp op) {
    switch (op) {
        case CmpOp::EQ: return "EQ";
        case CmpOp::NE: return "NE";
        case CmpOp::LT: return "LT";
        case CmpOp::LE: return "LE";
        case CmpOp::GT: return "GT";
        case CmpOp::GE: return "GE";
    }
    return "UNKNOWN";
}

std::string genericRelOpName(GenericRelOp op) {
    switch (op) {
        case GenericRelOp::Scan: return "Scan";
        case GenericRelOp::Filter: return "Filter";
        case GenericRelOp::Project: return "Project";
        case GenericRelOp::Join: return "Join";
        case GenericRelOp::SemiJoin: return "SemiJoin";
        case GenericRelOp::AntiJoin: return "AntiJoin";
        case GenericRelOp::Aggregate: return "Aggregate";
        case GenericRelOp::Sort: return "Sort";
        case GenericRelOp::Limit: return "Limit";
        case GenericRelOp::Materialize: return "Materialize";
    }
    return "Unknown";
}

std::string genericJoinKindName(GenericJoinKind kind) {
    switch (kind) {
        case GenericJoinKind::Inner: return "Inner";
        case GenericJoinKind::LeftOuter: return "LeftOuter";
        case GenericJoinKind::Semi: return "Semi";
        case GenericJoinKind::Anti: return "Anti";
    }
    return "Unknown";
}

nlohmann::json genericExprToJSON(const GenericExprPtr& expr) {
    if (!expr) return nullptr;
    nlohmann::json j;
    j["id"] = idJSON(expr->id.value);
    j["type"] = typeInfoToJSON(expr->type);

    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            j["kind"] = "Column";
            j["relationInstance"] = idJSON(node.relationInstance.value);
            j["table"] = node.table;
            j["alias"] = node.alias;
            j["column"] = node.column;
            j["hasGroupDomain"] = node.hasGroupDomain;
            if (node.hasGroupDomain && node.domainMin <= node.domainMax) {
                j["domainMin"] = node.domainMin;
                j["domainMax"] = node.domainMax;
            }
            if (!node.charDomain.empty()) {
                j["charDomain"] = nlohmann::json::array();
                for (char ch : node.charDomain)
                    j["charDomain"].push_back(std::string(1, ch));
            }
            if (node.numericScale > 0)
                j["numericScale"] = node.numericScale;
            if (!node.keyDomainSymbol.empty())
                j["keyDomainSymbol"] = node.keyDomainSymbol;
            if (!node.distinctDomainSymbol.empty())
                j["distinctDomainSymbol"] = node.distinctDomainSymbol;
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            j["kind"] = "Literal";
            std::visit([&](const auto& value) { j["value"] = value; }, node.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            j["kind"] = "Binary";
            j["op"] = exprOpName(node.op);
            j["left"] = genericExprToJSON(node.left);
            j["right"] = genericExprToJSON(node.right);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            j["kind"] = "Case";
            j["branches"] = nlohmann::json::array();
            for (const auto& branch : node.branches) {
                j["branches"].push_back({
                    {"condition", genericPredicateToJSON(branch.condition)},
                    {"result", genericExprToJSON(branch.result)}
                });
            }
            j["else"] = genericExprToJSON(node.elseResult);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            j["kind"] = "Function";
            j["name"] = node.name;
            j["args"] = nlohmann::json::array();
            for (const auto& arg : node.args)
                j["args"].push_back(genericExprToJSON(arg));
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            j["kind"] = "Aggregate";
            j["func"] = aggFuncName(node.func);
            j["star"] = node.star;
            j["distinct"] = node.distinct;
            j["alias"] = node.alias;
            j["arg"] = genericExprToJSON(node.arg);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            j["kind"] = "ScalarLookup";
            j["source"] = idJSON(node.source.value);
            j["outputName"] = node.outputName;
            j["keys"] = nlohmann::json::array();
            for (const auto& key : node.keys)
                j["keys"].push_back(genericExprToJSON(key));
        }
    }, expr->node);

    return j;
}

nlohmann::json genericPredicateToJSON(const GenericPredicatePtr& pred) {
    if (!pred) return nullptr;
    nlohmann::json j;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            j["kind"] = "Comparison";
            j["op"] = cmpOpName(node.op);
            j["left"] = genericExprToJSON(node.left);
            j["right"] = genericExprToJSON(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            j["kind"] = "Between";
            j["expr"] = genericExprToJSON(node.expr);
            j["low"] = genericExprToJSON(node.low);
            j["high"] = genericExprToJSON(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            j["kind"] = "InList";
            j["expr"] = genericExprToJSON(node.expr);
            j["values"] = nlohmann::json::array();
            for (const auto& value : node.values)
                j["values"].push_back(genericExprToJSON(value));
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            j["kind"] = "Like";
            j["expr"] = genericExprToJSON(node.expr);
            j["pattern"] = node.pattern;
            j["negated"] = node.negated;
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            j["kind"] = node.op == GenericLogicalPred::Op::And ? "And"
                      : node.op == GenericLogicalPred::Op::Or ? "Or"
                      : "Not";
            j["children"] = nlohmann::json::array();
            for (const auto& child : node.children)
                j["children"].push_back(genericPredicateToJSON(child));
        } else if constexpr (std::is_same_v<T, GenericExistsPred>) {
            j["kind"] = "Exists";
            j["negated"] = node.negated;
            j["subqueryIndex"] = node.subqueryIndex;
        }
    }, pred->node);
    return j;
}

nlohmann::json genericSchemaToJSON(const GenericOutputSchema& schema) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& col : schema.columns) {
        j.push_back({
            {"id", idJSON(col.id.value)},
            {"relationInstance", idJSON(col.relationInstance.value)},
            {"name", col.name},
            {"displayName", col.displayName},
            {"type", typeInfoToJSON(col.type)}
        });
    }
    return j;
}

nlohmann::json genericNodeToJSON(const GenericRelNode& node) {
    nlohmann::json j;
    j["id"] = idJSON(node.id.value);
    j["op"] = genericRelOpName(node.op);
    j["inputs"] = nlohmann::json::array();
    for (const auto& input : node.inputs)
        j["inputs"].push_back(idJSON(input.value));
    j["output"] = genericSchemaToJSON(node.output);

    std::visit([&](const auto& detail) {
        using T = std::decay_t<decltype(detail)>;
        if constexpr (std::is_same_v<T, GenericScanDetail>) {
            j["detail"] = {
                {"relationInstance", idJSON(detail.relationInstance.value)},
                {"table", detail.table},
                {"alias", detail.alias}
            };
        } else if constexpr (std::is_same_v<T, GenericFilterDetail>) {
            j["detail"] = {{"predicate", genericPredicateToJSON(detail.predicate)}};
        } else if constexpr (std::is_same_v<T, GenericProjectDetail>) {
            nlohmann::json projections = nlohmann::json::array();
            for (const auto& proj : detail.projections) {
                projections.push_back({
                    {"name", proj.name},
                    {"type", typeInfoToJSON(proj.type)},
                    {"expr", genericExprToJSON(proj.expr)}
                });
            }
            j["detail"] = {{"projections", projections}};
        } else if constexpr (std::is_same_v<T, GenericJoinDetail>) {
            j["detail"] = {
                {"kind", genericJoinKindName(detail.kind)},
                {"predicate", genericPredicateToJSON(detail.predicate)}
            };
        } else if constexpr (std::is_same_v<T, GenericAggregateDetail>) {
            nlohmann::json groupBy = nlohmann::json::array();
            for (const auto& expr : detail.groupBy)
                groupBy.push_back(genericExprToJSON(expr));
            nlohmann::json aggregates = nlohmann::json::array();
            for (const auto& agg : detail.aggregates) {
                aggregates.push_back({
                    {"name", agg.name},
                    {"type", typeInfoToJSON(agg.type)},
                    {"expr", genericExprToJSON(agg.expr)}
                });
            }
            j["detail"] = {
                {"groupBy", groupBy},
                {"groupNames", detail.groupNames},
                {"aggregates", aggregates},
                {"aggregateOutputFuncs", detail.aggregateOutputFuncs},
                {"outputOrder", detail.outputOrder},
                {"having", genericPredicateToJSON(detail.having)}
            };
        } else if constexpr (std::is_same_v<T, GenericSortDetail>) {
            nlohmann::json keys = nlohmann::json::array();
            for (const auto& key : detail.keys) {
                keys.push_back({
                    {"expr", genericExprToJSON(key.expr)},
                    {"descending", key.descending}
                });
            }
            j["detail"] = {{"keys", keys}};
        } else if constexpr (std::is_same_v<T, GenericLimitDetail>) {
            j["detail"] = {{"limit", detail.limit}};
        } else if constexpr (std::is_same_v<T, GenericMaterializeDetail>) {
            j["detail"] = {{"outputName", detail.outputName}};
        }
    }, node.detail);

    return j;
}

} // namespace codegen
