#include "generic/lowering/generic_ir_physical_planner.h"

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_join_carry.h"
#include "generic/lowering/generic_multi_table_checks.h"
#include "generic/lowering/generic_multi_table_join_lowering.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_scalar_lookup.h"
#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

void prependPlanPhases(MetalQueryPlan& target, MetalQueryPlan& prefix) {
    for (auto& helper : prefix.helpers) {
        if (std::find(target.helpers.begin(), target.helpers.end(), helper) ==
            target.helpers.end()) {
            target.helpers.push_back(std::move(helper));
        }
    }
    if (!prefix.phases.empty()) {
        target.phases.insert(
            target.phases.begin(),
            std::make_move_iterator(prefix.phases.begin()),
            std::make_move_iterator(prefix.phases.end()));
    }
}

std::string metalCharLiteralLocal(char c) {
    if (c == '\\') return "'\\\\'";
    if (c == '\'') return "'\\''";
    if (c == '\0') return "'\\0'";
    return std::string("'") + c + "'";
}

std::string materializedValueAt(const GenericMatColumnDesc& col,
                                const std::string& row) {
    if (col.stringLen > 0) {
        return col.bufferName + "[" + row + " * " +
               std::to_string(col.stringLen) + "u]";
    }
    return col.bufferName + "[" + row + "]";
}

std::string charDomainBucketExpr(const std::string& raw,
                                 const std::vector<char>& domain) {
    if (domain.empty()) return "";
    if (domain.size() == 1) return "0";
    std::string expr = std::to_string(domain.size() - 1);
    for (int i = static_cast<int>(domain.size()) - 2; i >= 0; --i) {
        expr = "(" + raw + " == " + metalCharLiteralLocal(domain[(size_t)i]) +
               " ? " + std::to_string(i) + " : " + expr + ")";
    }
    return expr;
}

std::string fixedStringDomainBucketExpr(const std::string& buffer,
                                        const std::string& row,
                                        int width,
                                        const std::vector<std::string>& domain) {
    if (domain.empty() || width <= 0) return "";
    if (domain.size() == 1) return "0";
    const std::string ptr = buffer + " + " + row + " * " +
                            std::to_string(width) + "u";
    std::string expr = std::to_string(domain.size() - 1);
    for (int i = static_cast<int>(domain.size()) - 2; i >= 0; --i) {
        expr = "(" + fixedStringEqMetalFromPointer(ptr, width, domain[(size_t)i]) +
               " ? " + std::to_string(i) + " : " + expr + ")";
    }
    return expr;
}

const GenericMatColumnDesc* findMaterializedColumn(
        const std::vector<GenericMatColumnDesc>& cols,
        const std::string& displayName) {
    for (const auto& col : cols) {
        if (col.displayName == displayName) return &col;
    }
    return nullptr;
}

std::vector<std::string> uniqueStrings(std::vector<std::string> values) {
    std::vector<std::string> out;
    for (auto& value : values) {
        if (std::find(out.begin(), out.end(), value) == out.end())
            out.push_back(std::move(value));
    }
    return out;
}

std::vector<std::string> intersectStrings(const std::vector<std::string>& left,
                                          const std::vector<std::string>& right) {
    std::vector<std::string> out;
    for (const auto& value : left) {
        if (std::find(right.begin(), right.end(), value) != right.end())
            out.push_back(value);
    }
    return out;
}

int maxStringLen(const std::vector<std::string>& values) {
    int width = 0;
    for (const auto& value : values)
        width = std::max(width, static_cast<int>(value.size()));
    return width;
}

std::vector<int64_t> uniqueInts(std::vector<int64_t> values) {
    std::vector<int64_t> out;
    for (auto value : values) {
        if (std::find(out.begin(), out.end(), value) == out.end())
            out.push_back(value);
    }
    return out;
}

std::vector<int64_t> intersectInts(const std::vector<int64_t>& left,
                                   const std::vector<int64_t>& right) {
    std::vector<int64_t> out;
    for (auto value : left) {
        if (std::find(right.begin(), right.end(), value) != right.end())
            out.push_back(value);
    }
    return out;
}

std::optional<int64_t> integerLiteralForDomain(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    if (auto parsed = integerStringLiteralValue(expr)) return *parsed;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    if (auto* value = std::get_if<int64_t>(&lit->value)) return *value;
    return std::nullopt;
}

std::optional<std::vector<int64_t>> finiteIntDomainForPredicate(
        const GenericPredicatePtr& pred,
        const GenericExprPtr& target) {
    if (!pred || !target) return std::nullopt;
    return std::visit([&](const auto& node)
            -> std::optional<std::vector<int64_t>> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (node.op != CmpOp::EQ) return std::nullopt;
            if (genericExprEquivalent(node.left, target)) {
                if (auto lit = integerLiteralForDomain(node.right))
                    return uniqueInts({*lit});
            }
            if (genericExprEquivalent(node.right, target)) {
                if (auto lit = integerLiteralForDomain(node.left))
                    return uniqueInts({*lit});
            }
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (!genericExprEquivalent(node.expr, target)) return std::nullopt;
            std::vector<int64_t> values;
            for (const auto& value : node.values) {
                auto lit = integerLiteralForDomain(value);
                if (!lit) return std::nullopt;
                values.push_back(*lit);
            }
            return uniqueInts(std::move(values));
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.children.empty()) return std::nullopt;
            if (node.op == GenericLogicalPred::Op::And) {
                std::optional<std::vector<int64_t>> domain;
                for (const auto& child : node.children) {
                    auto childDomain = finiteIntDomainForPredicate(child, target);
                    if (!childDomain) continue;
                    domain = domain
                        ? intersectInts(*domain, *childDomain)
                        : *childDomain;
                }
                return domain;
            }
            if (node.op == GenericLogicalPred::Op::Or) {
                std::vector<int64_t> out;
                for (const auto& child : node.children) {
                    auto childDomain = finiteIntDomainForPredicate(child, target);
                    if (!childDomain) return std::nullopt;
                    out.insert(out.end(), childDomain->begin(), childDomain->end());
                }
                return uniqueInts(std::move(out));
            }
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    }, pred->node);
}

std::optional<std::vector<std::string>> finiteStringDomainForPredicate(
        const GenericPredicatePtr& pred,
        const GenericExprPtr& target) {
    if (!pred || !target) return std::nullopt;
    return std::visit([&](const auto& node)
            -> std::optional<std::vector<std::string>> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (node.op != CmpOp::EQ) return std::nullopt;
            if (genericExprEquivalent(node.left, target)) {
                if (auto lit = stringLiteralValue(node.right))
                    return uniqueStrings({*lit});
            }
            if (genericExprEquivalent(node.right, target)) {
                if (auto lit = stringLiteralValue(node.left))
                    return uniqueStrings({*lit});
            }
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (!genericExprEquivalent(node.expr, target)) return std::nullopt;
            std::vector<std::string> values;
            for (const auto& value : node.values) {
                auto lit = stringLiteralValue(value);
                if (!lit) return std::nullopt;
                values.push_back(*lit);
            }
            return uniqueStrings(std::move(values));
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.children.empty()) return std::nullopt;
            if (node.op == GenericLogicalPred::Op::And) {
                std::optional<std::vector<std::string>> domain;
                for (const auto& child : node.children) {
                    auto childDomain = finiteStringDomainForPredicate(child, target);
                    if (!childDomain) continue;
                    domain = domain
                        ? intersectStrings(*domain, *childDomain)
                        : *childDomain;
                }
                return domain;
            }
            if (node.op == GenericLogicalPred::Op::Or) {
                std::vector<std::string> out;
                for (const auto& child : node.children) {
                    auto childDomain = finiteStringDomainForPredicate(child, target);
                    if (!childDomain) return std::nullopt;
                    out.insert(out.end(), childDomain->begin(), childDomain->end());
                }
                return uniqueStrings(std::move(out));
            }
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    }, pred->node);
}

std::string lowerAsciiLocal(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::string columnKey(const GenericColumnExpr& col) {
    return std::to_string(col.relationInstance.value) + ":" + col.column;
}

class ColumnEquivalence {
public:
    void add(const GenericColumnExpr& col) {
        const std::string key = columnKey(col);
        parent_.try_emplace(key, key);
        if (seen_.insert(key).second) columns_.push_back(col);
    }

    void unite(const GenericColumnExpr& left, const GenericColumnExpr& right) {
        add(left);
        add(right);
        const std::string a = find(columnKey(left));
        const std::string b = find(columnKey(right));
        if (a != b) parent_[a] = b;
    }

    bool equivalent(const GenericColumnExpr& left,
                    const GenericColumnExpr& right) {
        if (columnKey(left) == columnKey(right)) return true;
        add(left);
        add(right);
        return find(columnKey(left)) == find(columnKey(right));
    }

    bool equivalentToRelationColumn(const GenericColumnExpr& target,
                                    int relationInstance) {
        add(target);
        const size_t n = columns_.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& col = columns_[i];
            if (col.relationInstance.value != relationInstance) continue;
            if (equivalent(target, col)) return true;
        }
        return false;
    }

private:
    std::string find(const std::string& key) {
        auto it = parent_.find(key);
        if (it == parent_.end()) {
            parent_[key] = key;
            return key;
        }
        if (it->second == key) return key;
        it->second = find(it->second);
        return it->second;
    }

    std::map<std::string, std::string> parent_;
    std::set<std::string> seen_;
    std::vector<GenericColumnExpr> columns_;
};

void collectConjunctiveEqColumns(const GenericPredicatePtr& pred,
                                 ColumnEquivalence& eq) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (node.op != CmpOp::EQ) return;
            auto* left = node.left
                ? std::get_if<GenericColumnExpr>(&node.left->node)
                : nullptr;
            auto* right = node.right
                ? std::get_if<GenericColumnExpr>(&node.right->node)
                : nullptr;
            if (left && right) eq.unite(*left, *right);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op != GenericLogicalPred::Op::And) return;
            for (const auto& child : node.children)
                collectConjunctiveEqColumns(child, eq);
        }
    }, pred->node);
}

ColumnEquivalence buildJoinColumnEquivalence(
        const std::vector<const GenericRelNode*>& joins,
        const GenericRelNode* filterNode) {
    ColumnEquivalence eq;
    for (const auto* joinNode : joins) {
        auto* join = joinNode ? std::get_if<GenericJoinDetail>(&joinNode->detail) : nullptr;
        if (!join) continue;
        collectConjunctiveEqColumns(join->predicate, eq);
    }
    if (auto* filter = filterNode
            ? std::get_if<GenericFilterDetail>(&filterNode->detail)
            : nullptr) {
        collectConjunctiveEqColumns(filter->predicate, eq);
    }
    return eq;
}

void collectExprColumns(const GenericExprPtr& expr,
                        std::vector<GenericColumnExpr>& out) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            out.push_back(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectExprColumns(node.left, out);
            collectExprColumns(node.right, out);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches)
                collectExprColumns(branch.result, out);
            collectExprColumns(node.elseResult, out);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectExprColumns(arg, out);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectExprColumns(node.arg, out);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectExprColumns(key, out);
        }
    }, expr->node);
}

struct DateBounds {
    std::optional<int64_t> minValue;
    std::optional<int64_t> maxValue;
};

DateBounds intersectDateBounds(DateBounds left, const DateBounds& right) {
    if (right.minValue)
        left.minValue = left.minValue
            ? std::max(*left.minValue, *right.minValue)
            : right.minValue;
    if (right.maxValue)
        left.maxValue = left.maxValue
            ? std::min(*left.maxValue, *right.maxValue)
            : right.maxValue;
    return left;
}

DateBounds unionDateBounds(DateBounds left, const DateBounds& right) {
    if (right.minValue)
        left.minValue = left.minValue
            ? std::min(*left.minValue, *right.minValue)
            : right.minValue;
    if (right.maxValue)
        left.maxValue = left.maxValue
            ? std::max(*left.maxValue, *right.maxValue)
            : right.maxValue;
    return left;
}

std::optional<DateBounds> dateBoundsForPredicate(
        const GenericPredicatePtr& pred,
        const GenericExprPtr& target) {
    if (!pred || !target) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<DateBounds> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            if (!genericExprEquivalent(node.expr, target)) return std::nullopt;
            auto low = integerLiteralForDomain(node.low);
            auto high = integerLiteralForDomain(node.high);
            if (!low || !high) return std::nullopt;
            return DateBounds{*low, *high};
        } else if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            CmpOp op = node.op;
            std::optional<int64_t> lit;
            if (genericExprEquivalent(node.left, target)) {
                lit = integerLiteralForDomain(node.right);
            } else if (genericExprEquivalent(node.right, target)) {
                lit = integerLiteralForDomain(node.left);
                if (op == CmpOp::LT) op = CmpOp::GT;
                else if (op == CmpOp::LE) op = CmpOp::GE;
                else if (op == CmpOp::GT) op = CmpOp::LT;
                else if (op == CmpOp::GE) op = CmpOp::LE;
            }
            if (!lit) return std::nullopt;
            DateBounds out;
            switch (op) {
                case CmpOp::EQ:
                    out.minValue = *lit;
                    out.maxValue = *lit;
                    break;
                case CmpOp::LT:
                    out.maxValue = *lit - 1;
                    break;
                case CmpOp::LE:
                    out.maxValue = *lit;
                    break;
                case CmpOp::GT:
                    out.minValue = *lit + 1;
                    break;
                case CmpOp::GE:
                    out.minValue = *lit;
                    break;
                case CmpOp::NE:
                    return std::nullopt;
            }
            return out;
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.children.empty()) return std::nullopt;
            if (node.op == GenericLogicalPred::Op::And) {
                std::optional<DateBounds> out;
                for (const auto& child : node.children) {
                    auto childBounds = dateBoundsForPredicate(child, target);
                    if (!childBounds) continue;
                    out = out
                        ? intersectDateBounds(*out, *childBounds)
                        : *childBounds;
                }
                return out;
            }
            if (node.op == GenericLogicalPred::Op::Or) {
                std::optional<DateBounds> out;
                for (const auto& child : node.children) {
                    auto childBounds = dateBoundsForPredicate(child, target);
                    if (!childBounds) return std::nullopt;
                    out = out
                        ? unionDateBounds(*out, *childBounds)
                        : *childBounds;
                }
                return out;
            }
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    }, pred->node);
}

std::optional<int64_t> datePartDomainSizeForPredicate(
        const GenericPredicatePtr& pred,
        const GenericExprPtr& dateExpr,
        const std::string& unit) {
    auto bounds = dateBoundsForPredicate(pred, dateExpr);
    if (!bounds || !bounds->minValue || !bounds->maxValue ||
        *bounds->minValue > *bounds->maxValue) {
        return std::nullopt;
    }
    if (unit == "year") {
        int64_t minYear = *bounds->minValue / 10000;
        int64_t maxYear = *bounds->maxValue / 10000;
        if (maxYear >= minYear) return maxYear - minYear + 1;
    }
    if (unit == "month") return 12;
    if (unit == "day") return 31;
    return std::nullopt;
}

std::optional<int64_t> finiteGroupExprCardinality(
        const GenericExprPtr& expr,
        const GenericPredicatePtr& pred) {
    if (!expr) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<int64_t> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (auto ints = finiteIntDomainForPredicate(pred, expr);
                ints && !ints->empty()) {
                return static_cast<int64_t>(ints->size());
            }
            if (auto strings = finiteStringDomainForPredicate(pred, expr);
                strings && !strings->empty()) {
                return static_cast<int64_t>(strings->size());
            }
            if (!node.charDomain.empty())
                return static_cast<int64_t>(node.charDomain.size());
            if (node.hasGroupDomain && node.domainMax >= node.domainMin) {
                int64_t size = (int64_t)node.domainMax - (int64_t)node.domainMin + 1;
                if (size > 0 && size <= 4096) return size;
            }
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            std::string name = lowerAsciiLocal(node.name);
            if ((name == "date_part" || name == "extract") &&
                node.args.size() >= 2 && node.args[0]) {
                std::string unit;
                if (auto* lit = std::get_if<GenericLiteralExpr>(&node.args[0]->node)) {
                    if (auto* s = std::get_if<std::string>(&lit->value))
                        unit = lowerAsciiLocal(*s);
                }
                if (!unit.empty())
                    return datePartDomainSizeForPredicate(pred, node.args[1], unit);
            }
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    }, expr->node);
}

std::optional<int64_t> finiteGroupOutputBound(
        const GenericAggregateDetail& aggregate,
        const GenericPredicatePtr& pred) {
    int64_t total = 1;
    for (const auto& group : aggregate.groupBy) {
        auto card = finiteGroupExprCardinality(group, pred);
        if (!card || *card <= 0) return std::nullopt;
        if (total > std::numeric_limits<int64_t>::max() / *card)
            return std::nullopt;
        total *= *card;
    }
    return total;
}

std::optional<std::string> primaryKeyGroupOutputBound(
        const GenericRelPlan& ir,
        const GenericAggregateDetail& aggregate,
        const ColumnEquivalence& baseEq,
        const IrCarryMap& carryMap) {
    auto makePkColumn = [&](const GenericRelationInstance& inst,
                            const GenericRelation& rel) {
        GenericColumnExpr pk;
        pk.relationInstance = inst.id;
        pk.table = inst.baseName;
        pk.alias = inst.alias;
        pk.column = rel.primaryKeyColumn;
        return pk;
    };

    for (const auto& inst : ir.relationInstances) {
        const auto* rel = ir.findRelation(inst.relation);
        if (!rel || rel->primaryKeyColumn.empty() ||
            rel->primaryKeyDomainSymbol.empty()) {
            continue;
        }

        ColumnEquivalence eq = baseEq;
        GenericColumnExpr pk = makePkColumn(inst, *rel);
        bool pkInGroup = false;
        for (const auto& group : aggregate.groupBy) {
            std::vector<GenericColumnExpr> cols;
            collectExprColumns(group, cols);
            for (const auto& col : cols) {
                if (eq.equivalent(col, pk)) {
                    pkInGroup = true;
                    break;
                }
            }
            if (pkInGroup) break;
        }
        if (!pkInGroup) continue;

        std::set<int> determinedRelations{inst.id.value};
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& otherInst : ir.relationInstances) {
                if (determinedRelations.count(otherInst.id.value)) continue;
                const auto* otherRel = ir.findRelation(otherInst.relation);
                if (!otherRel || otherRel->primaryKeyColumn.empty()) continue;
                GenericColumnExpr otherPk = makePkColumn(otherInst, *otherRel);
                for (int determined : determinedRelations) {
                    if (eq.equivalentToRelationColumn(otherPk, determined)) {
                        determinedRelations.insert(otherInst.id.value);
                        changed = true;
                        break;
                    }
                }
            }
        }

        bool allDetermined = true;
        for (const auto& group : aggregate.groupBy) {
            std::vector<GenericColumnExpr> cols;
            collectExprColumns(group, cols);
            if (cols.empty()) {
                allDetermined = false;
                break;
            }
            for (const auto& col : cols) {
                bool determined = determinedRelations.count(
                    col.relationInstance.value) > 0;
                if (!determined) {
                    for (int relation : determinedRelations) {
                        if (eq.equivalentToRelationColumn(col, relation)) {
                            determined = true;
                            break;
                        }
                    }
                }
                if (!determined) {
                    auto relIt = carryMap.find(col.relationInstance.value);
                    auto colIt = relIt == carryMap.end()
                        ? std::map<std::string, IrCarryColumn>::const_iterator{}
                        : relIt->second.find(col.column);
                    if (relIt != carryMap.end() && colIt != relIt->second.end()) {
                        const auto& carried = colIt->second.column;
                        determined = determinedRelations.count(
                            carried.relationInstance.value) > 0;
                        if (!determined) {
                            for (int relation : determinedRelations) {
                                if (eq.equivalentToRelationColumn(carried, relation)) {
                                    determined = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (!determined) {
                    allDetermined = false;
                    break;
                }
            }
            if (!allDetermined) break;
        }
        if (allDetermined) {
            if (!rel->virtualRelation && !inst.baseName.empty())
                return tableSizeName(inst.baseName);
            return rel->primaryKeyDomainSymbol;
        }
    }
    return std::nullopt;
}

class MetalMaterializedRangeScan : public MetalOperator {
public:
    MetalMaterializedRangeScan(std::string rowsSymbol,
                               std::string idxVar,
                               std::vector<GenericMatColumnDesc> columns)
        : rowsSymbol_(std::move(rowsSymbol)),
          idxVar_(std::move(idxVar)),
          columns_(std::move(columns)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string nParam = tableSizeName(rowsSymbol_);
        cg.setPhaseScannedTable(rowsSymbol_);
        cg.addResolvedScalarParam(nParam, "uint", rowsSymbol_);
        for (const auto& col : columns_) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
        }
        cg.addBlock("for (uint " + idxVar_ + " = tid; " + idxVar_ + " < " +
                    nParam + "; " + idxVar_ + " += tpg)", [&]() {
            consume();
        });
    }

    std::string describe() const override {
        return "MaterializedRangeScan(" + rowsSymbol_ + ")";
    }

private:
    std::string rowsSymbol_;
    std::string idxVar_;
    std::vector<GenericMatColumnDesc> columns_;
};

} // namespace

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableGroupedAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!aggregate)
        return fail(error, "IR multi-table grouped aggregate lowerer: malformed aggregate detail.");
    if (aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR multi-table grouped aggregate lowerer: no aggregate outputs.");

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& group : aggregate->groupBy)
        neededExprs.push_back(group);
    for (const auto& projection : aggregate->aggregates) {
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table grouped aggregate lowerer: non-aggregate projection.");
        if (agg->arg)
            neededExprs.push_back(agg->arg);
    }

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq) && groupedAggregateNeedsScalarPreAgg(*shape)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_GROUP_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table grouped aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "GENERIC_IR_MULTI_TABLE_GROUP", aq,
        scalarLookups.empty() ? nullptr : &scalarLookups, error);
    if (!lowering) return std::nullopt;
    prependPlanPhases(lowering->plan, scalarPreAggPlan);

    const std::string idxVar = "i";
    const std::string resultCounter = "d_ir_multi_group_input_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(lowering->probePipe), resultCounter, "1");

    std::vector<GenericMatColumnDesc> materializedCols;
    GenericGroupSpec groupSpec;
    std::vector<IrGroupKeyDesc> groupKeys;
    int matColIdx = 0;

    auto addInputColumn = [&](const std::string& displayName,
                              const TypeInfo& type,
                              const GenericExprPtr& expr,
                              int scaleDown,
                              const std::string& distinctDomainSymbol) -> bool {
        if (!materializeExprSupported(expr)) {
            if (error)
                *error = "IR multi-table grouped aggregate lowerer: input expression '" +
                         displayName + "' is not supported.";
            return false;
        }
        if (exprNeedsCarriedString(expr, lowering->carryMap)) {
            if (error)
                *error = "IR multi-table grouped aggregate lowerer: carried string input '" +
                         displayName + "' is not supported yet.";
            return false;
        }
        const IrCarryColumn* carriedFixedString = nullptr;
        if (auto* col = expr ? std::get_if<GenericColumnExpr>(&expr->node) : nullptr) {
            auto relIt = lowering->carryMap.find(col->relationInstance.value);
            if (relIt != lowering->carryMap.end()) {
                auto colIt = relIt->second.find(col->column);
                if (colIt != relIt->second.end() &&
                    col->type.type == DataType::CHAR_FIXED) {
                    carriedFixedString = &colIt->second;
                }
            }
        }
        int stringLen = materializedStringLenForExpr(expr, lowering->carryMap);
        std::string sizeExpr = lowering->outputSize;
        std::string bufferName = "d_ir_multi_group_" + std::to_string(matColIdx++) +
                                 "_" + sanitizeIdentifier(displayName);
        std::string metalType = carriedFixedString ? "uint" : metalTypeForType(type);
        if (stringLen > 0 && !carriedFixedString)
            sizeExpr += " * " + std::to_string(stringLen);
        materialize->addColumn(bufferName, metalType,
                               carriedFixedString
                                   ? carriedFixedString->rowVarName
                                   : materializeExprToMetalWithCarryMap(
                                         expr, idxVar, lowering->carryMap),
                               displayName, sizeExpr,
                               carriedFixedString ? 0 : stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            displayName, bufferName, metalType, stringLen, scaleDown, false,
            distinctDomainSymbol, carriedFixedString != nullptr,
            carriedFixedString ? carriedFixedString->column.table : std::string{},
            carriedFixedString ? carriedFixedString->column.column : std::string{}});
        return true;
    };

    for (size_t i = 0; i < aggregate->groupBy.size(); ++i) {
        const auto& group = aggregate->groupBy[i];
        const std::string displayName = groupDisplayNameForAggregate(*aggregate, i);
        groupSpec.keyColumns.push_back(displayName);
        IrGroupKeyDesc key;
        key.displayName = displayName;
        groupKeys.push_back(std::move(key));
        if (!addInputColumn(displayName, group ? group->type : TypeInfo{DataType::INT, 0},
                            group, 0, "")) {
            return std::nullopt;
        }
    }

    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table grouped aggregate lowerer: non-aggregate projection.");
        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;

        GenericExprPtr inputExpr;
        TypeInfo inputType{DataType::FLOAT, 0};
        int inputScaleDown = 0;
        std::string distinctDomainSymbol;
        std::string funcName = aggregateOutputFuncFor(*aggregate, i, agg->func);

        if (agg->func == AggFunc::COUNT) {
            GenericExpr lit;
            lit.type = inputType;
            lit.node = GenericLiteralExpr{1.0, inputType};
            inputExpr = std::make_shared<GenericExpr>(std::move(lit));
            funcName = "COUNT";
        } else {
            if (!agg->arg)
                return fail(error, "IR multi-table grouped aggregate lowerer: aggregate '" +
                                   aggFuncName(agg->func) + "' requires an argument.");
            inputExpr = agg->arg;
            if (agg->func == AggFunc::COUNT_DISTINCT) {
                if (agg->arg->type.type == DataType::CHAR_FIXED)
                    return fail(error, "IR multi-table grouped aggregate lowerer: COUNT(DISTINCT) over fixed strings is not supported yet.");
                distinctDomainSymbol = distinctDomainSymbolForExpr(agg->arg);
                inputType = agg->arg->type;
                funcName = "COUNT_DISTINCT";
            } else if (agg->func == AggFunc::SUM || agg->func == AggFunc::AVG) {
                inputScaleDown = numericScaleForExpr(agg->arg);
            } else if (agg->func != AggFunc::MIN && agg->func != AggFunc::MAX) {
                return fail(error, "IR multi-table grouped aggregate lowerer: unsupported aggregate " +
                                   aggFuncName(agg->func) + ".");
            }
        }

        groupSpec.aggColumns.push_back(displayName);
        groupSpec.aggFuncs.push_back(funcName);
        if (!addInputColumn(displayName, inputType, inputExpr, inputScaleDown,
                            distinctDomainSymbol)) {
            return std::nullopt;
        }
    }

    groupSpec.outputColumns = aggregate->outputOrder;
    if (!configureAggregateHaving(*aggregate, groupSpec, aq, &*shape, error))
        return std::nullopt;

    auto& matPhase = appendPhase(lowering->plan, "GENERIC_ir_multi_table_group_materialize",
                                 std::move(materialize));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(matPhase, scalarLookups);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, *aggregate, groupKeys);
            if (!name)
                return fail(error, "IR multi-table grouped aggregate lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!aggregate->having && !aggregateNeedsHashGroupOutput(*aggregate)) {
        bool directOk = true;
        int totalBuckets = 1;
        std::vector<IrGroupKeyDesc> denseKeys;
        denseKeys.reserve(aggregate->groupBy.size());

        const std::string idxVar = "i";
        for (size_t i = 0; i < aggregate->groupBy.size(); ++i) {
            const auto& group = aggregate->groupBy[i];
            auto* col = group ? std::get_if<GenericColumnExpr>(&group->node) : nullptr;
            const std::string displayName = groupDisplayNameForAggregate(*aggregate, i);
            const auto* matCol = findMaterializedColumn(materializedCols, displayName);
            if (!matCol) {
                directOk = false;
                break;
            }

            IrGroupKeyDesc key;
            key.displayName = displayName;
            key.stride = totalBuckets;
            const std::string raw = materializedValueAt(*matCol, idxVar);
            const auto* fd = shape->filter ? filterDetail(shape->filter) : nullptr;
            auto finiteIntDomain = finiteIntDomainForPredicate(
                fd ? fd->predicate : GenericPredicatePtr{}, group);
            if ((group->type.type == DataType::INT ||
                 group->type.type == DataType::DATE) &&
                finiteIntDomain && !finiteIntDomain->empty()) {
                auto [minIt, maxIt] = std::minmax_element(
                    finiteIntDomain->begin(), finiteIntDomain->end());
                int64_t minValue = *minIt;
                int64_t maxValue = *maxIt;
                int64_t domainSize = maxValue - minValue + 1;
                if (domainSize <= 0 || domainSize > 4096 ||
                    minValue < std::numeric_limits<int>::min() ||
                    maxValue > std::numeric_limits<int>::max()) {
                    directOk = false;
                    break;
                }
                key.numValues = static_cast<int>(domainSize);
                key.keyBase = static_cast<int>(minValue);
                key.keyExpr = minValue != 0
                    ? "(" + raw + " - " + std::to_string(minValue) + ")"
                    : raw;
                key.keyExpr = "clamp(" + key.keyExpr + ", 0, " +
                              std::to_string(key.numValues - 1) + ")";
            } else if (col && col->type.type == DataType::CHAR1) {
                key.keyExpr = charDomainBucketExpr(raw, col->charDomain);
                key.numValues = static_cast<int>(col->charDomain.size());
                key.charMap = col->charDomain;
                if (key.keyExpr.empty() || key.numValues <= 0) {
                    directOk = false;
                    break;
                }
            } else if (col && col->type.type == DataType::CHAR_FIXED) {
                auto domain = finiteStringDomainForPredicate(
                    fd ? fd->predicate : GenericPredicatePtr{}, group);
                if (!domain || domain->empty() || matCol->stringLen <= 0) {
                    directOk = false;
                    break;
                }
                key.stringMap = *domain;
                key.stringLen = std::max(matCol->stringLen, maxStringLen(key.stringMap));
                key.numValues = static_cast<int>(key.stringMap.size());
                key.keyExpr = fixedStringDomainBucketExpr(
                    matCol->bufferName, idxVar, matCol->stringLen, key.stringMap);
                if (key.keyExpr.empty() || key.numValues <= 0) {
                    directOk = false;
                    break;
                }
            } else if (col && (col->type.type == DataType::INT ||
                        col->type.type == DataType::DATE) &&
                       col->hasGroupDomain &&
                       col->domainMax >= col->domainMin) {
                key.numValues = col->domainMax - col->domainMin + 1;
                key.keyBase = col->domainMin;
                key.keyExpr = col->domainMin != 0
                    ? "(" + raw + " - " + std::to_string(col->domainMin) + ")"
                    : raw;
                key.keyExpr = "clamp(" + key.keyExpr + ", 0, " +
                              std::to_string(key.numValues - 1) + ")";
            } else {
                directOk = false;
                break;
            }

            totalBuckets *= key.numValues;
            if (key.numValues <= 0 || totalBuckets > 4096) {
                directOk = false;
                break;
            }
            denseKeys.push_back(std::move(key));
        }

        std::vector<IrPendingAgg> pending;
        int valuesPerBucket = 0;
        if (directOk) {
            for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
                const auto& projection = aggregate->aggregates[i];
                auto* agg = projection.expr
                    ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
                    : nullptr;
                if (!agg) {
                    directOk = false;
                    break;
                }
                const std::string displayName = projection.name.empty()
                    ? "agg_" + std::to_string(i)
                    : projection.name;

                if (agg->func == AggFunc::COUNT_DISTINCT ||
                    agg->func == AggFunc::MIN ||
                    agg->func == AggFunc::MAX) {
                    directOk = false;
                    break;
                }

                if (agg->func == AggFunc::COUNT) {
                    IrPendingAgg out;
                    out.displayName = displayName;
                    out.offset = valuesPerBucket++;
                    out.valueExpr = "1u";
                    out.funcName = "COUNT";
                    pending.push_back(std::move(out));
                    continue;
                }

                const auto* matCol = findMaterializedColumn(materializedCols, displayName);
                if (!agg->arg || !matCol || matCol->stringLen > 0) {
                    directOk = false;
                    break;
                }
                std::string valueExpr = materializedValueAt(*matCol, idxVar);
                if (agg->func == AggFunc::AVG) {
                    const int fixedScale = matCol->scaleDown;
                    IrPendingAgg sum;
                    sum.displayName = displayName;
                    sum.offset = valuesPerBucket;
                    sum.valueExpr = valueExpr;
                    if (agg->arg->type.type == DataType::FLOAT && fixedScale > 0) {
                        sum.valueExpr = scaledLongExpr(valueExpr, fixedScale);
                        sum.isLongPair = true;
                        sum.scaleDown = -fixedScale;
                        valuesPerBucket += 2;
                    } else if (agg->arg->type.type == DataType::FLOAT) {
                        sum.isFloatSum = true;
                        sum.scaleDown = -1;
                        valuesPerBucket += 1;
                    } else {
                        sum.isLongPair = true;
                        sum.scaleDown = -1;
                        valuesPerBucket += 2;
                    }
                    sum.funcName = "AVG";
                    sum.innerColumn = innerColumnName(agg->arg);
                    pending.push_back(std::move(sum));

                    IrPendingAgg cnt;
                    cnt.displayName = displayName + "_cnt";
                    cnt.offset = valuesPerBucket++;
                    cnt.valueExpr = "1u";
                    cnt.funcName = "AVG";
                    pending.push_back(std::move(cnt));
                    continue;
                }

                if (agg->func == AggFunc::SUM) {
                    IrPendingAgg out;
                    out.displayName = displayName;
                    out.offset = valuesPerBucket;
                    out.valueExpr = valueExpr;
                    out.funcName = "SUM";
                    out.innerColumn = innerColumnName(agg->arg);
                    if (agg->arg->type.type == DataType::FLOAT && matCol->scaleDown > 0) {
                        out.valueExpr = scaledLongExpr(valueExpr, matCol->scaleDown);
                        out.isLongPair = true;
                        out.scaleDown = matCol->scaleDown;
                        valuesPerBucket += 2;
                    } else if (agg->arg->type.type == DataType::FLOAT) {
                        out.isFloatSum = true;
                        valuesPerBucket += 1;
                    } else {
                        out.isLongPair = true;
                        valuesPerBucket += 2;
                    }
                    pending.push_back(std::move(out));
                    continue;
                }

                directOk = false;
                break;
            }
        }

        if (directOk && !pending.empty()) {
            std::string bucketExpr = "(" + denseKeys.front().keyExpr + ")";
            for (size_t i = 1; i < denseKeys.size(); ++i) {
                bucketExpr = "(" + bucketExpr + " + (" + denseKeys[i].keyExpr + ") * " +
                             std::to_string(denseKeys[i].stride) + ")";
            }

            const std::string rowsSym = "ir_multi_direct_group_rows";
            auto scan = std::make_unique<MetalMaterializedRangeScan>(
                rowsSym, idxVar, materializedCols);
            auto keyed = std::make_unique<MetalKeyedAgg>(
                std::move(scan), "d_ir_multi_direct_group_aggs", bucketExpr,
                totalBuckets, valuesPerBucket,
                std::to_string(totalBuckets * valuesPerBucket));

            std::vector<std::string> keyNames;
            std::vector<GroupKeyDecode> decodeInfo;
            for (const auto& key : denseKeys) {
                keyNames.push_back(key.displayName);
                GroupKeyDecode d;
                d.name = key.displayName;
                d.numValues = key.numValues;
                d.stride = key.stride;
                d.charMap = key.charMap;
                d.keyBase = key.keyBase;
                decodeInfo.push_back(std::move(d));
            }
            keyed->setMultiKeyResult(keyNames, decodeInfo, totalBuckets);
            for (const auto& agg : pending) {
                keyed->addAggregateWithMeta(agg.displayName, agg.offset, agg.valueExpr,
                                            agg.atomicOp, agg.isLongPair, agg.scaleDown,
                                            agg.isFloatSum, agg.isMinMax,
                                            agg.funcName, agg.innerColumn);
            }

            attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
            auto& directPhase = appendPhase(lowering->plan,
                "GENERIC_ir_multi_table_direct_group", std::move(keyed));

            std::vector<KeyedCompactKeySpec> compactKeys;
            std::vector<GenericMatColumnDesc> compactCols;
            for (const auto& key : denseKeys) {
                compactKeys.push_back({key.displayName, key.numValues, key.stride,
                                       key.charMap, key.keyBase, key.stringMap,
                                       key.stringLen});
                std::string buf = "d_ir_multi_direct_out_" +
                                  std::to_string(compactCols.size()) + "_" +
                                  sanitizeIdentifier(key.displayName);
                const bool isStringKey = !key.charMap.empty() || !key.stringMap.empty();
                compactCols.push_back(GenericMatColumnDesc{
                    key.displayName, buf, isStringKey ? "char" : "int",
                    key.stringMap.empty() ? 0 : key.stringLen});
            }

            std::vector<KeyedCompactAggSpec> compactAggs;
            for (size_t pi = 0; pi < pending.size(); ++pi) {
                const auto& p = pending[pi];
                KeyedCompactAggSpec out;
                out.displayName = p.displayName;
                out.offset = p.offset;
                out.isLongPair = p.isLongPair;
                out.scaleDown = p.scaleDown;
                out.isFloatSum = p.isFloatSum;
                out.isMinMax = p.isMinMax;
                out.atomicOp = p.atomicOp;
                out.avgSumIsLongPair = p.isLongPair;

                std::string metalType = "uint";
                int outScale = 0;
                bool outLongPair = false;
                if (p.scaleDown < 0 && pi + 1 < pending.size()) {
                    const auto& cnt = pending[pi + 1];
                    out.isAvg = true;
                    out.countOffset = cnt.offset;
                    out.countIsFloat = cnt.isFloatSum;
                    metalType = "float";
                    ++pi;
                } else if (p.isLongPair) {
                    metalType = "uint";
                    outScale = p.scaleDown > 0 ? p.scaleDown : 0;
                    outLongPair = true;
                    out.isLongPair = true;
                } else if (p.isFloatSum || p.isMinMax) {
                    metalType = "float";
                }
                std::string buf = "d_ir_multi_direct_out_" +
                                  std::to_string(compactCols.size()) + "_" +
                                  sanitizeIdentifier(out.displayName);
                compactAggs.push_back(out);
                compactCols.push_back(GenericMatColumnDesc{
                    out.displayName, buf, metalType, 0, outScale, outLongPair});
            }

            const std::string compactCounter = "d_ir_multi_direct_result_count";
            auto& compactPhase = appendPhase(lowering->plan,
                "GENERIC_ir_multi_table_direct_group_compact",
                makeKeyedAggCompactOperator(
                    "d_ir_multi_direct_group_aggs", compactCounter, totalBuckets,
                    valuesPerBucket, compactKeys, compactAggs, compactCols));
            (void)directPhase;

            if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
                const std::string sortRowsSym = "n_gpu_sort_ir_multi_direct_group_rows";
                attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);
                if (!appendGenericGpuSort(lowering->plan, "ir_multi_direct_group",
                                          sortRowsSym, std::to_string(totalBuckets),
                                          compactCols, sortSpec, error)) {
                    return std::nullopt;
                }
            }

            return std::move(lowering->plan);
        }
    }

    std::string groupOutputBoundExpr = lowering->outputSize;
    const auto* fd = shape->filter ? filterDetail(shape->filter) : nullptr;
    auto eq = buildJoinColumnEquivalence(shape->joins, shape->filter);
    if (auto pkBound = primaryKeyGroupOutputBound(
            ir, *aggregate, eq, lowering->carryMap)) {
        groupOutputBoundExpr = *pkBound;
    } else if (auto finiteBound = finiteGroupOutputBound(
            *aggregate, fd ? fd->predicate : GenericPredicatePtr{})) {
        groupOutputBoundExpr = std::to_string(*finiteBound);
    }

    const std::string groupTag = "ir_multi_table_group";
    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = groupOutputBoundExpr == lowering->outputSize
        ? "next_pow2(" + lowering->outputSize + " * 2)"
        : "next_pow2(" + groupOutputBoundExpr + " * 2 + 16)";
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.maxOutputRowsExpr = groupOutputBoundExpr;
    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
    gbSpec.inputColumns = std::move(materializedCols);
    gbSpec.groupBy = std::move(groupSpec);
    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
    appendGenericGpuGroupBy(lowering->plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(lowering->plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        if (!appendGenericGpuSort(lowering->plan, "group_" + groupTag,
                                  sortRowsSym, gbSpec.maxOutputRowsExpr,
                                  genericGpuGroupOutputColumns(gbSpec),
                                  sortSpec, error)) {
            return std::nullopt;
        }
    }

    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableGroupedAggregateIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableGroupedAggregateIRToMetalImpl(ir, &aq, error);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableScalarAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!aggregate)
        return fail(error, "IR multi-table scalar aggregate lowerer: malformed aggregate detail.");
    if (!aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR multi-table scalar aggregate lowerer: no aggregate outputs.");
    if (aggregate->having)
        return fail(error, "IR multi-table scalar aggregate lowerer: HAVING is not supported.");

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& projection : aggregate->aggregates) {
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table scalar aggregate lowerer: non-aggregate projection.");
        if (agg->arg)
            neededExprs.push_back(agg->arg);
    }

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq)) {
        scalarPreAggPlan.name = "GENERIC_IR_MULTI_TABLE_SCALAR_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table scalar aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "GENERIC_IR_MULTI_TABLE_SCALAR", aq,
        scalarLookups.empty() ? nullptr : &scalarLookups, error);
    if (!lowering) return std::nullopt;
    prependPlanPhases(lowering->plan, scalarPreAggPlan);

    auto expressionSupported = [&](const GenericExprPtr& expr,
                                   const std::string& displayName) -> bool {
        if (!expr) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' requires an argument.";
            return false;
        }
        if (!materializeExprSupported(expr)) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' argument is not supported.";
            return false;
        }
        if (exprNeedsCarriedString(expr, lowering->carryMap)) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: carried string aggregate input '" +
                         displayName + "' is not supported yet.";
            return false;
        }
        if (expr->type.type != DataType::INT &&
            expr->type.type != DataType::DATE &&
            expr->type.type != DataType::FLOAT) {
            if (error)
                *error = "IR multi-table scalar aggregate lowerer: aggregate '" +
                         displayName + "' argument must be numeric.";
            return false;
        }
        return true;
    };

    const std::string idxVar = "i";
    auto reduce = std::make_unique<MetalTGReduce>(
        std::move(lowering->probePipe), "d_ir_multi_scalar");
    std::vector<bool> consumed(aggregate->aggregates.size(), false);

    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        if (consumed[i]) continue;
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR multi-table scalar aggregate lowerer: non-aggregate projection.");

        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        const std::string accName = "a" + std::to_string(i) + "_" +
                                    sanitizeIdentifier(displayName);
        const std::string outputFunc = aggregateOutputFuncFor(*aggregate, i,
                                                              agg->func);

        if (outputFunc == "RATIO_DEN") {
            consumed[i] = true;
            continue;
        }

        if (outputFunc == "RATIO") {
            if (i + 1 >= aggregate->aggregates.size())
                return fail(error, "IR multi-table scalar aggregate lowerer: RATIO denominator is missing.");
            const auto& denProjection = aggregate->aggregates[i + 1];
            auto* denAgg = denProjection.expr
                ? std::get_if<GenericAggregateExpr>(&denProjection.expr->node)
                : nullptr;
            if (!denAgg || aggregateOutputFuncFor(*aggregate, i + 1,
                                                  denAgg->func) != "RATIO_DEN") {
                return fail(error, "IR multi-table scalar aggregate lowerer: RATIO denominator metadata is invalid.");
            }
            if (!expressionSupported(agg->arg, displayName) ||
                !expressionSupported(denAgg->arg, denProjection.name)) {
                return std::nullopt;
            }
            std::string numExpr = materializeExprToMetalWithCarryMap(
                agg->arg, idxVar, lowering->carryMap);
            std::string denExpr = materializeExprToMetalWithCarryMap(
                denAgg->arg, idxVar, lowering->carryMap);
            int numIdx = reduce->addAccumulator(accName + "_num", numExpr, "float");
            int denIdx = reduce->addAccumulator(accName + "_den", denExpr, "float");
            reduce->setAverageResultAlias(displayName, numIdx, denIdx, 0, nullptr);
            consumed[i] = true;
            consumed[i + 1] = true;
            continue;
        }

        if (agg->func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(displayName, accIndex, 0, nullptr);
            consumed[i] = true;
            continue;
        }

        if (!expressionSupported(agg->arg, displayName))
            return std::nullopt;
        std::string valueExpr = materializeExprToMetalWithCarryMap(
            agg->arg, idxVar, lowering->carryMap);

        if (agg->func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum",
                                                  valueExpr, "float");
            int countIndex = reduce->addAccumulator(accName + "_count",
                                                    "1.0f", "float");
            reduce->setAverageResultAlias(displayName, sumIndex, countIndex,
                                          0, nullptr);
            consumed[i] = true;
            continue;
        }

        std::string outputType =
            agg->arg->type.type == DataType::FLOAT ? "float" : "long";
        MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
        if (agg->func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
        else if (agg->func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
        else if (agg->func != AggFunc::SUM) {
            return fail(error, "IR multi-table scalar aggregate lowerer: unsupported aggregate '" +
                               aggFuncName(agg->func) + "'.");
        }
        if (op != MetalTGReduce::ReduceOp::SUM &&
            agg->arg->type.type != DataType::FLOAT) {
            outputType = "int";
        }
        int accIndex = reduce->addAccumulator(accName, valueExpr, outputType,
                                              "", "", op);
        reduce->setAccumulatorResultAlias(displayName, accIndex, 0, nullptr);
        consumed[i] = true;
    }

    auto& scalarPhase = appendPhase(lowering->plan, "GENERIC_ir_multi_table_scalar",
                                    std::move(reduce));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(scalarPhase, scalarLookups);
    if (scalarLookups.empty())
        lowering->plan.chunkable = true;
    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetalImpl(ir, &aq, error);
}

} // namespace codegen
