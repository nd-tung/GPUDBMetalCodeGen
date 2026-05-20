#include "generic/lowering/generic_ir_physical_planner.h"

#include "generic/gpu_ops/generic_gpu_physical_ops.h"
#include "generic/lowering/generic_aggregate_helpers.h"
#include "generic/lowering/generic_cost_model.h"
#include "generic/lowering/generic_expression_metal.h"
#include "generic/lowering/generic_join_carry.h"
#include "generic/lowering/generic_multi_table_checks.h"
#include "generic/lowering/generic_multi_table_join_lowering.h"
#include "generic/lowering/generic_plan_shapes.h"
#include "generic/lowering/generic_relation_analysis.h"
#include "generic/lowering/generic_scalar_lookup.h"
#include "generic/lowering/generic_scalar_preagg_lowering.h"
#include "execution/metal_generic_executor.h"
#include "metal_plan_common.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace codegen {

namespace {

constexpr const char* kFdHiddenBucketDisplay = "__hidden_fd_bucket";

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
    if (!prefix.costTraces.empty()) {
        target.costTraces.insert(
            target.costTraces.begin(),
            std::make_move_iterator(prefix.costTraces.begin()),
            std::make_move_iterator(prefix.costTraces.end()));
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
    if (col.stringRowRef) {
        return col.bufferName + "[" + row + "]";
    }
    if (col.stringLen > 0) {
        return col.bufferName + "[" + row + " * " +
               std::to_string(col.stringLen) + "ul]";
    }
    return col.bufferName + "[" + row + "]";
}

std::string domainCarryStorageBufferName(
        const GenericJoinDomainBitmapInfo& storage,
        const GenericColumnExpr& col) {
    std::string storageScope =
        !storage.storageAlias.empty() ? storage.storageAlias
                                      : storage.storageTable;
    std::string originScope = !col.alias.empty() ? col.alias : col.table;
    return "d_ir_carry_" + sanitizeIdentifier(storageScope) + "_" +
           sanitizeIdentifier(originScope) + "_" +
           sanitizeIdentifier(col.column);
}

void appendPostDispatchHook(MetalQueryPlan::Phase& phase,
                            PostDispatchHook hook) {
    auto previous = std::move(phase.postDispatchHook);
    phase.postDispatchHook =
        [previous = std::move(previous), hook = std::move(hook)](
                MetalGenericExecutor& executor) mutable {
            double extraMs = 0.0;
            if (previous) extraMs += previous(executor);
            if (hook) extraMs += hook(executor);
            return extraMs;
        };
}

void attachHavingTotalScalarHook(MetalQueryPlan::Phase& phase,
                                 std::string totalBuffer,
                                 std::string scalarName,
                                 bool isLongPair,
                                 bool isFloatSum) {
    appendPostDispatchHook(
        phase,
        [totalBuffer = std::move(totalBuffer),
         scalarName = std::move(scalarName),
         isLongPair,
         isFloatSum](MetalGenericExecutor& executor) {
            auto* buffer = executor.getAllocatedBuffer(totalBuffer);
            if (!buffer) return 0.0;
            const auto* words =
                static_cast<const uint32_t*>(buffer->contents());
            float value = 0.0f;
            if (isLongPair) {
                uint64_t raw = (static_cast<uint64_t>(words[1]) << 32) |
                               static_cast<uint64_t>(words[0]);
                int64_t signedValue = 0;
                std::memcpy(&signedValue, &raw, sizeof(signedValue));
                value = static_cast<float>(signedValue);
            } else if (isFloatSum) {
                std::memcpy(&value, words, sizeof(value));
            } else {
                value = static_cast<float>(words[0]);
            }
            executor.registerScalarFloat(scalarName, value);
            return 0.0;
        });
}

void replaceAllInPlace(std::string& value,
                       const std::string& from,
                       const std::string& to) {
    if (from.empty() || from == to) return;
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
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

std::string fixedStringPointerDomainBucketExpr(
        const std::string& ptr,
        int width,
        const std::vector<std::string>& domain) {
    if (domain.empty() || width <= 0) return "";
    if (domain.size() == 1) return "0";
    std::string expr = std::to_string(domain.size() - 1);
    for (int i = static_cast<int>(domain.size()) - 2; i >= 0; --i) {
        expr = "(" + fixedStringEqMetalFromPointer(ptr, width, domain[(size_t)i]) +
               " ? " + std::to_string(i) + " : " + expr + ")";
    }
    return expr;
}

std::string fixedStringDomainBucketExpr(const std::string& buffer,
                                        const std::string& row,
                                        int width,
                                        const std::vector<std::string>& domain) {
    if (domain.empty() || width <= 0) return "";
    const std::string ptr = buffer + " + " + row + " * " +
                            std::to_string(width) + "ul";
    return fixedStringPointerDomainBucketExpr(ptr, width, domain);
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

    std::vector<GenericColumnExpr> columnsEquivalentTo(
            const GenericColumnExpr& target) {
        add(target);
        std::vector<GenericColumnExpr> out;
        const size_t n = columns_.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& col = columns_[i];
            if (equivalent(target, col)) out.push_back(col);
        }
        return out;
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
        if (auto* col = dateExpr
                ? std::get_if<GenericColumnExpr>(&dateExpr->node)
                : nullptr) {
            if (col->hasGroupDomain && col->domainMax >= col->domainMin) {
                if (unit == "year") {
                    int64_t minYear = col->domainMin / 10000;
                    int64_t maxYear = col->domainMax / 10000;
                    if (maxYear >= minYear) return maxYear - minYear + 1;
                }
                if (unit == "month") return 12;
                if (unit == "day") return 31;
            }
        }
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

std::optional<std::pair<int64_t, int64_t>> datePartBoundsForPredicate(
        const GenericPredicatePtr& pred,
        const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* fn = std::get_if<GenericFunctionExpr>(&expr->node);
    if (!fn || fn->args.size() < 2 || !fn->args[0]) return std::nullopt;
    std::string name = lowerAsciiLocal(fn->name);
    if (name != "date_part" && name != "extract") return std::nullopt;
    std::string unit;
    if (auto* lit = std::get_if<GenericLiteralExpr>(&fn->args[0]->node)) {
        if (auto* s = std::get_if<std::string>(&lit->value))
            unit = lowerAsciiLocal(*s);
    }
    if (unit != "year" && unit != "month" && unit != "day")
        return std::nullopt;
    auto bounds = dateBoundsForPredicate(pred, fn->args[1]);
    if (!bounds || !bounds->minValue || !bounds->maxValue ||
        *bounds->minValue > *bounds->maxValue) {
        if (auto* col = std::get_if<GenericColumnExpr>(&fn->args[1]->node)) {
            if (col->hasGroupDomain && col->domainMax >= col->domainMin) {
                if (unit == "year") {
                    int64_t minYear = col->domainMin / 10000;
                    int64_t maxYear = col->domainMax / 10000;
                    if (maxYear >= minYear)
                        return std::make_pair(minYear, maxYear);
                }
                if (unit == "month") return std::make_pair<int64_t, int64_t>(1, 12);
                if (unit == "day") return std::make_pair<int64_t, int64_t>(1, 31);
            }
        }
        return std::nullopt;
    }
    if (unit == "year") {
        int64_t minYear = *bounds->minValue / 10000;
        int64_t maxYear = *bounds->maxValue / 10000;
        if (maxYear < minYear) return std::nullopt;
        return std::make_pair(minYear, maxYear);
    }
    if (unit == "month") return std::make_pair<int64_t, int64_t>(1, 12);
    if (unit == "day") return std::make_pair<int64_t, int64_t>(1, 31);
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

std::optional<std::string> relationInstanceRowBoundExpr(
        const GenericRelPlan& ir,
        int relationInstanceId) {
    const auto* inst = ir.findRelationInstance(
        GenericRelationInstanceId{relationInstanceId});
    if (!inst) return std::nullopt;
    const auto* rel = ir.findRelation(inst->relation);
    if (!rel) return std::nullopt;
    if (!rel->virtualRelation && !inst->baseName.empty())
        return tableSizeName(inst->baseName);
    if (!rel->primaryKeyDomainSymbol.empty())
        return rel->primaryKeyDomainSymbol;
    if (!rel->maxKeySymbol.empty())
        return rel->maxKeySymbol;
    return std::nullopt;
}

std::optional<int> positiveIntLiteralString(const std::string& text) {
    if (text.empty()) return std::nullopt;
    int64_t value = 0;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c)))
            return std::nullopt;
        value = value * 10 + (c - '0');
        if (value > std::numeric_limits<int>::max())
            return std::nullopt;
    }
    if (value <= 0) return std::nullopt;
    return static_cast<int>(value);
}

std::optional<int> staticBaseTableRowBound(const GenericRelPlan& ir,
                                           const std::string& table) {
    if (table.empty()) return std::nullopt;
    for (const auto& inst : ir.relationInstances) {
        const auto* rel = ir.findRelation(inst.relation);
        if (!rel || rel->virtualRelation) continue;
        if (inst.baseName != table && rel->name != table) continue;
        if (auto domain = positiveIntLiteralString(rel->primaryKeyDomainSymbol))
            return domain;
    }
    return std::nullopt;
}

std::string multiplyBoundTerms(std::vector<std::string> terms) {
    std::vector<std::string> filtered;
    for (auto& term : terms) {
        if (term.empty() || term == "1") continue;
        filtered.push_back(std::move(term));
    }
    if (filtered.empty()) return "1";
    std::string out = "(" + filtered.front() + ")";
    for (size_t i = 1; i < filtered.size(); ++i)
        out += " * (" + filtered[i] + ")";
    return out;
}

std::optional<std::string> relationBoundedGroupOutputBound(
        const GenericRelPlan& ir,
        const GenericAggregateDetail& aggregate,
        const GenericPredicatePtr& pred) {
    std::set<int> relationIds;
    std::vector<std::string> terms;
    int64_t finiteProduct = 1;
    for (const auto& group : aggregate.groupBy) {
        if (auto card = finiteGroupExprCardinality(group, pred)) {
            if (*card <= 0) return std::nullopt;
            if (finiteProduct > std::numeric_limits<int64_t>::max() / *card)
                return std::nullopt;
            finiteProduct *= *card;
            continue;
        }

        std::vector<GenericColumnExpr> cols;
        collectExprColumns(group, cols);
        for (const auto& col : cols) {
            if (!col.relationInstance.valid()) return std::nullopt;
            relationIds.insert(col.relationInstance.value);
        }
    }

    if (finiteProduct != 1)
        terms.push_back(std::to_string(finiteProduct));
    for (int relationId : relationIds) {
        auto bound = relationInstanceRowBoundExpr(ir, relationId);
        if (!bound) return std::nullopt;
        terms.push_back(*bound);
    }
    return multiplyBoundTerms(std::move(terms));
}

std::string groupHashCapacityExpr(const std::string& inputSizeExpr,
                                  const std::string& outputBoundExpr) {
    if (outputBoundExpr == inputSizeExpr)
        return "next_pow2(" + inputSizeExpr + " * 2)";
    return "next_pow2(" + outputBoundExpr + " * 2 + 4096)";
}

struct PrimaryKeyGroupReduction {
    size_t groupIndex = 0;
    std::string outputBoundExpr;
    std::string keyDomainExpr;
    int relationInstance = -1;
    std::string table;
    std::string alias;
    std::string keyColumn;
};

std::optional<PrimaryKeyGroupReduction> primaryKeyGroupReduction(
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
        std::optional<size_t> pkGroupIndex;
        for (size_t gi = 0; gi < aggregate.groupBy.size(); ++gi) {
            auto* groupCol = aggregate.groupBy[gi]
                ? std::get_if<GenericColumnExpr>(&aggregate.groupBy[gi]->node)
                : nullptr;
            if (groupCol && eq.equivalent(*groupCol, pk)) {
                pkGroupIndex = gi;
                break;
            }
        }
        if (!pkGroupIndex) continue;

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
            PrimaryKeyGroupReduction out;
            out.groupIndex = *pkGroupIndex;
            out.outputBoundExpr = (!rel->virtualRelation && !inst.baseName.empty())
                ? tableSizeName(inst.baseName)
                : rel->primaryKeyDomainSymbol;
            out.keyDomainExpr = rel->primaryKeyDomainSymbol;
            out.relationInstance = inst.id.value;
            out.table = inst.baseName;
            out.alias = inst.alias;
            out.keyColumn = rel->primaryKeyColumn;
            return out;
        }
    }
    return std::nullopt;
}

std::optional<std::string> primaryKeyGroupOutputBound(
        const GenericRelPlan& ir,
        const GenericAggregateDetail& aggregate,
        const ColumnEquivalence& baseEq,
        const IrCarryMap& carryMap) {
    auto reduction = primaryKeyGroupReduction(ir, aggregate, baseEq, carryMap);
    if (!reduction) return std::nullopt;
    return reduction->outputBoundExpr;
}

class MetalMaterializedRangeScan : public MetalOperator {
public:
    using ExtraBuffer = std::pair<std::string, std::string>;

    MetalMaterializedRangeScan(std::string rowsSymbol,
                               std::string idxVar,
                               std::vector<GenericMatColumnDesc> columns,
                               std::vector<ExtraBuffer> extraBuffers = {})
        : rowsSymbol_(std::move(rowsSymbol)),
          idxVar_(std::move(idxVar)),
          columns_(std::move(columns)),
          extraBuffers_(std::move(extraBuffers)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string nParam = tableSizeName(rowsSymbol_);
        cg.setPhaseScannedTable(rowsSymbol_);
        cg.addResolvedScalarParam(nParam, "uint", rowsSymbol_);
        for (const auto& col : columns_) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
            if (col.stringRowRef && !col.stringSourceColumn.empty()) {
                cg.addColumnParam(col.stringSourceColumn, "char",
                                  col.stringSourceTable);
            }
        }
        for (const auto& extra : extraBuffers_) {
            cg.addBufferParam(extra.first, extra.second, "", false);
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
    std::vector<ExtraBuffer> extraBuffers_;
};

struct DomainAggReuseColumnSpec {
    GenericMatColumnDesc output;
    std::string sourceBuffer;
    std::string sourceMetalType;
    bool fromDomainKey = false;
    bool sourceIsStringRowRef = false;
    std::string stringSourceTable;
    std::string stringSourceColumn;
};

class MetalDomainAggReuseMaterialize : public MetalOperator {
public:
    MetalDomainAggReuseMaterialize(std::string bitmapName,
                                   std::string keyDomain,
                                   std::string aggBuffer,
                                   std::string keyListBuffer,
                                   std::string keyListCountBuffer,
                                   std::string outputCounter,
                                   std::vector<DomainAggReuseColumnSpec> columns,
                                   GenericMatColumnDesc aggregateOutput)
        : bitmapName_(std::move(bitmapName)),
          keyDomain_(std::move(keyDomain)),
          aggBuffer_(std::move(aggBuffer)),
          keyListBuffer_(std::move(keyListBuffer)),
          keyListCountBuffer_(std::move(keyListCountBuffer)),
          outputCounter_(std::move(outputCounter)),
          columns_(std::move(columns)),
          aggregateOutput_(std::move(aggregateOutput)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string domainParam =
            "n_domain_agg_reuse_" + sanitizeIdentifier(outputCounter_);
        cg.addResolvedScalarParam(domainParam, "uint", keyDomain_);
        cg.addBitmapReadParam(bitmapName_, "(" + keyDomain_ + " + 31) / 32");
        cg.addBufferParam(aggBuffer_, "atomic_uint", keyDomain_, false);
        const bool useKeyList =
            !keyListBuffer_.empty() && !keyListCountBuffer_.empty();
        if (useKeyList) {
            cg.addBufferParam(keyListBuffer_, "uint", keyDomain_, false);
            cg.addBufferParam(keyListCountBuffer_, "const atomic_uint", "1",
                              false);
        }
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");

        for (const auto& col : columns_) {
            if (!col.fromDomainKey) {
                cg.addBufferParam(col.sourceBuffer,
                                  col.sourceIsStringRowRef
                                      ? "uint"
                                      : col.sourceMetalType,
                                  keyDomain_, false);
            }
            if (col.sourceIsStringRowRef &&
                !col.stringSourceTable.empty() &&
                !col.stringSourceColumn.empty()) {
                cg.addColumnParam(col.stringSourceColumn, "char",
                                  col.stringSourceTable);
            }
        }

        auto addOutput = [&](const GenericMatColumnDesc& out) {
            std::string sizeExpr = keyDomain_;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
            cg.registerOutputColumn(out.displayName, out.bufferName,
                                    out.metalType, out.stringLen,
                                    out.scaleDown, out.isLongPair);
        };

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& col : columns_)
            addOutput(col.output);
        addOutput(aggregateOutput_);

        const std::string scanLimit = useKeyList
            ? "atomic_load_explicit(&" + keyListCountBuffer_ +
                  "[0], memory_order_relaxed)"
            : domainParam;
        cg.addBlock("for (uint _scan = tid; _scan < " + scanLimit +
                    "; _scan += tpg)", [&]() {
            if (useKeyList)
                cg.addLine("uint _key = " + keyListBuffer_ + "[_scan];");
            else
                cg.addLine("uint _key = _scan;");
            cg.addIf("bitmap_test_atomic(" + bitmapName_ + ", _key)", [&]() {
                cg.addLine("uint _pos = atomic_fetch_add_explicit(&" +
                           outputCounter_ +
                           "[0], 1u, memory_order_relaxed);");
                for (const auto& col : columns_)
                    emitColumnWrite(cg, col);
                cg.addLine(aggregateOutput_.bufferName +
                           "[_pos] = as_type<float>(atomic_load_explicit(&" +
                           aggBuffer_ + "[_key], memory_order_relaxed));");
                consume();
            });
        });
    }

    std::string describe() const override {
        return "DomainAggReuseMaterialize(" + aggBuffer_ + ")";
    }

private:
    std::string bitmapName_;
    std::string keyDomain_;
    std::string aggBuffer_;
    std::string keyListBuffer_;
    std::string keyListCountBuffer_;
    std::string outputCounter_;
    std::vector<DomainAggReuseColumnSpec> columns_;
    GenericMatColumnDesc aggregateOutput_;

    void emitColumnWrite(MetalCodegen& cg,
                         const DomainAggReuseColumnSpec& col) const {
        const auto& out = col.output;
        if (out.stringLen > 0 && col.sourceIsStringRowRef) {
            const int width = std::max(1, out.stringLen);
            cg.addLine("uint _src_row_" + sanitizeIdentifier(out.displayName) +
                       " = " + col.sourceBuffer + "[_key];");
            cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(width) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[_pos * " +
                           std::to_string(width) + "u + _oc] = " +
                           col.stringSourceColumn + "[_src_row_" +
                           sanitizeIdentifier(out.displayName) + " * " +
                           std::to_string(width) + "u + _oc];");
            });
            return;
        }
        if (col.fromDomainKey) {
            cg.addLine(out.bufferName + "[_pos] = (" + out.metalType +
                       ")_key;");
            return;
        }
        cg.addLine(out.bufferName + "[_pos] = " + col.sourceBuffer +
                   "[_key];");
    }
};

enum class DomainAggReusePkSourceKind {
    DomainKey,
    PkRowColumn,
    CarryByPkColumn
};

struct DomainAggReusePkColumnSpec {
    GenericMatColumnDesc output;
    DomainAggReusePkSourceKind kind = DomainAggReusePkSourceKind::DomainKey;
    std::string sourceTable;
    std::string sourceColumn;
    std::string sourceMetalType;
    std::string carryBuffer;
    std::string carryDomain;
    std::string carryKeyColumn;
    std::string carryKeyMetalType;
    bool sourceIsStringRowRef = false;
    std::string stringSourceTable;
    std::string stringSourceColumn;
};

class MetalDomainAggReusePkLookupMaterialize : public MetalOperator {
public:
    MetalDomainAggReusePkLookupMaterialize(
            std::string pkTable,
            std::string pkRowLookupBuffer,
            std::string keyDomain,
            std::string aggBuffer,
            std::string keyListBuffer,
            std::string keyListCountBuffer,
            std::string outputCounter,
            std::vector<DomainAggReusePkColumnSpec> columns,
            GenericMatColumnDesc aggregateOutput)
        : pkTable_(std::move(pkTable)),
          pkRowLookupBuffer_(std::move(pkRowLookupBuffer)),
          keyDomain_(std::move(keyDomain)),
          aggBuffer_(std::move(aggBuffer)),
          keyListBuffer_(std::move(keyListBuffer)),
          keyListCountBuffer_(std::move(keyListCountBuffer)),
          outputCounter_(std::move(outputCounter)),
          columns_(std::move(columns)),
          aggregateOutput_(std::move(aggregateOutput)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string domainParam =
            "n_domain_agg_pk_reuse_" + sanitizeIdentifier(outputCounter_);
        cg.addResolvedScalarParam(domainParam, "uint", keyDomain_);
        cg.addBufferParam(pkRowLookupBuffer_, "int", keyDomain_, false);
        cg.addBufferParam(aggBuffer_, "atomic_uint", keyDomain_, false);
        const bool useKeyList =
            !keyListBuffer_.empty() && !keyListCountBuffer_.empty();
        if (useKeyList) {
            cg.addBufferParam(keyListBuffer_, "uint", keyDomain_, false);
            cg.addBufferParam(keyListCountBuffer_, "const atomic_uint", "1",
                              false);
        }
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");

        for (const auto& col : columns_) {
            if (col.kind == DomainAggReusePkSourceKind::PkRowColumn) {
                cg.addColumnParam(col.sourceColumn,
                                  col.output.stringLen > 0 ? "char"
                                                           : col.sourceMetalType,
                                  pkTable_);
            } else if (col.kind ==
                       DomainAggReusePkSourceKind::CarryByPkColumn) {
                cg.addColumnParam(col.carryKeyColumn, col.carryKeyMetalType,
                                  pkTable_);
                cg.addBufferParam(col.carryBuffer,
                                  col.sourceIsStringRowRef
                                      ? "uint"
                                      : col.sourceMetalType,
                                  col.carryDomain, false);
                if (col.sourceIsStringRowRef &&
                    !col.stringSourceTable.empty() &&
                    !col.stringSourceColumn.empty()) {
                    cg.addColumnParam(col.stringSourceColumn, "char",
                                      col.stringSourceTable);
                }
            }
        }

        auto addOutput = [&](const GenericMatColumnDesc& out) {
            std::string sizeExpr = keyDomain_;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
            cg.registerOutputColumn(out.displayName, out.bufferName,
                                    out.metalType, out.stringLen,
                                    out.scaleDown, out.isLongPair);
        };

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& col : columns_)
            addOutput(col.output);
        addOutput(aggregateOutput_);

        const std::string scanLimit = useKeyList
            ? "atomic_load_explicit(&" + keyListCountBuffer_ +
                  "[0], memory_order_relaxed)"
            : domainParam;
        cg.addBlock("for (uint _scan = tid; _scan < " + scanLimit +
                    "; _scan += tpg)", [&]() {
            if (useKeyList)
                cg.addLine("uint _key = " + keyListBuffer_ + "[_scan];");
            else
                cg.addLine("uint _key = _scan;");
            cg.addLine("int _pk_row_i = " + pkRowLookupBuffer_ + "[_key];");
            cg.addIf("_pk_row_i >= 0", [&]() {
                cg.addLine("uint _pk_row = (uint)_pk_row_i;");
                cg.addLine("bool _row_ok = true;");
                for (size_t ci = 0; ci < columns_.size(); ++ci)
                    emitGuard(cg, columns_[ci], ci);
                cg.addIf("_row_ok", [&]() {
                    cg.addLine("uint _pos = atomic_fetch_add_explicit(&" +
                               outputCounter_ +
                               "[0], 1u, memory_order_relaxed);");
                    for (size_t ci = 0; ci < columns_.size(); ++ci)
                        emitColumnWrite(cg, columns_[ci], ci);
                    cg.addLine(
                        aggregateOutput_.bufferName +
                        "[_pos] = as_type<float>(atomic_load_explicit(&" +
                        aggBuffer_ + "[_key], memory_order_relaxed));");
                    consume();
                });
            });
        });
    }

    std::string describe() const override {
        return "DomainAggReusePkLookupMaterialize(" + aggBuffer_ + ")";
    }

private:
    std::string pkTable_;
    std::string pkRowLookupBuffer_;
    std::string keyDomain_;
    std::string aggBuffer_;
    std::string keyListBuffer_;
    std::string keyListCountBuffer_;
    std::string outputCounter_;
    std::vector<DomainAggReusePkColumnSpec> columns_;
    GenericMatColumnDesc aggregateOutput_;

    static std::string suffixForColumn(const DomainAggReusePkColumnSpec& col,
                                       size_t idx) {
        return std::to_string(idx) + "_" +
               sanitizeIdentifier(col.output.displayName);
    }

    static std::string carryKeyExpr(const DomainAggReusePkColumnSpec& col) {
        return "(uint)(" + col.carryKeyColumn + "[_pk_row])";
    }

    static bool hasReliableCarrySentinel(
            const DomainAggReusePkColumnSpec& col) {
        if (col.sourceIsStringRowRef) return true;
        return col.sourceMetalType == "int" || col.sourceMetalType == "uint";
    }

    static std::string sentinelCondition(
            const DomainAggReusePkColumnSpec& col,
            const std::string& valueExpr) {
        if (col.sourceIsStringRowRef || col.sourceMetalType == "uint")
            return valueExpr + " != 0xFFFFFFFFu";
        return valueExpr + " != (" + col.sourceMetalType + ")-1";
    }

    void emitGuard(MetalCodegen& cg,
                   const DomainAggReusePkColumnSpec& col,
                   size_t idx) const {
        if (col.kind != DomainAggReusePkSourceKind::CarryByPkColumn ||
            !hasReliableCarrySentinel(col)) {
            return;
        }
        const std::string suffix = suffixForColumn(col, idx);
        const std::string valueType =
            col.sourceIsStringRowRef ? "uint" : col.sourceMetalType;
        cg.addLine("uint _carry_key_" + suffix + " = " + carryKeyExpr(col) +
                   ";");
        cg.addLine(valueType + " _carry_guard_" + suffix + " = " +
                   col.carryBuffer + "[_carry_key_" + suffix + "];");
        cg.addLine("if (!(" +
                   sentinelCondition(col, "_carry_guard_" + suffix) +
                   ")) _row_ok = false;");
    }

    void emitColumnWrite(MetalCodegen& cg,
                         const DomainAggReusePkColumnSpec& col,
                         size_t idx) const {
        const auto& out = col.output;
        if (col.kind == DomainAggReusePkSourceKind::DomainKey) {
            cg.addLine(out.bufferName + "[_pos] = (" + out.metalType +
                       ")_key;");
            return;
        }

        if (col.kind == DomainAggReusePkSourceKind::PkRowColumn) {
            if (out.stringLen > 0) {
                const int width = std::max(1, out.stringLen);
                cg.addBlock("for (uint _oc = 0; _oc < " +
                            std::to_string(width) + "u; ++_oc)", [&]() {
                    cg.addLine(out.bufferName + "[_pos * " +
                               std::to_string(width) + "u + _oc] = " +
                               col.sourceColumn + "[_pk_row * " +
                               std::to_string(width) + "u + _oc];");
                });
            } else {
                cg.addLine(out.bufferName + "[_pos] = " + col.sourceColumn +
                           "[_pk_row];");
            }
            return;
        }

        const std::string suffix = suffixForColumn(col, idx);
        const std::string carryKey = "_carry_key_" + suffix;
        if (!hasReliableCarrySentinel(col))
            cg.addLine("uint " + carryKey + " = " + carryKeyExpr(col) + ";");
        if (out.stringLen > 0 && col.sourceIsStringRowRef) {
            const int width = std::max(1, out.stringLen);
            const std::string srcRow = "_src_row_" + suffix;
            cg.addLine("uint " + srcRow + " = " + col.carryBuffer + "[" +
                       carryKey + "];");
            cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(width) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[_pos * " +
                           std::to_string(width) + "u + _oc] = " +
                           col.stringSourceColumn + "[" + srcRow + " * " +
                           std::to_string(width) + "u + _oc];");
            });
            return;
        }

        cg.addLine(out.bufferName + "[_pos] = " + col.carryBuffer + "[" +
                   carryKey + "];");
    }
};

class MetalFiniteStringRowRefMapBuild : public MetalOperator {
public:
    MetalFiniteStringRowRefMapBuild(std::string sourceTable,
                                    std::string sourceColumn,
                                    int width,
                                    std::string mapBuffer,
                                    std::vector<std::string> domain)
        : sourceTable_(std::move(sourceTable)),
          sourceColumn_(std::move(sourceColumn)),
          width_(width),
          mapBuffer_(std::move(mapBuffer)),
          domain_(std::move(domain)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        if (sourceTable_.empty() || sourceColumn_.empty() || width_ <= 0 ||
            domain_.empty()) {
            return;
        }

        const std::string nRows = tableSizeName(sourceTable_);
        cg.setPhaseScannedTable(sourceTable_);
        cg.addTableSizeParam(sourceTable_);
        cg.addColumnParam(sourceColumn_, "char", sourceTable_);
        cg.addBufferParam(mapBuffer_, "uint", nRows, true);

        cg.addBlock("for (uint i = tid; i < " + nRows + "; i += tpg)", [&]() {
            if (domain_.size() == 1) {
                cg.addLine(mapBuffer_ + "[i] = 0u;");
                return;
            }

            const std::string ptr = sourceColumn_ + " + i * " +
                std::to_string(width_) + "u";
            std::string expr = "0u";
            for (int di = static_cast<int>(domain_.size()) - 1; di >= 0; --di) {
                expr = "(" + fixedStringEqMetalFromPointer(
                    ptr, width_, domain_[(size_t)di]) + " ? " +
                    std::to_string(di) + "u : " + expr + ")";
            }
            cg.addLine(mapBuffer_ + "[i] = " + expr + ";");
        });
    }

    std::string describe() const override {
        return "FiniteStringRowRefMapBuild(" + sourceTable_ + "." +
               sourceColumn_ + ")";
    }

private:
    std::string sourceTable_;
    std::string sourceColumn_;
    int width_ = 0;
    std::string mapBuffer_;
    std::vector<std::string> domain_;
};

class MetalFdKeyedGroupBuild : public MetalOperator {
public:
    MetalFdKeyedGroupBuild(std::string rowsSymbol,
                           std::string bucketCountExpr,
                           std::string bucketCountSymbol,
                           std::string stateBuffer,
                           std::string repRowBuffer,
                           std::string aggBuffer,
                           GenericMatColumnDesc keyColumn,
                           std::vector<GenericMatColumnDesc> inputColumns,
                           std::vector<IrPendingAgg> pending,
                           int valuesPerBucket)
        : rowsSymbol_(std::move(rowsSymbol)),
          bucketCountExpr_(std::move(bucketCountExpr)),
          bucketCountSymbol_(std::move(bucketCountSymbol)),
          stateBuffer_(std::move(stateBuffer)),
          repRowBuffer_(std::move(repRowBuffer)),
          aggBuffer_(std::move(aggBuffer)),
          keyColumn_(std::move(keyColumn)),
          inputColumns_(std::move(inputColumns)),
          pending_(std::move(pending)),
          valuesPerBucket_(valuesPerBucket) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        const std::string nRows = tableSizeName(rowsSymbol_);
        cg.setPhaseScannedTable(rowsSymbol_);
        cg.addResolvedScalarParam(nRows, "uint", rowsSymbol_);
        cg.addResolvedScalarParam(bucketCountSymbol_, "uint", bucketCountExpr_);
        for (const auto& col : inputColumns_) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
        }
        cg.addAtomicBufferParam(stateBuffer_, "atomic_uint", bucketCountExpr_);
        cg.addBufferParam(repRowBuffer_, "uint", bucketCountExpr_, false);
        cg.addAtomicBufferParam(aggBuffer_, "atomic_uint",
                                bucketCountExpr_ + " * " +
                                    std::to_string(valuesPerBucket_));

        cg.addBlock("for (uint _r = tid; _r < " + nRows + "; _r += tpg)", [&]() {
            cg.addLine("uint _bucket = (uint)(" + materializedValueAt(keyColumn_, "_r") + ");");
            cg.addIf("_bucket < " + bucketCountSymbol_, [&]() {
                cg.addLine("uint _was_seen = atomic_exchange_explicit(&" +
                           stateBuffer_ + "[_bucket], 1u, memory_order_relaxed);");
                cg.addIf("_was_seen == 0u", [&]() {
                    cg.addLine(repRowBuffer_ + "[_bucket] = _r;");
                });
                emitAggregateUpdates(cg, "_bucket");
            });
            consume();
        });
    }

    std::string describe() const override {
        return "FdKeyedGroupBuild(" + keyColumn_.displayName + ")";
    }

private:
    std::string rowsSymbol_;
    std::string bucketCountExpr_;
    std::string bucketCountSymbol_;
    std::string stateBuffer_;
    std::string repRowBuffer_;
    std::string aggBuffer_;
    GenericMatColumnDesc keyColumn_;
    std::vector<GenericMatColumnDesc> inputColumns_;
    std::vector<IrPendingAgg> pending_;
    int valuesPerBucket_ = 0;

    void emitAggregateUpdates(MetalCodegen& cg,
                              const std::string& bucket) const {
        const std::string base = bucket + " * " +
            std::to_string(valuesPerBucket_) + "u";
        for (const auto& agg : pending_) {
            const std::string slot = base + " + " + std::to_string(agg.offset) + "u";
            if (agg.isFloatSum) {
                cg.addLine("atomic_add_float(&" + aggBuffer_ + "[" + slot +
                           "], (float)(" + agg.valueExpr + "));");
            } else if (agg.isLongPair) {
                const std::string hiSlot = base + " + " +
                    std::to_string(agg.offset + 1) + "u";
                cg.addLine("atomic_add_long_pair(&" + aggBuffer_ + "[" + slot +
                           "], &" + aggBuffer_ + "[" + hiSlot +
                           "], (long)(" + agg.valueExpr + "));");
            } else {
                cg.addLine("atomic_fetch_add_explicit(&" + aggBuffer_ + "[" +
                           slot + "], (uint)(" + agg.valueExpr +
                           "), memory_order_relaxed);");
            }
        }
    }
};

class MetalFdKeyedGroupCompact : public MetalOperator {
public:
    MetalFdKeyedGroupCompact(std::string bucketCountExpr,
                             std::string bucketCountSymbol,
                             std::string outputCapacityExpr,
                             std::string stateBuffer,
                             std::string repRowBuffer,
                             std::string aggBuffer,
                             std::string outputCounter,
                             std::vector<GenericMatColumnDesc> inputColumns,
                             std::vector<std::string> groupColumns,
                             std::vector<IrPendingAgg> pending,
                             std::vector<GenericMatColumnDesc> outputs,
                             int valuesPerBucket)
        : bucketCountExpr_(std::move(bucketCountExpr)),
          bucketCountSymbol_(std::move(bucketCountSymbol)),
          outputCapacityExpr_(std::move(outputCapacityExpr)),
          stateBuffer_(std::move(stateBuffer)),
          repRowBuffer_(std::move(repRowBuffer)),
          aggBuffer_(std::move(aggBuffer)),
          outputCounter_(std::move(outputCounter)),
          inputColumns_(std::move(inputColumns)),
          groupColumns_(std::move(groupColumns)),
          pending_(std::move(pending)),
          outputs_(std::move(outputs)),
          valuesPerBucket_(valuesPerBucket) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addResolvedScalarParam(bucketCountSymbol_, "uint", bucketCountExpr_);
        cg.addBufferParam(stateBuffer_, "atomic_uint", bucketCountExpr_, false);
        cg.addBufferParam(repRowBuffer_, "uint", bucketCountExpr_, false);
        cg.addBufferParam(aggBuffer_, "atomic_uint",
                          bucketCountExpr_ + " * " +
                              std::to_string(valuesPerBucket_),
                          false);
        for (const auto& col : inputColumns_) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
            if (col.stringRowRef && !col.stringSourceColumn.empty()) {
                cg.addColumnParam(col.stringSourceColumn, "char",
                                  col.stringSourceTable);
            }
        }
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");
        for (const auto& out : outputs_) {
            std::string sizeExpr = outputCapacityExpr_.empty()
                ? bucketCountExpr_
                : outputCapacityExpr_;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
        }

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& out : outputs_) {
            cg.registerOutputColumn(out.displayName, out.bufferName, out.metalType,
                                    out.stringLen, out.scaleDown, out.isLongPair);
        }

        cg.addBlock("for (uint _bucket = tid; _bucket < " + bucketCountSymbol_ +
                    "; _bucket += tpg)", [&]() {
            cg.addIf("atomic_load_explicit(&" + stateBuffer_ +
                     "[_bucket], memory_order_relaxed) != 0u", [&]() {
                cg.addLine("uint _rep = " + repRowBuffer_ + "[_bucket];");
                cg.addLine("uint _pos = atomic_fetch_add_explicit(&" +
                           outputCounter_ + "[0], 1u, memory_order_relaxed);");
                emitOutputWrites(cg, "_bucket", "_rep", "_pos");
            });
            consume();
        });
    }

    std::string describe() const override { return "FdKeyedGroupCompact"; }

private:
    std::string bucketCountExpr_;
    std::string bucketCountSymbol_;
    std::string outputCapacityExpr_;
    std::string stateBuffer_;
    std::string repRowBuffer_;
    std::string aggBuffer_;
    std::string outputCounter_;
    std::vector<GenericMatColumnDesc> inputColumns_;
    std::vector<std::string> groupColumns_;
    std::vector<IrPendingAgg> pending_;
    std::vector<GenericMatColumnDesc> outputs_;
    int valuesPerBucket_ = 0;

    bool isGroupColumn(const std::string& display) const {
        return std::find(groupColumns_.begin(), groupColumns_.end(), display) !=
               groupColumns_.end();
    }

    const IrPendingAgg* pendingForDisplay(const std::string& display,
                                          size_t* index = nullptr) const {
        for (size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i].displayName == display) {
                if (index) *index = i;
                return &pending_[i];
            }
        }
        return nullptr;
    }

    std::string longPairAsFloatExpr(int offset,
                                    const std::string& bucket) const {
        const std::string base = bucket + " * " +
            std::to_string(valuesPerBucket_) + "u";
        return "((float)atomic_load_explicit(&" + aggBuffer_ + "[" + base +
               " + " + std::to_string(offset + 1) +
               "u], memory_order_relaxed) * 4294967296.0f + "
               "(float)atomic_load_explicit(&" + aggBuffer_ + "[" + base +
               " + " + std::to_string(offset) +
               "u], memory_order_relaxed))";
    }

    std::string aggSlotExpr(int offset,
                            const std::string& bucket) const {
        return bucket + " * " + std::to_string(valuesPerBucket_) +
               "u + " + std::to_string(offset) + "u";
    }

    std::string aggregateValueExpr(const IrPendingAgg& agg,
                                   size_t pendingIndex,
                                   const std::string& bucket) const {
        if (agg.scaleDown < 0 && pendingIndex + 1 < pending_.size()) {
            std::string sum;
            if (agg.isLongPair) {
                sum = longPairAsFloatExpr(agg.offset, bucket);
                if (agg.scaleDown < -1) {
                    sum = "(" + sum + " / " +
                          std::to_string(-agg.scaleDown) + ".0f)";
                }
            } else {
                sum = "as_type<float>(atomic_load_explicit(&" + aggBuffer_ +
                      "[" + aggSlotExpr(agg.offset, bucket) +
                      "], memory_order_relaxed))";
            }
            const auto& cnt = pending_[pendingIndex + 1];
            std::string count = "atomic_load_explicit(&" + aggBuffer_ +
                "[" + aggSlotExpr(cnt.offset, bucket) +
                "], memory_order_relaxed)";
            return "((" + count + ") != 0u ? (" + sum + ") / (float)(" +
                   count + ") : 0.0f)";
        }
        if (agg.isFloatSum || agg.isMinMax) {
            return "as_type<float>(atomic_load_explicit(&" + aggBuffer_ +
                   "[" + aggSlotExpr(agg.offset, bucket) +
                   "], memory_order_relaxed))";
        }
        return "atomic_load_explicit(&" + aggBuffer_ + "[" +
               aggSlotExpr(agg.offset, bucket) + "], memory_order_relaxed)";
    }

    std::string stringByteAt(const GenericMatColumnDesc& col,
                             const std::string& rep,
                             const std::string& offset) const {
        if (col.stringRowRef) {
            return col.stringSourceColumn + "[" + col.bufferName + "[" + rep +
                   "] * " + std::to_string(col.stringLen) + "ul + " +
                   offset + "]";
        }
        return col.bufferName + "[" + rep + " * " +
               std::to_string(col.stringLen) + "ul + " + offset + "]";
    }

    void emitGroupOutput(MetalCodegen& cg,
                         const GenericMatColumnDesc& out,
                         const GenericMatColumnDesc& col,
                         const std::string& rep,
                         const std::string& pos) const {
        if (col.stringLen > 0) {
                cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(col.stringLen) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[(ulong)(" + pos + ") * " +
                           std::to_string(col.stringLen) + "ul + _oc] = " +
                           stringByteAt(col, rep, "_oc") + ";");
            });
        } else {
            cg.addLine(out.bufferName + "[" + pos + "] = " +
                       col.bufferName + "[" + rep + "];");
        }
    }

    void emitOutputWrites(MetalCodegen& cg,
                          const std::string& bucket,
                          const std::string& rep,
                          const std::string& pos) const {
        for (const auto& out : outputs_) {
            if (out.displayName == kFdHiddenBucketDisplay) {
                cg.addLine(out.bufferName + "[" + pos + "] = (uint)(" +
                           bucket + ");");
                continue;
            }
            if (isGroupColumn(out.displayName)) {
                if (const auto* col =
                        findMaterializedColumn(inputColumns_, out.displayName)) {
                    emitGroupOutput(cg, out, *col, rep, pos);
                }
                continue;
            }

            size_t pendingIndex = 0;
            const auto* agg = pendingForDisplay(out.displayName, &pendingIndex);
            if (!agg) continue;
            if (out.isLongPair) {
                cg.addLine(out.bufferName + "[" + pos +
                           " * 2u] = atomic_load_explicit(&" + aggBuffer_ +
                           "[" + aggSlotExpr(agg->offset, bucket) +
                           "], memory_order_relaxed);");
                cg.addLine(out.bufferName + "[" + pos +
                           " * 2u + 1u] = atomic_load_explicit(&" +
                           aggBuffer_ + "[" + aggSlotExpr(agg->offset + 1, bucket) +
                           "], memory_order_relaxed);");
            } else {
                cg.addLine(out.bufferName + "[" + pos + "] = " +
                           aggregateValueExpr(*agg, pendingIndex, bucket) + ";");
            }
        }
    }
};

class MetalFdKeyedGroupTopKGather : public MetalOperator {
public:
    MetalFdKeyedGroupTopKGather(std::string sortedIndexBuffer,
                                std::string sortedRowsSymbol,
                                int limit,
                                GenericMatColumnDesc compactBucketColumn,
                                std::string bucketCountExpr,
                                std::string repRowBuffer,
                                std::string aggBuffer,
                                std::string outputCounter,
                                std::vector<GenericMatColumnDesc> inputColumns,
                                std::vector<std::string> groupColumns,
                                std::vector<IrPendingAgg> pending,
                                std::vector<GenericMatColumnDesc> outputs,
                                int valuesPerBucket)
        : sortedIndexBuffer_(std::move(sortedIndexBuffer)),
          sortedRowsSymbol_(std::move(sortedRowsSymbol)),
          limit_(limit),
          compactBucketColumn_(std::move(compactBucketColumn)),
          bucketCountExpr_(std::move(bucketCountExpr)),
          repRowBuffer_(std::move(repRowBuffer)),
          aggBuffer_(std::move(aggBuffer)),
          outputCounter_(std::move(outputCounter)),
          inputColumns_(std::move(inputColumns)),
          groupColumns_(std::move(groupColumns)),
          pending_(std::move(pending)),
          outputs_(std::move(outputs)),
          valuesPerBucket_(valuesPerBucket) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.setPhaseMaxThreadgroups(1);
        cg.addScalarParam(sortedRowsSymbol_, "uint");
        cg.addBufferParam(sortedIndexBuffer_, "int", "", false);
        cg.addBufferParam(compactBucketColumn_.bufferName,
                          compactBucketColumn_.metalType, "", false);
        cg.addBufferParam(repRowBuffer_, "uint", bucketCountExpr_, false);
        cg.addBufferParam(aggBuffer_, "atomic_uint",
                          bucketCountExpr_ + " * " +
                              std::to_string(valuesPerBucket_),
                          false);
        for (const auto& col : inputColumns_) {
            cg.addBufferParam(col.bufferName, col.metalType, "", false);
            if (col.stringRowRef && !col.stringSourceColumn.empty()) {
                cg.addColumnParam(col.stringSourceColumn, "char",
                                  col.stringSourceTable);
            }
        }

        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");
        const std::string outCapacity = std::to_string(limit_);
        for (const auto& out : outputs_) {
            std::string sizeExpr = outCapacity;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            if (out.isLongPair)
                sizeExpr += " * 2";
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
        }

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& out : outputs_) {
            cg.registerOutputColumn(out.displayName, out.bufferName,
                                    out.metalType, out.stringLen,
                                    out.scaleDown, out.isLongPair);
        }

        cg.addLine("uint _fd_limit = " + std::to_string(limit_) + "u;");
        cg.addLine("uint _fd_out_count = " + sortedRowsSymbol_ +
                   " < _fd_limit ? " + sortedRowsSymbol_ + " : _fd_limit;");
        cg.addIf("tid == 0", [&]() {
            cg.addLine("atomic_store_explicit(&" + outputCounter_ +
                       "[0], _fd_out_count, memory_order_relaxed);");
        });
        cg.addBlock("for (uint _rank = tid; _rank < _fd_out_count; _rank += tpg)",
                    [&]() {
            cg.addLine("int _compact_pos_i = " + sortedIndexBuffer_ + "[_rank];");
            cg.addLine("uint _compact_pos = (_compact_pos_i >= 0 ? "
                       "(uint)_compact_pos_i : _rank);");
            cg.addIf("_compact_pos >= " + sortedRowsSymbol_, [&]() {
                cg.addLine("_compact_pos = _rank;");
            });
            cg.addLine("uint _bucket = " + compactBucketColumn_.bufferName +
                       "[_compact_pos];");
            cg.addLine("uint _rep = " + repRowBuffer_ + "[_bucket];");
            emitOutputWrites(cg, "_bucket", "_rep", "_rank");
            consume();
        });
    }

    std::string describe() const override {
        return "FdKeyedGroupTopKGather";
    }

private:
    std::string sortedIndexBuffer_;
    std::string sortedRowsSymbol_;
    int limit_ = 0;
    GenericMatColumnDesc compactBucketColumn_;
    std::string bucketCountExpr_;
    std::string repRowBuffer_;
    std::string aggBuffer_;
    std::string outputCounter_;
    std::vector<GenericMatColumnDesc> inputColumns_;
    std::vector<std::string> groupColumns_;
    std::vector<IrPendingAgg> pending_;
    std::vector<GenericMatColumnDesc> outputs_;
    int valuesPerBucket_ = 0;

    bool isGroupColumn(const std::string& display) const {
        return std::find(groupColumns_.begin(), groupColumns_.end(), display) !=
               groupColumns_.end();
    }

    const IrPendingAgg* pendingForDisplay(const std::string& display,
                                          size_t* index = nullptr) const {
        for (size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i].displayName == display) {
                if (index) *index = i;
                return &pending_[i];
            }
        }
        return nullptr;
    }

    std::string longPairAsFloatExpr(int offset,
                                    const std::string& bucket) const {
        const std::string base = bucket + " * " +
            std::to_string(valuesPerBucket_) + "u";
        return "((float)atomic_load_explicit(&" + aggBuffer_ + "[" + base +
               " + " + std::to_string(offset + 1) +
               "u], memory_order_relaxed) * 4294967296.0f + "
               "(float)atomic_load_explicit(&" + aggBuffer_ + "[" + base +
               " + " + std::to_string(offset) +
               "u], memory_order_relaxed))";
    }

    std::string aggSlotExpr(int offset,
                            const std::string& bucket) const {
        return bucket + " * " + std::to_string(valuesPerBucket_) +
               "u + " + std::to_string(offset) + "u";
    }

    std::string aggregateValueExpr(const IrPendingAgg& agg,
                                   size_t pendingIndex,
                                   const std::string& bucket) const {
        if (agg.scaleDown < 0 && pendingIndex + 1 < pending_.size()) {
            std::string sum;
            if (agg.isLongPair) {
                sum = longPairAsFloatExpr(agg.offset, bucket);
                if (agg.scaleDown < -1) {
                    sum = "(" + sum + " / " +
                          std::to_string(-agg.scaleDown) + ".0f)";
                }
            } else {
                sum = "as_type<float>(atomic_load_explicit(&" + aggBuffer_ +
                      "[" + aggSlotExpr(agg.offset, bucket) +
                      "], memory_order_relaxed))";
            }
            const auto& cnt = pending_[pendingIndex + 1];
            std::string count = "atomic_load_explicit(&" + aggBuffer_ +
                "[" + aggSlotExpr(cnt.offset, bucket) +
                "], memory_order_relaxed)";
            return "((" + count + ") != 0u ? (" + sum + ") / (float)(" +
                   count + ") : 0.0f)";
        }
        if (agg.isFloatSum || agg.isMinMax) {
            return "as_type<float>(atomic_load_explicit(&" + aggBuffer_ +
                   "[" + aggSlotExpr(agg.offset, bucket) +
                   "], memory_order_relaxed))";
        }
        return "atomic_load_explicit(&" + aggBuffer_ + "[" +
               aggSlotExpr(agg.offset, bucket) + "], memory_order_relaxed)";
    }

    std::string stringByteAt(const GenericMatColumnDesc& col,
                             const std::string& rep,
                             const std::string& offset) const {
        if (col.stringRowRef) {
            return col.stringSourceColumn + "[" + col.bufferName + "[" + rep +
                   "] * " + std::to_string(col.stringLen) + "ul + " +
                   offset + "]";
        }
        return col.bufferName + "[" + rep + " * " +
               std::to_string(col.stringLen) + "ul + " + offset + "]";
    }

    void emitGroupOutput(MetalCodegen& cg,
                         const GenericMatColumnDesc& out,
                         const GenericMatColumnDesc& col,
                         const std::string& rep,
                         const std::string& pos) const {
        if (col.stringLen > 0) {
            cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(col.stringLen) + "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[(ulong)(" + pos + ") * " +
                           std::to_string(col.stringLen) + "ul + _oc] = " +
                           stringByteAt(col, rep, "_oc") + ";");
            });
        } else {
            cg.addLine(out.bufferName + "[" + pos + "] = " +
                       col.bufferName + "[" + rep + "];");
        }
    }

    void emitOutputWrites(MetalCodegen& cg,
                          const std::string& bucket,
                          const std::string& rep,
                          const std::string& pos) const {
        for (const auto& out : outputs_) {
            if (isGroupColumn(out.displayName)) {
                if (const auto* col =
                        findMaterializedColumn(inputColumns_, out.displayName)) {
                    emitGroupOutput(cg, out, *col, rep, pos);
                }
                continue;
            }

            size_t pendingIndex = 0;
            const auto* agg = pendingForDisplay(out.displayName, &pendingIndex);
            if (!agg) continue;
            if (out.isLongPair) {
                cg.addLine(out.bufferName + "[" + pos +
                           " * 2u] = atomic_load_explicit(&" + aggBuffer_ +
                           "[" + aggSlotExpr(agg->offset, bucket) +
                           "], memory_order_relaxed);");
                cg.addLine(out.bufferName + "[" + pos +
                           " * 2u + 1u] = atomic_load_explicit(&" +
                           aggBuffer_ + "[" + aggSlotExpr(agg->offset + 1, bucket) +
                           "], memory_order_relaxed);");
            } else {
                cg.addLine(out.bufferName + "[" + pos + "] = " +
                           aggregateValueExpr(*agg, pendingIndex, bucket) + ";");
            }
        }
    }
};

std::string materializedStringPtrExpr(const GenericMatColumnDesc& col,
                                      const std::string& row) {
    const int width = std::max(1, col.stringLen);
    if (col.stringRowRef) {
        return "(" + col.stringSourceColumn + " + " + col.bufferName + "[" +
               row + "] * " + std::to_string(width) + "ul)";
    }
    return "(" + col.bufferName + " + " + row + " * " +
           std::to_string(width) + "ul)";
}

void bindMaterializedColumns(MetalCodegen& cg,
                             const std::vector<GenericMatColumnDesc>& columns) {
    for (const auto& col : columns) {
        cg.addBufferParam(col.bufferName, col.metalType, "", false);
        if (col.stringRowRef && !col.stringSourceColumn.empty()) {
            cg.addColumnParam(col.stringSourceColumn, "char",
                              col.stringSourceTable);
        }
    }
}

class MetalCountDistinctGroupBuild : public MetalOperator {
public:
    MetalCountDistinctGroupBuild(std::string rowsSymbol,
                                 std::string capacityExpr,
                                 std::string capacitySymbol,
                                 std::string stateBuffer,
                                 std::string hashBuffer,
                                 std::string slotGroupBuffer,
                                 std::string slotRepRowBuffer,
                                 std::string groupRepRowBuffer,
                                 std::string rowGroupBuffer,
                                 std::string rowGroupSizeExpr,
                                 std::string groupCounter,
                                 std::vector<GenericMatColumnDesc> groupKeys)
        : rowsSymbol_(std::move(rowsSymbol)),
          capacityExpr_(std::move(capacityExpr)),
          capacitySymbol_(std::move(capacitySymbol)),
          stateBuffer_(std::move(stateBuffer)),
          hashBuffer_(std::move(hashBuffer)),
          slotGroupBuffer_(std::move(slotGroupBuffer)),
          slotRepRowBuffer_(std::move(slotRepRowBuffer)),
          groupRepRowBuffer_(std::move(groupRepRowBuffer)),
          rowGroupBuffer_(std::move(rowGroupBuffer)),
          rowGroupSizeExpr_(std::move(rowGroupSizeExpr)),
          groupCounter_(std::move(groupCounter)),
          groupKeys_(std::move(groupKeys)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string nRows = tableSizeName(rowsSymbol_);
        cg.setPhaseScannedTable(rowsSymbol_);
        cg.addResolvedScalarParam(nRows, "uint", rowsSymbol_);
        cg.addResolvedScalarParam(capacitySymbol_, "uint", capacityExpr_);
        bindMaterializedColumns(cg, groupKeys_);
        cg.addAtomicBufferParam(stateBuffer_, "atomic_uint", capacityExpr_);
        cg.addAtomicBufferParam(hashBuffer_, "atomic_uint", capacityExpr_);
        cg.addAtomicBufferParam(slotGroupBuffer_, "atomic_uint", capacityExpr_);
        cg.addAtomicBufferParam(slotRepRowBuffer_, "atomic_uint", capacityExpr_);
        cg.addBufferParam(groupRepRowBuffer_, "uint", capacityExpr_, false);
        cg.addBufferParam(rowGroupBuffer_, "uint", rowGroupSizeExpr_, false);
        cg.addAtomicBufferParam(groupCounter_, "atomic_uint", "1");

        cg.addBlock("for (uint _r = tid; _r < " + nRows + "; _r += tpg)", [&]() {
            emitHash(cg, "_cd_hash", "_r");
            cg.addLine("uint _cd_slot = _cd_hash % " + capacitySymbol_ + ";");
            cg.addBlock("while (true)", [&]() {
                cg.addLine("uint _cd_state = atomic_load_explicit(&" +
                           stateBuffer_ + "[_cd_slot], memory_order_relaxed);");
                cg.addLine("if (_cd_state == 0u) {");
                cg.increaseIndent();
                cg.addLine("uint _cd_expected = 0u;");
                cg.addLine("if (atomic_compare_exchange_weak_explicit(&" +
                           stateBuffer_ + "[_cd_slot], &_cd_expected, 1u, "
                           "memory_order_relaxed, memory_order_relaxed)) {");
                cg.increaseIndent();
                cg.addLine("uint _cd_gid = atomic_fetch_add_explicit(&" +
                           groupCounter_ + "[0], 1u, memory_order_relaxed);");
                cg.addLine("atomic_store_explicit(&" + hashBuffer_ +
                           "[_cd_slot], _cd_hash, memory_order_relaxed);");
                cg.addLine("atomic_store_explicit(&" + slotRepRowBuffer_ +
                           "[_cd_slot], _r, memory_order_relaxed);");
                cg.addLine("atomic_store_explicit(&" + slotGroupBuffer_ +
                           "[_cd_slot], _cd_gid, memory_order_relaxed);");
                cg.addLine(groupRepRowBuffer_ + "[_cd_gid] = _r;");
                cg.addLine(rowGroupBuffer_ + "[_r] = _cd_gid;");
                cg.addLine("atomic_store_explicit(&" + stateBuffer_ +
                           "[_cd_slot], 2u, memory_order_relaxed);");
                cg.addLine("break;");
                cg.decreaseIndent();
                cg.addLine("}");
                cg.decreaseIndent();
                cg.addLine("} else if (_cd_state == 2u) {");
                cg.increaseIndent();
                cg.addLine("bool _cd_same = false;");
                cg.addBlock("for (uint _cd_retry = 0u; _cd_retry < 32u && !_cd_same; ++_cd_retry)", [&]() {
                    cg.addLine("uint _cd_slot_hash = atomic_load_explicit(&" +
                               hashBuffer_ + "[_cd_slot], memory_order_relaxed);");
                    cg.addLine("uint _cd_rep = atomic_load_explicit(&" +
                               slotRepRowBuffer_ + "[_cd_slot], memory_order_relaxed);");
                    cg.addLine("_cd_same = (_cd_slot_hash == _cd_hash);");
                    emitKeyEquals(cg, "_cd_same", "_r", "_cd_rep");
                });
                cg.addLine("if (_cd_same) {");
                cg.increaseIndent();
                cg.addLine(rowGroupBuffer_ + "[_r] = atomic_load_explicit(&" +
                           slotGroupBuffer_ +
                           "[_cd_slot], memory_order_relaxed);");
                cg.addLine("break;");
                cg.decreaseIndent();
                cg.addLine("}");
                cg.addLine("_cd_slot = (_cd_slot + 1u) % " +
                           capacitySymbol_ + ";");
                cg.decreaseIndent();
                cg.addLine("}");
            });
        });
    }

    std::string describe() const override {
        return "CountDistinctGroupBuild";
    }

private:
    std::string rowsSymbol_;
    std::string capacityExpr_;
    std::string capacitySymbol_;
    std::string stateBuffer_;
    std::string hashBuffer_;
    std::string slotGroupBuffer_;
    std::string slotRepRowBuffer_;
    std::string groupRepRowBuffer_;
    std::string rowGroupBuffer_;
    std::string rowGroupSizeExpr_;
    std::string groupCounter_;
    std::vector<GenericMatColumnDesc> groupKeys_;

    void emitHash(MetalCodegen& cg,
                  const std::string& hashVar,
                  const std::string& row) const {
        cg.addLine("uint " + hashVar + " = 2166136261u;");
        for (const auto& key : groupKeys_) {
            if (key.stringLen > 0) {
                const std::string ptr = materializedStringPtrExpr(key, row);
                cg.addBlock("for (uint _cd_hc = 0; _cd_hc < " +
                            std::to_string(std::max(1, key.stringLen)) +
                            "u; ++_cd_hc)", [&]() {
                    cg.addLine(hashVar + " ^= (uint)(uchar)" + ptr +
                               "[_cd_hc];");
                    cg.addLine(hashVar + " *= 16777619u;");
                });
            } else {
                cg.addLine(hashVar + " ^= (uint)(" +
                           materializedValueAt(key, row) + ");");
                cg.addLine(hashVar + " *= 16777619u;");
            }
        }
    }

    void emitKeyEquals(MetalCodegen& cg,
                       const std::string& sameVar,
                       const std::string& leftRow,
                       const std::string& rightRow) const {
        for (const auto& key : groupKeys_) {
            if (key.stringLen > 0) {
                const std::string leftPtr = materializedStringPtrExpr(key, leftRow);
                const std::string rightPtr = materializedStringPtrExpr(key, rightRow);
                cg.addBlock("if (" + sameVar + ")", [&]() {
                    cg.addBlock("for (uint _cd_eqc = 0; _cd_eqc < " +
                                std::to_string(std::max(1, key.stringLen)) +
                                "u; ++_cd_eqc)", [&]() {
                        cg.addIf(leftPtr + "[_cd_eqc] != " + rightPtr +
                                 "[_cd_eqc]", [&]() {
                            cg.addLine(sameVar + " = false;");
                            cg.addLine("break;");
                        });
                    });
                });
            } else {
                cg.addLine(sameVar + " = " + sameVar + " && (" +
                           materializedValueAt(key, leftRow) + " == " +
                           materializedValueAt(key, rightRow) + ");");
            }
        }
    }
};

class MetalCountDistinctBitmapFill : public MetalOperator {
public:
    MetalCountDistinctBitmapFill(std::string rowsSymbol,
                                 std::string distinctDomainExpr,
                                 std::string distinctDomainSymbol,
                                 std::string bitmapStrideSymbol,
                                 std::string rowGroupBuffer,
                                 std::string rowGroupSizeExpr,
                                 std::string bitmapBuffer,
                                 std::string activeWordsBuffer,
                                 std::string activeCountBuffer,
                                 GenericMatColumnDesc distinctColumn)
        : rowsSymbol_(std::move(rowsSymbol)),
          distinctDomainExpr_(std::move(distinctDomainExpr)),
          distinctDomainSymbol_(std::move(distinctDomainSymbol)),
          bitmapStrideSymbol_(std::move(bitmapStrideSymbol)),
          rowGroupBuffer_(std::move(rowGroupBuffer)),
          rowGroupSizeExpr_(std::move(rowGroupSizeExpr)),
          bitmapBuffer_(std::move(bitmapBuffer)),
          activeWordsBuffer_(std::move(activeWordsBuffer)),
          activeCountBuffer_(std::move(activeCountBuffer)),
          distinctColumn_(std::move(distinctColumn)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string nRows = tableSizeName(rowsSymbol_);
        cg.setPhaseScannedTable(rowsSymbol_);
        cg.addResolvedScalarParam(nRows, "uint", rowsSymbol_);
        cg.addResolvedScalarParam(distinctDomainSymbol_, "uint",
                                  distinctDomainExpr_);
        cg.addResolvedScalarParam(bitmapStrideSymbol_, "uint",
                                  "(" + distinctDomainExpr_ + " + 31) / 32");
        bindMaterializedColumns(cg, {distinctColumn_});
        cg.addBufferParam(rowGroupBuffer_, "uint", rowGroupSizeExpr_, false);
        cg.addBufferParam(bitmapBuffer_, "atomic_uint", "", false);
        cg.addBufferParam(activeWordsBuffer_, "uint", "", false);
        cg.addBufferParam(activeCountBuffer_, "atomic_uint", "", false);

        cg.addBlock("for (uint _r = tid; _r < " + nRows + "; _r += tpg)", [&]() {
            cg.addLine("uint _cd_gid = " + rowGroupBuffer_ + "[_r];");
            cg.addIf("_cd_gid != 0xFFFFFFFFu", [&]() {
                cg.addLine("uint _cd_value = (uint)(" +
                           materializedValueAt(distinctColumn_, "_r") + ");");
                cg.addIf("_cd_value < " + distinctDomainSymbol_, [&]() {
                    cg.addLine("uint _cd_word = _cd_gid * " +
                               bitmapStrideSymbol_ + " + (_cd_value >> 5u);");
                    cg.addLine("uint _cd_mask = 1u << (_cd_value & 31u);");
                    cg.addLine("uint _cd_old = atomic_fetch_or_explicit(&" +
                               bitmapBuffer_ + "[_cd_word], _cd_mask, "
                               "memory_order_relaxed);");
                    cg.addIf("_cd_old == 0u", [&]() {
                        cg.addLine("uint _cd_slot = atomic_fetch_add_explicit(&" +
                                   activeCountBuffer_ +
                                   "[0], 1u, memory_order_relaxed);");
                        cg.addLine(activeWordsBuffer_ + "[_cd_slot] = _cd_word;");
                    });
                });
            });
        });
    }

    std::string describe() const override {
        return "CountDistinctBitmapFill";
    }

private:
    std::string rowsSymbol_;
    std::string distinctDomainExpr_;
    std::string distinctDomainSymbol_;
    std::string bitmapStrideSymbol_;
    std::string rowGroupBuffer_;
    std::string rowGroupSizeExpr_;
    std::string bitmapBuffer_;
    std::string activeWordsBuffer_;
    std::string activeCountBuffer_;
    GenericMatColumnDesc distinctColumn_;
};

class MetalCountDistinctBitmapPopcount : public MetalOperator {
public:
    MetalCountDistinctBitmapPopcount(std::string activeWordsSymbol,
                                     std::string bitmapStrideSymbol,
                                     std::string distinctDomainExpr,
                                     std::string activeWordsBuffer,
                                     std::string bitmapBuffer,
                                     std::string countBuffer)
        : activeWordsSymbol_(std::move(activeWordsSymbol)),
          bitmapStrideSymbol_(std::move(bitmapStrideSymbol)),
          distinctDomainExpr_(std::move(distinctDomainExpr)),
          activeWordsBuffer_(std::move(activeWordsBuffer)),
          bitmapBuffer_(std::move(bitmapBuffer)),
          countBuffer_(std::move(countBuffer)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string nRows = tableSizeName(activeWordsSymbol_);
        cg.setPhaseScannedTable(activeWordsSymbol_);
        cg.addResolvedScalarParam(nRows, "uint", activeWordsSymbol_);
        cg.addResolvedScalarParam(bitmapStrideSymbol_, "uint",
                                  "(" + distinctDomainExpr_ + " + 31) / 32");
        cg.addBufferParam(activeWordsBuffer_, "uint", "", false);
        cg.addBufferParam(bitmapBuffer_, "uint", "", false);
        cg.addBufferParam(countBuffer_, "atomic_uint", "", false);

        cg.addBlock("for (uint _i = tid; _i < " + nRows + "; _i += tpg)", [&]() {
            cg.addLine("uint _w = " + activeWordsBuffer_ + "[_i];");
            cg.addLine("uint _cd_gid = _w / " + bitmapStrideSymbol_ + ";");
            cg.addLine("uint _cd_count = popcount(" + bitmapBuffer_ + "[_w]);");
            cg.addIf("_cd_count != 0u", [&]() {
                cg.addLine("atomic_fetch_add_explicit(&" + countBuffer_ +
                           "[_cd_gid], _cd_count, memory_order_relaxed);");
            });
        });
    }

    std::string describe() const override {
        return "CountDistinctBitmapPopcount";
    }

private:
    std::string activeWordsSymbol_;
    std::string bitmapStrideSymbol_;
    std::string distinctDomainExpr_;
    std::string activeWordsBuffer_;
    std::string bitmapBuffer_;
    std::string countBuffer_;
};

class MetalCountDistinctGroupCompact : public MetalOperator {
public:
    MetalCountDistinctGroupCompact(std::string groupRowsSymbol,
                                   std::string outputCapacityExpr,
                                   std::string outputCapacitySymbol,
                                   std::string groupRepRowBuffer,
                                   std::string countBuffer,
                                   std::string outputCounter,
                                   std::string aggregateDisplayName,
                                   std::vector<GenericMatColumnDesc> groupKeys,
                                   std::vector<GenericMatColumnDesc> outputs)
        : groupRowsSymbol_(std::move(groupRowsSymbol)),
          outputCapacityExpr_(std::move(outputCapacityExpr)),
          outputCapacitySymbol_(std::move(outputCapacitySymbol)),
          groupRepRowBuffer_(std::move(groupRepRowBuffer)),
          countBuffer_(std::move(countBuffer)),
          outputCounter_(std::move(outputCounter)),
          aggregateDisplayName_(std::move(aggregateDisplayName)),
          groupKeys_(std::move(groupKeys)),
          outputs_(std::move(outputs)) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        (void)consume;
        const std::string nRows = tableSizeName(groupRowsSymbol_);
        cg.setPhaseScannedTable(groupRowsSymbol_);
        cg.addResolvedScalarParam(nRows, "uint", groupRowsSymbol_);
        cg.addResolvedScalarParam(outputCapacitySymbol_, "uint",
                                  outputCapacityExpr_);
        bindMaterializedColumns(cg, groupKeys_);
        cg.addBufferParam(groupRepRowBuffer_, "uint", "", false);
        cg.addBufferParam(countBuffer_, "uint", "", false);
        cg.addAtomicBufferParam(outputCounter_, "atomic_uint", "1");
        for (const auto& out : outputs_) {
            std::string sizeExpr = outputCapacityExpr_;
            if (out.stringLen > 0)
                sizeExpr += " * " + std::to_string(out.stringLen);
            cg.addBufferParam(out.bufferName, out.metalType, sizeExpr, false);
        }

        cg.registerMaterializeOutput(outputCounter_);
        for (const auto& out : outputs_) {
            cg.registerOutputColumn(out.displayName, out.bufferName,
                                    out.metalType, out.stringLen, out.scaleDown,
                                    out.isLongPair);
        }

        cg.addBlock("for (uint _gid = tid; _gid < " + nRows + "; _gid += tpg)", [&]() {
            cg.addLine("uint _cd_count = " + countBuffer_ + "[_gid];");
            cg.addIf("_cd_count != 0u", [&]() {
                cg.addLine("uint _pos = atomic_fetch_add_explicit(&" +
                           outputCounter_ + "[0], 1u, memory_order_relaxed);");
                cg.addIf("_pos < " + outputCapacitySymbol_, [&]() {
                    cg.addLine("uint _rep = " + groupRepRowBuffer_ + "[_gid];");
                    emitOutputWrites(cg, "_rep", "_pos", "_cd_count");
                });
            });
        });
    }

    std::string describe() const override {
        return "CountDistinctGroupCompact";
    }

private:
    std::string groupRowsSymbol_;
    std::string outputCapacityExpr_;
    std::string outputCapacitySymbol_;
    std::string groupRepRowBuffer_;
    std::string countBuffer_;
    std::string outputCounter_;
    std::string aggregateDisplayName_;
    std::vector<GenericMatColumnDesc> groupKeys_;
    std::vector<GenericMatColumnDesc> outputs_;

    const GenericMatColumnDesc* groupKeyForDisplay(
            const std::string& displayName) const {
        for (const auto& key : groupKeys_) {
            if (key.displayName == displayName) return &key;
        }
        return nullptr;
    }

    void emitGroupOutput(MetalCodegen& cg,
                         const GenericMatColumnDesc& out,
                         const GenericMatColumnDesc& key,
                         const std::string& rep,
                         const std::string& pos) const {
        if (key.stringLen > 0) {
            const std::string ptr = materializedStringPtrExpr(key, rep);
            cg.addBlock("for (uint _oc = 0; _oc < " +
                        std::to_string(std::max(1, key.stringLen)) +
                        "u; ++_oc)", [&]() {
                cg.addLine(out.bufferName + "[" + pos + " * " +
                           std::to_string(std::max(1, key.stringLen)) +
                           "u + _oc] = " + ptr + "[_oc];");
            });
        } else {
            cg.addLine(out.bufferName + "[" + pos + "] = " +
                       materializedValueAt(key, rep) + ";");
        }
    }

    void emitOutputWrites(MetalCodegen& cg,
                          const std::string& rep,
                          const std::string& pos,
                          const std::string& countValue) const {
        for (const auto& out : outputs_) {
            if (out.displayName == aggregateDisplayName_) {
                cg.addLine(out.bufferName + "[" + pos + "] = " + countValue + ";");
                continue;
            }
            if (const auto* key = groupKeyForDisplay(out.displayName)) {
                emitGroupOutput(cg, out, *key, rep, pos);
            }
        }
    }
};

void attachCountDistinctBitmapAllocationHook(
        MetalQueryPlan::Phase& phase,
        std::string groupCounter,
        std::string distinctDomainExpr,
        std::string groupRowsSymbol,
        std::string popWordsSymbol,
        std::string bitmapStrideSymbol,
        std::string bitmapBuffer,
        std::string countBuffer,
        std::string activeWordsBuffer,
        std::string activeCountBuffer) {
    phase.postDispatchHook =
        [groupCounter = std::move(groupCounter),
         distinctDomainExpr = std::move(distinctDomainExpr),
         groupRowsSymbol = std::move(groupRowsSymbol),
         popWordsSymbol = std::move(popWordsSymbol),
         bitmapStrideSymbol = std::move(bitmapStrideSymbol),
         bitmapBuffer = std::move(bitmapBuffer),
         countBuffer = std::move(countBuffer),
         activeWordsBuffer = std::move(activeWordsBuffer),
         activeCountBuffer = std::move(activeCountBuffer)](
                MetalGenericExecutor& executor) {
            auto* counter = executor.getAllocatedBuffer(groupCounter);
            if (!counter) return 0.0;
            uint32_t groupCount =
                *static_cast<const uint32_t*>(counter->contents());

            size_t domain = 0;
            if (!executor.tryResolveSizeExpression(distinctDomainExpr, domain))
                return 0.0;
            const size_t stride = (domain + 31) / 32;
            const size_t bitmapWords = static_cast<size_t>(groupCount) * stride;

            auto* device = executor.device();
            const size_t bitmapBytes =
                std::max<size_t>(bitmapWords * sizeof(uint32_t), sizeof(uint32_t));
            auto* bitmap = device->newBuffer(bitmapBytes,
                                             MTL::ResourceStorageModeShared);
            if (bitmap) {
                std::memset(bitmap->contents(), 0, bitmapBytes);
                bitmap->didModifyRange(NS::Range::Make(0, bitmapBytes));
                executor.registerAllocatedBuffer(bitmapBuffer, bitmap);
            }

            const size_t countBytes =
                std::max<size_t>(static_cast<size_t>(groupCount) *
                                 sizeof(uint32_t), sizeof(uint32_t));
            auto* counts = device->newBuffer(countBytes,
                                             MTL::ResourceStorageModeShared);
            if (counts) {
                std::memset(counts->contents(), 0, countBytes);
                counts->didModifyRange(NS::Range::Make(0, countBytes));
                executor.registerAllocatedBuffer(countBuffer, counts);
            }

            const size_t activeWordsBytes =
                std::max<size_t>(bitmapWords * sizeof(uint32_t), sizeof(uint32_t));
            auto* activeWords =
                device->newBuffer(activeWordsBytes, MTL::ResourceStorageModeShared);
            if (activeWords) {
                executor.registerAllocatedBuffer(activeWordsBuffer, activeWords);
            }

            auto* activeCount =
                device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
            if (activeCount) {
                std::memset(activeCount->contents(), 0, sizeof(uint32_t));
                activeCount->didModifyRange(NS::Range::Make(0, sizeof(uint32_t)));
                executor.registerAllocatedBuffer(activeCountBuffer, activeCount);
            }

            executor.registerSymbol(groupRowsSymbol, groupCount);
            executor.registerScalarInt(groupRowsSymbol, static_cast<int>(groupCount));
            executor.registerSymbol(tableSizeName(groupRowsSymbol), groupCount);
            executor.registerScalarInt(tableSizeName(groupRowsSymbol),
                                       static_cast<int>(groupCount));
            executor.registerSymbol(popWordsSymbol, bitmapWords);
            executor.registerScalarInt(popWordsSymbol, static_cast<int>(bitmapWords));
            executor.registerSymbol(tableSizeName(popWordsSymbol), bitmapWords);
            executor.registerScalarInt(tableSizeName(popWordsSymbol),
                                       static_cast<int>(bitmapWords));
            executor.registerSymbol(bitmapStrideSymbol, stride);
            executor.registerScalarInt(bitmapStrideSymbol, static_cast<int>(stride));
            return 0.0;
        };
}

void attachCountDistinctActiveWordCountHook(
        MetalQueryPlan::Phase& phase,
        std::string activeCountBuffer,
        std::string activeWordsSymbol) {
    phase.postDispatchHook =
        [activeCountBuffer = std::move(activeCountBuffer),
         activeWordsSymbol = std::move(activeWordsSymbol)](
                MetalGenericExecutor& executor) {
            auto* counter = executor.getAllocatedBuffer(activeCountBuffer);
            if (!counter) return 0.0;
            uint32_t activeWords =
                *static_cast<const uint32_t*>(counter->contents());
            executor.registerSymbol(activeWordsSymbol, activeWords);
            executor.registerScalarInt(activeWordsSymbol,
                                       static_cast<int>(activeWords));
            executor.registerSymbol(tableSizeName(activeWordsSymbol), activeWords);
            executor.registerScalarInt(tableSizeName(activeWordsSymbol),
                                       static_cast<int>(activeWords));
            return 0.0;
        };
}

std::vector<GenericMatColumnDesc> fdKeyedGroupOutputColumns(
        const std::string& tag,
        const GenericGroupSpec& groupSpec,
        const std::vector<GenericMatColumnDesc>& inputColumns,
        const std::vector<IrPendingAgg>& pending) {
    std::vector<GenericMatColumnDesc> out;
    std::set<std::string> seen;
    const std::string prefix = "d_" + sanitizeIdentifier(tag) + "_out_";

    auto pendingForDisplay = [&](const std::string& display) -> const IrPendingAgg* {
        for (const auto& p : pending) {
            if (p.displayName == display) return &p;
        }
        return nullptr;
    };

    auto appendDisplay = [&](const std::string& display) {
        if (display.empty() || display.rfind("__hidden_", 0) == 0 ||
            seen.count(display)) {
            return;
        }
        if (std::find(groupSpec.keyColumns.begin(), groupSpec.keyColumns.end(),
                      display) != groupSpec.keyColumns.end()) {
            const auto* col = findMaterializedColumn(inputColumns, display);
            if (!col) return;
            const std::string outType = col->stringLen > 0 ? "char" : col->metalType;
            out.push_back({display,
                           prefix + std::to_string(out.size()) + "_" +
                               sanitizeIdentifier(display),
                           outType, col->stringLen});
            seen.insert(display);
            return;
        }
        const auto* agg = pendingForDisplay(display);
        if (!agg) return;
        std::string outType = "uint";
        int scaleDown = 0;
        bool isLongPair = false;
        if (agg->scaleDown < 0) {
            outType = "float";
        } else if (agg->isLongPair) {
            outType = "uint";
            scaleDown = agg->scaleDown > 0 ? agg->scaleDown : 0;
            isLongPair = true;
        } else if (agg->isFloatSum || agg->isMinMax) {
            outType = "float";
        }
        out.push_back({display,
                       prefix + std::to_string(out.size()) + "_" +
                           sanitizeIdentifier(display),
                       outType, 0, scaleDown, isLongPair});
        seen.insert(display);
    };

    for (const auto& display : groupSpec.outputColumns)
        appendDisplay(display);
    if (out.empty()) {
        for (const auto& display : groupSpec.keyColumns)
            appendDisplay(display);
        for (const auto& display : groupSpec.aggColumns)
            appendDisplay(display);
    }
    return out;
}

std::vector<GenericMatColumnDesc> fdKeyedGroupTopKNarrowColumns(
        const std::string& tag,
        const std::vector<GenericMatColumnDesc>& fullOutputs,
        const GenericSortSpec& sortSpec) {
    std::vector<GenericMatColumnDesc> out;
    std::set<std::string> seen;
    const std::string prefix = "d_" + sanitizeIdentifier(tag) + "_out_";

    auto appendClone = [&](const GenericMatColumnDesc& src) {
        if (src.displayName.empty() || seen.count(src.displayName)) return;
        out.push_back({src.displayName,
                       prefix + std::to_string(out.size()) + "_" +
                           sanitizeIdentifier(src.displayName),
                       src.metalType, src.stringLen, src.scaleDown,
                       src.isLongPair});
        seen.insert(src.displayName);
    };

    for (const auto& sk : sortSpec.keys) {
        if (const auto* col = findMaterializedColumn(fullOutputs, sk.column))
            appendClone(*col);
    }
    out.push_back({kFdHiddenBucketDisplay,
                   prefix + std::to_string(out.size()) + "_bucket",
                   "uint", 0});
    return out;
}

bool materializedCountDistinctTypeSupported(const GenericMatColumnDesc& col) {
    if (col.stringLen > 0) return false;
    return col.metalType == "int" || col.metalType == "uint" ||
           col.metalType == "char" || col.metalType == "uchar";
}

bool materializedGroupKeySupported(const GenericMatColumnDesc& col) {
    if (col.stringLen > 0) {
        return !col.stringRowRef ||
               (!col.stringSourceTable.empty() &&
                !col.stringSourceColumn.empty());
    }
    return col.metalType == "int" || col.metalType == "uint" ||
           col.metalType == "char" || col.metalType == "uchar";
}

bool tryAppendMaterializedCountDistinctGroup(
        MetalQueryPlan& plan,
        MetalQueryPlan::Phase& materializePhase,
        const GenericAggregateDetail& aggregate,
        const GenericGroupSpec& groupSpec,
        const std::vector<GenericMatColumnDesc>& materializedCols,
        const GenericSortSpec& sortSpec,
        const std::string& materializedCounter,
        const std::string& inputUpperBoundExpr,
        const std::string& outputCapacityExpr,
        bool* applied,
        std::string* error) {
    if (applied) *applied = false;
    if (aggregate.having || aggregate.aggregates.size() != 1)
        return true;

    const auto& projection = aggregate.aggregates.front();
    auto* agg = projection.expr
        ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
        : nullptr;
    if (!agg || agg->func != AggFunc::COUNT_DISTINCT || !agg->arg)
        return true;

    const std::string aggregateDisplay =
        projection.name.empty() ? "agg_0" : projection.name;
    const auto* distinctCol =
        findMaterializedColumn(materializedCols, aggregateDisplay);
    if (!distinctCol ||
        !materializedCountDistinctTypeSupported(*distinctCol) ||
        distinctCol->distinctDomainSymbol.empty()) {
        return true;
    }

    std::vector<GenericMatColumnDesc> groupKeys;
    groupKeys.reserve(groupSpec.keyColumns.size());
    for (const auto& display : groupSpec.keyColumns) {
        const auto* col = findMaterializedColumn(materializedCols, display);
        if (!col || !materializedGroupKeySupported(*col))
            return true;
        groupKeys.push_back(*col);
    }
    if (groupKeys.empty())
        return true;

    std::vector<GenericMatColumnDesc> outputs;
    std::set<std::string> seen;
    const std::string outPrefix = "d_ir_multi_cd_group_out_";
    auto appendOutput = [&](const std::string& display) -> bool {
        if (display.empty() || seen.count(display) ||
            display.rfind("__hidden_", 0) == 0) {
            return true;
        }
        if (display == aggregateDisplay) {
            outputs.push_back({display,
                               outPrefix + std::to_string(outputs.size()) +
                                   "_" + sanitizeIdentifier(display),
                               "uint", 0});
            seen.insert(display);
            return true;
        }
        for (const auto& key : groupKeys) {
            if (key.displayName != display) continue;
            outputs.push_back({display,
                               outPrefix + std::to_string(outputs.size()) +
                                   "_" + sanitizeIdentifier(display),
                               key.stringLen > 0 ? "char" : key.metalType,
                               key.stringLen});
            seen.insert(display);
            return true;
        }
        return false;
    };

    for (const auto& display : groupSpec.outputColumns) {
        if (!appendOutput(display)) {
            if (error)
                *error = "IR multi-table count-distinct lowerer: output column is not produced by the aggregate.";
            return false;
        }
    }
    if (outputs.empty()) {
        for (const auto& display : groupSpec.keyColumns) {
            if (!appendOutput(display)) return false;
        }
        if (!appendOutput(aggregateDisplay)) return false;
    }

    const std::string tag = "ir_multi_cd_group";
    const std::string rowsSymbol = tag + "_input_rows";
    const std::string groupRowsSymbol = tag + "_rows";
    const std::string popWordsSymbol = tag + "_pop_words";
    const std::string activeWordsSymbol = tag + "_active_words";
    const std::string capacityExpr =
        groupHashCapacityExpr(inputUpperBoundExpr, outputCapacityExpr);
    const std::string capacitySymbol = "n_" + tag + "_slots";
    const std::string outputCapacitySymbol = "n_" + tag + "_out_cap";
    const std::string distinctDomainSymbol = "n_" + tag + "_distinct_domain";
    const std::string bitmapStrideSymbol = "n_" + tag + "_bitmap_stride";
    const std::string stateBuffer = "d_" + tag + "_state";
    const std::string hashBuffer = "d_" + tag + "_hash";
    const std::string slotGroupBuffer = "d_" + tag + "_slot_group";
    const std::string slotRepRowBuffer = "d_" + tag + "_slot_rep_row";
    const std::string groupRepRowBuffer = "d_" + tag + "_group_rep_row";
    const std::string rowGroupBuffer = "d_" + tag + "_row_group";
    const std::string groupCounter = "d_" + tag + "_count";
    const std::string bitmapBuffer = "d_" + tag + "_bitmap";
    const std::string countBuffer = "d_" + tag + "_counts";
    const std::string activeWordsBuffer = "d_" + tag + "_active_words";
    const std::string activeCountBuffer = "d_" + tag + "_active_word_count";
    const std::string outputCounter = "d_" + tag + "_result_count";

    attachMaterializedCountHook(materializePhase, materializedCounter,
                                rowsSymbol);
    auto& buildPhase = appendPhase(
        plan, "GENERIC_ir_multi_table_count_distinct_group_build",
        std::make_unique<MetalCountDistinctGroupBuild>(
            rowsSymbol, capacityExpr, capacitySymbol, stateBuffer, hashBuffer,
            slotGroupBuffer, slotRepRowBuffer, groupRepRowBuffer,
            rowGroupBuffer, inputUpperBoundExpr, groupCounter, groupKeys));
    attachCountDistinctBitmapAllocationHook(
        buildPhase, groupCounter, distinctCol->distinctDomainSymbol,
        groupRowsSymbol, popWordsSymbol, bitmapStrideSymbol, bitmapBuffer,
        countBuffer, activeWordsBuffer, activeCountBuffer);

    auto& fillPhase = appendPhase(
        plan, "GENERIC_ir_multi_table_count_distinct_bitmap_fill",
        std::make_unique<MetalCountDistinctBitmapFill>(
            rowsSymbol, distinctCol->distinctDomainSymbol, distinctDomainSymbol,
            bitmapStrideSymbol, rowGroupBuffer, inputUpperBoundExpr,
            bitmapBuffer, activeWordsBuffer, activeCountBuffer, *distinctCol));
    attachCountDistinctActiveWordCountHook(
        fillPhase, activeCountBuffer, activeWordsSymbol);

    appendPhase(
        plan, "GENERIC_ir_multi_table_count_distinct_popcount",
        std::make_unique<MetalCountDistinctBitmapPopcount>(
            activeWordsSymbol, bitmapStrideSymbol,
            distinctCol->distinctDomainSymbol, activeWordsBuffer, bitmapBuffer,
            countBuffer));

    auto& compactPhase = appendPhase(
        plan, "GENERIC_ir_multi_table_count_distinct_compact",
        std::make_unique<MetalCountDistinctGroupCompact>(
            groupRowsSymbol, outputCapacityExpr, outputCapacitySymbol,
            groupRepRowBuffer, countBuffer, outputCounter, aggregateDisplay,
            groupKeys, outputs));

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string sortRowsSymbol =
            "n_gpu_sort_ir_multi_cd_group_rows";
        attachMaterializedCountHook(compactPhase, outputCounter,
                                    sortRowsSymbol);
        if (!appendBestGenericGpuOrder(plan, "ir_multi_cd_group",
                                       sortRowsSymbol, outputCapacityExpr,
                                       outputs, sortSpec, error)) {
            return false;
        }
    }

    if (applied) *applied = true;
    return true;
}

void collectPredicateColumnsLocal(const GenericPredicatePtr& pred,
                                  std::vector<GenericColumnExpr>& out) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectExprColumns(node.left, out);
            collectExprColumns(node.right, out);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectExprColumns(node.expr, out);
            collectExprColumns(node.low, out);
            collectExprColumns(node.high, out);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectExprColumns(node.expr, out);
            for (const auto& value : node.values)
                collectExprColumns(value, out);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectExprColumns(node.expr, out);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectPredicateColumnsLocal(child, out);
        }
    }, pred->node);
}

bool predicateOnlyReferencesRelation(const GenericPredicatePtr& pred,
                                     int relationInstance) {
    std::vector<GenericColumnExpr> cols;
    collectPredicateColumnsLocal(pred, cols);
    for (const auto& col : cols) {
        if (col.relationInstance.value != relationInstance)
            return false;
    }
    return true;
}

const GenericScanDetail* scanForRelation(const MultiTableGroupedAggShape& shape,
                                         int relationInstance) {
    for (const auto* scanNode : shape.scans) {
        auto* scan = scanDetail(scanNode);
        if (scan && scan->relationInstance.value == relationInstance)
            return scan;
    }
    return nullptr;
}

GenericColumnExpr relationColumn(const GenericScanDetail& scan,
                                 const std::string& column,
                                 const SchemaProvider& schema) {
    GenericColumnExpr out;
    out.relationInstance = scan.relationInstance;
    out.table = scan.table;
    out.alias = scan.alias;
    out.column = column;
    if (schema.hasColumn(scan.table, column)) {
        out.type = TypeInfo{schema.columnType(scan.table, column),
                            schema.columnFixedWidth(scan.table, column)};
        if (auto domain = schema.groupDomain(scan.table, column)) {
            out.hasGroupDomain = true;
            out.domainMin = domain->minValue;
            out.domainMax = domain->maxValue;
        }
        out.charDomain = schema.charDomain(scan.table, column);
        out.numericScale = schema.numericScale(scan.table, column);
        out.keyDomainSymbol = schema.keyDomainSymbol(scan.table, column);
        out.distinctDomainSymbol =
            schema.distinctDomainSymbol(scan.table, column);
    }
    return out;
}

std::string scanScopeName(const GenericScanDetail& scan) {
    return sanitizeIdentifier(scan.alias.empty() ? scan.table : scan.alias);
}

std::optional<MetalQueryPlan> lowerSharedDimensionCount(
        const GenericRelPlan& ir,
        const MultiTableGroupedAggShape& shape,
        const GenericAggregateDetail& aggregate,
        const AnalyzedQuery* aq,
        std::string* error) {
    if (!aq || !aq->schema) return std::nullopt;
    if (shape.scans.size() != 3) return std::nullopt;

    if (aggregate.groupBy.size() != 1 || aggregate.aggregates.size() != 1)
        return std::nullopt;
    auto* groupCol = aggregate.groupBy.front()
        ? std::get_if<GenericColumnExpr>(&aggregate.groupBy.front()->node)
        : nullptr;
    if (!groupCol)
        return std::nullopt;

    const auto& aggProjection = aggregate.aggregates.front();
    auto* agg = aggProjection.expr
        ? std::get_if<GenericAggregateExpr>(&aggProjection.expr->node)
        : nullptr;
    if (!agg || agg->func != AggFunc::COUNT) return std::nullopt;

    const auto* dimensionScan =
        scanForRelation(shape, groupCol->relationInstance.value);
    if (!dimensionScan) return std::nullopt;
    const GenericRelNode* dimensionScanNode = nullptr;
    for (const auto* scanNode : shape.scans) {
        auto* scan = scanDetail(scanNode);
        if (scan &&
            scan->relationInstance.value ==
                dimensionScan->relationInstance.value) {
            dimensionScanNode = scanNode;
            break;
        }
    }
    const auto* dimensionRel = relationForScan(ir, dimensionScanNode);
    if (!dimensionRel || dimensionRel->primaryKeyColumn.empty() ||
        dimensionRel->primaryKeyDomainSymbol.empty()) {
        return std::nullopt;
    }

    if (auto* filter = filterDetail(shape.filter)) {
        if (!predicateOnlyReferencesRelation(filter->predicate,
                                             dimensionScan->relationInstance.value)) {
            return std::nullopt;
        }
        if (!predicateSupported(filter->predicate)) {
            if (error)
                *error = "IR shared-dimension count lowerer: dimension filter predicate is not supported.";
            return std::nullopt;
        }
    }

    ColumnEquivalence eq = buildJoinColumnEquivalence(shape.joins, shape.filter);
    GenericColumnExpr dimensionPk =
        relationColumn(*dimensionScan, dimensionRel->primaryKeyColumn,
                       *aq->schema);
    auto equivalentCols = eq.columnsEquivalentTo(dimensionPk);

    struct CountInput {
        const GenericScanDetail* scan = nullptr;
        GenericColumnExpr keyCol;
        std::string countBuffer;
    };
    std::vector<CountInput> countInputs;
    for (const auto* scanNode : shape.scans) {
        auto* scan = scanDetail(scanNode);
        if (!scan ||
            scan->relationInstance.value ==
                dimensionScan->relationInstance.value) {
            continue;
        }

        auto it = std::find_if(
            equivalentCols.begin(), equivalentCols.end(),
            [&](const GenericColumnExpr& col) {
                return col.relationInstance.value == scan->relationInstance.value;
            });
        if (it == equivalentCols.end())
            return std::nullopt;

        CountInput input;
        input.scan = scan;
        input.keyCol = *it;
        input.countBuffer = "d_ir_shared_dim_count_" +
                            scanScopeName(*scan) + "_" +
                            sanitizeIdentifier(it->column);
        countInputs.push_back(std::move(input));
    }
    if (countInputs.size() != 2) return std::nullopt;

    const std::string idxVar = "i";

    MetalQueryPlan plan;
    plan.name = "GENERIC_IR_SHARED_DIMENSION_COUNT";

    for (const auto& input : countInputs) {
        auto scan = makeScanForCols(input.scan->table, idxVar,
                                    {input.keyCol.column},
                                    aq->schema);
        auto count = std::make_unique<MetalAtomicCount>(
            std::move(scan), input.countBuffer,
            input.keyCol.column + "[" + idxVar + "]",
            dimensionRel->primaryKeyDomainSymbol);
        appendPhase(plan, "GENERIC_ir_shared_dimension_count_" +
                          scanScopeName(*input.scan),
                    std::move(count));
    }

    int groupStringWidth = 0;
    if (groupCol->type.type == DataType::CHAR_FIXED) {
        groupStringWidth = groupCol->type.fixedWidth > 0
            ? groupCol->type.fixedWidth
            : aq->schema->columnFixedWidth(dimensionScan->table,
                                           groupCol->column);
        if (groupStringWidth <= 0) return std::nullopt;
    }

    std::set<std::string> dimensionCols{
        dimensionRel->primaryKeyColumn, groupCol->column};
    std::unique_ptr<MetalOperator> dimensionPipe =
        makeScanForCols(dimensionScan->table, idxVar, dimensionCols,
                        aq->schema);
    if (auto* filter = filterDetail(shape.filter)) {
        dimensionPipe = maybeSelect(
            std::move(dimensionPipe),
            genericPredicateToMetal(filter->predicate, idxVar));
    }

    const std::string resultCounter = "d_ir_shared_dim_result_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(dimensionPipe), resultCounter, "1");
    const std::string groupDisplayName =
        groupDisplayNameForAggregate(aggregate, 0);
    const std::string groupBuffer = "d_ir_shared_dim_0_" +
        sanitizeIdentifier(groupDisplayName);
    const std::string countBuffer = "d_ir_shared_dim_1_" +
        sanitizeIdentifier(aggProjection.name.empty() ? "cnt"
                                                      : aggProjection.name);
    std::string countExpr = "(int)(";
    for (size_t i = 0; i < countInputs.size(); ++i) {
        if (i > 0) countExpr += " * ";
        countExpr += "atomic_load_explicit(&" + countInputs[i].countBuffer +
                     "[" + dimensionRel->primaryKeyColumn + "[" + idxVar +
                     "]], memory_order_relaxed)";
    }
    countExpr += ")";

    std::vector<GenericMatColumnDesc> outputCols;
    if (groupCol->type.type == DataType::CHAR_FIXED) {
        materialize->addColumn(
            groupBuffer, "char",
            groupCol->column + " + " + idxVar + " * " +
                std::to_string(groupStringWidth) + "ul",
            groupDisplayName,
            tableSizeName(dimensionScan->table) + " * " +
                std::to_string(groupStringWidth),
            groupStringWidth);
        outputCols.push_back(
            {groupDisplayName, groupBuffer, "char", groupStringWidth});
    } else {
        materialize->addColumn(
            groupBuffer, metalTypeForType(groupCol->type),
            groupCol->column + "[" + idxVar + "]",
            groupDisplayName, tableSizeName(dimensionScan->table), 0);
        outputCols.push_back(
            {groupDisplayName, groupBuffer, metalTypeForType(groupCol->type), 0});
    }

    const std::string countDisplay =
        aggProjection.name.empty() ? "cnt" : aggProjection.name;
    materialize->addColumn(countBuffer, "int", countExpr, countDisplay,
                           tableSizeName(dimensionScan->table), 0);
    auto& matPhase = appendPhase(plan, "GENERIC_ir_shared_dimension_materialize",
                                 std::move(materialize));
    for (const auto& input : countInputs) {
        matPhase.extraBuffers.push_back(
            {input.countBuffer, "atomic_uint", true, false});
    }

    outputCols.push_back({countDisplay, countBuffer, "int", 0});

    std::vector<IrGroupKeyDesc> groupKeys;
    IrGroupKeyDesc groupKey;
    groupKey.displayName = groupDisplayName;
    groupKeys.push_back(std::move(groupKey));
    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape.limit);
    if (auto* sort = sortDetail(shape.sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, aggregate, groupKeys);
            if (!name) {
                if (error)
                    *error = "IR shared-dimension count lowerer: ORDER BY key is not an output.";
                return std::nullopt;
            }
            sortSpec.keys.push_back({*name, key.descending});
        }
    }
    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_shared_dim_rows";
        attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
        if (!appendBestGenericGpuOrder(plan, "ir_shared_dim", rowsSym,
                                       tableSizeName(dimensionScan->table),
                                       outputCols, sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

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

    if (auto p = lowerSharedDimensionCount(ir, *shape, *aggregate, aq, error))
        return p;

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

    std::vector<MetalMaterialize::Column> pendingMaterializeCols;
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
        pendingMaterializeCols.push_back(MetalMaterialize::Column{
            bufferName, metalType,
            carriedFixedString
                ? carriedFixedString->rowVarName
                : materializeExprToMetalWithCarryMap(
                      expr, idxVar, lowering->carryMap),
            displayName, sizeExpr, carriedFixedString ? 0 : stringLen});
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

    std::optional<size_t> materializePhaseIndex;
    auto ensureMaterializePhase = [&]() -> MetalQueryPlan::Phase& {
        if (!materializePhaseIndex) {
            auto materialize = std::make_unique<MetalMaterialize>(
                std::move(lowering->probePipe), resultCounter, "1");
            for (const auto& col : pendingMaterializeCols) {
                materialize->addColumn(col.arrayName, col.type, col.valueExpr,
                                       col.displayName, col.sizeExpr,
                                       col.stringLen);
            }
            auto& phase = appendPhase(
                lowering->plan, "GENERIC_ir_multi_table_group_materialize",
                std::move(materialize));
            if (!scalarLookups.empty())
                attachGenericScalarLookupBuffers(phase, scalarLookups);
            materializePhaseIndex = lowering->plan.phases.size() - 1;
        }
        return lowering->plan.phases[*materializePhaseIndex];
    };

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

    const auto* fd = shape->filter ? filterDetail(shape->filter) : nullptr;
    auto joinEqForPk = buildJoinColumnEquivalence(shape->joins, shape->filter);
    std::string groupOutputBoundExpr = lowering->outputSize;
    if (auto pkBound = primaryKeyGroupOutputBound(
            ir, *aggregate, joinEqForPk, lowering->carryMap)) {
        groupOutputBoundExpr = *pkBound;
    } else if (auto finiteBound = finiteGroupOutputBound(
            *aggregate, fd ? fd->predicate : GenericPredicatePtr{})) {
        groupOutputBoundExpr = std::to_string(*finiteBound);
	    } else if (auto relationBound = relationBoundedGroupOutputBound(
	            ir, *aggregate, fd ? fd->predicate : GenericPredicatePtr{})) {
	        groupOutputBoundExpr = *relationBound;
	    }

    auto tryInSubAggDomainReuse = [&]() -> std::optional<MetalQueryPlan> {
        if (aggregate->having || aggregate->aggregates.size() != 1 ||
            lowering->inSubAggs.empty()) {
            return std::nullopt;
        }
        const auto& projection = aggregate->aggregates.front();
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        auto* aggCol = agg && agg->arg
            ? std::get_if<GenericColumnExpr>(&agg->arg->node)
            : nullptr;
        if (!agg || !aggCol || agg->func != AggFunc::SUM ||
            aggCol->type.type != DataType::FLOAT) {
            return std::nullopt;
        }

        const GenericInSubAggInfo* reuseAgg = nullptr;
        for (const auto& info : lowering->inSubAggs) {
            if (info.aggFunc == "sum" && info.table == aggCol->table &&
                info.valueColumn == aggCol->column) {
                reuseAgg = &info;
                break;
            }
        }
        if (!reuseAgg) return std::nullopt;

        auto pkReduction =
            primaryKeyGroupReduction(ir, *aggregate, joinEqForPk,
                                     lowering->carryMap);
        if (!pkReduction || pkReduction->keyDomainExpr != reuseAgg->keyDomain)
            return std::nullopt;
        auto* pkGroupCol = aggregate->groupBy[pkReduction->groupIndex]
            ? std::get_if<GenericColumnExpr>(
                  &aggregate->groupBy[pkReduction->groupIndex]->node)
            : nullptr;
        if (!pkGroupCol) return std::nullopt;

        const GenericJoinDomainBitmapInfo* domainBitmap = nullptr;
        for (const auto& bitmap : lowering->domainBitmaps) {
            if (bitmap.relationInstance == pkReduction->relationInstance &&
                bitmap.keyColumn == pkReduction->keyColumn &&
                bitmap.keyDomain == reuseAgg->keyDomain) {
                domainBitmap = &bitmap;
                break;
            }
        }
        if (!domainBitmap) return std::nullopt;

        auto scanForRelation = [&](int relationInstance)
                -> const GenericScanDetail* {
            for (const auto* scanNode : shape->scans) {
                auto* scan = scanDetail(scanNode);
                if (scan && scan->relationInstance.value == relationInstance)
                    return scan;
            }
            return nullptr;
        };

        auto phaseIndexByName = [&](const std::string& phaseName)
                -> std::optional<size_t> {
            if (phaseName.empty()) return std::nullopt;
            for (size_t i = 0; i < lowering->plan.phases.size(); ++i) {
                if (lowering->plan.phases[i].name == phaseName)
                    return i;
            }
            return std::nullopt;
        };

        auto relationInstanceById = [&](int relationInstance)
                -> const GenericRelationInstance* {
            GenericRelationInstanceId id;
            id.value = relationInstance;
            return ir.findRelationInstance(id);
        };

        auto makePkColumnExpr = [&]() -> GenericColumnExpr {
            GenericColumnExpr col;
            col.relationInstance.value = pkReduction->relationInstance;
            col.table = pkReduction->table;
            col.alias = pkReduction->alias;
            col.column = pkReduction->keyColumn;
            if (pkGroupCol) col.type = pkGroupCol->type;
            return col;
        };

        auto tryPkRowLookupReuse = [&]() -> std::optional<MetalQueryPlan> {
            const auto* pkScan = scanForRelation(pkReduction->relationInstance);
            if (!pkScan || pkReduction->keyColumn.empty())
                return std::nullopt;

            auto phaseIndex = phaseIndexByName(domainBitmap->buildPhaseName);
            if (!phaseIndex || *phaseIndex + 1 != lowering->plan.phases.size())
                return std::nullopt;

            const GenericColumnExpr pkColumnExpr = makePkColumnExpr();
            std::map<std::string, DomainAggReusePkColumnSpec> groupByDisplay;
            int outIdx = 0;
            for (size_t gi = 0; gi < aggregate->groupBy.size(); ++gi) {
                auto* col = aggregate->groupBy[gi]
                    ? std::get_if<GenericColumnExpr>(
                          &aggregate->groupBy[gi]->node)
                    : nullptr;
                if (!col) return std::nullopt;

                const std::string display =
                    groupDisplayNameForAggregate(*aggregate, gi);
                const bool isString = col->type.type == DataType::CHAR_FIXED;
                DomainAggReusePkColumnSpec spec;
                spec.output = GenericMatColumnDesc{
                    display,
                    "d_ir_multi_reuse_pk_out_" + std::to_string(outIdx++) +
                        "_" + sanitizeIdentifier(display),
                    isString ? "char" : metalTypeForType(col->type),
                    isString ? std::max(1, col->type.fixedWidth) : 0};
                spec.sourceMetalType = metalTypeForType(col->type);

                if (joinEqForPk.equivalent(*col, pkColumnExpr)) {
                    spec.kind = DomainAggReusePkSourceKind::DomainKey;
                } else if (col->relationInstance.value ==
                           pkReduction->relationInstance) {
                    spec.kind = DomainAggReusePkSourceKind::PkRowColumn;
                    spec.sourceTable = pkScan->table;
                    spec.sourceColumn = col->column;
                } else {
                    auto relIt =
                        lowering->carryMap.find(col->relationInstance.value);
                    if (relIt == lowering->carryMap.end())
                        return std::nullopt;
                    auto colIt = relIt->second.find(col->column);
                    if (colIt == relIt->second.end())
                        return std::nullopt;
                    const auto& carry = colIt->second;

                    const auto* carryInst =
                        relationInstanceById(carry.column.relationInstance.value);
                    const auto* carryRel = carryInst
                        ? ir.findRelation(carryInst->relation)
                        : nullptr;
                    if (!carryRel || carryRel->primaryKeyColumn.empty())
                        return std::nullopt;

                    const GenericJoinDomainBitmapInfo* carryBitmap = nullptr;
                    for (const auto& bitmap : lowering->domainBitmaps) {
                        if (bitmap.relationInstance ==
                                carry.column.relationInstance.value &&
                            bitmap.keyColumn == carryRel->primaryKeyColumn) {
                            carryBitmap = &bitmap;
                            break;
                        }
                    }
                    if (!carryBitmap) return std::nullopt;

                    GenericColumnExpr carriedPk;
                    carriedPk.relationInstance =
                        carry.column.relationInstance;
                    carriedPk.table = carryInst->baseName;
                    carriedPk.alias = carryInst->alias;
                    carriedPk.column = carryRel->primaryKeyColumn;

                    std::optional<GenericColumnExpr> pkSideKey;
                    for (const auto& equivalent :
                         joinEqForPk.columnsEquivalentTo(carriedPk)) {
                        if (equivalent.relationInstance.value ==
                            pkReduction->relationInstance) {
                            pkSideKey = equivalent;
                            break;
                        }
                    }
                    if (!pkSideKey) return std::nullopt;

                    const std::string carryMetalType =
                        metalTypeForType(carry.column.type);
                    const bool carryIsString =
                        carry.column.type.type == DataType::CHAR_FIXED;
                    if (!carryIsString && carryMetalType != "int" &&
                        carryMetalType != "uint") {
                        return std::nullopt;
                    }

                    spec.kind = DomainAggReusePkSourceKind::CarryByPkColumn;
                    spec.sourceMetalType = carryMetalType;
                    spec.carryBuffer =
                        domainCarryStorageBufferName(*carryBitmap,
                                                     carry.column);
                    spec.carryDomain = carryBitmap->keyDomain;
                    spec.carryKeyColumn = pkSideKey->column;
                    spec.carryKeyMetalType = metalTypeForType(pkSideKey->type);
                    spec.sourceIsStringRowRef = carryIsString;
                    if (carryIsString) {
                        spec.stringSourceTable = carry.column.table;
                        spec.stringSourceColumn = carry.column.column;
                    }
                }
                groupByDisplay[display] = std::move(spec);
            }

            const std::string aggregateDisplay = projection.name.empty()
                ? "agg_0"
                : projection.name;
            std::vector<DomainAggReusePkColumnSpec> orderedGroups;
            if (!aggregate->outputOrder.empty()) {
                bool seenAggregate = false;
                for (const auto& name : aggregate->outputOrder) {
                    if (name == aggregateDisplay) {
                        seenAggregate = true;
                        continue;
                    }
                    if (seenAggregate) return std::nullopt;
                    auto it = groupByDisplay.find(name);
                    if (it == groupByDisplay.end()) return std::nullopt;
                    orderedGroups.push_back(it->second);
                }
            } else {
                for (size_t gi = 0; gi < aggregate->groupBy.size(); ++gi) {
                    const std::string display =
                        groupDisplayNameForAggregate(*aggregate, gi);
                    auto it = groupByDisplay.find(display);
                    if (it == groupByDisplay.end()) return std::nullopt;
                    orderedGroups.push_back(it->second);
                }
            }

            GenericMatColumnDesc aggregateOutput{
                aggregateDisplay,
                "d_ir_multi_reuse_pk_out_" + std::to_string(outIdx++) + "_" +
                    sanitizeIdentifier(aggregateDisplay),
                "float"};

            std::vector<GenericMatColumnDesc> outputCols;
            outputCols.reserve(orderedGroups.size() + 1);
            for (const auto& col : orderedGroups)
                outputCols.push_back(col.output);
            outputCols.push_back(aggregateOutput);

            lowering->plan.phases.erase(
                lowering->plan.phases.begin() +
                static_cast<std::ptrdiff_t>(*phaseIndex));

            const std::string pkTag = sanitizeIdentifier(
                pkScan->alias.empty() ? pkScan->table : pkScan->alias);
            const std::string pkRowLookup =
                "d_ir_multi_reuse_pk_row_" + pkTag;
            auto rowLookupPipe = std::make_unique<MetalArrayStore>(
                makeAutoScan(pkScan->table, idxVar), pkRowLookup,
                pkReduction->keyColumn + "[" + idxVar + "]",
                "(int)" + idxVar, "int", reuseAgg->keyDomain, 0xFF);
            appendPhase(lowering->plan,
                        "GENERIC_ir_multi_table_pk_row_lookup_" + pkTag,
                        std::move(rowLookupPipe));

            const std::string reuseCounter =
                "d_ir_multi_reuse_result_count";
            const std::string reuseRows = "n_ir_multi_reuse_rows";
            auto& phase = appendPhase(
                lowering->plan, "GENERIC_ir_multi_table_in_agg_reuse_compact",
                std::make_unique<MetalDomainAggReusePkLookupMaterialize>(
                    pkScan->table, pkRowLookup, reuseAgg->keyDomain,
                    reuseAgg->aggBuffer, reuseAgg->keyListBuffer,
                    reuseAgg->keyListCountBuffer, reuseCounter,
                    std::move(orderedGroups), aggregateOutput));
            attachMaterializedCountHook(phase, reuseCounter, reuseRows);

            if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
                std::string sortError;
                if (!appendBestGenericGpuOrder(
                        lowering->plan, "ir_multi_in_agg_reuse",
                        reuseRows, reuseAgg->keyDomain, outputCols,
                        sortSpec, &sortError)) {
                    return std::nullopt;
                }
            }

            lowering->plan.costTraces.push_back(
                "cost[ir_multi_in_agg_reuse]: primary-key row lookup " +
                pkRowLookup + " replaces tail domain build " +
                domainBitmap->buildPhaseName);
            return std::move(lowering->plan);
        };

        if (auto pkLookupPlan = tryPkRowLookupReuse())
            return pkLookupPlan;

        std::map<std::string, DomainAggReuseColumnSpec> groupByDisplay;
        int outIdx = 0;
        for (size_t gi = 0; gi < aggregate->groupBy.size(); ++gi) {
            auto* col = aggregate->groupBy[gi]
                ? std::get_if<GenericColumnExpr>(&aggregate->groupBy[gi]->node)
                : nullptr;
            if (!col) return std::nullopt;

            const std::string display =
                groupDisplayNameForAggregate(*aggregate, gi);
            DomainAggReuseColumnSpec spec;
            const bool isString = col->type.type == DataType::CHAR_FIXED;
            spec.output = GenericMatColumnDesc{
                display,
                "d_ir_multi_reuse_out_" + std::to_string(outIdx++) + "_" +
                    sanitizeIdentifier(display),
                isString ? "char" : metalTypeForType(col->type),
                isString ? std::max(1, col->type.fixedWidth) : 0};
            spec.sourceMetalType = metalTypeForType(col->type);

            if (col->relationInstance.value == domainBitmap->relationInstance &&
                col->column == domainBitmap->keyColumn) {
                spec.fromDomainKey = true;
            } else {
                auto relIt = lowering->carryMap.find(col->relationInstance.value);
                if (relIt == lowering->carryMap.end()) return std::nullopt;
                auto colIt = relIt->second.find(col->column);
                if (colIt == relIt->second.end()) return std::nullopt;
                const auto& carry = colIt->second;
                spec.sourceBuffer =
                    domainCarryStorageBufferName(*domainBitmap,
                                                 carry.column);
                if (isString) {
                    spec.sourceIsStringRowRef = true;
                    spec.stringSourceTable = carry.column.table;
                    spec.stringSourceColumn = carry.column.column;
                }
            }
            groupByDisplay[display] = std::move(spec);
        }

        const std::string aggregateDisplay = projection.name.empty()
            ? "agg_0"
            : projection.name;
        std::vector<DomainAggReuseColumnSpec> orderedGroups;
        if (!aggregate->outputOrder.empty()) {
            bool seenAggregate = false;
            for (const auto& name : aggregate->outputOrder) {
                if (name == aggregateDisplay) {
                    seenAggregate = true;
                    continue;
                }
                if (seenAggregate) return std::nullopt;
                auto it = groupByDisplay.find(name);
                if (it == groupByDisplay.end()) return std::nullopt;
                orderedGroups.push_back(it->second);
            }
        } else {
            for (size_t gi = 0; gi < aggregate->groupBy.size(); ++gi) {
                const std::string display =
                    groupDisplayNameForAggregate(*aggregate, gi);
                auto it = groupByDisplay.find(display);
                if (it == groupByDisplay.end()) return std::nullopt;
                orderedGroups.push_back(it->second);
            }
        }

        GenericMatColumnDesc aggregateOutput{
            aggregateDisplay,
            "d_ir_multi_reuse_out_" + std::to_string(outIdx++) + "_" +
                sanitizeIdentifier(aggregateDisplay),
            "float"};

        std::vector<GenericMatColumnDesc> outputCols;
        outputCols.reserve(orderedGroups.size() + 1);
        for (const auto& col : orderedGroups)
            outputCols.push_back(col.output);
        outputCols.push_back(aggregateOutput);

        const std::string reuseCounter = "d_ir_multi_reuse_result_count";
        const std::string reuseRows = "n_ir_multi_reuse_rows";
        auto& phase = appendPhase(
            lowering->plan, "GENERIC_ir_multi_table_in_agg_reuse_compact",
            std::make_unique<MetalDomainAggReuseMaterialize>(
                domainBitmap->bitmapName, domainBitmap->keyDomain,
                reuseAgg->aggBuffer, reuseAgg->keyListBuffer,
                reuseAgg->keyListCountBuffer, reuseCounter,
                std::move(orderedGroups),
                aggregateOutput));
        attachMaterializedCountHook(phase, reuseCounter, reuseRows);

        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            std::string sortError;
            if (!appendBestGenericGpuOrder(
                    lowering->plan, "ir_multi_in_agg_reuse", reuseRows,
                    domainBitmap->keyDomain, outputCols, sortSpec,
                    &sortError)) {
                return std::nullopt;
            }
        }

        return std::move(lowering->plan);
    };

    if (auto reused = tryInSubAggDomainReuse())
        return reused;

	    auto makeAggregationCostInput =
	        [&](std::vector<GenericMatColumnDesc> outputColumns) {
	            GenericAggregationCostInput input;
	            input.tag = "ir_multi_table_group";
	            input.inputRowsExpr = lowering->outputSize;
	            input.outputRowsExpr = groupOutputBoundExpr;
	            input.materializedInputColumns = materializedCols;
	            input.outputColumns = std::move(outputColumns);
	            input.groupKeyCount = aggregate->groupBy.size();
	            input.aggregateCount = aggregate->aggregates.size();
	            input.sortLimit = sortSpec.limit;
	            return input;
	        };
	    auto appendAggregationCostChoice =
	        [&](const GenericAggregationCostInput& input,
	            const std::string& chosenCandidate) {
                auto choice = chooseMultiTableAggregationPlan(
                    input, chosenCandidate);
	            appendGenericCostDecisionTrace(
	                lowering->plan, choice.trace);
	        };

	    bool hasCountDistinctAgg = false;
    for (const auto& projection : aggregate->aggregates) {
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (agg && agg->func == AggFunc::COUNT_DISTINCT) {
            hasCountDistinctAgg = true;
            break;
        }
    }
    if (hasCountDistinctAgg) {
        auto& matPhase = ensureMaterializePhase();
        bool countDistinctGroupApplied = false;
	        if (!tryAppendMaterializedCountDistinctGroup(
	                lowering->plan, matPhase, *aggregate, groupSpec, materializedCols,
	                sortSpec, resultCounter, lowering->outputSize,
	                groupOutputBoundExpr, &countDistinctGroupApplied, error)) {
	            return std::nullopt;
	        }
	        if (countDistinctGroupApplied) {
	            auto costInput = makeAggregationCostInput(materializedCols);
	            costInput.materializedHashAvailable = false;
	            costInput.countDistinctAvailable = true;
	            costInput.aggregateSlots =
	                static_cast<int>(std::max<size_t>(1, aggregate->aggregates.size()));
	            appendAggregationCostChoice(costInput,
	                                        "materialized_count_distinct");
	            return std::move(lowering->plan);
	        }
	    }

    struct FdGroupPlanCandidate {
        bool available = false;
        PrimaryKeyGroupReduction reduction;
        GenericMatColumnDesc keyColumn;
        std::vector<IrPendingAgg> pending;
        int valuesPerBucket = 0;
        std::vector<GenericMatColumnDesc> outputColumns;
        std::vector<GenericMatColumnDesc> narrowOutputColumns;
        FdTopKLateMaterializationChoice lateTopKChoice;
        bool lateTopKMaterialize = false;
    };
    FdGroupPlanCandidate fdCandidate;

    if (!aggregateNeedsHashGroupOutput(*aggregate) && !aggregate->having) {
        auto pkReduction = primaryKeyGroupReduction(
            ir, *aggregate, joinEqForPk, lowering->carryMap);
        if (pkReduction && !pkReduction->keyDomainExpr.empty()) {
            bool fdDirectOk = true;
            const std::string keyDisplay =
                groupDisplayNameForAggregate(*aggregate, pkReduction->groupIndex);
            const auto* keyCol = findMaterializedColumn(materializedCols, keyDisplay);
            if (!keyCol || keyCol->stringLen > 0 ||
                (keyCol->metalType != "int" && keyCol->metalType != "uint")) {
                fdDirectOk = false;
            }

            std::vector<IrPendingAgg> fdPending;
            int fdValuesPerBucket = 0;
            if (fdDirectOk) {
                for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
                    const auto& projection = aggregate->aggregates[i];
                    auto* agg = projection.expr
                        ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
                        : nullptr;
                    if (!agg) {
                        fdDirectOk = false;
                        break;
                    }
                    const std::string displayName = projection.name.empty()
                        ? "agg_" + std::to_string(i)
                        : projection.name;

                    if (agg->func == AggFunc::COUNT_DISTINCT ||
                        agg->func == AggFunc::MIN ||
                        agg->func == AggFunc::MAX) {
                        fdDirectOk = false;
                        break;
                    }

                    if (agg->func == AggFunc::COUNT) {
                        IrPendingAgg out;
                        out.displayName = displayName;
                        out.offset = fdValuesPerBucket++;
                        out.valueExpr = "1u";
                        out.funcName = "COUNT";
                        fdPending.push_back(std::move(out));
                        continue;
                    }

                    const auto* matCol =
                        findMaterializedColumn(materializedCols, displayName);
                    if (!agg->arg || !matCol || matCol->stringLen > 0) {
                        fdDirectOk = false;
                        break;
                    }
                    std::string valueExpr = materializedValueAt(*matCol, "_r");
                    if (agg->func == AggFunc::AVG) {
                        const int fixedScale = matCol->scaleDown;
                        IrPendingAgg sum;
                        sum.displayName = displayName;
                        sum.offset = fdValuesPerBucket;
                        sum.valueExpr = valueExpr;
                        if (agg->arg->type.type == DataType::FLOAT && fixedScale > 0) {
                            sum.valueExpr = scaledLongExpr(valueExpr, fixedScale);
                            sum.isLongPair = true;
                            sum.scaleDown = -fixedScale;
                            fdValuesPerBucket += 2;
                        } else if (agg->arg->type.type == DataType::FLOAT) {
                            sum.isFloatSum = true;
                            sum.scaleDown = -1;
                            fdValuesPerBucket += 1;
                        } else {
                            sum.isLongPair = true;
                            sum.scaleDown = -1;
                            fdValuesPerBucket += 2;
                        }
                        sum.funcName = "AVG";
                        sum.innerColumn = innerColumnName(agg->arg);
                        fdPending.push_back(std::move(sum));

                        IrPendingAgg cnt;
                        cnt.displayName = displayName + "_cnt";
                        cnt.offset = fdValuesPerBucket++;
                        cnt.valueExpr = "1u";
                        cnt.funcName = "AVG";
                        fdPending.push_back(std::move(cnt));
                        continue;
                    }

                    if (agg->func == AggFunc::SUM) {
                        IrPendingAgg out;
                        out.displayName = displayName;
                        out.offset = fdValuesPerBucket;
                        out.valueExpr = valueExpr;
                        out.funcName = "SUM";
                        out.innerColumn = innerColumnName(agg->arg);
                        if (agg->arg->type.type == DataType::FLOAT &&
                            matCol->scaleDown > 0) {
                            out.valueExpr = scaledLongExpr(valueExpr, matCol->scaleDown);
                            out.isLongPair = true;
                            out.scaleDown = matCol->scaleDown;
                            fdValuesPerBucket += 2;
                        } else if (agg->arg->type.type == DataType::FLOAT) {
                            out.isFloatSum = true;
                            fdValuesPerBucket += 1;
                        } else {
                            out.isLongPair = true;
                            fdValuesPerBucket += 2;
                        }
                        fdPending.push_back(std::move(out));
                        continue;
                    }

                    fdDirectOk = false;
                    break;
                }
            }

            std::vector<GenericMatColumnDesc> fdOutputCols;
            if (fdDirectOk && fdValuesPerBucket > 0) {
                fdOutputCols = fdKeyedGroupOutputColumns(
                    "ir_multi_fd_group", groupSpec, materializedCols, fdPending);
                for (const auto& display : groupSpec.outputColumns) {
                    if (!findMaterializedColumn(fdOutputCols, display)) {
                        fdDirectOk = false;
                        break;
                    }
                }
            }

            if (fdDirectOk && fdValuesPerBucket > 0) {
                auto narrowFdOutputCols = fdKeyedGroupTopKNarrowColumns(
                    "ir_multi_fd_group_topk", fdOutputCols, sortSpec);
                auto lateTopKChoice = chooseFdTopKLateMaterialization(
                    sortSpec, fdOutputCols, narrowFdOutputCols,
                    pkReduction->outputBoundExpr,
                    pkReduction->keyDomainExpr, "ir_multi_fd_group_topk");
                fdCandidate.available = true;
                fdCandidate.reduction = *pkReduction;
                fdCandidate.keyColumn = *keyCol;
                fdCandidate.pending = std::move(fdPending);
                fdCandidate.valuesPerBucket = fdValuesPerBucket;
                fdCandidate.outputColumns = std::move(fdOutputCols);
                fdCandidate.narrowOutputColumns = std::move(narrowFdOutputCols);
                fdCandidate.lateTopKChoice = std::move(lateTopKChoice);
                fdCandidate.lateTopKMaterialize =
                    fdCandidate.lateTopKChoice.useLateMaterialization;
            }
        }
    }

    auto fdCandidateName = [&]() {
        return fdCandidate.lateTopKMaterialize ? std::string("fd_keyed_late_topk")
                                               : std::string("fd_keyed_group");
    };
    auto fdCostCandidate = [&]() {
        GenericAggregationCandidateCostInput candidate;
        candidate.name = fdCandidateName();
        candidate.available = fdCandidate.available;
        candidate.reason = fdCandidate.available
            ? ""
            : "finite-domain keyed group candidate is not applicable";
        candidate.outputColumns = fdCandidate.outputColumns;
        candidate.aggregateSlots = fdCandidate.valuesPerBucket;
        candidate.denseBucketsExpr = fdCandidate.reduction.keyDomainExpr;
        candidate.fdLateTopK = fdCandidate.lateTopKMaterialize;
        return candidate;
    };
    auto emitFdCandidate = [&]() -> std::optional<MetalQueryPlan> {
        if (!fdCandidate.available) return std::nullopt;
        const std::string rowsSym = "ir_multi_fd_group_rows";
        const std::string bucketCountSymbol = "n_ir_multi_fd_group_buckets";
        const std::string stateBuffer = "d_ir_multi_fd_group_state";
        const std::string repRowBuffer = "d_ir_multi_fd_group_rep_row";
        const std::string aggBuffer = "d_ir_multi_fd_group_aggs";
        const std::string compactCounter = "d_ir_multi_fd_group_count";

        appendGenericCostDecisionTrace(lowering->plan,
                                       fdCandidate.lateTopKChoice.trace);
        auto& matPhase = ensureMaterializePhase();
        attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
        appendPhase(lowering->plan,
            "GENERIC_ir_multi_table_fd_group_build",
            std::make_unique<MetalFdKeyedGroupBuild>(
                rowsSym, fdCandidate.reduction.keyDomainExpr, bucketCountSymbol,
                stateBuffer, repRowBuffer, aggBuffer, fdCandidate.keyColumn,
                materializedCols, fdCandidate.pending,
                fdCandidate.valuesPerBucket));

        const auto& compactOutputs = fdCandidate.lateTopKMaterialize
            ? fdCandidate.narrowOutputColumns
            : fdCandidate.outputColumns;
        auto& compactPhase = appendPhase(
            lowering->plan,
            fdCandidate.lateTopKMaterialize
                ? "GENERIC_ir_multi_table_fd_group_topk_compact"
                : "GENERIC_ir_multi_table_fd_group_compact",
            std::make_unique<MetalFdKeyedGroupCompact>(
                fdCandidate.reduction.keyDomainExpr, bucketCountSymbol,
                fdCandidate.reduction.outputBoundExpr, stateBuffer, repRowBuffer,
                aggBuffer, compactCounter,
                materializedCols, groupSpec.keyColumns, fdCandidate.pending,
                compactOutputs, fdCandidate.valuesPerBucket));

        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            const std::string orderTag = fdCandidate.lateTopKMaterialize
                ? "ir_multi_fd_group_topk"
                : "ir_multi_fd_group";
            const std::string sortRowsSym =
                "n_gpu_sort_" + orderTag + "_rows";
            attachMaterializedCountHook(compactPhase, compactCounter,
                                        sortRowsSym);
            if (!appendBestGenericGpuOrder(
                    lowering->plan, orderTag, sortRowsSym,
                    fdCandidate.reduction.outputBoundExpr, compactOutputs,
                    sortSpec, error)) {
                return std::nullopt;
            }
            if (fdCandidate.lateTopKMaterialize) {
                const auto* bucketCol = findMaterializedColumn(
                    compactOutputs, kFdHiddenBucketDisplay);
                if (!bucketCol || !lowering->plan.gpuSort) {
                    return fail(error,
                        "IR multi-table FD top-k lowerer: missing narrow bucket or GPU order.");
                }
                const std::string finalCounter =
                    "d_ir_multi_fd_group_topk_result_count";
                appendPhase(lowering->plan,
                    "GENERIC_ir_multi_table_fd_group_topk_gather",
                    std::make_unique<MetalFdKeyedGroupTopKGather>(
                        lowering->plan.gpuSort->sortedIndexBuffer,
                        sortRowsSym, sortSpec.limit, *bucketCol,
                        fdCandidate.reduction.keyDomainExpr, repRowBuffer,
                        aggBuffer, finalCounter, materializedCols,
                        groupSpec.keyColumns, fdCandidate.pending,
                        fdCandidate.outputColumns,
                        fdCandidate.valuesPerBucket),
                    256);
                lowering->plan.gpuSort.reset();
            }
        }
        return std::move(lowering->plan);
    };

    const std::string groupTag = "ir_multi_table_group";
    auto makeHashOutputColumns = [&]() {
        GenericGpuGroupSpec spec;
        spec.tag = groupTag;
        spec.inputColumns = materializedCols;
        spec.groupBy = groupSpec;
        return genericGpuGroupOutputColumns(spec);
    };
    auto hashCostCandidate =
        [&](const std::vector<GenericMatColumnDesc>& hashOutputColumns) {
            GenericAggregationCandidateCostInput candidate;
            candidate.name = "materialized_hash_group";
            candidate.outputColumns = hashOutputColumns;
            candidate.aggregateSlots = static_cast<int>(
                std::max<size_t>(1, aggregate->aggregates.size()));
            return candidate;
        };
    auto chooseAggregationCandidate =
        [&](std::vector<GenericAggregationCandidateCostInput> candidates) {
            GenericAggregationCostInput costInput =
                makeAggregationCostInput(candidates.empty()
                    ? std::vector<GenericMatColumnDesc>{}
                    : candidates.front().outputColumns);
            costInput.candidates = std::move(candidates);
            auto choice = chooseMultiTableAggregationPlan(costInput);
            appendGenericCostDecisionTrace(lowering->plan, choice.trace);
            return choice;
        };

    bool aggregationChoiceMade = false;
    if (!aggregate->groupBy.empty()) {
        bool directOk = true;
        int totalBuckets = 1;
        const std::string dynamicBucketCountSymbol =
            "n_ir_multi_direct_group_buckets";
        std::string bucketCountExpr;
        bool dynamicDomain = false;
        std::vector<IrGroupKeyDesc> denseKeys;
        denseKeys.reserve(aggregate->groupBy.size());
        struct DirectStringKeyMapSpec {
            std::string sourceTable;
            std::string sourceColumn;
            int width = 0;
            std::string mapBuffer;
            std::vector<std::string> domain;
        };
        std::vector<DirectStringKeyMapSpec> directStringKeyMaps;
        std::vector<MetalMaterializedRangeScan::ExtraBuffer> directExtraBuffers;
        std::map<std::string, std::string> directStringKeyMapBySignature;
        bool directInputFusionOk = true;
        int directCarriedStringRowRefRewrites = 0;
        std::vector<std::pair<std::string, std::string>> directInputRewrites;

        auto canFuseDirectScalarExpr = [&](const GenericExprPtr& expr) {
            if (!expr || !materializeExprSupported(expr)) return false;
            if (exprNeedsCarriedString(expr, lowering->carryMap)) return false;
            return expr->type.type != DataType::CHAR_FIXED;
        };
        auto carriedStringRowRefExpr = [&](const GenericExprPtr& expr)
                -> std::optional<std::string> {
            if (!expr) return std::nullopt;
            const auto* col = std::get_if<GenericColumnExpr>(&expr->node);
            if (!col || col->type.type != DataType::CHAR_FIXED)
                return std::nullopt;
            auto relIt = lowering->carryMap.find(col->relationInstance.value);
            if (relIt == lowering->carryMap.end())
                return std::nullopt;
            auto colIt = relIt->second.find(col->column);
            if (colIt == relIt->second.end() ||
                colIt->second.rowVarName.empty()) {
                return std::nullopt;
            }
            return colIt->second.rowVarName;
        };

        auto finiteStringKeyMapSignature = [](
                const GenericMatColumnDesc& col,
                const std::vector<std::string>& domain) {
            std::string sig = col.stringSourceTable + "." +
                              col.stringSourceColumn + ":" +
                              std::to_string(col.stringLen);
            for (const auto& value : domain) {
                sig += "#" + std::to_string(value.size()) + ":" + value;
            }
            return sig;
        };
        auto ensureFiniteStringKeyMap = [&](
                const GenericMatColumnDesc& col,
                const std::vector<std::string>& domain,
                const std::string& displayName) -> std::string {
            if (col.stringSourceTable.empty() ||
                col.stringSourceColumn.empty() ||
                col.stringLen <= 0 || domain.empty()) {
                return {};
            }
            const std::string sig = finiteStringKeyMapSignature(col, domain);
            auto existing = directStringKeyMapBySignature.find(sig);
            if (existing != directStringKeyMapBySignature.end())
                return existing->second;

            const std::string mapName =
                "d_ir_multi_direct_string_keymap_" +
                std::to_string(directStringKeyMaps.size()) + "_" +
                sanitizeIdentifier(displayName);
            directStringKeyMapBySignature.emplace(sig, mapName);
            directStringKeyMaps.push_back(DirectStringKeyMapSpec{
                col.stringSourceTable, col.stringSourceColumn, col.stringLen,
                mapName, domain});
            directExtraBuffers.push_back({mapName, "const uint"});
            return mapName;
        };

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
            key.stride = 1;
            const std::string raw = materializedValueAt(*matCol, idxVar);
            if (directInputFusionOk && canFuseDirectScalarExpr(group)) {
                directInputRewrites.push_back({
                    raw,
                    genericExprToMetalWithCarryMap(
                        group, idxVar, lowering->carryMap)});
            } else if (directInputFusionOk && matCol->stringRowRef) {
                if (auto rowRefExpr = carriedStringRowRefExpr(group)) {
                    directInputRewrites.push_back({raw, *rowRefExpr});
                    ++directCarriedStringRowRefRewrites;
                } else {
                    directInputFusionOk = false;
                }
            } else {
                directInputFusionOk = false;
            }
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
            } else if (auto datePartBounds = datePartBoundsForPredicate(
                           fd ? fd->predicate : GenericPredicatePtr{}, group);
                       (group->type.type == DataType::INT ||
                        group->type.type == DataType::DATE) &&
                       datePartBounds) {
                int64_t minValue = datePartBounds->first;
                int64_t maxValue = datePartBounds->second;
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
                if (domain && !domain->empty() && matCol->stringLen > 0) {
                    key.stringMap = *domain;
                    key.stringLen = std::max(matCol->stringLen,
                                             maxStringLen(key.stringMap));
                    key.numValues = static_cast<int>(key.stringMap.size());
                    if (matCol->stringRowRef) {
                        const std::string mapName = ensureFiniteStringKeyMap(
                            *matCol, key.stringMap, displayName);
                        if (mapName.empty()) {
                            directOk = false;
                            break;
                        }
                        key.keyExpr = mapName + "[" + raw + "]";
                    } else {
                        key.keyExpr = fixedStringDomainBucketExpr(
                            matCol->bufferName, idxVar, matCol->stringLen,
                            key.stringMap);
                    }
                    if (key.keyExpr.empty() || key.numValues <= 0) {
                        directOk = false;
                        break;
                    }
                } else if (matCol->stringRowRef &&
                           matCol->stringLen > 0 && matCol->metalType == "uint" &&
                           !matCol->stringSourceTable.empty() &&
                           !matCol->stringSourceColumn.empty()) {
                    if (auto staticRows = staticBaseTableRowBound(
                            ir, matCol->stringSourceTable);
                        staticRows && *staticRows <= 4096) {
                        key.numValues = *staticRows;
                        key.keyExpr = "clamp((int)(" + raw + "), 0, " +
                                      std::to_string(*staticRows - 1) + ")";
                    } else {
                        key.numValuesExpr = tableSizeName(matCol->stringSourceTable);
                        key.keyExpr = raw;
                    }
                    key.stringRowRef = true;
                    key.stringLen = matCol->stringLen;
                    key.stringSourceTable = matCol->stringSourceTable;
                    key.stringSourceColumn = matCol->stringSourceColumn;
                } else {
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
            } else if (col && (col->type.type == DataType::INT ||
                        col->type.type == DataType::DATE) &&
                       aggregate->groupBy.size() == 1 &&
                       !col->keyDomainSymbol.empty()) {
                key.numValuesExpr = col->keyDomainSymbol;
                key.keyExpr = raw;
            } else {
                directOk = false;
                break;
            }

            if (key.numValuesExpr.empty()) {
                if (key.numValues <= 0) {
                    directOk = false;
                    break;
                }
            }
            denseKeys.push_back(std::move(key));
        }

        auto assignDenseKeyStrides = [&]() -> bool {
            int staticProduct = 1;
            int dynamicIndex = -1;
            for (size_t i = 0; i < denseKeys.size(); ++i) {
                auto& key = denseKeys[i];
                if (!key.numValuesExpr.empty()) {
                    if (dynamicIndex >= 0) return false;
                    dynamicIndex = static_cast<int>(i);
                    continue;
                }
                if (key.numValues <= 0) return false;
                if (staticProduct > 4096 / key.numValues) return false;
                key.stride = staticProduct;
                staticProduct *= key.numValues;
            }

            if (dynamicIndex >= 0) {
                auto& dynamicKey = denseKeys[(size_t)dynamicIndex];
                if (staticProduct <= 0) return false;
                dynamicDomain = true;
                dynamicKey.stride = staticProduct;
                bucketCountExpr = "(" + dynamicKey.numValuesExpr + ")";
                if (staticProduct != 1)
                    bucketCountExpr += " * " + std::to_string(staticProduct);
                dynamicKey.numValuesExpr = staticProduct == 1
                    ? dynamicBucketCountSymbol
                    : "(" + dynamicBucketCountSymbol + " / " +
                          std::to_string(staticProduct) + "u)";
                totalBuckets = staticProduct;
            } else {
                dynamicDomain = false;
                totalBuckets = staticProduct;
                bucketCountExpr = std::to_string(totalBuckets);
            }
            return totalBuckets > 0;
        };

        if (directOk)
            directOk = assignDenseKeyStrides();

        std::vector<IrPendingAgg> pending;
        std::vector<int> aggregatePendingIndex(aggregate->aggregates.size(), -1);
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
                const std::string outputFunc = aggregateOutputFuncFor(
                    *aggregate, i, agg->func);

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
                    aggregatePendingIndex[i] = static_cast<int>(pending.size());
                    pending.push_back(std::move(out));
                    continue;
                }

                const auto* matCol = findMaterializedColumn(materializedCols, displayName);
                if (!agg->arg || !matCol || matCol->stringLen > 0) {
                    directOk = false;
                    break;
                }
                std::string valueExpr = materializedValueAt(*matCol, idxVar);
                if (directInputFusionOk && canFuseDirectScalarExpr(agg->arg)) {
                    directInputRewrites.push_back({
                        valueExpr,
                        genericExprToMetalWithCarryMap(
                            agg->arg, idxVar, lowering->carryMap)});
                } else {
                    directInputFusionOk = false;
                }
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
                    aggregatePendingIndex[i] = static_cast<int>(pending.size());
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
                    out.funcName = outputFunc;
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
                    aggregatePendingIndex[i] = static_cast<int>(pending.size());
                    pending.push_back(std::move(out));
                    continue;
                }

                directOk = false;
                break;
            }
        }

        KeyedCompactHavingSpec havingSpec;
        if (directOk && !pending.empty()) {
            auto configureDirectHavingSlot = [&](int aggIdx,
                                                 bool scalar) -> bool {
                if (aggIdx < 0 ||
                    aggIdx >= static_cast<int>(aggregatePendingIndex.size())) {
                    return false;
                }
                const int pendingIdx = aggregatePendingIndex[(size_t)aggIdx];
                if (pendingIdx < 0 ||
                    pendingIdx >= static_cast<int>(pending.size())) {
                    return false;
                }
                const auto& p = pending[(size_t)pendingIdx];
                if (p.funcName == "AVG")
                    return false;
                if (scalar) {
                    havingSpec.scalarAggOffset = p.offset;
                    havingSpec.scalarAggIsLongPair = p.isLongPair;
                    havingSpec.scalarAggIsFloatSum = p.isFloatSum;
                    havingSpec.scalarAggScaleDown = p.scaleDown;
                    havingSpec.scalarTotalBuffer =
                        "d_ir_multi_direct_group_having_total";
                    havingSpec.scalarTotalParam =
                        "s_ir_multi_direct_group_having_total";
                    havingSpec.scalarCompareOp = groupSpec.havingScalarCompareOp;
                    havingSpec.scalarMultiplier = groupSpec.havingMultiplier;
                } else {
                    havingSpec.compareAggOffset = p.offset;
                    havingSpec.compareAggIsLongPair = p.isLongPair;
                    havingSpec.compareAggIsFloatSum = p.isFloatSum;
                    havingSpec.compareAggScaleDown = p.scaleDown;
                    havingSpec.compareOp = groupSpec.havingCompareOp;
                    havingSpec.compareValue = groupSpec.havingCompareValue;
                }
                return true;
            };
            if (aggregate->having) {
                bool havingOk = true;
                if (groupSpec.havingAggIdx >= 0)
                    havingOk = configureDirectHavingSlot(groupSpec.havingAggIdx, true);
                if (havingOk && groupSpec.havingCompareAggIdx >= 0 &&
                    !groupSpec.havingCompareOp.empty()) {
                    havingOk = configureDirectHavingSlot(
                        groupSpec.havingCompareAggIdx, false);
                }
                directOk = havingOk;
            }
        }

        DenseGroupCostChoice denseChoice;
        bool haveDenseChoice = false;
        if (directOk && !pending.empty()) {
            denseChoice = chooseDenseGroupPlan(
                denseKeys, pending, totalBuckets, dynamicDomain, havingSpec,
                "ir_multi_table_direct_group");
            haveDenseChoice = true;
            if (!denseChoice.useDense)
                directOk = false;
        }

        if (directOk && !pending.empty()) {
            const bool pipelineInputAvailable = directInputFusionOk;
            const bool activeBucketCompactionApplicable =
                !havingSpec.scalarTotalBuffer.empty() ||
                havingSpec.compareAggOffset >= 0;
            ActiveBucketCompactionCostChoice activeChoice;
            bool haveActiveChoice = false;
            bool useActiveBucketCompaction = false;
            if (activeBucketCompactionApplicable) {
                auto activeCostInput = makeAggregationCostInput({});
                activeCostInput.directDenseAvailable = true;
                activeCostInput.aggregateSlots = valuesPerBucket;
                activeCostInput.denseBuckets = dynamicDomain ? 0 : totalBuckets;
                activeCostInput.denseBucketsExpr =
                    dynamicDomain ? bucketCountExpr : "";
                activeCostInput.dynamicDenseDomain = dynamicDomain;
                activeChoice = chooseActiveBucketCompaction(
                    activeCostInput, "ir_multi_table_direct_group_active");
                haveActiveChoice = true;
                useActiveBucketCompaction = activeChoice.useActiveList;
            }
            auto inputModeCost = makeAggregationCostInput({});
            inputModeCost.directDenseAvailable = true;
            inputModeCost.directInputFused = pipelineInputAvailable;
            inputModeCost.directMaterializedAvailable = true;
            inputModeCost.directPipelineCarriedStringRowRefs =
                directCarriedStringRowRefRewrites;
            inputModeCost.directPipelineExtraBuffers =
                static_cast<int>(directExtraBuffers.size());
            inputModeCost.activeBucketCompaction = useActiveBucketCompaction;
            inputModeCost.aggregateSlots = valuesPerBucket;
            inputModeCost.denseBuckets = dynamicDomain ? 0 : totalBuckets;
            inputModeCost.denseBucketsExpr =
                dynamicDomain ? bucketCountExpr : "";
            inputModeCost.dynamicDenseDomain = dynamicDomain;
            auto inputModeChoice = chooseDirectDenseInputMode(
                inputModeCost, "ir_multi_table_direct_group_input");
            const bool fuseDirectInput = inputModeChoice.usePipeline;
            if (fuseDirectInput) {
                for (auto& key : denseKeys) {
                    for (const auto& rewrite : directInputRewrites)
                        replaceAllInPlace(key.keyExpr, rewrite.first,
                                          rewrite.second);
                }
                for (auto& agg : pending) {
                    for (const auto& rewrite : directInputRewrites)
                        replaceAllInPlace(agg.valueExpr, rewrite.first,
                                          rewrite.second);
                }
            }

            auto directCompactOutputColumns =
                [&]() -> std::vector<GenericMatColumnDesc> {
                    std::vector<GenericMatColumnDesc> columns;
                    for (const auto& key : denseKeys) {
                        std::string buf = "d_ir_multi_direct_out_" +
                                          std::to_string(columns.size()) + "_" +
                                          sanitizeIdentifier(key.displayName);
                        const bool isStringKey = !key.charMap.empty() ||
                                                 !key.stringMap.empty() ||
                                                 key.stringRowRef;
                        columns.push_back(GenericMatColumnDesc{
                            key.displayName, buf, isStringKey ? "char" : "int",
                            isStringKey ? key.stringLen : 0});
                    }

                    for (size_t pi = 0; pi < pending.size(); ++pi) {
                        const auto& p = pending[pi];
                        if (p.funcName == "RATIO_DEN" ||
                            p.displayName.rfind("__hidden_", 0) == 0) {
                            continue;
                        }

                        std::string metalType = "uint";
                        int outScale = 0;
                        bool outLongPair = false;
                        if (p.funcName == "RATIO") {
                            metalType = "float";
                            if (pi + 1 < pending.size() &&
                                pending[pi + 1].funcName == "RATIO_DEN") {
                                ++pi;
                            }
                        } else if (p.scaleDown < 0 && pi + 1 < pending.size()) {
                            metalType = "float";
                            ++pi;
                        } else if (p.isLongPair) {
                            outScale = p.scaleDown > 0 ? p.scaleDown : 0;
                            outLongPair = true;
                        } else if (p.isFloatSum || p.isMinMax) {
                            metalType = "float";
                        }

                        std::string buf = "d_ir_multi_direct_out_" +
                                          std::to_string(columns.size()) + "_" +
                                          sanitizeIdentifier(p.displayName);
                        columns.push_back(GenericMatColumnDesc{
                            p.displayName, buf, metalType, 0, outScale, outLongPair});
                    }
                    return columns;
                };

            const int keyedBucketCount = dynamicDomain ? 0 : totalBuckets;
            const std::string directCandidateName = fuseDirectInput
                ? "direct_dense_pipeline"
                : "direct_dense_materialized";
            auto hashOutputColumns = makeHashOutputColumns();
            GenericAggregationCandidateCostInput directCandidate;
            directCandidate.name = directCandidateName;
            directCandidate.outputColumns = directCompactOutputColumns();
            directCandidate.aggregateSlots = valuesPerBucket;
            directCandidate.denseBuckets = keyedBucketCount;
            directCandidate.denseBucketsExpr = dynamicDomain ? bucketCountExpr : "";
            directCandidate.dynamicDenseDomain = dynamicDomain;
            directCandidate.directInputFused = fuseDirectInput;
            directCandidate.directPipelineCarriedStringRowRefs =
                directCarriedStringRowRefRewrites;
            directCandidate.directPipelineExtraBuffers =
                static_cast<int>(directExtraBuffers.size());
            directCandidate.activeBucketCompaction = useActiveBucketCompaction;

            std::vector<GenericAggregationCandidateCostInput> candidates;
            candidates.push_back(hashCostCandidate(hashOutputColumns));
            if (fdCandidate.available)
                candidates.push_back(fdCostCandidate());
            candidates.push_back(directCandidate);
            auto globalChoice = chooseAggregationCandidate(std::move(candidates));
            aggregationChoiceMade = true;
            if (fdCandidate.available &&
                globalChoice.chosenCandidate == fdCandidateName()) {
                return emitFdCandidate();
            }
            if (globalChoice.chosenCandidate != directCandidateName) {
                directOk = false;
            }

            if (!directOk) {
                // Fall through to the materialized hash-group path.
            } else {
                if (haveDenseChoice) {
                    appendGenericCostDecisionTrace(lowering->plan,
                                                   denseChoice.trace);
                }
                if (haveActiveChoice) {
                    appendGenericCostDecisionTrace(lowering->plan,
                                                   activeChoice.trace);
                }
                appendGenericCostDecisionTrace(lowering->plan,
                                               inputModeChoice.trace);

            std::string bucketExpr;
            for (const auto& key : denseKeys) {
                std::string term = "(" + key.keyExpr + ")";
                if (key.stride != 1)
                    term = "(" + term + " * " +
                           std::to_string(key.stride) + ")";
                bucketExpr = bucketExpr.empty()
                    ? term
                    : "(" + bucketExpr + " + " + term + ")";
            }
            bucketCountExpr = dynamicDomain
                ? bucketCountExpr
                : std::to_string(totalBuckets);

            std::unique_ptr<MetalOperator> directInput;
            for (const auto& spec : directStringKeyMaps) {
                appendPhase(lowering->plan,
                    "GENERIC_ir_multi_table_finite_string_keymap_" +
                        sanitizeIdentifier(spec.mapBuffer),
                    std::make_unique<MetalFiniteStringRowRefMapBuild>(
                        spec.sourceTable, spec.sourceColumn, spec.width,
                        spec.mapBuffer, spec.domain),
                    256);
            }
            if (fuseDirectInput) {
                directInput = std::move(lowering->probePipe);
            } else {
                auto& matPhase = ensureMaterializePhase();
                const std::string rowsSym = "ir_multi_direct_group_rows";
                attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
                directInput = std::make_unique<MetalMaterializedRangeScan>(
                    rowsSym, idxVar, materializedCols, directExtraBuffers);
            }
            auto keyed = std::make_unique<MetalKeyedAgg>(
                std::move(directInput), "d_ir_multi_direct_group_aggs", bucketExpr,
                keyedBucketCount, valuesPerBucket,
                bucketCountExpr + " * " + std::to_string(valuesPerBucket));
            if (fuseDirectInput) {
                for (const auto& extra : directExtraBuffers)
                    keyed->addExtraBufferParam(extra.first, extra.second);
            }
	            const std::string activeFlagBuffer =
	                useActiveBucketCompaction
                    ? "d_ir_multi_direct_group_active_flags"
                    : "";
            const std::string activeListBuffer =
                useActiveBucketCompaction
                    ? "d_ir_multi_direct_group_active_list"
                    : "";
            const std::string activeCounterBuffer =
                useActiveBucketCompaction
                    ? "d_ir_multi_direct_group_active_count"
                    : "";
            const std::string activeCountSymbol =
                useActiveBucketCompaction
                    ? "n_ir_multi_direct_group_active_buckets"
                    : "";
            if (useActiveBucketCompaction) {
                keyed->setActiveBucketTracking(activeFlagBuffer, activeListBuffer,
                                               activeCounterBuffer,
                                               bucketCountExpr);
            }

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
            if (!havingSpec.scalarTotalBuffer.empty()) {
                keyed->setHavingTotal(havingSpec.scalarTotalBuffer,
                                      havingSpec.scalarAggOffset);
            }

            auto& directPhase = appendPhase(lowering->plan,
                "GENERIC_ir_multi_table_direct_group", std::move(keyed));
            if (fuseDirectInput && !scalarLookups.empty()) {
                attachGenericScalarLookupBuffers(directPhase, scalarLookups);
            }
            if (useActiveBucketCompaction) {
                attachMaterializedCountHook(directPhase, activeCounterBuffer,
                                            activeCountSymbol);
            }
            if (!havingSpec.scalarTotalBuffer.empty() &&
                !havingSpec.scalarTotalParam.empty()) {
                attachHavingTotalScalarHook(
                    directPhase, havingSpec.scalarTotalBuffer,
                    havingSpec.scalarTotalParam,
                    havingSpec.scalarAggIsLongPair,
                    havingSpec.scalarAggIsFloatSum);
            }

            std::vector<KeyedCompactKeySpec> compactKeys;
            std::vector<GenericMatColumnDesc> compactCols;
            for (const auto& key : denseKeys) {
                compactKeys.push_back({key.displayName, key.numValues,
                                       key.numValuesExpr, key.stride,
                                       key.charMap, key.keyBase, key.stringMap,
                                       key.stringLen, key.stringRowRef,
                                       key.stringSourceTable,
                                       key.stringSourceColumn});
                std::string buf = "d_ir_multi_direct_out_" +
                                  std::to_string(compactCols.size()) + "_" +
                                  sanitizeIdentifier(key.displayName);
                const bool isStringKey = !key.charMap.empty() ||
                                         !key.stringMap.empty() ||
                                         key.stringRowRef;
                compactCols.push_back(GenericMatColumnDesc{
                    key.displayName, buf, isStringKey ? "char" : "int",
                    isStringKey ? key.stringLen : 0});
            }

            std::vector<KeyedCompactAggSpec> compactAggs;
            for (size_t pi = 0; pi < pending.size(); ++pi) {
                const auto& p = pending[pi];
                if (p.funcName == "RATIO_DEN" ||
                    p.displayName.rfind("__hidden_", 0) == 0) {
                    continue;
                }
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
                if (p.funcName == "RATIO") {
                    if (pi + 1 >= pending.size() ||
                        pending[pi + 1].funcName != "RATIO_DEN") {
                        return fail(error,
                            "IR multi-table direct group lowerer: RATIO denominator is missing.");
                    }
                    const auto& den = pending[pi + 1];
                    out.isRatio = true;
                    out.ratioDenOffset = den.offset;
                    out.ratioDenIsLongPair = den.isLongPair;
                    out.ratioDenIsFloatSum = den.isFloatSum;
                    out.ratioDenScaleDown = den.scaleDown;
                    metalType = "float";
                    ++pi;
                } else if (p.scaleDown < 0 && pi + 1 < pending.size()) {
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
                    "d_ir_multi_direct_group_aggs", compactCounter, keyedBucketCount,
                    valuesPerBucket, compactKeys, compactAggs, compactCols,
                    bucketCountExpr, dynamicDomain ? dynamicBucketCountSymbol : "",
                    havingSpec, activeListBuffer, activeCountSymbol));
            (void)directPhase;

	            if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
	                const std::string sortRowsSym = "n_gpu_sort_ir_multi_direct_group_rows";
	                attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);
                if (!appendBestGenericGpuOrder(lowering->plan, "ir_multi_direct_group",
                                               sortRowsSym, bucketCountExpr,
                                               compactCols, sortSpec, error)) {
	                    return std::nullopt;
	                }
	            }

	            return std::move(lowering->plan);
            }
	        }
    }

    if (!aggregationChoiceMade) {
        auto hashOutputColumnsForChoice = makeHashOutputColumns();
        std::vector<GenericAggregationCandidateCostInput> candidates;
        candidates.push_back(hashCostCandidate(hashOutputColumnsForChoice));
        if (fdCandidate.available)
            candidates.push_back(fdCostCandidate());
        auto globalChoice = chooseAggregationCandidate(std::move(candidates));
        aggregationChoiceMade = true;
        if (fdCandidate.available &&
            globalChoice.chosenCandidate == fdCandidateName()) {
            return emitFdCandidate();
        }
    }

    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = groupHashCapacityExpr(
        lowering->outputSize, groupOutputBoundExpr);
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.maxOutputRowsExpr = groupOutputBoundExpr;
	    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
	    gbSpec.inputColumns = std::move(materializedCols);
	    gbSpec.groupBy = std::move(groupSpec);
	    auto hashOutputColumns = genericGpuGroupOutputColumns(gbSpec);
	    auto& matPhase = ensureMaterializePhase();
	    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
	    appendGenericGpuGroupBy(lowering->plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(lowering->plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
	        if (!appendBestGenericGpuOrder(lowering->plan, "group_" + groupTag,
	                                       sortRowsSym, gbSpec.maxOutputRowsExpr,
	                                       hashOutputColumns,
	                                       sortSpec, error)) {
            return std::nullopt;
        }
    }

    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return attachGenericCostTrace(
        lowerMultiTableGroupedAggregateIRToMetalImpl(ir, nullptr, error),
        ir, "multi_table_grouped_aggregate");
}

std::optional<MetalQueryPlan> lowerMultiTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return attachGenericCostTrace(
        lowerMultiTableGroupedAggregateIRToMetalImpl(ir, &aq, error),
        ir, "multi_table_grouped_aggregate");
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
    return attachGenericCostTrace(
        lowerMultiTableScalarAggregateIRToMetalImpl(ir, nullptr, error),
        ir, "multi_table_scalar_aggregate");
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return attachGenericCostTrace(
        lowerMultiTableScalarAggregateIRToMetalImpl(ir, &aq, error),
        ir, "multi_table_scalar_aggregate");
}

} // namespace codegen
