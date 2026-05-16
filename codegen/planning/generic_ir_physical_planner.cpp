#include "generic_ir_physical_planner.h"
#include "generic_scalar_lookup_helpers.h"
#include "metal_generic_sql_physical_ops.h"
#include "metal_plan_common.h"
#include "../core/metal_param_binding.h"
#include "../core/schema_provider.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <type_traits>

namespace codegen {

namespace {

std::optional<MetalQueryPlan> fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front())))
        name = "c_" + name;
    return name;
}

std::string metalTypeForType(const TypeInfo& type) {
    switch (type.type) {
        case DataType::INT:
        case DataType::DATE:
            return "int";
        case DataType::FLOAT:
            return "float";
        case DataType::CHAR1:
        case DataType::CHAR_FIXED:
            return "char";
    }
    return "int";
}

int fixedStringLenForExpr(const GenericExprPtr& expr) {
    if (!expr) return 0;
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (col->type.type == DataType::CHAR1) return 1;
        if (col->type.type == DataType::CHAR_FIXED) return col->type.fixedWidth;
    }
    return 0;
}

std::string literalToMetal(const GenericLiteralExpr& lit) {
    return std::visit([](const auto& value) -> std::string {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, int64_t>) {
            return std::to_string(value);
        } else if constexpr (std::is_same_v<V, double>) {
            return std::to_string(value) + "f";
        } else {
            std::string escaped;
            escaped.reserve(value.size() + 2);
            escaped.push_back('"');
            for (char ch : value) {
                if (ch == '\\' || ch == '"') escaped.push_back('\\');
                escaped.push_back(ch);
            }
            escaped.push_back('"');
            return escaped;
        }
    }, lit.value);
}

std::optional<std::string> stringLiteralValue(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    auto* value = std::get_if<std::string>(&lit->value);
    if (!value) return std::nullopt;
    return *value;
}

std::optional<int64_t> integerStringLiteralValue(const GenericExprPtr& expr) {
    auto value = stringLiteralValue(expr);
    if (!value || value->empty()) return std::nullopt;
    size_t pos = 0;
    if ((*value)[0] == '-' || (*value)[0] == '+') pos = 1;
    if (pos >= value->size()) return std::nullopt;
    for (size_t i = pos; i < value->size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>((*value)[i])))
            return std::nullopt;
    }
    try {
        return std::stoll(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::string genericMetalCharLiteral(char ch) {
    if (ch == '\\') return "'\\\\'";
    if (ch == '\'') return "'\\\''";
    if (ch == '\0') return "'\\0'";
    return std::string("'") + ch + "'";
}

std::string fixedStringEqMetalFromPointer(const std::string& basePtr,
                                          int width,
                                          const std::string& literal) {
    int cmpLen = std::min(static_cast<int>(literal.size()), width);
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += "(" + basePtr + ")[" + std::to_string(i) + "] == " +
                genericMetalCharLiteral(literal[(size_t)i]);
    }
    for (int i = cmpLen; i < width; ++i) {
        if (!cond.empty()) cond += " && ";
        std::string slot = "(" + basePtr + ")[" + std::to_string(i) + "]";
        cond += "(" + slot + " == '\\0' || " + slot + " == ' ')";
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

std::string fixedStringEqMetal(const GenericColumnExpr& col,
                               const std::string& literal,
                               const std::string& idxVar) {
    int width = col.type.fixedWidth > 0 ? col.type.fixedWidth : 1;
    int cmpLen = std::min(static_cast<int>(literal.size()), width);
    std::string base = col.column + "[" + idxVar + " * " +
                       std::to_string(width) + " + ";
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += base + std::to_string(i) + "] == " +
                genericMetalCharLiteral(literal[(size_t)i]);
    }
    for (int i = cmpLen; i < width; ++i) {
        if (!cond.empty()) cond += " && ";
        std::string slot = base + std::to_string(i) + "]";
        cond += "(" + slot + " == '\\0' || " + slot + " == ' ')";
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

std::optional<std::string> fixedStringLikeMetal(
        const GenericLikePred& like,
        const std::string& idxVar) {
    auto* col = like.expr ? std::get_if<GenericColumnExpr>(&like.expr->node) : nullptr;
    if (!col || col->type.type != DataType::CHAR_FIXED ||
        like.pattern.find('_') != std::string::npos ||
        like.pattern.find('\\') != std::string::npos) {
        return std::nullopt;
    }
    int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;

    if (like.pattern.find('%') == std::string::npos) {
        std::string exact = fixedStringEqMetal(*col, like.pattern, idxVar);
        return like.negated ? "!(" + exact + ")" : exact;
    }

    return fixedStringLikeDataMetal(col->column, "(uint)(" + idxVar + ")",
                                    width, like.pattern, like.negated);
}

std::optional<std::string> charStringComparisonToMetal(
        const GenericExprPtr& left,
        CmpOp op,
        const GenericExprPtr& right,
        const std::string& idxVar) {
    if (op != CmpOp::EQ && op != CmpOp::NE) return std::nullopt;

    const GenericColumnExpr* col = left ? std::get_if<GenericColumnExpr>(&left->node) : nullptr;
    auto literal = stringLiteralValue(right);
    if (!col || !literal) {
        col = right ? std::get_if<GenericColumnExpr>(&right->node) : nullptr;
        literal = stringLiteralValue(left);
    }
    if (!col || !literal) return std::nullopt;

    std::string eq;
    if (col->type.type == DataType::CHAR_FIXED) {
        eq = fixedStringEqMetal(*col, *literal, idxVar);
    } else if (col->type.type == DataType::CHAR1) {
        if (literal->empty()) return std::nullopt;
        eq = "(" + col->column + "[" + idxVar + "] == " +
             genericMetalCharLiteral(literal->front()) + ")";
    } else {
        return std::nullopt;
    }
    return op == CmpOp::NE ? "!(" + eq + ")" : eq;
}

std::string genericExprToMetal(const GenericExprPtr& expr, const std::string& idxVar);
std::string genericPredicateToMetal(const GenericPredicatePtr& pred,
                                    const std::string& idxVar);

std::string inListValueToMetal(const GenericExprPtr& value,
                               const TypeInfo& exprType,
                               const std::string& idxVar) {
    if ((exprType.type == DataType::INT || exprType.type == DataType::DATE) &&
        integerStringLiteralValue(value)) {
        return std::to_string(*integerStringLiteralValue(value));
    }
    return genericExprToMetal(value, idxVar);
}

std::string functionExprToMetal(const GenericFunctionExpr& fn,
                                const std::string& idxVar) {
    std::string name = fn.name;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });

    if (name == "date_part" || name == "extract") {
        std::string unit;
        if (!fn.args.empty() && fn.args[0]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[0]->node)) {
                if (auto* s = std::get_if<std::string>(&lit->value))
                    unit = *s;
            }
        }
        std::string col = fn.args.size() > 1 ? genericExprToMetal(fn.args[1], idxVar) : "0";
        if (unit == "year") return "(" + col + " / 10000)";
        if (unit == "month") return "((" + col + " / 100) % 100)";
        if (unit == "day") return "(" + col + " % 100)";
        return col;
    }

    if (name == "substring") {
        int start = 1;
        int len = 1;
        if (fn.args.size() > 1 && fn.args[1]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[1]->node)) {
                if (auto* v = std::get_if<int64_t>(&lit->value))
                    start = (int)*v;
            }
        }
        if (fn.args.size() > 2 && fn.args[2]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[2]->node)) {
                if (auto* v = std::get_if<int64_t>(&lit->value))
                    len = (int)*v;
            }
        }

        auto* col = fn.args.empty() || !fn.args[0]
            ? nullptr
            : std::get_if<GenericColumnExpr>(&fn.args[0]->node);
        if (!col) return "0";
        int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
        std::string result = "(";
        for (int i = 0; i < len; ++i) {
            if (i) result += " + ";
            int pos = start - 1 + i;
            int weight = 1;
            for (int w = 0; w < len - 1 - i; ++w) weight *= 10;
            std::string access = col->column + "[" + idxVar + " * " +
                                 std::to_string(width) + " + " +
                                 std::to_string(pos) + "]";
            if (weight > 1)
                result += "(" + access + " - '0') * " + std::to_string(weight);
            else
                result += "(" + access + " - '0')";
        }
        result += ")";
        return result;
    }

    return "0";
}

std::string genericExprToMetal(const GenericExprPtr& expr, const std::string& idxVar) {
    if (!expr) return "0";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return node.column + "[" + idxVar + "]";
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return literalToMetal(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            std::string left = genericExprToMetal(node.left, idxVar);
            std::string right = genericExprToMetal(node.right, idxVar);
            switch (node.op) {
                case ExprOp::ADD: return "(" + left + " + " + right + ")";
                case ExprOp::SUB: return "(" + left + " - " + right + ")";
                case ExprOp::MUL: return "(" + left + " * " + right + ")";
                case ExprOp::DIV: return "(" + left + " / " + right + ")";
            }
            return left;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.branches.empty())
                return genericExprToMetal(node.elseResult, idxVar);
            std::string result;
            for (const auto& branch : node.branches) {
                result += "((" + genericPredicateToMetal(branch.condition, idxVar) +
                          ") ? (" + genericExprToMetal(branch.result, idxVar) +
                          ") : ";
            }
            result += "(" + genericExprToMetal(node.elseResult, idxVar) + ")";
            for (size_t i = 0; i < node.branches.size(); ++i)
                result += ")";
            return result;
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            return functionExprToMetal(node, idxVar);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return node.arg ? genericExprToMetal(node.arg, idxVar) : "1";
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            return node.outputName;
        }
        return "0";
    }, expr->node);
}

std::string materializeExprToMetal(const GenericExprPtr& expr,
                                   const std::string& idxVar) {
    if (!expr) return "0";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (col->type.type == DataType::CHAR1)
            return col->column + " + " + idxVar;
        if (col->type.type == DataType::CHAR_FIXED) {
            int len = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
            return col->column + " + " + idxVar + " * " + std::to_string(len);
        }
    }
    return genericExprToMetal(expr, idxVar);
}

std::string cmpOpToMetal(CmpOp op) {
    switch (op) {
        case CmpOp::EQ: return "==";
        case CmpOp::NE: return "!=";
        case CmpOp::LT: return "<";
        case CmpOp::LE: return "<=";
        case CmpOp::GT: return ">";
        case CmpOp::GE: return ">=";
    }
    return "==";
}

std::string genericPredicateToMetal(const GenericPredicatePtr& pred,
                                    const std::string& idxVar) {
    if (!pred) return "true";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (auto cmp = charStringComparisonToMetal(node.left, node.op,
                                                       node.right, idxVar)) {
                return *cmp;
            }
            return "(" + genericExprToMetal(node.left, idxVar) + " " +
                   cmpOpToMetal(node.op) + " " +
                   genericExprToMetal(node.right, idxVar) + ")";
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            std::string expr = genericExprToMetal(node.expr, idxVar);
            return "(" + expr + " >= " + genericExprToMetal(node.low, idxVar) +
                   " && " + expr + " <= " + genericExprToMetal(node.high, idxVar) + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (node.values.empty()) return "true";
            std::string expr = genericExprToMetal(node.expr, idxVar);
            std::string out = "(";
            for (size_t i = 0; i < node.values.size(); ++i) {
                if (i) out += " || ";
                if (auto cmp = charStringComparisonToMetal(node.expr, CmpOp::EQ,
                                                           node.values[i], idxVar)) {
                    out += *cmp;
                } else {
                    out += expr + " == " +
                           inListValueToMetal(node.values[i],
                                               node.expr ? node.expr->type : TypeInfo{DataType::INT, 0},
                                               idxVar);
                }
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            if (auto like = fixedStringLikeMetal(node, idxVar))
                return *like;
            return node.negated ? "true" : "false";
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op == GenericLogicalPred::Op::Not) {
                return "!(" + (node.children.empty()
                    ? std::string("true")
                    : genericPredicateToMetal(node.children.front(), idxVar)) + ")";
            }
            if (node.children.empty()) {
                return node.op == GenericLogicalPred::Op::And ? "true" : "false";
            }
            std::string joiner = node.op == GenericLogicalPred::Op::And ? " && " : " || ";
            std::string out = "(";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i) out += joiner;
                out += genericPredicateToMetal(node.children[i], idxVar);
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericExistsPred>) {
            return node.negated ? "true" : "false";
        }
        return "true";
    }, pred->node);
}

bool predicateSupported(const GenericPredicatePtr& pred);

bool materializeExprSupported(const GenericExprPtr& expr) {
    if (!expr) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return true;
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return !std::holds_alternative<std::string>(node.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return materializeExprSupported(node.left) &&
                   materializeExprSupported(node.right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            std::string name = node.name;
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return name == "date_part" || name == "extract" || name == "substring";
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.type.type != DataType::INT &&
                node.type.type != DataType::DATE &&
                node.type.type != DataType::FLOAT) {
                return false;
            }
            for (const auto& branch : node.branches) {
                if (!predicateSupported(branch.condition) ||
                    !materializeExprSupported(branch.result)) {
                    return false;
                }
            }
            return materializeExprSupported(node.elseResult);
        } else {
            return false;
        }
    }, expr->node);
}

struct SingleTableShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* project = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct SingleTableScalarAggShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
};

struct SingleTableGroupedAggShape {
    const GenericRelNode* scan = nullptr;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct MultiTableMaterializeShape {
    std::vector<const GenericRelNode*> scans;
    std::vector<const GenericRelNode*> joins;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* project = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

struct MultiTableGroupedAggShape {
    std::vector<const GenericRelNode*> scans;
    std::vector<const GenericRelNode*> joins;
    const GenericRelNode* filter = nullptr;
    const GenericRelNode* aggregate = nullptr;
    const GenericRelNode* sort = nullptr;
    const GenericRelNode* limit = nullptr;
};

template <typename Shape>
std::optional<Shape> shapeFail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return std::nullopt;
}

bool predicateSupported(const GenericPredicatePtr& pred) {
    if (!pred) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (charStringComparisonToMetal(node.left, node.op, node.right, "i"))
                return true;
            return materializeExprSupported(node.left) &&
                   materializeExprSupported(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return materializeExprSupported(node.expr) &&
                   materializeExprSupported(node.low) &&
                   materializeExprSupported(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            auto* inCol = node.expr
                ? std::get_if<GenericColumnExpr>(&node.expr->node)
                : nullptr;
            if (inCol && (inCol->type.type == DataType::CHAR_FIXED ||
                          inCol->type.type == DataType::CHAR1)) {
                for (const auto& value : node.values) {
                    if (!stringLiteralValue(value)) return false;
                }
                return true;
            }
            if (!materializeExprSupported(node.expr)) return false;
            for (const auto& value : node.values) {
                if (materializeExprSupported(value)) continue;
                if (node.expr &&
                    (node.expr->type.type == DataType::INT ||
                     node.expr->type.type == DataType::DATE) &&
                    integerStringLiteralValue(value)) {
                    continue;
                }
                return false;
            }
            return true;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return fixedStringLikeMetal(node, "i").has_value();
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (!predicateSupported(child)) return false;
            }
            return true;
        } else {
            return false;
        }
    }, pred->node);
}

std::optional<SingleTableShape> parseSingleTableMaterializeShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Project)
        return std::nullopt;
    shape.project = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableShape>(
            error, "IR materialize lowerer: expected Scan under Project/Filter.");
    }
    shape.scan = node;
    return shape;
}

std::optional<SingleTableScalarAggShape> parseSingleTableScalarAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableScalarAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableScalarAggShape>(
            error, "IR scalar aggregate lowerer: expected Scan under Aggregate/Filter.");
    }
    shape.scan = node;
    return shape;
}

std::optional<SingleTableGroupedAggShape> parseSingleTableGroupedAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    SingleTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Scan) {
        return shapeFail<SingleTableGroupedAggShape>(
            error, "IR grouped aggregate lowerer: expected Scan under Aggregate/Filter.");
    }
    shape.scan = node;
    return shape;
}

bool isSupportedJoinTreeOp(GenericRelOp op) {
    return op == GenericRelOp::Join ||
           op == GenericRelOp::SemiJoin ||
           op == GenericRelOp::AntiJoin;
}

bool collectInnerJoinScans(const GenericRelPlan& ir,
                           const GenericRelNode* node,
                           std::vector<const GenericRelNode*>& scans,
                           std::vector<const GenericRelNode*>& joins,
                           std::string* error) {
    if (!node) {
        if (error) *error = "IR multi-table materialize lowerer: null join-tree node.";
        return false;
    }
    if (node->op == GenericRelOp::Scan) {
        scans.push_back(node);
        return true;
    }
    if (!isSupportedJoinTreeOp(node->op)) {
        if (error) *error = "IR multi-table materialize lowerer: join tree contains " +
                            genericRelOpName(node->op) + ".";
        return false;
    }
    auto* detail = std::get_if<GenericJoinDetail>(&node->detail);
    if (!detail || detail->kind == GenericJoinKind::LeftOuter) {
        if (error) *error = "IR multi-table materialize lowerer: only inner/semi/anti joins are supported.";
        return false;
    }
    if (node->inputs.size() != 2) {
        if (error) *error = "IR multi-table materialize lowerer: join must have two inputs.";
        return false;
    }
    joins.push_back(node);
    return collectInnerJoinScans(ir, ir.findNode(node->inputs[0]), scans, joins, error) &&
           collectInnerJoinScans(ir, ir.findNode(node->inputs[1]), scans, joins, error);
}

void collectScanRelationInstances(const GenericRelPlan& ir,
                                  const GenericRelNode* node,
                                  std::set<int>& relationInstances) {
    if (!node) return;
    if (node->op == GenericRelOp::Scan) {
        if (auto* scan = std::get_if<GenericScanDetail>(&node->detail)) {
            if (scan->relationInstance.valid())
                relationInstances.insert(scan->relationInstance.value);
        }
        return;
    }
    for (const auto& input : node->inputs)
        collectScanRelationInstances(ir, ir.findNode(input), relationInstances);
}

std::optional<MultiTableMaterializeShape> parseMultiTableMaterializeShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableMaterializeShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Project)
        return std::nullopt;
    shape.project = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableMaterializeShape>(
            error, "IR multi-table materialize lowerer: expected at least two scans.");
    return shape;
}

std::optional<MultiTableGroupedAggShape> parseMultiTableGroupedAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node) return std::nullopt;

    if (node->op == GenericRelOp::Limit) {
        shape.limit = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (node && node->op == GenericRelOp::Sort) {
        shape.sort = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableGroupedAggShape>(
            error, "IR multi-table grouped aggregate lowerer: expected at least two scans.");
    return shape;
}

std::optional<MultiTableGroupedAggShape> parseMultiTableScalarAggShape(
        const GenericRelPlan& ir,
        std::string* error) {
    MultiTableGroupedAggShape shape;
    const GenericRelNode* node = ir.findNode(ir.root);
    if (!node || node->op != GenericRelOp::Aggregate)
        return std::nullopt;
    shape.aggregate = node;

    node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    if (node && node->op == GenericRelOp::Filter) {
        shape.filter = node;
        node = node->inputs.empty() ? nullptr : ir.findNode(node->inputs.front());
    }
    if (!node || !isSupportedJoinTreeOp(node->op))
        return std::nullopt;
    if (!collectInnerJoinScans(ir, node, shape.scans, shape.joins, error))
        return std::nullopt;
    if (shape.scans.size() < 2)
        return shapeFail<MultiTableGroupedAggShape>(
            error, "IR multi-table scalar aggregate lowerer: expected at least two scans.");
    return shape;
}

const GenericScanDetail* scanDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericScanDetail>(&node->detail);
}

const GenericProjectDetail* projectDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericProjectDetail>(&node->detail);
}

const GenericFilterDetail* filterDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericFilterDetail>(&node->detail);
}

const GenericSortDetail* sortDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericSortDetail>(&node->detail);
}

int limitValue(const GenericRelNode* node) {
    if (!node) return -1;
    if (auto* detail = std::get_if<GenericLimitDetail>(&node->detail))
        return detail->limit;
    return -1;
}

const GenericAggregateDetail* aggregateDetail(const GenericRelNode* node) {
    if (!node) return nullptr;
    return std::get_if<GenericAggregateDetail>(&node->detail);
}

bool projectedColumnMatches(const GenericProjection& projection,
                            const GenericColumnExpr& sortColumn) {
    if (projection.name == sortColumn.column) return true;
    if (!projection.expr) return false;
    auto* col = std::get_if<GenericColumnExpr>(&projection.expr->node);
    if (!col) return false;
    bool sameColumn = col->column == sortColumn.column;
    bool sameTable = sortColumn.table.empty() || col->table == sortColumn.table;
    bool sameAlias = sortColumn.alias.empty() || col->alias == sortColumn.alias;
    return sameColumn && sameTable && sameAlias;
}

struct IrGroupKeyDesc {
    std::string displayName;
    std::string keyExpr;
    int numValues = 0;
    int stride = 1;
    std::vector<char> charMap;
    int keyBase = 0;
};

struct IrPendingAgg {
    std::string displayName;
    int offset = 0;
    std::string valueExpr;
    bool isLongPair = false;
    int scaleDown = 0;
    bool isFloatSum = false;
    bool isMinMax = false;
    std::string atomicOp = "add";
    std::string funcName;
    std::string innerColumn;
};

bool genericExprEquivalent(const GenericExprPtr& left, const GenericExprPtr& right);

std::optional<std::string> sortKeyDisplayName(const GenericSortKey& key,
                                              const GenericProjectDetail& project) {
    if (!key.expr) return std::nullopt;
    if (auto* col = std::get_if<GenericColumnExpr>(&key.expr->node)) {
        for (const auto& projection : project.projections) {
            if (projectedColumnMatches(projection, *col))
                return projection.name;
        }
    }
    return std::nullopt;
}

std::optional<std::string> sortKeyDisplayNameForGroupedAgg(
        const GenericSortKey& key,
        const GenericAggregateDetail& aggregate,
        const std::vector<IrGroupKeyDesc>& groupKeys) {
    if (!key.expr) return std::nullopt;
    for (size_t i = 0; i < aggregate.groupBy.size() && i < groupKeys.size(); ++i) {
        if (genericExprEquivalent(key.expr, aggregate.groupBy[i]))
            return groupKeys[i].displayName;
    }
    if (auto* col = std::get_if<GenericColumnExpr>(&key.expr->node)) {
        for (size_t i = 0; i < aggregate.groupNames.size() && i < groupKeys.size(); ++i) {
            if (col->column == aggregate.groupNames[i])
                return groupKeys[i].displayName;
        }
        for (size_t i = 0; i < aggregate.groupBy.size() && i < groupKeys.size(); ++i) {
            auto* groupCol = std::get_if<GenericColumnExpr>(&aggregate.groupBy[i]->node);
            if (!groupCol) continue;
            if (groupCol->column == col->column &&
                (col->table.empty() || groupCol->table == col->table) &&
                (col->alias.empty() || groupCol->alias == col->alias)) {
                return groupKeys[i].displayName;
            }
        }
        for (const auto& agg : aggregate.aggregates) {
            if (agg.name == col->column) return agg.name;
        }
    }
    return std::nullopt;
}

std::string char1BucketExpr(const GenericColumnExpr& col, const std::string& idxVar) {
    if (col.charDomain.empty()) return "";
    if (col.charDomain.size() == 1) return "0";
    std::string expr = col.column + "[" + idxVar + "]";
    std::string result;
    for (size_t i = 0; i + 1 < col.charDomain.size(); ++i) {
        result += "(" + expr + " == '" + col.charDomain[i] + "' ? " +
                  std::to_string(i) + " : ";
    }
    result += std::to_string(col.charDomain.size() - 1);
    for (size_t i = 0; i + 1 < col.charDomain.size(); ++i)
        result += ")";
    return result;
}

std::string scaledLongExpr(const std::string& rawExpr, int scale) {
    return "(long)round((" + rawExpr + ") * " + std::to_string(scale) + ".0f)";
}

int numericScaleForExpr(const GenericExprPtr& expr) {
    if (!expr) return 0;
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->numericScale;
    return 0;
}

std::string distinctDomainSymbolForExpr(const GenericExprPtr& expr) {
    if (!expr) return "";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->distinctDomainSymbol;
    return "";
}

std::string innerColumnName(const GenericExprPtr& expr) {
    if (!expr) return "";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
        return col->column;
    return "";
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::optional<AggFunc> aggregateFuncFromName(const std::string& name) {
    const std::string lower = lowerAscii(name);
    if (lower == "sum") return AggFunc::SUM;
    if (lower == "count") return AggFunc::COUNT;
    if (lower == "avg") return AggFunc::AVG;
    if (lower == "min") return AggFunc::MIN;
    if (lower == "max") return AggFunc::MAX;
    return std::nullopt;
}

bool genericExprEquivalent(const GenericExprPtr& left, const GenericExprPtr& right) {
    if (!left || !right) return !left && !right;
    return std::visit([&](const auto& lnode) -> bool {
        using L = std::decay_t<decltype(lnode)>;
        auto* rnode = std::get_if<L>(&right->node);
        if (!rnode) return false;

        if constexpr (std::is_same_v<L, GenericColumnExpr>) {
            if (lnode.column != rnode->column) return false;
            if (lnode.relationInstance.valid() && rnode->relationInstance.valid())
                return lnode.relationInstance.value == rnode->relationInstance.value;
            if (!lnode.table.empty() && !rnode->table.empty() &&
                lnode.table != rnode->table) {
                return false;
            }
            if (!lnode.alias.empty() && !rnode->alias.empty() &&
                lnode.alias != rnode->alias) {
                return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericLiteralExpr>) {
            return lnode.value == rnode->value;
        } else if constexpr (std::is_same_v<L, GenericBinaryExpr>) {
            return lnode.op == rnode->op &&
                   genericExprEquivalent(lnode.left, rnode->left) &&
                   genericExprEquivalent(lnode.right, rnode->right);
        } else if constexpr (std::is_same_v<L, GenericFunctionExpr>) {
            if (lowerAscii(lnode.name) != lowerAscii(rnode->name) ||
                lnode.args.size() != rnode->args.size()) {
                return false;
            }
            for (size_t i = 0; i < lnode.args.size(); ++i) {
                if (!genericExprEquivalent(lnode.args[i], rnode->args[i]))
                    return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericAggregateExpr>) {
            return lnode.func == rnode->func &&
                   lnode.star == rnode->star &&
                   lnode.distinct == rnode->distinct &&
                   genericExprEquivalent(lnode.arg, rnode->arg);
        } else if constexpr (std::is_same_v<L, GenericScalarLookupExpr>) {
            if (lnode.source.value != rnode->source.value ||
                lnode.outputName != rnode->outputName ||
                lnode.keys.size() != rnode->keys.size()) {
                return false;
            }
            for (size_t i = 0; i < lnode.keys.size(); ++i) {
                if (!genericExprEquivalent(lnode.keys[i], rnode->keys[i]))
                    return false;
            }
            return true;
        } else if constexpr (std::is_same_v<L, GenericCaseExpr>) {
            return false;
        }
        return false;
    }, left->node);
}

bool havingExprMatchesAggregate(const GenericExprPtr& expr,
                                const GenericProjection& projection) {
    if (!expr || !projection.expr) return false;
    auto* projected = std::get_if<GenericAggregateExpr>(&projection.expr->node);
    if (!projected) return false;

    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        return !projection.name.empty() && col->column == projection.name;
    }

    if (auto* agg = std::get_if<GenericAggregateExpr>(&expr->node)) {
        return agg->func == projected->func &&
               agg->star == projected->star &&
               agg->distinct == projected->distinct &&
               genericExprEquivalent(agg->arg, projected->arg);
    }

    auto* fn = std::get_if<GenericFunctionExpr>(&expr->node);
    if (!fn) return false;
    auto func = aggregateFuncFromName(fn->name);
    if (!func || *func != projected->func) return false;

    const bool star = *func == AggFunc::COUNT && fn->args.empty();
    if (star || projected->star)
        return star == projected->star;
    if (fn->args.empty()) return !projected->arg;
    return genericExprEquivalent(fn->args.front(), projected->arg);
}

std::optional<int> aggregateIndexForHavingExpr(
        const GenericExprPtr& expr,
        const GenericAggregateDetail& aggregate) {
    for (size_t i = 0; i < aggregate.aggregates.size(); ++i) {
        if (havingExprMatchesAggregate(expr, aggregate.aggregates[i]))
            return static_cast<int>(i);
    }
    return std::nullopt;
}

std::optional<double> numericLiteralValue(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return std::nullopt;
    return std::visit([](const auto& value) -> std::optional<double> {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, int64_t>)
            return static_cast<double>(value);
        else if constexpr (std::is_same_v<V, double>)
            return value;
        else
            return std::nullopt;
    }, lit->value);
}

bool isScalarSubquerySentinelLiteral(const GenericExprPtr& expr) {
    if (!expr) return false;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    if (!lit) return false;
    auto* value = std::get_if<int64_t>(&lit->value);
    return value && *value < -1000000;
}

bool predicateReferencesScalarSentinel(const GenericPredicatePtr& pred);

bool exprReferencesScalarSentinel(const GenericExprPtr& expr) {
    if (!expr) return false;
    if (isScalarSubquerySentinelLiteral(expr)) return true;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            return exprReferencesScalarSentinel(node.left) ||
                   exprReferencesScalarSentinel(node.right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args) {
                if (exprReferencesScalarSentinel(arg)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                if (predicateReferencesScalarSentinel(branch.condition) ||
                    exprReferencesScalarSentinel(branch.result)) {
                    return true;
                }
            }
            return exprReferencesScalarSentinel(node.elseResult);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return exprReferencesScalarSentinel(node.arg);
        }
        return false;
    }, expr->node);
}

bool predicateReferencesScalarSentinel(const GenericPredicatePtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            return exprReferencesScalarSentinel(node.left) ||
                   exprReferencesScalarSentinel(node.right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            return exprReferencesScalarSentinel(node.expr) ||
                   exprReferencesScalarSentinel(node.low) ||
                   exprReferencesScalarSentinel(node.high);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (exprReferencesScalarSentinel(node.expr)) return true;
            for (const auto& value : node.values) {
                if (exprReferencesScalarSentinel(value)) return true;
            }
            return false;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            return exprReferencesScalarSentinel(node.expr);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predicateReferencesScalarSentinel(child)) return true;
            }
            return false;
        }
        return false;
    }, pred->node);
}

std::optional<int> scalarSubqueryIndexFromSentinelLiteral(const GenericExprPtr& expr) {
    if (!isScalarSubquerySentinelLiteral(expr)) return std::nullopt;
    auto* lit = std::get_if<GenericLiteralExpr>(&expr->node);
    auto* value = lit ? std::get_if<int64_t>(&lit->value) : nullptr;
    if (!value) return std::nullopt;
    const int64_t idx = *value - static_cast<int64_t>(std::numeric_limits<int>::min());
    if (idx < 0 || idx > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return std::nullopt;
    return static_cast<int>(idx);
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

std::string formatSignatureDouble(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::string aggFuncSignatureName(AggFunc func) {
    return lowerAscii(aggFuncName(func));
}

std::string exprOpSignatureToken(ExprOp op) {
    switch (op) {
        case ExprOp::ADD: return "+";
        case ExprOp::SUB: return "-";
        case ExprOp::MUL: return "*";
        case ExprOp::DIV: return "/";
    }
    return "?";
}

std::optional<std::string> jsonStringValueForIr(const nlohmann::json& node) {
    if (node.is_string()) return node.get<std::string>();
    if (node.is_object() && node.contains("String") && node["String"].contains("sval"))
        return node["String"]["sval"].get<std::string>();
    return std::nullopt;
}

std::string jsonAExprOpForIr(const nlohmann::json& ae) {
    if (!ae.contains("name") || !ae["name"].is_array() || ae["name"].empty())
        return {};
    if (auto s = jsonStringValueForIr(ae["name"][0])) return *s;
    return {};
}

std::string jsonFuncNameForIr(const nlohmann::json& fc) {
    if (!fc.contains("funcname") || !fc["funcname"].is_array() || fc["funcname"].empty())
        return {};
    auto s = jsonStringValueForIr(fc["funcname"].back());
    return s ? lowerAscii(*s) : "";
}

std::optional<double> jsonNumericConstForIr(const nlohmann::json& node) {
    const nlohmann::json* ac = nullptr;
    if (node.is_object() && node.contains("A_Const")) ac = &node["A_Const"];
    else if (node.is_object()) ac = &node;
    if (!ac) return std::nullopt;

    auto readNum = [](const nlohmann::json& v) -> std::optional<double> {
        try {
            if (v.is_number()) return v.get<double>();
            if (v.is_string()) return std::stod(v.get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
        return std::nullopt;
    };

    try {
        if (ac->contains("fval")) {
            const auto& f = (*ac)["fval"];
            if (f.is_object() && f.contains("fval")) return readNum(f["fval"]);
            return readNum(f);
        }
        if (ac->contains("ival")) {
            const auto& i = (*ac)["ival"];
            if (i.is_object() && i.contains("ival")) return readNum(i["ival"]);
            return readNum(i);
        }
        if (ac->contains("val")) {
            const auto& v = (*ac)["val"];
            if (v.contains("Float")) return readNum(v["Float"].at("fval"));
            if (v.contains("Integer")) return readNum(v["Integer"].at("ival"));
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

std::string resolveJsonColumnTable(const std::string& qualifier,
                                   const std::string& column,
                                   const std::map<std::string, std::string>& aliases,
                                   const std::vector<std::string>& tables,
                                   const SchemaProvider* schema) {
    if (!qualifier.empty()) {
        auto it = aliases.find(qualifier);
        return it == aliases.end() ? qualifier : it->second;
    }

    if (!schema) return "";
    std::string match;
    for (const auto& table : tables) {
        if (!schema->hasColumn(table, column)) continue;
        if (!match.empty()) return "";
        match = table;
    }
    return match;
}

std::optional<std::string> jsonColumnSignatureForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
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
    const std::string column = fields.back();
    const std::string qualifier = fields.size() >= 2 ? fields[fields.size() - 2] : "";
    const std::string table = resolveJsonColumnTable(qualifier, column, aliases, tables, schema);
    return table.empty() ? "col:" + column : "col:" + table + "." + column;
}

std::string genericColumnSignatureForIr(const GenericColumnExpr& col) {
    return col.table.empty() ? "col:" + col.column : "col:" + col.table + "." + col.column;
}

std::optional<std::string> genericExprSignatureForIr(const GenericExprPtr& expr);

std::string combineBinarySignature(const std::string& op,
                                   std::string left,
                                   std::string right) {
    if (op == "+" || op == "*") {
        if (right < left) std::swap(left, right);
    }
    return "bin:" + op + "(" + left + "," + right + ")";
}

std::optional<std::string> genericExprSignatureForIr(const GenericExprPtr& expr) {
    if (!expr) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<std::string> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            return genericColumnSignatureForIr(node);
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return std::visit([](const auto& value) -> std::string {
                using V = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<V, int64_t>)
                    return "lit:i:" + std::to_string(value);
                else if constexpr (std::is_same_v<V, double>)
                    return "lit:f:" + formatSignatureDouble(value);
                else
                    return "lit:s:" + value;
            }, node.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            auto left = genericExprSignatureForIr(node.left);
            auto right = genericExprSignatureForIr(node.right);
            if (!left || !right) return std::nullopt;
            return combineBinarySignature(exprOpSignatureToken(node.op), *left, *right);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            std::vector<std::string> args;
            for (const auto& arg : node.args) {
                auto sig = genericExprSignatureForIr(arg);
                if (!sig) return std::nullopt;
                args.push_back(*sig);
            }
            std::ostringstream oss;
            oss << "fn:" << lowerAscii(node.name) << "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i) oss << ",";
                oss << args[i];
            }
            oss << ")";
            return oss.str();
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            if (node.star) return "agg:" + aggFuncSignatureName(node.func) + "(*)";
            auto arg = genericExprSignatureForIr(node.arg);
            if (!arg) return std::nullopt;
            return "agg:" + aggFuncSignatureName(node.func) + "(" + *arg + ")";
        } else {
            return std::nullopt;
        }
    }, expr->node);
}

std::optional<std::string> jsonExprSignatureForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema) {
    if (!node.is_object()) return std::nullopt;
    if (node.contains("TypeCast"))
        return jsonExprSignatureForIr(node["TypeCast"].value("arg", nlohmann::json{}),
                                      aliases, tables, schema);
    if (node.contains("ColumnRef") || node.contains("fields"))
        return jsonColumnSignatureForIr(node, aliases, tables, schema);
    if (node.contains("A_Const")) {
        const auto& ac = node["A_Const"];
        try {
            auto readStringConst = [](const nlohmann::json& value)
                    -> std::optional<std::string> {
                if (value.is_string()) return value.get<std::string>();
                if (value.is_object() && value.contains("sval"))
                    return value["sval"].get<std::string>();
                return std::nullopt;
            };
            auto readIntConst = [](const nlohmann::json& value)
                    -> std::optional<int64_t> {
                if (value.is_number_integer()) return value.get<int64_t>();
                if (value.is_string()) return std::stoll(value.get<std::string>());
                if (value.is_object() && value.contains("ival")) {
                    const auto& inner = value["ival"];
                    if (inner.is_number_integer()) return inner.get<int64_t>();
                    if (inner.is_string()) return std::stoll(inner.get<std::string>());
                }
                return std::nullopt;
            };
            auto readFloatConst = [](const nlohmann::json& value)
                    -> std::optional<double> {
                if (value.is_number()) return value.get<double>();
                if (value.is_string()) return std::stod(value.get<std::string>());
                if (value.is_object() && value.contains("fval")) {
                    const auto& inner = value["fval"];
                    if (inner.is_number()) return inner.get<double>();
                    if (inner.is_string()) return std::stod(inner.get<std::string>());
                }
                return std::nullopt;
            };
            if (ac.contains("sval")) {
                if (auto s = readStringConst(ac["sval"])) return "lit:s:" + *s;
            }
            if (ac.contains("ival")) {
                if (auto i = readIntConst(ac["ival"])) return "lit:i:" + std::to_string(*i);
            }
            if (ac.contains("fval")) {
                if (auto f = readFloatConst(ac["fval"]))
                    return "lit:f:" + formatSignatureDouble(*f);
            }
            if (ac.contains("val") && ac["val"].contains("String"))
                return "lit:s:" + ac["val"]["String"].at("sval").get<std::string>();
            if (ac.contains("val") && ac["val"].contains("Integer"))
                return "lit:i:" + std::to_string(ac["val"]["Integer"].at("ival").get<int64_t>());
            if (ac.contains("val") && ac["val"].contains("Float"))
                return "lit:f:" + formatSignatureDouble(
                    std::stod(ac["val"]["Float"].at("fval").get<std::string>()));
        } catch (...) {
            return std::nullopt;
        }
        return std::nullopt;
    }
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        const std::string name = jsonFuncNameForIr(fc);
        std::vector<std::string> args;
        if (fc.contains("args") && fc["args"].is_array()) {
            for (const auto& arg : fc["args"]) {
                if (arg.is_object() && arg.contains("A_Star")) {
                    args.push_back("*");
                    continue;
                }
                auto sig = jsonExprSignatureForIr(arg, aliases, tables, schema);
                if (!sig) return std::nullopt;
                args.push_back(*sig);
            }
        }
        std::ostringstream oss;
        oss << "fn:" << name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i) oss << ",";
            oss << args[i];
        }
        oss << ")";
        return oss.str();
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        const std::string op = jsonAExprOpForIr(ae);
        auto left = jsonExprSignatureForIr(ae.value("lexpr", nlohmann::json{}),
                                           aliases, tables, schema);
        auto right = jsonExprSignatureForIr(ae.value("rexpr", nlohmann::json{}),
                                            aliases, tables, schema);
        if (!left || !right || op.empty()) return std::nullopt;
        return combineBinarySignature(op, *left, *right);
    }
    return std::nullopt;
}

std::string predicateSignatureFromComparison(std::string op,
                                             std::string left,
                                             std::string right) {
    if (op == "=") op = "==";
    if (op == "<>") op = "!=";
    if (op == "==") {
        if (right < left) std::swap(left, right);
    }
    return "cmp:" + op + "(" + left + "," + right + ")";
}

std::optional<std::string> genericPredicateSignatureForIr(const GenericPredicatePtr& pred) {
    if (!pred) return std::nullopt;
    return std::visit([&](const auto& node) -> std::optional<std::string> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            auto left = genericExprSignatureForIr(node.left);
            auto right = genericExprSignatureForIr(node.right);
            if (!left || !right) return std::nullopt;
            return predicateSignatureFromComparison(cmpOpToMetal(node.op), *left, *right);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            auto low = genericExprSignatureForIr(node.low);
            auto high = genericExprSignatureForIr(node.high);
            if (!expr || !low || !high) return std::nullopt;
            return "between:" + *expr + "(" + *low + "," + *high + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            if (!expr) return std::nullopt;
            std::vector<std::string> values;
            for (const auto& value : node.values) {
                auto sig = genericExprSignatureForIr(value);
                if (!sig) return std::nullopt;
                values.push_back(*sig);
            }
            std::sort(values.begin(), values.end());
            std::ostringstream oss;
            oss << "in:" << *expr << "(";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) oss << ",";
                oss << values[i];
            }
            oss << ")";
            return oss.str();
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            auto expr = genericExprSignatureForIr(node.expr);
            if (!expr) return std::nullopt;
            return std::string(node.negated ? "notlike:" : "like:") + *expr + ":" + node.pattern;
        } else {
            return std::nullopt;
        }
    }, pred->node);
}

bool collectGenericPredicateAtomSignatures(const GenericPredicatePtr& pred,
                                           std::vector<std::string>& out) {
    if (!pred) return true;
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op == GenericLogicalPred::Op::And) {
            for (const auto& child : logical->children) {
                if (!collectGenericPredicateAtomSignatures(child, out))
                    return false;
            }
            return true;
        }
    }
    auto sig = genericPredicateSignatureForIr(pred);
    if (!sig) return false;
    out.push_back(*sig);
    return true;
}

bool collectJsonPredicateAtomSignatures(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        std::vector<std::string>& out) {
    if (!node.is_object()) return false;
    if (node.contains("BoolExpr") && node["BoolExpr"].value("boolop", "") == "AND_EXPR") {
        const auto& args = node["BoolExpr"].value("args", nlohmann::json::array());
        for (const auto& arg : args) {
            if (!collectJsonPredicateAtomSignatures(arg, aliases, tables, schema, out))
                return false;
        }
        return true;
    }
    if (node.contains("A_Expr")) {
        const auto& ae = node["A_Expr"];
        const std::string op = jsonAExprOpForIr(ae);
        auto left = jsonExprSignatureForIr(ae.value("lexpr", nlohmann::json{}),
                                           aliases, tables, schema);
        auto right = jsonExprSignatureForIr(ae.value("rexpr", nlohmann::json{}),
                                            aliases, tables, schema);
        if (!left || !right || op.empty()) return false;
        out.push_back(predicateSignatureFromComparison(op, *left, *right));
        return true;
    }
    return false;
}

struct JsonScalarAggTarget {
    AggFunc func = AggFunc::SUM;
    bool star = false;
    double multiplier = 1.0;
    std::string argSignature;
};

bool extractJsonScalarAggTargetForIr(
        const nlohmann::json& node,
        const std::map<std::string, std::string>& aliases,
        const std::vector<std::string>& tables,
        const SchemaProvider* schema,
        JsonScalarAggTarget& out,
        double multiplier = 1.0) {
    if (!node.is_object()) return false;
    if (node.contains("TypeCast"))
        return extractJsonScalarAggTargetForIr(
            node["TypeCast"].value("arg", nlohmann::json{}), aliases, tables,
            schema, out, multiplier);
    if (node.contains("FuncCall")) {
        const auto& fc = node["FuncCall"];
        auto func = aggregateFuncFromName(jsonFuncNameForIr(fc));
        if (!func) return false;
        out.func = *func;
        out.multiplier = multiplier;
        out.star = !fc.contains("args") || !fc["args"].is_array() || fc["args"].empty();
        if (!out.star) {
            const auto& arg = fc["args"][0];
            out.star = arg.is_object() && arg.contains("A_Star");
        }
        if (!out.star) {
            auto sig = jsonExprSignatureForIr(fc["args"][0], aliases, tables, schema);
            if (!sig) return false;
            out.argSignature = *sig;
        }
        return true;
    }
    if (!node.contains("A_Expr")) return false;

    const auto& ae = node["A_Expr"];
    const std::string op = jsonAExprOpForIr(ae);
    if (op == "*") {
        if (auto lit = jsonNumericConstForIr(ae.value("lexpr", nlohmann::json{})))
            return extractJsonScalarAggTargetForIr(ae.value("rexpr", nlohmann::json{}),
                                                   aliases, tables, schema, out,
                                                   multiplier * *lit);
        if (auto lit = jsonNumericConstForIr(ae.value("rexpr", nlohmann::json{})))
            return extractJsonScalarAggTargetForIr(ae.value("lexpr", nlohmann::json{}),
                                                   aliases, tables, schema, out,
                                                   multiplier * *lit);
    }
    if (op == "/") {
        if (auto lit = jsonNumericConstForIr(ae.value("rexpr", nlohmann::json{}))) {
            if (*lit != 0.0)
                return extractJsonScalarAggTargetForIr(ae.value("lexpr", nlohmann::json{}),
                                                       aliases, tables, schema, out,
                                                       multiplier / *lit);
        }
    }
    return false;
}

struct ScalarHavingSubquerySummary {
    JsonScalarAggTarget aggregate;
    std::vector<std::string> tables;
    std::vector<std::string> predicateSignatures;
};

std::optional<ScalarHavingSubquerySummary> parseScalarHavingSubquerySummary(
        const AnalyzedQuery& aq,
        int sqIdx) {
    if (sqIdx < 0 || sqIdx >= static_cast<int>(aq.subqueries.size()))
        return std::nullopt;
    const auto& sq = aq.subqueries[(size_t)sqIdx];
    if (sq.type != AnalyzedQuery::Subquery::SCALAR_SUBQUERY)
        return std::nullopt;

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(sq.sql);
    } catch (...) {
        return std::nullopt;
    }
    if (!root.contains("SelectStmt")) return std::nullopt;
    const auto& ss = root["SelectStmt"];
    if (ss.contains("groupClause") && !ss["groupClause"].is_null()) return std::nullopt;
    if (ss.contains("havingClause") && !ss["havingClause"].is_null()) return std::nullopt;
    if (ss.contains("limitCount") && !ss["limitCount"].is_null()) return std::nullopt;

    ScalarHavingSubquerySummary summary;
    std::map<std::string, std::string> aliases;
    if (!ss.contains("fromClause") || !ss["fromClause"].is_array())
        return std::nullopt;
    for (const auto& from : ss["fromClause"]) {
        if (!from.contains("RangeVar")) return std::nullopt;
        const auto& rv = from["RangeVar"];
        const std::string rel = rv.value("relname", "");
        if (rel.empty()) return std::nullopt;
        summary.tables.push_back(rel);
        aliases[rel] = rel;
        if (rv.contains("alias")) {
            if (rv["alias"].contains("Alias")) {
                aliases[rv["alias"]["Alias"].value("aliasname", rel)] = rel;
            } else if (rv["alias"].contains("aliasname")) {
                aliases[rv["alias"].value("aliasname", rel)] = rel;
            }
        }
    }
    if (summary.tables.empty()) return std::nullopt;

    if (!ss.contains("targetList") || !ss["targetList"].is_array())
        return std::nullopt;
    bool foundAgg = false;
    for (const auto& target : ss["targetList"]) {
        if (!target.contains("ResTarget") || !target["ResTarget"].contains("val"))
            continue;
        if (extractJsonScalarAggTargetForIr(target["ResTarget"]["val"], aliases,
                                            summary.tables, aq.schema,
                                            summary.aggregate)) {
            foundAgg = true;
            break;
        }
    }
    if (!foundAgg) return std::nullopt;

    if (ss.contains("whereClause") && !ss["whereClause"].is_null()) {
        if (!collectJsonPredicateAtomSignatures(ss["whereClause"], aliases,
                                                summary.tables, aq.schema,
                                                summary.predicateSignatures)) {
            return std::nullopt;
        }
        std::sort(summary.predicateSignatures.begin(),
                  summary.predicateSignatures.end());
    }
    std::sort(summary.tables.begin(), summary.tables.end());
    return summary;
}

std::vector<std::string> scanTableSignaturesForHaving(
        const std::vector<const GenericRelNode*>& scans) {
    std::vector<std::string> tables;
    for (const auto* scanNode : scans) {
        if (auto* scan = scanDetail(scanNode))
            tables.push_back(scan->table);
    }
    std::sort(tables.begin(), tables.end());
    return tables;
}

std::optional<std::vector<std::string>> outerPredicateSignaturesForHaving(
        const MultiTableGroupedAggShape& shape) {
    std::vector<std::string> signatures;
    for (const auto* joinNode : shape.joins) {
        auto* join = joinNode ? std::get_if<GenericJoinDetail>(&joinNode->detail) : nullptr;
        if (!join) return std::nullopt;
        if (!collectGenericPredicateAtomSignatures(join->predicate, signatures))
            return std::nullopt;
    }
    if (auto* filter = filterDetail(shape.filter)) {
        if (!collectGenericPredicateAtomSignatures(filter->predicate, signatures))
            return std::nullopt;
    }
    std::sort(signatures.begin(), signatures.end());
    return signatures;
}

bool validateHavingAggregateIndex(const GenericAggregateDetail& aggregate,
                                  int aggIdx,
                                  const GenericAggregateExpr*& agg,
                                  std::string* error) {
    if (aggIdx < 0 || aggIdx >= static_cast<int>(aggregate.aggregates.size())) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING aggregate index is out of range.";
        return false;
    }

    agg = aggregate.aggregates[(size_t)aggIdx].expr
        ? std::get_if<GenericAggregateExpr>(&aggregate.aggregates[(size_t)aggIdx].expr->node)
        : nullptr;
    if (!agg || agg->func == AggFunc::COUNT_DISTINCT) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING over COUNT(DISTINCT) is not supported yet.";
        return false;
    }
    return true;
}

bool configureAggregateScalarHaving(const GenericAggregateDetail& aggregate,
                                    int aggIdx,
                                    CmpOp op,
                                    int sqIdx,
                                    int sentinel,
                                    const AnalyzedQuery* aq,
                                    const MultiTableGroupedAggShape* shape,
                                    GenericGroupSpec& groupSpec,
                                    std::string* error) {
    if (!aq || !shape) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING requires analyzed subquery metadata.";
        return false;
    }

    const GenericAggregateExpr* agg = nullptr;
    if (!validateHavingAggregateIndex(aggregate, aggIdx, agg, error))
        return false;
    auto summary = parseScalarHavingSubquerySummary(*aq, sqIdx);
    if (!summary) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING is not a supported ungrouped aggregate subquery.";
        return false;
    }
    if (summary->aggregate.multiplier <= 0.0) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING multiplier must be positive.";
        return false;
    }
    if (summary->aggregate.func != agg->func) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING aggregate function differs from grouped aggregate.";
        return false;
    }
    if (summary->aggregate.star != agg->star) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING COUNT(*) shape differs from grouped aggregate.";
        return false;
    }
    if (!agg->star) {
        auto argSig = genericExprSignatureForIr(agg->arg);
        if (!argSig || *argSig != summary->aggregate.argSignature) {
            if (error)
                *error = "IR grouped aggregate lowerer: scalar-subquery HAVING aggregate input differs from grouped aggregate.";
            return false;
        }
    }

    if (scanTableSignaturesForHaving(shape->scans) != summary->tables) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING table set differs from outer grouped input.";
        return false;
    }

    auto outerPredicates = outerPredicateSignaturesForHaving(*shape);
    if (!outerPredicates || *outerPredicates != summary->predicateSignatures) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING predicates differ from outer grouped input.";
        return false;
    }

    groupSpec.havingAggIdx = aggIdx;
    groupSpec.havingMultiplier = summary->aggregate.multiplier;
    groupSpec.havingSentinel = sentinel;
    groupSpec.havingScalarCompareOp = cmpOpToMetal(op);
    return true;
}

bool configureAggregateHaving(const GenericAggregateDetail& aggregate,
                              GenericGroupSpec& groupSpec,
                              const AnalyzedQuery* aq,
                              const MultiTableGroupedAggShape* shape,
                              std::string* error) {
    if (!aggregate.having) return true;

    auto* cmp = std::get_if<GenericComparisonPred>(&aggregate.having->node);
    if (!cmp) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING currently supports aggregate comparisons only.";
        return false;
    }

    auto aggIdx = aggregateIndexForHavingExpr(cmp->left, aggregate);
    CmpOp op = cmp->op;
    auto sqIdx = scalarSubqueryIndexFromSentinelLiteral(cmp->right);
    int sentinel = 0;
    if (sqIdx) {
        auto* lit = std::get_if<GenericLiteralExpr>(&cmp->right->node);
        auto* value = lit ? std::get_if<int64_t>(&lit->value) : nullptr;
        sentinel = value ? static_cast<int>(*value) : 0;
    }

    if (!aggIdx || !sqIdx) {
        aggIdx = aggregateIndexForHavingExpr(cmp->right, aggregate);
        op = reverseCmpOp(cmp->op);
        sqIdx = scalarSubqueryIndexFromSentinelLiteral(cmp->left);
        if (sqIdx) {
            auto* lit = std::get_if<GenericLiteralExpr>(&cmp->left->node);
            auto* value = lit ? std::get_if<int64_t>(&lit->value) : nullptr;
            sentinel = value ? static_cast<int>(*value) : 0;
        }
    }

    if (aggIdx && sqIdx) {
        return configureAggregateScalarHaving(aggregate, *aggIdx, op, *sqIdx,
                                              sentinel, aq, shape, groupSpec, error);
    }

    aggIdx = aggregateIndexForHavingExpr(cmp->left, aggregate);
    auto literal = numericLiteralValue(cmp->right);
    op = cmp->op;

    if (!aggIdx || !literal) {
        aggIdx = aggregateIndexForHavingExpr(cmp->right, aggregate);
        literal = numericLiteralValue(cmp->left);
        op = reverseCmpOp(cmp->op);
    }

    if ((literal && isScalarSubquerySentinelLiteral(cmp->left)) ||
        (literal && isScalarSubquerySentinelLiteral(cmp->right))) {
        if (error)
            *error = "IR grouped aggregate lowerer: scalar-subquery HAVING requires decorrelation.";
        return false;
    }

    if (!aggIdx || !literal) {
        if (error)
            *error = "IR grouped aggregate lowerer: HAVING must compare a projected aggregate with a numeric literal.";
        return false;
    }
    const GenericAggregateExpr* agg = nullptr;
    if (!validateHavingAggregateIndex(aggregate, *aggIdx, agg, error))
        return false;

    groupSpec.havingCompareAggIdx = *aggIdx;
    groupSpec.havingCompareOp = cmpOpToMetal(op);
    groupSpec.havingCompareValue = *literal;
    return true;
}

std::string groupDisplayNameForExpr(const GenericExprPtr& expr, size_t index) {
    if (expr) {
        if (auto* col = std::get_if<GenericColumnExpr>(&expr->node))
            return col->column.empty() ? "group_" + std::to_string(index) : col->column;
        if (auto* fn = std::get_if<GenericFunctionExpr>(&expr->node))
            return fn->name.empty() ? "group_" + std::to_string(index) : fn->name;
    }
    return "group_" + std::to_string(index);
}

std::string groupDisplayNameForAggregate(const GenericAggregateDetail& aggregate,
                                         size_t index) {
    if (index < aggregate.groupNames.size() && !aggregate.groupNames[index].empty())
        return aggregate.groupNames[index];
    if (index < aggregate.groupBy.size())
        return groupDisplayNameForExpr(aggregate.groupBy[index], index);
    return "group_" + std::to_string(index);
}

std::string aggregateOutputFuncFor(const GenericAggregateDetail& aggregate,
                                   size_t index,
                                   AggFunc fallback) {
    if (index < aggregate.aggregateOutputFuncs.size() &&
        !aggregate.aggregateOutputFuncs[index].empty()) {
        return aggregate.aggregateOutputFuncs[index];
    }
    return aggFuncName(fallback);
}

bool aggregateNeedsHashGroupOutput(const GenericAggregateDetail& aggregate) {
    for (const auto& func : aggregate.aggregateOutputFuncs) {
        if (func == "RATIO" || func == "RATIO_DEN")
            return true;
    }
    return false;
}

const GenericRelation* relationForScan(const GenericRelPlan& ir,
                                       const GenericRelNode* scanNode) {
    auto* scan = scanDetail(scanNode);
    if (!scan) return nullptr;
    auto* inst = ir.findRelationInstance(scan->relationInstance);
    if (!inst) return nullptr;
    return ir.findRelation(inst->relation);
}

void collectPredicateRelations(const GenericPredicatePtr& pred,
                               std::set<int>& relationInstances);

void collectExprRelations(const GenericExprPtr& expr,
                          std::set<int>& relationInstances) {
    if (!expr) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (node.relationInstance.valid())
                relationInstances.insert(node.relationInstance.value);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            collectExprRelations(node.left, relationInstances);
            collectExprRelations(node.right, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches) {
                collectPredicateRelations(branch.condition, relationInstances);
                collectExprRelations(branch.result, relationInstances);
            }
            collectExprRelations(node.elseResult, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                collectExprRelations(arg, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            collectExprRelations(node.arg, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericScalarLookupExpr>) {
            for (const auto& key : node.keys)
                collectExprRelations(key, relationInstances);
        }
    }, expr->node);
}

void collectPredicateRelations(const GenericPredicatePtr& pred,
                               std::set<int>& relationInstances) {
    if (!pred) return;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            collectExprRelations(node.left, relationInstances);
            collectExprRelations(node.right, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            collectExprRelations(node.expr, relationInstances);
            collectExprRelations(node.low, relationInstances);
            collectExprRelations(node.high, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            collectExprRelations(node.expr, relationInstances);
            for (const auto& value : node.values)
                collectExprRelations(value, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            collectExprRelations(node.expr, relationInstances);
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children)
                collectPredicateRelations(child, relationInstances);
        }
    }, pred->node);
}

void splitConjuncts(const GenericPredicatePtr& pred,
                    std::vector<GenericPredicatePtr>& out) {
    if (!pred) return;
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        if (logical->op == GenericLogicalPred::Op::And) {
            for (const auto& child : logical->children)
                splitConjuncts(child, out);
            return;
        }
    }
    out.push_back(pred);
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

struct IrCarryColumn {
    GenericColumnExpr column;
    std::string varName;
    std::string bufferName;
};

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

using IrCarryMap = std::map<int, std::map<std::string, IrCarryColumn>>;

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

struct MultiTableJoinLowering {
    MetalQueryPlan plan;
    std::unique_ptr<MetalOperator> probePipe;
    const GenericScanDetail* probeScan = nullptr;
    std::string outputSize;
    IrCarryMap carryMap;
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

bool hasScalarSubqueries(const AnalyzedQuery* aq) {
    if (!aq) return false;
    for (const auto& sq : aq->subqueries) {
        if (sq.type == AnalyzedQuery::Subquery::SCALAR_SUBQUERY)
            return true;
    }
    return false;
}

bool groupedAggregateNeedsScalarPreAgg(const MultiTableGroupedAggShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateReferencesScalarSentinel(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinNode
            ? std::get_if<GenericJoinDetail>(&joinNode->detail)
            : nullptr;
        if (join && predicateReferencesScalarSentinel(join->predicate))
            return true;
    }
    return false;
}

bool materializeNeedsScalarPreAgg(const MultiTableMaterializeShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateReferencesScalarSentinel(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinNode
            ? std::get_if<GenericJoinDetail>(&joinNode->detail)
            : nullptr;
        if (join && predicateReferencesScalarSentinel(join->predicate))
            return true;
    }
    return false;
}

bool predicateContainsEmptyInList(const GenericPredicatePtr& pred) {
    if (!pred) return false;
    return std::visit([&](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericInListPred>) {
            return node.values.empty();
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            for (const auto& child : node.children) {
                if (predicateContainsEmptyInList(child)) return true;
            }
            return false;
        }
        return false;
    }, pred->node);
}

bool materializeHasEmptyInListPlaceholder(const MultiTableMaterializeShape& shape) {
    if (auto* filter = filterDetail(shape.filter)) {
        if (predicateContainsEmptyInList(filter->predicate)) return true;
    }
    for (const auto* joinNode : shape.joins) {
        auto* join = joinNode
            ? std::get_if<GenericJoinDetail>(&joinNode->detail)
            : nullptr;
        if (join && predicateContainsEmptyInList(join->predicate))
            return true;
    }
    return false;
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
    plan.name = "ADHOC_IR_FROM_SUBQUERY_TOP_SCALAR";
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
        appendPhase(plan, "ADHOC_ir_from_subquery_aggregate_" + tag,
                    std::move(agg));
    }

    {
        auto range = std::make_unique<MetalRangeScan>(sizeSymbol, idxVar);
        auto extremum = std::make_unique<MetalIrAtomicExtremumFloatArray>(
            std::move(range), aggBuffer, aggSeenBuffer, extremumBuffer,
            extremumState, idxVar, scalarExtremum.func == AggFunc::MAX);
        appendPhase(plan, "ADHOC_ir_from_subquery_extremum_" + tag,
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
            plan, "ADHOC_ir_from_subquery_materialize_" + tag,
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
    plan.name = "ADHOC_IR_FROM_SUBQUERY_HISTOGRAM";
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
        appendPhase(plan, "ADHOC_ir_from_subquery_count_" + tag,
                    std::move(count));
    }

    {
        std::set<std::string> scanCols{groupJoinCol};
        auto scan = makeScanForCols(groupBase, idxVar, scanCols, aq.schema);
        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(scan), "d_ir_from_subquery_" + tag + "_result_count",
            "1");

        const std::string outputSize = tableSizeName(groupBase);
        const std::string groupKeyExpr = groupJoinCol + "[" + idxVar + "]";
        const std::string countExpr =
            "(int)atomic_load_explicit(&" + countBuffer + "[" +
            groupKeyExpr + "], memory_order_relaxed)";
        const std::string outerCountName =
            analyzedDisplayNameForTarget(*outerCount, outerCountIndex);
        std::vector<GenericMatColumnDesc> materializedCols;

        const std::string countCol = "d_ir_from_subquery_" + tag + "_0_" +
            sanitizeIdentifier(innerAggAlias);
        const std::string outerCountCol = "d_ir_from_subquery_" + tag + "_1_" +
            sanitizeIdentifier(outerCountName);
        materialize->addColumn(countCol, "int", countExpr,
                               innerAggAlias, outputSize);
        materializedCols.push_back({innerAggAlias, countCol, "int", 0, 0, false});
        materialize->addColumn(outerCountCol, "float", "1.0f",
                               outerCountName, outputSize);
        materializedCols.push_back({outerCountName, outerCountCol,
                                    "float", 0, 0, false});

        auto& phase = appendPhase(
            plan, "ADHOC_ir_from_subquery_materialize_" + tag,
            std::move(materialize));
        phase.extraBuffers.push_back({countBuffer, "atomic_uint", true, false});

        GenericGroupSpec groupSpec;
        groupSpec.keyColumns.push_back(innerAggAlias);
        groupSpec.aggColumns.push_back(outerCountName);
        groupSpec.aggFuncs.push_back("COUNT");

        const std::string groupTag = "ir_from_subquery_hist_" + tag;
        GenericGpuGroupSpec gbSpec;
        gbSpec.tag = groupTag;
        gbSpec.inputCounter = "d_ir_from_subquery_" + tag + "_result_count";
        gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
        gbSpec.capacityExpr = "next_pow2(" + outputSize + " * 2)";
        gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
        gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
        gbSpec.inputColumns = materializedCols;
        gbSpec.groupBy = groupSpec;
        attachMaterializedCountHook(phase, gbSpec.inputCounter,
                                    gbSpec.inputRowsSymbol);
        appendGenericGpuGroupBy(plan, gbSpec);
        attachMaterializedCountHook(plan.phases.back(), gbSpec.outputCounter,
                                    "n_gpu_sort_" + groupTag + "_rows");

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
                                      "n_gpu_sort_" + groupTag + "_rows",
                                      gbSpec.capacityExpr,
                                      genericGpuGroupOutputColumns(gbSpec),
                                      sortSpec, error)) {
                return std::nullopt;
            }
        }
    }

    return plan;
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

const IrCarryColumn* findCarry(const IrCarryMap& carries,
                               const GenericColumnExpr& col) {
    auto relIt = carries.find(col.relationInstance.value);
    if (relIt == carries.end()) return nullptr;
    auto colIt = relIt->second.find(col.column);
    if (colIt == relIt->second.end()) return nullptr;
    return &colIt->second;
}

std::optional<std::string> charStringComparisonToMetalWithCarryMap(
        const GenericExprPtr& left,
        CmpOp op,
        const GenericExprPtr& right,
        const std::string& idxVar,
        const IrCarryMap& carries) {
    if (op != CmpOp::EQ && op != CmpOp::NE) return std::nullopt;

    const GenericColumnExpr* col = left ? std::get_if<GenericColumnExpr>(&left->node) : nullptr;
    auto literal = stringLiteralValue(right);
    if (!col || !literal) {
        col = right ? std::get_if<GenericColumnExpr>(&right->node) : nullptr;
        literal = stringLiteralValue(left);
    }
    if (!col || !literal) return std::nullopt;

    std::string eq;
    const IrCarryColumn* carry = findCarry(carries, *col);
    if (col->type.type == DataType::CHAR_FIXED) {
        int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
        eq = carry
            ? fixedStringEqMetalFromPointer(carry->varName, width, *literal)
            : fixedStringEqMetal(*col, *literal, idxVar);
    } else if (col->type.type == DataType::CHAR1) {
        if (literal->empty()) return std::nullopt;
        std::string valueExpr = carry
            ? carry->varName
            : col->column + "[" + idxVar + "]";
        eq = "(" + valueExpr + " == " + genericMetalCharLiteral(literal->front()) + ")";
    } else {
        return std::nullopt;
    }
    return op == CmpOp::NE ? "!(" + eq + ")" : eq;
}

std::optional<std::string> fixedStringLikeMetalWithCarryMap(
        const GenericLikePred& like,
        const std::string& idxVar,
        const IrCarryMap& carries) {
    auto* col = like.expr ? std::get_if<GenericColumnExpr>(&like.expr->node) : nullptr;
    if (!col || col->type.type != DataType::CHAR_FIXED ||
        like.pattern.find('_') != std::string::npos ||
        like.pattern.find('\\') != std::string::npos) {
        return std::nullopt;
    }

    const IrCarryColumn* carry = findCarry(carries, *col);
    if (!carry)
        return fixedStringLikeMetal(like, idxVar);

    int width = col->type.fixedWidth > 0 ? col->type.fixedWidth : 1;
    if (like.pattern.find('%') == std::string::npos) {
        std::string exact = fixedStringEqMetalFromPointer(
            carry->varName, width, like.pattern);
        return like.negated ? "!(" + exact + ")" : exact;
    }

    return fixedStringLikeDataMetal(carry->varName, "0u", width,
                                    like.pattern, like.negated);
}

int materializedStringLenForExpr(const GenericExprPtr& expr,
                                 const IrCarryMap& carries) {
    if (expr) {
        if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
            if (findCarry(carries, *col) && col->type.type == DataType::CHAR1)
                return 0;
        }
    }
    return fixedStringLenForExpr(expr);
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

std::string genericExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                           const std::string& idxVar,
                                           const IrCarryMap& carries);
std::string genericPredicateToMetalWithCarryMap(const GenericPredicatePtr& pred,
                                                const std::string& idxVar,
                                                const IrCarryMap& carries);

std::string functionExprToMetalWithCarryMap(const GenericFunctionExpr& fn,
                                            const std::string& idxVar,
                                            const IrCarryMap& carries) {
    std::string name = fn.name;
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (name == "date_part" || name == "extract") {
        std::string unit;
        if (!fn.args.empty() && fn.args[0]) {
            if (auto* lit = std::get_if<GenericLiteralExpr>(&fn.args[0]->node)) {
                if (auto* s = std::get_if<std::string>(&lit->value))
                    unit = *s;
            }
        }
        std::string col = fn.args.size() > 1
            ? genericExprToMetalWithCarryMap(fn.args[1], idxVar, carries)
            : "0";
        if (unit == "year") return "(" + col + " / 10000)";
        if (unit == "month") return "((" + col + " / 100) % 100)";
        if (unit == "day") return "(" + col + " % 100)";
        return col;
    }
    return functionExprToMetal(fn, idxVar);
}

std::string genericExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                           const std::string& idxVar,
                                           const IrCarryMap& carries) {
    if (!expr) return "0";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericColumnExpr>) {
            if (const auto* carry = findCarry(carries, node))
                return carry->varName;
            return node.column + "[" + idxVar + "]";
        } else if constexpr (std::is_same_v<T, GenericLiteralExpr>) {
            return literalToMetal(node);
        } else if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            std::string left = genericExprToMetalWithCarryMap(node.left, idxVar, carries);
            std::string right = genericExprToMetalWithCarryMap(node.right, idxVar, carries);
            switch (node.op) {
                case ExprOp::ADD: return "(" + left + " + " + right + ")";
                case ExprOp::SUB: return "(" + left + " - " + right + ")";
                case ExprOp::MUL: return "(" + left + " * " + right + ")";
                case ExprOp::DIV: return "(" + left + " / " + right + ")";
            }
            return left;
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            if (node.branches.empty())
                return genericExprToMetalWithCarryMap(node.elseResult, idxVar, carries);
            std::string result;
            for (const auto& branch : node.branches) {
                result += "((" +
                    genericPredicateToMetalWithCarryMap(branch.condition, idxVar, carries) +
                    ") ? (" +
                    genericExprToMetalWithCarryMap(branch.result, idxVar, carries) +
                    ") : ";
            }
            result += "(" + genericExprToMetalWithCarryMap(node.elseResult, idxVar, carries) + ")";
            for (size_t i = 0; i < node.branches.size(); ++i)
                result += ")";
            return result;
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            return functionExprToMetalWithCarryMap(node, idxVar, carries);
        } else if constexpr (std::is_same_v<T, GenericAggregateExpr>) {
            return node.arg ? genericExprToMetalWithCarryMap(node.arg, idxVar, carries) : "1";
        } else {
            return genericExprToMetal(expr, idxVar);
        }
    }, expr->node);
}

std::string materializeExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                               const std::string& idxVar,
                                               const IrCarryMap& carries) {
    if (!expr) return "0";
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        if (const auto* carry = findCarry(carries, *col))
            return carry->varName;
        return materializeExprToMetal(expr, idxVar);
    }
    return genericExprToMetalWithCarryMap(expr, idxVar, carries);
}

std::string genericPredicateToMetalWithCarryMap(const GenericPredicatePtr& pred,
                                                const std::string& idxVar,
                                                const IrCarryMap& carries) {
    if (!pred) return "true";
    return std::visit([&](const auto& node) -> std::string {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericComparisonPred>) {
            if (auto cmp = charStringComparisonToMetalWithCarryMap(
                    node.left, node.op, node.right, idxVar, carries)) {
                return *cmp;
            }
            return "(" + genericExprToMetalWithCarryMap(node.left, idxVar, carries) +
                   " " + cmpOpToMetal(node.op) + " " +
                   genericExprToMetalWithCarryMap(node.right, idxVar, carries) + ")";
        } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
            std::string expr = genericExprToMetalWithCarryMap(node.expr, idxVar, carries);
            return "(" + expr + " >= " +
                   genericExprToMetalWithCarryMap(node.low, idxVar, carries) +
                   " && " + expr + " <= " +
                   genericExprToMetalWithCarryMap(node.high, idxVar, carries) + ")";
        } else if constexpr (std::is_same_v<T, GenericInListPred>) {
            if (node.values.empty()) return "true";
            std::string expr = genericExprToMetalWithCarryMap(node.expr, idxVar, carries);
            std::string out = "(";
            for (size_t i = 0; i < node.values.size(); ++i) {
                if (i) out += " || ";
                if (auto cmp = charStringComparisonToMetalWithCarryMap(
                        node.expr, CmpOp::EQ, node.values[i], idxVar, carries)) {
                    out += *cmp;
                } else {
                    if (node.expr &&
                        (node.expr->type.type == DataType::INT ||
                         node.expr->type.type == DataType::DATE) &&
                        integerStringLiteralValue(node.values[i])) {
                        out += expr + " == " +
                               std::to_string(*integerStringLiteralValue(node.values[i]));
                    } else {
                        out += expr + " == " +
                               genericExprToMetalWithCarryMap(node.values[i], idxVar, carries);
                    }
                }
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLogicalPred>) {
            if (node.op == GenericLogicalPred::Op::Not) {
                return "!(" + (node.children.empty()
                    ? std::string("true")
                    : genericPredicateToMetalWithCarryMap(node.children.front(),
                                                          idxVar, carries)) + ")";
            }
            if (node.children.empty()) {
                return node.op == GenericLogicalPred::Op::And ? "true" : "false";
            }
            std::string joiner = node.op == GenericLogicalPred::Op::And ? " && " : " || ";
            std::string out = "(";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i) out += joiner;
                out += genericPredicateToMetalWithCarryMap(node.children[i],
                                                           idxVar, carries);
            }
            out += ")";
            return out;
        } else if constexpr (std::is_same_v<T, GenericLikePred>) {
            if (auto like = fixedStringLikeMetalWithCarryMap(node, idxVar, carries))
                return *like;
            return node.negated ? "true" : "false";
        } else {
            return "true";
        }
    }, pred->node);
}

bool exprNeedsCarriedString(const GenericExprPtr& expr,
                            const IrCarryMap& carries) {
    if (!expr) return false;
    if (std::get_if<GenericColumnExpr>(&expr->node)) {
        return false;
    }
    bool found = false;
    std::visit([&](const auto& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, GenericBinaryExpr>) {
            found = exprNeedsCarriedString(node.left, carries) ||
                    exprNeedsCarriedString(node.right, carries);
        } else if constexpr (std::is_same_v<T, GenericFunctionExpr>) {
            for (const auto& arg : node.args)
                found = found || exprNeedsCarriedString(arg, carries);
        } else if constexpr (std::is_same_v<T, GenericCaseExpr>) {
            for (const auto& branch : node.branches)
                found = found || exprNeedsCarriedString(branch.result, carries);
            found = found || exprNeedsCarriedString(node.elseResult, carries);
        }
    }, expr->node);
    return found;
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
                lowering.plan, "ADHOC_ir_multi_table_build_" + buildTag,
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
                lowering.plan, "ADHOC_ir_multi_table_build_" + buildTag,
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
                lowering.plan, "ADHOC_ir_multi_table_agg_" + buildTag,
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
                lowering.plan, "ADHOC_ir_multi_table_build_" + buildTag,
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
                                  "ADHOC_ir_multi_table_build_" + buildTag,
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

} // namespace

std::optional<MetalQueryPlan> lowerFromSubqueryAggregateIRToMetal(
        const AnalyzedQuery& aq,
        std::string* error) {
    if (auto p = lowerFromSubqueryHistogramIRToMetal(aq, error))
        return p;
    return lowerFromSubqueryTopScalarIRToMetal(aq, error);
}

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetalImpl(
        const GenericRelPlan& ir,
        const AnalyzedQuery* aq,
        std::string* error) {
    auto shape = parseMultiTableMaterializeShape(ir, error);
    if (!shape) return std::nullopt;

    auto* project = projectDetail(shape->project);
    if (!project || project->projections.empty())
        return fail(error, "IR multi-table materialize lowerer: no projection columns.");
    if (materializeHasEmptyInListPlaceholder(*shape))
        return std::nullopt;

    MetalQueryPlan scalarPreAggPlan;
    std::vector<GenericScalarLookupInfo> scalarLookups;
    if (hasScalarSubqueries(aq) && materializeNeedsScalarPreAgg(*shape)) {
        scalarPreAggPlan.name = "ADHOC_IR_MULTI_TABLE_MATERIALIZE_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table materialize lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    std::vector<GenericExprPtr> neededExprs;
    for (const auto& projection : project->projections)
        neededExprs.push_back(projection.expr);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys)
            neededExprs.push_back(key.expr);
    }

    std::string sharedLowerError;
    if (auto lowering = buildMultiTableJoinLowering(
            ir, shape->scans, shape->joins, shape->filter, neededExprs,
            "ADHOC_IR_MULTI_TABLE_MATERIALIZE", aq,
            scalarLookups.empty() ? nullptr : &scalarLookups, &sharedLowerError)) {
        prependPlanPhases(lowering->plan, scalarPreAggPlan);
        const std::string idxVar = "i";
        const std::string resultCounter = "d_ir_multi_table_result_count";
        auto materialize = std::make_unique<MetalMaterialize>(
            std::move(lowering->probePipe), resultCounter, "1");
        std::vector<GenericMatColumnDesc> materializedCols;
        for (size_t i = 0; i < project->projections.size(); ++i) {
            const auto& projection = project->projections[i];
            if (!materializeExprSupported(projection.expr))
                return std::nullopt;
            if (exprNeedsCarriedString(projection.expr, lowering->carryMap))
                return std::nullopt;
            int stringLen = materializedStringLenForExpr(projection.expr,
                                                         lowering->carryMap);
            std::string sizeExpr = lowering->outputSize;
            if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
            const std::string bufferName = "d_ir_multi_table_" +
                std::to_string(i) + "_" + sanitizeIdentifier(projection.name);
            const std::string metalType = metalTypeForType(projection.type);
            materialize->addColumn(bufferName, metalType,
                                   materializeExprToMetalWithCarryMap(
                                       projection.expr, idxVar,
                                       lowering->carryMap),
                                   projection.name, sizeExpr, stringLen);
            materializedCols.push_back(GenericMatColumnDesc{
                projection.name, bufferName, metalType, stringLen});
        }

        auto& matPhase = appendPhase(lowering->plan,
                                     "ADHOC_ir_multi_table_materialize",
                                     std::move(materialize));
        if (!scalarLookups.empty())
            attachGenericScalarLookupBuffers(matPhase, scalarLookups);

        GenericSortSpec sortSpec;
        sortSpec.limit = limitValue(shape->limit);
        if (auto* sort = sortDetail(shape->sort)) {
            for (const auto& key : sort->keys) {
                auto name = sortKeyDisplayName(key, *project);
                if (!name)
                    return fail(error, "IR multi-table materialize lowerer: ORDER BY key is not projected.");
                sortSpec.keys.push_back({*name, key.descending});
            }
        }
        if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
            const std::string rowsSym = "n_gpu_sort_ir_multi_table_rows";
            attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
            if (!appendGenericGpuSort(lowering->plan, "ir_multi_table_materialize",
                                      rowsSym, lowering->outputSize,
                                      materializedCols, sortSpec, error)) {
                return std::nullopt;
            }
        }
        return std::move(lowering->plan);
    }

    struct IrBuildSide {
        const GenericRelNode* scanNode = nullptr;
        const GenericScanDetail* scan = nullptr;
        const GenericRelation* relation = nullptr;
        int relationInstance = -1;
        int parentRelationInstance = -1;
        GenericColumnExpr joinCol;
        GenericColumnExpr parentCol;
        std::vector<int> children;
        std::vector<GenericPredicatePtr> filters;
        std::map<std::string, IrCarryColumn> localCarries;
        std::vector<IrCarryColumn> subtreeCarries;
        std::string keyDomain;
        std::string bitmapName;
    };
    struct IrScanSide {
        const GenericRelNode* node = nullptr;
        const GenericScanDetail* scan = nullptr;
        const GenericRelation* relation = nullptr;
    };

    std::vector<IrScanSide> sides;
    std::set<std::string> baseTables;
    for (const auto* scanNode : shape->scans) {
        auto* scan = scanDetail(scanNode);
        auto* relation = relationForScan(ir, scanNode);
        if (!scan || !relation)
            return fail(error, "IR multi-table materialize lowerer: malformed scan metadata.");
        if (!baseTables.insert(scan->table).second)
            return std::nullopt;
        sides.push_back(IrScanSide{scanNode, scan, relation});
    }

    const IrScanSide* probe = nullptr;
    for (const auto& side : sides) {
        if (!probe || side.relation->probePriority > probe->relation->probePriority)
            probe = &side;
    }
    if (!probe) return std::nullopt;

    auto relationIdOf = [](const GenericColumnExpr& col) {
        return col.relationInstance.value;
    };

    std::map<int, IrScanSide> sideByRel;
    for (const auto& side : sides)
        sideByRel[side.scan->relationInstance.value] = side;

    std::vector<IrJoinColumns> edges;
    for (const auto* joinNode : shape->joins) {
        auto* join = std::get_if<GenericJoinDetail>(&joinNode->detail);
        if (!join || !join->predicate) return std::nullopt;
        std::vector<GenericPredicatePtr> conjuncts;
        splitConjuncts(join->predicate, conjuncts);
        for (const auto& pred : conjuncts) {
            IrJoinColumns edge;
            if (!extractEqJoinColumnsFromPredicate(pred, edge))
                return std::nullopt;
            if (!sideByRel.count(relationIdOf(edge.left)) ||
                !sideByRel.count(relationIdOf(edge.right))) {
                return std::nullopt;
            }
            edges.push_back(std::move(edge));
        }
    }

    if (edges.size() != sides.size() - 1)
        return std::nullopt;

    std::map<int, std::vector<IrJoinColumns>> adjacency;
    for (const auto& edge : edges) {
        adjacency[relationIdOf(edge.left)].push_back(edge);
        adjacency[relationIdOf(edge.right)].push_back(edge);
    }

    std::map<int, IrBuildSide> buildByRel;
    const int probeRel = probe->scan->relationInstance.value;
    std::set<int> visited{probeRel};
    std::vector<int> bfs{probeRel};
    for (size_t head = 0; head < bfs.size(); ++head) {
        int parentRel = bfs[head];
        for (const auto& edge : adjacency[parentRel]) {
            int leftRel = relationIdOf(edge.left);
            int rightRel = relationIdOf(edge.right);
            int childRel = leftRel == parentRel ? rightRel : leftRel;
            if (visited.count(childRel)) continue;

            const auto& childSide = sideByRel.at(childRel);
            IrBuildSide build;
            build.scanNode = childSide.node;
            build.scan = childSide.scan;
            build.relation = childSide.relation;
            build.relationInstance = childRel;
            build.parentRelationInstance = parentRel;
            if (leftRel == childRel) {
                build.joinCol = edge.left;
                build.parentCol = edge.right;
            } else {
                build.joinCol = edge.right;
                build.parentCol = edge.left;
            }
        if (build.relation->primaryKeyColumn.empty() ||
            build.relation->primaryKeyColumn != build.joinCol.column) {
            return std::nullopt;
        }
        build.keyDomain = build.joinCol.keyDomainSymbol;
        if (build.keyDomain.empty())
            build.keyDomain = build.relation->primaryKeyDomainSymbol;
        if (build.keyDomain.empty())
            build.keyDomain = build.relation->maxKeySymbol;
        if (build.keyDomain.empty())
            return fail(error, "IR multi-table materialize lowerer: build key has no schema domain symbol.");
        build.bitmapName = "d_ir_join_bitmap_" +
            sanitizeIdentifier(build.scan->alias.empty() ? build.scan->table
                                                         : build.scan->alias);
            buildByRel[childRel] = std::move(build);
            buildByRel[parentRel].children.push_back(childRel);
            visited.insert(childRel);
            bfs.push_back(childRel);
        }
    }
    if (visited.size() != sides.size() || buildByRel.empty())
        return std::nullopt;

    std::vector<GenericPredicatePtr> probeFilters;
    std::vector<GenericPredicatePtr> crossFilters;
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate))
            return std::nullopt;
        std::vector<GenericPredicatePtr> conjuncts;
        splitConjuncts(filter->predicate, conjuncts);
        for (const auto& pred : conjuncts) {
            std::set<int> rels;
            collectPredicateRelations(pred, rels);
            if (rels.empty() ||
                (rels.size() == 1 && rels.count(probe->scan->relationInstance.value))) {
                probeFilters.push_back(pred);
            } else if (rels.size() == 1) {
                auto it = buildByRel.find(*rels.begin());
                if (it == buildByRel.end()) return std::nullopt;
                it->second.filters.push_back(pred);
            } else {
                crossFilters.push_back(pred);
            }
        }
    }

    std::map<int, std::map<std::string, GenericColumnExpr>> neededByRel;
    auto addNeeded = [&](const GenericExprPtr& expr) {
        for (const auto& [rel, side] : sideByRel) {
            if (rel == probeRel) continue;
            collectColumnsForRelation(expr, side.scan->relationInstance,
                                      neededByRel[rel]);
        }
    };
    for (const auto& projection : project->projections)
        addNeeded(projection.expr);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys)
            addNeeded(key.expr);
    }
    for (const auto& pred : crossFilters) {
        std::visit([&](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, GenericComparisonPred>) {
                addNeeded(node.left);
                addNeeded(node.right);
            } else if constexpr (std::is_same_v<T, GenericBetweenPred>) {
                addNeeded(node.expr);
                addNeeded(node.low);
                addNeeded(node.high);
            } else if constexpr (std::is_same_v<T, GenericInListPred>) {
                addNeeded(node.expr);
                for (const auto& value : node.values) addNeeded(value);
            }
        }, pred->node);
    }

    for (auto& [rel, build] : buildByRel) {
        if (rel == probeRel) continue;
        std::map<std::string, GenericColumnExpr> needed;
        auto neededIt = neededByRel.find(rel);
        if (neededIt != neededByRel.end()) needed = neededIt->second;
        for (const auto& [name, col] : needed) {
            if (!typeCanUseArrayCarry(col.type.type))
                return std::nullopt;
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
    MetalQueryPlan plan;
    plan.name = "ADHOC_IR_MULTI_TABLE_MATERIALIZE";

    for (int rel : postorder) {
        const auto& build = buildByRel.at(rel);
        const std::string buildKeyExpr = build.joinCol.column + "[" + idxVar + "]";
        std::unique_ptr<MetalOperator> buildPipe =
            makeAutoScan(build.scan->table, idxVar);
        bool buildScalarLookupsLoaded = false;
        bool buildUsesScalarLookupBuffer = false;
        for (const auto& pred : build.filters) {
            std::string cond = genericPredicateToMetal(pred, idxVar);
            if (!scalarLookups.empty() &&
                referencesGenericScalarSentinel(cond, scalarLookups) &&
                !buildScalarLookupsLoaded) {
                buildPipe = appendScalarLookupLoads(
                    std::move(buildPipe), &scalarLookups, idxVar,
                    build.scan->table, aq ? aq->schema : nullptr);
                buildScalarLookupsLoaded = true;
            }
            cond = rewriteScalarLookupsInCondition(
                std::move(cond),
                scalarLookups.empty() ? nullptr : &scalarLookups,
                idxVar, build.scan->table, aq ? aq->schema : nullptr);
            buildUsesScalarLookupBuffer =
                buildUsesScalarLookupBuffer ||
                (!scalarLookups.empty() &&
                 referencesGenericScalarLookupBuffer(cond, scalarLookups));
            buildPipe = maybeSelect(std::move(buildPipe), cond);
        }

        for (int childRel : build.children) {
            const auto& child = buildByRel.at(childRel);
            const std::string childProbeKeyExpr = child.parentCol.column + "[" + idxVar + "]";
            buildPipe = std::make_unique<MetalBitmapProbe>(
                std::move(buildPipe), child.bitmapName, childProbeKeyExpr);
            for (const auto& carry : child.subtreeCarries) {
                buildPipe = appendCarryLookup(std::move(buildPipe), *child.scan,
                                               carry, childProbeKeyExpr);
            }
        }

        buildPipe = std::make_unique<MetalBitmapBuild>(
            std::move(buildPipe), build.bitmapName, buildKeyExpr,
            "(" + build.keyDomain + " + 31) / 32");
        for (const auto& carry : build.subtreeCarries) {
            buildPipe = appendCarryStore(std::move(buildPipe), *build.scan,
                                         carry, buildKeyExpr, rel, idxVar,
                                         build.keyDomain);
        }
        auto& buildPhase = appendPhase(plan, "ADHOC_ir_multi_table_build_" +
                          sanitizeIdentifier(build.scan->alias.empty()
                              ? build.scan->table : build.scan->alias),
                    std::move(buildPipe));
        if (buildUsesScalarLookupBuffer)
            attachGenericScalarLookupBuffers(buildPhase, scalarLookups);
    }

    std::unique_ptr<MetalOperator> probePipe =
        makeAutoScan(probe->scan->table, idxVar);
    bool probeScalarLookupsLoaded = false;
    bool probeUsesScalarLookupBuffer = false;
    for (const auto& pred : probeFilters) {
        std::string cond = genericPredicateToMetal(pred, idxVar);
        if (!scalarLookups.empty() &&
            referencesGenericScalarSentinel(cond, scalarLookups) &&
            !probeScalarLookupsLoaded) {
            probePipe = appendScalarLookupLoads(
                std::move(probePipe), &scalarLookups, idxVar,
                probe->scan->table, aq ? aq->schema : nullptr);
            probeScalarLookupsLoaded = true;
        }
        cond = rewriteScalarLookupsInCondition(
            std::move(cond),
            scalarLookups.empty() ? nullptr : &scalarLookups,
            idxVar, probe->scan->table, aq ? aq->schema : nullptr);
        probeUsesScalarLookupBuffer =
            probeUsesScalarLookupBuffer ||
            (!scalarLookups.empty() &&
             referencesGenericScalarLookupBuffer(cond, scalarLookups));
        probePipe = maybeSelect(std::move(probePipe), cond);
    }

    IrCarryMap carryMap;
    for (int childRel : buildByRel[probeRel].children) {
        const auto& build = buildByRel.at(childRel);
        const std::string probeKeyExpr = build.parentCol.column + "[" + idxVar + "]";
        probePipe = std::make_unique<MetalBitmapProbe>(
            std::move(probePipe), build.bitmapName, probeKeyExpr);
        for (const auto& carry : build.subtreeCarries) {
            probePipe = appendCarryLookup(std::move(probePipe), *build.scan,
                                          carry, probeKeyExpr);
            carryMap[carry.column.relationInstance.value][carry.column.column] = carry;
        }
    }

    for (const auto& pred : crossFilters) {
        std::string cond = genericPredicateToMetalWithCarryMap(pred, idxVar,
                                                               carryMap);
        if (!scalarLookups.empty() &&
            referencesGenericScalarSentinel(cond, scalarLookups) &&
            !probeScalarLookupsLoaded) {
            probePipe = appendScalarLookupLoads(
                std::move(probePipe), &scalarLookups, idxVar,
                probe->scan->table, aq ? aq->schema : nullptr);
            probeScalarLookupsLoaded = true;
        }
        cond = rewriteScalarLookupsInCondition(
            std::move(cond),
            scalarLookups.empty() ? nullptr : &scalarLookups,
            idxVar, probe->scan->table, aq ? aq->schema : nullptr);
        probeUsesScalarLookupBuffer =
            probeUsesScalarLookupBuffer ||
            (!scalarLookups.empty() &&
             referencesGenericScalarLookupBuffer(cond, scalarLookups));
        probePipe = maybeSelect(std::move(probePipe), cond);
    }

    const std::string resultCounter = "d_ir_multi_table_result_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(probePipe), resultCounter, "1");
    const std::string outputSize = tableSizeName(probe->scan->table);
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t i = 0; i < project->projections.size(); ++i) {
        const auto& projection = project->projections[i];
        if (!materializeExprSupported(projection.expr))
            return std::nullopt;
        if (exprNeedsCarriedString(projection.expr, carryMap))
            return std::nullopt;
        int stringLen = materializedStringLenForExpr(projection.expr, carryMap);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        const std::string bufferName = "d_ir_multi_table_" + std::to_string(i) +
            "_" + sanitizeIdentifier(projection.name);
        const std::string metalType = metalTypeForType(projection.type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetalWithCarryMap(
                                   projection.expr, idxVar, carryMap),
                               projection.name, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            projection.name, bufferName, metalType, stringLen});
    }

    auto& matPhase = appendPhase(plan, "ADHOC_ir_multi_table_materialize",
                                 std::move(materialize));
    if (probeUsesScalarLookupBuffer)
        attachGenericScalarLookupBuffers(matPhase, scalarLookups);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayName(key, *project);
            if (!name)
                return fail(error, "IR multi-table materialize lowerer: ORDER BY key is not projected.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }
    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_multi_table_rows";
        attachMaterializedCountHook(matPhase, resultCounter, rowsSym);
        if (!appendGenericGpuSort(plan, "ir_multi_table_materialize", rowsSym,
                                  outputSize, materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
    }
    prependPlanPhases(plan, scalarPreAggPlan);
    return plan;
}

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableMaterializeIRToMetalImpl(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableMaterializeIRToMetalImpl(ir, &aq, error);
}

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
        scalarPreAggPlan.name = "ADHOC_IR_MULTI_TABLE_GROUP_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table grouped aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "ADHOC_IR_MULTI_TABLE_GROUP", aq,
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
        int stringLen = materializedStringLenForExpr(expr, lowering->carryMap);
        std::string sizeExpr = lowering->outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_multi_group_" + std::to_string(matColIdx++) +
                                 "_" + sanitizeIdentifier(displayName);
        std::string metalType = metalTypeForType(type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetalWithCarryMap(
                                   expr, idxVar, lowering->carryMap),
                               displayName, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            displayName, bufferName, metalType, stringLen, scaleDown, false,
            distinctDomainSymbol});
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
                distinctDomainSymbol = distinctDomainSymbolForExpr(agg->arg);
                if (distinctDomainSymbol.empty())
                    return fail(error, "IR multi-table grouped aggregate lowerer: COUNT(DISTINCT) has no schema distinct-domain metadata.");
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

    auto& matPhase = appendPhase(lowering->plan, "ADHOC_ir_multi_table_group_materialize",
                                 std::move(materialize));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(matPhase, scalarLookups);

    const std::string groupTag = "ir_multi_table_group";
    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = "next_pow2(" + lowering->outputSize + " * 2)";
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
    gbSpec.inputColumns = std::move(materializedCols);
    gbSpec.groupBy = std::move(groupSpec);
    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
    appendGenericGpuGroupBy(lowering->plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(lowering->plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

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

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        if (!appendGenericGpuSort(lowering->plan, "group_" + groupTag,
                                  sortRowsSym, gbSpec.capacityExpr,
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

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
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
        scalarPreAggPlan.name = "ADHOC_IR_MULTI_TABLE_SCALAR_PREAGG";
        scalarLookups = buildGenericScalarPreAggs(*aq, scalarPreAggPlan);
        if (scalarLookups.empty()) {
            return fail(error, "IR multi-table scalar aggregate lowerer: scalar subquery decorrelation is not supported.");
        }
    }

    auto lowering = buildMultiTableJoinLowering(
        ir, shape->scans, shape->joins, shape->filter, neededExprs,
        "ADHOC_IR_MULTI_TABLE_SCALAR", aq,
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

    auto& scalarPhase = appendPhase(lowering->plan, "ADHOC_ir_multi_table_scalar",
                                    std::move(reduce));
    if (!scalarLookups.empty())
        attachGenericScalarLookupBuffers(scalarPhase, scalarLookups);
    return std::move(lowering->plan);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetal(ir, nullptr, error);
}

std::optional<MetalQueryPlan> lowerMultiTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        const AnalyzedQuery& aq,
        std::string* error) {
    return lowerMultiTableScalarAggregateIRToMetal(ir, &aq, error);
}

bool canUseKeyedSingleTableGroup(const GenericAggregateDetail& aggregate) {
    int totalBuckets = 1;
    for (const auto& group : aggregate.groupBy) {
        auto* col = group ? std::get_if<GenericColumnExpr>(&group->node) : nullptr;
        if (!col) return false;

        int numValues = 0;
        if (col->type.type == DataType::CHAR1) {
            numValues = static_cast<int>(col->charDomain.size());
        } else if ((col->type.type == DataType::INT ||
                    col->type.type == DataType::DATE) &&
                   col->hasGroupDomain &&
                   col->domainMax >= col->domainMin) {
            numValues = col->domainMax - col->domainMin + 1;
        } else {
            return false;
        }

        if (numValues <= 0) return false;
        totalBuckets *= numValues;
        if (totalBuckets > 4096) return false;
    }
    return true;
}

std::optional<MetalQueryPlan> lowerSingleTableHashGroupedAggregateIRToMetal(
        const GenericScanDetail& scan,
        const GenericAggregateDetail& aggregate,
        const GenericRelNode* filterNode,
        const GenericRelNode* sortNode,
        const GenericRelNode* limitNode,
        std::string* error) {
    const std::string idxVar = "i";
    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan.table, idxVar);
    if (auto* filter = filterDetail(filterNode)) {
        if (!predicateSupported(filter->predicate))
            return fail(error, "IR single-table hash group lowerer: filter predicate unsupported.");
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }

    const std::string resultCounter = "d_ir_single_hash_group_input_count";
    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(pipe), resultCounter, "1");
    const std::string outputSize = tableSizeName(scan.table);

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
                *error = "IR single-table hash group lowerer: input expression '" +
                         displayName + "' is not supported.";
            return false;
        }
        int stringLen = fixedStringLenForExpr(expr);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_single_hash_group_" +
                                 std::to_string(matColIdx++) + "_" +
                                 sanitizeIdentifier(displayName);
        std::string metalType = metalTypeForType(type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetal(expr, idxVar),
                               displayName, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            displayName, bufferName, metalType, stringLen, scaleDown, false,
            distinctDomainSymbol});
        return true;
    };

    for (size_t i = 0; i < aggregate.groupBy.size(); ++i) {
        const auto& group = aggregate.groupBy[i];
        const std::string displayName = groupDisplayNameForAggregate(aggregate, i);
        groupSpec.keyColumns.push_back(displayName);
        IrGroupKeyDesc key;
        key.displayName = displayName;
        groupKeys.push_back(std::move(key));
        if (!addInputColumn(displayName,
                            group ? group->type : TypeInfo{DataType::INT, 0},
                            group, 0, "")) {
            return std::nullopt;
        }
    }

    for (size_t i = 0; i < aggregate.aggregates.size(); ++i) {
        const auto& projection = aggregate.aggregates[i];
        auto* agg = projection.expr
            ? std::get_if<GenericAggregateExpr>(&projection.expr->node)
            : nullptr;
        if (!agg)
            return fail(error, "IR single-table hash group lowerer: non-aggregate projection.");
        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;

        GenericExprPtr inputExpr;
        TypeInfo inputType{DataType::FLOAT, 0};
        int inputScaleDown = 0;
        std::string distinctDomainSymbol;
        std::string funcName = aggregateOutputFuncFor(aggregate, i, agg->func);

        if (agg->func == AggFunc::COUNT) {
            GenericExpr lit;
            lit.type = inputType;
            lit.node = GenericLiteralExpr{1.0, inputType};
            inputExpr = std::make_shared<GenericExpr>(std::move(lit));
            funcName = "COUNT";
        } else {
            if (!agg->arg)
                return fail(error, "IR single-table hash group lowerer: aggregate '" +
                                   aggFuncName(agg->func) + "' requires an argument.");
            inputExpr = agg->arg;
            if (agg->func == AggFunc::COUNT_DISTINCT) {
                distinctDomainSymbol = distinctDomainSymbolForExpr(agg->arg);
                if (distinctDomainSymbol.empty())
                    return fail(error, "IR single-table hash group lowerer: COUNT(DISTINCT) has no schema distinct-domain metadata.");
                inputType = agg->arg->type;
                funcName = "COUNT_DISTINCT";
            } else if (agg->func == AggFunc::SUM || agg->func == AggFunc::AVG) {
                inputScaleDown = numericScaleForExpr(agg->arg);
            } else if (agg->func != AggFunc::MIN && agg->func != AggFunc::MAX) {
                return fail(error, "IR single-table hash group lowerer: unsupported aggregate " +
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

    groupSpec.outputColumns = aggregate.outputOrder;
    if (!configureAggregateHaving(aggregate, groupSpec, nullptr, nullptr, error))
        return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "ADHOC_IR_SINGLE_TABLE_HASH_GROUP";
    auto& matPhase = appendPhase(plan, "ADHOC_ir_single_table_hash_group_materialize",
                                 std::move(materialize));

    const std::string groupTag = "ir_single_table_hash_group";
    GenericGpuGroupSpec gbSpec;
    gbSpec.tag = groupTag;
    gbSpec.inputCounter = resultCounter;
    gbSpec.inputRowsSymbol = "n_gpu_gb_" + groupTag + "_input";
    gbSpec.capacityExpr = "next_pow2(" + outputSize + " * 2)";
    gbSpec.capacitySymbol = "n_gpu_gb_" + groupTag + "_cap";
    gbSpec.outputCounter = "d_gpu_gb_" + groupTag + "_count";
    gbSpec.inputColumns = std::move(materializedCols);
    gbSpec.groupBy = std::move(groupSpec);
    attachMaterializedCountHook(matPhase, gbSpec.inputCounter, gbSpec.inputRowsSymbol);
    appendGenericGpuGroupBy(plan, gbSpec);

    const std::string sortRowsSym = "n_gpu_sort_" + groupTag + "_rows";
    attachMaterializedCountHook(plan.phases.back(), gbSpec.outputCounter,
                                sortRowsSym);

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(limitNode);
    if (auto* sort = sortDetail(sortNode)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, aggregate, groupKeys);
            if (!name)
                return fail(error, "IR single-table hash group lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        if (!appendGenericGpuSort(plan, "group_" + groupTag,
                                  sortRowsSym, gbSpec.capacityExpr,
                                  genericGpuGroupOutputColumns(gbSpec),
                                  sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

std::optional<MetalQueryPlan> lowerSingleTableGroupedAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto shape = parseSingleTableGroupedAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!scan || !aggregate)
        return fail(error, "IR grouped aggregate lowerer: malformed scan/aggregate detail.");
    if (aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->having)
        return fail(error, "IR grouped aggregate lowerer: HAVING is not supported yet.");
    if (aggregate->aggregates.empty())
        return fail(error, "IR grouped aggregate lowerer: no aggregate outputs.");

    if (aggregateNeedsHashGroupOutput(*aggregate) ||
        !canUseKeyedSingleTableGroup(*aggregate)) {
        return lowerSingleTableHashGroupedAggregateIRToMetal(
            *scan, *aggregate, shape->filter, shape->sort, shape->limit, error);
    }

    const std::string idxVar = "i";
    std::vector<IrGroupKeyDesc> groupKeys;
    int totalBuckets = 1;
    for (size_t i = 0; i < aggregate->groupBy.size(); ++i) {
        auto* col = std::get_if<GenericColumnExpr>(&aggregate->groupBy[i]->node);
        if (!col)
            return std::nullopt;

        IrGroupKeyDesc key;
        key.displayName = groupDisplayNameForAggregate(*aggregate, i);
        key.stride = totalBuckets;
        if (col->type.type == DataType::CHAR1) {
            key.keyExpr = char1BucketExpr(*col, idxVar);
            key.numValues = static_cast<int>(col->charDomain.size());
            key.charMap = col->charDomain;
            if (key.keyExpr.empty() || key.numValues <= 0)
                return fail(error, "IR grouped aggregate lowerer: CHAR1 group key has no schema char domain.");
        } else {
            if (col->domainMin > col->domainMax)
                return fail(error, "IR grouped aggregate lowerer: group key has no schema domain.");
            key.numValues = col->domainMax - col->domainMin + 1;
            key.keyBase = col->domainMin;
            std::string raw = col->column + "[" + idxVar + "]";
            if (col->domainMin != 0)
                key.keyExpr = "(" + raw + " - " + std::to_string(col->domainMin) + ")";
            else
                key.keyExpr = raw;
            key.keyExpr = "clamp(" + key.keyExpr + ", 0, " +
                          std::to_string(key.numValues - 1) + ")";
        }
        if (key.numValues <= 0)
            return fail(error, "IR grouped aggregate lowerer: invalid group key domain.");
        totalBuckets *= key.numValues;
        groupKeys.push_back(std::move(key));
    }
    if (totalBuckets > 4096)
        return fail(error, "IR grouped aggregate lowerer: bucket count exceeds 4096.");

    std::string bucketExpr = "(" + groupKeys.front().keyExpr + ")";
    for (size_t i = 1; i < groupKeys.size(); ++i) {
        bucketExpr = "(" + bucketExpr + " + (" + groupKeys[i].keyExpr + ") * " +
                     std::to_string(groupKeys[i].stride) + ")";
    }

    std::vector<IrPendingAgg> pending;
    int valuesPerBucket = 0;
    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        auto* agg = projection.expr ? std::get_if<GenericAggregateExpr>(&projection.expr->node) : nullptr;
        if (!agg)
            return fail(error, "IR grouped aggregate lowerer: non-aggregate projection.");

        const std::string displayName = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        if (agg->func == AggFunc::COUNT) {
            IrPendingAgg out;
            out.displayName = displayName;
            out.offset = valuesPerBucket++;
            out.valueExpr = "1u";
            out.funcName = "COUNT";
            pending.push_back(std::move(out));
            continue;
        }
        if (!agg->arg)
            return fail(error, "IR grouped aggregate lowerer: aggregate argument required.");
        if (!materializeExprSupported(agg->arg))
            return fail(error, "IR grouped aggregate lowerer: aggregate argument unsupported.");

        if (agg->func == AggFunc::AVG) {
            const bool isFloat = agg->arg->type.type == DataType::FLOAT;
            const int fixedScale = isFloat ? numericScaleForExpr(agg->arg) : 0;
            IrPendingAgg sum;
            sum.displayName = displayName;
            sum.offset = valuesPerBucket;
            std::string valueExpr = genericExprToMetal(agg->arg, idxVar);
            if (isFloat && fixedScale > 0) {
                sum.valueExpr = scaledLongExpr(valueExpr, fixedScale);
                sum.isLongPair = true;
                sum.scaleDown = -fixedScale;
                valuesPerBucket += 2;
            } else if (isFloat) {
                sum.valueExpr = valueExpr;
                sum.isFloatSum = true;
                sum.scaleDown = -1;
                valuesPerBucket += 1;
            } else {
                sum.valueExpr = valueExpr;
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

        IrPendingAgg out;
        out.displayName = displayName;
        out.offset = valuesPerBucket;
        out.valueExpr = genericExprToMetal(agg->arg, idxVar);
        out.funcName = aggFuncName(agg->func);
        out.innerColumn = innerColumnName(agg->arg);
        if (agg->func == AggFunc::MIN || agg->func == AggFunc::MAX) {
            out.atomicOp = agg->func == AggFunc::MIN ? "min" : "max";
            out.isMinMax = true;
            if (agg->arg->type.type == DataType::FLOAT)
                out.isFloatSum = true;
            valuesPerBucket += 1;
        } else if (agg->func == AggFunc::SUM) {
            if (agg->arg->type.type == DataType::FLOAT) {
                const int fixedScale = numericScaleForExpr(agg->arg);
                if (fixedScale > 0) {
                    out.valueExpr = scaledLongExpr(out.valueExpr, fixedScale);
                    out.isLongPair = true;
                    out.scaleDown = fixedScale;
                    valuesPerBucket += 2;
                } else {
                    out.isFloatSum = true;
                    valuesPerBucket += 1;
                }
            } else {
                out.isLongPair = true;
                valuesPerBucket += 2;
            }
        } else {
            return fail(error, "IR grouped aggregate lowerer: unsupported aggregate " +
                               aggFuncName(agg->func) + ".");
        }
        pending.push_back(std::move(out));
    }
    if (pending.empty())
        return fail(error, "IR grouped aggregate lowerer: no aggregate slots.");

    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan->table, idxVar);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate))
            return fail(error, "IR grouped aggregate lowerer: filter predicate unsupported.");
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }
    pipe = maybeSelect(std::move(pipe), "(" + bucketExpr + " >= 0 && " +
                       bucketExpr + " < " + std::to_string(totalBuckets) + ")");

    MetalQueryPlan plan;
    plan.name = "ADHOC_IR_SINGLE_TABLE_GROUP";
    auto keyed = std::make_unique<MetalKeyedAgg>(
        std::move(pipe), "d_ir_group_aggs", bucketExpr,
        totalBuckets, valuesPerBucket, std::to_string(totalBuckets * valuesPerBucket));

    std::vector<std::string> keyNames;
    std::vector<GroupKeyDecode> decodeInfo;
    for (const auto& key : groupKeys) {
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

    auto& groupPhase = appendPhase(plan, "ADHOC_ir_single_table_group", std::move(keyed));

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayNameForGroupedAgg(key, *aggregate, groupKeys);
            if (!name)
                return fail(error, "IR grouped aggregate lowerer: ORDER BY key is not an output.");
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    (void)groupPhase;

    std::vector<KeyedCompactKeySpec> compactKeys;
    std::vector<GenericMatColumnDesc> compactCols;
    for (const auto& key : groupKeys) {
        compactKeys.push_back({key.displayName, key.numValues, key.stride,
                               key.charMap, key.keyBase});
        std::string buf = "d_ir_keyed_out_" + std::to_string(compactCols.size()) +
                          "_" + sanitizeIdentifier(key.displayName);
        compactCols.push_back(GenericMatColumnDesc{
            key.displayName, buf, key.charMap.empty() ? "int" : "char"});
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
        std::string buf = "d_ir_keyed_out_" + std::to_string(compactCols.size()) +
                          "_" + sanitizeIdentifier(out.displayName);
        compactAggs.push_back(out);
        compactCols.push_back(GenericMatColumnDesc{
            out.displayName, buf, metalType, 0, outScale, outLongPair});
    }

    const std::string compactCounter = "d_ir_keyed_result_count";
    auto& compactPhase = appendPhase(plan, "ADHOC_ir_single_table_group_compact",
        makeKeyedAggCompactOperator(
            "d_ir_group_aggs", compactCounter, totalBuckets, valuesPerBucket,
            compactKeys, compactAggs, compactCols));

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string sortRowsSym = "n_gpu_sort_ir_single_keyed_rows";
        attachMaterializedCountHook(compactPhase, compactCounter, sortRowsSym);
        if (!appendGenericGpuSort(plan, "ir_single_keyed", sortRowsSym,
                                  std::to_string(totalBuckets), compactCols,
                                  sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

std::optional<MetalQueryPlan> lowerSingleTableScalarAggregateIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto shape = parseSingleTableScalarAggShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* aggregate = aggregateDetail(shape->aggregate);
    if (!scan || !aggregate)
        return fail(error, "IR scalar aggregate lowerer: malformed scan/aggregate detail.");
    if (!aggregate->groupBy.empty())
        return std::nullopt;
    if (aggregate->aggregates.empty())
        return fail(error, "IR scalar aggregate lowerer: no aggregate outputs.");
    if (aggregate->having)
        return fail(error, "IR scalar aggregate lowerer: HAVING is not supported.");

    const std::string idxVar = "i";
    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan->table, idxVar);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate)) {
            return fail(error, "IR scalar aggregate lowerer: filter predicate is not supported.");
        }
        pipe = maybeSelect(std::move(pipe),
                           genericPredicateToMetal(filter->predicate, idxVar));
    }

    auto reduce = std::make_unique<MetalTGReduce>(std::move(pipe), "d_ir_scalar");
    for (size_t i = 0; i < aggregate->aggregates.size(); ++i) {
        const auto& projection = aggregate->aggregates[i];
        if (!projection.expr)
            return fail(error, "IR scalar aggregate lowerer: null aggregate expression.");
        auto* agg = std::get_if<GenericAggregateExpr>(&projection.expr->node);
        if (!agg)
            return fail(error, "IR scalar aggregate lowerer: projection is not an aggregate.");

        const std::string alias = projection.name.empty()
            ? "agg_" + std::to_string(i)
            : projection.name;
        const std::string accName = "a" + std::to_string(i) + "_" +
                                    sanitizeIdentifier(alias);

        if (agg->func == AggFunc::COUNT) {
            int accIndex = reduce->addAccumulator(accName, "1", "long");
            reduce->setAccumulatorResultAlias(alias, accIndex, 0, nullptr);
            continue;
        }

        if (!agg->arg)
            return fail(error, "IR scalar aggregate lowerer: aggregate '" +
                               aggFuncName(agg->func) + "' requires an argument.");
        if (!materializeExprSupported(agg->arg))
            return fail(error, "IR scalar aggregate lowerer: aggregate argument is not supported.");

        std::string valueExpr = genericExprToMetal(agg->arg, idxVar);
        if (agg->func == AggFunc::AVG) {
            int sumIndex = reduce->addAccumulator(accName + "_sum", valueExpr, "float");
            int countIndex = reduce->addAccumulator(accName + "_count", "1.0f", "float");
            reduce->setAverageResultAlias(alias, sumIndex, countIndex, 0, nullptr);
            continue;
        }

        std::string outputType = agg->arg->type.type == DataType::FLOAT ? "float" : "long";
        MetalTGReduce::ReduceOp op = MetalTGReduce::ReduceOp::SUM;
        if (agg->func == AggFunc::MIN) op = MetalTGReduce::ReduceOp::MIN;
        else if (agg->func == AggFunc::MAX) op = MetalTGReduce::ReduceOp::MAX;
        else if (agg->func != AggFunc::SUM) {
            return fail(error, "IR scalar aggregate lowerer: unsupported aggregate '" +
                               aggFuncName(agg->func) + "'.");
        }
        if (op != MetalTGReduce::ReduceOp::SUM && agg->arg->type.type != DataType::FLOAT)
            outputType = "int";

        int accIndex = reduce->addAccumulator(accName, valueExpr, outputType, "", "", op);
        reduce->setAccumulatorResultAlias(alias, accIndex, 0, nullptr);
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_IR_SINGLE_TABLE_SCALAR";
    appendPhase(plan, "ADHOC_ir_single_table_scalar", std::move(reduce));
    return plan;
}

std::optional<MetalQueryPlan> lowerSingleTableMaterializeIRToMetal(
        const GenericRelPlan& ir,
        std::string* error) {
    auto shape = parseSingleTableMaterializeShape(ir, error);
    if (!shape) return std::nullopt;

    auto* scan = scanDetail(shape->scan);
    auto* project = projectDetail(shape->project);
    if (!scan || !project)
        return fail(error, "IR materialize lowerer: malformed scan/project detail.");
    if (project->projections.empty())
        return fail(error, "IR materialize lowerer: no projection columns.");

    const std::string idxVar = "i";
    std::unique_ptr<MetalOperator> pipe = makeAutoScan(scan->table, idxVar);
    if (auto* filter = filterDetail(shape->filter)) {
        if (!predicateSupported(filter->predicate)) {
            return fail(error, "IR materialize lowerer: filter predicate is not supported.");
        }
        std::string predicate = genericPredicateToMetal(filter->predicate, idxVar);
        pipe = maybeSelect(std::move(pipe), predicate);
    }

    MetalQueryPlan plan;
    plan.name = "ADHOC_IR_SINGLE_TABLE_MATERIALIZE";

    auto materialize = std::make_unique<MetalMaterialize>(
        std::move(pipe), "d_adhoc_result_count", "1");

    const std::string outputSize = tableSizeName(scan->table);
    std::vector<GenericMatColumnDesc> materializedCols;
    for (size_t i = 0; i < project->projections.size(); ++i) {
        const auto& projection = project->projections[i];
        if (!materializeExprSupported(projection.expr)) {
            return fail(error, "IR materialize lowerer: projection '" +
                               projection.name + "' is not supported.");
        }

        int stringLen = fixedStringLenForExpr(projection.expr);
        std::string sizeExpr = outputSize;
        if (stringLen > 0) sizeExpr += " * " + std::to_string(stringLen);
        std::string bufferName = "d_ir_adhoc_" + std::to_string(i) + "_" +
                                 sanitizeIdentifier(projection.name);
        std::string metalType = metalTypeForType(projection.type);
        materialize->addColumn(bufferName, metalType,
                               materializeExprToMetal(projection.expr, idxVar),
                               projection.name, sizeExpr, stringLen);
        materializedCols.push_back(GenericMatColumnDesc{
            projection.name, bufferName, metalType, stringLen});
    }

    auto& matPhase = appendPhase(plan, "ADHOC_ir_single_table_materialize",
                                 std::move(materialize));

    GenericSortSpec sortSpec;
    sortSpec.limit = limitValue(shape->limit);
    if (auto* sort = sortDetail(shape->sort)) {
        for (const auto& key : sort->keys) {
            auto name = sortKeyDisplayName(key, *project);
            if (!name) {
                return fail(error, "IR materialize lowerer: ORDER BY key is not projected.");
            }
            sortSpec.keys.push_back({*name, key.descending});
        }
    }

    if (!sortSpec.keys.empty() || sortSpec.limit >= 0) {
        const std::string rowsSym = "n_gpu_sort_ir_single_rows";
        attachMaterializedCountHook(matPhase, "d_adhoc_result_count", rowsSym);
        if (!appendGenericGpuSort(plan, "ir_single_materialize", rowsSym,
                                  outputSize, materializedCols, sortSpec, error)) {
            return std::nullopt;
        }
    }

    return plan;
}

} // namespace codegen
