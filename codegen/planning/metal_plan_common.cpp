#include "metal_plan_common.h"
#include "tpch_schema.h"

#include <sstream>
#include <type_traits>
#include <variant>

namespace codegen {

// ===================================================================
// Helper: expression to Metal code string (columnar access: col[idx])
// ===================================================================

// Collect all column references in an expression
void collectColumns(const ExprPtr& expr, std::set<std::string>& cols) {
    if (!expr) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            cols.insert(node.column);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            collectColumns(node.left, cols);
            collectColumns(node.right, cols);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (auto& a : node.args) collectColumns(a, cols);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (auto& b : node.branches) {
                collectColumns(b.result, cols);
            }
            if (node.elseResult) collectColumns(node.elseResult, cols);
        }
    }, expr->node);
}

void collectColumns(const PredPtr& pred, std::set<std::string>& cols) {
    if (!pred) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            collectColumns(node.left, cols);
            collectColumns(node.right, cols);
        } else if constexpr (std::is_same_v<T, Between>) {
            collectColumns(node.expr, cols);
            collectColumns(node.low, cols);
            collectColumns(node.high, cols);
        } else if constexpr (std::is_same_v<T, InList>) {
            collectColumns(node.expr, cols);
            for (auto& v : node.values) collectColumns(v, cols);
        } else if constexpr (std::is_same_v<T, LogicalAnd>) {
            for (auto& c : node.children) collectColumns(c, cols);
        } else if constexpr (std::is_same_v<T, LogicalOr>) {
            for (auto& c : node.children) collectColumns(c, cols);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            collectColumns(node.child, cols);
        } else if constexpr (std::is_same_v<T, Like>) {
            collectColumns(node.expr, cols);
        }
    }, pred->node);
}

std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar) {
    if (!expr) return "";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, ColRef>) {
            // Columnar access: column_name[idx]
            return node.column + "[" + idxVar + "]";
        }
        else if constexpr (std::is_same_v<T, Literal>) {
            return std::visit([](auto&& v) -> std::string {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, int>)
                    return std::to_string(v);
                else if constexpr (std::is_same_v<V, float>)
                    return std::to_string(v) + "f";
                else
                    return "\"" + v + "\"";
            }, node.value);
        }
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
            std::string l = exprToMetal(node.left, idxVar);
            std::string r = exprToMetal(node.right, idxVar);
            switch (node.op) {
                case ExprOp::ADD: return "(" + l + " + " + r + ")";
                case ExprOp::SUB: return "(" + l + " - " + r + ")";
                case ExprOp::MUL: return "(" + l + " * " + r + ")";
                case ExprOp::DIV: return "(" + l + " / " + r + ")";
            }
            return l;
        }
        else if constexpr (std::is_same_v<T, FuncCall>) {
            std::ostringstream os;
            os << node.name << "(";
            for (size_t i = 0; i < node.args.size(); i++) {
                if (i) os << ", ";
                os << exprToMetal(node.args[i], idxVar);
            }
            os << ")";
            return os.str();
        }
        else if constexpr (std::is_same_v<T, CaseWhen>) {
            if (!node.branches.empty())
                return exprToMetal(node.branches[0].result, idxVar);
            if (node.elseResult)
                return exprToMetal(node.elseResult, idxVar);
            return "0";
        }
        else {
            return "/* unknown expr */";
        }
    }, expr->node);
}

std::string predToMetal(const PredPtr& pred, const std::string& idxVar) {
    if (!pred) return "true";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, Comparison>) {
            std::string l = exprToMetal(node.left, idxVar);
            std::string r = exprToMetal(node.right, idxVar);
            switch (node.op) {
                case CmpOp::EQ: return l + " == " + r;
                case CmpOp::NE: return l + " != " + r;
                case CmpOp::LT: return l + " < " + r;
                case CmpOp::LE: return l + " <= " + r;
                case CmpOp::GT: return l + " > " + r;
                case CmpOp::GE: return l + " >= " + r;
            }
            return l + " == " + r;
        }
        else if constexpr (std::is_same_v<T, Between>) {
            std::string e = exprToMetal(node.expr, idxVar);
            std::string lo = exprToMetal(node.low, idxVar);
            std::string hi = exprToMetal(node.high, idxVar);
            return "(" + e + " >= " + lo + " && " + e + " <= " + hi + ")";
        }
        else if constexpr (std::is_same_v<T, InList>) {
            std::string e = exprToMetal(node.expr, idxVar);
            std::string cond;
            for (size_t i = 0; i < node.values.size(); i++) {
                if (i) cond += " || ";
                cond += e + " == " + exprToMetal(node.values[i], idxVar);
            }
            return "(" + cond + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalAnd>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " && ";
                cond += "(" + predToMetal(node.children[i], idxVar) + ")";
            }
            return cond;
        }
        else if constexpr (std::is_same_v<T, LogicalOr>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " || ";
                cond += "(" + predToMetal(node.children[i], idxVar) + ")";
            }
            return "(" + cond + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalNot>) {
            return "!(" + predToMetal(node.child, idxVar) + ")";
        }
        else if constexpr (std::is_same_v<T, Like>) {
            return "/* LIKE not directly translatable */true";
        }
        else if constexpr (std::is_same_v<T, ExistsPred>) {
            return "/* EXISTS */true";
        }
        else {
            return "true";
        }
    }, pred->node);
}

// Combine all filter predicates into a single Metal condition string
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar) {
    if (filters.empty()) return "";
    if (filters.size() == 1) return predToMetal(filters[0], idxVar);

    std::string cond;
    for (size_t i = 0; i < filters.size(); i++) {
        if (i) cond += " && ";
        cond += "(" + predToMetal(filters[i], idxVar) + ")";
    }
    return cond;
}

// Map column name to Metal type using TPC-H schema
static std::string colMetalType(const std::string& table, const std::string& colName) {
    const auto& schema = TPCHSchema::instance();
    auto& tdef = schema.table(table);
    auto& cdef = tdef.col(colName);
    switch (cdef.type) {
        case DataType::INT:        return "int";
        case DataType::FLOAT:      return "float";
        case DataType::DATE:       return "int";
        case DataType::CHAR1:      return "char";
        case DataType::CHAR_FIXED: return "char";
    }
    return "int";
}

std::unique_ptr<MetalGridStrideScan> makeScan(const std::string& table,
                                              const std::string& idxVar,
                                              ColumnList columns) {
    auto scan = std::make_unique<MetalGridStrideScan>(table, "row", idxVar);
    for (const auto& [name, type] : columns) scan->addColumn(name, type);
    return scan;
}

std::unique_ptr<MetalGridStrideScan> makeScanForCols(const std::string& table,
                                                     const std::string& idxVar,
                                                     const std::set<std::string>& cols) {
    auto scan = std::make_unique<MetalGridStrideScan>(table, "row", idxVar);
    for (const auto& colName : cols) scan->addColumn(colName, colMetalType(table, colName));
    return scan;
}

std::unique_ptr<MetalOperator> maybeSelect(std::unique_ptr<MetalOperator> input,
                                           const std::string& filterCond) {
    if (filterCond.empty()) return input;
    return std::make_unique<MetalSelection>(std::move(input), filterCond);
}

MetalQueryPlan::Phase& appendPhase(MetalQueryPlan& plan, const std::string& name,
                                   std::unique_ptr<MetalOperator> root,
                                   int threadgroupSize) {
    MetalQueryPlan::Phase phase;
    phase.name = name;
    phase.root = std::move(root);
    phase.threadgroupSize = threadgroupSize;
    plan.phases.push_back(std::move(phase));
    return plan.phases.back();
}

} // namespace codegen
