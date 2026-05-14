#include "metal_plan_common.h"

#include <sstream>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace codegen {

namespace {

bool isDateColumnExpr(const ExprPtr& expr) {
    if (!expr) return false;
    if (auto* col = std::get_if<ColRef>(&expr->node)) {
        return col->dataType == DataType::DATE;
    }
    return false;
}

std::optional<int> parseDateStringExpr(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<Literal>(&expr->node);
    if (!lit) return std::nullopt;
    auto* s = std::get_if<std::string>(&lit->value);
    if (!s || s->size() < 10 || (*s)[4] != '-' || (*s)[7] != '-') {
        return std::nullopt;
    }
    try {
        int y = std::stoi(s->substr(0, 4));
        int m = std::stoi(s->substr(5, 2));
        int d = std::stoi(s->substr(8, 2));
        return y * 10000 + m * 100 + d;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> parseChar1StringExpr(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<Literal>(&expr->node);
    if (!lit) return std::nullopt;
    auto* s = std::get_if<std::string>(&lit->value);
    if (!s || s->empty()) return std::nullopt;
    // CHAR1 comparison: use first character of the string literal
    return std::string("'") + s->front() + "'";
}

const ColRef* fixedStringCol(const ExprPtr& expr) {
    if (!expr) return nullptr;
    auto* col = std::get_if<ColRef>(&expr->node);
    if (!col || col->dataType != DataType::CHAR_FIXED) return nullptr;
    if (getenv("GEN_DEBUG"))
        fprintf(stderr, "[GEN_DEBUG] fixedStringCol: %s.%s dt=%d\n", col->table.c_str(), col->column.c_str(), (int)col->dataType);
    return col;
}

std::optional<std::string> stringLiteralValue(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    auto* lit = std::get_if<Literal>(&expr->node);
    if (!lit) return std::nullopt;
    auto* value = std::get_if<std::string>(&lit->value);
    if (!value) return std::nullopt;
    return *value;
}

std::string metalCharLiteral(char ch) {
    if (ch == '\\') return "'\\\\'";
    if (ch == '\'') return "'\\\''";
    if (ch == '\0') return "'\\0'";
    return std::string("'") + ch + "'";
}

std::string fixedStringEqMetal(const ColRef& col,
                                const std::string& literal,
                                const std::string& idxVar,
                                const SchemaProvider* schema = nullptr) {
    int width = 0;
    if (schema) width = schema->columnFixedWidth(col.table, col.column);
    if (width <= 0) width = col.fixedWidth; // fallback: ColRef carries width from subquery aliases
    if (width <= 0) return "false";

    int cmpLen = std::min(static_cast<int>(literal.size()), width);

    std::string aliasPrefix;
    if (!col.tableAlias.empty()) aliasPrefix = "/*" + col.tableAlias + "*/";
    std::string base = aliasPrefix + col.column + "[" + idxVar + " * " + std::to_string(width) + " + ";
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += base + std::to_string(i) + "] == " + metalCharLiteral(literal[i]);
    }
    for (int i = cmpLen; i < width; ++i) {
        if (!cond.empty()) cond += " && ";
        std::string slot = base + std::to_string(i) + "]";
        cond += "(" + slot + " == '\\0' || " + slot + " == ' ')";
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

std::string fixedStringPrefixMetal(const ColRef& col,
                                    const std::string& prefix,
                                    const std::string& idxVar,
                                    const SchemaProvider* schema = nullptr) {
    int width = 0;
    if (schema) width = schema->columnFixedWidth(col.table, col.column);
    if (width <= 0) return "false";

    int cmpLen = std::min(static_cast<int>(prefix.size()), width);

    std::string aliasPrefix;
    if (!col.tableAlias.empty()) aliasPrefix = "/*" + col.tableAlias + "*/";
    std::string base = aliasPrefix + col.column + "[" + idxVar + " * " + std::to_string(width) + " + ";
    std::string cond;
    for (int i = 0; i < cmpLen; ++i) {
        if (!cond.empty()) cond += " && ";
        cond += base + std::to_string(i) + "] == " + metalCharLiteral(prefix[i]);
    }
    return cond.empty() ? "true" : "(" + cond + ")";
}

std::vector<std::string> likeLiteralSegments(const std::string& pattern) {
    std::vector<std::string> segments;
    std::string current;
    for (char ch : pattern) {
        if (ch == '%') {
            if (!current.empty()) segments.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) segments.push_back(current);
    return segments;
}

bool likePatternUsesUnsupportedWildcard(const std::string& pattern) {
    return pattern.find('_') != std::string::npos ||
           pattern.find('\\') != std::string::npos;
}

std::string packedSegmentWord(const std::string& segment, size_t begin) {
    uint64_t word = 0;
    size_t end = std::min(segment.size(), begin + 8);
    for (size_t i = begin; i < end; ++i) {
        word |= static_cast<uint64_t>(static_cast<unsigned char>(segment[i])) << ((i - begin) * 8);
    }
    std::ostringstream os;
    os << "0x" << std::hex << word << "ull";
    return os.str();
}

bool likeSegmentsArePackable(const std::vector<std::string>& segments) {
    for (const auto& segment : segments) {
        if (segment.size() > 16) return false;
    }
    return true;
}

std::optional<std::string> fixedStringLikeMetal(const Like& like,
                                                const std::string& idxVar,
                                                const SchemaProvider* schema = nullptr) {
    const ColRef* col = fixedStringCol(like.expr);
    if (!col || likePatternUsesUnsupportedWildcard(like.pattern)) return std::nullopt;

    if (col->table.empty() || col->column.empty()) return std::nullopt;
    int width = 0;
    if (schema) width = schema->columnFixedWidth(col->table, col->column);
    if (width <= 0) return std::nullopt;

    // Build LIKE Metal expression using column width

    if (like.pattern.find('%') == std::string::npos) {
        std::string exact = fixedStringEqMetal(*col, like.pattern, idxVar, schema);
        return like.negated ? "!(" + exact + ")" : exact;
    }

    std::vector<std::string> segments = likeLiteralSegments(like.pattern);
    if (segments.empty()) return like.negated ? std::string("false") : std::string("true");
    if (segments.size() > 2) return std::nullopt;

    const bool leadingWildcard = like.pattern.front() == '%';
    const bool trailingWildcard = like.pattern.back() == '%';

    if (segments.size() == 1 && !leadingWildcard && trailingWildcard && segments[0].size() > 16) {
        std::string match = fixedStringPrefixMetal(*col, segments[0], idxVar, schema);
        if (like.negated) return "!(" + match + ")";
        return match;
    }

    if (!likeSegmentsArePackable(segments)) return std::nullopt;

    std::ostringstream call;
    if (segments.size() == 1) {
        const auto& segment = segments[0];
        call << "fixed_like_one_segment(" << col->column << ", (uint)(" << idxVar << "), "
             << width << "u, " << packedSegmentWord(segment, 0) << ", "
             << packedSegmentWord(segment, 8) << ", " << segment.size() << "u, "
             << (leadingWildcard ? "true" : "false") << ", "
             << (trailingWildcard ? "true" : "false") << ")";
    } else {
        const auto& first = segments[0];
        const auto& second = segments[1];
        call << "fixed_like_two_segment(" << col->column << ", (uint)(" << idxVar << "), "
             << width << "u, " << packedSegmentWord(first, 0) << ", "
             << packedSegmentWord(first, 8) << ", " << first.size() << "u, "
             << packedSegmentWord(second, 0) << ", " << packedSegmentWord(second, 8) << ", "
             << second.size() << "u, " << (leadingWildcard ? "true" : "false") << ", "
             << (trailingWildcard ? "true" : "false") << ")";
    }

    std::string match = call.str();
    if (like.negated) return "!(" + match + ")";
    return match;
}

std::optional<std::string> fixedStringComparisonMetal(const Comparison& cmp,
                                                      const std::string& idxVar,
                                                      const SchemaProvider* schema = nullptr) {
    const ColRef* col = fixedStringCol(cmp.left);
    auto literal = stringLiteralValue(cmp.right);
    bool reversed = false;
    if (!col || !literal) {
        col = fixedStringCol(cmp.right);
        literal = stringLiteralValue(cmp.left);
        reversed = true;
    }
    if (!col || !literal) return std::nullopt;

    CmpOp op = cmp.op;
    if (reversed) {
        if (op == CmpOp::LT) op = CmpOp::GT;
        else if (op == CmpOp::LE) op = CmpOp::GE;
        else if (op == CmpOp::GT) op = CmpOp::LT;
        else if (op == CmpOp::GE) op = CmpOp::LE;
    }
    if (op != CmpOp::EQ && op != CmpOp::NE) return std::nullopt;

    std::string eq = fixedStringEqMetal(*col, *literal, idxVar, schema);
    if (getenv("GEN_DEBUG")) {
        int w = 0; bool has = false;
        if (schema) { w = schema->columnFixedWidth(col->table, col->column); has = schema->hasColumn(col->table, col->column); }
        fprintf(stderr, "[GEN_DEBUG] fixedStringComparison: col=%s.%s lit=%s hasCol=%d width=%d tbls=%zu eq=[%s]\n",
                col->table.c_str(), col->column.c_str(), literal->c_str(),
                (int)has, w, schema ? schema->tableNames().size() : 0, eq.c_str());
    }
    if (op == CmpOp::NE) return "!(" + eq + ")";
    return eq;
}

std::string exprToMetalForPredicate(const ExprPtr& expr,
                                    const ExprPtr& other,
                                    const std::string& idxVar) {
    if (isDateColumnExpr(other)) {
        if (auto dateValue = parseDateStringExpr(expr)) {
            return std::to_string(*dateValue);
        }
    }
    if (other) {
        if (auto* col = std::get_if<ColRef>(&other->node)) {
            if (col->dataType == DataType::CHAR1) {
                if (auto charValue = parseChar1StringExpr(expr)) return *charValue;
            }
        }
        // Handle string literal compared against non-ColRef (e.g. substring
        // result IN ('13', '31', ...)): convert string to integer literal.
        if (auto* lit = expr ? std::get_if<Literal>(&expr->node) : nullptr) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) {
                try { return std::to_string(std::stoi(*sv)); }
                catch (...) {}
            }
        }
    }
    return exprToMetal(expr, idxVar);
}

} // namespace

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
                collectColumns(b.condition, cols);
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

void collectColumnTables(const ExprPtr& expr, std::map<std::string, std::string>& colToTable) {
    if (!expr) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            if (!node.table.empty()) colToTable[node.column] = node.table;
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            collectColumnTables(node.left, colToTable);
            collectColumnTables(node.right, colToTable);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (auto& a : node.args) collectColumnTables(a, colToTable);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (auto& b : node.branches) {
                collectColumnTables(b.condition, colToTable);
                collectColumnTables(b.result, colToTable);
            }
            if (node.elseResult) collectColumnTables(node.elseResult, colToTable);
        }
    }, expr->node);
}

void collectColumnTables(const PredPtr& pred, std::map<std::string, std::string>& colToTable) {
    if (!pred) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            collectColumnTables(node.left, colToTable);
            collectColumnTables(node.right, colToTable);
        } else if constexpr (std::is_same_v<T, Between>) {
            collectColumnTables(node.expr, colToTable);
            collectColumnTables(node.low, colToTable);
            collectColumnTables(node.high, colToTable);
        } else if constexpr (std::is_same_v<T, InList>) {
            collectColumnTables(node.expr, colToTable);
            for (auto& v : node.values) collectColumnTables(v, colToTable);
        } else if constexpr (std::is_same_v<T, LogicalAnd>) {
            for (auto& c : node.children) collectColumnTables(c, colToTable);
        } else if constexpr (std::is_same_v<T, LogicalOr>) {
            for (auto& c : node.children) collectColumnTables(c, colToTable);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            collectColumnTables(node.child, colToTable);
        } else if constexpr (std::is_same_v<T, Like>) {
            collectColumnTables(node.expr, colToTable);
        }
    }, pred->node);
}

std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar,
                        const SchemaProvider* schema) {
    if (!expr) return "";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, ColRef>) {
            // Columnar access: column_name[idx]
            std::string aliasPrefix;
            if (!node.tableAlias.empty()) aliasPrefix = "/*" + node.tableAlias + "*/";
            return aliasPrefix + node.column + "[" + idxVar + "]";
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
            if (node.name == "date_part" || node.name == "extract") {
                // EXTRACT(YEAR/MONTH/DAY FROM date_col)
                // args[0] = unit string, args[1] = column expression
                std::string unit;
                if (!node.args.empty() && node.args[0]) {
                    if (auto* lit = std::get_if<Literal>(&node.args[0]->node)) {
                        if (auto* s = std::get_if<std::string>(&lit->value)) unit = *s;
                    }
                }
                std::string col = node.args.size() > 1 ? exprToMetal(node.args[1], idxVar) : "";
                if (unit == "year")   return "(" + col + " / 10000)";
                if (unit == "month") return "((" + col + " / 100) % 100)";
                if (unit == "day")    return "(" + col + " % 100)";
                return col; // fallback
            }
            if (node.name == "substring") {
                int start = 1, len = 1;
                if (node.args.size() > 1 && node.args[1]) {
                    if (auto* lit = std::get_if<Literal>(&node.args[1]->node)) {
                        if (auto* sv = std::get_if<int>(&lit->value)) start = *sv;
                    }
                }
                if (node.args.size() > 2 && node.args[2]) {
                    if (auto* lit = std::get_if<Literal>(&node.args[2]->node)) {
                        if (auto* sv = std::get_if<int>(&lit->value)) len = *sv;
                    }
                }
                // Look up the column's fixed width and name for proper row offset.
                std::string colName;
                int fw = 1;
                if (node.args.size() > 0 && node.args[0]) {
                    if (auto* cr = std::get_if<ColRef>(&node.args[0]->node)) {
                        colName = cr->column;
                    try {
                        if (schema) fw = schema->columnFixedWidth(cr->table, cr->column);
                    } catch (...) {}
                    }
                }
                if (colName.empty()) return "0";
                // Emit: ((col[idx*fw + offset] - '0') * weight + ...)
                std::string result = "(";
                for (int i = 0; i < len; ++i) {
                    if (i > 0) result += " + ";
                    int pos = start - 1 + i;
                    int weight = 1;
                    for (int w = 0; w < len - 1 - i; ++w) weight *= 10;
                    std::string access = colName + "[" + idxVar + " * " + std::to_string(fw) +
                                        " + " + std::to_string(pos) + "]";
                    if (weight > 1)
                        result += "(" + access + " - '0') * " + std::to_string(weight);
                    else
                        result += "(" + access + " - '0')";
                }
                result += ")";
                return result;
            }
            // Strip aggregate function calls (sum, count, avg, min, max)
            // — these are emitted as raw inner expressions for materialize/agg.
            if (node.name == "sum" || node.name == "count" || node.name == "avg" ||
                node.name == "min" || node.name == "max") {
                if (!node.args.empty()) return exprToMetal(node.args[0], idxVar);
                return "0";
            }
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
            if (node.branches.empty()) {
                if (node.elseResult) return exprToMetal(node.elseResult, idxVar);
                return "0";
            }
            std::string result = "(";
            for (size_t i = 0; i < node.branches.size(); ++i) {
                if (i > 0) result += " : ";
                std::string cond = predToMetal(node.branches[i].condition, idxVar, schema);
                if (getenv("GEN_DEBUG")) {
                    auto* cmp = std::get_if<Comparison>(&node.branches[i].condition->node);
                    if (cmp) {
                        auto* lc = std::get_if<ColRef>(&cmp->left->node);
                        if (lc) fprintf(stderr, "[GEN_DEBUG] CaseWhen: col=%s.%s dt=%d cond=[%s]\n",
                                        lc->table.c_str(), lc->column.c_str(), (int)lc->dataType, cond.c_str());
                    }
                }
                std::string val = exprToMetal(node.branches[i].result, idxVar, schema);
                result += cond + " ? " + val;
            }
            if (node.elseResult)
                result += " : " + exprToMetal(node.elseResult, idxVar);
            else
                result += " : 0";
            result += ")";
            return result;
        }
        else {
            return "/* unknown expr */";
        }
    }, expr->node);
}

std::string predToMetal(const PredPtr& pred, const std::string& idxVar,
                        const SchemaProvider* schema) {
    if (!pred) return "true";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, Comparison>) {
            if (auto fixedStringCmp = fixedStringComparisonMetal(node, idxVar, schema)) {
                return *fixedStringCmp;
            }
            std::string l = exprToMetalForPredicate(node.left, node.right, idxVar);
            std::string r = exprToMetalForPredicate(node.right, node.left, idxVar);
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
            std::string lo = exprToMetalForPredicate(node.low, node.expr, idxVar);
            std::string hi = exprToMetalForPredicate(node.high, node.expr, idxVar);
            return "(" + e + " >= " + lo + " && " + e + " <= " + hi + ")";
        }
        else if constexpr (std::is_same_v<T, InList>) {
            if (node.values.empty()) return "true";  // placeholder for unsupported subquery
            if (auto* col = fixedStringCol(node.expr)) {
                std::string cond;
                for (const auto& value : node.values) {
                    auto literal = stringLiteralValue(value);
                    if (!literal) return "false";
                    if (!cond.empty()) cond += " || ";
                    cond += fixedStringEqMetal(*col, *literal, idxVar, schema);
                }
                return cond.empty() ? "false" : "(" + cond + ")";
            }
            // CHAR1 column InList — compare first character of each literal value
            if (auto* col = std::get_if<ColRef>(&node.expr->node)) {
                if (col->dataType == DataType::CHAR1) {
                    std::string cond;
                    for (const auto& value : node.values) {
                        auto literal = stringLiteralValue(value);
                        if (!literal || literal->empty()) continue;
                        if (!cond.empty()) cond += " || ";
                        cond += col->column + "[" + idxVar + "] == " + metalCharLiteral((*literal)[0]);
                    }
                    return cond.empty() ? "false" : "(" + cond + ")";
                }
            }
            std::string e = exprToMetal(node.expr, idxVar);
            std::string cond;
            for (size_t i = 0; i < node.values.size(); i++) {
                if (i) cond += " || ";
                cond += e + " == " + exprToMetalForPredicate(node.values[i], node.expr, idxVar);
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
            if (auto fixedStringLike = fixedStringLikeMetal(node, idxVar, schema)) {
                return *fixedStringLike;
            }
            return "false";
        }
        else if constexpr (std::is_same_v<T, ExistsPred>) {
            return "/* EXISTS */true";
        }
        else {
            return "true";
        }
    }, pred->node);
}

// Combine all filter predicates into a single Metal condition string.
// For readability and Metal compiler efficiency, use line breaks with multiple filters.
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar,
                           const SchemaProvider* schema) {
    if (filters.empty()) return "";
    if (filters.size() == 1) return predToMetal(filters[0], idxVar, schema);

    std::string cond;
    const char* sep = filters.size() > 2 ? "\n               && " : " && ";
    for (size_t i = 0; i < filters.size(); i++) {
        if (i) cond += sep;
        cond += "(" + predToMetal(filters[i], idxVar, schema) + ")";
    }
    return cond;
}

// Map column name to Metal type.
static std::string colMetalType(const std::string& table, const std::string& colName,
                                const SchemaProvider* schema = nullptr) {
    DataType type = DataType::INT;
    if (schema) type = schema->columnType(table, colName);
    switch (type) {
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
                                                     const std::set<std::string>& cols,
                                                     const SchemaProvider* schema) {
    auto scan = std::make_unique<MetalGridStrideScan>(table, "row", idxVar);
    for (const auto& colName : cols) scan->addColumn(colName, colMetalType(table, colName, schema));
    return scan;
}

std::unique_ptr<MetalGridStrideScan> makeAutoScan(const std::string& table,
                                                   const std::string& idxVar) {
    return std::make_unique<MetalGridStrideScan>(table, "row", idxVar);
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

void addGpuSortToPlan(MetalQueryPlan& plan,
                      const std::string& sortColBuffer,
                      const std::string& sortColType,
                      const std::string& nResultsExpr,
                      bool sortDesc,
                      const std::string& sortKeyBuf,
                      const std::string& sortIdxBuf) {
    // Phase: init sort keys from the materialized column buffer
    auto initSort = std::make_unique<MetalInitSortKeys>(
        sortColBuffer, sortColType, sortKeyBuf, sortIdxBuf, nResultsExpr, sortDesc);
    appendPhase(plan, "GPU_sort_init", std::move(initSort));

    // Phase: bitonic sort step with post-dispatch (k,j) hook
    auto sortStep = std::make_unique<MetalBitonicSortStep>(sortKeyBuf, sortIdxBuf, nResultsExpr);
    auto& sortPhase = appendPhase(plan, "GPU_sort_step", std::move(sortStep));
    sortPhase.postDispatchHook = MetalInitSortKeys::makeBitonicHook(
        "GPU_sort_step", sortKeyBuf, sortIdxBuf, nResultsExpr);

    // Mark plan for sorted-index remapping during result collection
    plan.gpuSort = MetalQueryPlan::GpuSort{sortIdxBuf, nResultsExpr, sortDesc};
}

std::string exprToMetalForHaving(const ExprPtr& expr,
                                 const std::vector<MetalKeyedAggSlotForHaving>& slots) {
    if (!expr) return "0";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, Literal>) {
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
        else if constexpr (std::is_same_v<T, FuncCall>) {
            std::string funcUpper = node.name;
            for (auto& c : funcUpper) c = (char)std::toupper((unsigned char)c);

            std::string refCol;
            if (!node.args.empty()) {
                if (auto* cr = std::get_if<ColRef>(&node.args[0]->node)) {
                    refCol = cr->column;
                }
            }

            for (const auto& slot : slots) {
                if (slot.funcName == funcUpper) {
                    if (refCol.empty() || slot.innerColumn == refCol) {
                        return "((float)(_tg_" + slot.name + "))";
                    }
                }
            }
            // Fallback: match by funcName only (skip innerColumn check)
            for (const auto& slot : slots) {
                if (slot.funcName == funcUpper) {
                    return "((float)(_tg_" + slot.name + "))";
                }
            }
            return "0";
        }
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
            std::string l = exprToMetalForHaving(node.left, slots);
            std::string r = exprToMetalForHaving(node.right, slots);
            switch (node.op) {
                case ExprOp::ADD: return "(" + l + " + " + r + ")";
                case ExprOp::SUB: return "(" + l + " - " + r + ")";
                case ExprOp::MUL: return "(" + l + " * " + r + ")";
                case ExprOp::DIV: return "(" + l + " / " + r + ")";
            }
            return l;
        }
        else {
            return "0";
        }
    }, expr->node);
}

std::string predToMetalForHaving(const PredPtr& pred,
                                 const std::vector<MetalKeyedAggSlotForHaving>& slots) {
    if (!pred) return "true";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, Comparison>) {
            std::string l = exprToMetalForHaving(node.left, slots);
            std::string r = exprToMetalForHaving(node.right, slots);
            switch (node.op) {
                case CmpOp::EQ: return "(" + l + " == " + r + ")";
                case CmpOp::NE: return "(" + l + " != " + r + ")";
                case CmpOp::LT: return "(" + l + " < " + r + ")";
                case CmpOp::LE: return "(" + l + " <= " + r + ")";
                case CmpOp::GT: return "(" + l + " > " + r + ")";
                case CmpOp::GE: return "(" + l + " >= " + r + ")";
            }
            return "(" + l + " == " + r + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalAnd>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " && ";
                cond += "(" + predToMetalForHaving(node.children[i], slots) + ")";
            }
            return cond;
        }
        else if constexpr (std::is_same_v<T, LogicalOr>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " || ";
                cond += "(" + predToMetalForHaving(node.children[i], slots) + ")";
            }
            return "(" + cond + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalNot>) {
            return "!(" + predToMetalForHaving(node.child, slots) + ")";
        }
        else {
            return "true";
        }
    }, pred->node);
}

} // namespace codegen
