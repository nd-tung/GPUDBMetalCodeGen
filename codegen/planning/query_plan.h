#pragma once
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <cstdint>

namespace codegen {

// ===================================================================
// DATA TYPES
// ===================================================================

enum class DataType { INT, FLOAT, DATE, CHAR1, CHAR_FIXED };

struct TypeInfo {
    DataType type;
    int fixedWidth = 0; // only for CHAR_FIXED
};

// ===================================================================
// EXPRESSIONS
// ===================================================================

enum class ExprOp { ADD, SUB, MUL, DIV };
enum class AggFunc { SUM, COUNT, AVG, MIN, MAX, COUNT_DISTINCT };
enum class CmpOp { EQ, NE, LT, LE, GT, GE };

struct ColRef {
    std::string table;    // e.g. "lineitem"
    std::string column;   // e.g. "l_shipdate"
    int         colIndex; // resolved TPC-H column index
    DataType    dataType;
};

struct Literal {
    std::variant<int, float, std::string> value;
};

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct BinaryExpr {
    ExprOp op;
    ExprPtr left, right;
};

struct CaseWhen {
    struct Branch { ExprPtr condition, result; };
    std::vector<Branch> branches;
    ExprPtr elseResult;
};

struct FuncCall {
    std::string name; // EXTRACT, SUBSTRING, etc.
    std::vector<ExprPtr> args;
};

struct Expr {
    std::variant<ColRef, Literal, BinaryExpr, CaseWhen, FuncCall> node;

    static ExprPtr col(const std::string& table, const std::string& col, int idx, DataType dt) {
        auto e = std::make_shared<Expr>();
        e->node = ColRef{table, col, idx, dt};
        return e;
    }
    static ExprPtr lit(int v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr litf(float v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr lits(const std::string& v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr binary(ExprOp op, ExprPtr l, ExprPtr r) {
        auto e = std::make_shared<Expr>();
        e->node = BinaryExpr{op, l, r};
        return e;
    }
};

// ===================================================================
// PREDICATES
// ===================================================================

struct Predicate;
using PredPtr = std::shared_ptr<Predicate>;

struct Comparison {
    CmpOp op;
    ExprPtr left, right;
};

struct Between {
    ExprPtr expr, low, high;
};

struct InList {
    ExprPtr expr;
    std::vector<ExprPtr> values;
};

struct Like {
    ExprPtr expr;
    std::string pattern;
    bool negated = false;
};

struct LogicalAnd { std::vector<PredPtr> children; };
struct LogicalOr  { std::vector<PredPtr> children; };
struct LogicalNot { PredPtr child; };

struct ExistsPred {
    bool negated = false;
    // child query index in AnalyzedQuery::subqueries
    int subqueryIdx = -1;
};

struct Predicate {
    std::variant<Comparison, Between, InList, Like,
                 LogicalAnd, LogicalOr, LogicalNot, ExistsPred> node;

    static PredPtr cmp(CmpOp op, ExprPtr l, ExprPtr r) {
        auto p = std::make_shared<Predicate>();
        p->node = Comparison{op, l, r};
        return p;
    }
    static PredPtr between(ExprPtr e, ExprPtr lo, ExprPtr hi) {
        auto p = std::make_shared<Predicate>();
        p->node = Between{e, lo, hi};
        return p;
    }
    static PredPtr inList(ExprPtr e, std::vector<ExprPtr> vals) {
        auto p = std::make_shared<Predicate>();
        p->node = InList{e, std::move(vals)};
        return p;
    }
    static PredPtr like(ExprPtr e, const std::string& pat, bool neg = false) {
        auto p = std::make_shared<Predicate>();
        p->node = Like{e, pat, neg};
        return p;
    }
    static PredPtr logAnd(std::vector<PredPtr> ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalAnd{std::move(ch)};
        return p;
    }
    static PredPtr logOr(std::vector<PredPtr> ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalOr{std::move(ch)};
        return p;
    }
    static PredPtr logNot(PredPtr ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalNot{ch};
        return p;
    }
};

// ===================================================================
// AGGREGATION SPEC
// ===================================================================

struct AggSpec {
    AggFunc   func;
    ExprPtr   expr;       // null for COUNT(*)
    std::string alias;
};

struct ColumnBinding {
    std::string table;
    std::string column;
    int         colIndex;
    DataType    dataType;
    int         fixedWidth = 0;
};

} // namespace codegen
