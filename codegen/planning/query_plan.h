#pragma once
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <cstdint>
#include <map>
#include <optional>

namespace codegen {

// --- Data Types ---

enum class DataType { INT, FLOAT, DATE, CHAR1, CHAR_FIXED };

struct TypeInfo {
    DataType type;
    int fixedWidth = 0; // CHAR_FIXED width.
};

// --- Expressions ---

enum class ExprOp { ADD, SUB, MUL, DIV };
enum class AggFunc { SUM, COUNT, AVG, MIN, MAX, COUNT_DISTINCT };
enum class CmpOp { EQ, NE, LT, LE, GT, GE };

struct ColRef {
    std::string table;      // Base table for schema lookup.
    std::string column;     // Source column name.
    int         colIndex;   // Resolved schema column index.
    DataType    dataType;
    int         fixedWidth = 0; // CHAR_FIXED width.
    std::string tableAlias; // Alias used for join disambiguation.
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
    struct Branch { std::shared_ptr<struct Predicate> condition; ExprPtr result; };
    std::vector<Branch> branches;
    ExprPtr elseResult;
};

struct FuncCall {
    std::string name;
    std::vector<ExprPtr> args;
};

struct ScalarSubqueryRef {
    int index = -1;
};

struct Expr {
    std::variant<ColRef, Literal, BinaryExpr, CaseWhen, FuncCall,
                 ScalarSubqueryRef> node;

    static ExprPtr col(const std::string& table, const std::string& col, int idx, DataType dt, int fw = 0, const std::string& alias = "") {
        auto e = std::make_shared<Expr>();
        e->node = ColRef{table, col, idx, dt, fw, alias};
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
    static ExprPtr scalarSubquery(int index) {
        auto e = std::make_shared<Expr>();
        e->node = ScalarSubqueryRef{index};
        return e;
    }
};

// --- Predicates ---

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
    // Child query index in the source subquery list.
    int subqueryIdx = -1;

    ExistsPred() = default;
    ExistsPred(bool negated_, int subqueryIdx_)
        : negated(negated_), subqueryIdx(subqueryIdx_) {}
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

// --- Query fragments shared by SQL analysis and Generic IR source metadata ---

struct JoinClause {
    std::string leftTable, rightTable;
    std::string leftCol, rightCol;
    bool anti = false;       // NOT EXISTS anti-semi join.
    bool leftOuter = false;  // LEFT OUTER JOIN.
    bool semi = false;       // EXISTS semi join.
    std::string innerTable;  // EXISTS/NOT EXISTS inner table.
};

struct AggTarget {
    AggFunc func;
    ExprPtr innerExpr;  // Aggregate input expression.
    std::string alias;
    bool isStar = false; // COUNT(*)
};

struct SelectTarget {
    ExprPtr expr;
    std::string alias;
    bool isAgg = false;
    std::optional<AggTarget> agg;
};

struct FromSubqueryAggInfo {
    std::string alias;  // FROM-subquery alias.
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;
    std::vector<JoinClause> joins;
    std::vector<PredPtr> filters;
    std::vector<SelectTarget> targets;
    std::vector<ExprPtr> groupBy;
};

struct OrderByItem {
    ExprPtr expr;
    bool descending = false;
};

// --- Aggregation Spec ---

struct AggSpec {
    AggFunc   func;
    ExprPtr   expr;       // Null for COUNT(*).
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
