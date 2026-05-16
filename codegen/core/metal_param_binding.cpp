#include "metal_param_binding.h"
#include <sstream>
#include <cctype>
#include <algorithm>

namespace codegen {

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t");
    return s.substr(start, end - start + 1);
}

static bool isAllDigits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

size_t MetalSizeResolver::resolve(const std::string& expr) const {
    std::string e = trim(expr);
    if (e.empty())
        throw std::runtime_error("MetalSizeResolver: empty expression");

    // Strip matching outer parentheses.
    while (e.size() >= 2 && e.front() == '(' && e.back() == ')') {
        int depth = 0;
        bool matched = true;
        for (size_t i = 0; i < e.size() - 1; i++) {
            if (e[i] == '(') depth++;
            else if (e[i] == ')') depth--;
            if (depth == 0) { matched = false; break; }
        }
        if (matched) e = trim(e.substr(1, e.size() - 2));
        else break;
    }

    if (isAllDigits(e))
        return std::stoull(e);

    if (hasSymbol(e))
        return getSymbol(e);

    {
        const std::string fn = "next_pow2";
        if (e.size() > fn.size() + 2 && e.compare(0, fn.size(), fn) == 0 &&
            e[fn.size()] == '(' && e.back() == ')') {
            std::string inner = trim(e.substr(fn.size() + 1, e.size() - fn.size() - 2));
            size_t v = resolve(inner);
            if (v <= 1) return 1;
            size_t p = 1;
            while (p < v) p <<= 1;
            return p;
        }
    }

    auto findLastTopLevel = [&](const std::string& ops) -> std::pair<size_t, char> {
        int depth = 0;
        size_t bestPos = std::string::npos;
        char bestOp = 0;
        for (size_t i = 0; i < e.size(); i++) {
            if (e[i] == '(') depth++;
            else if (e[i] == ')') depth--;
            else if (depth == 0 && i > 0 && ops.find(e[i]) != std::string::npos) {
                bestPos = i;
                bestOp = e[i];
            }
        }
        return {bestPos, bestOp};
    };

    auto [pos1, op1] = findLastTopLevel("+-");
    if (pos1 != std::string::npos) {
        std::string lhs = trim(e.substr(0, pos1));
        std::string rhs = trim(e.substr(pos1 + 1));
        size_t lval = resolve(lhs);
        size_t rval = resolve(rhs);
        return (op1 == '+') ? lval + rval : lval - rval;
    }

    auto [pos2, op2] = findLastTopLevel("*/");
    if (pos2 != std::string::npos) {
        std::string lhs = trim(e.substr(0, pos2));
        std::string rhs = trim(e.substr(pos2 + 1));
        size_t lval = resolve(lhs);
        size_t rval = resolve(rhs);
        return (op2 == '*') ? lval * rval : (rval == 0 ? 0 : lval / rval);
    }

    throw std::runtime_error("MetalSizeResolver: cannot resolve '" + expr + "'");
}

} // namespace codegen
