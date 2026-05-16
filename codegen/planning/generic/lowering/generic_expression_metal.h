#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <cstdint>
#include <optional>
#include <string>

namespace codegen {

std::string sanitizeIdentifier(std::string name);
std::string metalTypeForType(const TypeInfo& type);
int fixedStringLenForExpr(const GenericExprPtr& expr);
std::string literalToMetal(const GenericLiteralExpr& lit);
std::optional<std::string> stringLiteralValue(const GenericExprPtr& expr);
std::optional<int64_t> integerStringLiteralValue(const GenericExprPtr& expr);
std::string genericMetalCharLiteral(char ch);
std::string fixedStringEqMetalFromPointer(const std::string& basePtr,
                                          int width,
                                          const std::string& literal);
std::string fixedStringEqMetal(const GenericColumnExpr& col,
                               const std::string& literal,
                               const std::string& idxVar);
std::optional<std::string> fixedStringLikeMetal(const GenericLikePred& like,
                                                const std::string& idxVar);
std::string functionExprToMetal(const GenericFunctionExpr& fn,
                                const std::string& idxVar);
std::string genericExprToMetal(const GenericExprPtr& expr,
                               const std::string& idxVar);
std::string materializeExprToMetal(const GenericExprPtr& expr,
                                   const std::string& idxVar);
std::string cmpOpToMetal(CmpOp op);
std::string genericPredicateToMetal(const GenericPredicatePtr& pred,
                                    const std::string& idxVar);
bool materializeExprSupported(const GenericExprPtr& expr);
bool predicateSupported(const GenericPredicatePtr& pred);

} // namespace codegen
