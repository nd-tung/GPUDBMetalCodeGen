#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <map>
#include <string>

namespace codegen {

struct IrCarryColumn {
    GenericColumnExpr column;
    std::string varName;
    std::string bufferName;
};

using IrCarryMap = std::map<int, std::map<std::string, IrCarryColumn>>;

int materializedStringLenForExpr(const GenericExprPtr& expr,
                                 const IrCarryMap& carries);

std::string genericExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                           const std::string& idxVar,
                                           const IrCarryMap& carries);

std::string materializeExprToMetalWithCarryMap(const GenericExprPtr& expr,
                                               const std::string& idxVar,
                                               const IrCarryMap& carries);

std::string genericPredicateToMetalWithCarryMap(const GenericPredicatePtr& pred,
                                                const std::string& idxVar,
                                                const IrCarryMap& carries);

bool exprNeedsCarriedString(const GenericExprPtr& expr,
                            const IrCarryMap& carries);

} // namespace codegen
