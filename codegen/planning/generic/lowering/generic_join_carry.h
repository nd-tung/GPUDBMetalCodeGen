#pragma once

#include "generic/ir/generic_relational_ir.h"

#include <map>
#include <string>

namespace codegen {

// Carries preserve values after a join advances past their source row.
struct IrCarryColumn {
    GenericColumnExpr column;
    std::string varName;
    // Fixed strings carry source row ids for later byte reads.
    std::string rowVarName;
    std::string bufferName;
};

// Relation-instance id -> source column name -> carried value.
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
