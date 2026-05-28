#pragma once

#include "catalog.hpp"
#include "generic/ir/analyzed_query.h"

#include "../../../../third_party/nlohmann/json.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace codegen {
namespace analyzed_query_internal {

// Scalar subqueries collected by walkExpr and transferred to Generic source metadata.
struct ScalarSubqueryInfo {
    std::string sql; // Serialized subselect AST.
};

struct AnalyzeScope {
    std::vector<std::string> tables;
    std::vector<std::string> aliases;
    std::unordered_map<std::string, std::string> aliasToTable;

    static AnalyzeScope fromTables(const std::vector<std::string>& tables,
                                   const std::vector<std::string>& aliases = {}) {
        AnalyzeScope scope;
        scope.tables = tables;
        scope.aliases = aliases;
        if (scope.aliases.size() < scope.tables.size()) {
            for (size_t i = scope.aliases.size(); i < scope.tables.size(); ++i)
                scope.aliases.push_back(scope.tables[i]);
        }
        for (size_t i = 0; i < scope.tables.size(); ++i) {
            const std::string& alias = scope.aliases[i].empty()
                ? scope.tables[i]
                : scope.aliases[i];
            scope.aliasToTable[alias] = scope.tables[i];
            if (!scope.aliasToTable.count(scope.tables[i]))
                scope.aliasToTable[scope.tables[i]] = scope.tables[i];
        }
        return scope;
    }

    static AnalyzeScope fromAnalyzed(const AnalyzedQuery& aq) {
        return fromTables(aq.tables, aq.tableAliases);
    }

    static AnalyzeScope fromSubqueryFirst(const AnalyzedQuery& aq,
                                          size_t subqueryStart) {
        AnalyzeScope scope;
        for (size_t i = subqueryStart; i < aq.tables.size(); ++i) {
            scope.tables.push_back(aq.tables[i]);
            scope.aliases.push_back(i < aq.tableAliases.size()
                ? aq.tableAliases[i]
                : aq.tables[i]);
        }
        for (size_t i = 0; i < subqueryStart && i < aq.tables.size(); ++i) {
            scope.tables.push_back(aq.tables[i]);
            scope.aliases.push_back(i < aq.tableAliases.size()
                ? aq.tableAliases[i]
                : aq.tables[i]);
        }
        return fromTables(scope.tables, scope.aliases);
    }
};

struct AnalyzeContext {
    const SchemaProvider& schema;
    Catalog catalog;
    std::unordered_map<std::string, std::string> aliasMap;
    std::unordered_map<std::string, std::string> aliasRewriteMap;
    std::unordered_map<std::string, ColRef> subqueryAliasMap;
    std::unordered_map<std::string, ExprPtr> subqueryExprMap;
    std::map<std::string, std::pair<nlohmann::json, std::vector<std::string>>> views;
    std::vector<ScalarSubqueryInfo> scalarSubqueries;

    explicit AnalyzeContext(const SchemaProvider& schema)
        : schema(schema), catalog(Catalog::fromSchemaProvider(schema)) {}
};

inline void rebuildAliasMap(AnalyzeContext& ctx, const AnalyzedQuery& aq,
                            bool includeTableNames = true) {
    ctx.aliasMap.clear();
    for (size_t i = 0; i < aq.tables.size() && i < aq.tableAliases.size(); ++i) {
        if (includeTableNames || aq.tableAliases[i] != aq.tables[i])
            ctx.aliasMap[aq.tableAliases[i]] = aq.tables[i];
    }
}

struct ResolvedName {
    std::string table;
    std::string column;
    DataType type = DataType::INT;
    int fixedWidth = 0;
    std::string tableAlias;
    bool catalogResolved = false;
};

class NameResolver {
public:
    NameResolver(const AnalyzeContext& ctx, const AnalyzeScope& scope)
        : ctx_(ctx), scope_(scope) {}

    std::optional<ResolvedName> resolveColumn(const std::string& colName,
                                              const std::string& qualifier) const {
        if (colName.empty()) return std::nullopt;
        std::string effectiveQualifier = qualifier;
        if (!effectiveQualifier.empty()) {
            auto rewrite = ctx_.aliasRewriteMap.find(effectiveQualifier);
            if (rewrite != ctx_.aliasRewriteMap.end())
                effectiveQualifier = rewrite->second;
        }

        if (!effectiveQualifier.empty())
            return resolveQualified(colName, effectiveQualifier);
        return resolveUnqualified(colName);
    }

private:
    const AnalyzeContext& ctx_;
    const AnalyzeScope& scope_;

    std::optional<ResolvedName> metadataFor(const std::string& table,
                                            const std::string& colName,
                                            const std::string& alias = {}) const {
        ResolvedName out;
        out.table = table;
        out.column = colName;
        out.tableAlias = alias;
        if (auto* catTable = ctx_.catalog.findTable(table)) {
            if (auto* col = catTable->findColumn(colName)) {
                out.type = col->type;
                out.fixedWidth = col->fixedWidth;
                out.catalogResolved = true;
                return out;
            }
        }
        if (ctx_.schema.hasColumn(table, colName)) {
            out.type = ctx_.schema.columnType(table, colName);
            if (out.type == DataType::CHAR_FIXED)
                out.fixedWidth = ctx_.schema.columnFixedWidth(table, colName);
            return out;
        }
        return std::nullopt;
    }

    std::optional<ResolvedName> resolveQualified(
            const std::string& colName,
            const std::string& qualifier) const {
        auto aliasIt = scope_.aliasToTable.find(qualifier);
        std::string table = aliasIt != scope_.aliasToTable.end()
            ? aliasIt->second
            : qualifier;
        std::string alias = (aliasIt != scope_.aliasToTable.end() &&
                             qualifier != table)
            ? qualifier
            : "";
        return metadataFor(table, colName, alias);
    }

    std::optional<ResolvedName> resolveUnqualified(
            const std::string& colName) const {
        for (size_t i = 0; i < scope_.tables.size(); ++i) {
            const std::string& table = scope_.tables[i];
            std::string alias;
            if (i < scope_.aliases.size() && scope_.aliases[i] != table)
                alias = scope_.aliases[i];
            if (auto resolved = metadataFor(table, colName, alias))
                return resolved;
        }
        return std::nullopt;
    }
};

} // namespace analyzed_query_internal
} // namespace codegen
