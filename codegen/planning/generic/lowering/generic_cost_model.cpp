#include "generic/lowering/generic_cost_model.h"
#include "core/infra.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace codegen {

namespace {

std::string costTypeName(const TypeInfo& type) {
    switch (type.type) {
        case DataType::INT: return "INT";
        case DataType::FLOAT: return "FLOAT";
        case DataType::DATE: return "DATE";
        case DataType::CHAR1: return "CHAR1";
        case DataType::CHAR_FIXED:
            return "CHAR_FIXED(" + std::to_string(type.fixedWidth) + ")";
    }
    return "UNKNOWN";
}

std::string relationRowBoundExpr(const GenericRelation& rel) {
    if (!rel.primaryKeyDomainSymbol.empty()) return rel.primaryKeyDomainSymbol;
    if (!rel.maxKeySymbol.empty()) return rel.maxKeySymbol;
    return "";
}

std::optional<int64_t> finiteValueCountForColumn(const GenericColumnExpr& col) {
    if (col.hasGroupDomain && col.domainMax >= col.domainMin) {
        return static_cast<int64_t>(col.domainMax) -
               static_cast<int64_t>(col.domainMin) + 1;
    }
    if (!col.charDomain.empty())
        return static_cast<int64_t>(col.charDomain.size());
    if (auto value = resolveGenericCostExpression(col.keyDomainSymbol))
        return static_cast<int64_t>(*value);
    return std::nullopt;
}

void collectCostColumnsFromExpr(const GenericExprPtr& expr,
                                std::vector<GenericColumnExpr>& out);

void collectCostColumnsFromPredicate(const GenericPredicatePtr& pred,
                                     std::vector<GenericColumnExpr>& out) {
    if (!pred) return;
    if (auto* cmp = std::get_if<GenericComparisonPred>(&pred->node)) {
        collectCostColumnsFromExpr(cmp->left, out);
        collectCostColumnsFromExpr(cmp->right, out);
        return;
    }
    if (auto* between = std::get_if<GenericBetweenPred>(&pred->node)) {
        collectCostColumnsFromExpr(between->expr, out);
        collectCostColumnsFromExpr(between->low, out);
        collectCostColumnsFromExpr(between->high, out);
        return;
    }
    if (auto* inList = std::get_if<GenericInListPred>(&pred->node)) {
        collectCostColumnsFromExpr(inList->expr, out);
        for (const auto& value : inList->values)
            collectCostColumnsFromExpr(value, out);
        return;
    }
    if (auto* like = std::get_if<GenericLikePred>(&pred->node)) {
        collectCostColumnsFromExpr(like->expr, out);
        return;
    }
    if (auto* logical = std::get_if<GenericLogicalPred>(&pred->node)) {
        for (const auto& child : logical->children)
            collectCostColumnsFromPredicate(child, out);
        return;
    }
}

void collectCostColumnsFromExpr(const GenericExprPtr& expr,
                                std::vector<GenericColumnExpr>& out) {
    if (!expr) return;
    if (auto* col = std::get_if<GenericColumnExpr>(&expr->node)) {
        out.push_back(*col);
        return;
    }
    if (std::holds_alternative<GenericLiteralExpr>(expr->node))
        return;
    if (auto* bin = std::get_if<GenericBinaryExpr>(&expr->node)) {
        collectCostColumnsFromExpr(bin->left, out);
        collectCostColumnsFromExpr(bin->right, out);
        return;
    }
    if (auto* caseExpr = std::get_if<GenericCaseExpr>(&expr->node)) {
        for (const auto& branch : caseExpr->branches) {
            collectCostColumnsFromPredicate(branch.condition, out);
            collectCostColumnsFromExpr(branch.result, out);
        }
        collectCostColumnsFromExpr(caseExpr->elseResult, out);
        return;
    }
    if (auto* func = std::get_if<GenericFunctionExpr>(&expr->node)) {
        for (const auto& arg : func->args)
            collectCostColumnsFromExpr(arg, out);
        return;
    }
    if (auto* agg = std::get_if<GenericAggregateExpr>(&expr->node)) {
        collectCostColumnsFromExpr(agg->arg, out);
        return;
    }
    if (auto* lookup = std::get_if<GenericScalarLookupExpr>(&expr->node)) {
        for (const auto& key : lookup->keys)
            collectCostColumnsFromExpr(key, out);
        return;
    }
}

void collectCostColumnsFromNode(const GenericRelNode& node,
                                std::vector<GenericColumnExpr>& out) {
    for (const auto& col : node.output.columns) {
        if (!col.relationInstance.valid() || col.name.empty()) continue;
        GenericColumnExpr expr;
        expr.relationInstance = col.relationInstance;
        expr.column = col.name;
        expr.type = col.type;
        out.push_back(std::move(expr));
    }

    if (auto* filter = std::get_if<GenericFilterDetail>(&node.detail)) {
        collectCostColumnsFromPredicate(filter->predicate, out);
        return;
    }
    if (auto* project = std::get_if<GenericProjectDetail>(&node.detail)) {
        for (const auto& projection : project->projections)
            collectCostColumnsFromExpr(projection.expr, out);
        return;
    }
    if (auto* join = std::get_if<GenericJoinDetail>(&node.detail)) {
        collectCostColumnsFromPredicate(join->predicate, out);
        return;
    }
    if (auto* aggregate = std::get_if<GenericAggregateDetail>(&node.detail)) {
        for (const auto& group : aggregate->groupBy)
            collectCostColumnsFromExpr(group, out);
        for (const auto& projection : aggregate->aggregates)
            collectCostColumnsFromExpr(projection.expr, out);
        collectCostColumnsFromPredicate(aggregate->having, out);
        return;
    }
    if (auto* sort = std::get_if<GenericSortDetail>(&node.detail)) {
        for (const auto& key : sort->keys)
            collectCostColumnsFromExpr(key.expr, out);
        return;
    }
}

std::string formatOptionalDouble(std::optional<double> value) {
    if (!value) return "unknown";
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << *value;
    return out.str();
}

std::string formatOptionalInt(std::optional<int64_t> value) {
    if (!value) return "unknown";
    return std::to_string(*value);
}

std::string formatCostNumber(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << value;
    return out.str();
}

bool materializedColumnExists(const std::vector<GenericMatColumnDesc>& cols,
                              const std::string& displayName) {
    return std::any_of(cols.begin(), cols.end(), [&](const auto& col) {
        return col.displayName == displayName;
    });
}

std::string trimCostExpression(const std::string& expr) {
    size_t first = 0;
    while (first < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[first]))) {
        ++first;
    }
    size_t last = expr.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(expr[last - 1]))) {
        --last;
    }
    return expr.substr(first, last - first);
}

std::optional<int> currentTpchScaleFactor() {
    const std::string& path = ::g_dataset_path;
    auto pos = path.find("SF-");
    if (pos == std::string::npos) return std::nullopt;
    pos += 3;
    int value = 0;
    bool any = false;
    while (pos < path.size() &&
           std::isdigit(static_cast<unsigned char>(path[pos]))) {
        any = true;
        value = value * 10 + (path[pos] - '0');
        ++pos;
    }
    if (!any || value <= 0) return std::nullopt;
    return value;
}

std::optional<size_t> readColbinRowCount(const std::string& table) {
    const std::string path = ::g_dataset_path + table + ".colbin";
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    colbin::FileHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in) return std::nullopt;
    if (std::memcmp(hdr.magic, colbin::MAGIC, sizeof(hdr.magic)) != 0 ||
        hdr.version != colbin::VERSION) {
        return std::nullopt;
    }
    return static_cast<size_t>(hdr.n_rows);
}

std::optional<size_t> fallbackTpchRowCount(const std::string& table) {
    auto sf = currentTpchScaleFactor();
    if (!sf) return std::nullopt;
    const size_t scale = static_cast<size_t>(*sf);
    if (table == "region") return 5;
    if (table == "nation") return 25;
    if (table == "supplier") return 10000 * scale;
    if (table == "part") return 200000 * scale;
    if (table == "partsupp") return 800000 * scale;
    if (table == "customer") return 150000 * scale;
    if (table == "orders") return 1500000 * scale;
    if (table == "lineitem") return 6000000 * scale;
    return std::nullopt;
}

std::optional<size_t> costTableRowCount(const std::string& table) {
    if (auto rows = readColbinRowCount(table)) return rows;
    return fallbackTpchRowCount(table);
}

std::optional<std::string> extractJsonStringField(const std::string& object,
                                                  const std::string& field) {
    const std::string key = "\"" + field + "\"";
    size_t pos = object.find(key);
    if (pos == std::string::npos) return std::nullopt;
    pos = object.find(':', pos + key.size());
    if (pos == std::string::npos) return std::nullopt;
    pos = object.find('"', pos + 1);
    if (pos == std::string::npos) return std::nullopt;
    size_t end = object.find('"', pos + 1);
    if (end == std::string::npos) return std::nullopt;
    return object.substr(pos + 1, end - pos - 1);
}

std::optional<size_t> extractJsonSizeField(const std::string& object,
                                           const std::string& field) {
    const std::string key = "\"" + field + "\"";
    size_t pos = object.find(key);
    if (pos == std::string::npos) return std::nullopt;
    pos = object.find(':', pos + key.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < object.size() &&
           std::isspace(static_cast<unsigned char>(object[pos]))) {
        ++pos;
    }
    if (pos >= object.size() || !std::isdigit(static_cast<unsigned char>(object[pos])))
        return std::nullopt;
    size_t value = 0;
    while (pos < object.size() &&
           std::isdigit(static_cast<unsigned char>(object[pos]))) {
        value = value * 10 + static_cast<size_t>(object[pos] - '0');
        ++pos;
    }
    return value;
}

std::map<std::pair<std::string, int>, size_t> loadMaxKeyCacheValues() {
    std::map<std::pair<std::string, int>, size_t> values;
    std::ifstream in(::g_dataset_path + ".maxkeys.json");
    if (!in) return values;
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = text.find('{', pos)) != std::string::npos) {
        size_t end = text.find('}', pos + 1);
        if (end == std::string::npos) break;
        const std::string object = text.substr(pos, end - pos + 1);
        auto file = extractJsonStringField(object, "file");
        auto col = extractJsonSizeField(object, "col");
        auto maxValue = extractJsonSizeField(object, "max");
        if (file && col && maxValue) {
            auto key = std::make_pair(*file, static_cast<int>(*col));
            auto it = values.find(key);
            if (it == values.end() || *maxValue > it->second)
                values[key] = *maxValue;
        }
        pos = end + 1;
    }
    return values;
}

std::optional<size_t> maxKeyFromCache(
        const std::map<std::pair<std::string, int>, size_t>& cache,
        const std::vector<std::pair<std::string, int>>& columns) {
    std::optional<size_t> out;
    for (const auto& column : columns) {
        auto it = cache.find(column);
        if (it == cache.end()) continue;
        out = out ? std::max(*out, it->second) : it->second;
    }
    return out;
}

void registerCostSymbolIfMissing(
        std::unordered_map<std::string, size_t>& symbols,
        const std::string& name,
        size_t value) {
    symbols.emplace(name, value);
}

std::unordered_map<std::string, size_t> buildScaleAwareCostSymbols() {
    std::unordered_map<std::string, size_t> symbols;
    const std::vector<std::string> tables = {
        "region", "nation", "supplier", "part", "partsupp",
        "customer", "orders", "lineitem"
    };
    std::unordered_map<std::string, size_t> rowsByTable;
    for (const auto& table : tables) {
        if (auto rows = costTableRowCount(table)) {
            rowsByTable[table] = *rows;
            registerCostSymbolIfMissing(symbols, tableSizeName(table), *rows);
            registerCostSymbolIfMissing(symbols, "num" + table, *rows);
        }
    }

    const auto cache = loadMaxKeyCacheValues();
    auto registerMaxKey = [&](const std::string& symbol,
                              const std::vector<std::pair<std::string, int>>& columns,
                              std::optional<size_t> fallbackMax) {
        std::optional<size_t> maxValue = maxKeyFromCache(cache, columns);
        if (!maxValue) maxValue = fallbackMax;
        if (maxValue) registerCostSymbolIfMissing(symbols, symbol, *maxValue + 1);
    };

    auto rowMax = [&](const std::string& table) -> std::optional<size_t> {
        auto it = rowsByTable.find(table);
        if (it == rowsByTable.end() || it->second == 0) return std::nullopt;
        return it->second;
    };

    registerMaxKey("maxCustkey",
                   {{"customer.colbin", 0}, {"orders.colbin", 1}},
                   rowMax("customer"));
    registerMaxKey("maxSuppkey",
                   {{"supplier.colbin", 0}, {"partsupp.colbin", 1},
                    {"lineitem.colbin", 2}},
                   rowMax("supplier"));
    registerMaxKey("maxPartkey",
                   {{"part.colbin", 0}, {"partsupp.colbin", 0},
                    {"lineitem.colbin", 1}},
                   rowMax("part"));
    std::optional<size_t> fallbackOrderMax;
    if (auto orderRows = rowMax("orders"))
        fallbackOrderMax = *orderRows * 4;
    registerMaxKey("maxOrderkey",
                   {{"orders.colbin", 0}, {"lineitem.colbin", 0}},
                   fallbackOrderMax);
    return symbols;
}

const std::unordered_map<std::string, size_t>& scaleAwareCostSymbols() {
    static std::string cachedDatasetPath;
    static std::unordered_map<std::string, size_t> cachedSymbols;
    if (cachedDatasetPath != ::g_dataset_path || cachedSymbols.empty()) {
        cachedDatasetPath = ::g_dataset_path;
        cachedSymbols = buildScaleAwareCostSymbols();
    }
    return cachedSymbols;
}

void populateScaleAwareCostSymbols(MetalSizeResolver& resolver) {
    for (const auto& [name, value] : scaleAwareCostSymbols()) {
        if (!resolver.hasSymbol(name)) resolver.registerSymbol(name, value);
    }
}

double symbolicRowEstimate(const std::string& expr) {
    if (auto parsed = resolveGenericCostExpression(expr)) return *parsed;
    return 1024.0 * 1024.0;
}

GenericCostAlternativeTrace makeAggregationCandidate(
        const std::string& name,
        bool available,
        double cost,
        std::string reason = {}) {
    if (!available && reason.empty()) reason = "candidate is not applicable";
    if (!available) cost = std::numeric_limits<double>::infinity();
    return {name, cost, std::move(reason)};
}

struct AggregationCostTerms {
    double inputRows = 0.0;
    double outputRows = 0.0;
    size_t inputWidth = 0;
    size_t outputWidth = 0;
    double aggregateSlots = 1.0;
    double denseBuckets = 1.0;
    double materializeBytes = 0.0;
    double outputBytes = 0.0;
    double denseStateBytes = 0.0;
};

AggregationCostTerms estimateAggregationCostTerms(
        const GenericAggregationCostInput& input,
        const GenericAggregationCandidateCostInput* candidate = nullptr) {
    AggregationCostTerms terms;
    terms.inputRows = symbolicRowEstimate(input.inputRowsExpr);
    terms.outputRows = std::min(
        terms.inputRows, symbolicRowEstimate(input.outputRowsExpr));
    terms.inputWidth = genericMatRowByteWidthEstimate(
        input.materializedInputColumns);
    const auto& outputColumns =
        candidate && !candidate->outputColumns.empty()
            ? candidate->outputColumns
            : input.outputColumns;
    terms.outputWidth = genericMatRowByteWidthEstimate(
        outputColumns, /*includeHidden=*/true);
    terms.aggregateSlots =
        static_cast<double>(std::max(
            1, candidate && candidate->aggregateSlots > 0
                   ? candidate->aggregateSlots
                   : input.aggregateSlots));
    const bool dynamicDenseDomain =
        candidate ? candidate->dynamicDenseDomain : input.dynamicDenseDomain;
    const int denseBuckets =
        candidate && candidate->denseBuckets > 0
            ? candidate->denseBuckets
            : input.denseBuckets;
    const std::string denseBucketsExpr =
        candidate ? candidate->denseBucketsExpr : input.denseBucketsExpr;
    if (!denseBucketsExpr.empty()) {
        terms.denseBuckets = symbolicRowEstimate(denseBucketsExpr);
    } else {
        terms.denseBuckets = dynamicDenseDomain
            ? terms.outputRows
            : static_cast<double>(std::max(1, denseBuckets));
    }
    terms.materializeBytes =
        terms.inputRows * static_cast<double>(std::max<size_t>(1, terms.inputWidth));
    terms.outputBytes =
        terms.outputRows * static_cast<double>(std::max<size_t>(1, terms.outputWidth));
    terms.denseStateBytes = terms.denseBuckets * terms.aggregateSlots * 4.0;
    return terms;
}

double directDensePipelineCost(const GenericAggregationCostInput& input,
                               const GenericAggregationCandidateCostInput* candidate,
                               const AggregationCostTerms& terms) {
    constexpr double kLaunchCost = 64.0 * 1024.0;
    constexpr double kDenseAtomicPenalty = 6.0;
    const int carriedStringRowRefs = candidate
        ? candidate->directPipelineCarriedStringRowRefs
        : input.directPipelineCarriedStringRowRefs;
    const int extraBuffers = candidate
        ? candidate->directPipelineExtraBuffers
        : input.directPipelineExtraBuffers;
    const bool activeBucketCompaction = candidate
        ? candidate->activeBucketCompaction
        : input.activeBucketCompaction;
    const double carriedStringPenalty =
        carriedStringRowRefs > 0
            ? terms.materializeBytes *
                  (1.10 + 0.15 *
                      static_cast<double>(
                          std::max(0, carriedStringRowRefs - 1)))
            : 0.0;
    const double extraBufferPenalty =
        static_cast<double>(std::max(0, extraBuffers)) *
        terms.inputRows * 2.0;
    return terms.outputBytes + terms.denseStateBytes +
           terms.inputRows * kDenseAtomicPenalty +
           carriedStringPenalty + extraBufferPenalty +
           kLaunchCost * (activeBucketCompaction ? 2.0 : 1.5);
}

double directDenseMaterializedCost(const GenericAggregationCostInput& input,
                                   const GenericAggregationCandidateCostInput* candidate,
                                   const AggregationCostTerms& terms) {
    constexpr double kLaunchCost = 64.0 * 1024.0;
    constexpr double kDenseAtomicPenalty = 6.0;
    const bool activeBucketCompaction = candidate
        ? candidate->activeBucketCompaction
        : input.activeBucketCompaction;
    return terms.materializeBytes + terms.outputBytes + terms.denseStateBytes +
           terms.inputRows * kDenseAtomicPenalty +
           kLaunchCost * (activeBucketCompaction ? 3.0 : 2.5);
}

} // namespace

const GenericCostRelationEstimate* GenericCostContext::relation(
        int relationId) const {
    for (const auto& rel : relations) {
        if (rel.relationId == relationId) return &rel;
    }
    return nullptr;
}

const GenericCostRelationInstanceEstimate* GenericCostContext::relationInstance(
        int relationInstanceId) const {
    for (const auto& inst : relationInstances) {
        if (inst.relationInstanceId == relationInstanceId) return &inst;
    }
    return nullptr;
}

const GenericCostColumnEstimate* GenericCostContext::column(
        int relationInstanceId,
        const std::string& columnName) const {
    for (const auto& col : columns) {
        if (col.relationInstanceId == relationInstanceId &&
            col.column == columnName) {
            return &col;
        }
    }
    return nullptr;
}

bool genericCostTraceEnabled() {
    const char* value = std::getenv("GPUDB_GENERIC_COST_TRACE");
    return value && value[0] != '\0' && value[0] != '0';
}

size_t genericCostTypeByteWidth(const TypeInfo& type) {
    switch (type.type) {
        case DataType::INT:
        case DataType::FLOAT:
        case DataType::DATE:
            return 4;
        case DataType::CHAR1:
            return 1;
        case DataType::CHAR_FIXED:
            return static_cast<size_t>(std::max(1, type.fixedWidth));
    }
    return 4;
}

std::optional<double> parseGenericCostPositiveNumber(const std::string& expr) {
    const std::string trimmed = trimCostExpression(expr);
    if (trimmed.empty()) return std::nullopt;

    double value = 0.0;
    for (char ch : trimmed) {
        if (ch < '0' || ch > '9') return std::nullopt;
        value = value * 10.0 + static_cast<double>(ch - '0');
        if (value > static_cast<double>(std::numeric_limits<int64_t>::max()))
            return std::nullopt;
    }
    if (value <= 0.0) return std::nullopt;
    return value;
}

std::optional<double> resolveGenericCostExpression(const std::string& expr) {
    const std::string trimmed = trimCostExpression(expr);
    if (trimmed.empty()) return std::nullopt;
    if (auto parsed = parseGenericCostPositiveNumber(trimmed)) return parsed;

    MetalSizeResolver resolver;
    populateScaleAwareCostSymbols(resolver);
    try {
        return static_cast<double>(resolver.resolve(trimmed));
    } catch (...) {
        return std::nullopt;
    }
}

GenericCostContext buildGenericCostContext(const GenericRelPlan& ir) {
    GenericCostContext context;
    context.traceEnabled = genericCostTraceEnabled();

    for (const auto& rel : ir.relations) {
        GenericCostRelationEstimate estimate;
        estimate.relationId = rel.id.value;
        estimate.name = rel.name;
        estimate.rowBoundExpr = relationRowBoundExpr(rel);
        estimate.rowBound = resolveGenericCostExpression(estimate.rowBoundExpr);
        estimate.primaryKeyColumn = rel.primaryKeyColumn;
        estimate.primaryKeyDomainExpr = rel.primaryKeyDomainSymbol;
        estimate.probePriority = rel.probePriority;
        estimate.virtualRelation = rel.virtualRelation;
        context.relations.push_back(std::move(estimate));
    }

    for (const auto& inst : ir.relationInstances) {
        GenericCostRelationInstanceEstimate estimate;
        estimate.relationInstanceId = inst.id.value;
        estimate.relationId = inst.relation.value;
        estimate.baseName = inst.baseName;
        estimate.alias = inst.alias;
        if (const auto* rel = context.relation(inst.relation.value)) {
            estimate.rowBoundExpr = rel->rowBoundExpr;
            estimate.rowBound = rel->rowBound;
        }
        context.relationInstances.push_back(std::move(estimate));
    }

    std::vector<GenericColumnExpr> sourceColumns;
    for (const auto& node : ir.nodes)
        collectCostColumnsFromNode(node, sourceColumns);

    std::set<std::tuple<int, std::string, std::string, std::string>> seen;
    for (const auto& col : sourceColumns) {
        if (!col.relationInstance.valid() || col.column.empty()) continue;
        auto key = std::make_tuple(col.relationInstance.value, col.table,
                                   col.alias, col.column);
        if (!seen.insert(key).second) continue;

        GenericCostColumnEstimate estimate;
        estimate.relationInstanceId = col.relationInstance.value;
        estimate.table = col.table;
        estimate.alias = col.alias;
        estimate.column = col.column;
        estimate.type = col.type;
        estimate.byteWidth = genericCostTypeByteWidth(col.type);
        estimate.finiteValueCount = finiteValueCountForColumn(col);
        estimate.keyDomainExpr = col.keyDomainSymbol;
        estimate.distinctDomainExpr = col.distinctDomainSymbol;

        if ((estimate.table.empty() || estimate.alias.empty()) &&
            col.relationInstance.valid()) {
            if (const auto* inst = ir.findRelationInstance(col.relationInstance)) {
                if (estimate.table.empty()) estimate.table = inst->baseName;
                if (estimate.alias.empty()) estimate.alias = inst->alias;
            }
        }
        context.columns.push_back(std::move(estimate));
    }

    return context;
}

std::string formatGenericCostDecision(const GenericCostDecisionTrace& decision) {
    std::ostringstream out;
    out << "COST_CHOICE operator=" << decision.operatorName;
    if (!decision.tag.empty()) out << " tag=" << decision.tag;
    out << " chosen=" << decision.chosen
        << " chosen_cost=" << formatCostNumber(decision.chosenCost);
    for (const auto& [name, value] : decision.estimates)
        out << " " << name << "=" << value;
    for (const auto& rejected : decision.rejected) {
        out << " rejected=" << rejected.name
            << " cost=" << formatCostNumber(rejected.cost);
        if (!rejected.reason.empty())
            out << " reason=\"" << rejected.reason << "\"";
    }
    return out.str();
}

std::string formatGenericCostContextSummary(const GenericCostContext& context,
                                            const std::string& route) {
    std::ostringstream out;
    out << "COST_CONTEXT route=" << route
        << " relations=" << context.relations.size()
        << " relation_instances=" << context.relationInstances.size()
        << " columns=" << context.columns.size();
    for (const auto& rel : context.relations) {
        out << " rel[" << rel.name
            << ":rows=" << (rel.rowBoundExpr.empty() ? "unknown"
                                                     : rel.rowBoundExpr)
            << ",static=" << formatOptionalDouble(rel.rowBound)
            << ",pk=" << (rel.primaryKeyColumn.empty() ? "none"
                                                       : rel.primaryKeyColumn)
            << "]";
    }
    for (const auto& col : context.columns) {
        out << " col[" << (col.alias.empty() ? col.table : col.alias)
            << "." << col.column
            << ":type=" << costTypeName(col.type)
            << ",width=" << col.byteWidth
            << ",domain=" << formatOptionalInt(col.finiteValueCount)
            << "]";
    }
    return out.str();
}

void appendGenericCostTrace(MetalQueryPlan& plan,
                            const GenericCostContext& context,
                            const std::string& route) {
    if (!context.traceEnabled) return;
    plan.costTraces.insert(plan.costTraces.begin(),
                           formatGenericCostContextSummary(context, route));
    for (const auto& decision : context.decisions)
        plan.costTraces.push_back(formatGenericCostDecision(decision));
}

void appendGenericCostDecisionTrace(MetalQueryPlan& plan,
                                    const GenericCostDecisionTrace& decision) {
    if (!genericCostTraceEnabled()) return;
    plan.costTraces.push_back(formatGenericCostDecision(decision));
}

std::optional<MetalQueryPlan> attachGenericCostTrace(
        std::optional<MetalQueryPlan>&& plan,
        const GenericRelPlan& ir,
        const std::string& route) {
    if (!plan) return std::nullopt;
    auto context = buildGenericCostContext(ir);
    appendGenericCostTrace(*plan, context, route);
    return std::move(plan);
}

DenseGroupCostChoice chooseDenseGroupPlan(
        const std::vector<IrGroupKeyDesc>& keys,
        const std::vector<IrPendingAgg>& pending,
        int totalBuckets,
        bool dynamicDomain,
        const KeyedCompactHavingSpec& havingSpec,
        const std::string& tag) {
    DenseGroupCostChoice choice;
    choice.trace.operatorName = "dense_group";
    choice.trace.tag = tag;

    auto finish = [&]() {
        choice.trace.chosen = choice.useDense ? "dense_group" : "hash_group";
        choice.trace.chosenCost = choice.useDense ? choice.denseCost
                                                  : choice.hashCost;
        choice.trace.estimates["keys"] = std::to_string(keys.size());
        choice.trace.estimates["aggregates"] = std::to_string(pending.size());
        choice.trace.estimates["total_buckets"] = std::to_string(totalBuckets);
        choice.trace.estimates["dynamic_domain"] =
            dynamicDomain ? "true" : "false";
        choice.trace.rejected.push_back({
            choice.useDense ? "hash_group" : "dense_group",
            choice.useDense ? choice.hashCost : choice.denseCost,
            choice.useDense ? "" : choice.reason
        });
        return choice;
    };

    if (keys.empty() || pending.empty() || totalBuckets <= 0) {
        choice.reason = "invalid dense group shape";
        return finish();
    }

    bool allAdds = true;
    int valueSlots = 0;
    for (const auto& agg : pending) {
        if (agg.atomicOp != "add") allAdds = false;
        valueSlots += agg.isLongPair ? 2 : 1;
    }
    valueSlots = std::max(1, valueSlots);

    bool hasDynamicKey = dynamicDomain;
    bool hasDynamicStringRowRef = false;
    for (const auto& key : keys) {
        if (!key.numValuesExpr.empty()) {
            hasDynamicKey = true;
            hasDynamicStringRowRef = hasDynamicStringRowRef ||
                                     key.stringRowRef;
        }
    }

    constexpr int kMaxBucketsForLocalReduce = 64;
    constexpr int kMinAggsForLocalReduce = 3;
    constexpr int kMaxSingleAggTinyBucketsForLocalReduce = 16;
    constexpr int kMaxBucketsForTgAtomicReduce = 256;
    const bool hasHavingTotal = !havingSpec.scalarTotalBuffer.empty();
    const bool localReduceEligible =
        allAdds && !hasHavingTotal && !dynamicDomain &&
        totalBuckets <= kMaxBucketsForLocalReduce &&
        (static_cast<int>(pending.size()) >= kMinAggsForLocalReduce ||
         totalBuckets <= kMaxSingleAggTinyBucketsForLocalReduce);
    const bool tgAtomicReduceEligible =
        allAdds && !hasHavingTotal && !dynamicDomain &&
        totalBuckets <= kMaxBucketsForTgAtomicReduce;

    const double keyCost = static_cast<double>(keys.size()) * 8.0;
    const double aggCost = static_cast<double>(pending.size()) * 12.0;
    choice.hashCost = 160.0 + keyCost + aggCost;
    choice.denseCost =
        static_cast<double>(totalBuckets * valueSlots) * 0.25 + keyCost +
        (localReduceEligible ? aggCost * 0.25 :
         tgAtomicReduceEligible ? aggCost * 1.5 : aggCost * 8.0);

    choice.trace.estimates["value_slots"] = std::to_string(valueSlots);
    choice.trace.estimates["all_adds"] = allAdds ? "true" : "false";
    choice.trace.estimates["has_dynamic_key"] =
        hasDynamicKey ? "true" : "false";
    choice.trace.estimates["has_dynamic_string_row_ref"] =
        hasDynamicStringRowRef ? "true" : "false";
    choice.trace.estimates["has_having_total"] =
        hasHavingTotal ? "true" : "false";
    choice.trace.estimates["local_reduce_eligible"] =
        localReduceEligible ? "true" : "false";
    choice.trace.estimates["tg_atomic_reduce_eligible"] =
        tgAtomicReduceEligible ? "true" : "false";

    if (hasDynamicKey) {
        choice.denseCost += hasDynamicStringRowRef ? 24.0 : 0.0;
        choice.denseCost += hasHavingTotal ? 32.0 : 0.0;
        choice.useDense = choice.denseCost < choice.hashCost;
        if (!choice.useDense)
            choice.reason = "hash group estimated cheaper for dynamic dense key";
        return finish();
    }

    if (!localReduceEligible && !tgAtomicReduceEligible) {
        choice.reason = hasHavingTotal
            ? "HAVING total prevents local dense reduction"
            : "dense group cannot reduce atomics locally";
        return finish();
    }

    choice.useDense = choice.denseCost < choice.hashCost;
    if (!choice.useDense)
        choice.reason = "hash group estimated cheaper";
    return finish();
}

int genericMatColumnByteWidthEstimate(const GenericMatColumnDesc& col) {
    if (col.stringLen > 0) return std::max(1, col.stringLen);
    if (col.isLongPair) return 8;
    if (col.metalType == "long" || col.metalType == "ulong" ||
        col.metalType == "double") {
        return 8;
    }
    if (col.metalType == "char" || col.metalType == "uchar") return 1;
    if (col.metalType == "short" || col.metalType == "ushort") return 2;
    return 4;
}

size_t genericMatRowByteWidthEstimate(
        const std::vector<GenericMatColumnDesc>& columns,
        bool includeHidden) {
    size_t bytes = 0;
    for (const auto& col : columns) {
        if (!includeHidden && col.displayName.rfind("__hidden_", 0) == 0)
            continue;
        bytes += static_cast<size_t>(genericMatColumnByteWidthEstimate(col));
    }
    return bytes;
}

double fdTopKGroupBoundEstimate(const std::string& outputBoundExpr,
                                const std::string& keyDomainExpr) {
    std::optional<double> bound =
        resolveGenericCostExpression(outputBoundExpr);
    if (auto keyDomain = resolveGenericCostExpression(keyDomainExpr)) {
        bound = bound ? std::min(*bound, *keyDomain) : keyDomain;
    }
    if (bound) return *bound;

    // Symbolic table/cardinality bounds are unknown at lowering time. Use a
    // generic large-table proxy so the guard still accounts for group count.
    return 1024.0 * 1024.0;
}

FdTopKLateMaterializationChoice chooseFdTopKLateMaterialization(
        const GenericSortSpec& sortSpec,
        const std::vector<GenericMatColumnDesc>& fullOutputs,
        const std::vector<GenericMatColumnDesc>& narrowOutputs,
        const std::string& outputBoundExpr,
        const std::string& keyDomainExpr,
        const std::string& tag) {
    FdTopKLateMaterializationChoice choice;
    choice.trace.operatorName = "fd_topk_late_materialization";
    choice.trace.tag = tag;

    auto finish = [&]() {
        const bool lateCandidateAvailable =
            choice.lateMaterializeBytes > 0.0 &&
            choice.fullCompactBytes > choice.lateMaterializeBytes &&
            choice.requiredSavings > 0.0;
        const double fullEffectiveCost = choice.fullCompactBytes;
        const double lateEffectiveCost = lateCandidateAvailable
            ? choice.lateMaterializeBytes + choice.requiredSavings
            : std::numeric_limits<double>::infinity();
        choice.trace.chosen = choice.useLateMaterialization
            ? "late_materialization"
            : "full_materialization";
        choice.trace.chosenCost = choice.useLateMaterialization
            ? lateEffectiveCost
            : fullEffectiveCost;
        choice.trace.estimates["full_width"] =
            std::to_string(choice.fullWidth);
        choice.trace.estimates["narrow_width"] =
            std::to_string(choice.narrowWidth);
        choice.trace.estimates["group_bound"] =
            formatCostNumber(choice.groupBound);
        choice.trace.estimates["limit_rows"] =
            formatCostNumber(choice.limitRows);
        choice.trace.estimates["gather_bytes"] =
            formatCostNumber(choice.gatherBytes);
        choice.trace.estimates["full_effective_cost"] =
            formatCostNumber(fullEffectiveCost);
        choice.trace.estimates["late_effective_cost"] =
            formatCostNumber(lateEffectiveCost);
        choice.trace.estimates["late_raw_cost"] =
            formatCostNumber(choice.lateMaterializeBytes);
        choice.trace.estimates["saved_bytes"] =
            formatCostNumber(choice.savedBytes);
        choice.trace.estimates["required_savings"] =
            formatCostNumber(choice.requiredSavings);
        choice.trace.rejected.push_back({
            choice.useLateMaterialization ? "full_materialization"
                                          : "late_materialization",
            choice.useLateMaterialization ? fullEffectiveCost
                                          : lateEffectiveCost,
            choice.useLateMaterialization ? "" : choice.reason
        });
        return choice;
    };

    if (sortSpec.limit <= 0 || sortSpec.keys.empty()) {
        choice.reason = "top-k requires a positive LIMIT and sort key";
        return finish();
    }
    if (narrowOutputs.empty()) {
        choice.reason = "narrow output set is empty";
        return finish();
    }
    for (const auto& sk : sortSpec.keys) {
        if (!materializedColumnExists(narrowOutputs, sk.column)) {
            choice.reason = "narrow output misses a sort key";
            return finish();
        }
    }

    choice.fullWidth = genericMatRowByteWidthEstimate(fullOutputs);
    choice.narrowWidth =
        genericMatRowByteWidthEstimate(narrowOutputs, /*includeHidden=*/true);
    if (choice.fullWidth <= choice.narrowWidth) {
        choice.reason = "narrow row is not smaller than full row";
        return finish();
    }

    choice.groupBound =
        fdTopKGroupBoundEstimate(outputBoundExpr, keyDomainExpr);
    choice.limitRows = std::min<double>(
        choice.groupBound, static_cast<double>(sortSpec.limit));
    if (choice.groupBound <= choice.limitRows * 2.0) {
        choice.reason = "LIMIT does not reduce enough groups";
        return finish();
    }

    choice.fullCompactBytes =
        static_cast<double>(choice.fullWidth) * choice.groupBound;
    const double narrowCompactBytes =
        static_cast<double>(choice.narrowWidth) * choice.groupBound;
    choice.gatherBytes =
        static_cast<double>(choice.fullWidth + choice.narrowWidth +
                            sizeof(uint32_t) * 2) *
        choice.limitRows;
    constexpr double kGatherLaunchBytes = 64.0 * 1024.0;
    constexpr double kMinSavingsBytes = 64.0 * 1024.0;
    constexpr double kMinSavingsToGatherCost = 3.0;

    choice.lateMaterializeBytes =
        narrowCompactBytes + choice.gatherBytes + kGatherLaunchBytes;
    if (choice.fullCompactBytes <= choice.lateMaterializeBytes) {
        choice.reason = "late materialization writes at least as many bytes";
        return finish();
    }
    choice.savedBytes = choice.fullCompactBytes - choice.lateMaterializeBytes;
    choice.requiredSavings =
        std::max(kMinSavingsBytes,
                 choice.gatherBytes * kMinSavingsToGatherCost);
    choice.useLateMaterialization =
        choice.savedBytes >= choice.requiredSavings;
    if (!choice.useLateMaterialization)
        choice.reason = "saved bytes do not cover gather overhead";
    return finish();
}

bool shouldUseFdTopKLateMaterialization(
        const GenericSortSpec& sortSpec,
        const std::vector<GenericMatColumnDesc>& fullOutputs,
        const std::vector<GenericMatColumnDesc>& narrowOutputs,
        const std::string& outputBoundExpr,
        const std::string& keyDomainExpr) {
    return chooseFdTopKLateMaterialization(
        sortSpec, fullOutputs, narrowOutputs, outputBoundExpr, keyDomainExpr)
        .useLateMaterialization;
}

MultiTableAggregationCostChoice chooseMultiTableAggregationPlan(
        const GenericAggregationCostInput& input,
        const std::string& preferredCandidate) {
    constexpr double kLaunchCost = 64.0 * 1024.0;
    constexpr double kHashRowPenalty = 16.0;
    constexpr double kFdStatePenalty = 8.0;

    std::vector<GenericAggregationCandidateCostInput> candidateInputs =
        input.candidates;
    if (candidateInputs.empty()) {
        candidateInputs.push_back(GenericAggregationCandidateCostInput{
            "materialized_hash_group", input.materializedHashAvailable,
            input.materializedHashAvailable ? "" : "candidate is not applicable",
            input.outputColumns,
            static_cast<int>(std::max<size_t>(1, input.aggregateCount)),
            input.denseBuckets,
            "",
            input.dynamicDenseDomain});

        candidateInputs.push_back(GenericAggregationCandidateCostInput{
            "direct_dense_pipeline",
            input.directDenseAvailable && input.directInputFused,
            input.directDenseAvailable
                ? "direct dense requires materialized input"
                : "direct dense candidate is not applicable",
            input.outputColumns,
            input.aggregateSlots,
            input.denseBuckets,
            input.denseBucketsExpr,
            input.dynamicDenseDomain,
            true,
            input.directPipelineCarriedStringRowRefs,
            input.directPipelineExtraBuffers,
            input.activeBucketCompaction});

        candidateInputs.push_back(GenericAggregationCandidateCostInput{
            "direct_dense_materialized",
            input.directDenseAvailable && input.directMaterializedAvailable,
            input.directDenseAvailable
                ? "direct dense materialized input is not applicable"
                : "direct dense candidate is not applicable",
            input.outputColumns,
            input.aggregateSlots,
            input.denseBuckets,
            input.denseBucketsExpr,
            input.dynamicDenseDomain,
            false,
            input.directPipelineCarriedStringRowRefs,
            input.directPipelineExtraBuffers,
            input.activeBucketCompaction});

        candidateInputs.push_back(GenericAggregationCandidateCostInput{
            input.fdLateTopK ? "fd_keyed_late_topk" : "fd_keyed_group",
            input.fdKeyedAvailable,
            "finite-domain keyed group candidate is not applicable",
            input.outputColumns,
            input.aggregateSlots,
            input.denseBuckets,
            "",
            input.dynamicDenseDomain,
            false,
            0,
            0,
            false,
            input.fdLateTopK});

        candidateInputs.push_back(GenericAggregationCandidateCostInput{
            "materialized_count_distinct",
            input.countDistinctAvailable,
            "count-distinct candidate is not applicable",
            input.outputColumns,
            input.aggregateSlots,
            input.denseBuckets,
            input.denseBucketsExpr,
            input.dynamicDenseDomain});
    }

    std::vector<GenericCostAlternativeTrace> candidates;
    const GenericAggregationCandidateCostInput* chosenInput = nullptr;
    double chosenCost = std::numeric_limits<double>::infinity();

    auto candidateCost = [&](const GenericAggregationCandidateCostInput& candidate,
                             const AggregationCostTerms& terms) {
        if (candidate.name == "materialized_hash_group") {
            return terms.materializeBytes + terms.outputBytes +
                   terms.inputRows * kHashRowPenalty + kLaunchCost * 3.0;
        }
        if (candidate.name == "direct_dense_pipeline") {
            return directDensePipelineCost(input, &candidate, terms);
        }
        if (candidate.name == "direct_dense_materialized") {
            return directDenseMaterializedCost(input, &candidate, terms);
        }
        if (candidate.name == "fd_keyed_group" ||
            candidate.name == "fd_keyed_late_topk") {
            return terms.materializeBytes + terms.outputBytes +
                   terms.denseStateBytes * kFdStatePenalty +
                   kLaunchCost * (candidate.fdLateTopK ? 4.0 : 3.0);
        }
        if (candidate.name == "materialized_count_distinct") {
            return terms.materializeBytes + terms.outputBytes +
                   terms.denseStateBytes * 2.0 +
                   terms.inputRows * (kHashRowPenalty + 4.0) +
                   kLaunchCost * 4.0;
        }
        return std::numeric_limits<double>::infinity();
    };

    for (const auto& candidate : candidateInputs) {
        const auto terms = estimateAggregationCostTerms(input, &candidate);
        double cost = candidate.available
            ? candidateCost(candidate, terms)
            : std::numeric_limits<double>::infinity();
        std::string reason = candidate.reason;
        if (!candidate.available && reason.empty())
            reason = "candidate is not applicable";
        candidates.push_back(makeAggregationCandidate(
            candidate.name, candidate.available, cost, reason));
        if (candidate.available && cost < chosenCost) {
            chosenCost = cost;
            chosenInput = &candidate;
        }
    }

    if (!chosenInput && !preferredCandidate.empty()) {
        auto preferredIt = std::find_if(candidateInputs.begin(), candidateInputs.end(),
            [&](const auto& candidate) {
                return candidate.name == preferredCandidate;
            });
        if (preferredIt != candidateInputs.end()) {
            chosenInput = &*preferredIt;
            chosenCost = candidates[static_cast<size_t>(
                std::distance(candidateInputs.begin(), preferredIt))].cost;
        }
    }
    if (!chosenInput && !candidateInputs.empty()) {
        chosenInput = &candidateInputs.front();
        chosenCost = candidates.front().cost;
    }

    const std::string chosenCandidate =
        chosenInput ? chosenInput->name : preferredCandidate;
    const auto chosenTerms = chosenInput
        ? estimateAggregationCostTerms(input, chosenInput)
        : estimateAggregationCostTerms(input);

    MultiTableAggregationCostChoice choice;
    choice.chosenCandidate = chosenCandidate;
    choice.chosenCost = chosenCost;
    GenericCostDecisionTrace& trace = choice.trace;
    trace.operatorName = "multi_table_aggregation";
    trace.tag = input.tag;
    trace.chosen = chosenCandidate;
    trace.chosenCost = chosenCost;
    trace.estimates["input_rows"] = formatCostNumber(chosenTerms.inputRows);
    trace.estimates["output_rows"] = formatCostNumber(chosenTerms.outputRows);
    trace.estimates["input_width"] = std::to_string(chosenTerms.inputWidth);
    trace.estimates["output_width"] = std::to_string(chosenTerms.outputWidth);
    trace.estimates["materialize_bytes"] =
        formatCostNumber(chosenTerms.materializeBytes);
    trace.estimates["group_keys"] = std::to_string(input.groupKeyCount);
    trace.estimates["aggregates"] = std::to_string(input.aggregateCount);
    trace.estimates["aggregate_slots"] =
        formatCostNumber(chosenTerms.aggregateSlots);
    trace.estimates["dense_buckets"] =
        formatCostNumber(chosenTerms.denseBuckets);
    trace.estimates["pipeline_carried_string_rowrefs"] =
        std::to_string(chosenInput
            ? chosenInput->directPipelineCarriedStringRowRefs
            : input.directPipelineCarriedStringRowRefs);
    trace.estimates["pipeline_extra_buffers"] =
        std::to_string(chosenInput
            ? chosenInput->directPipelineExtraBuffers
            : input.directPipelineExtraBuffers);
    trace.estimates["sort_limit"] = std::to_string(input.sortLimit);

    for (const auto& candidate : candidates) {
        if (candidate.name == chosenCandidate) continue;
        trace.rejected.push_back(candidate);
    }
    return choice;
}

DirectDenseInputModeChoice chooseDirectDenseInputMode(
        const GenericAggregationCostInput& input,
        const std::string& tag) {
    DirectDenseInputModeChoice choice;
    choice.trace.operatorName = "direct_dense_input_mode";
    choice.trace.tag = tag;

    const bool pipelineAvailable =
        input.directDenseAvailable && input.directInputFused;
    const bool materializedAvailable =
        input.directDenseAvailable && input.directMaterializedAvailable;
    const auto terms = estimateAggregationCostTerms(input);
    choice.pipelineCost = pipelineAvailable
        ? directDensePipelineCost(input, nullptr, terms)
        : std::numeric_limits<double>::infinity();
    choice.materializedCost = materializedAvailable
        ? directDenseMaterializedCost(input, nullptr, terms)
        : std::numeric_limits<double>::infinity();
    const double finiteBaseline = std::isfinite(choice.pipelineCost)
        ? choice.pipelineCost
        : choice.materializedCost;
    choice.requiredWin = std::isfinite(finiteBaseline)
        ? std::max(64.0 * 1024.0, finiteBaseline * 0.05)
        : std::numeric_limits<double>::infinity();
    const double pipelineEffectiveCost = pipelineAvailable
        ? choice.pipelineCost
        : std::numeric_limits<double>::infinity();
    const double materializedEffectiveCost = materializedAvailable
        ? choice.materializedCost +
              (pipelineAvailable ? choice.requiredWin : 0.0)
        : std::numeric_limits<double>::infinity();

    if (pipelineAvailable && materializedAvailable) {
        choice.usePipeline = pipelineEffectiveCost <= materializedEffectiveCost;
        if (!choice.usePipeline)
            choice.reason = "materialized input wins by the required margin";
        else if (choice.materializedCost < choice.pipelineCost)
            choice.reason = "materialized input does not win by the required margin";
    } else if (pipelineAvailable) {
        choice.usePipeline = true;
        choice.reason = "materialized input is not applicable";
    } else {
        choice.usePipeline = false;
        choice.reason = materializedAvailable
            ? "pipeline input is not applicable"
            : "no direct dense input mode is applicable";
    }

    choice.trace.chosen = choice.usePipeline
        ? "pipeline_input"
        : "materialized_input";
    choice.trace.chosenCost = choice.usePipeline
        ? pipelineEffectiveCost
        : materializedEffectiveCost;
    choice.trace.estimates["input_rows"] = formatCostNumber(terms.inputRows);
    choice.trace.estimates["input_width"] = std::to_string(terms.inputWidth);
    choice.trace.estimates["aggregate_slots"] =
        std::to_string(input.aggregateSlots);
    choice.trace.estimates["dense_buckets"] =
        formatCostNumber(terms.denseBuckets);
    choice.trace.estimates["materialize_bytes"] =
        formatCostNumber(terms.materializeBytes);
    choice.trace.estimates["materialized_effective_cost"] =
        formatCostNumber(materializedEffectiveCost);
    choice.trace.estimates["materialized_raw_cost"] =
        formatCostNumber(choice.materializedCost);
    choice.trace.estimates["pipeline_effective_cost"] =
        formatCostNumber(pipelineEffectiveCost);
    choice.trace.estimates["pipeline_carried_string_rowrefs"] =
        std::to_string(input.directPipelineCarriedStringRowRefs);
    choice.trace.estimates["pipeline_raw_cost"] =
        formatCostNumber(choice.pipelineCost);
    choice.trace.estimates["pipeline_extra_buffers"] =
        std::to_string(input.directPipelineExtraBuffers);
    choice.trace.estimates["required_win"] =
        formatCostNumber(choice.requiredWin);
    choice.trace.rejected.push_back({
        choice.usePipeline ? "materialized_input" : "pipeline_input",
        choice.usePipeline ? materializedEffectiveCost : pipelineEffectiveCost,
        choice.usePipeline && choice.reason.empty()
            ? "pipeline input estimated cheaper"
            : choice.reason
    });
    return choice;
}

ActiveBucketCompactionCostChoice chooseActiveBucketCompaction(
        const GenericAggregationCostInput& input,
        const std::string& tag) {
    constexpr double kLaunchCost = 64.0 * 1024.0;
    constexpr double kDenseSlotReadCost = 4.0;
    constexpr double kDenseBucketCost = 2.0;
    constexpr double kActiveTrackCost = 12.0;
    constexpr double kActiveRandomSlotReadCost = 18.0;
    constexpr double kActiveBucketCost = 8.0;
    constexpr double kMaxActiveFraction = 0.05;
    constexpr double kRequiredWinFraction = 0.20;

    ActiveBucketCompactionCostChoice choice;
    choice.trace.operatorName = "active_bucket_compaction";
    choice.trace.tag = tag;

    const auto terms = estimateAggregationCostTerms(input);
    choice.denseBuckets = std::max(1.0, terms.denseBuckets);
    choice.estimatedActiveBuckets =
        std::min(choice.denseBuckets, std::max(1.0, terms.outputRows));
    choice.activeFraction =
        choice.estimatedActiveBuckets / choice.denseBuckets;

    const double aggregateSlots = std::max(1.0, terms.aggregateSlots);
    choice.denseCompactCost =
        choice.denseBuckets *
            (aggregateSlots * kDenseSlotReadCost + kDenseBucketCost) +
        kLaunchCost;
    choice.activeCompactCost =
        terms.inputRows * kActiveTrackCost +
        choice.estimatedActiveBuckets *
            (aggregateSlots * kActiveRandomSlotReadCost +
             kActiveBucketCost) +
        kLaunchCost * 1.5;
    choice.requiredWin =
        std::max(64.0 * 1024.0,
                 choice.denseCompactCost * kRequiredWinFraction);

    const bool resolvedDynamicDomain =
        input.denseBuckets > 0 ||
        (!input.denseBucketsExpr.empty() &&
         resolveGenericCostExpression(input.denseBucketsExpr).has_value());
    const bool dynamicUnknownDomain =
        input.dynamicDenseDomain && !resolvedDynamicDomain;
    const bool sparseEnough = choice.activeFraction <= kMaxActiveFraction;
    double denseEffectiveCost = dynamicUnknownDomain
        ? std::numeric_limits<double>::infinity()
        : choice.denseCompactCost;
    double activeEffectiveCost =
        (!dynamicUnknownDomain && !sparseEnough)
            ? std::numeric_limits<double>::infinity()
            : choice.activeCompactCost +
                  (dynamicUnknownDomain ? 0.0 : choice.requiredWin);
    if (dynamicUnknownDomain) {
        choice.useActiveList = true;
        choice.reason = "dense scan requires a reliable dynamic domain bound";
    } else {
        choice.useActiveList =
            sparseEnough &&
            activeEffectiveCost < denseEffectiveCost;
    }
    if (!choice.useActiveList) {
        choice.reason = sparseEnough
            ? "active-list savings do not cover tracking and random-read cost"
            : "estimated active bucket fraction is too high";
    }

    choice.trace.chosen =
        choice.useActiveList ? "active_list" : "dense_scan";
    choice.trace.chosenCost =
        choice.useActiveList ? activeEffectiveCost
                             : denseEffectiveCost;
    choice.trace.estimates["active_buckets"] =
        formatCostNumber(choice.estimatedActiveBuckets);
    choice.trace.estimates["active_cost"] =
        formatCostNumber(choice.activeCompactCost);
    choice.trace.estimates["active_effective_cost"] =
        formatCostNumber(activeEffectiveCost);
    choice.trace.estimates["active_fraction"] =
        formatCostNumber(choice.activeFraction * 100.0) + "%";
    choice.trace.estimates["aggregate_slots"] =
        formatCostNumber(aggregateSlots);
    choice.trace.estimates["dense_buckets"] =
        formatCostNumber(choice.denseBuckets);
    choice.trace.estimates["dense_cost"] =
        formatCostNumber(choice.denseCompactCost);
    choice.trace.estimates["dense_effective_cost"] =
        formatCostNumber(denseEffectiveCost);
    choice.trace.estimates["dynamic_unknown_domain"] =
        dynamicUnknownDomain ? "true" : "false";
    choice.trace.estimates["input_rows"] =
        formatCostNumber(terms.inputRows);
    choice.trace.estimates["required_win"] =
        formatCostNumber(choice.requiredWin);
    choice.trace.rejected.push_back({
        choice.useActiveList ? "dense_scan" : "active_list",
        choice.useActiveList ? denseEffectiveCost
                             : activeEffectiveCost,
        choice.reason
    });
    return choice;
}

KeysetPropagationCostChoice chooseKeysetPropagation(
        const KeysetPropagationCostInput& input) {
    constexpr double kLaunchCost = 64.0 * 1024.0;
    constexpr double kBitmapProbePenalty = 4.0;
    constexpr double kMinSavingsBytes = 64.0 * 1024.0;
    constexpr double kSetupMargin = 0.25;

    KeysetPropagationCostChoice choice;
    choice.trace.operatorName = "keyset_propagation";
    choice.trace.tag = input.tag;

    const double buildRows = symbolicRowEstimate(input.buildRowsExpr);
    const double targetRows = symbolicRowEstimate(input.targetRowsExpr);
    const double keyDomain = std::max(
        1.0, resolveGenericCostExpression(input.keyDomainExpr)
                 .value_or(targetRows));
    const double activeFraction = std::clamp(
        input.estimatedActiveKeyFraction, 0.01, 1.0);
    const double bitmapBytes = std::ceil(keyDomain / 8.0);
    const double sourceProbeBytes = input.hasSourceBitmap
        ? buildRows * (static_cast<double>(input.keyByteWidth) +
                       kBitmapProbePenalty)
        : 0.0;
    const double depthPenalty =
        std::max(0, input.propagationDepth) * kLaunchCost * 0.5;
    const double reuse = static_cast<double>(std::max(1, input.reuseCount));

    choice.setupCost =
        buildRows * static_cast<double>(input.keyByteWidth) +
        sourceProbeBytes + bitmapBytes + kLaunchCost + depthPenalty;
    choice.probeCost =
        targetRows * (static_cast<double>(input.keyByteWidth) +
                      kBitmapProbePenalty);
    const double savedRows = targetRows * (1.0 - activeFraction);
    choice.savedBytes =
        savedRows * static_cast<double>(input.targetRowByteWidth) * reuse;
    choice.requiredSavings =
        (choice.setupCost + choice.probeCost) * (1.0 + kSetupMargin) +
        kMinSavingsBytes;
    choice.useKeyset = choice.savedBytes >= choice.requiredSavings;
    if (!choice.useKeyset)
        choice.reason = "estimated saved bytes do not cover keyset setup";

    const double useKeysetRawNetCost =
        choice.setupCost + choice.probeCost - choice.savedBytes;
    const double skipKeysetNetCost = 0.0;
    const double useKeysetEffectiveCost =
        choice.requiredSavings - choice.savedBytes;
    const double skipKeysetEffectiveCost = 0.0;
    choice.trace.chosen =
        choice.useKeyset ? "use_keyset" : "skip_keyset";
    choice.trace.chosenCost =
        choice.useKeyset ? useKeysetEffectiveCost : skipKeysetEffectiveCost;
    choice.trace.estimates["active_fraction"] =
        formatCostNumber(activeFraction * 100.0) + "%";
    choice.trace.estimates["bitmap_bytes"] = formatCostNumber(bitmapBytes);
    choice.trace.estimates["build_rows"] = formatCostNumber(buildRows);
    choice.trace.estimates["depth"] =
        std::to_string(std::max(0, input.propagationDepth));
    choice.trace.estimates["key_domain"] = formatCostNumber(keyDomain);
    choice.trace.estimates["key_width"] =
        std::to_string(input.keyByteWidth);
    choice.trace.estimates["probe_cost"] = formatCostNumber(choice.probeCost);
    choice.trace.estimates["required_savings"] =
        formatCostNumber(choice.requiredSavings);
    choice.trace.estimates["reuse"] =
        std::to_string(std::max(1, input.reuseCount));
    choice.trace.estimates["saved_bytes"] =
        formatCostNumber(choice.savedBytes);
    choice.trace.estimates["setup_cost"] =
        formatCostNumber(choice.setupCost);
    choice.trace.estimates["skip_keyset_raw_net_cost"] =
        formatCostNumber(skipKeysetNetCost);
    choice.trace.estimates["use_keyset_effective_cost"] =
        formatCostNumber(useKeysetEffectiveCost);
    choice.trace.estimates["use_keyset_raw_net_cost"] =
        formatCostNumber(useKeysetRawNetCost);
    choice.trace.estimates["target_rows"] = formatCostNumber(targetRows);
    choice.trace.estimates["target_width"] =
        std::to_string(input.targetRowByteWidth);
    choice.trace.rejected.push_back({
        choice.useKeyset ? "skip_keyset" : "use_keyset",
        choice.useKeyset ? skipKeysetEffectiveCost : useKeysetEffectiveCost,
        choice.reason
    });
    return choice;
}

} // namespace codegen
