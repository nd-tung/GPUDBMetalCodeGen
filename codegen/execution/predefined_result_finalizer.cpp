#include "predefined_result_finalizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>

namespace codegen {

namespace {

using HostResultSpec = MetalQueryPlan::HostResultSpec;

void mark(std::vector<std::string>* ops, const std::string& op) {
    if (ops) ops->push_back(op);
}

std::string intDateToStr(int d) {
    char buf[12];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", d / 10000, (d / 100) % 100, d % 100);
    return buf;
}

std::string fixedString(const char* base, int width, bool trimSpaces) {
    int len = 0;
    while (len < width && base[len] != '\0') len++;
    if (trimSpaces) {
        while (len > 0 && base[len - 1] == ' ') len--;
    }
    return std::string(base, len);
}

double valueAsDouble(const GenericResult::Value& value) {
    if (std::holds_alternative<double>(value)) return std::get<double>(value);
    if (std::holds_alternative<int64_t>(value)) return (double)std::get<int64_t>(value);
    return 0.0;
}

bool valueTruthy(const GenericResult::Value& value) {
    if (std::holds_alternative<double>(value)) return std::get<double>(value) != 0.0;
    if (std::holds_alternative<int64_t>(value)) return std::get<int64_t>(value) != 0;
    if (std::holds_alternative<std::string>(value)) return !std::get<std::string>(value).empty();
    return false;
}

int compareValues(const GenericResult::Value& a, const GenericResult::Value& b) {
    if (std::holds_alternative<std::string>(a) || std::holds_alternative<std::string>(b)) {
        std::string as = std::holds_alternative<std::string>(a)
            ? std::get<std::string>(a) : std::to_string(valueAsDouble(a));
        std::string bs = std::holds_alternative<std::string>(b)
            ? std::get<std::string>(b) : std::to_string(valueAsDouble(b));
        if (as < bs) return -1;
        if (as > bs) return 1;
        return 0;
    }
    double av = valueAsDouble(a);
    double bv = valueAsDouble(b);
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

void sortRows(GenericResult& result,
              const std::vector<HostResultSpec::SortKey>& keys) {
    if (keys.empty()) return;
    std::sort(result.rows.begin(), result.rows.end(),
        [&](const GenericResult::Row& a, const GenericResult::Row& b) {
            for (const auto& key : keys) {
                if (key.columnIndex < 0 ||
                    (size_t)key.columnIndex >= a.size() ||
                    (size_t)key.columnIndex >= b.size()) {
                    continue;
                }
                int cmp = compareValues(a[(size_t)key.columnIndex],
                                        b[(size_t)key.columnIndex]);
                if (cmp == 0) continue;
                return key.descending ? cmp > 0 : cmp < 0;
            }
            return false;
        });
}

MTL::Buffer* bufferFor(MetalGenericExecutor& executor, const std::string& name) {
    return name.empty() ? nullptr : executor.getAllocatedBuffer(name);
}

uint32_t readCounter(MetalGenericExecutor& executor, const std::string& name) {
    auto* buf = bufferFor(executor, name);
    if (!buf || buf->length() < sizeof(uint32_t)) return 0;
    return *static_cast<uint32_t*>(buf->contents());
}

size_t columnCapacity(MetalGenericExecutor& executor,
                      const HostResultSpec::BufferColumn& col) {
    auto* buf = bufferFor(executor, col.bufferName);
    if (!buf) return 0;
    if (col.elementType == "char") {
        return col.stringLen > 0 ? buf->length() / (size_t)col.stringLen : buf->length();
    }
    if (col.elementType == "long") return buf->length() / sizeof(int64_t);
    if (col.elementType == "float") return buf->length() / sizeof(float);
    return buf->length() / sizeof(uint32_t);
}

GenericResult::Value readBufferColumn(MetalGenericExecutor& executor,
                                      const HostResultSpec::BufferColumn& col,
                                      uint32_t row) {
    auto* buf = bufferFor(executor, col.bufferName);
    if (!buf) return (int64_t)0;
    if (col.elementType == "float") {
        const auto* data = static_cast<const float*>(buf->contents());
        return (double)data[row];
    }
    if (col.elementType == "long") {
        const auto* data = static_cast<const int64_t*>(buf->contents());
        return (int64_t)data[row];
    }
    if (col.elementType == "char") {
        const auto* data = static_cast<const char*>(buf->contents());
        if (col.stringLen > 0) {
            return fixedString(data + (size_t)row * (size_t)col.stringLen,
                               col.stringLen, col.trimSpaces);
        }
        return std::string(1, data[row]);
    }
    const auto* data = static_cast<const uint32_t*>(buf->contents());
    int64_t v = (int64_t)data[row];
    if (col.asDateString) return intDateToStr((int)v);
    return v;
}

GenericResult::Row readBufferRow(MetalGenericExecutor& executor,
                                 const HostResultSpec& spec,
                                 uint32_t row) {
    GenericResult::Row out;
    out.reserve(spec.bufferColumns.size());
    for (const auto& col : spec.bufferColumns) {
        out.push_back(readBufferColumn(executor, col, row));
    }
    return out;
}

std::optional<GenericResult::Value> readStaticCell(
        MetalGenericExecutor& executor,
        const GenericResult& existing,
        const HostResultSpec::Cell& cell) {
    auto existingCell = [&](int r, int c) -> GenericResult::Value {
        if (r < 0 || c < 0 ||
            (size_t)r >= existing.rows.size() ||
            (size_t)c >= existing.rows[(size_t)r].size()) {
            return (int64_t)0;
        }
        return existing.rows[(size_t)r][(size_t)c];
    };
    auto readFloat = [&](const std::string& name, int idx) -> double {
        auto* buf = bufferFor(executor, name);
        if (!buf || idx < 0 || (size_t)(idx + 1) * sizeof(float) > buf->length()) return 0.0;
        return (double)static_cast<const float*>(buf->contents())[idx];
    };
    auto readUInt = [&](const std::string& name, int idx) -> int64_t {
        auto* buf = bufferFor(executor, name);
        if (!buf || idx < 0 || (size_t)(idx + 1) * sizeof(uint32_t) > buf->length()) return 0;
        return (int64_t)static_cast<const uint32_t*>(buf->contents())[idx];
    };

    switch (cell.kind) {
        case HostResultSpec::CellKind::IntLiteral:
            return cell.intValue;
        case HostResultSpec::CellKind::StringLiteral:
            return cell.stringValue;
        case HostResultSpec::CellKind::BufferUInt:
            return readUInt(cell.bufferName, cell.index);
        case HostResultSpec::CellKind::BufferFloat:
            return readFloat(cell.bufferName, cell.index);
        case HostResultSpec::CellKind::ExistingCell:
            return existingCell(cell.row, cell.column);
        case HostResultSpec::CellKind::ExistingRatio: {
            double denom = valueAsDouble(existingCell(cell.denominatorRow,
                                                     cell.denominatorColumn));
            double num = valueAsDouble(existingCell(cell.numeratorRow,
                                                   cell.numeratorColumn));
            return denom != 0.0 ? (num / denom) * cell.multiplier : 0.0;
        }
        case HostResultSpec::CellKind::BufferRatio: {
            double denom = readFloat(cell.bufferName, cell.denominatorIndex);
            double num = readFloat(cell.bufferName, cell.numeratorIndex);
            return denom != 0.0 ? (num / denom) * cell.multiplier : 0.0;
        }
    }
    return std::nullopt;
}

void applyGpuSortRemap(GenericResult& result,
                       const MetalQueryPlan::GpuSort& gpuSort,
                       MetalGenericExecutor& executor) {
    auto* idxBuf = executor.getAllocatedBuffer(gpuSort.sortedIndexBuffer);
    if (!idxBuf) return;

    size_t nResults = 0;
    executor.tryGetSymbol(gpuSort.nResults, nResults);
    if (nResults == 0 || nResults > result.rows.size()) nResults = result.rows.size();

    const int* indices = static_cast<const int*>(idxBuf->contents());
    if (!indices) return;

    const size_t outRows = (gpuSort.limit >= 0)
        ? std::min(nResults, static_cast<size_t>(gpuSort.limit))
        : nResults;

    std::vector<GenericResult::Row> sortedRows;
    sortedRows.reserve(outRows);
    for (size_t i = 0; i < outRows; ++i) {
        int src = indices[i];
        size_t row = (src >= 0 && static_cast<size_t>(src) < result.rows.size())
            ? static_cast<size_t>(src) : i;
        sortedRows.push_back(std::move(result.rows[row]));
    }
    result.rows = std::move(sortedRows);
}

void finalizeExistingSort(const HostResultSpec& spec,
                          GenericResult& result,
                          std::vector<std::string>* hostOps) {
    if (!spec.columns.empty()) {
        result.columns.clear();
        for (const auto& col : spec.columns) result.columns.push_back({col.displayName, col.type});
    }
    sortRows(result, spec.existingSort);
    if (!spec.existingSort.empty()) mark(hostOps, "hostSort");
}

void finalizeStaticRows(const HostResultSpec& spec,
                        MetalGenericExecutor& executor,
                        GenericResult& result) {
    GenericResult out;
    for (const auto& col : spec.columns) out.columns.push_back({col.displayName, col.type});
    for (const auto& rowSpec : spec.staticRows) {
        if (rowSpec.includeIf) {
            auto cond = readStaticCell(executor, result, *rowSpec.includeIf);
            if (!cond || !valueTruthy(*cond)) continue;
        }
        GenericResult::Row row;
        row.reserve(rowSpec.values.size());
        for (const auto& cell : rowSpec.values) {
            auto v = readStaticCell(executor, result, cell);
            row.push_back(v ? *v : GenericResult::Value((int64_t)0));
        }
        out.rows.push_back(std::move(row));
    }
    result = std::move(out);
}

void finalizeBufferRows(const MetalQueryPlan& plan,
                        const HostResultSpec& spec,
                        MetalGenericExecutor& executor,
                        GenericResult& result,
                        std::vector<std::string>* hostOps) {
    GenericResult out;
    for (const auto& col : spec.bufferColumns) {
        std::string type = col.asDateString ? "string" :
            (col.elementType == "char" ? "string" : col.elementType);
        out.columns.push_back({col.displayName, type});
    }

    uint32_t n = readCounter(executor, spec.countBuffer);
    size_t cap = n;
    for (const auto& col : spec.bufferColumns) {
        size_t colCap = columnCapacity(executor, col);
        cap = cap == 0 ? colCap : std::min(cap, colCap);
    }
    if (n > cap) n = (uint32_t)cap;
    int limit = spec.limit >= 0 ? std::min((int)n, spec.limit) : (int)n;

    auto appendSource = [&](uint32_t src) {
        if (src >= n) return;
        out.rows.push_back(readBufferRow(executor, spec, src));
    };

    if (spec.useGpuSort && plan.gpuSort) {
        auto* idxBuf = executor.getAllocatedBuffer(plan.gpuSort->sortedIndexBuffer);
        if (idxBuf) {
            const int* order = static_cast<const int*>(idxBuf->contents());
            int gpuLimit = plan.gpuSort->limit >= 0 ? std::min(limit, plan.gpuSort->limit) : limit;
            for (int i = 0; i < gpuLimit; ++i) {
                int src = order[i];
                if (src >= 0) appendSource((uint32_t)src);
            }
        }
    } else if (!spec.identityCountBuffer.empty()) {
        uint32_t identityN = readCounter(executor, spec.identityCountBuffer);
        limit = std::min(limit, (int)identityN);
        for (int i = 0; i < limit; ++i) appendSource((uint32_t)i);
    }

    const size_t expectedRows = (size_t)limit;
    if (out.rows.size() < expectedRows) {
        out.rows.clear();
        for (uint32_t i = 0; i < n; ++i) out.rows.push_back(readBufferRow(executor, spec, i));
        sortRows(out, spec.fallbackSort);
        if (!spec.fallbackSort.empty()) mark(hostOps, "hostSort");
        if ((int)out.rows.size() > limit) out.rows.resize((size_t)limit);
    }

    result = std::move(out);
}

} // namespace

void finalizeHostResult(const MetalQueryPlan& plan,
                        MetalGenericExecutor& executor,
                        GenericResult& result,
                        std::vector<std::string>* hostOps) {
    if (!plan.hostResult) {
        if (plan.gpuSort && !result.columns.empty()) {
            applyGpuSortRemap(result, *plan.gpuSort, executor);
        }
        return;
    }

    const auto& spec = *plan.hostResult;
    switch (spec.kind) {
        case HostResultSpec::Kind::ExistingSort:
            if (plan.gpuSort && !result.columns.empty()) {
                applyGpuSortRemap(result, *plan.gpuSort, executor);
            }
            finalizeExistingSort(spec, result, hostOps);
            break;
        case HostResultSpec::Kind::StaticRows:
            finalizeStaticRows(spec, executor, result);
            break;
        case HostResultSpec::Kind::BufferRows:
            finalizeBufferRows(plan, spec, executor, result, hostOps);
            break;
    }
}

} // namespace codegen
