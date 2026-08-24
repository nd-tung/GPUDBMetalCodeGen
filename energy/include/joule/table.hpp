#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace joule {

enum class DataType {
    int32,
    int64,
    float32,
    float64,
};

[[nodiscard]] constexpr std::size_t byte_width(DataType type) {
    switch (type) {
        case DataType::int32:
        case DataType::float32:
            return 4;
        case DataType::int64:
        case DataType::float64:
            return 8;
    }
    return 0;
}

struct Column {
    void* data{};
    std::uint64_t count{};
    DataType type{DataType::int32};

    [[nodiscard]] std::size_t size_bytes() const {
        return static_cast<std::size_t>(count) * byte_width(type);
    }
};

struct Table {
    std::vector<Column> columns;

    [[nodiscard]] std::uint64_t row_count() const {
        if (columns.empty()) {
            return 0;
        }
        const auto expected = columns.front().count;
        for (const auto& column : columns) {
            if (column.count != expected) {
                throw std::logic_error("all columns in a table must have the same row count");
            }
        }
        return expected;
    }
};

}  // namespace joule

