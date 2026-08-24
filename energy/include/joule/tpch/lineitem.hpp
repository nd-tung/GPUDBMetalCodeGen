#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace joule::tpch {

struct LineitemView {
    std::span<const std::int32_t> order_key;
    std::span<const std::int32_t> part_key;
    std::span<const float> quantity;
    std::span<const float> extended_price;
    std::span<const float> discount;
    std::span<const float> tax;
    std::span<const char> return_flag;
    std::span<const char> line_status;
    std::span<const std::int32_t> ship_date_yyyymmdd;

    [[nodiscard]] std::size_t row_count() const noexcept {
        return quantity.size();
    }
};

class LineitemStore {
public:
    // Opens GPUDBMetalCodeGen's TPCHCB01/v2 lineitem.colbin and projects the
    // columns used by the analytical operator suite.
    explicit LineitemStore(const std::filesystem::path& colbin_path);
    ~LineitemStore();

    LineitemStore(LineitemStore&&) noexcept;
    LineitemStore& operator=(LineitemStore&&) noexcept;
    LineitemStore(const LineitemStore&) = delete;
    LineitemStore& operator=(const LineitemStore&) = delete;

    [[nodiscard]] LineitemView view() const noexcept;
    [[nodiscard]] std::uint64_t row_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::tpch
