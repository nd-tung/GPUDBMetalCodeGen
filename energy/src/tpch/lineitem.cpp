#include "joule/tpch/lineitem.hpp"

#include "joule/tpch/colbin.hpp"

namespace joule::tpch {

struct LineitemStore::Impl {
    ColbinStore store;
    LineitemView columns;

    explicit Impl(const std::filesystem::path& path) : store(path) {
        const auto optional_integer = [this](std::int32_t index) {
            return store.has_column(index)
                ? store.integer_column(index)
                : std::span<const std::int32_t>{};
        };
        const auto optional_float = [this](std::int32_t index) {
            return store.has_column(index)
                ? store.float_column(index)
                : std::span<const float>{};
        };
        const auto optional_char1 = [this](std::int32_t index) {
            return store.has_column(index)
                ? store.char_column(index, ColbinType::char1)
                : std::span<const char>{};
        };
        columns = LineitemView{
            optional_integer(0),
            optional_integer(1),
            store.float_column(4),
            store.float_column(5),
            store.float_column(6),
            optional_float(7),
            optional_char1(8),
            optional_char1(9),
            store.integer_column(10, ColbinType::date)};
    }
};

LineitemStore::LineitemStore(const std::filesystem::path& colbin_path)
    : impl_(std::make_unique<Impl>(colbin_path)) {}
LineitemStore::~LineitemStore() = default;
LineitemStore::LineitemStore(LineitemStore&&) noexcept = default;
LineitemStore& LineitemStore::operator=(LineitemStore&&) noexcept = default;
LineitemView LineitemStore::view() const noexcept { return impl_->columns; }
std::uint64_t LineitemStore::row_count() const noexcept { return impl_->store.row_count(); }

}  // namespace joule::tpch
