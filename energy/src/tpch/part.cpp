#include "joule/tpch/part.hpp"

#include "joule/tpch/colbin.hpp"

namespace joule::tpch {

struct PartStore::Impl {
    ColbinStore store;
    PartView columns;
    explicit Impl(const std::filesystem::path& path) : store(path) {
        columns = PartView{
            store.integer_column(0),
            store.char_column(4, ColbinType::char_fixed, 25)};
    }
};

PartStore::PartStore(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}
PartStore::~PartStore() = default;
PartStore::PartStore(PartStore&&) noexcept = default;
PartStore& PartStore::operator=(PartStore&&) noexcept = default;
PartView PartStore::view() const noexcept { return impl_->columns; }

}  // namespace joule::tpch
