#include "joule/tpch/orders.hpp"

#include "joule/tpch/colbin.hpp"

namespace joule::tpch {

struct OrdersStore::Impl {
    ColbinStore store;
    OrdersView columns;
    explicit Impl(const std::filesystem::path& path) : store(path) {
        columns = OrdersView{store.integer_column(0), store.float_column(3)};
    }
};

OrdersStore::OrdersStore(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}
OrdersStore::~OrdersStore() = default;
OrdersStore::OrdersStore(OrdersStore&&) noexcept = default;
OrdersStore& OrdersStore::operator=(OrdersStore&&) noexcept = default;
OrdersView OrdersStore::view() const noexcept { return impl_->columns; }

}  // namespace joule::tpch
