#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace joule::tpch {

struct PartView {
    std::span<const std::int32_t> part_key;
    // Fixed-width 25-byte p_type values.
    std::span<const char> type;
    [[nodiscard]] std::size_t row_count() const noexcept { return part_key.size(); }
};

class PartStore {
public:
    explicit PartStore(const std::filesystem::path& colbin_path);
    ~PartStore();
    PartStore(PartStore&&) noexcept;
    PartStore& operator=(PartStore&&) noexcept;
    PartStore(const PartStore&) = delete;
    PartStore& operator=(const PartStore&) = delete;
    [[nodiscard]] PartView view() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::tpch
