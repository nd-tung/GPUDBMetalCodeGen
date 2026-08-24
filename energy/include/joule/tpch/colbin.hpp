#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace joule::tpch {

enum class ColbinType : std::uint8_t {
    integer = 0,
    floating = 1,
    date = 2,
    char1 = 3,
    char_fixed = 4,
};

class ColbinStore {
public:
    explicit ColbinStore(const std::filesystem::path& path);
    ~ColbinStore();

    ColbinStore(ColbinStore&&) noexcept;
    ColbinStore& operator=(ColbinStore&&) noexcept;
    ColbinStore(const ColbinStore&) = delete;
    ColbinStore& operator=(const ColbinStore&) = delete;

    [[nodiscard]] std::uint64_t row_count() const noexcept;
    [[nodiscard]] bool has_column(std::int32_t index) const noexcept;
    [[nodiscard]] std::span<const std::int32_t> integer_column(
        std::int32_t index,
        ColbinType expected_type = ColbinType::integer) const;
    [[nodiscard]] std::span<const float> float_column(std::int32_t index) const;
    [[nodiscard]] std::span<const char> char_column(
        std::int32_t index,
        ColbinType expected_type,
        std::uint16_t fixed_width = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joule::tpch
