#include "joule/tpch/colbin.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace joule::tpch {
namespace {

constexpr std::array<char, 8> colbin_magic{'T', 'P', 'C', 'H', 'C', 'B', '0', '1'};
constexpr std::uint32_t colbin_version = 2;
constexpr std::uint64_t colbin_alignment = 16'384;

struct ColbinHeader {
    std::array<char, 8> magic{};
    std::uint32_t version{};
    std::uint32_t column_count{};
    std::uint64_t row_count{};
    std::uint64_t source_size{};
    std::int64_t source_mtime_ns{};
    std::uint64_t reserved{};
};

struct ColbinDescriptor {
    std::int32_t column_index{};
    std::uint8_t data_type{};
    std::uint8_t reserved0{};
    std::uint16_t fixed_width{};
    std::uint64_t offset{};
    std::uint64_t size_bytes{};
    std::uint64_t reserved1{};
};

static_assert(sizeof(ColbinHeader) == 48);
static_assert(sizeof(ColbinDescriptor) == 32);

[[nodiscard]] std::runtime_error system_error(
    std::string_view action,
    const std::filesystem::path& path) {
    return std::runtime_error(
        std::string(action) + " '" + path.string() + "': " + std::strerror(errno));
}

[[nodiscard]] std::size_t element_size(
    ColbinType type,
    std::uint16_t fixed_width) {
    switch (type) {
        case ColbinType::integer:
        case ColbinType::floating:
        case ColbinType::date:
            return 4;
        case ColbinType::char1:
            return 1;
        case ColbinType::char_fixed:
            return fixed_width;
    }
    return 0;
}

}  // namespace

struct ColbinStore::Impl {
    int descriptor{-1};
    void* mapping{};
    std::size_t mapping_size{};
    std::uint64_t rows{};
    std::filesystem::path path;
    std::unordered_map<std::int32_t, ColbinDescriptor> columns;

    explicit Impl(const std::filesystem::path& requested_path) : path(requested_path) {
        descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor < 0) {
            throw system_error("could not open", path);
        }
        try {
            struct stat status {};
            if (::fstat(descriptor, &status) != 0) {
                throw system_error("could not stat", path);
            }
            if (status.st_size < static_cast<off_t>(sizeof(ColbinHeader))) {
                throw std::runtime_error("colbin file is too small: " + path.string());
            }
            mapping_size = static_cast<std::size_t>(status.st_size);
            mapping = ::mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (mapping == MAP_FAILED) {
                mapping = nullptr;
                throw system_error("could not mmap", path);
            }
            static_cast<void>(::madvise(mapping, mapping_size, MADV_SEQUENTIAL));

            ColbinHeader header;
            std::memcpy(&header, mapping, sizeof(header));
            const auto descriptor_bytes =
                static_cast<std::uint64_t>(header.column_count) * sizeof(ColbinDescriptor);
            if (header.magic != colbin_magic || header.version != colbin_version ||
                header.row_count == 0 ||
                header.row_count > std::numeric_limits<std::uint32_t>::max() ||
                descriptor_bytes > mapping_size - sizeof(ColbinHeader)) {
                throw std::runtime_error(
                    "unsupported GPUDBMetalCodeGen colbin: " + path.string());
            }

            rows = header.row_count;
            const auto* descriptors = reinterpret_cast<const ColbinDescriptor*>(
                static_cast<const char*>(mapping) + sizeof(ColbinHeader));
            columns.reserve(header.column_count);
            for (std::uint32_t index = 0; index < header.column_count; ++index) {
                const auto& column = descriptors[index];
                const auto type = static_cast<ColbinType>(column.data_type);
                const auto item_bytes = element_size(type, column.fixed_width);
                if (item_bytes == 0 || column.offset % colbin_alignment != 0 ||
                    column.offset > mapping_size ||
                    column.size_bytes > mapping_size - column.offset ||
                    rows > std::numeric_limits<std::uint64_t>::max() / item_bytes ||
                    column.size_bytes != rows * item_bytes) {
                    throw std::runtime_error(
                        "invalid column descriptor " + std::to_string(column.column_index) +
                        " in " + path.string());
                }
                columns[column.column_index] = column;
            }
        } catch (...) {
            reset();
            throw;
        }
    }

    ~Impl() { reset(); }

    void reset() noexcept {
        if (mapping != nullptr) {
            ::munmap(mapping, mapping_size);
            mapping = nullptr;
        }
        if (descriptor >= 0) {
            ::close(descriptor);
            descriptor = -1;
        }
    }

    [[nodiscard]] const ColbinDescriptor& require(
        std::int32_t index,
        ColbinType type,
        std::uint16_t width = 0) const {
        const auto found = columns.find(index);
        if (found == columns.end()) {
            throw std::runtime_error(
                path.filename().string() + " is missing column " + std::to_string(index));
        }
        const auto& column = found->second;
        if (column.data_type != static_cast<std::uint8_t>(type) ||
            (type == ColbinType::char_fixed && column.fixed_width != width)) {
            throw std::runtime_error(
                "unexpected type for column " + std::to_string(index) + " in " +
                path.string());
        }
        return column;
    }

    template <typename T>
    [[nodiscard]] std::span<const T> span_for(const ColbinDescriptor& column) const {
        return {
            reinterpret_cast<const T*>(
                static_cast<const char*>(mapping) + column.offset),
            static_cast<std::size_t>(rows)};
    }
};

ColbinStore::ColbinStore(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

ColbinStore::~ColbinStore() = default;
ColbinStore::ColbinStore(ColbinStore&&) noexcept = default;
ColbinStore& ColbinStore::operator=(ColbinStore&&) noexcept = default;

std::uint64_t ColbinStore::row_count() const noexcept { return impl_->rows; }

bool ColbinStore::has_column(std::int32_t index) const noexcept {
    return impl_->columns.contains(index);
}

std::span<const std::int32_t> ColbinStore::integer_column(
    std::int32_t index,
    ColbinType expected_type) const {
    if (expected_type != ColbinType::integer && expected_type != ColbinType::date) {
        throw std::invalid_argument("integer_column expects INT or DATE");
    }
    return impl_->span_for<std::int32_t>(impl_->require(index, expected_type));
}

std::span<const float> ColbinStore::float_column(std::int32_t index) const {
    return impl_->span_for<float>(impl_->require(index, ColbinType::floating));
}

std::span<const char> ColbinStore::char_column(
    std::int32_t index,
    ColbinType expected_type,
    std::uint16_t fixed_width) const {
    if (expected_type != ColbinType::char1 && expected_type != ColbinType::char_fixed) {
        throw std::invalid_argument("char_column expects CHAR1 or CHAR_FIXED");
    }
    const auto& column = impl_->require(index, expected_type, fixed_width);
    const auto count = expected_type == ColbinType::char_fixed
        ? static_cast<std::size_t>(impl_->rows) * fixed_width
        : static_cast<std::size_t>(impl_->rows);
    return {
        reinterpret_cast<const char*>(
            static_cast<const char*>(impl_->mapping) + column.offset),
        count};
}

}  // namespace joule::tpch
