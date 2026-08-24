#include "joule/operators/cpu/tpch_q6.hpp"
#include "joule/operators/cpu/tpch_q6_unfused.hpp"
#include "joule/operators/cpu/relational.hpp"
#include "joule/operators/gpu/tpch_q6.hpp"
#include "joule/operators/gpu/tpch_q6_unfused.hpp"
#include "joule/operators/gpu/relational.hpp"
#include "joule/tpch/lineitem.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::size_t rows = 4096;
constexpr std::uint64_t alignment = 16'384;

struct Header {
    std::array<char, 8> magic;
    std::uint32_t version;
    std::uint32_t columns;
    std::uint64_t rows;
    std::uint64_t source_size;
    std::int64_t source_mtime;
    std::uint64_t reserved;
};

struct Descriptor {
    std::int32_t index;
    std::uint8_t type;
    std::uint8_t reserved0;
    std::uint16_t width;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::uint64_t reserved1;
};

static_assert(sizeof(Header) == 48);
static_assert(sizeof(Descriptor) == 32);

template <typename T>
void write_column(std::ofstream& output, std::uint64_t offset, const std::vector<T>& data) {
    output.seekp(static_cast<std::streamoff>(offset));
    output.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!output) {
        throw std::runtime_error("could not write fixture column");
    }
}

void create_fixture(const std::filesystem::path& path) {
    std::vector<float> quantity(rows, 30.0F);
    std::vector<float> price(rows, 1.0F);
    std::vector<float> discount(rows, 0.06F);
    std::vector<std::int32_t> date(rows, 19'940'601);

    quantity[0] = 23.0F;
    price[0] = 100.25F;
    quantity[1] = 24.0F;
    price[1] = 999.99F;
    quantity[2] = 12.0F;
    price[2] = 10.0F;
    discount[2] = 0.07F;
    quantity[3] = 1.0F;
    date[3] = 19'950'101;
    quantity[4] = 1.0F;
    price[4] = 0.01F;
    discount[4] = 0.05F;
    quantity[5] = 1.0F;
    discount[5] = 0.04F;

    constexpr std::uint64_t bytes = rows * sizeof(float);
    constexpr std::array<std::uint64_t, 4> offsets{
        alignment, alignment * 2, alignment * 3, alignment * 4};
    const Header header{
        {'T', 'P', 'C', 'H', 'C', 'B', '0', '1'}, 2, 4, rows, 0, 0, 0};
    const std::array<Descriptor, 4> descriptors{{
        {4, 1, 0, 0, offsets[0], bytes, 0},
        {5, 1, 0, 0, offsets[1], bytes, 0},
        {6, 1, 0, 0, offsets[2], bytes, 0},
        {10, 2, 0, 0, offsets[3], bytes, 0},
    }};

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(
        reinterpret_cast<const char*>(descriptors.data()),
        static_cast<std::streamsize>(sizeof(descriptors)));
    write_column(output, offsets[0], quantity);
    write_column(output, offsets[1], price);
    write_column(output, offsets[2], discount);
    write_column(output, offsets[3], date);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("joule-tpch-q6-" + std::to_string(::getpid()) + ".colbin");
    try {
        create_fixture(path);
        const joule::tpch::LineitemStore store(path);
        const auto input = store.view();
        require(input.row_count() == rows, "colbin row count mismatch");
        const auto reference = joule::operators::cpu::tpch_q6_reference(input, true);
        require(reference.match_count == 3, "reference count mismatch");
        require(reference.revenue_1e4_usd == 67'155, "reference revenue mismatch");
        require(reference.bitmap.front() == 0b10101U, "reference bitmap mismatch");
        const auto materialized =
            joule::operators::cpu::q6_materialize_reference(input);
        require(
            materialized == std::vector<std::uint32_t>({0, 2, 4}),
            "materialize reference mismatch");

        for (const auto thread_count : {2U, 24U, 32U}) {
            joule::operators::cpu::ScanCopyF32 copy(
                input.extended_price, thread_count);
            joule::operators::cpu::Q6FilterMaterialize materialize(
                input, thread_count);
            for (int repetition = 0; repetition < 5; ++repetition) {
                static_cast<void>(copy.execute());
                require(std::equal(
                    copy.output().begin(), copy.output().end(),
                    input.extended_price.begin()),
                    "CPU scan-copy mismatch");

                const auto result = materialize.execute();
                require(
                    result.output_count == materialized.size(),
                    "CPU materialize count mismatch");
                require(std::equal(
                    materialize.output().begin(), materialize.output().end(),
                    materialized.begin()),
                    "CPU materialize output mismatch");
            }
        }

        // Qualifying rows span many logical chunks. Dynamic workers may claim
        // those chunks in any order, but materialization must remain stable.
        {
            constexpr std::size_t stress_rows = 8'193;
            std::vector<float> quantity(stress_rows, 30.0F);
            std::vector<float> discount(stress_rows, 0.01F);
            std::vector<std::int32_t> ship_date(stress_rows, 19'960'101);
            for (std::size_t row = 3; row < stress_rows; row += 97) {
                quantity[row] = 1.0F;
                discount[row] = 0.06F;
                ship_date[row] = 19'940'601;
            }
            joule::tpch::LineitemView stress_input{};
            stress_input.quantity = quantity;
            stress_input.discount = discount;
            stress_input.ship_date_yyyymmdd = ship_date;
            const auto stress_reference =
                joule::operators::cpu::q6_materialize_reference(stress_input);
            joule::operators::cpu::Q6FilterMaterialize materialize(
                stress_input, 32);
            for (int repetition = 0; repetition < 10; ++repetition) {
                const auto result = materialize.execute();
                require(
                    result.output_count == stress_reference.size() &&
                        std::equal(
                            materialize.output().begin(), materialize.output().end(),
                            stress_reference.begin()),
                    "CPU dynamic materialize stability mismatch");
            }
        }

        // Repeated executions verify cursor reset; 24 and 32 workers exercise
        // P-core-sized and full heterogeneous worker-pool configurations.
        for (const auto thread_count : {2U, 24U, 32U}) {
            for (const auto mode : {
                     joule::operators::cpu::TpchQ6Mode::filter_count,
                     joule::operators::cpu::TpchQ6Mode::filter_bitmap,
                     joule::operators::cpu::TpchQ6Mode::revenue}) {
                joule::operators::cpu::TpchQ6Config config;
                config.mode = mode;
                config.thread_count = thread_count;
                joule::operators::cpu::TpchQ6 op(input, config);
                for (int repetition = 0; repetition < 5; ++repetition) {
                    const auto result = op.execute();
                    if (mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap) {
                        require(std::equal(
                            op.bitmap().begin(), op.bitmap().end(),
                            reference.bitmap.begin()),
                            "CPU bitmap mismatch");
                    } else {
                        require(
                            result.match_count == reference.match_count,
                            "CPU count mismatch");
                    }
                    if (mode == joule::operators::cpu::TpchQ6Mode::revenue) {
                        require(
                            result.revenue_1e4_usd == reference.revenue_1e4_usd,
                            "CPU revenue mismatch");
                    }
                }
            }
        }

        for (const auto thread_count : {2U, 24U, 32U}) {
            joule::operators::cpu::TpchQ6UnfusedConfig config;
            config.thread_count = thread_count;
            joule::operators::cpu::TpchQ6Unfused op(input, config);
            for (int repetition = 0; repetition < 5; ++repetition) {
                const auto result = op.execute();
                require(
                    result.match_count == reference.match_count &&
                        result.revenue_1e4_usd == reference.revenue_1e4_usd,
                    "CPU unfused Q6 result mismatch");
                require(
                    std::equal(
                        op.bitmap().begin(), op.bitmap().end(),
                        reference.bitmap.begin()),
                    "CPU unfused Q6 bitmap mismatch");
            }
        }

#if defined(JOULE_DEFAULT_METALLIB_PATH)
        {
            joule::operators::gpu::ScanCopyF32 copy(input.extended_price);
            static_cast<void>(copy.execute());
            require(std::equal(
                copy.output().begin(), copy.output().end(), input.extended_price.begin()),
                "GPU scan-copy mismatch");
        }
        {
            joule::operators::gpu::Q6FilterMaterialize op(
                JOULE_DEFAULT_METALLIB_PATH, input);
            const auto result = op.execute();
            require(result.output_count == materialized.size(), "GPU materialize count mismatch");
            require(std::equal(
                op.output().begin(), op.output().end(), materialized.begin()),
                "GPU materialize output mismatch");
        }
        for (const auto mode : {
                 joule::operators::cpu::TpchQ6Mode::filter_count,
                 joule::operators::cpu::TpchQ6Mode::filter_bitmap,
                 joule::operators::cpu::TpchQ6Mode::revenue}) {
            joule::operators::gpu::TpchQ6Config config;
            config.mode = mode;
            config.threadgroup_width = 32;
            joule::operators::gpu::TpchQ6 op(JOULE_DEFAULT_METALLIB_PATH, input, config);
            const auto result = op.execute();
            if (mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap) {
                require(std::equal(
                    op.bitmap().begin(), op.bitmap().end(), reference.bitmap.begin()),
                    "GPU bitmap mismatch");
            } else {
                require(result.match_count == reference.match_count, "GPU count mismatch");
            }
            if (mode == joule::operators::cpu::TpchQ6Mode::revenue) {
                require(
                    result.revenue_1e4_usd == reference.revenue_1e4_usd,
                    "GPU revenue mismatch");
            } else if (mode == joule::operators::cpu::TpchQ6Mode::filter_count) {
                require(
                    result.revenue_1e4_usd == 0,
                    "GPU filter-count produced an unused revenue result");
            }
        }
        {
            joule::operators::gpu::TpchQ6Unfused op(
                JOULE_DEFAULT_METALLIB_PATH, input);
            const auto result = op.execute();
            require(
                result.match_count == reference.match_count &&
                    result.revenue_1e4_usd == reference.revenue_1e4_usd,
                "GPU unfused Q6 result mismatch");
        }
#endif
        std::filesystem::remove(path);
        std::cout << "TPC-H Q6 colbin CPU/GPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
