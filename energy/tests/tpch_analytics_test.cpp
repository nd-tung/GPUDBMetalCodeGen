#include "joule/operators/cpu/aggregates.hpp"
#include "joule/operators/cpu/filter_project.hpp"
#include "joule/operators/cpu/hash_join.hpp"
#include "joule/operators/cpu/topk.hpp"
#include "joule/operators/cpu/tpch_q1.hpp"
#include "joule/operators/cpu/tpch_q14.hpp"
#include "joule/operators/gpu/aggregates.hpp"
#include "joule/operators/gpu/filter_project.hpp"
#include "joule/operators/gpu/hash_join.hpp"
#include "joule/operators/gpu/topk.hpp"
#include "joule/operators/gpu/tpch_q1.hpp"
#include "joule/operators/gpu/tpch_q14.hpp"
#include "joule/tpch/lineitem.hpp"
#include "joule/tpch/orders.hpp"
#include "joule/tpch/part.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

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

struct Column {
    std::int32_t index;
    std::uint8_t type;
    std::uint16_t width;
    std::vector<char> bytes;
};

template <typename T>
[[nodiscard]] Column column(
    std::int32_t index,
    std::uint8_t type,
    const std::vector<T>& values,
    std::uint16_t width = 0) {
    Column result{index, type, width, {}};
    result.bytes.resize(values.size() * sizeof(T));
    std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}

void write_colbin(
    const std::filesystem::path& path,
    std::uint64_t rows,
    const std::vector<Column>& columns) {
    const Header header{
        {'T', 'P', 'C', 'H', 'C', 'B', '0', '1'}, 2,
        static_cast<std::uint32_t>(columns.size()), rows, 0, 0, 0};
    std::vector<Descriptor> descriptors;
    descriptors.reserve(columns.size());
    for (std::size_t index = 0; index < columns.size(); ++index) {
        descriptors.push_back({
            columns[index].index, columns[index].type, 0, columns[index].width,
            alignment * (index + 1), columns[index].bytes.size(), 0});
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(
        reinterpret_cast<const char*>(descriptors.data()),
        static_cast<std::streamsize>(descriptors.size() * sizeof(Descriptor)));
    for (std::size_t index = 0; index < columns.size(); ++index) {
        output.seekp(static_cast<std::streamoff>(descriptors[index].offset));
        output.write(columns[index].bytes.data(),
                     static_cast<std::streamsize>(columns[index].bytes.size()));
    }
    if (!output) throw std::runtime_error("could not write colbin fixture");
}

void create_lineitem(const std::filesystem::path& path) {
    constexpr std::size_t row_count = 4096;
    std::vector<std::int32_t> order_key(row_count);
    std::vector<std::int32_t> part_key(row_count);
    std::vector<float> quantity(row_count);
    std::vector<float> price(row_count);
    std::vector<float> discount(row_count);
    std::vector<float> tax(row_count);
    std::vector<char> return_flag(row_count);
    std::vector<char> line_status(row_count);
    std::vector<std::int32_t> date(row_count, 19'960'101);
    constexpr std::array<char, 3> flags{'A', 'N', 'R'};
    constexpr std::array<char, 2> statuses{'F', 'O'};
    for (std::size_t row = 0; row < row_count; ++row) {
        order_key[row] = static_cast<std::int32_t>(row + 1);
        // Missing build keys are spread across chunks so stable hash
        // materialization must compact gaps without reordering matches.
        part_key[row] = row % 11 == 0
            ? 7
            : static_cast<std::int32_t>(row % 4 + 1);
        quantity[row] = static_cast<float>(row % 50) + 0.25F;
        price[row] = static_cast<float>(row % 1000) + 10.25F;
        discount[row] = static_cast<float>(row % 11) / 100.0F;
        tax[row] = static_cast<float>(row % 9) / 100.0F;
        return_flag[row] = flags[row % flags.size()];
        line_status[row] = statuses[row % statuses.size()];
    }
    date[0] = date[1] = date[2] = date[3] = 19'950'915;
    date[5] = date[6] = date[7] = 19'940'915;
    for (std::size_t row = 64; row < row_count; row += 127) {
        quantity[row] = 1.0F;
        discount[row] = 0.06F;
        date[row] = 19'940'915;
    }
    date[4] = 19'981'201;

    write_colbin(path, row_count, {
        column(0, 0, order_key), column(1, 0, part_key),
        column(4, 1, quantity), column(5, 1, price), column(6, 1, discount),
        column(7, 1, tax), column(8, 3, return_flag), column(9, 3, line_status),
        column(10, 2, date)});
}

void create_part(const std::filesystem::path& path) {
    // 4096 rows force the GPU hash-build statistics through multiple
    // threadgroup partials. Keys form four collision clusters for the exact
    // 8192-slot table used by the build, exercising concurrent CAS probing.
    constexpr std::size_t row_count = 4096;
    constexpr std::int32_t hash_capacity = 8192;
    std::vector<std::int32_t> keys(row_count);
    std::vector<char> types(row_count * 25, ' ');
    constexpr std::string_view promo = "PROMO BURNISHED";
    constexpr std::string_view standard = "STANDARD BRASS";
    for (std::size_t row = 0; row < row_count; ++row) {
        keys[row] = static_cast<std::int32_t>(row % 4 + 1) +
            static_cast<std::int32_t>(row / 4) * hash_capacity;
        const auto text = row % 2 == 0 ? promo : standard;
        std::memcpy(types.data() + row * 25, text.data(), text.size());
    }
    write_colbin(
        path, row_count, {column(0, 0, keys), column(4, 4, types, 25)});
}

void create_orders(const std::filesystem::path& path) {
    constexpr std::size_t row_count = 257;
    std::vector<std::int32_t> keys(row_count);
    std::vector<float> prices(row_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        keys[row] = static_cast<std::int32_t>(1000 - row);
        prices[row] = static_cast<float>(row % 31) + 0.25F;
    }
    prices[7] = prices[17] = prices[27] = 999.99F;
    write_colbin(path, row_count, {column(0, 0, keys), column(3, 1, prices)});
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("joule-tpch-analytics-" + std::to_string(::getpid()));
    try {
        std::filesystem::create_directories(directory);
        create_lineitem(directory / "lineitem.colbin");
        create_part(directory / "part.colbin");
        create_orders(directory / "orders.colbin");
        const joule::tpch::LineitemStore lineitem_store(directory / "lineitem.colbin");
        const joule::tpch::PartStore part_store(directory / "part.colbin");
        const joule::tpch::OrdersStore orders_store(directory / "orders.colbin");
        const auto lineitem = lineitem_store.view();
        const auto part = part_store.view();
        const auto orders = orders_store.view();

        for (const auto mode : {
                 joule::operators::cpu::PriceAggregateMode::sum,
                 joule::operators::cpu::PriceAggregateMode::minmax,
                 joule::operators::cpu::PriceAggregateMode::stats}) {
            const auto reference =
                joule::operators::cpu::price_aggregate_reference(
                    lineitem.extended_price, mode);
            for (const auto thread_count : {3U, 24U, 32U}) {
                joule::operators::cpu::PriceAggregate aggregate_cpu(
                    lineitem.extended_price, mode, thread_count);
                for (int repetition = 0; repetition < 5; ++repetition) {
                    const auto result = aggregate_cpu.execute();
                    require(
                        result.count == reference.count &&
                            result.sum_price_cents == reference.sum_price_cents &&
                            result.min_price_cents == reference.min_price_cents &&
                            result.max_price_cents == reference.max_price_cents,
                        "CPU price aggregate result mismatch");
                }
            }
#if defined(JOULE_DEFAULT_METALLIB_PATH)
            for (const auto reduction : {
                     joule::operators::gpu::PriceAggregateReduction::threadgroup_tree,
                     joule::operators::gpu::PriceAggregateReduction::simdgroup}) {
                joule::operators::gpu::PriceAggregate aggregate_gpu(
                    JOULE_DEFAULT_METALLIB_PATH, lineitem.extended_price, mode,
                    256, reduction);
                const auto gpu_result = aggregate_gpu.execute();
                require(
                    gpu_result.count == reference.count &&
                        gpu_result.sum_price_cents == reference.sum_price_cents &&
                        gpu_result.min_price_cents == reference.min_price_cents &&
                        gpu_result.max_price_cents == reference.max_price_cents,
                    "GPU price aggregate result mismatch");
            }
#endif
        }

#if defined(JOULE_DEFAULT_METALLIB_PATH)
        // Width 32 and 70,003 rows require two recursive reduce passes,
        // covering every mode-specialized first/reduce pipeline.
        std::vector<float> recursive_prices(70'003);
        for (std::size_t index = 0; index < recursive_prices.size(); ++index) {
            recursive_prices[index] =
                static_cast<float>(static_cast<int>(index % 257) - 128);
        }
        for (const auto mode : {
                 joule::operators::cpu::PriceAggregateMode::sum,
                 joule::operators::cpu::PriceAggregateMode::minmax,
                 joule::operators::cpu::PriceAggregateMode::stats}) {
            const auto reference =
                joule::operators::cpu::price_aggregate_reference(
                    recursive_prices, mode);
            for (const auto reduction : {
                     joule::operators::gpu::PriceAggregateReduction::threadgroup_tree,
                     joule::operators::gpu::PriceAggregateReduction::simdgroup}) {
                joule::operators::gpu::PriceAggregate aggregate_gpu(
                    JOULE_DEFAULT_METALLIB_PATH, recursive_prices, mode,
                    32, reduction);
                const auto result = aggregate_gpu.execute();
                require(
                    result.count == reference.count &&
                        result.sum_price_cents == reference.sum_price_cents &&
                        result.min_price_cents == reference.min_price_cents &&
                        result.max_price_cents == reference.max_price_cents,
                    "GPU recursive price aggregate result mismatch");
            }
        }
#endif

        constexpr std::uint32_t group_cardinality = 2;
        const auto group_reference =
            joule::operators::cpu::part_key_group_count_reference(
                lineitem.part_key, group_cardinality);
        for (const auto thread_count : {3U, 24U, 32U}) {
            joule::operators::cpu::PartKeyGroupCount group_cpu(
                lineitem.part_key, group_cardinality, thread_count);
            for (int repetition = 0; repetition < 5; ++repetition) {
                static_cast<void>(group_cpu.execute());
                require(
                    std::equal(
                        group_cpu.output().begin(), group_cpu.output().end(),
                        group_reference.begin()),
                    "CPU group-by count result mismatch");
            }
        }
#if defined(JOULE_DEFAULT_METALLIB_PATH)
        for (const auto strategy : {
                 joule::operators::gpu::GroupByCountStrategy::global_atomic,
                 joule::operators::gpu::GroupByCountStrategy::bounded_threadgroup}) {
            joule::operators::gpu::PartKeyGroupCount group_gpu(
                JOULE_DEFAULT_METALLIB_PATH, lineitem.part_key,
                group_cardinality, 256, strategy);
            static_cast<void>(group_gpu.execute());
            require(
                std::equal(
                    group_gpu.output().begin(), group_gpu.output().end(),
                    group_reference.begin()),
                "GPU group-by count result mismatch");
        }
#endif

        const auto project_reference =
            joule::operators::cpu::q6_filter_project_reference(lineitem);
        require(!project_reference.empty(), "filter-project fixture matched no rows");
        for (const auto thread_count : {3U, 24U, 32U}) {
            joule::operators::cpu::Q6FilterProject project_cpu(
                lineitem, thread_count);
            for (int repetition = 0; repetition < 5; ++repetition) {
                static_cast<void>(project_cpu.execute());
                require(
                    project_cpu.output().size() == project_reference.size() &&
                        std::equal(
                            project_cpu.output().begin(), project_cpu.output().end(),
                            project_reference.begin()),
                    "CPU filter-project result mismatch");
            }
        }
#if defined(JOULE_DEFAULT_METALLIB_PATH)
        joule::operators::gpu::Q6FilterProject project_gpu(
            JOULE_DEFAULT_METALLIB_PATH, lineitem);
        static_cast<void>(project_gpu.execute());
        require(
            project_gpu.output().size() == project_reference.size(),
            "GPU filter-project output count mismatch");
        for (std::size_t index = 0; index < project_reference.size(); ++index) {
            const auto& got = project_gpu.output()[index];
            const auto& want = project_reference[index];
            require(
                got.row_id == want.row_id &&
                    got.part_key == want.part_key &&
                    got.revenue_1e4_usd == want.revenue_1e4_usd,
                "GPU filter-project result mismatch");
        }
#endif

        const auto hash_build_reference =
            joule::operators::cpu::part_hash_build_reference(part);
        joule::operators::cpu::PartHashBuild hash_build_cpu(part, 3);
        const auto hash_build_cpu_run = hash_build_cpu.execute();
        require(
            hash_build_cpu_run.entry_count ==
                    hash_build_reference.entry_count &&
                hash_build_cpu_run.promo_entry_count ==
                    hash_build_reference.promo_entry_count &&
                hash_build_cpu.verify() == hash_build_reference,
            "CPU hash-build result mismatch");
        const auto hash_build_cpu_repeat = hash_build_cpu.execute();
        require(
            hash_build_cpu_repeat.entry_count ==
                    hash_build_reference.entry_count &&
                hash_build_cpu_repeat.promo_entry_count ==
                    hash_build_reference.promo_entry_count &&
                hash_build_cpu.verify() == hash_build_reference,
            "CPU repeated hash-build result mismatch");

        const auto hash_count_reference =
            joule::operators::cpu::part_hash_probe_count_reference(
                lineitem, part);
        joule::operators::cpu::PartHashProbeCount hash_count_cpu(
            lineitem, part, 3);
        const auto hash_count_cpu_result = hash_count_cpu.execute();
        require(
            hash_count_cpu_result.match_count ==
                    hash_count_reference.match_count &&
                hash_count_cpu_result.promo_match_count ==
                    hash_count_reference.promo_match_count,
            "CPU hash-probe-count result mismatch");

        const auto hash_materialize_reference =
            joule::operators::cpu::part_hash_probe_materialize_reference(
                lineitem, part);
        joule::operators::cpu::PartHashProbeMaterialize
            hash_materialize_cpu(lineitem, part, 3);
        static_cast<void>(hash_materialize_cpu.execute());
        require(
            std::equal(
                hash_materialize_cpu.output().begin(),
                hash_materialize_cpu.output().end(),
                hash_materialize_reference.begin()),
            "CPU hash-probe-materialize result mismatch");

        // Repeated high worker counts exercise cursor reset and stable
        // chunk-order compaction when P- and E-core workers finish unevenly.
        for (const auto thread_count : {24U, 32U}) {
            joule::operators::cpu::PartHashBuild build_cpu(part, thread_count);
            joule::operators::cpu::PartHashProbeCount count_cpu(
                lineitem, part, thread_count);
            joule::operators::cpu::PartHashProbeMaterialize materialize_cpu(
                lineitem, part, thread_count);
            for (int repetition = 0; repetition < 5; ++repetition) {
                const auto build_result = build_cpu.execute();
                require(
                    build_result.entry_count == hash_build_reference.entry_count &&
                        build_result.promo_entry_count ==
                            hash_build_reference.promo_entry_count &&
                        build_cpu.verify() == hash_build_reference,
                    "CPU heterogeneous hash-build result mismatch");

                const auto count_result = count_cpu.execute();
                require(
                    count_result.match_count == hash_count_reference.match_count &&
                        count_result.promo_match_count ==
                            hash_count_reference.promo_match_count,
                    "CPU heterogeneous hash-probe-count result mismatch");

                static_cast<void>(materialize_cpu.execute());
                require(
                    materialize_cpu.output().size() ==
                            hash_materialize_reference.size() &&
                        std::equal(
                            materialize_cpu.output().begin(),
                            materialize_cpu.output().end(),
                            hash_materialize_reference.begin()),
                    "CPU heterogeneous hash-probe-materialize result mismatch");
            }
        }

#if defined(JOULE_DEFAULT_METALLIB_PATH)
        joule::operators::gpu::PartHashBuild hash_build_gpu(
            JOULE_DEFAULT_METALLIB_PATH, part);
        const auto hash_build_gpu_run = hash_build_gpu.execute();
        require(
            hash_build_gpu_run.entry_count ==
                    hash_build_reference.entry_count &&
                hash_build_gpu_run.promo_entry_count ==
                    hash_build_reference.promo_entry_count &&
                hash_build_gpu.verify() == hash_build_reference,
            "GPU hash-build result mismatch");
        const auto hash_build_gpu_repeat = hash_build_gpu.execute();
        require(
            hash_build_gpu_repeat.entry_count ==
                    hash_build_reference.entry_count &&
                hash_build_gpu_repeat.promo_entry_count ==
                    hash_build_reference.promo_entry_count &&
                hash_build_gpu.verify() == hash_build_reference,
            "GPU repeated hash-build result mismatch");

        joule::operators::gpu::PartHashProbeCount hash_count_gpu(
            JOULE_DEFAULT_METALLIB_PATH, lineitem, part);
        const auto hash_count_gpu_result = hash_count_gpu.execute();
        require(
            hash_count_gpu_result.match_count ==
                    hash_count_reference.match_count &&
                hash_count_gpu_result.promo_match_count ==
                    hash_count_reference.promo_match_count,
            "GPU hash-probe-count result mismatch");

        joule::operators::gpu::PartHashProbeMaterialize
            hash_materialize_gpu(
                JOULE_DEFAULT_METALLIB_PATH, lineitem, part);
        static_cast<void>(hash_materialize_gpu.execute());
        require(
            std::equal(
                hash_materialize_gpu.output().begin(),
                hash_materialize_gpu.output().end(),
                hash_materialize_reference.begin()),
            "GPU hash-probe-materialize result mismatch");
#endif

        const auto q1_reference = joule::operators::cpu::tpch_q1_reference(lineitem);
        const auto q14_reference = joule::operators::cpu::tpch_q14_reference(lineitem, part);
        const auto topk_reference = joule::operators::cpu::top10_reference(orders);
        for (const auto thread_count : {3U, 24U, 32U}) {
            joule::operators::cpu::TpchQ1GroupBy q1_cpu(lineitem, thread_count);
            joule::operators::cpu::TpchQ14HashJoin q14_cpu(
                lineitem, part, thread_count);
            joule::operators::cpu::OrdersTopK topk_cpu(orders, thread_count);
            for (int repetition = 0; repetition < 5; ++repetition) {
                require(
                    q1_cpu.execute().groups == q1_reference,
                    "CPU Q1 result mismatch");
                const auto q14_cpu_result = q14_cpu.execute();
                require(
                    q14_cpu_result.promo_revenue_1e4_usd ==
                            q14_reference.promo_revenue_1e4_usd &&
                        q14_cpu_result.total_revenue_1e4_usd ==
                            q14_reference.total_revenue_1e4_usd,
                    "CPU Q14 result mismatch");
                require(
                    topk_cpu.execute().rows == topk_reference,
                    "CPU top-k result mismatch");
            }
        }

#if defined(JOULE_DEFAULT_METALLIB_PATH)
        joule::operators::gpu::TpchQ1GroupBy q1_gpu(JOULE_DEFAULT_METALLIB_PATH, lineitem);
        // Q1 reuses one threadgroup scratch array for all six keys. Repeated
        // dispatches make a missing reuse barrier observable as an intermittent
        // cross-key reduction race.
        for (int repetition = 0; repetition < 100; ++repetition) {
            require(
                q1_gpu.execute().groups == q1_reference,
                "GPU repeated Q1 result mismatch");
        }

        joule::operators::gpu::TpchQ14HashJoin q14_gpu(
            JOULE_DEFAULT_METALLIB_PATH, lineitem, part);
        const auto q14_gpu_result = q14_gpu.execute();
        require(
            q14_gpu_result.promo_revenue_1e4_usd == q14_reference.promo_revenue_1e4_usd &&
            q14_gpu_result.total_revenue_1e4_usd == q14_reference.total_revenue_1e4_usd,
            "GPU Q14 result mismatch");

        joule::operators::gpu::OrdersTopK topk_gpu(JOULE_DEFAULT_METALLIB_PATH, orders);
        require(topk_gpu.execute().rows == topk_reference, "GPU top-k result mismatch");
#endif
        std::filesystem::remove_all(directory);
        std::cout << "TPC-H analytics colbin CPU/GPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
