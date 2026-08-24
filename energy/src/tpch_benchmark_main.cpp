#include "joule/measurement_handshake.hpp"
#include "joule/operators/cpu/aggregates.hpp"
#include "joule/operators/cpu/filter_project.hpp"
#include "joule/operators/cpu/hash_join.hpp"
#include "joule/operators/cpu/relational.hpp"
#include "joule/operators/cpu/topk.hpp"
#include "joule/operators/cpu/tpch_q1.hpp"
#include "joule/operators/cpu/tpch_q14.hpp"
#include "joule/operators/cpu/tpch_q6.hpp"
#include "joule/operators/cpu/tpch_q6_unfused.hpp"
#include "joule/operators/gpu/aggregates.hpp"
#include "joule/operators/gpu/filter_project.hpp"
#include "joule/operators/gpu/hash_join.hpp"
#include "joule/operators/gpu/relational.hpp"
#include "joule/operators/gpu/topk.hpp"
#include "joule/operators/gpu/tpch_q1.hpp"
#include "joule/operators/gpu/tpch_q14.hpp"
#include "joule/operators/gpu/tpch_q6.hpp"
#include "joule/operators/gpu/tpch_q6_unfused.hpp"
#include "joule/tpch/lineitem.hpp"
#include "joule/tpch/orders.hpp"
#include "joule/tpch/part.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

struct Options {
    std::filesystem::path data;
    std::filesystem::path part;
    std::filesystem::path orders;
    std::string operator_name{"q6-revenue"};
    std::string backend{"cpu"};
    std::uint64_t duration_ms{1'800'000};
    std::uint32_t warmup_iterations{5};
    std::uint32_t cpu_threads{0};
    std::uint32_t threadgroup_width{256};
    std::uint32_t group_cardinality{1U << 20};
    std::string gpu_aggregate_reduction{"simdgroup"};
    std::string gpu_groupby_strategy{"global-atomic"};
    std::filesystem::path result_json;
#if defined(JOULE_DEFAULT_METALLIB_PATH)
    std::filesystem::path metal_library{JOULE_DEFAULT_METALLIB_PATH};
#else
    std::filesystem::path metal_library{"build/metal/joule.metallib"};
#endif
};

struct Summary {
    std::uint64_t rows{};
    std::uint64_t iterations{};
    std::uint64_t output_count{};
    std::uint64_t result_checksum{};
    std::int64_t reference_value_0{};
    std::int64_t reference_value_1{};
    std::int64_t reference_value_2{};
    std::int64_t reference_value_3{};
    std::int64_t result_value_0{};
    std::int64_t result_value_1{};
    std::int64_t result_value_2{};
    std::int64_t result_value_3{};
    double wall_time_ms{};
    double accumulated_compute_ms{};
    long double logical_bytes_per_iteration{};
    std::uint32_t cpu_threads{};
    std::uint32_t execution_width{};
    std::uint32_t max_threads{};
    std::string device;
    std::string semantics;
    std::string data_files;
    std::string implementation_variant{"default"};
};

[[nodiscard]] std::uint64_t parse_uint64(
    std::string_view value,
    std::string_view option) {
    std::size_t parsed = 0;
    const auto number = std::stoull(std::string(value), &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return number;
}

void print_help() {
    std::cout
        << "Usage: joule-tpch-benchmark [options]\n\n"
        << "Options:\n"
        << "  --data PATH          SF directory or lineitem.colbin (required)\n"
        << "  --part PATH          part.colbin (defaults beside lineitem)\n"
        << "  --orders PATH        orders.colbin (defaults beside lineitem)\n"
        << "  --operator NAME      scan-copy|filter-count|filter-bitmap|"
           "filter-materialize|filter-project|hash-build|hash-probe-count|"
           "hash-probe-materialize|aggregate-sum|aggregate-minmax|aggregate-stats|"
           "groupby-part-count|q6-revenue|q6-revenue-unfused|q1-groupby|"
           "q14-join|orders-topk\n"
        << "  --backend cpu|gpu    execution backend (default: cpu)\n"
        << "  --duration-ms N      timed loop duration (default: 1800000)\n"
        << "  --warmup-iterations N  operators before timing (default: 5)\n"
        << "  --cpu-threads N      0 selects all logical CPUs (default: 0)\n"
        << "  --threadgroup-width N  Metal threads per group (default: 256)\n"
        << "  --group-cardinality N  power-of-two group buckets (default: 1048576)\n"
        << "  --gpu-aggregate-reduction NAME  simdgroup|threadgroup-tree "
           "(default: simdgroup)\n"
        << "  --gpu-groupby-strategy NAME  global-atomic|bounded-threadgroup "
           "(default: global-atomic)\n"
        << "  --result-json PATH  write the benchmark summary to a dedicated JSON file\n"
        << "  --metallib PATH      compiled Metal library path\n";
}

[[nodiscard]] Options parse_arguments(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "-h" || argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(argument));
        }
        const std::string_view value(argv[++index]);
        if (argument == "--data") options.data = value;
        else if (argument == "--part") options.part = value;
        else if (argument == "--orders") options.orders = value;
        else if (argument == "--operator") options.operator_name = value;
        else if (argument == "--backend") options.backend = value;
        else if (argument == "--duration-ms") options.duration_ms = parse_uint64(value, argument);
        else if (argument == "--warmup-iterations") {
            const auto parsed = parse_uint64(value, argument);
            if (parsed > UINT32_MAX) throw std::invalid_argument("warmup count is too large");
            options.warmup_iterations = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--cpu-threads") {
            const auto parsed = parse_uint64(value, argument);
            if (parsed > UINT32_MAX) throw std::invalid_argument("CPU thread count is too large");
            options.cpu_threads = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--threadgroup-width") {
            const auto parsed = parse_uint64(value, argument);
            if (parsed > UINT32_MAX) throw std::invalid_argument("threadgroup width is too large");
            options.threadgroup_width = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--group-cardinality") {
            const auto parsed = parse_uint64(value, argument);
            if (parsed == 0 || parsed > UINT32_MAX ||
                !std::has_single_bit(static_cast<std::uint32_t>(parsed))) {
                throw std::invalid_argument(
                    "group cardinality must be a power of two fitting uint32");
            }
            options.group_cardinality = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--gpu-aggregate-reduction") {
            options.gpu_aggregate_reduction = value;
        } else if (argument == "--gpu-groupby-strategy") {
            options.gpu_groupby_strategy = value;
        } else if (argument == "--result-json") {
            options.result_json = value;
        } else if (argument == "--metallib") options.metal_library = value;
        else throw std::invalid_argument("unknown option: " + std::string(argument));
    }
    if (options.data.empty()) throw std::invalid_argument("--data is required");
    constexpr std::array<std::string_view, 17> operators{
        "scan-copy", "filter-count", "filter-bitmap", "filter-materialize",
        "filter-project", "hash-build", "hash-probe-count",
        "hash-probe-materialize",
        "aggregate-sum", "aggregate-minmax", "aggregate-stats",
        "groupby-part-count", "q6-revenue", "q6-revenue-unfused",
        "q1-groupby", "q14-join", "orders-topk"};
    if (std::find(operators.begin(), operators.end(), options.operator_name) == operators.end()) {
        throw std::invalid_argument("unsupported --operator; use --help for the list");
    }
    if (options.backend != "cpu" && options.backend != "gpu") {
        throw std::invalid_argument("--backend must be cpu or gpu");
    }
    if (options.gpu_aggregate_reduction != "simdgroup" &&
        options.gpu_aggregate_reduction != "threadgroup-tree") {
        throw std::invalid_argument(
            "--gpu-aggregate-reduction must be simdgroup or threadgroup-tree");
    }
    if (options.gpu_groupby_strategy != "global-atomic" &&
        options.gpu_groupby_strategy != "bounded-threadgroup") {
        throw std::invalid_argument(
            "--gpu-groupby-strategy must be global-atomic or bounded-threadgroup");
    }
    if (options.operator_name == "groupby-part-count" &&
        options.backend == "gpu" &&
        options.gpu_groupby_strategy == "bounded-threadgroup" &&
        options.group_cardinality > 4096) {
        throw std::invalid_argument(
            "bounded-threadgroup supports --group-cardinality up to 4096");
    }
    if (options.threadgroup_width < 32 || options.threadgroup_width > 512 ||
        !std::has_single_bit(options.threadgroup_width)) {
        throw std::invalid_argument(
            "--threadgroup-width must be a power of two between 32 and 512");
    }
    return options;
}

[[nodiscard]] std::filesystem::path lineitem_path(const Options& options) {
    return std::filesystem::is_directory(options.data)
        ? options.data / "lineitem.colbin" : options.data;
}

[[nodiscard]] std::filesystem::path sibling_path(
    const Options& options,
    const std::filesystem::path& explicit_path,
    std::string_view filename) {
    if (!explicit_path.empty()) return explicit_path;
    const auto lineitem = lineitem_path(options);
    return lineitem.parent_path() / filename;
}

template <typename T>
[[nodiscard]] std::uint64_t checksum(std::span<const T> values) {
    const auto bytes = std::as_bytes(values);
    std::uint64_t result = 14'695'981'039'346'656'037ULL;
    for (const auto byte : bytes) {
        result ^= std::to_integer<std::uint8_t>(byte);
        result *= 1'099'511'628'211ULL;
    }
    return result;
}

[[nodiscard]] std::string json_escape(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char value : text) {
        if (value == '\\' || value == '"') result.push_back('\\');
        result.push_back(value);
    }
    return result;
}

template <typename Execute, typename Validate, typename ComputeTime>
auto run_loop(
    const Options& options,
    joule::MeasurementHandshake& handshake,
    Execute&& execute,
    Validate&& validate,
    ComputeTime&& compute_time,
    Summary& summary) {
    using Result = std::remove_cvref_t<std::invoke_result_t<Execute>>;
    for (std::uint32_t warmup = 0; warmup < options.warmup_iterations; ++warmup) {
        const auto result = execute();
        validate(result, "warm-up");
    }
    handshake.ready_and_wait();
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(options.duration_ms);
    std::optional<Result> last;
    do {
        last = execute();
        summary.accumulated_compute_ms += compute_time(*last);
        ++summary.iterations;
    } while (std::chrono::steady_clock::now() < deadline);
    const auto end = std::chrono::steady_clock::now();
    handshake.complete();
    summary.wall_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    validate(*last, "final");
    return *last;
}

[[nodiscard]] joule::operators::cpu::TpchQ6Mode q6_mode(std::string_view name) {
    if (name == "filter-count") return joule::operators::cpu::TpchQ6Mode::filter_count;
    if (name == "filter-bitmap") return joule::operators::cpu::TpchQ6Mode::filter_bitmap;
    return joule::operators::cpu::TpchQ6Mode::revenue;
}

void validate_q1(
    const joule::operators::cpu::TpchQ1Groups& actual,
    const joule::operators::cpu::TpchQ1Groups& expected,
    std::string_view phase,
    std::string_view backend) {
    for (std::size_t group = 0; group < expected.size(); ++group) {
        if (actual[group] == expected[group]) continue;
        const auto& got = actual[group];
        const auto& want = expected[group];
        throw std::runtime_error(
            std::string(phase) + " " + std::string(backend) + " Q1 mismatch in group " +
            std::to_string(group) + ": actual=[" + std::to_string(got.count) + "," +
            std::to_string(got.sum_quantity_1e2) + "," +
            std::to_string(got.sum_base_price_1e2) + "," +
            std::to_string(got.sum_discount_price_1e4_usd) + "," +
            std::to_string(got.sum_charge_1e6_usd) + "," +
            std::to_string(got.sum_discount_1e2) + "] expected=[" +
            std::to_string(want.count) + "," +
            std::to_string(want.sum_quantity_1e2) + "," +
            std::to_string(want.sum_base_price_1e2) + "," +
            std::to_string(want.sum_discount_price_1e4_usd) + "," +
            std::to_string(want.sum_charge_1e6_usd) + "," +
            std::to_string(want.sum_discount_1e2) + "]");
    }
}

void run_lineitem_operator(
    const Options& options,
    joule::MeasurementHandshake& handshake,
    Summary& summary) {
    const auto path = lineitem_path(options);
    const joule::tpch::LineitemStore store(path);
    const auto input = store.view();
    summary.rows = input.row_count();
    summary.data_files = std::filesystem::absolute(path).string();

    if (options.operator_name == "scan-copy") {
        summary.semantics = "materialize l_extendedprice without transformation";
        summary.output_count = input.row_count();
        summary.logical_bytes_per_iteration = input.row_count() * 8.0L;
        if (options.backend == "cpu") {
            joule::operators::cpu::ScanCopyF32 op(input.extended_price, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(options, handshake,
                [&] { return op.execute(); },
                [&](const auto&, std::string_view phase) {
                    if (!std::equal(op.output().begin(), op.output().end(), input.extended_price.begin())) {
                        throw std::runtime_error(std::string(phase) + " CPU scan-copy mismatch");
                    }
                }, [](const auto& result) { return result.compute_time_ms; }, summary);
            summary.output_count = final.output_count;
            summary.result_checksum = checksum<float>(op.output());
        } else {
            joule::operators::gpu::ScanCopyF32 op(input.extended_price);
            summary.device = op.device_name();
            const auto final = run_loop(options, handshake,
                [&] { return op.execute(); },
                [&](const auto&, std::string_view phase) {
                    if (!std::equal(op.output().begin(), op.output().end(), input.extended_price.begin())) {
                        throw std::runtime_error(std::string(phase) + " GPU scan-copy mismatch");
                    }
                }, [](const auto& result) { return result.gpu_time_ms; }, summary);
            summary.output_count = final.output_count;
            summary.result_checksum = checksum<float>(op.output());
        }
        return;
    }

    if (options.operator_name == "filter-project") {
        const auto reference =
            joule::operators::cpu::q6_filter_project_reference(input);
        std::int64_t reference_revenue = 0;
        for (const auto& record : reference) {
            reference_revenue += record.revenue_1e4_usd;
        }
        summary.semantics =
            "TPC-H Q6 predicate plus stable projection of row-id, part-key, and revenue";
        summary.output_count = reference.size();
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.size());
        summary.reference_value_1 = reference_revenue;
        summary.logical_bytes_per_iteration =
            input.row_count() * 20.0L +
            reference.size() *
                sizeof(joule::operators::cpu::FilterProjectRecord);
        const auto validate_records =
            [&](const auto& actual, std::string_view phase,
                std::string_view backend) {
                if (actual.size() != reference.size()) {
                    throw std::runtime_error(
                        std::string(phase) + " " + std::string(backend) +
                        " filter-project output count mismatch");
                }
                for (std::size_t index = 0; index < reference.size(); ++index) {
                    const auto& got = actual[index];
                    const auto& want = reference[index];
                    if (got.row_id != want.row_id ||
                        got.part_key != want.part_key ||
                        got.revenue_1e4_usd != want.revenue_1e4_usd) {
                        throw std::runtime_error(
                            std::string(phase) + " " +
                            std::string(backend) +
                            " filter-project record mismatch");
                    }
                }
            };
        if (options.backend == "cpu") {
            summary.implementation_variant = "cpu-two-pass-stable";
            joule::operators::cpu::Q6FilterProject op(
                input, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size()) {
                        throw std::runtime_error(
                            std::string(phase) +
                            " CPU filter-project count mismatch");
                    }
                    validate_records(op.output(), phase, "CPU");
                },
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.output_count);
            summary.result_value_1 = reference_revenue;
            summary.result_checksum =
                checksum<joule::operators::cpu::FilterProjectRecord>(
                    op.output());
        } else {
            summary.implementation_variant =
                "gpu-bitmap-prefix-scatter";
            joule::operators::gpu::Q6FilterProject op(
                options.metal_library, input, options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size()) {
                        throw std::runtime_error(
                            std::string(phase) +
                            " GPU filter-project count mismatch");
                    }
                    validate_records(op.output(), phase, "GPU");
                },
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.output_count);
            summary.result_value_1 = reference_revenue;
            summary.result_checksum =
                checksum<joule::operators::gpu::FilterProjectRecord>(
                    op.output());
        }
        return;
    }

    if (options.operator_name == "hash-build") {
        const auto part_path =
            sibling_path(options, options.part, "part.colbin");
        const joule::tpch::PartStore part_store(part_path);
        const auto part = part_store.view();
        const auto reference =
            joule::operators::cpu::part_hash_build_reference(part);
        summary.rows = part.row_count();
        summary.data_files = std::filesystem::absolute(part_path).string();
        summary.semantics =
            "clear and build a unique-key open-addressed part hash table";
        summary.output_count = reference.entry_count;
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.entry_count);
        summary.reference_value_1 =
            static_cast<std::int64_t>(reference.promo_entry_count);
        summary.reference_value_2 =
            static_cast<std::int64_t>(reference.key_sum);
        summary.logical_bytes_per_iteration =
            part.row_count() * 37.0L;
        const auto validate_run =
            [&](const auto& result, std::string_view phase) {
                if (result.entry_count != reference.entry_count ||
                    result.promo_entry_count !=
                        reference.promo_entry_count) {
                    throw std::runtime_error(
                        std::string(phase) + " " + options.backend +
                        " hash-build count mismatch");
                }
            };
        joule::operators::cpu::HashBuildVerification verification;
        if (options.backend == "cpu") {
            summary.implementation_variant =
                "cpu-open-addressed-atomic-build";
            joule::operators::cpu::PartHashBuild op(
                part, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                validate_run,
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            verification = op.verify();
            summary.result_value_0 =
                static_cast<std::int64_t>(final.entry_count);
            summary.result_value_1 =
                static_cast<std::int64_t>(final.promo_entry_count);
        } else {
            summary.implementation_variant =
                "gpu-open-addressed-atomic-cas-build";
            joule::operators::gpu::PartHashBuild op(
                options.metal_library, part, options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                validate_run,
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            verification = op.verify();
            summary.result_value_0 =
                static_cast<std::int64_t>(final.entry_count);
            summary.result_value_1 =
                static_cast<std::int64_t>(final.promo_entry_count);
        }
        if (verification != reference) {
            throw std::runtime_error(
                options.backend + " hash-build verification mismatch");
        }
        summary.result_value_2 =
            static_cast<std::int64_t>(verification.key_sum);
        const std::array values{
            summary.result_value_0, summary.result_value_1,
            summary.result_value_2};
        summary.result_checksum = checksum<std::int64_t>(values);
        return;
    }

    if (options.operator_name == "hash-probe-count") {
        const auto part_path =
            sibling_path(options, options.part, "part.colbin");
        const joule::tpch::PartStore part_store(part_path);
        const auto part = part_store.view();
        const auto reference =
            joule::operators::cpu::part_hash_probe_count_reference(
                input, part);
        summary.data_files +=
            "," + std::filesystem::absolute(part_path).string();
        summary.semantics =
            "probe an immutable part hash table and count all/promo matches";
        summary.output_count = 2;
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.match_count);
        summary.reference_value_1 =
            static_cast<std::int64_t>(reference.promo_match_count);
        summary.logical_bytes_per_iteration =
            input.row_count() * 4.0L + 16.0L;
        const auto validate =
            [&](const auto& result, std::string_view phase) {
                if (result.match_count != reference.match_count ||
                    result.promo_match_count !=
                        reference.promo_match_count) {
                    throw std::runtime_error(
                        std::string(phase) + " " + options.backend +
                        " hash-probe-count mismatch");
                }
            };
        if (options.backend == "cpu") {
            summary.implementation_variant =
                "cpu-prebuilt-hash-probe";
            joule::operators::cpu::PartHashProbeCount op(
                input, part, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); }, validate,
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.match_count);
            summary.result_value_1 =
                static_cast<std::int64_t>(final.promo_match_count);
        } else {
            summary.implementation_variant =
                "gpu-prebuilt-hash-probe";
            joule::operators::gpu::PartHashProbeCount op(
                options.metal_library, input, part,
                options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); }, validate,
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.match_count);
            summary.result_value_1 =
                static_cast<std::int64_t>(final.promo_match_count);
        }
        const std::array values{
            summary.result_value_0, summary.result_value_1};
        summary.result_checksum = checksum<std::int64_t>(values);
        return;
    }

    if (options.operator_name == "hash-probe-materialize") {
        const auto part_path =
            sibling_path(options, options.part, "part.colbin");
        const joule::tpch::PartStore part_store(part_path);
        const auto part = part_store.view();
        const auto reference =
            joule::operators::cpu::part_hash_probe_materialize_reference(
                input, part);
        summary.data_files +=
            "," + std::filesystem::absolute(part_path).string();
        summary.semantics =
            "probe an immutable part hash table and stably materialize row-id/promo";
        summary.output_count = reference.size();
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.size());
        summary.logical_bytes_per_iteration =
            input.row_count() * 4.0L +
            reference.size() *
                sizeof(joule::operators::cpu::HashMatchRecord);
        if (options.backend == "cpu") {
            summary.implementation_variant =
                "cpu-two-pass-stable-hash-materialize";
            joule::operators::cpu::PartHashProbeMaterialize op(
                input, part, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size() ||
                        !std::equal(
                            op.output().begin(), op.output().end(),
                            reference.begin())) {
                        throw std::runtime_error(
                            std::string(phase) +
                            " CPU hash-probe-materialize mismatch");
                    }
                },
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.output_count);
            summary.result_checksum =
                checksum<joule::operators::cpu::HashMatchRecord>(
                    op.output());
        } else {
            summary.implementation_variant =
                "gpu-block-prefix-stable-hash-materialize";
            joule::operators::gpu::PartHashProbeMaterialize op(
                options.metal_library, input, part,
                options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size() ||
                        !std::equal(
                            op.output().begin(), op.output().end(),
                            reference.begin())) {
                        throw std::runtime_error(
                            std::string(phase) +
                            " GPU hash-probe-materialize mismatch");
                    }
                },
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.output_count);
            summary.result_checksum =
                checksum<joule::operators::gpu::HashMatchRecord>(
                    op.output());
        }
        return;
    }

    if (options.operator_name == "aggregate-sum" ||
        options.operator_name == "aggregate-minmax" ||
        options.operator_name == "aggregate-stats") {
        auto mode = joule::operators::cpu::PriceAggregateMode::stats;
        if (options.operator_name == "aggregate-sum") {
            mode = joule::operators::cpu::PriceAggregateMode::sum;
        } else if (options.operator_name == "aggregate-minmax") {
            mode = joule::operators::cpu::PriceAggregateMode::minmax;
        }
        const auto reference =
            joule::operators::cpu::price_aggregate_reference(
                input.extended_price, mode);
        summary.semantics =
            "fixed-point aggregate over l_extendedprice using float32 round(x*100)";
        summary.output_count =
            mode == joule::operators::cpu::PriceAggregateMode::stats ? 4 : 2;
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.count);
        summary.reference_value_1 = reference.sum_price_cents;
        summary.reference_value_2 = reference.min_price_cents;
        summary.reference_value_3 = reference.max_price_cents;
        summary.logical_bytes_per_iteration =
            input.row_count() * sizeof(float) + 4.0L * sizeof(std::int64_t);
        const auto validate = [&](const auto& result, std::string_view phase) {
            if (result.count != reference.count ||
                result.sum_price_cents != reference.sum_price_cents ||
                result.min_price_cents != reference.min_price_cents ||
                result.max_price_cents != reference.max_price_cents) {
                throw std::runtime_error(
                    std::string(phase) + " " + options.backend +
                    " price aggregate mismatch");
            }
        };
        if (options.backend == "cpu") {
            summary.implementation_variant = "cpu-parallel";
            joule::operators::cpu::PriceAggregate op(
                input.extended_price, mode, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); }, validate,
                [](const auto& result) { return result.compute_time_ms; }, summary);
            summary.result_value_0 = static_cast<std::int64_t>(final.count);
            summary.result_value_1 = final.sum_price_cents;
            summary.result_value_2 = final.min_price_cents;
            summary.result_value_3 = final.max_price_cents;
        } else {
            const auto reduction =
                options.gpu_aggregate_reduction == "simdgroup"
                ? joule::operators::gpu::PriceAggregateReduction::simdgroup
                : joule::operators::gpu::PriceAggregateReduction::threadgroup_tree;
            summary.implementation_variant =
                options.gpu_aggregate_reduction == "simdgroup"
                ? "gpu-simdgroup"
                : "gpu-threadgroup-tree";
            joule::operators::gpu::PriceAggregate op(
                options.metal_library, input.extended_price, mode,
                options.threadgroup_width, reduction);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); }, validate,
                [](const auto& result) { return result.gpu_time_ms; }, summary);
            summary.result_value_0 = static_cast<std::int64_t>(final.count);
            summary.result_value_1 = final.sum_price_cents;
            summary.result_value_2 = final.min_price_cents;
            summary.result_value_3 = final.max_price_cents;
        }
        const std::array values{
            summary.result_value_0, summary.result_value_1,
            summary.result_value_2, summary.result_value_3};
        summary.result_checksum = checksum<std::int64_t>(values);
        return;
    }

    if (options.operator_name == "groupby-part-count") {
        const auto reference =
            joule::operators::cpu::part_key_group_count_reference(
                input.part_key, options.group_cardinality);
        const auto nonempty = static_cast<std::uint64_t>(std::count_if(
            reference.begin(), reference.end(),
            [](std::uint32_t count) { return count != 0; }));
        summary.semantics =
            "COUNT(*) GROUP BY ((l_partkey-1) & (G-1)), dense uint32 output";
        summary.output_count = reference.size();
        summary.reference_value_0 =
            static_cast<std::int64_t>(input.row_count());
        summary.reference_value_1 = static_cast<std::int64_t>(nonempty);
        summary.logical_bytes_per_iteration =
            input.row_count() * sizeof(std::int32_t) +
            reference.size() * sizeof(std::uint32_t);
        if (options.backend == "cpu") {
            summary.implementation_variant = "cpu-parallel";
            joule::operators::cpu::PartKeyGroupCount op(
                input.part_key, options.group_cardinality, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto&, std::string_view phase) {
                    if (!std::equal(
                            op.output().begin(), op.output().end(),
                            reference.begin())) {
                        throw std::runtime_error(
                            std::string(phase) + " CPU group-by mismatch");
                    }
                },
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            summary.result_value_0 = static_cast<std::int64_t>(input.row_count());
            summary.result_value_1 = static_cast<std::int64_t>(nonempty);
            summary.result_checksum = checksum<std::uint32_t>(op.output());
        } else {
            const auto strategy =
                options.gpu_groupby_strategy == "global-atomic"
                ? joule::operators::gpu::GroupByCountStrategy::global_atomic
                : joule::operators::gpu::GroupByCountStrategy::bounded_threadgroup;
            summary.implementation_variant =
                options.gpu_groupby_strategy == "global-atomic"
                ? "gpu-global-atomic"
                : "gpu-bounded-threadgroup";
            joule::operators::gpu::PartKeyGroupCount op(
                options.metal_library, input.part_key,
                options.group_cardinality, options.threadgroup_width, strategy);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto&, std::string_view phase) {
                    if (!std::equal(
                            op.output().begin(), op.output().end(),
                            reference.begin())) {
                        throw std::runtime_error(
                            std::string(phase) + " GPU group-by mismatch");
                    }
                },
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            summary.result_value_0 = static_cast<std::int64_t>(input.row_count());
            summary.result_value_1 = static_cast<std::int64_t>(nonempty);
            summary.result_checksum = checksum<std::uint32_t>(op.output());
        }
        return;
    }

    if (options.operator_name == "filter-materialize") {
        const auto reference = joule::operators::cpu::q6_materialize_reference(input);
        summary.semantics = "TPC-H Q6 predicate with deterministic row-id materialization";
        summary.reference_value_0 = static_cast<std::int64_t>(reference.size());
        summary.logical_bytes_per_iteration = input.row_count() * 12.0L + reference.size() * 4.0L;
        if (options.backend == "cpu") {
            joule::operators::cpu::Q6FilterMaterialize op(input, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(options, handshake,
                [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size() ||
                        !std::equal(op.output().begin(), op.output().end(), reference.begin())) {
                        throw std::runtime_error(std::string(phase) + " CPU materialize mismatch");
                    }
                }, [](const auto& result) { return result.compute_time_ms; }, summary);
            summary.output_count = final.output_count;
            summary.result_checksum = checksum<std::uint32_t>(op.output());
        } else {
            joule::operators::gpu::Q6FilterMaterialize op(
                options.metal_library, input, options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(options, handshake,
                [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.output_count != reference.size() ||
                        !std::equal(op.output().begin(), op.output().end(), reference.begin())) {
                        throw std::runtime_error(std::string(phase) + " GPU materialize mismatch");
                    }
                }, [](const auto& result) { return result.gpu_time_ms; }, summary);
            summary.output_count = final.output_count;
            summary.result_checksum = checksum<std::uint32_t>(op.output());
        }
        summary.result_value_0 = static_cast<std::int64_t>(summary.output_count);
        return;
    }

    if (options.operator_name == "q6-revenue-unfused") {
        const auto reference =
            joule::operators::cpu::tpch_q6_reference(input, true);
        const auto bitmap_bytes =
            static_cast<long double>(
                reference.bitmap.size() * sizeof(std::uint32_t));
        summary.semantics =
            "TPC-H Q6 bitmap materialization followed by separate revenue reduction";
        summary.output_count = 2;
        summary.reference_value_0 =
            static_cast<std::int64_t>(reference.match_count);
        summary.reference_value_1 = reference.revenue_1e4_usd;
        summary.logical_bytes_per_iteration =
            input.row_count() * 12.0L + bitmap_bytes * 2.0L +
            reference.match_count * 8.0L + 16.0L;
        if (options.backend == "cpu") {
            summary.implementation_variant = "cpu-unfused";
            joule::operators::cpu::TpchQ6UnfusedConfig config;
            config.thread_count = options.cpu_threads;
            joule::operators::cpu::TpchQ6Unfused op(input, config);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.match_count != reference.match_count ||
                        result.revenue_1e4_usd != reference.revenue_1e4_usd ||
                        !std::equal(
                            op.bitmap().begin(), op.bitmap().end(),
                            reference.bitmap.begin())) {
                        throw std::runtime_error(
                            std::string(phase) + " CPU unfused Q6 mismatch");
                    }
                },
                [](const auto& result) { return result.compute_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.match_count);
            summary.result_value_1 = final.revenue_1e4_usd;
        } else {
            summary.implementation_variant = "gpu-unfused";
            joule::operators::gpu::TpchQ6UnfusedConfig config;
            config.threadgroup_width = options.threadgroup_width;
            joule::operators::gpu::TpchQ6Unfused op(
                options.metal_library, input, config);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            summary.max_threads = op.max_threads_per_threadgroup();
            const auto final = run_loop(
                options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.match_count != reference.match_count ||
                        result.revenue_1e4_usd != reference.revenue_1e4_usd) {
                        throw std::runtime_error(
                            std::string(phase) + " GPU unfused Q6 mismatch");
                    }
                },
                [](const auto& result) { return result.gpu_time_ms; },
                summary);
            summary.result_value_0 =
                static_cast<std::int64_t>(final.match_count);
            summary.result_value_1 = final.revenue_1e4_usd;
        }
        const std::array values{
            summary.result_value_0, summary.result_value_1};
        summary.result_checksum = checksum<std::int64_t>(values);
        return;
    }

    if (options.operator_name == "q1-groupby") {
        const auto reference = joule::operators::cpu::tpch_q1_reference(input);
        summary.semantics = "TPC-H Q1 filter plus six-key grouped fixed-point aggregates";
        summary.output_count = reference.size();
        summary.result_checksum = checksum<joule::operators::cpu::TpchQ1Group>(reference);
        summary.logical_bytes_per_iteration = input.row_count() * 22.0L + sizeof(reference);
        if (options.backend == "cpu") {
            joule::operators::cpu::TpchQ1GroupBy op(input, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            run_loop(options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    validate_q1(result.groups, reference, phase, "CPU");
                }, [](const auto& result) { return result.compute_time_ms; }, summary);
        } else {
            joule::operators::gpu::TpchQ1GroupBy op(
                options.metal_library, input, options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            run_loop(options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    validate_q1(result.groups, reference, phase, "GPU");
                }, [](const auto& result) { return result.gpu_time_ms; }, summary);
        }
        return;
    }

    if (options.operator_name == "q14-join") {
        const auto part_path = sibling_path(options, options.part, "part.colbin");
        const joule::tpch::PartStore part_store(part_path);
        const auto part = part_store.view();
        const auto reference = joule::operators::cpu::tpch_q14_reference(input, part);
        summary.data_files += "," + std::filesystem::absolute(part_path).string();
        summary.semantics = "TPC-H Q14 September 1995 hash probe plus promo/total aggregates";
        summary.output_count = 2;
        summary.reference_value_0 = reference.promo_revenue_1e4_usd;
        summary.reference_value_1 = reference.total_revenue_1e4_usd;
        summary.logical_bytes_per_iteration = input.row_count() * 16.0L + 16.0L;
        if (options.backend == "cpu") {
            joule::operators::cpu::TpchQ14HashJoin op(input, part, options.cpu_threads);
            summary.cpu_threads = op.thread_count();
            const auto final = run_loop(options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.promo_revenue_1e4_usd != reference.promo_revenue_1e4_usd ||
                        result.total_revenue_1e4_usd != reference.total_revenue_1e4_usd) {
                        throw std::runtime_error(std::string(phase) + " CPU Q14 mismatch");
                    }
                }, [](const auto& result) { return result.compute_time_ms; }, summary);
            summary.result_value_0 = final.promo_revenue_1e4_usd;
            summary.result_value_1 = final.total_revenue_1e4_usd;
        } else {
            joule::operators::gpu::TpchQ14HashJoin op(
                options.metal_library, input, part, options.threadgroup_width);
            summary.device = op.device_name();
            summary.execution_width = op.execution_width();
            const auto final = run_loop(options, handshake, [&] { return op.execute(); },
                [&](const auto& result, std::string_view phase) {
                    if (result.promo_revenue_1e4_usd != reference.promo_revenue_1e4_usd ||
                        result.total_revenue_1e4_usd != reference.total_revenue_1e4_usd) {
                        throw std::runtime_error(std::string(phase) + " GPU Q14 mismatch");
                    }
                }, [](const auto& result) { return result.gpu_time_ms; }, summary);
            summary.result_value_0 = final.promo_revenue_1e4_usd;
            summary.result_value_1 = final.total_revenue_1e4_usd;
        }
        const std::array values{summary.result_value_0, summary.result_value_1};
        summary.result_checksum = checksum<std::int64_t>(values);
        return;
    }

    const auto mode = q6_mode(options.operator_name);
    const auto reference = joule::operators::cpu::tpch_q6_reference(
        input, mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap);
    summary.semantics = "TPC-H Q6 (1994, discount 0.05..0.07, quantity < 24)";
    summary.reference_value_0 = static_cast<std::int64_t>(reference.match_count);
    summary.reference_value_1 = reference.revenue_1e4_usd;
    summary.logical_bytes_per_iteration = input.row_count() *
        (mode == joule::operators::cpu::TpchQ6Mode::revenue ? 16.0L : 12.0L);
    if (options.backend == "cpu") {
        if (mode == joule::operators::cpu::TpchQ6Mode::revenue) {
            summary.implementation_variant = "cpu-fused";
        }
        joule::operators::cpu::TpchQ6Config config{mode, options.cpu_threads};
        joule::operators::cpu::TpchQ6 op(input, config);
        summary.cpu_threads = op.thread_count();
        const auto final = run_loop(options, handshake, [&] { return op.execute(); },
            [&](const auto& result, std::string_view phase) {
                const bool bitmap_ok = mode != joule::operators::cpu::TpchQ6Mode::filter_bitmap ||
                    std::equal(op.bitmap().begin(), op.bitmap().end(), reference.bitmap.begin());
                const bool scalar_ok = mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap ||
                    (result.match_count == reference.match_count &&
                     (mode != joule::operators::cpu::TpchQ6Mode::revenue ||
                      result.revenue_1e4_usd == reference.revenue_1e4_usd));
                if (!bitmap_ok || !scalar_ok) {
                    throw std::runtime_error(std::string(phase) + " CPU Q6 mismatch");
                }
            }, [](const auto& result) { return result.compute_time_ms; }, summary);
        summary.result_value_0 = static_cast<std::int64_t>(final.match_count);
        summary.result_value_1 = final.revenue_1e4_usd;
        if (mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap) {
            summary.output_count = op.bitmap().size();
            summary.result_checksum = checksum<std::uint32_t>(op.bitmap());
        }
    } else {
        if (mode == joule::operators::cpu::TpchQ6Mode::revenue) {
            summary.implementation_variant = "gpu-fused";
        }
        joule::operators::gpu::TpchQ6Config config{mode, options.threadgroup_width};
        joule::operators::gpu::TpchQ6 op(options.metal_library, input, config);
        summary.device = op.device_name();
        summary.execution_width = op.execution_width();
        summary.max_threads = op.max_threads_per_threadgroup();
        const auto final = run_loop(options, handshake, [&] { return op.execute(); },
            [&](const auto& result, std::string_view phase) {
                const bool bitmap_ok = mode != joule::operators::cpu::TpchQ6Mode::filter_bitmap ||
                    std::equal(op.bitmap().begin(), op.bitmap().end(), reference.bitmap.begin());
                const bool scalar_ok = mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap ||
                    (result.match_count == reference.match_count &&
                     (mode != joule::operators::cpu::TpchQ6Mode::revenue ||
                      result.revenue_1e4_usd == reference.revenue_1e4_usd));
                if (!bitmap_ok || !scalar_ok) {
                    throw std::runtime_error(std::string(phase) + " GPU Q6 mismatch");
                }
            }, [](const auto& result) { return result.gpu_time_ms; }, summary);
        summary.result_value_0 = static_cast<std::int64_t>(final.match_count);
        summary.result_value_1 = final.revenue_1e4_usd;
        if (mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap) {
            summary.output_count = op.bitmap().size();
            summary.result_checksum = checksum<std::uint32_t>(op.bitmap());
        }
    }
    if (mode == joule::operators::cpu::TpchQ6Mode::filter_bitmap) {
        // The bitmap itself is the timed output. Exact validation above makes
        // its verified popcount available for unambiguous result metadata
        // without adding a reduction to the measured operator.
        summary.result_value_0 =
            static_cast<std::int64_t>(reference.match_count);
    } else if (mode == joule::operators::cpu::TpchQ6Mode::revenue) {
        summary.output_count = 2;
        const std::array values{
            summary.result_value_0, summary.result_value_1};
        summary.result_checksum = checksum<std::int64_t>(values);
    }
}

void run_orders_topk(
    const Options& options,
    joule::MeasurementHandshake& handshake,
    Summary& summary) {
    const auto path = sibling_path(options, options.orders, "orders.colbin");
    const joule::tpch::OrdersStore store(path);
    const auto input = store.view();
    const auto reference = joule::operators::cpu::top10_reference(input);
    summary.rows = input.row_count();
    summary.output_count = reference.size();
    summary.data_files = std::filesystem::absolute(path).string();
    summary.semantics = "orders top 10 by totalprice descending, orderkey ascending";
    summary.logical_bytes_per_iteration = input.row_count() * 8.0L + sizeof(reference);
    summary.reference_value_0 = reference.front().total_price_cents;
    summary.reference_value_1 = reference.front().order_key;
    summary.result_checksum = checksum<joule::operators::cpu::TopKEntry>(reference);
    if (options.backend == "cpu") {
        joule::operators::cpu::OrdersTopK op(input, options.cpu_threads);
        summary.cpu_threads = op.thread_count();
        const auto final = run_loop(options, handshake, [&] { return op.execute(); },
            [&](const auto& result, std::string_view phase) {
                if (result.rows != reference) {
                    throw std::runtime_error(std::string(phase) + " CPU top-k mismatch");
                }
            }, [](const auto& result) { return result.compute_time_ms; }, summary);
        summary.result_value_0 = final.rows.front().total_price_cents;
        summary.result_value_1 = final.rows.front().order_key;
    } else {
        joule::operators::gpu::OrdersTopK op(
            options.metal_library, input, options.threadgroup_width);
        summary.device = op.device_name();
        summary.execution_width = op.execution_width();
        const auto final = run_loop(options, handshake, [&] { return op.execute(); },
            [&](const auto& result, std::string_view phase) {
                if (result.rows != reference) {
                    throw std::runtime_error(std::string(phase) + " GPU top-k mismatch");
                }
            }, [](const auto& result) { return result.gpu_time_ms; }, summary);
        summary.result_value_0 = final.rows.front().total_price_cents;
        summary.result_value_1 = final.rows.front().order_key;
    }
}

void print_summary(
    std::ostream& output,
    const Options& options,
    const Summary& summary) {
    const auto wall_seconds = summary.wall_time_ms / 1'000.0;
    const auto throughput = static_cast<double>(
        summary.logical_bytes_per_iteration * summary.iterations / wall_seconds / 1e9L);
    output << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"operator\": \"" << json_escape(options.operator_name) << "\",\n"
              << "  \"backend\": \"" << json_escape(options.backend) << "\",\n"
              << "  \"dataset_format\": \"GPUDBMetalCodeGen-colbin-v2\",\n"
              << "  \"data\": \"" << json_escape(summary.data_files) << "\",\n"
              << "  \"rows\": " << summary.rows << ",\n"
              << "  \"semantics\": \"" << json_escape(summary.semantics) << "\",\n"
              << "  \"implementation_variant\": \""
              << json_escape(summary.implementation_variant) << "\",\n"
              << "  \"device\": \"" << json_escape(summary.device) << "\",\n"
              << "  \"cpu_threads_requested\": " << options.cpu_threads << ",\n"
              << "  \"cpu_threads\": " << summary.cpu_threads << ",\n"
              << "  \"threadgroup_width\": " << options.threadgroup_width << ",\n"
              << "  \"group_cardinality\": " << options.group_cardinality << ",\n"
              << "  \"execution_width\": " << summary.execution_width << ",\n"
              << "  \"max_threads_per_threadgroup\": " << summary.max_threads << ",\n"
              << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
              << "  \"memory_state\": \"warm\",\n"
              << "  \"input_storage\": \"mmap-colbin-zero-copy\",\n"
              << "  \"setup_in_timed_region\": false,\n"
              << "  \"iterations\": " << summary.iterations << ",\n"
              << "  \"wall_time_ms\": " << summary.wall_time_ms << ",\n"
              << "  \"end_to_end_ms_per_operator\": "
              << summary.wall_time_ms / static_cast<double>(summary.iterations) << ",\n"
              << "  \"compute_time_ms\": " << summary.accumulated_compute_ms << ",\n"
              << "  \"compute_ms_per_operator\": "
              << summary.accumulated_compute_ms / static_cast<double>(summary.iterations) << ",\n"
              << "  \"logical_throughput_gbps\": " << throughput << ",\n"
              << "  \"output_count\": " << summary.output_count << ",\n"
              << "  \"reference_value_0\": " << summary.reference_value_0 << ",\n"
              << "  \"reference_value_1\": " << summary.reference_value_1 << ",\n"
              << "  \"reference_value_2\": " << summary.reference_value_2 << ",\n"
              << "  \"reference_value_3\": " << summary.reference_value_3 << ",\n"
              << "  \"result_value_0\": " << summary.result_value_0 << ",\n"
              << "  \"result_value_1\": " << summary.result_value_1 << ",\n"
              << "  \"result_value_2\": " << summary.result_value_2 << ",\n"
              << "  \"result_value_3\": " << summary.result_value_3 << ",\n"
              << "  \"result_checksum\": " << summary.result_checksum << ",\n"
              << "  \"reference_match_count\": " << summary.reference_value_0 << ",\n"
              << "  \"reference_revenue_1e4_usd\": " << summary.reference_value_1 << ",\n"
              << "  \"result_match_count\": " << summary.result_value_0 << ",\n"
              << "  \"result_revenue_1e4_usd\": " << summary.result_value_1 << ",\n"
              << "  \"bitmap_checksum\": " << summary.result_checksum << "\n"
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        joule::MeasurementHandshake handshake;
        Summary summary;
        if (options.operator_name == "orders-topk") {
            run_orders_topk(options, handshake, summary);
        } else {
            run_lineitem_operator(options, handshake, summary);
        }
        if (options.result_json.empty()) {
            print_summary(std::cout, options, summary);
        } else {
            const auto path = std::filesystem::absolute(options.result_json);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream output(path);
            if (!output) {
                throw std::runtime_error(
                    "could not create benchmark result file: " + path.string());
            }
            print_summary(output, options, summary);
            if (!output) {
                throw std::runtime_error(
                    "could not write benchmark result file: " + path.string());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "joule-tpch-benchmark: " << error.what() << '\n';
        return 1;
    }
}
