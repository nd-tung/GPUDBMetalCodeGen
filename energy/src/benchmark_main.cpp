#include "joule/operators/cpu/scan_sum.hpp"
#include "joule/operators/gpu/scan_sum.hpp"
#include "joule/measurement_handshake.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint64_t rows{1ULL << 24};
    std::uint64_t duration_ms{2'000};
    std::string backend{"cpu"};
    std::string cpu_kernel{"simd"};
    std::uint32_t cpu_threads{0};
    std::string gpu_kernel{"simdgroup"};
    std::uint32_t threadgroup_width{256};
    std::uint32_t batch_size{1};
    std::uint32_t warmup_iterations{5};
    std::string input_pattern{"positive"};
    std::filesystem::path result_json;
#if defined(JOULE_DEFAULT_METALLIB_PATH)
    std::filesystem::path metal_library{JOULE_DEFAULT_METALLIB_PATH};
#else
    std::filesystem::path metal_library{"build/metal/joule.metallib"};
#endif
};

[[nodiscard]] std::uint64_t parse_uint64(std::string_view value, std::string_view option) {
    std::size_t parsed_characters = 0;
    const auto number = std::stoull(std::string(value), &parsed_characters);
    if (parsed_characters != value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return number;
}

void print_help() {
    std::cout
        << "Usage: joule-benchmark [options]\n\n"
        << "Options:\n"
        << "  --backend cpu|gpu   execution backend (default: cpu)\n"
        << "  --rows N            number of int32 rows (default: 16777216)\n"
        << "  --input-pattern NAME  positive|signed (default: positive)\n"
        << "  --duration-ms N     minimum timed duration (default: 2000)\n"
        << "  --warmup-iterations N  complete operators before timing (default: 5)\n"
        << "  --cpu-kernel NAME   scalar|parallel|simd (default: simd)\n"
        << "  --cpu-threads N     worker count; 0 uses all logical CPUs (default: 0)\n"
        << "  --gpu-kernel NAME   baseline|multi-item|simdgroup (default: simdgroup)\n"
        << "  --threadgroup-width N  Metal threads per group (default: 256)\n"
        << "  --batch-size N      scans per Metal command buffer; 1 is end-to-end (default: 1)\n"
        << "  --result-json PATH  write the benchmark summary to a dedicated JSON file\n"
        << "  --metallib PATH     compiled Metal library path\n";
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
        if (argument == "--rows") {
            options.rows = parse_uint64(value, argument);
        } else if (argument == "--input-pattern") {
            options.input_pattern = value;
        } else if (argument == "--duration-ms") {
            options.duration_ms = parse_uint64(value, argument);
        } else if (argument == "--backend") {
            options.backend = value;
        } else if (argument == "--cpu-kernel") {
            options.cpu_kernel = value == "parallel-simd" ? "simd" : std::string(value);
        } else if (argument == "--cpu-threads") {
            const auto count = parse_uint64(value, argument);
            if (count > UINT32_MAX) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            options.cpu_threads = static_cast<std::uint32_t>(count);
        } else if (argument == "--gpu-kernel") {
            options.gpu_kernel = value == "optimized" ? "simdgroup" : std::string(value);
        } else if (argument == "--threadgroup-width") {
            const auto width = parse_uint64(value, argument);
            if (width > UINT32_MAX) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            options.threadgroup_width = static_cast<std::uint32_t>(width);
        } else if (argument == "--batch-size") {
            const auto size = parse_uint64(value, argument);
            if (size > UINT32_MAX) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            options.batch_size = static_cast<std::uint32_t>(size);
        } else if (argument == "--warmup-iterations") {
            const auto count = parse_uint64(value, argument);
            if (count > UINT32_MAX) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            options.warmup_iterations = static_cast<std::uint32_t>(count);
        } else if (argument == "--metallib") {
            options.metal_library = std::string(value);
        } else if (argument == "--result-json") {
            options.result_json = std::string(value);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    if (options.rows == 0) {
        throw std::invalid_argument("--rows must be greater than zero");
    }
    if (options.backend != "cpu" && options.backend != "gpu") {
        throw std::invalid_argument("--backend must be cpu or gpu");
    }
    if (options.input_pattern != "positive" && options.input_pattern != "signed") {
        throw std::invalid_argument("--input-pattern must be positive or signed");
    }
    if (options.cpu_kernel != "scalar" && options.cpu_kernel != "parallel" &&
        options.cpu_kernel != "simd") {
        throw std::invalid_argument("--cpu-kernel must be scalar, parallel, or simd");
    }
    if (options.gpu_kernel != "baseline" && options.gpu_kernel != "multi-item" &&
        options.gpu_kernel != "simdgroup") {
        throw std::invalid_argument(
            "--gpu-kernel must be baseline, multi-item, or simdgroup");
    }
    if (options.threadgroup_width < 32 || options.threadgroup_width > 512 ||
        !std::has_single_bit(options.threadgroup_width)) {
        throw std::invalid_argument(
            "--threadgroup-width must be a power of two between 32 and 512");
    }
    if (options.batch_size == 0) {
        throw std::invalid_argument("--batch-size must be greater than zero");
    }
    return options;
}

[[nodiscard]] std::vector<std::int32_t> make_input(
    std::uint64_t rows,
    std::string_view pattern) {
    std::vector<std::int32_t> input(static_cast<std::size_t>(rows));
    std::uint32_t state = 0x9e3779b9U;
    for (auto& value : input) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = pattern == "signed"
            ? static_cast<std::int32_t>(state)
            : static_cast<std::int32_t>(state & 1'023U);
    }
    return input;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        joule::MeasurementHandshake measurement_handshake;
        const auto input = make_input(options.rows, options.input_pattern);
        const std::span<const std::int32_t> input_view(input);

        // Allocation, input generation, pipeline creation, and warm-up are outside
        // the benchmark's timed region. joule-measure reports its wider process window.
        const auto reference_sum = joule::operators::cpu::scan_sum_i32(input_view);

        std::int64_t last_sum = 0;
        std::uint64_t iterations = 0;
        double accumulated_cpu_compute_time_ms = 0.0;
        double accumulated_gpu_time_ms = 0.0;
        std::string device_name;
        std::uint32_t cpu_thread_count = 0;
        std::uint32_t execution_width = 0;
        std::uint32_t max_threads_per_threadgroup = 0;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;

        if (options.backend == "cpu") {
            joule::operators::cpu::ScanSumConfig config;
            if (options.cpu_kernel == "scalar") {
                config.kernel = joule::operators::cpu::ScanSumKernel::scalar;
            } else if (options.cpu_kernel == "parallel") {
                config.kernel = joule::operators::cpu::ScanSumKernel::parallel;
            } else {
                config.kernel = joule::operators::cpu::ScanSumKernel::parallel_simd;
            }
            config.thread_count = options.cpu_threads;
            joule::operators::cpu::ScanSum cpu_scan(input_view, config);
            cpu_thread_count = cpu_scan.thread_count();
            for (std::uint32_t warmup = 0; warmup < options.warmup_iterations; ++warmup) {
                if (cpu_scan.execute().sum != reference_sum) {
                    throw std::runtime_error(
                        "CPU warm-up checksum does not match the scalar reference");
                }
            }
            measurement_handshake.ready_and_wait();

            start = std::chrono::steady_clock::now();
            const auto deadline = start + std::chrono::milliseconds(options.duration_ms);
            do {
                const auto run = cpu_scan.execute();
                last_sum = run.sum;
                accumulated_cpu_compute_time_ms += run.compute_time_ms;
                ++iterations;
            } while (std::chrono::steady_clock::now() < deadline);
            end = std::chrono::steady_clock::now();
            measurement_handshake.complete();
        } else {
            joule::operators::gpu::ScanSumConfig config;
            if (options.gpu_kernel == "baseline") {
                config.kernel = joule::operators::gpu::ScanSumKernel::baseline;
            } else if (options.gpu_kernel == "multi-item") {
                config.kernel = joule::operators::gpu::ScanSumKernel::multi_item;
            } else {
                config.kernel = joule::operators::gpu::ScanSumKernel::simdgroup;
            }
            config.threadgroup_width = options.threadgroup_width;
            joule::operators::gpu::ScanSum gpu_scan(
                options.metal_library, input_view, config);
            device_name = gpu_scan.device_name();
            execution_width = gpu_scan.execution_width();
            max_threads_per_threadgroup = gpu_scan.max_threads_per_threadgroup();
            for (std::uint32_t warmup = 0; warmup < options.warmup_iterations; ++warmup) {
                if (gpu_scan.execute().sum != reference_sum) {
                    throw std::runtime_error(
                        "GPU warm-up checksum does not match the scalar reference");
                }
            }
            measurement_handshake.ready_and_wait();

            start = std::chrono::steady_clock::now();
            const auto deadline = start + std::chrono::milliseconds(options.duration_ms);
            do {
                const auto run = gpu_scan.execute_batch(options.batch_size);
                last_sum = run.sum;
                accumulated_gpu_time_ms += run.gpu_time_ms;
                iterations += run.repetitions;
            } while (std::chrono::steady_clock::now() < deadline);
            end = std::chrono::steady_clock::now();
            measurement_handshake.complete();
        }

        if (last_sum != reference_sum) {
            throw std::runtime_error(
                "final operator checksum does not match the scalar reference");
        }

        const auto wall_time_s = std::chrono::duration<double>(end - start).count();
        const auto bytes_processed =
            static_cast<long double>(options.rows) * sizeof(std::int32_t) * iterations;
        const auto throughput_gbps = static_cast<double>(bytes_processed / wall_time_s / 1e9L);

        std::ostringstream output;
        output << std::fixed << std::setprecision(6)
                  << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"operator\": \"scan_sum\",\n"
                  << "  \"backend\": \"" << options.backend << "\",\n"
                  << "  \"device\": \"" << device_name << "\",\n"
                  << "  \"cpu_kernel\": \"" << options.cpu_kernel << "\",\n"
                  << "  \"cpu_threads_requested\": " << options.cpu_threads << ",\n"
                  << "  \"cpu_threads\": " << cpu_thread_count << ",\n"
                  << "  \"gpu_kernel\": \"" << options.gpu_kernel << "\",\n"
                  << "  \"threadgroup_width\": " << options.threadgroup_width << ",\n"
                  << "  \"batch_size\": " << options.batch_size << ",\n"
                  << "  \"execution_width\": " << execution_width << ",\n"
                  << "  \"max_threads_per_threadgroup\": "
                  << max_threads_per_threadgroup << ",\n"
                  << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
                  << "  \"memory_state\": \"warm\",\n"
                  << "  \"input_storage\": \""
                  << (options.backend == "gpu" ? "metal_shared_copy" : "std_vector")
                  << "\",\n"
                  << "  \"input_pattern\": \"" << options.input_pattern << "\",\n"
                  << "  \"rows\": " << options.rows << ",\n"
                  << "  \"iterations\": " << iterations << ",\n"
                  << "  \"wall_time_ms\": " << wall_time_s * 1'000.0 << ",\n"
                  << "  \"end_to_end_ms_per_operator\": "
                  << wall_time_s * 1'000.0 / static_cast<double>(iterations) << ",\n"
                  << "  \"cpu_compute_time_ms\": "
                  << accumulated_cpu_compute_time_ms << ",\n"
                  << "  \"gpu_time_ms\": " << accumulated_gpu_time_ms << ",\n"
                  << "  \"compute_ms_per_operator\": "
                  << (options.backend == "cpu"
                          ? accumulated_cpu_compute_time_ms
                          : accumulated_gpu_time_ms) /
                         static_cast<double>(iterations)
                  << ",\n"
                  << "  \"throughput_gbps\": " << throughput_gbps << ",\n"
                  << "  \"reference_sum\": " << reference_sum << ",\n"
                  << "  \"checksum\": " << last_sum << "\n"
                  << "}\n";
        if (options.result_json.empty()) {
            std::cout << output.str();
        } else {
            const auto path = std::filesystem::absolute(options.result_json);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream result(path);
            if (!result) {
                throw std::runtime_error(
                    "could not create benchmark result file: " + path.string());
            }
            result << output.str();
            if (!result) {
                throw std::runtime_error(
                    "could not write benchmark result file: " + path.string());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "joule-benchmark: " << error.what() << '\n';
        return 1;
    }
}
