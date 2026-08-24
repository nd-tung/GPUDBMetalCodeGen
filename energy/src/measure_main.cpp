#include "joule/metrics.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CliOptions {
    joule::metrics::MeasurementConfig measurement;
    std::filesystem::path output_path{"-"};
    std::vector<std::string> command;
};

[[nodiscard]] std::string default_trace_name() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_time{};
    localtime_r(&now, &local_time);
    std::ostringstream name;
    name << "results/raw/powermetrics-" << std::put_time(&local_time, "%Y%m%d-%H%M%S")
         << ".plist";
    return name.str();
}

[[nodiscard]] std::string default_shelly_trace_name() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_time{};
    localtime_r(&now, &local_time);
    std::ostringstream name;
    name << "results/raw/shelly-" << std::put_time(&local_time, "%Y%m%d-%H%M%S")
         << ".jsonl";
    return name.str();
}

[[nodiscard]] std::uint32_t parse_uint32(std::string_view value, std::string_view option) {
    std::size_t parsed_characters = 0;
    const auto number = std::stoul(std::string(value), &parsed_characters);
    if (parsed_characters != value.size() || number > UINT32_MAX) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return static_cast<std::uint32_t>(number);
}

void print_help(std::ostream& output) {
    output
        << "Usage: joule-measure [options] -- command [arguments...]\n\n"
        << "Options:\n"
        << "  --sample-rate-ms N  powermetrics interval (default: 100)\n"
        << "  --baseline-ms N     idle baseline duration (default: 1000)\n"
        << "  --raw PATH          raw NUL-separated plist trace path\n"
        << "  --output PATH       result JSON path, or - for stdout (default: -)\n"
        << "  --cooperative      measure only the child's prepared timed region\n"
        << "  --cooperative-timeout-ms N  deadline per cooperative phase (0 disables)\n"
        << "  --no-sudo           run powermetrics directly (requires root)\n"
        << "  --shelly-host HOST  enable Shelly wall-power sampling\n"
        << "  --shelly-interface NAME  bind Shelly HTTP to this macOS interface\n"
        << "  --shelly-port N     Shelly HTTP port (default: 80)\n"
        << "  --shelly-sample-rate-ms N  Shelly polling interval (default: 1000)\n"
        << "  --shelly-timeout-ms N      per-request timeout (default: 2000)\n"
        << "  --shelly-attempts N        requests per scheduled sample (default: 3)\n"
        << "  --shelly-raw PATH   raw Shelly JSONL trace path\n"
        << "  --shelly-device-id ID      require this Shelly device id\n"
        << "  -h, --help          show this help\n\n"
        << "Run 'sudo -v' before measuring. Only powermetrics is launched through sudo.\n";
}

[[nodiscard]] CliOptions parse_arguments(int argc, char** argv) {
    CliOptions options;
    options.measurement.raw_trace_path = default_trace_name();

    const auto shelly_config = [&]() -> joule::metrics::ShellyConfig& {
        if (!options.measurement.shelly) {
            options.measurement.shelly.emplace();
        }
        return *options.measurement.shelly;
    };

    int index = 1;
    for (; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--") {
            ++index;
            break;
        }
        if (argument == "-h" || argument == "--help") {
            print_help(std::cout);
            std::exit(0);
        }
        if (argument == "--no-sudo") {
            options.measurement.use_sudo = false;
            continue;
        }
        if (argument == "--cooperative") {
            options.measurement.cooperative_boundary = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(argument));
        }
        const std::string_view value(argv[++index]);
        if (argument == "--sample-rate-ms") {
            options.measurement.sample_rate_ms = parse_uint32(value, argument);
        } else if (argument == "--baseline-ms") {
            options.measurement.baseline_ms = parse_uint32(value, argument);
        } else if (argument == "--cooperative-timeout-ms") {
            options.measurement.cooperative_timeout_ms = parse_uint32(value, argument);
        } else if (argument == "--raw") {
            options.measurement.raw_trace_path = value;
        } else if (argument == "--output") {
            options.output_path = value;
        } else if (argument == "--shelly-host") {
            shelly_config().host = value;
        } else if (argument == "--shelly-interface") {
            if (value.empty()) {
                throw std::invalid_argument("--shelly-interface must not be empty");
            }
            shelly_config().interface = value;
        } else if (argument == "--shelly-port") {
            const auto port = parse_uint32(value, argument);
            if (port == 0 || port > UINT16_MAX) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            shelly_config().port = static_cast<std::uint16_t>(port);
        } else if (argument == "--shelly-sample-rate-ms") {
            shelly_config().sample_rate_ms = parse_uint32(value, argument);
        } else if (argument == "--shelly-timeout-ms") {
            shelly_config().timeout_ms = parse_uint32(value, argument);
        } else if (argument == "--shelly-attempts") {
            shelly_config().attempts = parse_uint32(value, argument);
        } else if (argument == "--shelly-raw") {
            shelly_config().raw_trace_path = value;
        } else if (argument == "--shelly-device-id") {
            shelly_config().expected_device_id = std::string(value);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    for (; index < argc; ++index) {
        options.command.emplace_back(argv[index]);
    }
    if (options.command.empty()) {
        throw std::invalid_argument("missing command after --");
    }
    if (options.measurement.shelly) {
        auto& shelly = *options.measurement.shelly;
        if (shelly.host.empty()) {
            throw std::invalid_argument("--shelly-host is required when using Shelly options");
        }
        if (shelly.sample_rate_ms == 0 || shelly.timeout_ms == 0 ||
            shelly.attempts == 0) {
            throw std::invalid_argument(
                "Shelly sample rate, timeout, and attempts must be greater than zero");
        }
        if (shelly.expected_device_id && shelly.expected_device_id->empty()) {
            throw std::invalid_argument("--shelly-device-id must not be empty");
        }
        if (shelly.raw_trace_path.empty()) {
            shelly.raw_trace_path = default_shelly_trace_name();
        }
    }
    return options;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << character; break;
        }
    }
    escaped << '"';
    return escaped.str();
}

void write_optional(std::ostream& output, const std::optional<double>& value) {
    if (value) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_summary(
    std::ostream& output,
    const joule::metrics::PowerSummary& summary,
    std::string_view indentation) {
    output << "{\n"
           << indentation << "  \"sample_count\": " << summary.sample_count << ",\n"
           << indentation << "  \"rail_sample_count\": {\n"
           << indentation << "    \"cpu\": " << summary.cpu_sample_count << ",\n"
           << indentation << "    \"gpu\": " << summary.gpu_sample_count << ",\n"
           << indentation << "    \"ane\": " << summary.ane_sample_count << ",\n"
           << indentation << "    \"soc\": " << summary.total_sample_count << ",\n"
           << indentation << "    \"total\": " << summary.total_sample_count << "\n"
           << indentation << "  },\n"
           << indentation << "  \"sampled_time_s\": " << summary.sampled_time_s << ",\n"
           << indentation << "  \"energy_j\": {\n"
           << indentation << "    \"cpu\": ";
    write_optional(output, summary.cpu_energy_j);
    output << ",\n" << indentation << "    \"gpu\": ";
    write_optional(output, summary.gpu_energy_j);
    output << ",\n" << indentation << "    \"ane\": ";
    write_optional(output, summary.ane_energy_j);
    output << ",\n" << indentation << "    \"soc\": ";
    write_optional(output, summary.soc_energy_j);
    output << ",\n" << indentation << "    \"total\": ";
    write_optional(output, summary.total_energy_j);
    output << "\n" << indentation << "  },\n"
           << indentation << "  \"average_power_w\": {\n"
           << indentation << "    \"cpu\": ";
    write_optional(output, summary.average_cpu_power_w);
    output << ",\n" << indentation << "    \"gpu\": ";
    write_optional(output, summary.average_gpu_power_w);
    output << ",\n" << indentation << "    \"ane\": ";
    write_optional(output, summary.average_ane_power_w);
    output << ",\n" << indentation << "    \"soc\": ";
    write_optional(output, summary.average_soc_power_w);
    output << ",\n" << indentation << "    \"total\": ";
    write_optional(output, summary.average_total_power_w);
    output << "\n" << indentation << "  }\n" << indentation << '}';
}

void write_shelly_window(
    std::ostream& output,
    const joule::metrics::ShellyWindowSummary& summary,
    std::string_view indentation) {
    output << "{\n"
           << indentation << "  \"request_count\": " << summary.request_count << ",\n"
           << indentation << "  \"sample_count\": " << summary.sample_count << ",\n"
           << indentation << "  \"failure_count\": " << summary.failure_count << ",\n"
           << indentation << "  \"sampled_time_s\": " << summary.sampled_time_s << ",\n"
           << indentation << "  \"counter_monotonic\": "
           << (summary.counter_monotonic ? "true" : "false") << ",\n"
           << indentation << "  \"counter_start_wh\": ";
    write_optional(output, summary.counter_start_wh);
    output << ",\n" << indentation << "  \"counter_end_wh\": ";
    write_optional(output, summary.counter_end_wh);
    output << ",\n" << indentation << "  \"counter_delta_wh\": ";
    write_optional(output, summary.counter_delta_wh);
    output << ",\n" << indentation << "  \"energy_j\": ";
    write_optional(output, summary.energy_j);
    output << ",\n" << indentation << "  \"average_power_w\": ";
    write_optional(output, summary.average_power_w);
    output << ",\n" << indentation << "  \"sampled_apower_energy_j\": ";
    write_optional(output, summary.sampled_apower_energy_j);
    output << ",\n" << indentation << "  \"sampled_average_power_w\": ";
    write_optional(output, summary.sampled_average_power_w);
    output << "\n" << indentation << '}';
}

void write_wall_power(
    std::ostream& output,
    const joule::metrics::ShellyMeasurementResult& wall) {
    output << "{\n"
           << "    \"host\": " << json_escape(wall.host) << ",\n"
           << "    \"interface\": " << json_escape(wall.interface) << ",\n"
           << "    \"port\": " << wall.port << ",\n"
           << "    \"sample_rate_ms\": " << wall.sample_rate_ms << ",\n"
           << "    \"timeout_ms\": " << wall.timeout_ms << ",\n"
           << "    \"attempts\": " << wall.attempts << ",\n"
           << "    \"device_id\": ";
    if (wall.device_id.empty()) {
        output << "null";
    } else {
        output << json_escape(wall.device_id);
    }
    output << ",\n    \"expected_device_id\": ";
    if (wall.expected_device_id) {
        output << json_escape(*wall.expected_device_id);
    } else {
        output << "null";
    }
    output << ",\n    \"device_id_match\": ";
    if (wall.device_id_match) {
        output << (*wall.device_id_match ? "true" : "false");
    } else {
        output << "null";
    }
    output << ",\n"
           << "    \"energy_source\": \"aenergy.total\",\n"
           << "    \"raw_trace\": " << json_escape(wall.raw_trace_path.string()) << ",\n"
           << "    \"baseline\": ";
    write_shelly_window(output, wall.baseline, "    ");
    output << ",\n    \"workload\": ";
    write_shelly_window(output, wall.workload, "    ");
    output << ",\n    \"dynamic_energy_j\": ";
    write_optional(output, wall.dynamic_energy_j);
    output << ",\n    \"dynamic_sampled_apower_energy_j\": ";
    write_optional(output, wall.dynamic_sampled_apower_energy_j);
    output << "\n  }";
}

[[nodiscard]] std::string result_json(
    const CliOptions& options,
    const joule::metrics::MeasurementResult& result) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(9);
    output << "{\n"
           << "  \"schema_version\": 2,\n"
           << "  \"command\": [";
    for (std::size_t index = 0; index < options.command.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << json_escape(options.command[index]);
    }
    output << "],\n"
           << "  \"command_exit_code\": " << result.command_exit_code << ",\n"
           << "  \"command_wall_time_ms\": " << result.command_wall_time_ms << ",\n"
           << "  \"sample_rate_ms\": " << options.measurement.sample_rate_ms << ",\n"
           << "  \"cooperative_boundary\": "
           << (options.measurement.cooperative_boundary ? "true" : "false") << ",\n"
           << "  \"cooperative_timeout_ms\": "
           << options.measurement.cooperative_timeout_ms << ",\n"
           << "  \"baseline\": ";
    write_summary(output, result.baseline, "  ");
    output << ",\n  \"workload\": ";
    write_summary(output, result.workload, "  ");
    output << ",\n"
           << "  \"dynamic_energy_j\": {\n"
           << "    \"cpu\": ";
    write_optional(output, result.dynamic_cpu_energy_j);
    output << ",\n    \"gpu\": ";
    write_optional(output, result.dynamic_gpu_energy_j);
    output << ",\n    \"ane\": ";
    write_optional(output, result.dynamic_ane_energy_j);
    output << ",\n    \"soc\": ";
    write_optional(output, result.dynamic_soc_energy_j);
    output << ",\n    \"total\": ";
    write_optional(output, result.dynamic_total_energy_j);
    output << "\n  },\n";
    if (result.wall_power) {
        output << "  \"wall_power\": ";
        write_wall_power(output, *result.wall_power);
        output << ",\n";
    }
    output << "  \"measurement_window_error_ms\": "
           << result.workload.sampled_time_s * 1'000.0 - result.command_wall_time_ms << ",\n"
           << "  \"raw_trace\": " << json_escape(result.raw_trace_path.string()) << ",\n"
           << "  \"powermetrics_energy_is_estimated\": true\n"
           << "}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        const auto result = joule::metrics::measure_command(
            options.measurement, options.command);
        const auto json = result_json(options, result);

        if (options.output_path == "-") {
            std::cout << json;
        } else {
            const auto absolute_output = std::filesystem::absolute(options.output_path);
            if (!absolute_output.parent_path().empty()) {
                std::filesystem::create_directories(absolute_output.parent_path());
            }
            std::ofstream output(absolute_output);
            if (!output) {
                throw std::runtime_error("could not create result file: " + absolute_output.string());
            }
            output << json;
            std::cout << absolute_output.string() << '\n';
        }
        return result.command_exit_code;
    } catch (const std::exception& error) {
        std::cerr << "joule-measure: " << error.what() << '\n';
        return 1;
    }
}
