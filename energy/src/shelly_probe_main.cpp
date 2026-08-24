#include "joule/shelly.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

struct CliOptions {
    joule::metrics::ShellyConfig shelly;
    std::uint32_t duration_ms{5'000};
};

[[nodiscard]] std::filesystem::path default_trace_path() {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return std::filesystem::path("results/raw") /
        ("shelly-probe-" + std::to_string(now_ms) + ".jsonl");
}

[[nodiscard]] std::uint32_t parse_uint32(
    std::string_view value,
    std::string_view option) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    std::size_t parsed_characters = 0;
    const auto number = std::stoull(std::string(value), &parsed_characters);
    if (parsed_characters != value.size() ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return static_cast<std::uint32_t>(number);
}

void print_help(std::ostream& output) {
    output
        << "Usage: joule-shelly-probe --host HOST [options]\n\n"
        << "Options:\n"
        << "  --host HOST          Shelly host or IP address (required)\n"
        << "  --interface NAME     bind HTTP traffic to this macOS interface\n"
        << "  --port N             Shelly HTTP port (default: 80)\n"
        << "  --sample-rate-ms N   polling interval (default: 1000)\n"
        << "  --timeout-ms N       per-request timeout (default: 2000)\n"
        << "  --attempts N         attempts per scheduled sample (default: 3)\n"
        << "  --device-id ID       require this Shelly device id\n"
        << "  --duration-ms N      sampling time after first success (default: 5000)\n"
        << "  --raw PATH           raw JSONL trace path\n"
        << "  -h, --help           show this help\n";
}

[[nodiscard]] CliOptions parse_arguments(int argc, char** argv) {
    CliOptions options;
    options.shelly.raw_trace_path = default_trace_path();
    bool interface_supplied = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "-h" || argument == "--help") {
            print_help(std::cout);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(argument));
        }
        const std::string_view value(argv[++index]);
        if (argument == "--host") {
            options.shelly.host = value;
        } else if (argument == "--interface") {
            interface_supplied = true;
            options.shelly.interface = value;
        } else if (argument == "--port") {
            const auto port = parse_uint32(value, argument);
            if (port == 0 || port > std::numeric_limits<std::uint16_t>::max()) {
                throw std::invalid_argument("invalid value for " + std::string(argument));
            }
            options.shelly.port = static_cast<std::uint16_t>(port);
        } else if (argument == "--sample-rate-ms") {
            options.shelly.sample_rate_ms = parse_uint32(value, argument);
        } else if (argument == "--timeout-ms") {
            options.shelly.timeout_ms = parse_uint32(value, argument);
        } else if (argument == "--attempts") {
            options.shelly.attempts = parse_uint32(value, argument);
        } else if (argument == "--device-id") {
            options.shelly.expected_device_id = std::string(value);
        } else if (argument == "--duration-ms") {
            options.duration_ms = parse_uint32(value, argument);
        } else if (argument == "--raw") {
            options.shelly.raw_trace_path = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    if (options.shelly.host.empty()) {
        throw std::invalid_argument("--host is required");
    }
    if (options.shelly.sample_rate_ms == 0 || options.shelly.timeout_ms == 0 ||
        options.shelly.attempts == 0) {
        throw std::invalid_argument(
            "sample rate, timeout, and attempts must be greater than zero");
    }
    if (interface_supplied && options.shelly.interface.empty()) {
        throw std::invalid_argument("--interface must not be empty");
    }
    if (options.shelly.expected_device_id && options.shelly.expected_device_id->empty()) {
        throw std::invalid_argument("--device-id must not be empty");
    }
    if (options.shelly.raw_trace_path.empty()) {
        throw std::invalid_argument("--raw must not be empty");
    }
    return options;
}

[[nodiscard]] std::chrono::milliseconds first_success_timeout(
    const joule::metrics::ShellyConfig& config) {
    constexpr long double maximum_backoff_ms = 500.0L;
    const auto backoff_ms = std::min<long double>(config.sample_rate_ms, maximum_backoff_ms);
    const auto wait_ms =
        static_cast<long double>(config.timeout_ms) * config.attempts +
        backoff_ms * (config.attempts - 1U) + config.sample_rate_ms;
    const auto bounded_ms = std::min<long double>(
        std::max<long double>(5'000.0L, wait_ms),
        static_cast<long double>(std::chrono::milliseconds::max().count()));
    return std::chrono::milliseconds(
        static_cast<std::chrono::milliseconds::rep>(bounded_ms));
}

[[nodiscard]] std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    output << "\\u00";
                    constexpr char digits[] = "0123456789abcdef";
                    output << digits[(static_cast<unsigned char>(character) >> 4U) & 0x0fU]
                           << digits[static_cast<unsigned char>(character) & 0x0fU];
                } else {
                    output << character;
                }
        }
    }
    output << '"';
    return output.str();
}

void print_summary(
    const joule::metrics::ShellyCollection& collection,
    bool first_success) {
    const auto successes = static_cast<std::size_t>(std::count_if(
        collection.samples.begin(),
        collection.samples.end(),
        [](const joule::metrics::ShellySample& sample) { return sample.success; }));
    const auto total = collection.samples.size();
    std::cout << "{\"device_id\":" << json_string(collection.device_id)
              << ",\"first_success\":" << (first_success ? "true" : "false")
              << ",\"total_samples\":" << total
              << ",\"success_samples\":" << successes
              << ",\"failure_samples\":" << (total - successes)
              << ",\"raw_path\":"
              << json_string(collection.raw_trace_path.string()) << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        joule::metrics::ShellySampler sampler(options.shelly);
        sampler.start();
        const bool first_success = sampler.wait_for_first_success(
            first_success_timeout(options.shelly));
        if (first_success) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.duration_ms));
        }
        const auto collection = sampler.stop();
        print_summary(collection, first_success);
        return first_success ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "joule-shelly-probe: " << error.what() << '\n';
        return 1;
    }
}
