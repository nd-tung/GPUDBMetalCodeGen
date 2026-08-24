#include "joule/shelly.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

double require(const std::optional<double>& value, const char* message) {
    expect(value.has_value(), message);
    return *value;
}

void expect_near(double actual, double expected, const char* message) {
    if (std::abs(actual - expected) > 1e-8) {
        std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

Clock::time_point at(double seconds) {
    return Clock::time_point{} +
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

joule::metrics::ShellySample sample(
    std::uint64_t sequence,
    double seconds,
    bool success,
    std::optional<double> counter_wh) {
    joule::metrics::ShellySample result;
    result.sequence = sequence;
    result.scheduled_at = at(seconds);
    result.request_started_at = at(seconds);
    result.response_received_at = at(seconds);
    result.attempt_count = 1;
    result.success = success;
    if (success) {
        result.status.apower_w = 100.0;
        result.status.voltage_v = 230.0;
        result.status.current_a = 0.5;
        result.status.aenergy_total_wh = counter_wh;
    } else {
        result.error = "timeout";
    }
    return result;
}

std::string http_response(std::string_view status, std::string_view body) {
    return "HTTP/1.1 " + std::string(status) + "\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" +
        std::string(body);
}

void run_retry_test() {
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    expect(listener >= 0, "could not create retry-test listener");

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    expect(
        ::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
        "could not bind retry-test listener");
    socklen_t address_size = sizeof(address);
    expect(
        ::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_size) == 0,
        "could not read retry-test listener port");
    expect(::listen(listener, 4) == 0, "could not listen for retry test");

    const std::vector<std::string> responses{
        http_response("503 Service Unavailable", ""),
        http_response("200 OK", R"({"id":"retry-test-device"})"),
        http_response("503 Service Unavailable", ""),
        http_response(
            "200 OK",
            R"({"id":0,"apower":42.5,"voltage":230,"current":0.2,"aenergy":{"total":12.5}})"),
    };
    std::exception_ptr server_error;
    std::thread server([&] {
        try {
            for (const auto& response : responses) {
                pollfd ready{listener, POLLIN, 0};
                if (::poll(&ready, 1, 3'000) != 1) {
                    throw std::runtime_error("timed out waiting for retry-test request");
                }
                const int client = ::accept(listener, nullptr, nullptr);
                if (client < 0) {
                    throw std::runtime_error("could not accept retry-test request");
                }
                std::string request;
                char buffer[1'024];
                while (request.find("\r\n\r\n") == std::string::npos) {
                    const auto count = ::recv(client, buffer, sizeof(buffer), 0);
                    if (count <= 0) {
                        ::close(client);
                        throw std::runtime_error("could not receive retry-test request");
                    }
                    request.append(buffer, static_cast<std::size_t>(count));
                }
                std::size_t sent = 0;
                while (sent < response.size()) {
                    const auto count = ::send(
                        client, response.data() + sent, response.size() - sent, 0);
                    if (count <= 0) {
                        ::close(client);
                        throw std::runtime_error("could not send retry-test response");
                    }
                    sent += static_cast<std::size_t>(count);
                }
                ::close(client);
            }
        } catch (...) {
            server_error = std::current_exception();
        }
    });

    const auto trace_path = std::filesystem::temp_directory_path() /
        ("joule-shelly-retry-test-" + std::to_string(::getpid()) + ".jsonl");
    std::filesystem::remove(trace_path);
    joule::metrics::ShellyConfig config;
    config.host = "127.0.0.1";
#if defined(__APPLE__)
    config.interface = "lo0";
#endif
    config.port = ntohs(address.sin_port);
    config.sample_rate_ms = 60'000;
    config.timeout_ms = 500;
    config.attempts = 3;
    config.raw_trace_path = trace_path;
    config.expected_device_id = "retry-test-device";

    joule::metrics::ShellySampler sampler(config);
    sampler.start();
    expect(
        sampler.wait_for_first_success(std::chrono::seconds(3)),
        "retry did not produce a usable Shelly sample");
    const auto collection = sampler.stop();
    server.join();
    ::close(listener);
    if (server_error) {
        std::rethrow_exception(server_error);
    }

    expect(collection.samples.size() == 1, "retry emitted more than one logical sample");
    expect(collection.samples.front().success, "retried Shelly sample was not successful");
    expect(collection.samples.front().attempt_count == 2, "retry attempt count was not recorded");
    std::ifstream trace(trace_path);
    const std::string raw(
        (std::istreambuf_iterator<char>(trace)), std::istreambuf_iterator<char>());
    expect(
        raw.find("\"attempt_count\":2") != std::string::npos,
        "retry attempt count is missing from raw trace");
    std::filesystem::remove(trace_path);
}

void run_invalid_interface_test() {
#if defined(__APPLE__)
    joule::metrics::ShellyConfig config;
    config.host = "127.0.0.1";
    config.interface = "joule-no-such";
    config.raw_trace_path = std::filesystem::temp_directory_path() /
        "joule-shelly-invalid-interface-test.jsonl";

    bool rejected = false;
    try {
        joule::metrics::ShellySampler sampler(config);
    } catch (const std::system_error&) {
        rejected = true;
    }
    expect(rejected, "nonexistent Shelly interface was accepted");
#endif
}

}  // namespace

int main() {
    expect(joule::metrics::ShellyConfig{}.attempts == 3, "incorrect default retry count");
    expect(
        joule::metrics::ShellyConfig{}.interface.empty(),
        "Shelly interface should be unspecified by default");

    const std::string status_json = R"JSON({
      "id": 0,
      "output": true,
      "apower": 17.1,
      "voltage": 231.7,
      "current": 0.151,
      "unrelated": {"total": 999},
      "aenergy": {"total": 493.046, "by_minute": [1, 2, 3]}
    })JSON";
    const auto status = joule::metrics::parse_shelly_switch_status(status_json);
    expect(status.has_value(), "valid Switch.GetStatus was rejected");
    expect_near(require(status->apower_w, "missing apower"), 17.1, "apower");
    expect_near(require(status->voltage_v, "missing voltage"), 231.7, "voltage");
    expect_near(require(status->current_a, "missing current"), 0.151, "current");
    expect_near(
        require(status->aenergy_total_wh, "missing aenergy total"),
        493.046,
        "aenergy total");
    expect(
        !joule::metrics::parse_shelly_switch_status("{\"id\":1,\"apower\":2}"),
        "non-switch-0 response was accepted");
    expect(
        !joule::metrics::parse_shelly_switch_status(
            "{\"id\":0,\"apower\":2,\"aenergy\":{\"total\":1}"),
        "truncated Switch.GetStatus response was accepted");
    expect(
        !joule::metrics::parse_shelly_switch_status("{\"id\":0,\"apower\":2}"),
        "Switch.GetStatus response without energy was accepted");

    const auto device_id = joule::metrics::parse_shelly_device_id(
        "{\"name\":\"test\",\"id\":\"shellyplugmg3-08927259b5ec\"}");
    expect(
        device_id && *device_id == "shellyplugmg3-08927259b5ec",
        "device id parser failed");

    constexpr double watts_to_wh_per_second = 100.0 / 3'600.0;
    std::vector<joule::metrics::ShellySample> samples;
    for (std::uint64_t index = 0; index < 4; ++index) {
        samples.push_back(sample(
            index,
            static_cast<double>(index),
            true,
            10.0 + static_cast<double>(index) * watts_to_wh_per_second));
    }
    samples[2] = sample(2, 2.0, false, std::nullopt);

    const auto summary = joule::metrics::summarize_shelly_samples(
        samples, at(0.5), at(2.5));
    expect(summary.request_count == 2, "incorrect request count");
    expect(summary.sample_count == 1, "incorrect successful sample count");
    expect(summary.failure_count == 1, "incorrect failure count");
    expect(summary.counter_monotonic, "monotonic counter was rejected");
    expect_near(summary.sampled_time_s, 2.0, "sampled time");
    expect_near(require(summary.energy_j, "missing counter energy"), 200.0, "counter energy");
    expect_near(
        require(summary.average_power_w, "missing counter average power"),
        100.0,
        "counter average power");
    expect_near(
        require(summary.sampled_apower_energy_j, "missing apower energy"),
        200.0,
        "apower energy");

    samples[3].status.aenergy_total_wh = 9.0;
    const auto reset = joule::metrics::summarize_shelly_samples(
        samples, at(0.5), at(2.5));
    expect(!reset.counter_monotonic, "counter reset was not detected");
    expect(!reset.energy_j, "counter-reset energy should be unavailable");
    run_invalid_interface_test();
    run_retry_test();
    return 0;
}
