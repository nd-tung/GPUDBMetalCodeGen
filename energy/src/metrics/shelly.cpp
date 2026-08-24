#include "joule/shelly.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace joule::metrics {
namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

class Socket {
public:
    explicit Socket(int descriptor = -1) : descriptor_(descriptor) {}
    ~Socket() {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
    }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    Socket(Socket&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            if (descriptor_ >= 0) {
                ::close(descriptor_);
            }
            descriptor_ = std::exchange(other.descriptor_, -1);
        }
        return *this;
    }

    [[nodiscard]] int get() const { return descriptor_; }
    [[nodiscard]] explicit operator bool() const { return descriptor_ >= 0; }

private:
    int descriptor_;
};

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
    return output.str();
}

[[nodiscard]] std::optional<std::size_t> find_json_value(
    std::string_view json,
    std::string_view key) {
    const std::string marker = '"' + std::string(key) + '"';
    std::size_t cursor = 0;
    while ((cursor = json.find(marker, cursor)) != std::string_view::npos) {
        auto colon = cursor + marker.size();
        while (colon < json.size() && std::isspace(static_cast<unsigned char>(json[colon]))) {
            ++colon;
        }
        if (colon >= json.size() || json[colon] != ':') {
            cursor += marker.size();
            continue;
        }
        ++colon;
        while (colon < json.size() && std::isspace(static_cast<unsigned char>(json[colon]))) {
            ++colon;
        }
        return colon;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<double> parse_json_number_at(
    std::string_view json,
    std::size_t position) {
    if (position >= json.size() || json.substr(position, 4) == "null") {
        return std::nullopt;
    }
    const std::string owned(json.substr(position));
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtod(owned.c_str(), &end);
    if (errno != 0 || end == owned.c_str() || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] std::optional<double> find_json_number(
    std::string_view json,
    std::string_view key) {
    const auto position = find_json_value(json, key);
    return position ? parse_json_number_at(json, *position) : std::nullopt;
}

[[nodiscard]] std::optional<std::string> parse_json_string_at(
    std::string_view json,
    std::size_t position) {
    if (position >= json.size() || json[position] != '"') {
        return std::nullopt;
    }
    std::string result;
    for (std::size_t cursor = position + 1; cursor < json.size(); ++cursor) {
        const char character = json[cursor];
        if (character == '"') {
            return result;
        }
        if (character != '\\') {
            result.push_back(character);
            continue;
        }
        if (++cursor >= json.size()) {
            return std::nullopt;
        }
        switch (json[cursor]) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> find_json_object(
    std::string_view json,
    std::string_view key) {
    const auto position = find_json_value(json, key);
    if (!position || *position >= json.size() || json[*position] != '{') {
        return std::nullopt;
    }
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t cursor = *position; cursor < json.size(); ++cursor) {
        const char character = json[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                in_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_string = true;
        } else if (character == '{') {
            ++depth;
        } else if (character == '}') {
            if (--depth == 0) {
                return json.substr(*position, cursor - *position + 1);
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool is_complete_json_object(std::string_view json) {
    if (json.size() < 2 || json.front() != '{') {
        return false;
    }
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t cursor = 0; cursor < json.size(); ++cursor) {
        const char character = json[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                in_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_string = true;
        } else if (character == '{') {
            ++depth;
        } else if (character == '}') {
            if (depth == 0 || --depth == 0) {
                return depth == 0 && cursor + 1 == json.size();
            }
        }
    }
    return false;
}

[[nodiscard]] int remaining_timeout_ms(TimePoint deadline) {
    const auto remaining = deadline - Clock::now();
    if (remaining <= Clock::duration::zero()) {
        return 0;
    }
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    return static_cast<int>(std::min<std::int64_t>(
        std::numeric_limits<int>::max(), std::max<std::int64_t>(1, milliseconds.count() + 1)));
}

void wait_for_socket(int descriptor, short events, TimePoint deadline) {
    while (true) {
        pollfd descriptor_state{descriptor, events, 0};
        const int timeout = remaining_timeout_ms(deadline);
        if (timeout == 0) {
            throw std::runtime_error("Shelly HTTP request timed out");
        }
        const int result = ::poll(&descriptor_state, 1, timeout);
        if (result > 0) {
            if ((descriptor_state.revents & (POLLERR | POLLNVAL)) != 0) {
                throw std::runtime_error("Shelly HTTP socket failed");
            }
            return;
        }
        if (result == 0) {
            throw std::runtime_error("Shelly HTTP request timed out");
        }
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "Shelly HTTP poll failed");
        }
    }
}

[[nodiscard]] unsigned int resolve_interface_index(std::string_view interface) {
#if defined(__APPLE__)
    errno = 0;
    const auto index = ::if_nametoindex(std::string(interface).c_str());
    if (index == 0) {
        const int error = errno == 0 ? ENXIO : errno;
        throw std::system_error(
            error,
            std::generic_category(),
            "could not resolve Shelly network interface '" + std::string(interface) + "'");
    }
    return index;
#else
    static_cast<void>(interface);
    throw std::runtime_error(
        "Shelly network-interface binding is supported only on macOS");
#endif
}

void bind_socket_to_interface(
    int descriptor,
    int address_family,
    unsigned int interface_index,
    std::string_view interface) {
#if defined(__APPLE__)
    int protocol_level = 0;
    int option = 0;
    if (address_family == AF_INET) {
        protocol_level = IPPROTO_IP;
        option = IP_BOUND_IF;
    } else if (address_family == AF_INET6) {
        protocol_level = IPPROTO_IPV6;
        option = IPV6_BOUND_IF;
    } else {
        throw std::runtime_error(
            "cannot bind Shelly socket for unsupported address family " +
            std::to_string(address_family));
    }
    if (::setsockopt(
            descriptor,
            protocol_level,
            option,
            &interface_index,
            sizeof(interface_index)) != 0) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "could not bind Shelly socket to network interface '" +
                std::string(interface) + "'");
    }
#else
    static_cast<void>(descriptor);
    static_cast<void>(address_family);
    static_cast<void>(interface_index);
    static_cast<void>(interface);
    throw std::runtime_error(
        "Shelly network-interface binding is supported only on macOS");
#endif
}

[[nodiscard]] Socket connect_socket(
    const std::string& host,
    std::uint16_t port,
    const std::string& interface,
    TimePoint deadline) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* addresses = nullptr;
    const auto service = std::to_string(port);
    const int lookup = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (lookup != 0) {
        throw std::runtime_error(
            "could not resolve Shelly host '" + host + "': " + gai_strerror(lookup));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> address_list(addresses, freeaddrinfo);

    const auto interface_index = interface.empty()
        ? std::optional<unsigned int>{}
        : std::optional<unsigned int>{resolve_interface_index(interface)};

    std::string last_error = "connection failed";
    for (auto* address = addresses; address != nullptr; address = address->ai_next) {
        Socket socket(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (!socket) {
            last_error = std::strerror(errno);
            continue;
        }
        if (interface_index) {
            bind_socket_to_interface(
                socket.get(), address->ai_family, *interface_index, interface);
        }
#if defined(__APPLE__)
        int no_sigpipe = 1;
        static_cast<void>(::setsockopt(
            socket.get(), SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)));
#endif
        const int original_flags = ::fcntl(socket.get(), F_GETFL, 0);
        if (original_flags < 0 ||
            ::fcntl(socket.get(), F_SETFL, original_flags | O_NONBLOCK) != 0) {
            last_error = std::strerror(errno);
            continue;
        }
        if (::connect(socket.get(), address->ai_addr, address->ai_addrlen) == 0) {
            return socket;
        }
        if (errno != EINPROGRESS) {
            last_error = std::strerror(errno);
            continue;
        }
        try {
            wait_for_socket(socket.get(), POLLOUT, deadline);
        } catch (const std::exception& error) {
            last_error = error.what();
            continue;
        }
        int socket_error = 0;
        socklen_t error_size = sizeof(socket_error);
        if (::getsockopt(
                socket.get(), SOL_SOCKET, SO_ERROR, &socket_error, &error_size) != 0 ||
            socket_error != 0) {
            last_error = std::strerror(socket_error == 0 ? errno : socket_error);
            continue;
        }
        return socket;
    }
    throw std::runtime_error("could not connect to Shelly: " + last_error);
}

void send_all(int descriptor, std::string_view request, TimePoint deadline) {
    std::size_t sent = 0;
    while (sent < request.size()) {
        wait_for_socket(descriptor, POLLOUT, deadline);
#if defined(MSG_NOSIGNAL)
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const auto count = ::send(
            descriptor, request.data() + sent, request.size() - sent, flags);
        if (count > 0) {
            sent += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        throw std::system_error(errno, std::generic_category(), "Shelly HTTP send failed");
    }
}

[[nodiscard]] std::string receive_all(int descriptor, TimePoint deadline) {
    std::string response;
    std::array<char, 4'096> buffer{};
    constexpr std::size_t maximum_response_size = 1U << 20U;
    while (true) {
        wait_for_socket(descriptor, POLLIN, deadline);
        const auto count = ::recv(descriptor, buffer.data(), buffer.size(), 0);
        if (count == 0) {
            break;
        }
        if (count < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "Shelly HTTP receive failed");
        }
        response.append(buffer.data(), static_cast<std::size_t>(count));
        if (response.size() > maximum_response_size) {
            throw std::runtime_error("Shelly HTTP response is unexpectedly large");
        }
    }
    return response;
}

[[nodiscard]] std::string lowercase(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

[[nodiscard]] std::string decode_chunked(std::string_view encoded) {
    std::string decoded;
    std::size_t cursor = 0;
    while (true) {
        const auto line_end = encoded.find("\r\n", cursor);
        if (line_end == std::string_view::npos) {
            throw std::runtime_error("invalid chunked Shelly HTTP response");
        }
        const auto size_text = encoded.substr(cursor, line_end - cursor);
        const auto extension = size_text.find(';');
        const auto hexadecimal = size_text.substr(0, extension);
        std::size_t chunk_size = 0;
        const auto [end, error] = std::from_chars(
            hexadecimal.data(), hexadecimal.data() + hexadecimal.size(), chunk_size, 16);
        if (error != std::errc{} || end != hexadecimal.data() + hexadecimal.size()) {
            throw std::runtime_error("invalid Shelly HTTP chunk size");
        }
        cursor = line_end + 2;
        if (chunk_size == 0) {
            return decoded;
        }
        if (chunk_size > encoded.size() - cursor ||
            encoded.size() - cursor - chunk_size < 2 ||
            encoded.substr(cursor + chunk_size, 2) != "\r\n") {
            throw std::runtime_error("truncated chunked Shelly HTTP response");
        }
        decoded.append(encoded.substr(cursor, chunk_size));
        cursor += chunk_size + 2;
    }
}

[[nodiscard]] std::string http_get(
    const ShellyConfig& config,
    std::string_view path) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
    auto socket = connect_socket(config.host, config.port, config.interface, deadline);
    const std::string request =
        "GET " + std::string(path) + " HTTP/1.1\r\nHost: " + config.host +
        "\r\nAccept: application/json\r\nConnection: close\r\n"
        "User-Agent: joule-measure/1\r\n\r\n";
    send_all(socket.get(), request, deadline);
    const auto response = receive_all(socket.get(), deadline);

    const auto header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        throw std::runtime_error("invalid Shelly HTTP response");
    }
    const auto status_end = response.find("\r\n");
    if (status_end == std::string::npos) {
        throw std::runtime_error("invalid Shelly HTTP status line");
    }
    const auto status_line = std::string_view(response).substr(0, status_end);
    const auto first_space = status_line.find(' ');
    if (first_space == std::string_view::npos || first_space + 4 > status_line.size()) {
        throw std::runtime_error("invalid Shelly HTTP status line");
    }
    int status_code = 0;
    const auto code_text = status_line.substr(first_space + 1, 3);
    const auto [code_end, code_error] = std::from_chars(
        code_text.data(), code_text.data() + code_text.size(), status_code);
    if (code_error != std::errc{} || code_end != code_text.data() + code_text.size()) {
        throw std::runtime_error("invalid Shelly HTTP status code");
    }
    if (status_code != 200) {
        throw std::runtime_error("Shelly HTTP request returned status " + std::to_string(status_code));
    }

    const auto headers = lowercase(
        std::string_view(response).substr(status_end + 2, header_end - status_end - 2));
    const auto body = std::string_view(response).substr(header_end + 4);
    if (headers.find("transfer-encoding: chunked") != std::string::npos) {
        return decode_chunked(body);
    }
    return std::string(body);
}

void wait_before_retry(const ShellyConfig& config) {
    // Fast failures such as ENETUNREACH otherwise consume every attempt in a
    // few microseconds while Wi-Fi is still changing routes. Keep the backoff
    // below one normal polling interval and bounded so a failed request cannot
    // stall a measurement indefinitely.
    constexpr std::uint32_t maximum_backoff_ms = 500;
    std::this_thread::sleep_for(std::chrono::milliseconds(
        std::min(config.sample_rate_ms, maximum_backoff_ms)));
}

[[nodiscard]] std::int64_t monotonic_ns(TimePoint time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        time.time_since_epoch()).count();
}

void write_optional_json(std::ostream& output, const std::optional<double>& value) {
    if (value) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_raw_trace(
    const std::filesystem::path& path,
    const std::vector<ShellySample>& samples) {
    const auto absolute_path = std::filesystem::absolute(path);
    if (!absolute_path.parent_path().empty()) {
        std::filesystem::create_directories(absolute_path.parent_path());
    }
    std::ofstream output(absolute_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create Shelly trace: " + absolute_path.string());
    }
    output << std::setprecision(17);
    for (const auto& sample : samples) {
        const auto latency_ms = std::chrono::duration<double, std::milli>(
            sample.response_received_at - sample.request_started_at).count();
        output << "{\"schema_version\":1"
               << ",\"sequence\":" << sample.sequence
               << ",\"scheduled_monotonic_ns\":" << monotonic_ns(sample.scheduled_at)
               << ",\"request_start_monotonic_ns\":" << monotonic_ns(sample.request_started_at)
               << ",\"response_monotonic_ns\":" << monotonic_ns(sample.response_received_at)
               << ",\"latency_ms\":" << latency_ms
               << ",\"attempt_count\":" << sample.attempt_count
               << ",\"success\":" << (sample.success ? "true" : "false")
               << ",\"apower_w\":";
        write_optional_json(output, sample.status.apower_w);
        output << ",\"voltage_v\":";
        write_optional_json(output, sample.status.voltage_v);
        output << ",\"current_a\":";
        write_optional_json(output, sample.status.current_a);
        output << ",\"aenergy_total_wh\":";
        write_optional_json(output, sample.status.aenergy_total_wh);
        if (!sample.error.empty()) {
            output << ",\"error\":" << json_escape(sample.error);
        }
        output << "}\n";
    }
    if (!output) {
        throw std::runtime_error("could not write Shelly trace: " + absolute_path.string());
    }
}

struct TimedValue {
    TimePoint time;
    double value{};
};

[[nodiscard]] std::optional<double> interpolate(
    const std::vector<TimedValue>& points,
    TimePoint time,
    bool require_nondecreasing) {
    const auto upper = std::lower_bound(
        points.begin(), points.end(), time,
        [](const TimedValue& point, TimePoint requested) { return point.time < requested; });
    if (upper != points.end() && upper->time == time) {
        return upper->value;
    }
    if (upper == points.begin() || upper == points.end()) {
        return std::nullopt;
    }
    const auto lower = std::prev(upper);
    if (require_nondecreasing && upper->value < lower->value) {
        return std::nullopt;
    }
    const auto interval = std::chrono::duration<double>(upper->time - lower->time).count();
    if (interval <= 0.0) {
        return std::nullopt;
    }
    const auto offset = std::chrono::duration<double>(time - lower->time).count();
    return lower->value + (upper->value - lower->value) * (offset / interval);
}

[[nodiscard]] std::optional<double> integrate_apower(
    const std::vector<TimedValue>& points,
    TimePoint begin,
    TimePoint end) {
    const auto begin_power = interpolate(points, begin, false);
    const auto end_power = interpolate(points, end, false);
    if (!begin_power || !end_power) {
        return std::nullopt;
    }
    std::vector<TimedValue> window;
    window.push_back({begin, *begin_power});
    for (const auto& point : points) {
        if (point.time > begin && point.time < end) {
            window.push_back(point);
        }
    }
    window.push_back({end, *end_power});

    double energy_j = 0.0;
    for (std::size_t index = 1; index < window.size(); ++index) {
        const auto elapsed_s = std::chrono::duration<double>(
            window[index].time - window[index - 1].time).count();
        energy_j += (window[index - 1].value + window[index].value) * 0.5 * elapsed_s;
    }
    return energy_j;
}

}  // namespace

std::chrono::steady_clock::time_point ShellySample::measurement_time() const {
    return request_started_at + (response_received_at - request_started_at) / 2;
}

std::optional<ShellySwitchStatus> parse_shelly_switch_status(std::string_view json) {
    while (!json.empty() && std::isspace(static_cast<unsigned char>(json.front()))) {
        json.remove_prefix(1);
    }
    while (!json.empty() && std::isspace(static_cast<unsigned char>(json.back()))) {
        json.remove_suffix(1);
    }
    if (!is_complete_json_object(json)) {
        return std::nullopt;
    }
    const auto switch_id = find_json_number(json, "id");
    if (!switch_id || *switch_id != 0.0) {
        return std::nullopt;
    }
    ShellySwitchStatus status;
    status.apower_w = find_json_number(json, "apower");
    status.voltage_v = find_json_number(json, "voltage");
    status.current_a = find_json_number(json, "current");
    if (const auto energy = find_json_object(json, "aenergy")) {
        status.aenergy_total_wh = find_json_number(*energy, "total");
    }
    if (!status.apower_w || !status.aenergy_total_wh) {
        return std::nullopt;
    }
    return status;
}

std::optional<std::string> parse_shelly_device_id(std::string_view json) {
    const auto position = find_json_value(json, "id");
    return position ? parse_json_string_at(json, *position) : std::nullopt;
}

ShellyWindowSummary summarize_shelly_samples(
    const std::vector<ShellySample>& samples,
    TimePoint begin,
    TimePoint end) {
    if (end < begin) {
        throw std::invalid_argument("Shelly summary end precedes begin");
    }
    ShellyWindowSummary summary;
    summary.sampled_time_s = std::chrono::duration<double>(end - begin).count();
    std::vector<TimedValue> counters;
    std::vector<TimedValue> powers;
    counters.reserve(samples.size());
    powers.reserve(samples.size());

    for (const auto& sample : samples) {
        const auto time = sample.measurement_time();
        if (time >= begin && time <= end) {
            ++summary.request_count;
            if (sample.success) {
                ++summary.sample_count;
            } else {
                ++summary.failure_count;
            }
        }
        if (!sample.success) {
            continue;
        }
        if (sample.status.aenergy_total_wh) {
            counters.push_back({time, *sample.status.aenergy_total_wh});
        }
        if (sample.status.apower_w) {
            powers.push_back({time, *sample.status.apower_w});
        }
    }
    const auto by_time = [](const TimedValue& left, const TimedValue& right) {
        return left.time < right.time;
    };
    std::sort(counters.begin(), counters.end(), by_time);
    std::sort(powers.begin(), powers.end(), by_time);

    if (!counters.empty()) {
        auto first = std::lower_bound(
            counters.begin(), counters.end(), begin,
            [](const TimedValue& point, TimePoint requested) { return point.time < requested; });
        if (first != counters.begin()) {
            --first;
        }
        auto last = std::upper_bound(
            counters.begin(), counters.end(), end,
            [](TimePoint requested, const TimedValue& point) { return requested < point.time; });
        if (last != counters.end()) {
            ++last;
        }
        for (auto current = first; current != last && current != counters.end(); ++current) {
            const auto next = std::next(current);
            if (next != last && next != counters.end() && next->value < current->value) {
                summary.counter_monotonic = false;
                break;
            }
        }
    }

    if (summary.counter_monotonic) {
        summary.counter_start_wh = interpolate(counters, begin, true);
        summary.counter_end_wh = interpolate(counters, end, true);
        if (summary.counter_start_wh && summary.counter_end_wh &&
            *summary.counter_end_wh >= *summary.counter_start_wh) {
            summary.counter_delta_wh = *summary.counter_end_wh - *summary.counter_start_wh;
            summary.energy_j = *summary.counter_delta_wh * 3'600.0;
            if (summary.sampled_time_s > 0.0) {
                summary.average_power_w = *summary.energy_j / summary.sampled_time_s;
            }
        }
    }
    summary.sampled_apower_energy_j = integrate_apower(powers, begin, end);
    if (summary.sampled_apower_energy_j && summary.sampled_time_s > 0.0) {
        summary.sampled_average_power_w =
            *summary.sampled_apower_energy_j / summary.sampled_time_s;
    }
    return summary;
}

struct ShellySampler::Impl {
    explicit Impl(ShellyConfig supplied_config) : config(std::move(supplied_config)) {}

    void run() noexcept {
        try {
            const auto period = std::chrono::milliseconds(config.sample_rate_ms);
            auto next = Clock::now();
            std::uint64_t sequence = 0;
            while (true) {
                {
                    std::unique_lock lock(mutex);
                    if (condition.wait_until(lock, next, [this] { return stopping; })) {
                        break;
                    }
                }

                ShellySample sample;
                sample.sequence = sequence++;
                sample.scheduled_at = next;
                sample.request_started_at = Clock::now();
                std::ostringstream attempt_errors;
                for (std::uint32_t attempt = 0; attempt < config.attempts; ++attempt) {
                    const auto attempt_number = attempt + 1;
                    sample.attempt_count = attempt_number;
                    try {
                        const auto body = http_get(config, "/rpc/Switch.GetStatus?id=0");
                        const auto status = parse_shelly_switch_status(body);
                        if (!status) {
                            throw std::runtime_error(
                                "invalid Shelly Switch.GetStatus response");
                        }
                        sample.status = *status;
                        sample.success = true;
                        break;
                    } catch (const std::exception& error) {
                        if (attempt > 0) {
                            attempt_errors << "; ";
                        }
                        attempt_errors << "attempt " << attempt_number << '/' << config.attempts
                                       << ": " << error.what();
                        if (attempt_number < config.attempts) {
                            wait_before_retry(config);
                        }
                    }
                }
                if (!sample.success) {
                    sample.error = attempt_errors.str();
                }
                sample.response_received_at = Clock::now();
                {
                    std::lock_guard lock(mutex);
                    samples.push_back(std::move(sample));
                }
                condition.notify_all();

                next += period;
                if (next < Clock::now()) {
                    next = Clock::now();
                }
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            thread_error = std::current_exception();
            condition.notify_all();
        }
    }

    [[nodiscard]] bool has_usable_sample_after(std::optional<TimePoint> boundary) const {
        return std::any_of(samples.begin(), samples.end(), [&](const ShellySample& sample) {
            return sample.success && sample.status.apower_w && sample.status.aenergy_total_wh &&
                (!boundary || sample.measurement_time() > *boundary);
        });
    }

    ShellyConfig config;
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<ShellySample> samples;
    std::thread worker;
    std::exception_ptr thread_error;
    std::string device_id;
    bool started{};
    bool stopping{};
    bool stopped{};
    bool trace_written{};
};

ShellySampler::ShellySampler(ShellyConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    if (impl_->config.host.empty()) {
        throw std::invalid_argument("Shelly host must not be empty");
    }
    if (impl_->config.port == 0) {
        throw std::invalid_argument("Shelly port must be greater than zero");
    }
    if (impl_->config.sample_rate_ms == 0 || impl_->config.timeout_ms == 0 ||
        impl_->config.attempts == 0) {
        throw std::invalid_argument(
            "Shelly sample rate, timeout, and attempts must be greater than zero");
    }
    if (impl_->config.raw_trace_path.empty()) {
        throw std::invalid_argument("Shelly raw trace path must not be empty");
    }
    if (!impl_->config.interface.empty()) {
        static_cast<void>(resolve_interface_index(impl_->config.interface));
    }
    impl_->config.raw_trace_path = std::filesystem::absolute(impl_->config.raw_trace_path);
}

ShellySampler::~ShellySampler() {
    try {
        static_cast<void>(stop());
    } catch (...) {
    }
}

void ShellySampler::start() {
    if (impl_->started) {
        throw std::logic_error("Shelly sampler has already been started");
    }

    // Device identity is a one-time gate before the baseline. Give transient
    // Wi-Fi route loss a wider recovery window than an individual scheduled
    // power sample, which may safely be recorded as failed and retried on the
    // next schedule.
    constexpr std::uint32_t minimum_device_attempts = 10;
    const auto device_attempts =
        std::max(impl_->config.attempts, minimum_device_attempts);
    std::ostringstream device_errors;
    for (std::uint32_t attempt = 0; attempt < device_attempts; ++attempt) {
        const auto attempt_number = attempt + 1;
        try {
            const auto body = http_get(impl_->config, "/rpc/Shelly.GetDeviceInfo");
            const auto id = parse_shelly_device_id(body);
            if (!id || id->empty()) {
                throw std::runtime_error("Shelly.GetDeviceInfo response has no device id");
            }
            impl_->device_id = *id;
            break;
        } catch (const std::exception& error) {
            if (attempt > 0) {
                device_errors << "; ";
            }
            device_errors << "attempt " << attempt_number << '/' << device_attempts
                          << ": " << error.what();
            if (attempt_number < device_attempts) {
                wait_before_retry(impl_->config);
            }
        }
    }
    if (impl_->config.expected_device_id) {
        if (impl_->device_id.empty()) {
            throw std::runtime_error(
                "could not validate Shelly device id: " + device_errors.str());
        }
        if (impl_->device_id != *impl_->config.expected_device_id) {
            throw std::runtime_error(
                "Shelly device id mismatch: expected '" + *impl_->config.expected_device_id +
                "', got '" + impl_->device_id + "'");
        }
    }

    impl_->started = true;
    impl_->worker = std::thread([implementation = impl_.get()] { implementation->run(); });
}

bool ShellySampler::wait_for_first_success(std::chrono::milliseconds timeout) {
    std::unique_lock lock(impl_->mutex);
    return impl_->condition.wait_for(lock, timeout, [this] {
        return impl_->thread_error || impl_->has_usable_sample_after(std::nullopt);
    }) && !impl_->thread_error && impl_->has_usable_sample_after(std::nullopt);
}

bool ShellySampler::wait_for_success_after(
    TimePoint boundary,
    std::chrono::milliseconds timeout) {
    std::unique_lock lock(impl_->mutex);
    return impl_->condition.wait_for(lock, timeout, [this, boundary] {
        return impl_->thread_error || impl_->has_usable_sample_after(boundary);
    }) && !impl_->thread_error && impl_->has_usable_sample_after(boundary);
}

ShellyCollection ShellySampler::stop() {
    if (!impl_->started) {
        return {impl_->device_id, {}, impl_->config.raw_trace_path};
    }
    {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->stopped) {
            impl_->stopping = true;
        }
    }
    impl_->condition.notify_all();
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
    impl_->stopped = true;

    std::vector<ShellySample> samples;
    std::exception_ptr error;
    {
        std::lock_guard lock(impl_->mutex);
        samples = impl_->samples;
        error = impl_->thread_error;
    }
    if (!impl_->trace_written) {
        write_raw_trace(impl_->config.raw_trace_path, samples);
        impl_->trace_written = true;
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return {impl_->device_id, std::move(samples), impl_->config.raw_trace_path};
}

}  // namespace joule::metrics
