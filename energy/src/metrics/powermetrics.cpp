#include "joule/metrics.hpp"
#include "joule/measurement_protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <spawn.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace joule::metrics {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::string_view trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

[[nodiscard]] std::optional<double> parse_double(std::string_view value) {
    const std::string owned(trim(value));
    if (owned.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(parsed)) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::optional<std::uint64_t> parse_uint64(std::string_view value) {
    value = trim(value);
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::optional<std::string_view> extract_dict(
    std::string_view document,
    std::string_view key) {
    const std::string marker = "<key>" + std::string(key) + "</key>";
    const auto key_position = document.find(marker);
    if (key_position == std::string_view::npos) {
        return std::nullopt;
    }

    const auto root_open = document.find("<dict>", key_position + marker.size());
    if (root_open == std::string_view::npos) {
        return std::nullopt;
    }

    std::size_t cursor = root_open;
    std::size_t depth = 0;
    while (cursor < document.size()) {
        const auto next_open = document.find("<dict>", cursor);
        const auto next_close = document.find("</dict>", cursor);
        if (next_close == std::string_view::npos) {
            return std::nullopt;
        }
        if (next_open != std::string_view::npos && next_open < next_close) {
            ++depth;
            cursor = next_open + std::string_view("<dict>").size();
            continue;
        }
        if (depth == 0) {
            return std::nullopt;
        }
        --depth;
        if (depth == 0) {
            const auto content_begin = root_open + std::string_view("<dict>").size();
            return document.substr(content_begin, next_close - content_begin);
        }
        cursor = next_close + std::string_view("</dict>").size();
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> extract_numeric_text(
    std::string_view scope,
    std::string_view key) {
    const std::string marker = "<key>" + std::string(key) + "</key>";
    const auto key_position = scope.find(marker);
    if (key_position == std::string_view::npos) {
        return std::nullopt;
    }

    const auto value_begin = key_position + marker.size();
    const auto next_key = scope.find("<key>", value_begin);
    for (const std::string_view tag : {std::string_view{"real"}, std::string_view{"integer"}}) {
        const std::string open_tag = "<" + std::string(tag) + ">";
        const std::string close_tag = "</" + std::string(tag) + ">";
        const auto open = scope.find(open_tag, value_begin);
        if (open == std::string_view::npos ||
            (next_key != std::string_view::npos && open > next_key)) {
            continue;
        }
        const auto content_begin = open + open_tag.size();
        const auto close = scope.find(close_tag, content_begin);
        if (close != std::string_view::npos) {
            return scope.substr(content_begin, close - content_begin);
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<double> extract_double(
    std::string_view scope,
    std::string_view key) {
    const auto text = extract_numeric_text(scope, key);
    return text ? parse_double(*text) : std::nullopt;
}

[[nodiscard]] std::optional<std::uint64_t> extract_uint64(
    std::string_view scope,
    std::string_view key) {
    const auto text = extract_numeric_text(scope, key);
    return text ? parse_uint64(*text) : std::nullopt;
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open powermetrics trace: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] std::size_t count_complete_documents(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return 0;
    }
    const std::string trace{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    constexpr std::string_view closing_tag = "</plist>";
    std::size_t count = 0;
    std::size_t cursor = 0;
    while ((cursor = trace.find(closing_tag, cursor)) != std::string::npos) {
        ++count;
        cursor += closing_tag.size();
    }
    return count;
}

[[nodiscard]] int exit_code_from_status(int status) {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
}

[[nodiscard]] pid_t spawn_process(
    const std::vector<std::string>& arguments,
    bool create_process_group,
    bool quiet = false,
    const struct SpawnHandshake* handshake = nullptr);

struct SpawnHandshake {
    int ready_write{};
    int start_read{};
    int done_write{};
    std::array<int, 6> pipe_descriptors{};
};

[[nodiscard]] pid_t spawn_process(
    const std::vector<std::string>& arguments,
    bool create_process_group,
    bool quiet,
    const SpawnHandshake* handshake) {
    if (arguments.empty()) {
        throw std::invalid_argument("cannot spawn an empty command");
    }

    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const auto& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    if (create_process_group) {
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attributes, 0);
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_t* action_pointer = nullptr;
    if (quiet || handshake != nullptr) {
        posix_spawn_file_actions_init(&actions);
        action_pointer = &actions;
    }
    if (quiet) {
        posix_spawn_file_actions_addopen(
            &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    }
    if (handshake != nullptr) {
        posix_spawn_file_actions_adddup2(
            &actions, handshake->ready_write, measurement_protocol::ready_fd);
        posix_spawn_file_actions_adddup2(
            &actions, handshake->start_read, measurement_protocol::start_fd);
        posix_spawn_file_actions_adddup2(
            &actions, handshake->done_write, measurement_protocol::done_fd);
        for (const auto descriptor : handshake->pipe_descriptors) {
            posix_spawn_file_actions_addclose(&actions, descriptor);
        }
    }

    pid_t process_id = -1;
    const int error = posix_spawnp(
        &process_id,
        arguments.front().c_str(),
        action_pointer,
        &attributes,
        argv.data(),
        environ);

    if (action_pointer != nullptr) {
        posix_spawn_file_actions_destroy(&actions);
    }
    posix_spawnattr_destroy(&attributes);
    if (error != 0) {
        throw std::runtime_error(
            "could not start '" + arguments.front() + "': " + std::strerror(error));
    }
    return process_id;
}

class ProtocolPipes {
public:
    ProtocolPipes() {
        try {
            create_pipe(ready_);
            create_pipe(start_);
            create_pipe(done_);
        } catch (...) {
            close_all();
            throw;
        }
    }

    ~ProtocolPipes() {
        close_all();
    }

    [[nodiscard]] SpawnHandshake spawn_handshake() const {
        return SpawnHandshake{
            ready_[1], start_[0], done_[1],
            {ready_[0], ready_[1], start_[0], start_[1], done_[0], done_[1]}};
    }

    void close_child_ends() {
        close_descriptor(ready_[1]);
        close_descriptor(start_[0]);
        close_descriptor(done_[1]);
    }

    void wait_until_ready(std::chrono::milliseconds timeout) {
        read_byte(
            ready_[0],
            "benchmark exited before reporting ready",
            "benchmark timed out before reporting ready",
            timeout);
        close_descriptor(ready_[0]);
    }

    void start_workload() {
        write_byte(start_[1], 'S');
        close_descriptor(start_[1]);
    }

    void wait_until_done(std::chrono::milliseconds timeout) {
        read_byte(
            done_[0],
            "benchmark exited before completing its timed region",
            "benchmark timed region exceeded its deadline",
            timeout);
        close_descriptor(done_[0]);
    }

private:
    static void create_pipe(std::array<int, 2>& descriptors) {
        if (pipe(descriptors.data()) != 0) {
            throw std::system_error(errno, std::generic_category(), "pipe failed");
        }
        for (const auto descriptor : descriptors) {
            const auto flags = fcntl(descriptor, F_GETFD);
            if (flags < 0 || fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
                throw std::system_error(
                    errno, std::generic_category(), "could not configure handshake pipe");
            }
        }
    }

    static void close_descriptor(int& descriptor) noexcept {
        if (descriptor >= 0) {
            close(descriptor);
            descriptor = -1;
        }
    }

    static void read_byte(
        int descriptor,
        const char* eof_message,
        const char* timeout_message,
        std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        char value = 0;
        while (true) {
            if (timeout.count() > 0) {
                const auto remaining = deadline - std::chrono::steady_clock::now();
                if (remaining <= std::chrono::steady_clock::duration::zero()) {
                    throw std::runtime_error(timeout_message);
                }
                const auto remaining_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
                const int poll_timeout = static_cast<int>(std::min<std::int64_t>(
                    std::numeric_limits<int>::max(), std::max<std::int64_t>(1, remaining_ms + 1)));
                pollfd descriptor_state{descriptor, POLLIN, 0};
                const int poll_result = ::poll(&descriptor_state, 1, poll_timeout);
                if (poll_result == 0) {
                    throw std::runtime_error(timeout_message);
                }
                if (poll_result < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw std::system_error(
                        errno, std::generic_category(), "handshake poll failed");
                }
            }
            const auto count = read(descriptor, &value, 1);
            if (count == 1) {
                return;
            }
            if (count == 0) {
                throw std::runtime_error(eof_message);
            }
            if (errno != EINTR) {
                throw std::system_error(
                    errno, std::generic_category(), "handshake read failed");
            }
        }
    }

    static void write_byte(int descriptor, char value) {
        while (write(descriptor, &value, 1) < 0) {
            if (errno != EINTR) {
                throw std::system_error(
                    errno, std::generic_category(), "handshake write failed");
            }
        }
    }

    void close_all() noexcept {
        for (auto* descriptors : {&ready_, &start_, &done_}) {
            close_descriptor((*descriptors)[0]);
            close_descriptor((*descriptors)[1]);
        }
    }

    std::array<int, 2> ready_{-1, -1};
    std::array<int, 2> start_{-1, -1};
    std::array<int, 2> done_{-1, -1};
};

[[nodiscard]] int wait_for_process(pid_t process_id) {
    int status = 0;
    while (waitpid(process_id, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "waitpid failed");
        }
    }
    return exit_code_from_status(status);
}

[[nodiscard]] std::optional<int> poll_process(pid_t process_id) {
    int status = 0;
    const auto result = waitpid(process_id, &status, WNOHANG);
    if (result == 0) {
        return std::nullopt;
    }
    if (result == process_id) {
        return exit_code_from_status(status);
    }
    if (errno == EINTR) {
        return std::nullopt;
    }
    if (errno == ECHILD) {
        return 0;
    }
    throw std::system_error(errno, std::generic_category(), "waitpid failed");
}

[[nodiscard]] std::optional<int> wait_for_process_until(
    pid_t process_id,
    std::chrono::steady_clock::time_point deadline) {
    while (std::chrono::steady_clock::now() < deadline) {
        if (const auto code = poll_process(process_id)) {
            return code;
        }
        std::this_thread::sleep_for(10ms);
    }
    return std::nullopt;
}

void signal_process_group(pid_t process_id, int signal_number) {
    if (kill(-process_id, signal_number) != 0 && errno != ESRCH) {
        if (kill(process_id, signal_number) != 0 && errno != ESRCH) {
            throw std::system_error(errno, std::generic_category(), "could not signal powermetrics");
        }
    }
}

[[nodiscard]] const char* signal_option(int signal_number) {
    switch (signal_number) {
#if defined(SIGINFO)
        case SIGINFO:
            return "-INFO";
#endif
        case SIGINT:
            return "-INT";
        case SIGKILL:
            return "-KILL";
        default:
            throw std::invalid_argument("unsupported sampler signal");
    }
}

void signal_sampler_group(pid_t process_id, int signal_number, bool use_sudo) {
    if (!use_sudo) {
        signal_process_group(process_id, signal_number);
        return;
    }

    const auto signal_process = spawn_process(
        {"/usr/bin/sudo", "-n", "--", "/bin/kill", signal_option(signal_number),
         "--", "-" + std::to_string(process_id)},
        false,
        true);
    if (wait_for_process(signal_process) != 0) {
        throw std::runtime_error("sudo could not signal the privileged powermetrics process");
    }
}

[[nodiscard]] bool process_group_exists(pid_t process_id) {
    if (kill(-process_id, 0) == 0) {
        return true;
    }
    return errno == EPERM;
}

[[nodiscard]] bool wait_for_process_group_exit(
    pid_t process_id,
    std::chrono::steady_clock::time_point deadline) {
    while (std::chrono::steady_clock::now() < deadline) {
        if (!process_group_exists(process_id)) {
            return true;
        }
        std::this_thread::sleep_for(10ms);
    }
    return !process_group_exists(process_id);
}

void stop_sampler(pid_t process_id, bool use_sudo) {
    signal_sampler_group(process_id, SIGINT, use_sudo);
    static_cast<void>(wait_for_process_until(
        process_id, std::chrono::steady_clock::now() + 2s));
    if (wait_for_process_group_exit(process_id, std::chrono::steady_clock::now() + 2s)) {
        return;
    }

    signal_sampler_group(process_id, SIGKILL, use_sudo);
    static_cast<void>(wait_for_process_until(
        process_id, std::chrono::steady_clock::now() + 2s));
    if (!wait_for_process_group_exit(process_id, std::chrono::steady_clock::now() + 2s)) {
        throw std::runtime_error("powermetrics process group did not terminate");
    }
}

void stop_sampler_noexcept(pid_t process_id, bool use_sudo) noexcept {
    try {
        stop_sampler(process_id, use_sudo);
    } catch (...) {
        // Cleanup must not hide the original measurement error.
    }
}

[[nodiscard]] std::size_t wait_for_document_count(
    const std::filesystem::path& path,
    std::size_t minimum_count,
    pid_t sampler_process,
    std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const auto count = count_complete_documents(path);
        if (count >= minimum_count) {
            return count;
        }
        if (const auto code = poll_process(sampler_process)) {
            throw std::runtime_error(
                "powermetrics exited before producing a sample (exit code " +
                std::to_string(*code) + ")");
        }
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error("timed out waiting for a powermetrics sample");
}

[[nodiscard]] std::size_t mark_sample_boundary(
    const std::filesystem::path& path,
    pid_t sampler_process,
    std::uint32_t sample_rate_ms,
    bool use_sudo) {
#if defined(SIGINFO)
    const auto before = count_complete_documents(path);
    signal_sampler_group(sampler_process, SIGINFO, use_sudo);
    const auto timeout = std::chrono::milliseconds(
        std::max<std::uint32_t>(2'000, sample_rate_ms * 4));
    static_cast<void>(wait_for_document_count(
        path, before + 1, sampler_process, timeout));
    std::this_thread::sleep_for(5ms);
    return count_complete_documents(path);
#else
    static_cast<void>(path);
    static_cast<void>(sampler_process);
    static_cast<void>(sample_rate_ms);
    static_cast<void>(use_sudo);
    throw std::runtime_error("SIGINFO is unavailable on this platform");
#endif
}

[[nodiscard]] bool sudo_is_ready() {
    const auto process_id = spawn_process({"/usr/bin/sudo", "-n", "true"}, false, true);
    return wait_for_process(process_id) == 0;
}

class SudoKeepalive {
public:
    SudoKeepalive()
        : thread_([this] {
              std::unique_lock lock(mutex_);
              while (!condition_.wait_for(lock, 60s, [this] { return stopping_; })) {
                  lock.unlock();
                  try {
                      const auto process_id = spawn_process(
                          {"/usr/bin/sudo", "-n", "-v"}, false, true);
                      static_cast<void>(wait_for_process(process_id));
                  } catch (...) {
                      // The final privileged signal reports a useful error if refresh failed.
                  }
                  lock.lock();
              }
          }) {}

    ~SudoKeepalive() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        thread_.join();
    }

    SudoKeepalive(const SudoKeepalive&) = delete;
    SudoKeepalive& operator=(const SudoKeepalive&) = delete;

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_{};
    std::thread thread_;
};

struct RailAccumulator {
    double energy_j{};
    double time_s{};
    std::size_t sample_count{};
    bool has_value{};

    void add(const std::optional<double>& power_mw, std::uint64_t elapsed_ns) {
        if (!power_mw) {
            return;
        }
        const auto elapsed_s = static_cast<double>(elapsed_ns) * 1e-9;
        energy_j += *power_mw * static_cast<double>(elapsed_ns) * 1e-12;
        time_s += elapsed_s;
        ++sample_count;
        has_value = true;
    }

    [[nodiscard]] std::optional<double> energy() const {
        return has_value ? std::optional<double>{energy_j} : std::nullopt;
    }

    [[nodiscard]] std::optional<double> average_power_w() const {
        return has_value && time_s > 0.0
            ? std::optional<double>{energy_j / time_s}
            : std::nullopt;
    }
};

[[nodiscard]] std::optional<double> dynamic_energy(
    const std::optional<double>& workload_energy,
    const std::optional<double>& baseline_power_w,
    double workload_time_s) {
    if (!workload_energy || !baseline_power_w) {
        return std::nullopt;
    }
    return *workload_energy - (*baseline_power_w * workload_time_s);
}

}  // namespace

std::vector<PowerSample> parse_powermetrics_plist_trace(std::string_view trace) {
    std::vector<PowerSample> samples;
    constexpr std::string_view closing_tag = "</plist>";
    std::size_t document_begin = 0;

    while (document_begin < trace.size()) {
        const auto document_end = trace.find(closing_tag, document_begin);
        if (document_end == std::string_view::npos) {
            break;
        }
        const auto document = trace.substr(
            document_begin, document_end + closing_tag.size() - document_begin);
        document_begin = document_end + closing_tag.size();

        const auto elapsed_ns = extract_uint64(document, "elapsed_ns");
        if (!elapsed_ns) {
            continue;
        }

        const auto processor = extract_dict(document, "processor");
        const auto power_scope = processor.value_or(document);
        PowerSample sample;
        sample.elapsed_ns = *elapsed_ns;
        sample.cpu_power_mw = extract_double(power_scope, "cpu_power");
        sample.gpu_power_mw = extract_double(power_scope, "gpu_power");
        sample.ane_power_mw = extract_double(power_scope, "ane_power");
        sample.combined_power_mw = extract_double(power_scope, "combined_power");
        samples.push_back(sample);
    }
    return samples;
}

std::vector<PowerSample> read_powermetrics_plist_trace(
    const std::filesystem::path& path) {
    return parse_powermetrics_plist_trace(read_file(path));
}

PowerSummary summarize_power_samples(
    const std::vector<PowerSample>& samples,
    std::size_t begin,
    std::size_t end) {
    begin = std::min(begin, samples.size());
    end = std::min(std::max(begin, end), samples.size());

    PowerSummary summary;
    RailAccumulator cpu;
    RailAccumulator gpu;
    RailAccumulator ane;
    RailAccumulator soc;

    for (std::size_t index = begin; index < end; ++index) {
        const auto& sample = samples[index];
        if (sample.elapsed_ns == 0) {
            continue;
        }
        ++summary.sample_count;
        summary.sampled_time_s += static_cast<double>(sample.elapsed_ns) * 1e-9;
        cpu.add(sample.cpu_power_mw, sample.elapsed_ns);
        gpu.add(sample.gpu_power_mw, sample.elapsed_ns);
        ane.add(sample.ane_power_mw, sample.elapsed_ns);

        auto combined = sample.combined_power_mw;
        if (!combined && sample.cpu_power_mw && sample.gpu_power_mw && sample.ane_power_mw) {
            combined = *sample.cpu_power_mw + *sample.gpu_power_mw + *sample.ane_power_mw;
        }
        soc.add(combined, sample.elapsed_ns);
    }

    summary.cpu_energy_j = cpu.energy();
    summary.gpu_energy_j = gpu.energy();
    summary.ane_energy_j = ane.energy();
    summary.soc_energy_j = soc.energy();
    summary.cpu_sample_count = cpu.sample_count;
    summary.gpu_sample_count = gpu.sample_count;
    summary.ane_sample_count = ane.sample_count;
    summary.total_sample_count = soc.sample_count;
    summary.total_energy_j = summary.soc_energy_j;
    summary.average_cpu_power_w = cpu.average_power_w();
    summary.average_gpu_power_w = gpu.average_power_w();
    summary.average_ane_power_w = ane.average_power_w();
    summary.average_soc_power_w = soc.average_power_w();
    summary.average_total_power_w = summary.average_soc_power_w;
    return summary;
}

MeasurementResult measure_command(
    const MeasurementConfig& config,
    const std::vector<std::string>& command) {
#if !defined(__APPLE__)
    static_cast<void>(config);
    static_cast<void>(command);
    throw std::runtime_error("powermetrics measurement requires macOS");
#else
    if (command.empty()) {
        throw std::invalid_argument("a benchmark command is required");
    }
    if (config.sample_rate_ms == 0) {
        throw std::invalid_argument("sample rate must be greater than zero");
    }
    if (config.raw_trace_path.empty()) {
        throw std::invalid_argument("a raw trace path is required");
    }

    const bool needs_sudo = geteuid() != 0;
    if (needs_sudo && !config.use_sudo) {
        throw std::runtime_error("powermetrics requires root privileges; remove --no-sudo");
    }
    if (needs_sudo && !sudo_is_ready()) {
        throw std::runtime_error(
            "sudo credentials are not cached; run 'sudo -v' before joule-measure");
    }
    std::unique_ptr<SudoKeepalive> sudo_keepalive;
    if (needs_sudo) {
        sudo_keepalive = std::make_unique<SudoKeepalive>();
    }

    const auto raw_path = std::filesystem::absolute(config.raw_trace_path);
    if (!raw_path.parent_path().empty()) {
        std::filesystem::create_directories(raw_path.parent_path());
    }
    std::error_code remove_error;
    std::filesystem::remove(raw_path, remove_error);

    std::vector<std::string> sampler_command;
    if (needs_sudo) {
        sampler_command = {"/usr/bin/sudo", "-n", "--"};
    }
    sampler_command.insert(
        sampler_command.end(),
        {"/usr/bin/powermetrics",
         "--samplers", "cpu_power,gpu_power",
         "--sample-rate", std::to_string(config.sample_rate_ms),
         "--poweravg", "0",
         "--buffer-size", "1",
         "--format", "plist",
         "--handle-invalid-values",
         "--output-file", raw_path.string()});

    const auto sampler_process = spawn_process(sampler_command, true);
    bool sampler_running = true;
    pid_t command_process = -1;
    bool command_running = false;
    std::unique_ptr<ShellySampler> shelly_sampler;
    try {
        // A freshly released powermetrics sampler can take several seconds to
        // reacquire the hardware service. Keep the startup guard comfortably
        // above one sampling interval so a valid long trial is not abandoned
        // during this one-time handoff.
        const auto startup_timeout = std::chrono::milliseconds(std::max<std::uint64_t>(
            15'000, static_cast<std::uint64_t>(config.sample_rate_ms) * 10));
        const auto initial_count = wait_for_document_count(
            raw_path, 1, sampler_process, startup_timeout);

        auto baseline_begin = initial_count;
        std::optional<std::chrono::steady_clock::time_point> baseline_start_time;
        std::optional<std::chrono::steady_clock::time_point> baseline_end_time;
        if (config.shelly) {
            shelly_sampler = std::make_unique<ShellySampler>(*config.shelly);
            shelly_sampler->start();
            const auto shelly_ready_timeout = std::chrono::milliseconds(
                std::max<std::uint64_t>(
                    5'000,
                    static_cast<std::uint64_t>(config.shelly->timeout_ms) *
                            config.shelly->attempts +
                        static_cast<std::uint64_t>(config.shelly->sample_rate_ms) * 2));
            if (!shelly_sampler->wait_for_first_success(shelly_ready_timeout)) {
                throw std::runtime_error(
                    "timed out waiting for a usable Shelly power sample");
            }
            // Establish a fresh baseline boundary after network/device setup so
            // neither sampler includes Shelly connection setup in the baseline.
            baseline_begin = mark_sample_boundary(
                raw_path, sampler_process, config.sample_rate_ms, needs_sudo);
            baseline_start_time = std::chrono::steady_clock::now();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(config.baseline_ms));
        const auto baseline_end = mark_sample_boundary(
            raw_path, sampler_process, config.sample_rate_ms, needs_sudo);
        if (shelly_sampler) {
            baseline_end_time = std::chrono::steady_clock::now();
            const auto baseline_bracket_timeout = std::chrono::milliseconds(
                std::max<std::uint64_t>(
                    5'000,
                    static_cast<std::uint64_t>(config.shelly->timeout_ms) *
                            config.shelly->attempts +
                        static_cast<std::uint64_t>(config.shelly->sample_rate_ms) * 2));
            if (!shelly_sampler->wait_for_success_after(
                    *baseline_end_time, baseline_bracket_timeout)) {
                throw std::runtime_error(
                    "timed out waiting for a Shelly sample after the baseline boundary");
            }
        }

        std::size_t workload_begin = baseline_end;
        std::size_t workload_end = baseline_end;
        std::chrono::steady_clock::time_point command_start;
        std::chrono::steady_clock::time_point command_end;
        int command_exit_code = 1;

        if (config.cooperative_boundary) {
            ProtocolPipes protocol;
            const auto cooperative_timeout =
                std::chrono::milliseconds(config.cooperative_timeout_ms);
            const auto handshake = protocol.spawn_handshake();
            command_process = spawn_process(command, false, false, &handshake);
            command_running = true;
            protocol.close_child_ends();
            protocol.wait_until_ready(cooperative_timeout);

            // Discard process setup and warm-up samples. The measured window
            // begins only after the prepared child is blocked on the start pipe.
            workload_begin = mark_sample_boundary(
                raw_path, sampler_process, config.sample_rate_ms, needs_sudo);
            command_start = std::chrono::steady_clock::now();
            protocol.start_workload();
            protocol.wait_until_done(cooperative_timeout);
            command_end = std::chrono::steady_clock::now();
            workload_end = mark_sample_boundary(
                raw_path, sampler_process, config.sample_rate_ms, needs_sudo);

            command_exit_code = wait_for_process(command_process);
            command_running = false;
        } else {
            command_start = std::chrono::steady_clock::now();
            command_process = spawn_process(command, false);
            command_running = true;
            command_exit_code = wait_for_process(command_process);
            command_running = false;
            command_end = std::chrono::steady_clock::now();
            workload_end = mark_sample_boundary(
                raw_path, sampler_process, config.sample_rate_ms, needs_sudo);
        }
        stop_sampler(sampler_process, needs_sudo);
        sampler_running = false;

        std::optional<ShellyCollection> shelly_collection;
        if (shelly_sampler) {
            const auto end_sample_timeout = std::chrono::milliseconds(
                std::max<std::uint64_t>(
                    5'000,
                    static_cast<std::uint64_t>(config.shelly->timeout_ms) *
                            config.shelly->attempts +
                        static_cast<std::uint64_t>(config.shelly->sample_rate_ms) * 2));
            if (!shelly_sampler->wait_for_success_after(command_end, end_sample_timeout)) {
                throw std::runtime_error(
                    "timed out waiting for a Shelly sample after the workload boundary");
            }
            shelly_collection = shelly_sampler->stop();
        }

        const auto samples = read_powermetrics_plist_trace(raw_path);
        if (samples.size() < workload_end) {
            throw std::runtime_error("powermetrics trace ended with an incomplete sample");
        }

        MeasurementResult result;
        result.command_exit_code = command_exit_code;
        result.command_wall_time_ms =
            std::chrono::duration<double, std::milli>(command_end - command_start).count();
        result.baseline = summarize_power_samples(samples, baseline_begin, baseline_end);
        result.workload = summarize_power_samples(samples, workload_begin, workload_end);
        result.dynamic_cpu_energy_j = dynamic_energy(
            result.workload.cpu_energy_j,
            result.baseline.average_cpu_power_w,
            result.workload.sampled_time_s);
        result.dynamic_gpu_energy_j = dynamic_energy(
            result.workload.gpu_energy_j,
            result.baseline.average_gpu_power_w,
            result.workload.sampled_time_s);
        result.dynamic_ane_energy_j = dynamic_energy(
            result.workload.ane_energy_j,
            result.baseline.average_ane_power_w,
            result.workload.sampled_time_s);
        result.dynamic_soc_energy_j = dynamic_energy(
            result.workload.soc_energy_j,
            result.baseline.average_soc_power_w,
            result.workload.sampled_time_s);
        result.dynamic_total_energy_j = result.dynamic_soc_energy_j;
        if (shelly_collection && baseline_start_time && baseline_end_time) {
            ShellyMeasurementResult wall;
            wall.host = config.shelly->host;
            wall.interface = config.shelly->interface;
            wall.port = config.shelly->port;
            wall.sample_rate_ms = config.shelly->sample_rate_ms;
            wall.timeout_ms = config.shelly->timeout_ms;
            wall.attempts = config.shelly->attempts;
            wall.device_id = shelly_collection->device_id;
            wall.expected_device_id = config.shelly->expected_device_id;
            if (wall.expected_device_id) {
                wall.device_id_match = wall.device_id == *wall.expected_device_id;
            }
            wall.raw_trace_path = shelly_collection->raw_trace_path;
            wall.baseline = summarize_shelly_samples(
                shelly_collection->samples, *baseline_start_time, *baseline_end_time);
            wall.workload = summarize_shelly_samples(
                shelly_collection->samples, command_start, command_end);
            wall.dynamic_energy_j = dynamic_energy(
                wall.workload.energy_j,
                wall.baseline.average_power_w,
                wall.workload.sampled_time_s);
            wall.dynamic_sampled_apower_energy_j = dynamic_energy(
                wall.workload.sampled_apower_energy_j,
                wall.baseline.sampled_average_power_w,
                wall.workload.sampled_time_s);
            result.wall_power = std::move(wall);
        }
        result.raw_trace_path = raw_path;
        return result;
    } catch (...) {
        if (command_running) {
            if (kill(command_process, SIGTERM) == 0 || errno == ESRCH) {
                if (!wait_for_process_until(
                        command_process, std::chrono::steady_clock::now() + 2s)) {
                    static_cast<void>(kill(command_process, SIGKILL));
                    static_cast<void>(wait_for_process_until(
                        command_process, std::chrono::steady_clock::now() + 2s));
                }
            }
        }
        if (sampler_running) {
            stop_sampler_noexcept(sampler_process, needs_sudo);
        }
        if (shelly_sampler) {
            try {
                static_cast<void>(shelly_sampler->stop());
            } catch (...) {
                // Preserve the primary measurement failure.
            }
        }
        throw;
    }
#endif
}

}  // namespace joule::metrics
