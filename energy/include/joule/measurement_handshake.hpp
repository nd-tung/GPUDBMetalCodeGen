#pragma once

#include "joule/measurement_protocol.hpp"

#include <cerrno>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

namespace joule {
namespace detail {

inline void write_protocol_byte(int descriptor, char value) {
    while (::write(descriptor, &value, 1) < 0) {
        if (errno != EINTR) {
            throw std::system_error(
                errno, std::generic_category(), "measurement handshake write failed");
        }
    }
}

inline void read_protocol_byte(int descriptor) {
    char value = 0;
    while (true) {
        const auto count = ::read(descriptor, &value, 1);
        if (count == 1) {
            return;
        }
        if (count == 0) {
            throw std::runtime_error("measurement controller closed the handshake");
        }
        if (errno != EINTR) {
            throw std::system_error(
                errno, std::generic_category(), "measurement handshake read failed");
        }
    }
}

}  // namespace detail

class MeasurementHandshake {
public:
    MeasurementHandshake() {
        const bool has_ready = ::fcntl(measurement_protocol::ready_fd, F_GETFD) >= 0;
        const bool has_start = ::fcntl(measurement_protocol::start_fd, F_GETFD) >= 0;
        const bool has_done = ::fcntl(measurement_protocol::done_fd, F_GETFD) >= 0;
        if (has_ready != has_start || has_ready != has_done) {
            throw std::runtime_error("incomplete cooperative measurement descriptors");
        }
        active_ = has_ready;
    }

    void ready_and_wait() {
        if (!active_) {
            return;
        }
        detail::write_protocol_byte(measurement_protocol::ready_fd, 'R');
        ::close(measurement_protocol::ready_fd);
        detail::read_protocol_byte(measurement_protocol::start_fd);
        ::close(measurement_protocol::start_fd);
    }

    void complete() {
        if (!active_) {
            return;
        }
        detail::write_protocol_byte(measurement_protocol::done_fd, 'D');
        ::close(measurement_protocol::done_fd);
        active_ = false;
    }

private:
    bool active_{};
};

}  // namespace joule
