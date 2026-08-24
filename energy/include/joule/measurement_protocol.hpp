#pragma once

namespace joule::measurement_protocol {

// joule-measure maps cooperative child pipes to these descriptors. A benchmark
// that does not inherit all three descriptors runs normally without a handshake.
inline constexpr int ready_fd = 198;
inline constexpr int start_fd = 199;
inline constexpr int done_fd = 200;

}  // namespace joule::measurement_protocol
