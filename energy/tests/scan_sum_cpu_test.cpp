#include "joule/operators/cpu/scan_sum.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::int32_t> make_signed_input() {
    std::vector<std::int32_t> input(100'003);
    std::uint32_t state = 0x243f6a88U;
    for (auto& value : input) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<std::int32_t>(state);
    }
    input[0] = INT32_MIN;
    input[1] = INT32_MAX;
    return input;
}

}  // namespace

int main() {
    const auto input = make_signed_input();
    const std::span<const std::int32_t> view(input);
    const auto reference = joule::operators::cpu::scan_sum_i32(view);

    const std::array kernels{
        joule::operators::cpu::ScanSumKernel::scalar,
        joule::operators::cpu::ScanSumKernel::parallel,
        joule::operators::cpu::ScanSumKernel::parallel_simd};
    // Exercise both P-core-sized and full heterogeneous pools. The odd row count
    // also verifies that dynamic aligned chunks cover the tail exactly once.
    const std::array<std::uint32_t, 6> thread_counts{1, 2, 4, 8, 24, 32};

    for (const auto kernel : kernels) {
        for (const auto thread_count : thread_counts) {
            joule::operators::cpu::ScanSumConfig config;
            config.kernel = kernel;
            config.thread_count = thread_count;
            joule::operators::cpu::ScanSum scan(view, config);
            for (int repetition = 0; repetition < 20; ++repetition) {
                const auto result = scan.execute();
                assert(result.sum == reference);
                assert(result.host_time_ms >= 0.0);
                assert(result.compute_time_ms >= 0.0);
            }
        }
    }

    std::cout << "scan_sum_cpu_test: ok\n";
    return 0;
}
