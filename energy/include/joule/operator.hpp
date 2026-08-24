#pragma once

#include <cstdint>

namespace joule {

enum class Backend {
    cpu,
    gpu,
    cpu_gpu,
};

struct OperatorConfig {
    Backend backend{Backend::cpu};
    std::uint64_t input_size{};
    double selectivity{1.0};
    std::uint32_t tuple_width{4};
    std::uint64_t group_cardinality{1};
    double skew{};
    bool materialize_output{};
    double cpu_fraction{1.0};
};

struct RunResult {
    double wall_time_ms{};
    double cpu_time_ms{};
    double gpu_time_ms{};
    double cpu_energy_j{};
    double gpu_energy_j{};
    double soc_energy_j{};
    std::uint64_t output_count{};
    std::uint64_t checksum{};
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual void prepare(const OperatorConfig& config) = 0;
    [[nodiscard]] virtual RunResult execute(const OperatorConfig& config) = 0;
    [[nodiscard]] virtual bool verify() const = 0;
};

}  // namespace joule
