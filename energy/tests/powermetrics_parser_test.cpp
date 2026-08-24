#include "joule/metrics.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void expect_near(double actual, double expected, const char* label) {
    if (std::abs(actual - expected) > 1e-12) {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

double require(const std::optional<double>& value, const char* label) {
    if (!value) {
        std::cerr << label << ": missing value\n";
        std::exit(1);
    }
    return *value;
}

}  // namespace

int main() {
    const std::string document_one = R"PLIST(<?xml version="1.0"?>
<plist version="1.0"><dict>
<key>elapsed_ns</key><integer>100000000</integer>
<key>gpu_power</key><dict><key>gpu_power</key><real>9999</real></dict>
<key>processor</key><dict>
  <key>cpu_power</key><real>1000</real>
  <key>gpu_power</key><real>200</real>
  <key>ane_power</key><real>0</real>
  <key>combined_power</key><real>1200</real>
</dict></dict></plist>)PLIST";

    const std::string document_two = R"PLIST(<?xml version="1.0"?>
<plist version="1.0"><dict>
<key>elapsed_ns</key><integer>200000000</integer>
<key>processor</key><dict>
  <key>cpu_power</key><real>500</real>
  <key>gpu_power</key><real>100</real>
  <key>ane_power</key><real>0</real>
  <key>combined_power</key><real>600</real>
</dict></dict></plist>)PLIST";

    std::string trace = document_one;
    trace.push_back('\0');
    trace += document_two;
    trace.push_back('\0');

    const auto samples = joule::metrics::parse_powermetrics_plist_trace(trace);
    if (samples.size() != 2) {
        std::cerr << "expected 2 samples, got " << samples.size() << '\n';
        return 1;
    }
    expect_near(require(samples.front().gpu_power_mw, "first GPU power"), 200.0, "GPU scope");

    const auto summary = joule::metrics::summarize_power_samples(samples, 0, samples.size());
    if (summary.sample_count != 2) {
        std::cerr << "expected 2 summarized samples\n";
        return 1;
    }
    if (summary.cpu_sample_count != 2 || summary.gpu_sample_count != 2 ||
        summary.ane_sample_count != 2 || summary.total_sample_count != 2) {
        std::cerr << "expected complete per-rail sample coverage\n";
        return 1;
    }
    expect_near(summary.sampled_time_s, 0.3, "sampled time");
    expect_near(require(summary.cpu_energy_j, "CPU energy"), 0.2, "CPU energy");
    expect_near(require(summary.gpu_energy_j, "GPU energy"), 0.04, "GPU energy");
    expect_near(require(summary.soc_energy_j, "SoC energy"), 0.24, "SoC energy");
    expect_near(require(summary.total_energy_j, "total energy"), 0.24, "total energy alias");
    expect_near(require(summary.average_soc_power_w, "SoC power"), 0.8, "SoC power");
    expect_near(
        require(summary.average_total_power_w, "total power"),
        0.8,
        "total power alias");
    return 0;
}
