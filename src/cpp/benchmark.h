#pragma once

#include <string>
#include <vector>

struct BenchmarkSummary {
    std::string method = "UNKNOWN";
    long long n_steps_total = 0;
    long long n_force_evals_est = 0;
    int n_windows = 0;
    int n_hills = 0;
    unsigned int seed = 0;
    std::string hyperparameters = "";
    std::string fes_path = "none";
};

inline long long estimate_force_evals(long long n_steps_total) {
    return n_steps_total;
}

inline std::vector<std::string> benchmark_summary_lines(const BenchmarkSummary& summary) {
    return {
        "method=" + summary.method,
        "n_steps_total=" + std::to_string(summary.n_steps_total),
        "n_force_evals_est=" + std::to_string(summary.n_force_evals_est),
        "n_windows=" + std::to_string(summary.n_windows),
        "n_hills=" + std::to_string(summary.n_hills),
        "seed=" + std::to_string(summary.seed),
        "hyperparameters=" + summary.hyperparameters,
        "fes_out=" + summary.fes_path
    };
}

