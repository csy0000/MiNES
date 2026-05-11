#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "eq_neq.h"
#include "fes.h"

struct UmbrellaSamplingWindow {
    int id = -1;
    Vec2 center{};
    double k = 0.0;
    int n_steps = 0;
    int n_frames = 0;
    unsigned int seed = 0;
    std::string traj_file = "";
    std::vector<Vec2> samples;
};

struct UmbrellaSamplingConfig {
    int steps_per_window = 10000;
    int total_steps = 0;
    int output_samples = 200;
    double k = 1.0;
    double spacing = 0.5;
    double grid_dx = 0.0;
    double grid_dy = 0.0;
    unsigned int base_seed = 20260322u;
    std::string direction = "forward";
    Vec2 start{};
    Vec2 end{};
    std::string out_dir = "";
    std::string summary_out = "us_windows.csv";
    std::string fes_out = "us_fes.csv";
};

struct UmbrellaSamplingSummary {
    int n_windows_run = 0;
    int n_steps_total = 0;
    int n_force_evals_est = 0;
    std::string summary_path = "";
    std::string fes_path = "";
    std::vector<UmbrellaSamplingWindow> windows;
};

inline std::vector<int> build_us_window_steps(size_t n_windows,
                                              int steps_per_window,
                                              int total_steps) {
    std::vector<int> steps(n_windows, steps_per_window);
    if (n_windows == 0) {
        return steps;
    }
    if (total_steps <= 0) {
        return steps;
    }
    const int n = static_cast<int>(n_windows);
    const int base = total_steps / n;
    const int remainder = total_steps % n;
    steps.assign(n_windows, base);
    for (int i = 0; i < remainder; ++i) {
        steps[static_cast<size_t>(i)] += 1;
    }
    return steps;
}

inline Vec2 line_point(Vec2 start, Vec2 end, double lamb) {
    return {start.x + lamb * (end.x - start.x), start.y + lamb * (end.y - start.y)};
}

inline std::vector<Vec2> build_us_line_centers(const SimConfig& cfg,
                                               Vec2 start,
                                               Vec2 end,
                                               double spacing) {
    std::vector<Vec2> centers;
    if (spacing <= 0.0) {
        return centers;
    }

    double distance = 0.0;
    if (cfg.one_dimension == 'x') {
        distance = std::fabs(end.x - start.x);
    } else if (cfg.one_dimension == 'y') {
        distance = std::fabs(end.y - start.y);
    } else {
        distance = vec_norm(vec_sub(end, start));
    }

    if (distance <= 0.0) {
        centers.push_back(start);
        return centers;
    }

    int n_intervals = static_cast<int>(std::llround(distance / spacing));
    if (n_intervals <= 0) {
        n_intervals = 1;
    }
    const double reconstructed = static_cast<double>(n_intervals) * spacing;
    if (std::fabs(reconstructed - distance) > 1e-6 * std::max(1.0, distance)) {
        n_intervals = static_cast<int>(std::ceil(distance / spacing));
    }

    centers.reserve(static_cast<size_t>(n_intervals + 1));
    for (int i = 0; i <= n_intervals; ++i) {
        const double lamb = static_cast<double>(i) / static_cast<double>(n_intervals);
        centers.push_back(line_point(start, end, lamb));
    }
    return centers;
}

inline std::vector<Vec2> order_us_centers(const std::vector<Vec2>& centers,
                                          const std::string& direction) {
    std::vector<Vec2> ordered = centers;
    if (direction == "backward") {
        std::reverse(ordered.begin(), ordered.end());
        return ordered;
    }
    if (direction != "bidirectional") {
        return ordered;
    }

    std::vector<Vec2> alternating;
    alternating.reserve(ordered.size());
    int left = 0;
    int right = static_cast<int>(ordered.size()) - 1;
    while (left <= right) {
        alternating.push_back(ordered[static_cast<size_t>(left)]);
        if (left == right) {
            break;
        }
        alternating.push_back(ordered[static_cast<size_t>(right)]);
        ++left;
        --right;
    }
    return alternating;
}

inline void write_us_windows_csv(const std::string& out_path,
                                 const std::vector<UmbrellaSamplingWindow>& windows) {
    std::ofstream out(out_path);
    out << "window_id,center_x,center_y,k,n_steps,n_frames,seed,traj_file\n";
    for (const auto& window : windows) {
        out << window.id << ","
            << std::setprecision(10) << window.center.x << "," << window.center.y << ","
            << window.k << "," << window.n_steps << "," << window.n_frames << ","
            << window.seed << "," << window.traj_file << "\n";
    }
}

inline void write_us_fes_csv(const SimConfig& cfg,
                             const UmbrellaSamplingConfig& us_cfg,
                             const std::vector<UmbrellaSamplingWindow>& windows,
                             const std::string& out_path) {
    std::vector<Vec2> all_samples;
    std::vector<Vec2> centers;
    for (const auto& window : windows) {
        centers.push_back(window.center);
        all_samples.insert(all_samples.end(), window.samples.begin(), window.samples.end());
    }

    if (cfg.one_dimension == 'x' || cfg.one_dimension == 'y') {
        const GridSpec1D grid = auto_grid_1d(cfg, all_samples, centers, us_cfg.spacing, us_cfg.grid_dx);
        std::vector<double> prob_sum(grid.nx, 0.0);
        std::vector<double> count_sum(grid.nx, 0.0);
        std::vector<double> uncertainty(grid.nx, std::numeric_limits<double>::infinity());
        std::vector<int> contributions(grid.nx, 0);
        const double beta = 1.0 / cfg.kT;

        for (const auto& window : windows) {
            if (window.samples.empty()) {
                continue;
            }
            BiasHarmonic bias{window.k, window.center};
            std::vector<int> bin_ids(window.samples.size(), -1);
            std::vector<double> log_weights(window.samples.size(), -std::numeric_limits<double>::infinity());
            double max_log_w = -std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < window.samples.size(); ++i) {
                const int bin = grid_index_1d(grid, active_coordinate(cfg, window.samples[i]));
                if (bin < 0) {
                    continue;
                }
                const double log_w = beta * harmonic_bias_energy(cfg, bias, window.samples[i]);
                bin_ids[i] = bin;
                log_weights[i] = log_w;
                max_log_w = std::max(max_log_w, log_w);
            }
            if (!std::isfinite(max_log_w)) {
                continue;
            }

            std::vector<double> local_hist(grid.nx, 0.0);
            double local_norm = 0.0;
            for (size_t i = 0; i < window.samples.size(); ++i) {
                const int bin = bin_ids[i];
                if (bin < 0) {
                    continue;
                }
                const double weight = std::exp(log_weights[i] - max_log_w);
                local_hist[bin] += weight;
                local_norm += weight;
                count_sum[bin] += 1.0;
            }
            if (local_norm <= 0.0) {
                continue;
            }
            for (int ix = 0; ix < grid.nx; ++ix) {
                if (local_hist[ix] <= 0.0) {
                    continue;
                }
                prob_sum[ix] += local_hist[ix] / local_norm;
                contributions[ix] += 1;
            }
        }

        std::vector<double> free_energy(grid.nx, std::numeric_limits<double>::infinity());
        for (int ix = 0; ix < grid.nx; ++ix) {
            if (contributions[ix] <= 0 || prob_sum[ix] <= 0.0) {
                continue;
            }
            const double prob = prob_sum[ix] / static_cast<double>(contributions[ix]);
            free_energy[ix] = -cfg.kT * std::log(prob);
            if (count_sum[ix] > 0.0) {
                uncertainty[ix] = 1.0 / std::sqrt(count_sum[ix]);
            }
        }
        shift_free_energy_min_zero(free_energy);
        write_pmf_1d_csv(out_path, grid, free_energy, count_sum, uncertainty);
        return;
    }

    const GridSpec2D grid = auto_grid_2d(all_samples, centers, us_cfg.spacing, us_cfg.grid_dx, us_cfg.grid_dy);
    const int nxy = grid.nx * grid.ny;
    std::vector<double> prob_sum(nxy, 0.0);
    std::vector<double> count_sum(nxy, 0.0);
    std::vector<int> contributions(nxy, 0);
    const double beta = 1.0 / cfg.kT;

    for (const auto& window : windows) {
        if (window.samples.empty()) {
            continue;
        }
        BiasHarmonic bias{window.k, window.center};
        std::vector<int> flat_ids(window.samples.size(), -1);
        std::vector<double> log_weights(window.samples.size(), -std::numeric_limits<double>::infinity());
        double max_log_w = -std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < window.samples.size(); ++i) {
            const int ix = grid_index_x(grid, window.samples[i].x);
            const int iy = grid_index_y(grid, window.samples[i].y);
            if (ix < 0 || iy < 0) {
                continue;
            }
            const int idx = iy * grid.nx + ix;
            const double log_w = beta * harmonic_bias_energy(cfg, bias, window.samples[i]);
            flat_ids[i] = idx;
            log_weights[i] = log_w;
            max_log_w = std::max(max_log_w, log_w);
        }
        if (!std::isfinite(max_log_w)) {
            continue;
        }

        std::vector<double> local_hist(nxy, 0.0);
        double local_norm = 0.0;
        for (size_t i = 0; i < window.samples.size(); ++i) {
            const int idx = flat_ids[i];
            if (idx < 0) {
                continue;
            }
            const double weight = std::exp(log_weights[i] - max_log_w);
            local_hist[idx] += weight;
            local_norm += weight;
            count_sum[idx] += 1.0;
        }
        if (local_norm <= 0.0) {
            continue;
        }
        for (int idx = 0; idx < nxy; ++idx) {
            if (local_hist[idx] <= 0.0) {
                continue;
            }
            prob_sum[idx] += local_hist[idx] / local_norm;
            contributions[idx] += 1;
        }
    }

    std::vector<double> free_energy(nxy, std::numeric_limits<double>::infinity());
    for (int idx = 0; idx < nxy; ++idx) {
        if (contributions[idx] <= 0 || prob_sum[idx] <= 0.0) {
            continue;
        }
        const double prob = prob_sum[idx] / static_cast<double>(contributions[idx]);
        free_energy[idx] = -cfg.kT * std::log(prob);
    }
    shift_free_energy_min_zero(free_energy);
    write_fes_2d_csv(out_path, grid, free_energy, count_sum);
}

inline UmbrellaSamplingSummary run_umbrella_sampling(const SimConfig& cfg,
                                                     const UmbrellaSamplingConfig& us_cfg) {
    UmbrellaSamplingSummary summary{};
    std::filesystem::create_directories(us_cfg.out_dir);

    const std::vector<Vec2> centers = order_us_centers(
        build_us_line_centers(cfg, us_cfg.start, us_cfg.end, us_cfg.spacing),
        us_cfg.direction
    );
    const std::vector<int> window_steps = build_us_window_steps(
        centers.size(), us_cfg.steps_per_window, us_cfg.total_steps);

    summary.windows.reserve(centers.size());
    for (size_t idx = 0; idx < centers.size(); ++idx) {
        UmbrellaSamplingWindow window{};
        window.id = static_cast<int>(idx);
        window.center = centers[idx];
        window.k = us_cfg.k;
        window.seed = us_cfg.base_seed + static_cast<unsigned int>(idx);

        SimConfig run_cfg = cfg;
        run_cfg.n_eq_steps = window_steps[idx];
        run_cfg.eq_output_samples = us_cfg.output_samples;

        std::ostringstream traj_name;
        traj_name << us_cfg.out_dir << "/us_window_" << window.id << ".csv";
        window.traj_file = traj_name.str();
        window.samples = run_eq_harmonic_write(run_cfg, window.traj_file, window.center, window.k, window.seed);
        window.n_steps = run_cfg.n_eq_steps;
        window.n_frames = static_cast<int>(window.samples.size());

        summary.n_windows_run += 1;
        summary.n_steps_total += run_cfg.n_eq_steps;
        summary.n_force_evals_est += static_cast<int>(estimate_force_evals(run_cfg.n_eq_steps));
        summary.windows.push_back(window);
    }

    summary.summary_path = us_cfg.summary_out;
    summary.fes_path = us_cfg.fes_out;
    ensure_parent_dir(summary.summary_path);
    ensure_parent_dir(summary.fes_path);
    write_us_windows_csv(summary.summary_path, summary.windows);
    write_us_fes_csv(cfg, us_cfg, summary.windows, summary.fes_path);
    return summary;
}
