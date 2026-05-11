#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "bias.h"
#include "path.h"
#include "potential.h"
#include "sim_config.h"

struct LangevinRng {
    std::mt19937 rng;
    std::normal_distribution<double> normal{0.0, 1.0};

    explicit LangevinRng(unsigned int seed) : rng(seed) {}

    double randn() { return normal(rng); }
};

inline Vec2 sample_maxwell(const SimConfig& cfg, LangevinRng& rng) {
    const double sigma = std::sqrt(cfg.kT / cfg.mass);
    Vec2 vel{sigma * rng.randn(), sigma * rng.randn()};
    if (cfg.one_dimension == 'x') {
        vel.y = 0.0;
    } else if (cfg.one_dimension == 'y') {
        vel.x = 0.0;
    }
    return vel;
}

template <typename BiasType>
inline Vec2 total_grad(const SimConfig& cfg, const Vec2& pos, const BiasType& bias) {
    Vec2 g = potential_grad(cfg, pos.x, pos.y);
    Vec2 gb = bias.grad(pos.x, pos.y);
    if (cfg.one_dimension == 'x') {
        g.y = 0.0;
        gb.y = 0.0;
    } else if (cfg.one_dimension == 'y') {
        g.x = 0.0;
        gb.x = 0.0;
    }
    g.x += gb.x;
    g.y += gb.y;
    return g;
}

template <typename BiasType>
inline void langevin_vv_step(const SimConfig& cfg, const BiasType& bias,
                             Vec2& pos, Vec2& vel, LangevinRng& rng) {
    const double half_dt = 0.5 * cfg.dt;
    const double inv_m = 1.0 / cfg.mass;
    const double c = std::exp(-cfg.gamma * cfg.dt);
    const double sigma = std::sqrt((1.0 - c * c) * cfg.kT * inv_m);

    Vec2 g = total_grad(cfg, pos, bias);
    Vec2 F{-g.x, -g.y};
    if (cfg.one_dimension != 'y') {
        vel.x += half_dt * F.x * inv_m;
        pos.x += half_dt * vel.x;
        vel.x = c * vel.x + sigma * rng.randn();
        pos.x += half_dt * vel.x;
    } else {
        vel.x = 0.0;
    }
    if (cfg.one_dimension != 'x') {
        vel.y += half_dt * F.y * inv_m;
        pos.y += half_dt * vel.y;
        vel.y = c * vel.y + sigma * rng.randn();
        pos.y += half_dt * vel.y;
    } else {
        vel.y = 0.0;
    }

    g = total_grad(cfg, pos, bias);
    F.x = -g.x;
    F.y = -g.y;
    if (cfg.one_dimension != 'y') {
        vel.x += half_dt * F.x * inv_m;
    } else {
        vel.x = 0.0;
    }
    if (cfg.one_dimension != 'x') {
        vel.y += half_dt * F.y * inv_m;
    } else {
        vel.y = 0.0;
    }
}

inline void ensure_parent_dir(const std::string& path_str) {
    std::filesystem::path path(path_str);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

struct MetaRunSummary {
    Vec2 final_pos{};
    double final_base_u = 0.0;
    double final_meta_u = 0.0;
    int hills_deposited = 0;
};

inline double harmonic_bias_energy(const SimConfig& cfg, const BiasHarmonic& bias, const Vec2& pos) {
    if (cfg.one_dimension == 'x') {
        const double dx = pos.x - bias.center.x;
        return 0.5 * bias.k * dx * dx;
    }
    if (cfg.one_dimension == 'y') {
        const double dy = pos.y - bias.center.y;
        return 0.5 * bias.k * dy * dy;
    }
    return bias.U(pos.x, pos.y);
}

inline std::vector<Vec2> run_eq_harmonic_write(const SimConfig& cfg, const std::string& out_path,
                                               Vec2 center, double k, unsigned int seed,
                                               bool have_start = false, Vec2 start = {}) {
    ensure_parent_dir(out_path);
    LangevinRng rng(seed);
    Vec2 pos = have_start ? start : center;
    Vec2 vel{};

    const int target_samples = std::max(1, cfg.eq_output_samples);
    const int stride = std::max(1, cfg.n_eq_steps / target_samples);
    std::vector<Vec2> samples;
    samples.reserve(target_samples);

    std::ofstream out(out_path);
    out << "step,x,y,base_u,bias_u\n";

    BiasHarmonic bias{k, center};

    // Optional energy minimization before EQ sampling
    for (int step = 0; step < cfg.eq_minimize_steps; ++step) {
        Vec2 g = total_grad(cfg, pos, bias);
        pos.x += -cfg.eq_minimize_alpha * g.x;
        pos.y += -cfg.eq_minimize_alpha * g.y;
    }

    vel = sample_maxwell(cfg, rng);

    if (cfg.eq_write_initial && static_cast<int>(samples.size()) < target_samples) {
        const double base_u = potential_U(cfg, pos.x, pos.y);
        const double bias_u = harmonic_bias_energy(cfg, bias, pos);
        out << 0 << "," << std::setprecision(10) << pos.x << "," << pos.y << ","
            << base_u << "," << bias_u << "\n";
        samples.push_back(pos);
    }

    for (int step = 0; step < cfg.n_eq_steps; ++step) {
        langevin_vv_step(cfg, bias, pos, vel, rng);

        if (((step + 1) % stride == 0) && static_cast<int>(samples.size()) < target_samples) {
            samples.push_back(pos);
            const double base_u = potential_U(cfg, pos.x, pos.y);
            const double bias_u = harmonic_bias_energy(cfg, bias, pos);
            out << step << "," << std::setprecision(10) << pos.x << "," << pos.y << ","
                << base_u << "," << bias_u << "\n";
        }
    }

    return samples;
}

inline std::vector<Vec2> read_eq_samples(const std::string& path_csv) {
    std::vector<Vec2> samples;
    std::ifstream in(path_csv);
    if (!in.is_open()) {
        return samples;
    }

    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) {
            first = false;
            if (line.find("x") != std::string::npos && line.find("y") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<double> vals;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                vals.push_back(std::stod(token));
            }
        }
        if (vals.size() >= 3) {
            samples.push_back(Vec2{vals[1], vals[2]});
        } else if (vals.size() >= 2) {
            samples.push_back(Vec2{vals[0], vals[1]});
        }
    }
    return samples;
}

inline MetaRunSummary run_well_tempered_meta_write(const SimConfig& cfg,
                                                   const std::string& traj_out_path,
                                                   const std::string& hills_out_path,
                                                   Vec2 start,
                                                   unsigned int seed) {
    ensure_parent_dir(traj_out_path);
    ensure_parent_dir(hills_out_path);

    LangevinRng rng(seed);
    Vec2 pos = start;
    Vec2 vel = sample_maxwell(cfg, rng);

    BiasWellTemperedMeta bias{};
    bias.initial_height = cfg.meta_initial_height;
    bias.sigma_x = cfg.meta_sigma_x;
    bias.sigma_y = cfg.meta_sigma_y;
    bias.bias_factor = cfg.meta_bias_factor;
    bias.kT = cfg.kT;
    bias.one_dimension = cfg.one_dimension;
    bias.hills.reserve(1 + cfg.n_meta_steps / std::max(1, cfg.meta_deposition_stride));

    std::ofstream traj_out(traj_out_path);
    traj_out << "step,x,y,base_u,meta_u,total_u,n_hills,deposited_hill,hill_height\n";

    std::ofstream hills_out(hills_out_path);
    hills_out << "hill,step,x,y,height,sigma_x,sigma_y,bias_before,bias_after\n";

    auto write_frame = [&](int step, bool deposited, double hill_height) {
        const double base_u = potential_U(cfg, pos.x, pos.y);
        const double meta_u = bias.U(pos.x, pos.y);
        traj_out << step << "," << std::setprecision(10)
                 << pos.x << "," << pos.y << ","
                 << base_u << "," << meta_u << "," << (base_u + meta_u) << ","
                 << bias.size() << "," << (deposited ? 1 : 0) << "," << hill_height << "\n";
    };

    write_frame(0, false, 0.0);

    for (int step = 1; step <= cfg.n_meta_steps; ++step) {
        langevin_vv_step(cfg, bias, pos, vel, rng);

        bool deposited = false;
        double hill_height = 0.0;
        if (cfg.meta_deposition_stride > 0 && (step % cfg.meta_deposition_stride == 0)) {
            const double bias_before = bias.U(pos.x, pos.y);
            hill_height = bias.add_hill(pos.x, pos.y);
            const double bias_after = bias.U(pos.x, pos.y);
            deposited = true;

            hills_out << (bias.size() - 1) << "," << step << "," << std::setprecision(10)
                      << pos.x << "," << pos.y << ","
                      << hill_height << ","
                      << cfg.meta_sigma_x << "," << cfg.meta_sigma_y << ","
                      << bias_before << "," << bias_after << "\n";
        }

        if (cfg.meta_output_stride <= 1 || (step % cfg.meta_output_stride == 0) ||
            deposited || step == cfg.n_meta_steps) {
            write_frame(step, deposited, hill_height);
        }
    }

    MetaRunSummary summary{};
    summary.final_pos = pos;
    summary.final_base_u = potential_U(cfg, pos.x, pos.y);
    summary.final_meta_u = bias.U(pos.x, pos.y);
    summary.hills_deposited = static_cast<int>(bias.size());
    return summary;
}

inline Vec2 pick_start(LangevinRng& rng, const std::vector<Vec2>& samples, Vec2 fallback) {
    if (samples.empty()) {
        return fallback;
    }
    std::uniform_int_distribution<size_t> dist(0, samples.size() - 1);
    return samples[dist(rng.rng)];
}

inline void run_neq_forward(const SimConfig& cfg, int traj_idx, const std::vector<Vec2>& eq_samples,
                            const PathData& path) {
    LangevinRng rng(cfg.neq_seed + static_cast<unsigned int>(traj_idx));
    Vec2 pos = pick_start(rng, eq_samples, cfg.A);
    Vec2 vel = sample_maxwell(cfg, rng);

    std::ostringstream fname;
    fname << cfg.out_dir << "/neq_fwd_" << traj_idx << ".csv";
    std::ofstream out(fname.str());
    out << "step,lambda,x,y,base_u,bias_u,work\n";

    double work = 0.0;
    const int nsteps = static_cast<int>(path.points.size());
    for (int step = 0; step < nsteps; ++step) {
        const double lam = path.lambdas[step];
        BiasHarmonic bias{path.k[step], path.points[step]};
        langevin_vv_step(cfg, bias, pos, vel, rng);

        const double base_u = potential_U(cfg, pos.x, pos.y);
        const double bias_u = harmonic_bias_energy(cfg, bias, pos);

        const double work_before = work;
        if (cfg.neq_output_stride <= 1 || (step % cfg.neq_output_stride == 0) || (step + 1 == nsteps)) {
            out << step << "," << std::setprecision(10) << lam << ","
                << pos.x << "," << pos.y << ","
                << base_u << "," << bias_u << ","
                << work_before << "\n";
        }

        if (step + 1 < nsteps) {
            BiasHarmonic bias_next{path.k[step + 1], path.points[step + 1]};
            work += harmonic_bias_energy(cfg, bias_next, pos) - bias_u;
        }
    }
}

inline void run_neq_backward(const SimConfig& cfg, int traj_idx, const std::vector<Vec2>& eq_samples,
                             const PathData& path) {
    const unsigned int seed = cfg.neq_seed + static_cast<unsigned int>(cfg.n_neq_traj + traj_idx);
    LangevinRng rng(seed);
    Vec2 pos = pick_start(rng, eq_samples, cfg.B);
    Vec2 vel = sample_maxwell(cfg, rng);

    std::ostringstream fname;
    fname << cfg.out_dir << "/neq_bwd_" << traj_idx << ".csv";
    std::ofstream out(fname.str());
    out << "step,lambda,x,y,base_u,bias_u,work\n";

    double work = 0.0;
    const int nsteps = static_cast<int>(path.points.size());
    for (int step = 0; step < nsteps; ++step) {
        const int idx = nsteps - 1 - step;
        const double lam = path.lambdas[idx];
        BiasHarmonic bias{path.k[idx], path.points[idx]};
        langevin_vv_step(cfg, bias, pos, vel, rng);

        const double base_u = potential_U(cfg, pos.x, pos.y);
        const double bias_u = harmonic_bias_energy(cfg, bias, pos);

        const double work_before = work;
        if (cfg.neq_output_stride <= 1 || (step % cfg.neq_output_stride == 0) || (step + 1 == nsteps)) {
            out << step << "," << std::setprecision(10) << lam << ","
                << pos.x << "," << pos.y << ","
                << base_u << "," << bias_u << ","
                << work_before << "\n";
        }

        if (step + 1 < nsteps) {
            const int idx_next = nsteps - 2 - step;
            BiasHarmonic bias_next{path.k[idx_next], path.points[idx_next]};
            work += harmonic_bias_energy(cfg, bias_next, pos) - bias_u;
        }
    }
}
