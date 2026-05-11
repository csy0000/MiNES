#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "bias.h"
#include "sim_config.h"
#include "sim_types.h"

inline Vec2 vec_add(Vec2 a, Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}

inline Vec2 vec_sub(Vec2 a, Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}

inline Vec2 vec_scale(Vec2 a, double s) {
    return {a.x * s, a.y * s};
}

inline double vec_dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

inline double vec_norm2(Vec2 a) {
    return vec_dot(a, a);
}

inline double vec_norm(Vec2 a) {
    return std::sqrt(vec_norm2(a));
}

inline double active_coordinate(const SimConfig& cfg, const Vec2& pos) {
    return (cfg.one_dimension == 'y') ? pos.y : pos.x;
}

struct GridSpec1D {
    double xmin = 0.0;
    double xmax = 0.0;
    double dx = 0.1;
    int nx = 1;
};

struct GridSpec2D {
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
    double dx = 0.1;
    double dy = 0.1;
    int nx = 1;
    int ny = 1;
};

inline GridSpec1D make_grid_1d(double xmin, double xmax, double dx) {
    GridSpec1D grid{};
    grid.dx = std::max(dx, 1e-6);
    if (xmax <= xmin) {
        xmax = xmin + grid.dx;
    }
    grid.xmin = xmin;
    grid.xmax = xmax;
    grid.nx = 1 + static_cast<int>(std::ceil((grid.xmax - grid.xmin) / grid.dx));
    grid.xmax = grid.xmin + grid.dx * static_cast<double>(grid.nx - 1);
    return grid;
}

inline GridSpec2D make_grid_2d(double xmin, double xmax, double ymin, double ymax,
                               double dx, double dy) {
    GridSpec2D grid{};
    grid.dx = std::max(dx, 1e-6);
    grid.dy = std::max(dy, 1e-6);
    if (xmax <= xmin) {
        xmax = xmin + grid.dx;
    }
    if (ymax <= ymin) {
        ymax = ymin + grid.dy;
    }
    grid.xmin = xmin;
    grid.xmax = xmax;
    grid.ymin = ymin;
    grid.ymax = ymax;
    grid.nx = 1 + static_cast<int>(std::ceil((grid.xmax - grid.xmin) / grid.dx));
    grid.ny = 1 + static_cast<int>(std::ceil((grid.ymax - grid.ymin) / grid.dy));
    grid.xmax = grid.xmin + grid.dx * static_cast<double>(grid.nx - 1);
    grid.ymax = grid.ymin + grid.dy * static_cast<double>(grid.ny - 1);
    return grid;
}

inline GridSpec1D auto_grid_1d(const SimConfig& cfg,
                               const std::vector<Vec2>& samples,
                               const std::vector<Vec2>& centers,
                               double spacing,
                               double dx_hint) {
    double qmin = std::numeric_limits<double>::infinity();
    double qmax = -std::numeric_limits<double>::infinity();
    for (const auto& pos : samples) {
        const double q = active_coordinate(cfg, pos);
        qmin = std::min(qmin, q);
        qmax = std::max(qmax, q);
    }
    for (const auto& pos : centers) {
        const double q = active_coordinate(cfg, pos);
        qmin = std::min(qmin, q);
        qmax = std::max(qmax, q);
    }
    if (!std::isfinite(qmin) || !std::isfinite(qmax)) {
        qmin = -1.0;
        qmax = 1.0;
    }
    const double dx = (dx_hint > 0.0) ? dx_hint : std::max(spacing * 0.25, 0.02);
    const double margin = std::max(spacing, 4.0 * dx);
    return make_grid_1d(qmin - margin, qmax + margin, dx);
}

inline GridSpec2D auto_grid_2d(const std::vector<Vec2>& samples,
                               const std::vector<Vec2>& centers,
                               double spacing,
                               double dx_hint,
                               double dy_hint) {
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (const auto& pos : samples) {
        xmin = std::min(xmin, pos.x);
        xmax = std::max(xmax, pos.x);
        ymin = std::min(ymin, pos.y);
        ymax = std::max(ymax, pos.y);
    }
    for (const auto& pos : centers) {
        xmin = std::min(xmin, pos.x);
        xmax = std::max(xmax, pos.x);
        ymin = std::min(ymin, pos.y);
        ymax = std::max(ymax, pos.y);
    }
    if (!std::isfinite(xmin) || !std::isfinite(xmax) ||
        !std::isfinite(ymin) || !std::isfinite(ymax)) {
        xmin = -1.0;
        xmax = 1.0;
        ymin = -1.0;
        ymax = 1.0;
    }
    const double dx = (dx_hint > 0.0) ? dx_hint : std::max(spacing * 0.25, 0.02);
    const double dy = (dy_hint > 0.0) ? dy_hint : std::max(spacing * 0.25, 0.02);
    const double xmargin = std::max(spacing, 4.0 * dx);
    const double ymargin = std::max(spacing, 4.0 * dy);
    return make_grid_2d(xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin, dx, dy);
}

inline int grid_index_1d(const GridSpec1D& grid, double x) {
    const int idx = static_cast<int>(std::floor((x - grid.xmin) / grid.dx + 0.5));
    if (idx < 0 || idx >= grid.nx) {
        return -1;
    }
    return idx;
}

inline int grid_index_x(const GridSpec2D& grid, double x) {
    const int idx = static_cast<int>(std::floor((x - grid.xmin) / grid.dx + 0.5));
    if (idx < 0 || idx >= grid.nx) {
        return -1;
    }
    return idx;
}

inline int grid_index_y(const GridSpec2D& grid, double y) {
    const int idx = static_cast<int>(std::floor((y - grid.ymin) / grid.dy + 0.5));
    if (idx < 0 || idx >= grid.ny) {
        return -1;
    }
    return idx;
}

inline void shift_free_energy_min_zero(std::vector<double>& free_energy) {
    double min_f = std::numeric_limits<double>::infinity();
    for (double value : free_energy) {
        if (std::isfinite(value)) {
            min_f = std::min(min_f, value);
        }
    }
    if (!std::isfinite(min_f)) {
        return;
    }
    for (double& value : free_energy) {
        if (std::isfinite(value)) {
            value -= min_f;
        }
    }
}

inline void write_pmf_1d_csv(const std::string& out_path,
                             const GridSpec1D& grid,
                             const std::vector<double>& free_energy,
                             const std::vector<double>& count,
                             const std::vector<double>& uncertainty) {
    std::ofstream out(out_path);
    out << "x,F,count,uncertainty_estimate\n";
    for (int ix = 0; ix < grid.nx; ++ix) {
        const double x = grid.xmin + grid.dx * static_cast<double>(ix);
        out << std::setprecision(10) << x << ",";
        if (std::isfinite(free_energy[ix])) {
            out << free_energy[ix];
        }
        out << "," << count[ix] << ",";
        if (std::isfinite(uncertainty[ix])) {
            out << uncertainty[ix];
        }
        out << "\n";
    }
}

inline void write_fes_2d_csv(const std::string& out_path,
                             const GridSpec2D& grid,
                             const std::vector<double>& free_energy,
                             const std::vector<double>& count) {
    std::ofstream out(out_path);
    out << "x,y,F,count\n";
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int idx = iy * grid.nx + ix;
            const double x = grid.xmin + grid.dx * static_cast<double>(ix);
            const double y = grid.ymin + grid.dy * static_cast<double>(iy);
            out << std::setprecision(10) << x << "," << y << ",";
            if (std::isfinite(free_energy[idx])) {
                out << free_energy[idx];
            }
            out << "," << count[idx] << "\n";
        }
    }
}

struct MetaHillRecord {
    int hill = 0;
    int step = 0;
    GaussianHill gaussian{};
    double bias_before = 0.0;
    double bias_after = 0.0;
};

inline std::vector<MetaHillRecord> read_meta_hills_csv(const std::string& path, char one_dimension) {
    std::vector<MetaHillRecord> hills;
    std::ifstream in(path);
    if (!in.is_open()) {
        return hills;
    }

    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) {
            first = false;
            if (line.find("hill") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 9) {
            continue;
        }
        MetaHillRecord record{};
        record.hill = std::stoi(tokens[0]);
        record.step = std::stoi(tokens[1]);
        record.gaussian.center.x = std::stod(tokens[2]);
        record.gaussian.center.y = std::stod(tokens[3]);
        record.gaussian.height = std::stod(tokens[4]);
        record.gaussian.sigma_x = std::stod(tokens[5]);
        record.gaussian.sigma_y = std::stod(tokens[6]);
        record.gaussian.one_dimension = one_dimension;
        record.bias_before = std::stod(tokens[7]);
        record.bias_after = std::stod(tokens[8]);
        hills.push_back(record);
    }
    return hills;
}

inline GridSpec1D auto_meta_grid_1d(const std::vector<MetaHillRecord>& hills, char one_dimension,
                                    double xmin, double xmax, int nx) {
    double qmin = std::numeric_limits<double>::infinity();
    double qmax = -std::numeric_limits<double>::infinity();
    for (const auto& hill : hills) {
        const double q = (one_dimension == 'y') ? hill.gaussian.center.y : hill.gaussian.center.x;
        const double sigma = (one_dimension == 'y') ? hill.gaussian.sigma_y : hill.gaussian.sigma_x;
        qmin = std::min(qmin, q - 4.0 * sigma);
        qmax = std::max(qmax, q + 4.0 * sigma);
    }
    if (std::isfinite(xmin) && std::isfinite(xmax) && xmax > xmin) {
        qmin = xmin;
        qmax = xmax;
    }
    if (!std::isfinite(qmin) || !std::isfinite(qmax)) {
        qmin = -1.0;
        qmax = 1.0;
    }
    const int n = std::max(nx, 2);
    const double dx = (qmax - qmin) / static_cast<double>(n - 1);
    return make_grid_1d(qmin, qmax, dx);
}

inline GridSpec2D auto_meta_grid_2d(const std::vector<MetaHillRecord>& hills,
                                    double xmin, double xmax,
                                    double ymin, double ymax,
                                    int nx, int ny) {
    double xlo = std::numeric_limits<double>::infinity();
    double xhi = -std::numeric_limits<double>::infinity();
    double ylo = std::numeric_limits<double>::infinity();
    double yhi = -std::numeric_limits<double>::infinity();
    for (const auto& hill : hills) {
        xlo = std::min(xlo, hill.gaussian.center.x - 4.0 * hill.gaussian.sigma_x);
        xhi = std::max(xhi, hill.gaussian.center.x + 4.0 * hill.gaussian.sigma_x);
        ylo = std::min(ylo, hill.gaussian.center.y - 4.0 * hill.gaussian.sigma_y);
        yhi = std::max(yhi, hill.gaussian.center.y + 4.0 * hill.gaussian.sigma_y);
    }
    if (std::isfinite(xmin) && std::isfinite(xmax) && xmax > xmin) {
        xlo = xmin;
        xhi = xmax;
    }
    if (std::isfinite(ymin) && std::isfinite(ymax) && ymax > ymin) {
        ylo = ymin;
        yhi = ymax;
    }
    if (!std::isfinite(xlo) || !std::isfinite(xhi) || !std::isfinite(ylo) || !std::isfinite(yhi)) {
        xlo = -1.0;
        xhi = 1.0;
        ylo = -1.0;
        yhi = 1.0;
    }
    const int ngrid_x = std::max(nx, 2);
    const int ngrid_y = std::max(ny, 2);
    const double dx = (xhi - xlo) / static_cast<double>(ngrid_x - 1);
    const double dy = (yhi - ylo) / static_cast<double>(ngrid_y - 1);
    return make_grid_2d(xlo, xhi, ylo, yhi, dx, dy);
}

inline void write_meta_fes_1d_csv(const std::string& out_path,
                                  const GridSpec1D& grid,
                                  const std::vector<double>& free_energy,
                                  const std::vector<double>& bias) {
    std::ofstream out(out_path);
    out << "x,F,bias\n";
    for (int ix = 0; ix < grid.nx; ++ix) {
        const double x = grid.xmin + grid.dx * static_cast<double>(ix);
        out << std::setprecision(10) << x << ",";
        if (std::isfinite(free_energy[ix])) {
            out << free_energy[ix];
        }
        out << "," << bias[ix] << "\n";
    }
}

inline void write_meta_fes_2d_csv(const std::string& out_path,
                                  const GridSpec2D& grid,
                                  const std::vector<double>& free_energy,
                                  const std::vector<double>& bias) {
    std::ofstream out(out_path);
    out << "x,y,F,bias\n";
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int idx = iy * grid.nx + ix;
            const double x = grid.xmin + grid.dx * static_cast<double>(ix);
            const double y = grid.ymin + grid.dy * static_cast<double>(iy);
            out << std::setprecision(10) << x << "," << y << ",";
            if (std::isfinite(free_energy[idx])) {
                out << free_energy[idx];
            }
            out << "," << bias[idx] << "\n";
        }
    }
}

