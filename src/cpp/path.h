#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "sim_config.h"
#include "sim_types.h"

inline Vec2 lerp(Vec2 a, Vec2 b, double t) {
    return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
}

inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

inline double quad_mid_scaling_k(double lamb, double mid_scale) {
    if (mid_scale == 1.0) {
        return 1.0;
    }
    if (mid_scale > 1.0) {
        return 1.0 + mid_scale * (0.25 - (lamb - 0.5) * (lamb - 0.5));
    }
    double scale = 4.0 * (lamb - 0.5) * (lamb - 0.5);
    scale = scale * (1.0 - mid_scale) + mid_scale;
    return scale;
}

struct PathData {
    std::vector<double> lambdas;
    std::vector<Vec2> points;
    std::vector<double> k;
    bool has_k = false;
};

struct RawPathRow {
    Vec2 point{};
    double k = 0.0;
    bool has_k = false;
};

inline std::vector<RawPathRow> read_path_rows(const std::string& path_csv) {
    std::vector<RawPathRow> rows;
    std::ifstream in(path_csv);
    if (!in.is_open()) {
        return rows;
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

        RawPathRow row{};
        if (vals.size() >= 4) {
            row.point = Vec2{vals[1], vals[2]};
            row.k = vals[3];
            row.has_k = true;
        } else if (vals.size() == 3) {
            row.point = Vec2{vals[1], vals[2]};
        } else if (vals.size() >= 2) {
            row.point = Vec2{vals[0], vals[1]};
        } else {
            continue;
        }
        rows.push_back(row);
    }
    return rows;
}

inline void write_path_csv(const std::string& out_path, const PathData& path) {
    std::ofstream out(out_path);
    out << "lambda,x0,y0,k\n";
    for (size_t i = 0; i < path.points.size(); ++i) {
        out << std::setprecision(10) << path.lambdas[i] << ","
            << path.points[i].x << "," << path.points[i].y << ","
            << path.k[i] << "\n";
    }
}

inline PathData downsample_path(const PathData& path, int stride) {
    if (stride <= 1 || path.points.empty()) {
        return path;
    }
    PathData out{};
    out.has_k = path.has_k;
    const int n = static_cast<int>(path.points.size());
    for (int i = 0; i < n; ++i) {
        if ((i % stride == 0) || (i + 1 == n)) {
            out.lambdas.push_back(path.lambdas[i]);
            out.points.push_back(path.points[i]);
            out.k.push_back(path.k[i]);
        }
    }
    return out;
}

inline PathData build_path(const SimConfig& cfg) {
    PathData path{};
    const int requested = (cfg.n_path_points > 0) ? cfg.n_path_points : (cfg.n_neq_steps + 2);
    const int n = std::max(2, requested);
    path.lambdas.resize(n);
    for (int i = 0; i < n; ++i) {
        path.lambdas[i] = static_cast<double>(i) / static_cast<double>(n - 1);
    }

    if (!cfg.path_csv.empty()) {
        std::vector<RawPathRow> rows = read_path_rows(cfg.path_csv);
        if (static_cast<int>(rows.size()) >= 2) {
            path.points.reserve(rows.size());
            path.k.reserve(rows.size());
            bool all_have_k = true;
            for (const auto& row : rows) {
                path.points.push_back(row.point);
                if (row.has_k) {
                    path.k.push_back(row.k);
                } else {
                    all_have_k = false;
                }
            }
            path.has_k = all_have_k;
            if (!path.has_k) {
                path.k.clear();
            }
        }
    }

    if (path.points.empty()) {
        path.points.resize(n);
        for (int i = 0; i < n; ++i) {
            path.points[i] = lerp(cfg.A, cfg.B, path.lambdas[i]);
        }
    } else {
        const int m = static_cast<int>(path.points.size());
        if (m != n) {
            std::vector<Vec2> resampled(n);
            std::vector<double> k_resampled;
            if (path.has_k) {
                k_resampled.resize(n);
            }
            for (int i = 0; i < n; ++i) {
                const double t = path.lambdas[i];
                const double idx = t * (m - 1);
                const int i0 = static_cast<int>(std::floor(idx));
                const int i1 = std::min(m - 1, i0 + 1);
                const double w = idx - i0;
                resampled[i] = lerp(path.points[i0], path.points[i1], w);
                if (path.has_k) {
                    k_resampled[i] = lerp(path.k[i0], path.k[i1], w);
                }
            }
            path.points = std::move(resampled);
            if (path.has_k) {
                path.k = std::move(k_resampled);
            }
        }
    }

    if (!path.has_k) {
        path.k.resize(n);
        for (int i = 0; i < n; ++i) {
            const double lam = path.lambdas[i];
            path.k[i] = cfg.k * quad_mid_scaling_k(lam, cfg.k_midscale);
        }
    }

    return path;
}
