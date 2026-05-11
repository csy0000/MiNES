#pragma once

#include <cmath>
#include "sim_config.h"
#include "sim_types.h"

struct MullerBrown {
    static constexpr int kNum = 4;
    static constexpr double A[kNum]  = {-200.0, -100.0, -170.0,  15.0};
    static constexpr double a[kNum]  = {  -1.0,   -1.0,   -6.5,  0.7};
    static constexpr double b[kNum]  = {   0.0,    0.0,   11.0,  0.6};
    static constexpr double c[kNum]  = { -10.0,  -10.0,   -6.5,  0.7};
    static constexpr double x0[kNum] = {   1.0,    0.0,   -0.5, -1.0};
    static constexpr double y0[kNum] = {   0.0,    0.5,    1.5,  1.0};

    static double U(double x, double y) {
        double u = 0.0;
        for (int i = 0; i < kNum; ++i) {
            const double dx = x - x0[i];
            const double dy = y - y0[i];
            const double expo = a[i] * dx * dx + b[i] * dx * dy + c[i] * dy * dy;
            u += A[i] * std::exp(expo);
        }
        return u;
    }

    static Vec2 grad(double x, double y) {
        Vec2 g{};
        for (int i = 0; i < kNum; ++i) {
            const double dx = x - x0[i];
            const double dy = y - y0[i];
            const double expo = a[i] * dx * dx + b[i] * dx * dy + c[i] * dy * dy;
            const double u = A[i] * std::exp(expo);
            g.x += u * (2.0 * a[i] * dx + b[i] * dy);
            g.y += u * (b[i] * dx + 2.0 * c[i] * dy);
        }
        return g;
    }
};

struct SixHumpCamel {
    static double U(double x, double y) {
        const double term1 = (4.0 - 2.1 * x * x + (x * x * x * x) / 3.0) * x * x;
        const double term2 = x * y;
        const double term3 = (-4.0 + 4.0 * y * y) * y * y;
        return term1 + term2 + term3;
    }

    static Vec2 grad(double x, double y) {
        const double dx = 8.0 * x - 8.4 * x * x * x + 2.0 * x * x * x * x * x + y;
        const double dy = x - 8.0 * y + 16.0 * y * y * y;
        return {dx, dy};
    }
};

struct ThreeWell {
    static constexpr int kNum = 3;
    static constexpr double wells_x[kNum] = {-2.0, 0.0, 2.0};
    static constexpr double wells_y[kNum] = {-0.5, 0.5, -0.5};
    static constexpr double depths[kNum] = {3.0, 10.0, 3.0};
    static constexpr double sigmas[kNum] = {0.35, 0.35, 0.35};
    static constexpr double base_k = 1;
    static constexpr double base_x = 0.0;
    static constexpr double base_y = -1.0;

    static double U(double x, double y) {
        double u = 0; // 0.5 * base_k * ((x - base_x) * (x - base_x) + (y - base_y) * (y - base_y));
        for (int i = 0; i < kNum; ++i) {
            const double dx = x - wells_x[i];
            const double dy = y - wells_y[i];
            const double s2 = sigmas[i] * sigmas[i];
            u -= depths[i] * std::exp(-(dx * dx + dy * dy) / (2.0 * s2));
        }
        return u;
    }

    static Vec2 grad(double x, double y) {
        Vec2 g{};
        g.x = base_k * (x - base_x);
        g.y = base_k * (y - base_y);
        for (int i = 0; i < kNum; ++i) {
            const double dx = x - wells_x[i];
            const double dy = y - wells_y[i];
            const double s2 = sigmas[i] * sigmas[i];
            const double e = std::exp(-(dx * dx + dy * dy) / (2.0 * s2));
            const double coeff = depths[i] * e / s2;
            g.x += coeff * dx;
            g.y += coeff * dy;
        }
        return g;
    }
};

struct DoubleWell1D {
    static double active_coordinate(const SimConfig& cfg, double x, double y) {
        return (cfg.one_dimension == 'y') ? y : x;
    }

    static double U(const SimConfig& cfg, double x, double y) {
        const double q = active_coordinate(cfg, x, y);
        const double beta = 1.0 / cfg.kT;
        const double u0 = cfg.one_d_k0 * (q - cfg.one_d_x0) * (q - cfg.one_d_x0);
        const double u1 = cfg.one_d_k1 * (q - cfg.one_d_x1) * (q - cfg.one_d_x1);
        const double log_t0 = -beta * u0;
        const double log_t1 = -beta * u1 - cfg.one_d_E1;
        const double log_max = std::max(log_t0, log_t1);
        const double log_sum = log_max + std::log(std::exp(log_t0 - log_max) + std::exp(log_t1 - log_max));
        return -log_sum / beta;
    }

    static Vec2 grad(const SimConfig& cfg, double x, double y) {
        const double q = active_coordinate(cfg, x, y);
        const double beta = 1.0 / cfg.kT;
        const double u0 = cfg.one_d_k0 * (q - cfg.one_d_x0) * (q - cfg.one_d_x0);
        const double u1 = cfg.one_d_k1 * (q - cfg.one_d_x1) * (q - cfg.one_d_x1);
        const double du0 = 2.0 * cfg.one_d_k0 * (q - cfg.one_d_x0);
        const double du1 = 2.0 * cfg.one_d_k1 * (q - cfg.one_d_x1);

        const double log_t0 = -beta * u0;
        const double log_t1 = -beta * u1 - cfg.one_d_E1;
        const double log_max = std::max(log_t0, log_t1);
        const double w0 = std::exp(log_t0 - log_max);
        const double w1 = std::exp(log_t1 - log_max);
        const double denom = w0 + w1;
        const double dq = (w0 * du0 + w1 * du1) / denom;

        if (cfg.one_dimension == 'y') {
            return {0.0, dq};
        }
        return {dq, 0.0};
    }
};

inline double potential_U(const SimConfig& cfg, double x, double y) {
    switch (cfg.potential) {
        case SimConfig::PotentialType::MullerBrown:
            return MullerBrown::U(x, y);
        case SimConfig::PotentialType::SixHumpCamel:
            return SixHumpCamel::U(x, y);
        case SimConfig::PotentialType::ThreeWell:
            return ThreeWell::U(x, y);
        case SimConfig::PotentialType::DoubleWell1D:
            return DoubleWell1D::U(cfg, x, y);
    }
    return MullerBrown::U(x, y);
}

inline Vec2 potential_grad(const SimConfig& cfg, double x, double y) {
    switch (cfg.potential) {
        case SimConfig::PotentialType::MullerBrown:
            return MullerBrown::grad(x, y);
        case SimConfig::PotentialType::SixHumpCamel:
            return SixHumpCamel::grad(x, y);
        case SimConfig::PotentialType::ThreeWell:
            return ThreeWell::grad(x, y);
        case SimConfig::PotentialType::DoubleWell1D:
            return DoubleWell1D::grad(cfg, x, y);
    }
    return MullerBrown::grad(x, y);
}
