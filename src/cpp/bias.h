#pragma once

#include <cmath>
#include <vector>

#include "sim_types.h"

struct BiasHarmonic {
    double k = 200.0;
    Vec2 center{};

    double U(double x, double y) const {
        const double dx = x - center.x;
        const double dy = y - center.y;
        return 0.5 * k * (dx * dx + dy * dy);
    }

    Vec2 grad(double x, double y) const {
        const double dx = x - center.x;
        const double dy = y - center.y;
        return {k * dx, k * dy};
    }
};

struct GaussianHill {
    double height = 0.0;
    double sigma_x = 0.1;
    double sigma_y = 0.1;
    Vec2 center{};
    char one_dimension = 'n';

    double U(double x, double y) const {
        const double dx = x - center.x;
        const double dy = y - center.y;
        const double sx2 = sigma_x * sigma_x;
        const double sy2 = sigma_y * sigma_y;
        double exponent = 0.0;
        if (one_dimension == 'x') {
            exponent = -0.5 * (dx * dx) / sx2;
        } else if (one_dimension == 'y') {
            exponent = -0.5 * (dy * dy) / sy2;
        } else {
            exponent = -0.5 * ((dx * dx) / sx2 + (dy * dy) / sy2);
        }
        return height * std::exp(exponent);
    }

    Vec2 grad(double x, double y) const {
        const double dx = x - center.x;
        const double dy = y - center.y;
        const double sx2 = sigma_x * sigma_x;
        const double sy2 = sigma_y * sigma_y;
        const double u = U(x, y);
        if (one_dimension == 'x') {
            return {-u * dx / sx2, 0.0};
        }
        if (one_dimension == 'y') {
            return {0.0, -u * dy / sy2};
        }
        return {-u * dx / sx2, -u * dy / sy2};
    }
};

struct BiasWellTemperedMeta {
    double initial_height = 0.5;
    double sigma_x = 0.1;
    double sigma_y = 0.1;
    double bias_factor = 10.0;
    double kT = 1.0;
    char one_dimension = 'n';
    std::vector<GaussianHill> hills;

    double U(double x, double y) const {
        double total = 0.0;
        for (const auto& hill : hills) {
            total += hill.U(x, y);
        }
        return total;
    }

    Vec2 grad(double x, double y) const {
        Vec2 total{};
        for (const auto& hill : hills) {
            Vec2 g = hill.grad(x, y);
            total.x += g.x;
            total.y += g.y;
        }
        return total;
    }

    double next_height(double x, double y) const {
        const double denom = kT * (bias_factor - 1.0);
        if (denom <= 0.0) {
            return 0.0;
        }
        return initial_height * std::exp(-U(x, y) / denom);
    }

    double add_hill(double x, double y) {
        const double height = next_height(x, y);
        hills.push_back(GaussianHill{height, sigma_x, sigma_y, Vec2{x, y}, one_dimension});
        return height;
    }

    size_t size() const {
        return hills.size();
    }
};
