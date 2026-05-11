#pragma once

#include <string>
#include "sim_types.h"

struct SimConfig {
    // EQ and NEQ control
    int n_eq_steps = 20000;
    int n_neq_steps = 2000;
    int n_neq_traj = 10;
    double dt = 0.0005;
    double kT = 1; // kcal/mol at 300K
    double mass = 1.0;
    double gamma = 500.0;

    enum class PotentialType {
        MullerBrown,
        SixHumpCamel,
        ThreeWell,
        DoubleWell1D
    };
    PotentialType potential = PotentialType::MullerBrown;

    // 1D log-sum-exp double well:
    // U = -kT * ln(exp(-(k0*(q-x0)^2)/kT) + exp(-(k1*(q-x1)^2)/kT - E1))
    // where q is x or y depending on `one_dimension`.
    double one_d_k0 = 10.0;
    double one_d_x0 = -1.0;
    double one_d_k1 = 10.0;
    double one_d_x1 = 1.0;
    double one_d_E1 = 0.0;

    // Endpoints for restraint centers
    Vec2 A{-0.6, 1.4};
    Vec2 B{0.6, 0.0};

    // Isotropic harmonic bias strength.
    double k = 50.0;

    // Optional midpoint scaling for NEQ.
    double k_midscale = 1.0;

    char one_dimension = 'n'; // 'x', 'y', or 'n' for none

    // Optional path CSV; if empty, linear path between A and B is used
    std::string path_csv = "";
    int n_path_points = 0;

    int eq_output_samples = 1000;
    int neq_output_stride = 1;
    unsigned int eq_seed = 123u;
    unsigned int neq_seed = 42u;
    int n_meta_steps = 20000;
    int meta_output_stride = 20;
    int meta_deposition_stride = 200;

    double meta_initial_height = 0.5;
    double meta_sigma_x = 0.08;
    double meta_sigma_y = 0.08;
    double meta_bias_factor = 10.0;
    unsigned int meta_seed = 123u;

    int eq_minimize_steps = 0;
    double eq_minimize_alpha = 0.01;
    bool eq_write_initial = false;

    std::string out_dir = "simulations/cpp/outputs";
};
