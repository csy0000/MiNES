# `src/cpp`

This directory contains the lightweight C++ simulation engine used by the
benchmark scripts.

## Header Map

### [sim_types.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/sim_types.h)

Shared low-level types such as `Vec2`.

### [sim_config.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/sim_config.h)

Central simulation configuration:

- potential choice
- thermal parameters
- 1D double-well parameters
- harmonic restraint parameters
- MTD settings
- seeds and output strides

### [potential.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/potential.h)

Implements the model potentials and their gradients.

The active 1D double-well form is:

`U(x) = -(1 / beta) ln(exp(-beta * k0 * (x - x0)^2) + exp(-beta * k1 * (x - x1)^2 - E1))`

with `beta = 1 / kT`.

### [bias.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/bias.h)

Bias potentials:

- isotropic harmonic restraint
- well-tempered MTD hill bookkeeping

### [path.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/path.h)

Builds the nonequilibrium path used by the `NEQ` and `NES` workflows.

### [eq_neq.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/eq_neq.h)

Core integrator and simulation routines:

- equilibrium restrained trajectories
- nonequilibrium switching trajectories
- well-tempered metadynamics
- CSV writers and summary structs

### [us.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/us.h)

Fixed-window umbrella sampling (`US`) on top of the same integrator.

It:

- places evenly spaced windows along the line from the start basin to the end basin
- orders them by `forward`, `backward`, or `bidirectional`
- runs restrained EQ trajectories window by window
- writes `us_window_<id>.csv`, `us_windows.csv`, and `us_fes.csv`

### [fes.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/fes.h)

Grid utilities and FES / PMF CSV writers shared by `US` and MTD analysis.

### [benchmark.h](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/src/cpp/benchmark.h)

Benchmark summary structs and log-line helpers.

## Active Driver

The main executable is:

- [neq_sim.cpp](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/simulations/cpp/neq_sim.cpp)

It dispatches to:

- `EQ`
- `NEQ`
- `WT_META`
- `US`
- `PATH`
- `META_FES`

## Current Notes

- The active umbrella baseline is fixed-window `US`; the older adaptive `AUS`
  implementation is no longer the active path.
- The current benchmark is 1D-only and uses isotropic harmonic restraints.
