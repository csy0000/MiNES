# Baseline Design Note

## Adaptive umbrella sampling

The adaptive umbrella implementation is intentionally simple and engine-local:

- start from one or more seed windows
- run equilibrium Langevin sampling in each harmonic umbrella window
- inspect directional support near neighboring centers on a regular grid
- spawn a neighbor only if the parent window has both:
  - at least `aus_min_count` samples in that direction
  - directional support ratio above `aus_overlap_cutoff`
- stop when the target region is reached, the queue is exhausted, or the
  maximum number of windows is reached

For 1D, the active CV is the selected coordinate (`x` or `y`). For 2D, the
grid grows on the four cardinal neighbors of each accepted window.

The PMF/FES estimate is built from the collected umbrella trajectories with a
lightweight per-window reweighting by `exp(beta * U_bias)` followed by
histogram stitching on a regular grid. This stays transparent and avoids
introducing a heavy WHAM dependency for the first benchmark version.

## Well-tempered metadynamics

The metadynamics runner keeps the existing hill-deposition structure and
extends it in two ways:

- 1D direct-CV support through `-one-dimension x|y`
- gridded FES reconstruction from the deposited hills

The reconstruction uses the standard well-tempered relation

`F(s) ~= -(gamma / (gamma - 1)) * V_bias(s) + C`

with `gamma = meta_biasfactor`, and shifts the result so the minimum free
energy is zero. The reconstruction can be done immediately during `META` mode
when `-meta_fes_out` is given, or separately with `-meta_fes_mode`.

## Fairness and comparability

The baseline methods are kept comparable to the existing NES/MiNES workflows
by enforcing the same local conventions:

- same Langevin velocity-Verlet family
- same toy potentials
- same `kT`, `dt`, `gamma`, and mass conventions
- explicit CLI-exposed hyperparameters
- CSV outputs for trajectories, windows, hills, PMFs, and FES grids
- benchmark summary logs with step counts and estimated force evaluations

This keeps the benchmark baselines simple, reproducible, and easy to compare
without introducing external engines or a second simulation framework.

