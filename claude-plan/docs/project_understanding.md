# Project Understanding — MiNES

Last updated: 2026-05-02 (updated after [20:13] cleanup)

## What this project is

MiNES is a 1D/2D free-energy benchmark framework in C++ + Python. It compares
how fast different sampling methods converge to the correct PMF as a function
of simulation budget. The novel method is **MiNES** (adaptive milestone-chain
NEQ). Competing baselines are US, NES, and WT-MTD.

## Runtime flow (high-level)

```
neq_sim (C++ binary)
    ↑ called by
adaptive_methods.py / run_*.sh
    ↓ produces raw data
analysis/analysis_US_MTD.py  (MISSING)
    ↓ reduces to processed/seed_*.dat
benchmark/selected/*.dat
```

The C++ binary is the simulation engine for EQ sampling, NEQ switching,
umbrella sampling, and metadynamics. Python orchestrators drive the adaptive
logic (window placement, milestone chain growth). Python reducers do PMF
reconstruction (MBAR, Hummer-Szabo, BAR).

## C++ header dependency order

```
sim_types.h  →  sim_config.h  →  potential.h
                              →  bias.h
                              →  path.h
                              →  eq_neq.h  →  fes.h  →  us.h
```

`neq_sim.cpp` includes only `../../src/cpp/us.h` which transitively pulls in
everything else.

## Analysis modules (all now present in `analysis/`)

- `bidirectional_mts_pmf.py` — BAR/CFT delta_f, Hummer-Szabo MTS bootstrap
- `mines_current_protocol_analysis.py` — direct_eq_mbar_pmf, pair_js_divergence, bootstrap_direct_eq_mbar
- `mines_notebook_utils.py` — background_potential_1d, coverage_mask_from_samples, mode_x_from_samples
- `analysis_US_MTD.py` — CLI reducer (process-us-seed, process-aus-seed, process-mines-seed, etc.)

User moved these from the old project on 2026-05-02.

## 1D potentials: current vs. needed

| Potential | Status |
|---|---|
| DoubleWell1D (2-well) | Implemented in `src/cpp/potential.h` |
| TripleWell1D (3-well) | NOT implemented. `ThreeWell` struct is 2D. |
| MullerBrown (2D) | Implemented, but Python workflow only partial |

## Cleanup done 2026-05-02

The following were deleted:
- `src/cpp/anisotropic_backup/`, `src/cpp/neq_sim`, `src/analysis/` — stale
- `analysis/MiNES/` (pkl + png from past runs), `analysis/notebook/figures/`
- `analysis/mines_current_protocol_cache.py`, `analysis/rebuild_selected_from_raw.py`
- 3 diagnostic analysis shell scripts, 2 simulations scripts for T_NEQ/intermediate windows
- `simulations/run_muller_brown_aus_mtd.sh`, `simulations/plot_muller_brown_aus_mtd.py` — 2D deferred
- `simulations/run_benchmark_baselines.sh` — old parameter set
- `legacy/README.md` — stale paths
- 25 development/diagnostic notebooks and builder scripts in `analysis/notebook/`

## Workflow: 1D benchmark for one system

```bash
# 1. Build
clang++ -O2 -std=c++17 simulations/cpp/neq_sim.cpp -o simulations/cpp/neq_sim

# 2. Run (all methods, default DoubleWell system)
RUN_NOTEBOOK=0 bash simulations/run_US_MTD_NES.sh

# 3. Analyze
bash analysis/analysis_US_MTD_NES.sh data/1D/<system_slug>

# 4. Plot
python analysis/notebook/plot_doublewell_benchmark.py --system-root data/1D/<system_slug>
```

Steps 3 and 4 require the missing `analysis/` Python files.

## System slug format

`<SystemName>__k0_<k0>__x0_<x0>__k1_<k1>__x1_<x1>__E1_<E1>__kT_<kT>__dt_<dt>__gamma_<gamma>`

with decimal points replaced by `p` and minus signs by `m`.

Example: `DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0`

## Python env

Conda env name: `MiNES`. Requires `pymbar`, `numpy`, `pandas`, `scipy`,
and optionally `jax`.

## Open tasks (as of 2026-05-02)

- [ ] Smoke-test Python imports (adaptive_methods.py + mines_variance_fusion.py)
- [ ] Run 1D DoubleWell benchmark end-to-end (US, NES, MTD, MiNES)
- [ ] Set up `.gitignore` for `__pycache__`, `data/`, `*.pyc`, `*.DS_Store`
- [ ] Add `TripleWell1D` to `src/cpp/potential.h` (deferred — after 1D DoubleWell works)
- [ ] 2D Muller-Brown pipeline (deferred)
