# src/

Source code for the simulation engine and analysis modules.

## Subdirectories

### `src/cpp/` — C++ simulation headers

Header-only library included by `simulations/cpp/neq_sim.cpp`.

| Header | Role |
|---|---|
| `sim_types.h` | Shared types (`Vec2`, etc.) |
| `sim_config.h` | Central simulation configuration struct |
| `potential.h` | Model potentials: `DoubleWell1D`, `MullerBrown`, `SixHumpCamel`, `ThreeWell` |
| `bias.h` | Bias potentials: harmonic restraint, WT-MTD hills |
| `path.h` | NEQ path construction |
| `eq_neq.h` | Langevin integrator, EQ/NEQ/MTD routines, CSV writers |
| `fes.h` | Grid utilities and PMF CSV writers |
| `us.h` | Fixed-window umbrella sampling runner |
| `benchmark.h` | Benchmark summary structs and log helpers |

Header dependency order:

```
sim_types.h → sim_config.h → potential.h
                           → bias.h
                           → path.h
                           → eq_neq.h → fes.h → us.h
```

`neq_sim.cpp` includes only `us.h`, which pulls in everything else transitively.

### `src/analysis/` — Python analysis modules

PMF reconstruction and utility modules shared by all analysis scripts.

| Module | Role |
|---|---|
| `analysis_US_MTD.py` | CLI reducer: per-seed reduction and final aggregation for US, AUS, NES, MINES, MTD |
| `bidirectional_mts_pmf.py` | BAR/CFT ΔF solver, Hummer-Szabo MTS bootstrap |
| `mines_current_protocol_analysis.py` | Direct EQ MBAR PMF, JS divergence, bootstrap |
| `mines_notebook_utils.py` | Background potential, coverage mask, mode utilities |

`REPO_ROOT` is resolved as `Path(__file__).resolve().parents[2]` (two levels up from `src/analysis/`).

All four modules are imported by `simulations/adaptive_methods.py` and `scripts/mines_variance_fusion.py` via `sys.path` insertion.
