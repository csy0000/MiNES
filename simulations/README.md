# simulations/

C++ simulation engine and Python orchestrators for the benchmark.

## Contents

| Path | Purpose |
|---|---|
| `cpp/neq_sim.cpp` | Main C++ driver (compile once before running) |
| `cpp/neq_sim` | Compiled binary (git-ignored) |
| `adaptive_methods.py` | Python orchestrator for AUS and legacy MiNES |
| `write_method_contexts.py` | Writes `method_context.json` / `combo_context.json` files |

## Build

```bash
clang++ -O2 -std=c++17 simulations/cpp/neq_sim.cpp -o simulations/cpp/neq_sim
```

The binary is the sole simulation backend. All Python and shell scripts call it
by path via the `--bin` / `-bin` argument.

## adaptive_methods.py

CLI entry point for the two adaptive methods. Commands:

| Command | Method |
|---|---|
| `run-aus-seed` | Adaptive Umbrella Sampling — grows one paired left/right umbrella chain |
| `run-mines-seed` | MiNES (legacy) — grows one milestone chain from both endpoints |
| `run-mines-current-protocol` | MiNES current protocol (used by `run_MiNES.sh`) |

Called internally by `scripts/run_US_MTD_NES.sh` and `scripts/run_MiNES.sh`.
For the newer variance-fusion MiNES, use `scripts/mines_variance_fusion.py` instead.

## Dispatcher modes (`neq_sim`)

The C++ binary dispatches on the first mode flag:

| Mode | Description |
|---|---|
| `EQ` | Restrained equilibrium trajectory |
| `NEQ` | Nonequilibrium switching trajectory |
| `US` | Fixed-window umbrella sampling |
| `WT_META` | Well-tempered metadynamics |
| `PATH` | Path construction utility |
| `META_FES` | MTD FES reconstruction |

## Data layout

All simulation outputs are written under `data/1D/<system_slug>/`:

```
data/1D/<system_slug>/
  US/<combo>/raw/seed_<n>/        ← raw umbrella CSVs (deleted after reduction)
  US/<combo>/processed/seed_<n>.dat
  US/<combo>/reduced/seed_<n>.csv
  AUS/<combo>/raw/seed_<n>/
  AUS/<combo>/processed/seed_<n>.dat
  NES/<combo>/context/seed_<n>/   ← EQ endpoint samples (deleted after NEQ runs)
  NES/<combo>/raw/<T>/seed_<n>/
  NES/<combo>/processed/seed_<n>.dat
  MINES/<combo>/raw/seed_<n>/
  MINES/<combo>/processed/seed_<n>.dat
  MTD/<combo>/raw/seed_<n>/
  MTD/<combo>/processed/seed_<n>.dat
  benchmark/selected/             ← final comparison files
  benchmark/figures/
  run_context.json
```

System slug format:
`DoubleWell__k0_<k0>__x0_<x0>__k1_<k1>__x1_<x1>__E1_<E1>__kT_<kT>__dt_<dt>__gamma_<gamma>`
(decimal points → `p`, minus signs → `m`)

## Run

All simulation entry points are in `scripts/`. From the repo root:

```bash
# All methods
RUN_NOTEBOOK=0 bash scripts/run_US_MTD_NES.sh

# One method only
bash scripts/run_US.sh
bash scripts/run_NES.sh
bash scripts/run_MTD.sh
bash scripts/run_AUS.sh
bash scripts/run_MiNES.sh
```

See `scripts/README.md` for the full parameter reference.

For a deeper explanation of the adaptive algorithms, see `AUS_MINES_README.md`.
