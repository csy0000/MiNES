# scripts/

Shell and Python entry points for running and analyzing the benchmark.

## Per-method simulation scripts

Each script runs only one method arm. All system and seed parameters are
controlled via environment variables (same as `run_US_MTD_NES.sh`).

| Script | Method | Key env vars |
|---|---|---|
| `run_US.sh` | Umbrella Sampling (US) | `US_K_VALUES_CSV`, `US_DX_VALUES_CSV` |
| `run_AUS.sh` | Adaptive Umbrella Sampling (AUS) | `AUS_ALPHA_VALUES_CSV`, `AUS_QNEXT_VALUES_CSV` |
| `run_NES.sh` | Non-Equilibrium Switching (NES) | `NES_K_VALUES_CSV` |
| `run_MTD.sh` | Well-Tempered Metadynamics (MTD) | `MTD_BIASFACTOR_VALUES_CSV` |
| `run_MiNES.sh` | MiNES (legacy milestone protocol) | `MINES_K_PULL_VALUES_CSV` |

Quick example — run only US with one seed:

```bash
SEEDS_CSV=101 bash scripts/run_US.sh
```

## Combined simulation script

`run_US_MTD_NES.sh` runs all five method arms in sequence. Use the `RUN_*`
flags to enable/disable individual arms:

```bash
RUN_NOTEBOOK=0 bash scripts/run_US_MTD_NES.sh
```

## Analysis scripts

| Script | Purpose |
|---|---|
| `analysis_US_MTD_NES.sh` | Final aggregation pass for one system root (calls `src/analysis/analysis_US_MTD.py finalize`) |

```bash
bash scripts/analysis_US_MTD_NES.sh data/1D/<system_slug>
```

## MiNES variance-fusion scripts

| Script | Purpose |
|---|---|
| `mines_variance_fusion.py` | Python runner for the MiNES variance-fusion protocol |
| `run_mines_variance_fusion.sh` | Single-seed wrapper for `mines_variance_fusion.py` |
| `run_mines_variance_fusion_batch.sh` | Batch wrapper (multiple seeds / parameter sweeps) |
| `plot_MiNES.sh` | Regenerate MiNES overview figure |

```bash
bash scripts/run_mines_variance_fusion.sh \
  --system-root data/1D/<system_slug> \
  --bin simulations/cpp/neq_sim \
  --seed 101 \
  --label mines_vf
```

Add `--quick-test` for a fast smoke-test run.

## Multi-system batch

`run_target_1d_systems.sh` runs the three target 1D systems in sequence.

## Environment variable reference

All scripts accept the same system-level overrides:

| Variable | Default | Meaning |
|---|---|---|
| `POT_K0`, `POT_X0` | `0.05`, `-10.0` | Left-well spring and center |
| `POT_K1`, `POT_X1` | `0.05`, `10.0` | Right-well spring and center |
| `POT_E1` | `0.0` | Right-well energy offset |
| `THERMAL_KT` | `1.0` | Thermal energy |
| `DT` | `0.0005` | Langevin time step |
| `GAMMA` | `1.0` | Friction coefficient |
| `SEEDS_CSV` | `101` | Comma-separated replica seeds |
| `TIME_STEPS_CSV` | log-spaced 1e4–1e7 | Comma-separated budget checkpoints |
