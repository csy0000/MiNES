# analysis/

Post-processing outputs and notebooks for the 1D DoubleWell benchmark.

## Contents

| Path | Purpose |
|---|---|
| `notebook/doublewell_benchmark_results.ipynb` | Interactive results viewer (reads one `BENCHMARK_SYSTEM_ROOT`) |
| `notebook/plot_doublewell_benchmark.py` | Standalone figure generator |
| `mines_variance_fusion_visualization.ipynb` | MiNES variance-fusion step-by-step visualizer |

## Python modules

All analysis Python modules live in `src/analysis/`:

| Module | Role |
|---|---|
| `analysis_US_MTD.py` | CLI reducer: per-seed and final aggregation for US, AUS, NES, MINES, MTD |
| `bidirectional_mts_pmf.py` | BAR/CFT ΔF solver and Hummer-Szabo MTS bootstrap |
| `mines_current_protocol_analysis.py` | Direct EQ MBAR PMF, JS divergence, bootstrap helpers |
| `mines_notebook_utils.py` | Background potential, coverage mask, mode utilities |

## Run

Finalize one system root (aggregate seeds, rank combos, write `benchmark/selected/`):

```bash
bash scripts/analysis_US_MTD_NES.sh data/1D/<system_slug>
```

Regenerate benchmark figures for one system:

```bash
python3 analysis/notebook/plot_doublewell_benchmark.py \
  --system-root data/1D/<system_slug>
```

Open the notebook:

```bash
BENCHMARK_SYSTEM_ROOT=data/1D/<system_slug> \
  jupyter nbconvert --to notebook --execute --inplace \
  analysis/notebook/doublewell_benchmark_results.ipynb
```

## Benchmark-facing outputs

After `finalize`, the stable comparison files appear under `data/1D/<system_slug>/benchmark/selected/`:

- `us.dat`, `aus.dat`, `nes.dat`, `mines.dat`, `mtd.dat`
- `summary.dat`, `selection.json`

Figures are written to `data/1D/<system_slug>/benchmark/figures/`.
