# CLAUDE.md — MiNES Project Guide

## Project Goal

MiNES develops a method to speed up convergence of the potential-of-mean-force (PMF) and find the free-energy minimal path (FMEP) in high-dimensional space. The benchmark compares:

**Non-adaptive baselines:**
- Umbrella sampling (US) — fixed windows, screened over `k` and `dx`
- Nonequilibrium switching (NES) — bidirectional, screened over `k`

**Adaptive methods:**
- Well-tempered metadynamics (WT-MTD) — screened over `biasfactor`
- MiNES — adaptive milestone-chain NEQ, screened over `k_pull`

Target systems at project end:
- 1D: 2-well (DoubleWell1D) and 3-well (TripleWell1D)
- 2D: Muller-Brown potential

---

## Principle Rules

1. **Memory** — Store understanding, notes, and session context in `claude-plan/docs/`. Write anything that helps future sessions pick up without re-reading all files.

2. **Daily instructions** — Instructions live in `claude-plan/yyyy-mm/yyyy-mm-dd-Instruction.md`. Time-coded with `## [hh:mm]`. Longer special instructions may have distinct filenames like `yyyy-mm-dd-OPERATION.md`.

3. **Execution logs** — Write responses in `claude-plan/yyyy-mm/yyyy-mm-dd-Execution.md`, referencing the time code of the instruction being answered. Each execution file must open with a **Token Usage Summary** table (columns: time code, approx. input tokens, approx. output tokens, notes).

4. **Efficiency** — If the instruction does not explicitly ask to run or test scripts, record what to run in `claude-plan/yyyy-mm/yyyy-mm-dd-Operation.md` instead. Feel free to create new project files.

5. **Scope** — Read only same-day instructions unless explicitly told otherwise.

6. **GitHub Actions** — See the [GitHub Actions](#github-actions-suggestions) section below.

---

## Repo Layout

```
MiNES/
├── CLAUDE.md                          ← this file
├── src/
│   ├── cpp/                           ← reusable C++ headers (header-only)
│   │   ├── sim_types.h                ← Vec2
│   │   ├── sim_config.h               ← SimConfig (all run parameters)
│   │   ├── potential.h                ← MullerBrown, DoubleWell1D, ThreeWell (2D), SixHumpCamel
│   │   ├── bias.h                     ← BiasHarmonic, BiasWellTemperedMeta
│   │   ├── path.h                     ← PathData, build_path()
│   │   ├── eq_neq.h                   ← Langevin integrator, EQ/NEQ runners
│   │   ├── fes.h                      ← GridSpec1D/2D, PMF writers
│   │   ├── us.h                       ← UmbrellaSampling runner
│   │   └── benchmark.h                ← BenchmarkSummary
│   └── analysis/                      ← Python PMF analysis modules
│       ├── analysis_US_MTD.py         ← per-seed reduction CLI (REPO_ROOT = parents[2])
│       ├── bidirectional_mts_pmf.py   ← Hummer-Szabo PMF, MTS bootstrap, BAR/CFT delta_f
│       ├── mines_current_protocol_analysis.py ← EQ MBAR, JSD utilities
│       └── mines_notebook_utils.py    ← background potential, coverage mask helpers
├── simulations/
│   ├── cpp/
│   │   ├── neq_sim.cpp                ← CLI entrypoint; includes ../../src/cpp/us.h
│   │   └── neq_sim                    ← compiled binary (git-ignored)
│   ├── adaptive_methods.py            ← Python orchestrator for AUS and legacy MiNES
│   ├── write_method_contexts.py       ← writes method_context.json files
│   └── AUS_MINES_README.md            ← method workflow docs
├── analysis/
│   ├── mines_variance_fusion_visualization.ipynb  ← variance-fusion step-by-step viewer
│   └── notebook/
│       ├── doublewell_benchmark_results.ipynb  ← benchmark results viewer
│       └── plot_doublewell_benchmark.py        ← benchmark figure generator
├── scripts/
│   ├── run_US_MTD_NES.sh              ← combined benchmark runner (all five methods)
│   ├── run_US.sh                      ← US only
│   ├── run_AUS.sh                     ← AUS only
│   ├── run_NES.sh                     ← NES only
│   ├── run_MTD.sh                     ← MTD only
│   ├── run_MiNES.sh                   ← MiNES current-protocol screen
│   ├── analysis_US_MTD_NES.sh         ← final aggregation to benchmark/selected/
│   ├── mines_variance_fusion.py       ← standalone MiNES variance-fusion runner
│   ├── run_mines_variance_fusion.sh   ← single-seed wrapper
│   ├── run_mines_variance_fusion_batch.sh ← batch wrapper
│   ├── run_target_1d_systems.sh       ← three direct-sampling 1D target systems
│   └── plot_MiNES.sh                  ← MiNES overview plot
├── data/                              ← generated at runtime (not in repo)
│   └── 1D/<system_slug>/              ← US/, AUS/, NES/, MINES/, MTD/, benchmark/
├── legacy/
│   └── run_context.json               ← reference system config (DoubleWell1D example)
└── claude-plan/
    ├── docs/                          ← Claude memory and notes
    └── yyyy-mm/                       ← daily instructions and execution logs
```

---

## Build

```bash
conda activate MiNES
clang++ -O2 -std=c++17 simulations/cpp/neq_sim.cpp -o simulations/cpp/neq_sim
```

The binary includes all headers from `src/cpp/` via relative path `../../src/cpp/us.h`.

---

## Key Shell Runners

| Script | Purpose |
|---|---|
| `scripts/run_US_MTD_NES.sh` | Combined benchmark: all five methods in sequence |
| `scripts/run_US.sh` | US only |
| `scripts/run_AUS.sh` | AUS only |
| `scripts/run_NES.sh` | NES only |
| `scripts/run_MTD.sh` | MTD only |
| `scripts/run_MiNES.sh` | MiNES current-protocol screen |
| `scripts/run_mines_variance_fusion.sh` | Standalone variance-fusion runner |
| `scripts/run_target_1d_systems.sh` | Three direct-sampling 1D target systems |
| `scripts/analysis_US_MTD_NES.sh` | Final aggregation pass for one system root |

Environment flags: `RUN_US`, `RUN_AUS`, `RUN_NES`, `RUN_MINES`, `RUN_MTD`, `RUN_NOTEBOOK`, `SEEDS_CSV`.

---

## Current System Configuration (example from `legacy/run_context.json`)

- System: DoubleWell1D, `x0 = -10`, `x1 = 10`, `k0 = k1 = 1`, `E1 = 10`
- `thermal_kT = 1.0`, `dt = 0.0005`, `gamma = 1.0`
- Budget grid: logspace(1e4, 1e7, 21) steps
- PMF grid: `x ∈ [-12, 12]`, `dx = 0.1`; RMSE grid: `x ∈ [-10, 10]`, `dx = 0.2`

---

## Current Focus

1D DoubleWell1D benchmark only. The 3-well and 2D Muller-Brown extensions come later.

---

## 1D Potential Gap (deferred)

The existing `ThreeWell` struct in `src/cpp/potential.h` is **2D** (x, y). A `TripleWell1D` will be needed later for the 3-well benchmark but is deferred until 1D DoubleWell runs cleanly.

---

## GitHub Actions Suggestions

From [claude-code-action](https://github.com/anthropics/claude-code-action/tree/main), the following are relevant for this project:

1. **Automated CI build** — Add `.github/workflows/build.yml` to compile `simulations/cpp/neq_sim.cpp` on every push. Catches C++ compilation regressions early without manual rebuilds.

2. **`@claude` code review** — Configure claude-code-action so that adding `@claude` to a PR or issue triggers a review. Useful for reviewing Python analysis code changes (PMF estimators, JSD utilities) where subtle numerical bugs are easy to miss.

3. **PR auto-review** — Set `auto_review: true` in the action config so that every PR to `main` gets a review pass. Particularly helpful when changing `bidirectional_mts_pmf.py` or `potential.h` where correctness is critical.

4. **Issue-to-code** — Use `@claude` in an issue to implement small, well-specified additions (e.g., "add TripleWell1D to potential.h") and open a PR automatically.
