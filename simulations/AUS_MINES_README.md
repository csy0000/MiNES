# OBSOLETE NOTE

This document describes the **legacy** MiNES milestone-chain workflow using `simulations/adaptive_methods.py`.
The current MiNES implementation uses `scripts/mines_variance_fusion.py` and has a substantially different design.
See `README.md` and `docs/current_mines_protocol.md` for the current protocol.

**Do not use this file for current implementation decisions.**

---

# Benchmark Method Workflows (Legacy)

This note explains how the legacy 1D benchmark methods worked in the runner:

- `US` = fixed-window umbrella sampling
- `AUS` = adaptive umbrella sampling
- `NES` = nonequilibrium switching
- `MINES` = milestone-based nonequilibrium switching
- `MTD` = two-walker well-tempered metadynamics

The implementation is intentionally pragmatic. It is built on top of the
existing C++ `EQ` and `NEQ` primitives and uses the same streamed
reduce-and-cleanup pattern as the rest of the benchmark.

## 1D System Definition

All of the methods in this document are run on the same underlying 1D DoubleWell
system. The active coordinate is `x`, and the underlying potential is

```text
U(x) = -(1 / beta) * ln(
  exp(-beta * k0 * (x - x0)^2) +
  exp(-beta * k1 * (x - x1)^2 - E1)
)
```

with

```text
beta = 1 / kT
```

Parameter meaning:

- `k0`, `x0`: curvature and center of the left well
- `k1`, `x1`: curvature and center of the right well
- `E1`: energetic offset applied to the right well contribution
- `kT`: thermal energy used both in the dynamics and in the potential mixture

Interpretation:

- the left basin is centered near `x0`
- the right basin is centered near `x1`
- if `E1 > 0`, the right well is thermodynamically less favorable than the left
  one
- the exact benchmark system is chosen by the values written into each
  `run_context.json`

Shared dynamics settings used by the benchmark runner:

- active dimension: `x`
- `thermal_kT = 1.0`
- `dt = 0.0005`
- `gamma = 1.0`

Shared analysis grids:

- PMF storage grid: `x in [-12, 12]` with `dx = 0.1`
- RMSE evaluation grid: `x in [-10, 10]` with `dx = 0.2`

## Where These Methods Run

The entrypoint is
[run_US_MTD_NES.sh](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/simulations/run_US_MTD_NES.sh).

That shell runner writes one `run_context.json` for the selected 1D system and
then launches:

- `US` directly through the C++ binary with `-us_mode`
- `AUS` through
  [adaptive_methods.py](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/simulations/adaptive_methods.py)
  with `run-aus-seed`
- `NES` directly through repeated `EQ` + `NEQ` calls to the C++ binary
- `MINES` through
  [adaptive_methods.py](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/simulations/adaptive_methods.py)
  with `run-mines-seed`
- `MTD` directly through the C++ binary with one walker started from each basin

After each finished raw seed run, the runner immediately calls the reducer
[analysis_US_MTD.py](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/analysis/analysis_US_MTD.py)
and then deletes the large raw block.

## US Workflow

`US` is the fixed-window equilibrium baseline.

### Simulation Flow

For one `US` combo and one seed:

1. Build a uniform line of umbrella centers between the left and right basin.
2. Run one restrained equilibrium trajectory per umbrella center.
3. Save one trajectory file per window plus a `us_windows.csv` summary.

The raw seed output is written under:

- `US/<combo_label>/raw/seed_<seed>/us_windows.csv`
- `US/<combo_label>/raw/seed_<seed>/us_window_<id>.csv`

### Current Parameters

The active `US` controls in `run_context.json` are:

- `k_values`
- `dx_values`
- `total_steps`
- `sample_stride_steps`

Meaning:

- `k`: spring constant for every umbrella in one screened combo
- `dx`: spacing between fixed umbrella centers
- `total_steps`: total budget assigned to the whole umbrella set
- `sample_stride_steps`: retained stride used to determine `us_nout`

### Reduction Flow

For a target benchmark budget `T`, the reducer uses only the first `T / N`
steps from each window, where `N` is the number of umbrellas in that combo. It
then reconstructs the PMF with the benchmark MBAR routine.

The reducer writes:

- `US/<combo_label>/processed/seed_<seed>.dat`
- `US/<combo_label>/reduced/seed_<seed>.csv`

## NES Workflow

`NES` is the direct bidirectional nonequilibrium baseline.

### Simulation Flow

For one `NES` combo, one seed, and one target budget:

1. Run a restrained EQ block at the left basin.
2. Run a restrained EQ block at the right basin.
3. Launch 50 forward switching trajectories.
4. Launch 50 backward switching trajectories.
5. Store one raw `NEQ` block for that specific target budget.

The switching time for each trajectory is `T / 100`, because the benchmark uses
100 trajectories total per target budget.

The raw seed output is written under:

- `NES/<combo_label>/raw/T_<steps>/seed_<seed>/neq_fwd_<i>.csv`
- `NES/<combo_label>/raw/T_<steps>/seed_<seed>/neq_bwd_<i>.csv`

### Current Parameters

The active `NES` controls in `run_context.json` are:

- `k_values`
- `eq_steps`
- `eq_nout`
- `n_traj_per_direction`
- `neq_nout`
- `k_midscale`

Meaning:

- `k`: pulling stiffness used during switching
- `eq_steps`: restrained endpoint equilibration length
- `eq_nout`: saved frames in each endpoint EQ file
- `n_traj_per_direction`: forward and backward shots per target budget
- `neq_nout`: saved frames per switching trajectory
- `k_midscale`: optional midpoint stiffness modifier along the protocol

### Reduction Flow

The reducer processes one target budget at a time. It reads the forward and
backward switching trajectories, reweights them with the stored protocol work,
reconstructs one PMF snapshot with the Hummer-Szabo-style bidirectional
estimator, appends that snapshot to the seed PMF file, and deletes the raw
block.

The reducer writes:

- `NES/<combo_label>/processed/seed_<seed>.dat`
- `NES/<combo_label>/reduced/seed_<seed>.csv`

Only the longest switching-time case retains a reduced trajectory CSV.

## AUS Workflow

`AUS` currently grows one left equilibrium umbrella front and one right
equilibrium umbrella front together.

### Simulation Flow

For one `AUS` combo and one seed:

1. Start one restrained EQ window at the left endpoint and one at the right
   endpoint with `k_0 = 1`.
2. For each frontier separately, reconstruct a PMF from the parent umbrella
   window only, excluding all ancestor windows.
3. For the left frontier, use `q_next` as the target child mean.
4. For the right frontier, use `q_{1-next}` as the target child mean.
5. Compute each parent median `q_0.5`.
6. Propose one left child and one right child from the side-specific median and
   `alpha`.
7. Evaluate the local PMF slope `F'` at each side-specific target quantile.
   - the active code now uses only:
     - `poly_4term_parent`: cubic polynomial over all resolved parent-PMF bins
8. Set each child spring only when the slope sign matches that side:
   - left derives `k_{m+1} = F'(q_next) / (x_{m+1} - q_next)` when
     `F'(q_next) > 0`
   - right derives `k_{m+1} = -F'(q_{1-next}) / (q_{1-next} - x_{m+1})` when
     `F'(q_{1-next}) < 0`
   - otherwise place the child at the target quantile with `k_min`
9. Clamp each child `k` into `[k_min, k_max]` and adjust `x_{m+1}` when the
   clamp is active.
10. Stop before sampling the next pair once
    `q^m_{next,left} > q^m_{1-next,right}`. Otherwise sample both children and
    continue until `max_iterations` is reached.
11. After quantile crossing, reconstruct the matched-chain PMF from the
    retained tail of all sampled windows, evaluate the resolved bins between
    `start_x_left` and `start_x_right`, compute the average ESS in that
    interested region, and require every resolved bin to exceed
    `f_min^ESS * average_ESS`.
12. If the current lowest-ESS resolved bin is still below that fractional
    target, the active code now follows a combo-specific post-match rule.
13. For `poly_4term_parent`, use that lowest-fractional-ESS bin itself as the
    rescue target, estimate the local slope and curvature from a local
    `4`-term polynomial fit, use `k = |curvature| + 10` at the target when
    the fitted curvature is negative with `k_rescue_max = 100`, and otherwise
    reuse the same slope-based center / spring rule as the main growth chain.
14. For `cubic_spline_parent`, first try to extend an already allocated
    umbrella whose latest sampling segment places more than `f_min^ESS` of its
    samples in that bin. If several windows qualify, extend the one with the
    highest such fraction. If no existing umbrella satisfies that rule, fit a
    local cubic spline on `±5` grid points around that low-ESS bin and add one
    rescue umbrella at the most negative-curvature point with
    `k = -curvature + 10`, clamped into `[k_min, k_max]`.
15. Once the interested region clears the fractional ESS target, spend the
    next full cycle by adding one more `eq_steps` block to every allocated
    umbrella, then reconstruct the PMF again and repeat the same ESS checks.

The raw seed output is written under:

- `AUS/<combo_label>/raw/seed_<seed>/aus_windows.csv`
- `AUS/<combo_label>/raw/seed_<seed>/aus_summary.json`
- `AUS/<combo_label>/raw/seed_<seed>/window_<id>/eq_window.csv`

### Current Parameters

The active `AUS` controls in `run_context.json` are:

- `q_next_values`
- `alpha_values`
- `k_min_values`
- `k_max_values`
- `grid_dx`
- `start_x_left`
- `start_x_right`
- `endpoint_k`
- `eq_steps`
- `eq_nout`
- `analysis_tail_fraction`
- `max_iterations`
- `refine_metric`
- `refine_half_width_points`
- `refine_k_addition`
- `refine_ess_min_fraction`

Meaning:

- `q_next`: parent quantile targeted as the next child mean
- `alpha`: factor that moves the next equilibrium position beyond the median
- `k_min`, `k_max`: stiffness clamps for the child window
- `grid_dx`: placement grid spacing used to snap the next center
- `start_x_left`, `start_x_right`: starting centers for the two endpoint
  fronts
- `endpoint_k`: spring constant used for the first umbrella
- `eq_steps`: restrained EQ steps per adaptive window
- `eq_nout`: number of stored frames per adaptive window
- `analysis_tail_fraction`: retained tail fraction used for MBAR PMF analysis
- `max_iterations`: hard cap on adaptive growth depth
- `refine_metric`: current post-match certainty metric; active value is
  `interested_region_fraction_of_average_ess`
- `refine_half_width_points`: local PMF neighborhood size used for the rescue
  local fit
- `refine_k_addition`: additive stiffness term used when converting negative
  fitted curvature into the rescue umbrella spring
- `refine_rescue_k_max`: maximum allowed spring for the polynomial
  target-bin rescue path
- `refine_ess_min_fraction`: minimum allowed fraction of the interested-region
  average ESS that every resolved bin must exceed before redistribution starts

### Reduction Flow

The reducer command is:

```bash
python3 analysis/analysis_US_MTD.py \
  process-aus-seed \
  --system-root /abs/path/to/data/1D/<system_slug> \
  --combo-label qnext_<...>__alpha_<...>__kmin_<...>__kmax_<...> \
  --seed <seed>
```

The reducer:

1. Reads `aus_windows.csv`.
2. Reopens each adaptive window trajectory.
3. For each target budget `T` on the benchmark time grid, includes only windows
   whose cumulative start time is already within budget.
4. Truncates each included window to the first part of that window that fits
   inside `T`.
5. Keeps only the last `analysis_tail_fraction` of each truncated window.
6. Reuses the same MBAR routine used by fixed-window `US` to reconstruct the
   PMF on the 1D grid.
7. Writes:
   - `AUS/<combo_label>/processed/seed_<seed>.dat`
   - `AUS/<combo_label>/reduced/seed_<seed>.csv`
8. Deletes `AUS/<combo_label>/raw/seed_<seed>/`

The reduced CSV is only a compact retained sample record. The benchmark-facing
PMF data is in `processed/seed_<seed>.dat`.

## MiNES Workflow

`MINES` grows a milestone chain inward from both endpoints using restrained
equilibrium windows plus weak bidirectional switching.

### Simulation Flow

For one `MINES` combo and one seed:

1. Start one restrained EQ milestone ensemble at the left basin center.
2. Start one restrained EQ milestone ensemble at the right basin center.
3. Launch one parent bidirectional switching block between the current left and
   right frontiers.
4. From that parent block, estimate:
   - left-side reachable region
   - right-side reachable region
   - overlap of forward and reverse work distributions
5. If the fronts are not yet connected, place new milestone ensembles at the
   newly reachable left and right points.
6. Repeat until one of these happens:
   - endpoint fronts are already connected by EQ overlap
   - parent NEQ connectivity looks strong enough
   - no new milestone can be placed
   - `max_depth_per_side` is reached
7. After the milestone chain exists, launch adjacent bidirectional switching
   blocks along the final nearest-neighbor chain.

The raw seed output is written under:

- `MINES/<combo_label>/raw/seed_<seed>/milestones.csv`
- `MINES/<combo_label>/raw/seed_<seed>/edges.csv`
- `MINES/<combo_label>/raw/seed_<seed>/mines_summary.json`
- `MINES/<combo_label>/raw/seed_<seed>/parent_<i>/`
- `MINES/<combo_label>/raw/seed_<seed>/adjacent_<i>/`

For parent edges, the adaptive runner also writes compressed accumulator files:

- `forward_accumulators.json`
- `backward_accumulators.json`

These keep `H`, `Q`, `N`, `ESS`, and final work values for the parent edge.

### Current Parameters

The active `MINES` controls in `run_context.json` are:

- `k_pull_values`
- `grid_dx`
- `eq_steps`
- `eq_nout`
- `n_traj_per_direction`
- `t_neq`
- `neq_nout`
- `ess_min`
- `overlap_min`
- `work_overlap_min`
- `max_depth_per_side`

Meaning:

- `k_pull`: harmonic pulling stiffness used for milestone EQ and local NEQ edges
- `grid_dx`: frontier-placement grid spacing used for reach / overlap decisions
- `eq_steps`: restrained EQ steps per milestone ensemble
- `eq_nout`: stored frames per milestone EQ block
- `n_traj_per_direction`: number of forward and backward switching trajectories
  per edge
- `t_neq`: switching length for each edge trajectory
- `neq_nout`: saved frames per switching trajectory
- `ess_min`: support threshold for deciding how far a parent edge resolves
- `overlap_min`: equilibrium overlap threshold for declaring neighboring
  milestones connected
- `work_overlap_min`: forward/reverse work-overlap threshold for declaring NEQ
  connectivity
- `max_depth_per_side`: hard cap on milestone insertion depth from each side

### Reduction Flow

The reducer command is:

```bash
python3 analysis/analysis_US_MTD.py \
  process-mines-seed \
  --system-root /abs/path/to/data/1D/<system_slug> \
  --combo-label k_pull_<...> \
  --seed <seed>
```

The reducer:

1. Reads `milestones.csv` and `edges.csv`.
2. Reconstructs local PMF segments from each raw adjacent edge using the same
   Hummer-Szabo-style time-slice reweighting used in the `NES` path.
3. Estimates adjacent milestone offsets with a pairwise BAR-style solver from
   the forward and backward work values.
4. Builds a milestone free-energy scaffold by summing those adjacent offsets.
5. Aligns the local PMF segments onto that scaffold.
6. Combines the aligned segments into one PMF estimate for each target budget.
7. Writes:
   - `MINES/<combo_label>/processed/seed_<seed>.dat`
   - `MINES/<combo_label>/reduced/seed_<seed>.json`
8. Deletes `MINES/<combo_label>/raw/seed_<seed>/`

The reduced JSON keeps:

- milestone metadata
- edge metadata
- pairwise forward/backward work arrays
- parent-edge compressed accumulators where they exist

So the storage contract is: keep the reduced milestone/work objects, not the
full raw switching trajectories.

## MTD Workflow

`MTD` is the history-dependent adaptive baseline with two walkers.

### Simulation Flow

For one `MTD` combo and one seed:

1. Start one walker at the left basin.
2. Start one walker at the right basin.
3. Let both walkers deposit hills with the same metadynamics settings.
4. Save:
   - one trajectory file per walker
   - one hills file per walker

The raw seed output is written under:

- `MTD/<combo_label>/raw/seed_<seed>/left/meta_traj.csv`
- `MTD/<combo_label>/raw/seed_<seed>/left/meta_hills.csv`
- `MTD/<combo_label>/raw/seed_<seed>/right/meta_traj.csv`
- `MTD/<combo_label>/raw/seed_<seed>/right/meta_hills.csv`

### Current Parameters

The active `MTD` controls in `run_context.json` are:

- `biasfactor_values`
- `total_steps`
- `per_walker_steps`
- `sample_stride_steps`
- `meta_nout`
- `w0`
- `sigma`
- `stride`

Meaning:

- `biasfactor`: well-tempered biasfactor
- `total_steps`: total two-walker budget
- `per_walker_steps`: budget per walker
- `sample_stride_steps`: retained stride for the trajectory file
- `meta_nout`: saved frames per walker trajectory
- `w0`: initial hill height
- `sigma`: hill width in the 1D coordinate
- `stride`: hill deposition stride

### Reduction Flow

For a target benchmark budget `T`, the reducer uses only the first `T / 2`
steps from each walker, reconstructs the accumulated metadynamics bias from the
hills, converts that to a PMF estimate, writes the seed PMF trace, keeps one
reduced 1000-sample trajectory CSV, and deletes the raw block.

The reducer writes:

- `MTD/<combo_label>/processed/seed_<seed>.dat`
- `MTD/<combo_label>/reduced/seed_<seed>.csv`

## Final Aggregation

After all seed-level reductions exist, the final pass:

```bash
bash analysis/analysis_US_MTD_NES.sh /abs/path/to/data/1D/<system_slug>
```

This writes stable benchmark-facing outputs under `benchmark/selected/`:

- `us.dat`
- `aus.dat`
- `nes.dat`
- `mines.dat`
- `mtd.dat`
- `summary.dat`
- `selection.json`

The reducer ranks every screened combo within each method against the analytic
1D PMF and then promotes the best combo into the selected files above.

## Current Limitations

These adaptive methods are first-pass benchmark implementations.

Important current limits:

- `AUS` and `MINES` are 1D only.
- `AUS` currently uses a simple frontier-support rule on an adaptive placement
  grid with default `dx = 0.05`.
- `MINES` currently assumes a nearest-neighbor milestone chain.
- `MINES` uses the same explicit placement grid idea, with default `dx = 0.05`.
- `MINES` uses a pragmatic pairwise BAR-style edge offset plus local
  Hummer-Szabo segment assembly, not a full multistate NEQ estimator.
- Only a short adaptive smoke run has been validated so far in this branch; the
  full three-system production rerun with default adaptive screens has not been
  completed yet.

## Side-By-Side Comparison

| Method | Main idea | Raw simulation structure | PMF reconstruction | Adaptive? | Retained reduced file |
| --- | --- | --- | --- | --- | --- |
| `US` | Fixed equilibrium umbrella grid | One EQ trajectory per fixed umbrella window | MBAR on truncated window samples | No | `US/<combo>/reduced/seed_<seed>.csv` |
| `AUS` | Endpoint-grown equilibrium umbrella chain | One EQ trajectory per adaptive window as the chain grows inward | Same MBAR path as `US`, but on adaptive windows | Yes | `AUS/<combo>/reduced/seed_<seed>.csv` |
| `NES` | Direct bidirectional switching between endpoints | 50 forward + 50 backward switching trajectories for each target budget | Hummer-Szabo-style bidirectional reweighting from protocol work | No | `NES/<combo>/reduced/seed_<seed>.csv` only for longest budget |
| `MINES` | Endpoint-grown milestone chain with local NEQ edges | Parent and adjacent bidirectional switching blocks along a milestone chain | Local HS segments plus pairwise BAR-style milestone offsets | Yes | `MINES/<combo>/reduced/seed_<seed>.json` |
| `MTD` | Two-walker history-dependent biasing | One left walker and one right walker with hill deposition | PMF from accumulated metadynamics bias | Yes | `MTD/<combo>/reduced/seed_<seed>.csv` |

Practical interpretation:

- `US` is the simplest equilibrium reference.
- `AUS` is the adaptive equilibrium version of that idea.
- `NES` is the simplest direct nonequilibrium reference.
- `MINES` is the adaptive multistage nonequilibrium version.
- `MTD` is the history-dependent adaptive baseline, not a switching-chain
  method.

## Fast Reference

If you only want to know where to look:

- adaptive simulation orchestration:
  [adaptive_methods.py](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/simulations/adaptive_methods.py)
- per-seed reduction:
  [analysis_US_MTD.py](/Users/shuyuchen/Dropbox/ETH/Work/ellipse-bias-noneq-sim/analysis/analysis_US_MTD.py)
- method and combo metadata:
  `US/method_context.json`, `US/<combo>/combo_context.json`,
  `AUS/method_context.json`, `AUS/<combo>/combo_context.json`,
  `NES/method_context.json`, `NES/<combo>/combo_context.json`,
  `MINES/method_context.json`, `MINES/<combo>/combo_context.json`,
  `MTD/method_context.json`, `MTD/<combo>/combo_context.json`
- selected benchmark outputs:
  `benchmark/selected/us.dat`, `benchmark/selected/aus.dat`,
  `benchmark/selected/nes.dat`, `benchmark/selected/mines.dat`,
  `benchmark/selected/mtd.dat`
