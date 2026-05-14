# MiNES: Milestoned Nonequilibrium Switching for Adaptive PMF Estimation

## Overview

MiNES is an adaptive method for estimating the potential of mean force (PMF) along a predefined collective variable. It starts from endpoint equilibrium (EQ) ensembles, grows two chains of biased windows toward each other using nonequilibrium switching (NES), and iteratively refines coverage until the full collective-variable range is bridged and uncertainty converges.

Core ideas:

- Start from endpoint EQ ensembles at each boundary of the CV space.
- Grow two chains inward using KL-targeted Gaussian transport (KL-GT) child placement guided by a globally fixed endpoint-anchored width profile.
- Merge overlapping EQ windows into MBAR clusters using pairwise BAR/MBAR overlap (threshold 0.3).
- Use bidirectional NES/MTS patches for disconnected neighboring clusters without NES truncation.
- Fuse provisional PMF patches by inverse-variance-weighted least squares before EQ connectivity is complete; once the EQ network is connected, switch to connected-EQ MBAR only.

---

## Method Stages

### 1. Exploratory Chain Growth

MiNES starts from endpoint EQ windows L0 (at x_min) and R0 (at x_max). Their sampled means and standard deviations define a **globally fixed endpoint-anchored Gaussian width profile**:

```
s(m)        = (m - m_L0) / (m_R0 - m_L0)
sigma_GT(m) = (1 - s(m)) * sigma_L0 + s(m) * sigma_R0
```

This profile is computed once from the first-generation endpoints and is not updated generation by generation, avoiding progressive narrowing of child windows.

The globally fixed `sigma_GT(m)` profile and KL-target rule are used only during exploratory chain growth. Connectivity refinement may use local GT, global-PMF fits, or NES/MTS-supported placement.

#### Basin-like steps (KL-GT target)

For each frontier window with sampled mean `m_i`, MiNES solves a Gaussian KL-distance condition to find the ordinary target mean `m_KL`:

```
KL[ N(m_i, sigma_i^2) || N(m_KL, sigma_GT(m_KL)^2) ] = KL_target
```

`KL_target` is a configurable exploration spacing parameter.

#### Transition segment detection

A local variance-weighted quadratic PMF fit is used to infer a background harmonic:

```
F0(x) = 0.5 * k0 * (x - x0)^2 + C
```

The step is classified as a **transition segment (barrier-like step)** if:

```
k0 < 0   and   min(m_i, m_KL) < x0 < max(m_i, m_KL)
```

This is a local PMF geometry classification, not an EQ connectivity criterion.

#### Transition-segment target rule

For transition segments, MiNES ignores the KL target and uses a **reflected target mean**:

```
m_next   = 2 * x0 - m_i
sigma_next = sigma_GT(m_next)
```

Transition segments are intentionally allowed to remain EQ-disconnected after exploration and are repaired in refinement by bidirectional NES/MTS.

> **Future safeguard**: the reflected target can jump too far if the local negative-curvature fit is noisy. A maximum reflected displacement or trusted-interval clipping may be added later, but it is not part of the current protocol.

#### Bias parameter construction

Bias parameters are obtained by local harmonic inversion:

```
k_raw   = 1 / (beta_eff * sigma_next^2) - k0
x_raw   = ((k0 + k_raw) * m_next - k0 * x0) / k_raw
```

If `k_raw` is outside bounds, spring is clipped and center is recomputed to preserve the target mean:

```
k_child = clip(k_raw, k_min, k_max)
x_child = ((k0 + k_child) * m_next - k0 * x0) / k_child
```

Priority order: (1) preserve target mean `m_next`, (2) keep `k_child` inside bounds, (3) match `sigma_next` only when possible.

#### EQ connectivity

After sampling, EQ windows are merged into MBAR clusters by pairwise BAR/MBAR overlap:

```
neighboring windows are connected if BAR/MBAR overlap O_pair >= 0.3
```

Here `O_pair` is the pairwise off-diagonal overlap between two neighboring biased EQ ensembles, computed from the two-state BAR/MBAR overlap matrix. If directional values are reported, MiNES uses the conservative value `min(O_ij, O_ji)`.

JSD is not used for EQ connectivity decisions (it may still appear as a diagnostic output).

### 2. Adaptive Connectivity Refinement

Once chain growth stops, MiNES enters adaptive connectivity refinement (up to `--max-refinement-rounds` rounds). Each round processes one disconnected neighboring EQ-cluster pair, then rebuilds the cluster graph from scratch before the next round.

Each disconnected pair is classified by a mean-only local harmonic fit between the two boundary windows:

**Transition/barrier-like segment** (`k0 < 0` means the fitted stationary point `x0` is a local maximum / barrier top, and `x0` lies between the two boundary means):

```
If valid bidirectional NES already exists for the pair (both forward and reverse): reuse it.
Else: run new bidirectional NES between the two boundary windows.
Use the full bidirectional NES trajectories directly for CFT/MTS.
No NES truncation, ad-hoc final perturbation, or augmented protocol is applied.
If only one NES direction is available, MiNES runs a new bidirectional NES for the segment.
If MTS succeeds (finite PMF and variance coverage): keep MTS patch.
If MTS fails: fall back to midpoint mean-only GT EQ insertion.
```

**Basin-like segment** (fit not valid, `k0 >= 0`, or barrier top outside the boundary interval):

```
Do not run NES.
Insert a midpoint mean-only GT EQ window directly.
```

Output file: `refinement_summary.csv` (columns include `segment_type`, `nes_action`, `cft_solved`, `mts_patch_built`, `mts_failure_reason`, `fallback_used`, `fallback_rule`).

### 3. Final EQ-Extension Refinement

Once all EQ ensembles are connected into a single MBAR cluster (`len(clusters) == 1`), MiNES enters the final EQ-extension refinement phase. In this phase:

- No new windows are added.
- All windows in the single connected EQ cluster receive additional sampling.
- The global PMF and cluster patches are recomputed after each round.
- The loop stops when either:
  - `max(sqrt(global_variance)) < target_mbar_ddf` over the analysis interval, or
  - The remaining budget cannot afford another full extension round.

The number of extension rounds is **not a method parameter** — it is determined automatically from the remaining budget and `--eq-extension-steps`.

Output file: `final_eq_extension_summary.csv`.

---

## PMF Fusion

All patch PMFs are combined into a single global PMF by solving the inverse-variance-weighted least-squares problem:

```
min_{G, c} Σ_p Σ_{x in S_p} (G(x) - F_p(x) - c_p)^2 / (var_p(x) + ε)
```

where:
- `G(x)` is the global PMF estimate.
- `F_p(x)` is the local patch PMF (EQ-MBAR or NES/MTS).
- `c_p` is a fitted additive offset for patch `p`.
- `var_p(x)` is the patch uncertainty at bin `x`.
- `ε` is a variance floor (`--variance-floor`, default 1e-6).

The global PMF is not hard-stitched — it is a globally consistent inverse-variance-weighted combination.

---

## Final PMF Estimator and Refinement State Machine

MiNES uses nonequilibrium switching to construct and rescue coverage, but the final PMF estimator changes once the equilibrium network is fully connected.

### State A: Disconnected or Partially Connected EQ Network

When the EQ windows form multiple clusters, MiNES treats the PMF as provisional. In this stage, NEQ/MTS bridges, EQ refinement windows, or hybrid patch fusion may be used to grow, connect, and rescue the sampling network. The global PMF in this stage is mainly used for diagnostics and for deciding where additional sampling is needed.

The `--pmf-method` option controls the provisional PMF estimator in State A:

| `--pmf-method` | Description |
|---|---|
| `eq` | EQ-MBAR patches fused with NEQ-derived variance guidance |
| `hybrid` | EQ-MBAR patches combined with NEQ/MTS patches |
| `neq` | NEQ/MTS patches only |

### State B: One Connected EQ Cluster

Once all EQ windows merge into a single connected EQ cluster (`len(clusters) == 1`), MiNES switches to a pure connected-EQ MBAR estimator. In this state:

- The PMF is estimated only from EQ-MBAR patches.
- The uncertainty is estimated only from EQ bootstrap MBAR variance.
- NEQ/MTS patches are retained on disk for diagnostics only.
- NEQ/MTS patches are **not** included in the final PMF fit or variance estimate.

The `--pmf-method` option has no effect in State B. The estimator is always `connected_EQ_MBAR_only`.

The final connected-EQ phase extends every EQ window by `--eq-extension-steps` per round. After every extension round, MiNES recomputes the EQ overlap network. If the network remains connected, MBAR-only refinement continues until either `--target-mbar-ddf` is reached or the simulation budget is exhausted. If the EQ network loses connectivity during extension, MiNES exits MBAR-only refinement, writes diagnostic output (`eq_connectivity_lost.json`), and does not mark the run as converged.

This makes NEQ a construction and rescue tool, while the final converged PMF is a standard connected equilibrium MBAR estimate.

---

## Currently Supported Refinement Modes

```
1. Adaptive connectivity refinement (basin → midpoint EQ; transition → NES/MTS or fallback midpoint EQ)
2. EQ-extension until MBAR uncertainty convergence  (--final-refinement-mode eq-extend)
```

---

## Future Work

### MTD-supported refinement

Future work: initialize a metadynamics flooding potential from the negative of the estimated PMF (represented as a Gaussian mixture or tabulated bias). Use the resulting enhanced-sampling trajectories as additional PMF patches.

*Not currently implemented.*

### WE-supported refinement

Future work: use the estimated PMF and its uncertainty profile to allocate weighted-ensemble trajectories across CV bins, targeting high-variance regions.

*Not currently implemented.*

---

## Running the Workflow

### Minimal example

```bash
python scripts/mines_variance_fusion_v4.py \
  --system-root <path> \
  --bin <path-to-simulator> \
  --seed 123 \
  --label mines_variance_fusion_v4 \
  --total-budget-steps 2500000 \
  --target-kl 1.0 \
  --eq-overlap-threshold 0.3 \
  --max-refinement-rounds 10 \
  --final-refinement-mode eq-extend \
  --target-mbar-ddf 1e-3
```

### Key CLI options

| Option | Default | Description |
|---|---|---|
| `--total-budget-steps` | — | Total simulation budget in steps |
| `--k-max` | 100 | Maximum harmonic spring constant |
| `--k-min` | 1.0 | Minimum harmonic spring constant |
| `--target-kl` | 1.0 | Target directional Gaussian KL distance for basin-like KL-GT exploration steps |
| `--eq-overlap-threshold` | 0.3 | Pairwise BAR/MBAR overlap cutoff for merging neighboring EQ windows |
| `--n-eq-steps` | — | EQ steps per window |
| `--n-neq-traj` | — | NEQ trajectories per segment |
| `--t-neq` | — | NEQ switching time |
| `--max-generations` | 10 | Maximum chain-growth generations |
| `--max-refinement-rounds` | 10 | Maximum connectivity refinement rounds |
| `--final-refinement-mode` | `eq-extend` | Final phase mode (`none` or `eq-extend`) |
| `--target-mbar-ddf` | 1e-3 | MBAR ddF stopping target for EQ extension |
| `--eq-extension-steps` | None | EQ steps per window per extension round (default: `--n-eq-steps`) |
| `--quick-test` | False | Short smoke-test run (disables final EQ extension) |

### Batch runner (t_neq sweep)

```bash
bash scripts/run_mines_variance_fusion_v4_tneq_sweep.sh
```

Sweeps `t_neq = 300, 500, 1000, 2000, 3000, 5000`. Override defaults via environment variables:

```bash
SYSTEM_ROOT=<path> BIN=<simulator> SEED=456 bash scripts/run_mines_variance_fusion_v4_tneq_sweep.sh
```

---

## Important Output Files

| File | Description |
|---|---|
| `global_pmf.csv` | Final fused global PMF with variance |
| `global_fit_summary.json` | Patch fusion fit details and offsets |
| `windows.csv` | All EQ windows (center, spring, mean, std) |
| `clusters.csv` | EQ MBAR clusters |
| `segments.csv` | NEQ bridge segments |
| `patches.csv` | All PMF patches used in fusion |
| `neighbor_eq_overlap.csv` | BAR/MBAR pairwise overlap and connectivity between neighboring EQ windows |
| `refinement_summary.csv` | Adaptive connectivity refinement rounds (segment type, NES action, fallback) |
| `generation_summary.csv` | Per-generation child proposal diagnostics |
| `final_eq_extension_summary.csv` | Final EQ-extension rounds |
| `budget_ledger.csv` | Step-by-step budget accounting |
| `pmf_quality_vs_steps.csv` | PMF quality metrics vs. cumulative steps |
| `mines_variance_fusion_summary.json` | Run metadata and stop reason |
