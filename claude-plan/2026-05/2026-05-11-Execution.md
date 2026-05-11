# Execution Log — 2026-05-11

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| mines_mean_only_gt_update | Mean-only GT protocol + quantile crossing stop + EQ map | ~55 000 | ~12 000 | 7 changes in script; 2 new notebook cells |
| rescue_neq_gt_instruction | NEQ quadratic fit background for rescue window design | ~62 000 | ~14 000 | New CLI args; 2 new helpers; updated design function; 2 new notebook cells |

---

## [mines_mean_only_gt_update]

**Instruction file:** `claude-plan/2026-05/2026-05-02-claudecode_mines_mean_only_gt_update.md`

### Changes — `scripts/mines_variance_fusion.py`

#### New helper `estimate_mean_only_k0_x0_from_eq_pair` (inserted after `get_k0_x0_harmonic_fromEQ`)

Estimates one shared harmonic background from two EQ windows using force-balance from means only:

```
k0_segment = (k_R*(m_R - x_R) - k_L*(m_L - x_L)) / (m_L - m_R)
x0_segment = m_L + k_L*(m_L - x_L) / k0_segment
```

No standard deviations used. Returns `k0_segment`, `x0_segment`, `valid`, `fallback_used`, `fallback_reason`, `segment_type` (`transition` if `k0 < 0` and `x0 ∈ [m_L, m_R]`, else `regular`), plus window metadata. Fallbacks: `degenerate_means` (`|m_L - m_R| < eps`), `near_zero_k0` (uses midpoint for x0).

#### New helper `get_xs_ks_from_s_mean_only` (inserted after `get_xs_ks_from_s`)

Fixed-background GT step formula using constant `k0_segment, x0_segment` for the whole segment:

```
m_target = (1-s)*m_L + s*m_R
sigma_target = (1-s)*sigma_L + s*sigma_R
K_target = 1/sigma_target^2
k_raw = K_target - k0_segment
k_s = clip(k_raw, k_min, k_max)
x_raw = ((k0_segment + k_s)*m_target - k0_segment*x0_segment) / k_s
x_s = clip(x_raw, x_low, x_high)
```

Mode reported as `"GT_mean_only"`. Returns `x_s`, `k_s`, and metadata dict with all computed quantities. No `x0_s`/`k0_s` s-dependent interpolation.

#### Updated `build_gt_bridge_protocol`

- Calls `estimate_mean_only_k0_x0_from_eq_pair(left_window, right_window)` once to get `k0_segment`, `x0_segment`, `segment_type`.
- Calls `get_xs_ks_from_s_mean_only(...)` instead of `get_xs_ks_from_s(...)` for each step.
- Protocol rows now report `mode = "GT_mean_only"`, `k0_segment`, `x0_segment`, `segment_type`. No `x0_s`, `k0_s`, `x0_L`, `k0_L`, `x0_R`, `k0_R` in rows or metadata.
- `harmonic_{...}` metadata keys replaced with `mo_{...}` keys from `estimate_mean_only_k0_x0_from_eq_pair`.

#### New helpers `neq_eop_quantile_crossing` and `eq_tail_quantile_crossing` (inserted after `endpoint_x_from_trajectories`)

- **`neq_eop_quantile_crossing(segment, q_low=0.05, q_high=0.95)`**: tests `q0.95(forward_EoP) >= q0.05(reverse_EoP)`. Returns `{crossed, fwd_q_high, rev_q_low, q_low, q_high, reason}`.
- **`eq_tail_quantile_crossing(left_window, right_window, q_low=0.05, q_high=0.95)`**: tests `q0.95(left_EQ_tail) >= q0.05(right_EQ_tail)`. Returns `{crossed, left_q_high, right_q_low, q_low, q_high, reason}`.

Both return `crossed=False` with `reason="not_enough_samples"` when fewer than 2 finite samples available.

#### New helper `build_eq_map_segments` (inserted after `build_eq_bar_edge_rows`)

Builds the window-level EQ map:
1. Sort all windows by `(mean_x, name)`.
2. For each consecutive pair, call `estimate_mean_only_k0_x0_from_eq_pair`.
3. Classify `segment_type` (transition / regular).
4. Return rows for `eq_map_segments.csv`.

Columns: `segment_name`, `left_window`, `right_window`, `left_mean_x`, `right_mean_x`, `left_center_x`, `right_center_x`, `left_k`, `right_k`, `k0_segment`, `x0_segment`, `harmonic_valid`, `fallback_used`, `fallback_reason`, `segment_type`, `classification_reason`.

#### Updated `write_state_tables`

Added `eq_map_segments.csv` write immediately after `eq_bar_edges.csv`. Every state snapshot (via `write_state_snapshot → write_state_tables`) now writes this file.

#### Updated growth stopping logic (in main generation loop)

Replaced CFT-threshold stop criterion with quantile crossing:

```python
stop_by_cft = False  # diagnostic only
eop_cross = neq_eop_quantile_crossing(active_segment)
eq_cross  = eq_tail_quantile_crossing(left_frontier, right_frontier)
stop_by_eop      = eop_cross["crossed"]
stop_by_eq       = eq_cross["crossed"]
stop_by_quantile = stop_by_eop or stop_by_eq
```

Stop reason: `eop_quantile_crossing`, `eq_quantile_crossing`, or `eop_and_eq_quantile_crossing`.

`growth_stop_rows` now records: `stop_by_eop_crossing`, `stop_by_eq_crossing`, `forward_eop_q95`, `reverse_eop_q05`, `left_eq_q95`, `right_eq_q05`, `crossing_quantile_low/high`, `growth_stop_reason`, `stop_reason`. CFT fields kept in the row as diagnostics.

`generation_rows` similarly updated: `stop_by_cft = False`, `stop_by_eop_crossing`, `stop_by_eq_crossing`, `growth_stop_reason` added.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### New cells `eq-map-md` [15] and `eq-map-code` [16] (inserted before neq-network section)

- Markdown: explains `segment_type = transition/regular`, `k0_segment`, `x0_segment`.
- Code:
  - Loads `eq_map_segments.csv` via `read_csv_optional`.
  - Displays table with key columns.
  - Bar chart of `k0_segment` vs segment midpoint `mean_x`, colored red (transition) / blue (regular).
  - Scatter of `x0_segment` vs segment midpoint.

---

## [rescue_neq_gt_instruction]

**Instruction file:** `claude-plan/2026-05/2026-05-11-claudecode_rescue_neq_gt_instruction.md`

### Changes — `scripts/mines_variance_fusion.py`

#### New CLI arguments (in `parse_args`)

```
--rescue-background-fit-method {neq,mean-only,auto}   default: neq
--rescue-neq-fit-min-bins 5
--rescue-neq-fit-k0-min-abs 1e-8
--rescue-neq-fit-x0-margin-factor 0.25
```

#### New helper `fit_quadratic_background_from_segment_patch` (inserted after `estimate_mean_only_k0_x0_from_eq_pair`)

Fits a weighted quadratic `F_model(x) = a*x² + b*x + c` to a segment-local NEQ PMF patch via WLS.

- Valid bins: `coverage_mask[i] and finite(pmf[i]) and finite(variance[i])`
- Weights: `w_i = 1 / (variance[i] + variance_floor)`
- WLS via `np.linalg.lstsq` on `sqrt(w)*A` and `sqrt(w)*pmf`
- `k0 = 2*a`, `x0 = -b/(2*a)`, `F0 = c - k0/2*x0²`
- Acceptance: `n_bins >= min_fit_bins`, `|k0| >= k0_min_abs`, finite `k0/x0/F0/rmse`, `x0` within margin of segment boundaries
- Returns: `fit_accepted, fit_source, segment, patch_name, patch_kind, n_fit_bins, x_fit_min, x_fit_max, k0, x0, F0, a, b, c, weighted_rmse, reduced_chi2, variance_floor, fallback_reason`

#### New helper `get_neq_fit_for_rescue` (inserted after `fit_quadratic_background_from_segment_patch`)

Tries patches in order for the segment covering the target:
1. MTS patch from `neq_patch_store[segment.name]` if `segment.mts_patch_built`
2. Forward HS fallback via `bootstrap_hs_patch` (built on-the-fly)
3. Reverse HS fallback via `bootstrap_hs_patch` (built on-the-fly)

Returns `(best_fit_result, all_fit_rows)`. Best accepted fit chosen by lowest `weighted_rmse`. Returns `fit_accepted=False` if no valid patch or no accepted fit.

#### Updated `design_rescue_window_mean_only_gt`

New parameters (all optional for backward compat):
- `all_segments: list[NEQSegment] | None = None`
- `neq_patch_store: dict[str, PMFPatch] | None = None`
- `neq_quad_fit_rows: list[dict[str, Any]] | None = None`
- `rescue_round_root: Path | None = None`

NEQ fit logic (when `rescue_background_fit_method` is `neq` or `auto` and `all_segments`/`neq_patch_store` are provided):
1. Calls `get_neq_fit_for_rescue`; appends fit rows to `neq_quad_fit_rows`
2. If fit accepted: computes GT formula using fitted `k0`, `x0`:
   ```
   s = (m_target - m_L) / (m_R - m_L)   [fallback to 0.5 if out-of-range]
   sigma_target = (1-s)*sigma_L + s*sigma_R
   K_target = 1/sigma_target^2
   k_raw = K_target - k0
   k_rescue = clip(k_raw, k_min, k_max)
   x_raw = ((k0 + k_rescue)*m_target - k0*x0) / k_rescue
   x_rescue = clamp(clamp(x_raw, seg_lo, seg_hi), xmin, xmax)
   ```
   Returns early with `rescue_design_rule = "neq_quadratic_GT"`
3. If fit rejected: falls through to existing mean-only logic; `rescue_design_rule = "mean_only_GT_fallback_neq_fit_rejected"`
4. If `rescue_background_fit_method == "mean-only"` or params not provided: `rescue_design_rule = "mean_only_background_sigma_gt_width"`

All return dicts now include `rescue_background_fit_method`, `rescue_fit_*`, `rescue_gt_*`, and `rescue_design_rule` fields.

#### Updated `_RESCUE_SUMMARY_COLS`

36 new columns added after `mo_fallback_reason`:
```
rescue_background_fit_method, rescue_fit_source, rescue_fit_segment, rescue_fit_patch,
rescue_fit_accepted, rescue_fit_fallback_reason, rescue_fit_n_bins,
rescue_fit_x_min, rescue_fit_x_max, rescue_fit_k0, rescue_fit_x0, rescue_fit_F0,
rescue_fit_weighted_rmse, rescue_fit_reduced_chi2,
rescue_gt_m_L, rescue_gt_m_R, rescue_gt_sigma_L, rescue_gt_sigma_R, rescue_gt_m_target,
rescue_gt_s_raw, rescue_gt_s_used, rescue_gt_sigma_target, rescue_gt_K_target,
rescue_gt_k_raw, rescue_gt_k_final, rescue_gt_x_raw, rescue_gt_x_final,
rescue_gt_x_clipped_to_segment, rescue_gt_x_clipped_to_analysis_range,
rescue_design_rule
```

#### Updated main rescue loop

- Added `neq_quad_fit_rows: list[dict[str, Any]] = []` to tracked lists
- Passes `all_segments`, `neq_patch_store`, `neq_quad_fit_rows`, `rescue_round_root` to `design_rescue_window_mean_only_gt`
- After design call, annotates new fit rows with `round = rescue_counter`
- After `rescue_summary.csv` write: writes `rescue_neq_quadratic_fits.csv` and `rescue_round_<n>/neq_quadratic_fit_summary.json` when fit rows exist
- Writes empty `rescue_neq_quadratic_fits.csv` when no rescue rounds run
- Rescue row spread now includes `rescue_fit_*`, `rescue_gt_*`, `rescue_background_fit_method`, `rescue_design_rule` fields

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### New cells `neq-fit-rescue-md` [26] and `neq-fit-rescue-code` [27] (inserted before final empty cell)

- Markdown: explains `rescue_neq_quadratic_fits.csv`, fit columns, `rescue_design_rule`
- Code:
  - Loads `rescue_neq_quadratic_fits.csv` via `read_csv_optional`
  - Displays table with key fit columns
  - For each accepted fit: plots segment-local NEQ PMF patch + fitted quadratic background, marks `x₀`, `m_target`, `x_rescue`
  - Prints `rescue_design_rule` value counts from `rescue_summary.csv`
  - Gracefully skips when file not found or empty

### Verification

```
syntax: OK
fit_quadratic_background_from_segment_patch: OK
get_neq_fit_for_rescue: OK
design_rescue_window_mean_only_gt updated signature: OK (all_segments, neq_patch_store, neq_quad_fit_rows, rescue_round_root)
NEQ fit path with early return: OK
mean-only fallback path: OK
_RESCUE_SUMMARY_COLS rescue_background_fit_method: OK
_RESCUE_SUMMARY_COLS rescue_fit_source: OK
_RESCUE_SUMMARY_COLS rescue_design_rule: OK
_RESCUE_SUMMARY_COLS rescue_gt_m_L: OK
_RESCUE_SUMMARY_COLS rescue_gt_x_clipped_to_analysis_range: OK
neq_quad_fit_rows tracked in main: OK
rescue_neq_quadratic_fits.csv written: OK
neq_quadratic_fit_summary.json written per round: OK
empty rescue_neq_quadratic_fits.csv when no rounds: OK
notebook cells at idx 26-27: OK
total notebook cells: 29
CLI --rescue-background-fit-method: OK
CLI --rescue-neq-fit-min-bins: OK
CLI --rescue-neq-fit-k0-min-abs: OK
CLI --rescue-neq-fit-x0-margin-factor: OK
chain-growing unchanged: OK (no changes to build_gt_bridge_protocol or growth loop)
```

---

### Verification

```
syntax: OK
estimate_mean_only_k0_x0_from_eq_pair: OK
get_xs_ks_from_s_mean_only: OK
build_gt_bridge_protocol mean_only (no x0_s/k0_s keys): OK
neq_eop_quantile_crossing / eq_tail_quantile_crossing: OK
build_eq_map_segments: OK
eq_map_segments.csv write: OK
quantile stop in growth loop: OK
stop_by_cft demoted to False: OK
notebook eq-map cells at idx 15-16: OK
total notebook cells: 27
```
