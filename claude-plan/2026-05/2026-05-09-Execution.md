# Execution Log — 2026-05-09

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| gt_rescue_diagnostic_instruction | Notebook re-application + GT rescue harmonic diagnostics | ~65 000 | ~18 000 | Notebook was overwritten again; re-applied all previous instructions plus today's; 21 cells total |
| mean_only_gt_rescue_instruction | Mean-only GT rescue design in scripts + notebook visualization | ~55 000 | ~14 000 | New function `design_rescue_window_mean_only_gt`; 34 new `mo_` columns; notebook grows to 23 cells |

---

## [gt_rescue_diagnostic_instruction]

**Instruction file:** `claude-plan/2026-05/2026-05-09-claudecode_gt_rescue_diagnostic_instruction.md`

### Context

The notebook had been overwritten again (17-cell version, missing all previous session changes). This execution re-applied all changes from:

1. `2026-05-08-MINES_notebook_only_display_update_instruction.md`
2. `2026-05-08-claudecode_notebook_rescue_gt_diagnostics.md`
3. Today's `2026-05-09-claudecode_gt_rescue_diagnostic_instruction.md`

### Data files checked before writing

- `rescue_summary.csv` — has `gt_left_boundary`, `gt_right_boundary`, `gt_x0_L`, `gt_k0_L`, `gt_x0_R`, `gt_k0_R`, `gt_s_eff` etc.
- `segments.csv` — has `boundary_left`, `boundary_right`, `forward_protocol_diagnostics_file`, `reverse_protocol_diagnostics_file`
- `segments/<SEG>/protocols/forward_protocol_diagnostics.csv` — has `s`, `m_target`, `k0_L`, `x0_L`, `k0_R`, `x0_R`, `k0_s`, `x0_s`, `k_raw`, `k`, `x_raw`, `x`
- `windows.csv` — uses `name` column (not `window`), has `mean_x`, `std_x`, `center_x`

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `d974aea2` (global style + helpers)

Added all helper functions:
- `display_existing_columns(df, columns, max_rows=None)` — graceful column display
- `positive_for_log(values)` — masks non-positive/non-finite for log axes
- `add_rescue_gt_derived_columns(rdf)` — computes `gt_sigma_s`, `gt_x0_s`, `gt_k0_s`, `gt_m_s`
- `read_csv_optional(path)` — returns empty DataFrame if file missing
- `harmonic_curve(x, k0, x0)` — `0.5 * k0 * (x - x0)^2`
- `align_curve_to_pmf_at_x(x_grid, y_curve, pmf_df, x_ref)` — aligns harmonic to PMF at reference x
- `select_gt_target_row(proto_df, target_bin_x)` — finds protocol row with m_target closest to target

#### Cell `2240aee6` (global PMF/variance)

- Variance axis: `positive_for_log(var)` + `set_yscale("log")` + `grid(which="both")`

#### Cell `eq-ensemble-table-code` (EQ window table)

- Simplified window table: only `window, side, generation, x_m, mean_x, std_x, k, q_next, target_source`
- Removed `x_most`, `mean_minus_center_x`, `mean_minus_x_most`, `k_rule`, `barrier_crossing`
- Uses `display_existing_columns` helper
- Removed `fontsize=8` from legend

#### Cell `140782e7` (patch PMFs)

- Variance axis: `positive_for_log()` + `set_yscale("log")` + `grid(which="both")`
- Removed `fontsize=8` from all legends
- Added `set_x_coord_ticks` to all 3 axes
- Added `plt.show()`

#### Cell `717191c1` (quality metrics)

- RMSE axis: `positive_for_log()` + `set_yscale("log")` + `grid(which="both")`

#### Cell `514a054e` (max variance)

- Max variance axis: `positive_for_log()` + `set_yscale("log")` + `grid(which="both")`
- Fixed annotation guard: `if np.isfinite(sv) and sv > 0` (prevents crash on log axis)
- Added `plt.show()`

#### Cell `8f30eeb0` (PMF snapshots)

- Removed `fontsize=8` from legend
- Added `set_x_coord_ticks`
- Added `plt.show()`

#### Cell `645c3d81` (heatmaps)

- Added `plt.show()` for main heatmap figure
- Added `plt.show()` for dominant-source scatter
- Removed `fontsize=8` from ax2 legend
- Added `set_x_coord_ticks` for ax2

#### Cell `66f5959f` (retained NEQ patches)

- Removed `fontsize=8` from legend
- Added `plt.show()`

#### Cell `50f5f325` (frontier JSD / rescue)

- Rescue variance: `positive_for_log()` + `set_yscale("log")` + `grid(which="both")`
- Added `plt.show()`
- Extended rescue summary display with `gt_x0_L`, `gt_k0_L`, `gt_x0_R`, `gt_k0_R`, `gt_s_eff`
- Uses `display_existing_columns` for JSD tables

#### New cell `798133ea` (markdown) — `## Rescue GT diagnostics`

Table of GT interpolation quantity definitions.

#### New cell `2b1baeaf` (code) — Rescue GT diagnostics

- Calls `add_rescue_gt_derived_columns(rescue_df)` to compute `gt_sigma_s`, `gt_x0_s`, `gt_k0_s`, `gt_m_s`
- Detailed GT table (37 columns via `display_existing_columns`)
- Compact debug table (17 columns)
- Plot A: rescue target vs rescue center (with `gt_x_raw`)
- Plot B: stiffness decomposition log-scale (`rescue_k`, `gt_k_raw`, `gt_k0_s`)

#### New cell `d04aab74` (markdown) — `## GT rescue harmonic diagnostics`

Description of the three harmonics shown per rescue round.

#### New cell `f02a7b6b` (code) — GT rescue harmonic diagnostics

For each rescue row:
1. Looks up matching segment by `gt_left_boundary` / `gt_right_boundary` in `segments_df`
2. Loads `forward_protocol_diagnostics.csv` (falls back to reverse)
3. Selects s=0 row, s=1 row, target row (min `|m_target - target_bin_x|`)
4. Extracts `k0_L`, `x0_L`, `k0_R`, `x0_R`, `k0_s`, `x0_s`
5. Computes local plot range around the gap ± 1.5, clipped to analysis bounds
6. Plots three harmonics aligned to global PMF at `target_bin_x`
7. Overlays global PMF, target_bin_x (red), rescue_center_x (orange), rescue_mean_x (green)
8. Prints compact diagnostic table with all harmonic params and clipped values

### Verification

```
0 plt.close() calls
14 plt.show() calls
9 positive_for_log references
6 display_existing_columns calls
2 add_rescue_gt_derived_columns calls
2 harmonic_curve / select_gt_target_row definitions
21 total cells
```

`scripts/mines_variance_fusion.py` — unchanged.

---

## [mean_only_gt_rescue_instruction]

**Instruction file:** `claude-plan/2026-05/2026-05-09-claudecode_mean_only_gt_rescue_instruction.md`

### Changes — `scripts/mines_variance_fusion.py`

#### New function `design_rescue_window_mean_only_gt` (inserted after line 4043)

Full force-balance mean-only GT rescue design. Same cluster/boundary identification as `design_gt_rescue_window`. Departs from old design in the background estimation:

- `k0 = (k_L*(m_L-x_L) - k_R*(m_R-x_R)) / (m_R - m_L)` — force-balance from means only
- `x0 = m_L + (k_L/k0)*(m_L-x_L)` — harmonic center
- `x0_right_check = m_R + (k_R/k0)*(m_R-x_R)` — consistency check
- `s_raw = (x_target - m_L)/(m_R - m_L)`, fallback `s = 0.5` when `s_raw <= 0` or `s_raw >= 1`
- `sigma_s = (1-s)*sigma_L + s*sigma_R` (sigmas used only for width, not curvature)
- `kT_eff_L = (k0+k_L)*sigma_L^2`, `kT_eff_R = (k0+k_R)*sigma_R^2`, `kT_eff = 0.5*(L+R)` with fallback to `ctx["thermal_kT"]`
- `k_res_raw = kT_eff/sigma_s^2 - k0`, clipped to `[k_min, k_max]`
- `x_res_raw = x_target + (k0/k_res)*(x_target - x0)` — signed k0, no abs

Fallback cases:
- `degenerate_neighbor_means` (`|m_R - m_L| < 1e-9`): uses `max(k_rescue, k_min)` for k_res, `x_target` for x_res
- `near_zero_k0_center_shift_disabled` (`|k0| < 1e-12`): `x0=nan`, `x_res_raw = x_target`
- `invalid_sigma_or_kT_eff_for_k_res`: uses `max(k_rescue, k_min)` for k_res
- `no_bracketing_cluster_pair`: entire nan fallback, same as old design

Returns all original `gt_*` fields (backward compat with existing notebook cells) plus 34 new `mo_*` fields.

#### Call site updated

- Line ~4859: `operation` string changed from `"design_gt_rescue_window"` to `"design_rescue_window_mean_only_gt"`
- Line ~4863: function call changed from `design_gt_rescue_window(` to `design_rescue_window_mean_only_gt(`

#### `_RESCUE_SUMMARY_COLS` updated

Added 34 `mo_*` columns after `"gt_anchor_coordinate"`:
`mo_rescue_design_method`, `mo_left_window`, `mo_right_window`, `mo_x_L`, `mo_k_L`, `mo_m_L`, `mo_sigma_L`, `mo_x_R`, `mo_k_R`, `mo_m_R`, `mo_sigma_R`, `mo_x_target`, `mo_s_raw`, `mo_s_used`, `mo_s_fallback_to_midpoint`, `mo_sigma_s`, `mo_sigma_s_fallback_used`, `mo_k0_mean_only`, `mo_x0_mean_only`, `mo_x0_right_check`, `mo_x0_left_right_abs_diff`, `mo_kT_eff_L`, `mo_kT_eff_R`, `mo_kT_eff`, `mo_kT_eff_ratio`, `mo_kT_eff_fallback_used`, `mo_k_res_raw`, `mo_k_res`, `mo_k_res_clipped_to`, `mo_x_res_raw`, `mo_x_res`, `mo_x_res_clipped`, `mo_fallback_reason` (33 named + `mo_sigma_s_fallback_used`)

#### `rescue_row` builder updated

Added `**{k: v for k, v in rescue_design.items() if k.startswith("mo_")}` after the existing `gt_*` spread.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

Added 2 new cells after `f02a7b6b` (index 19 → 20 and 21):

#### Cell `mo-gt-rescue-md` (markdown) — `## Mean-only GT rescue diagnostics`

Description of the four panel elements: global PMF, F₀ harmonic background, U_res rescue umbrella, F₀+U_res biased potential, and vertical markers.

#### Cell `mo-gt-rescue-code` (code) — Mean-only GT rescue diagnostics

For each rescue row:
- Guards against missing `rescue_df` or missing `mo_` columns (prints warning and skips)
- Uses `global_df` for PMF alignment at `x_target`
- Plots global PMF (black), F₀ aligned to PMF (blue dashed), U_res with min=0 (red dashed), F₀+U_res aligned to PMF (green dash-dot)
- Vertical markers for x_target, x_res, m_L, m_R, x_L, x_R
- Compact diagnostic table with 17 mo_ fields displayed via `pd.DataFrame([_diag]).T`

### Verification

```
23 total cells
0 plt.close() calls
14 plt.show() calls
`design_rescue_window_mean_only_gt` defined at line ~4046
call site at line ~4231 (operation string) and ~4235 (function call)
34 new mo_ columns in _RESCUE_SUMMARY_COLS
mo_ spread in rescue_row builder
Python syntax check: OK
```
