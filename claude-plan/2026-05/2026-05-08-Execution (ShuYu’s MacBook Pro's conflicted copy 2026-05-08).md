# Execution Log — 2026-05-08

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| gt_rescue_mean_coordinate_changes | GT rescue, mean_x as core coordinate, EnsembleWindow fields, design_gt_rescue_window | ~70 000 | ~16 000 | 16 plan sections; EnsembleWindow + run_eq_window; eq_gt_tuple; cluster bounds; variance anchors; GT rescue function; notebook; AST OK |
| MINES_notebook_only_display_update | Notebook-only visualization update: plt.close→plt.show, rcParams fonts, grids, x ticks, window table | ~30 000 | ~8 000 | 9 cells updated; no scheduler changes; select_existing_columns + GT cols in rescue/EQ tables |
| claudecode_notebook_rescue_gt_diagnostics | Notebook GT rescue diagnostics: log-scale variance, simplified window table, new GT section | ~35 000 | ~9 000 | positive_for_log; display_existing_columns; add_rescue_gt_derived_columns; 2 new cells; 6 cells updated |

---

## [gt_rescue_mean_coordinate_changes]

**Instruction file:** `claude-plan/2026-05/2026-05-08-gt_rescue_mean_coordinate_changes.md`

### Changes — `scripts/mines_variance_fusion.py`

#### 1. `EnsembleWindow` dataclass — added `mean_x`, `std_x`

```python
@dataclass
class EnsembleWindow:
    ...
    mean_x: float
    std_x: float
    x_most: float    # kept as diagnostic
    ...
```

#### 2. `run_eq_window` — compute `mean_x`, `std_x`, `x_most`

```python
tail_x_finite = tail_x[np.isfinite(tail_x)]
if tail_x_finite.size < 2:
    raise RuntimeError(f"Need at least two finite tail samples for {name}.")
mean_x = float(np.mean(tail_x_finite))
std_x  = float(np.std(tail_x_finite, ddof=1))
x_most = float(mode_x_from_samples(tail_x_finite, grid))
window = EnsembleWindow(..., mean_x=mean_x, std_x=std_x, x_most=x_most, ...)
```

#### 3. `build_window_summary` — extended with new fields

```python
"mean_x": float(window.mean_x),
"std_x":  float(window.std_x),
"x_most": float(window.x_most),
"mean_minus_x_most":   float(window.mean_x - window.x_most),
"mean_minus_center_x": float(window.mean_x - window.center_x),
"x_most_minus_center_x": float(window.x_most - window.center_x),
```

#### 4. `eq_gt_tuple` — uses stored `mean_x`, `std_x`

Before: called `window_tail_mean_sigma(window)` (recomputed from tail).

After:
```python
def eq_gt_tuple(window) -> tuple[float, float, float, float]:
    return (float(window.center_x), float(window.k), float(window.mean_x), float(window.std_x))
```

#### 5. `source_left_anchor` / `source_right_anchor`

Changed `source.x_most` → `source.mean_x` for `EnsembleWindow` case.

#### 6. `build_eq_clusters` — cluster spatial bounds use `mean_x`

```python
left_x  = float(min(w.mean_x for w in current))
right_x = float(max(w.mean_x for w in current))
```

#### 7. New helpers: `rightmost_mean_window`, `leftmost_mean_window`

```python
def rightmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return max(cluster.windows, key=lambda w: float(w.mean_x))

def leftmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return min(cluster.windows, key=lambda w: float(w.mean_x))
```

#### 8. `choose_connected_boundary_pair` — updated to use `mean_x`

- `right_boundary_default = leftmost_mean_window(right_cluster)` (was `right_cluster.windows[0]`)
- `left_boundary = min(connected, key=lambda w: abs(w.mean_x - right_boundary_default.mean_x))` (was `min(connected, key=lambda w: w.center_x)`)
- Fallback: `left_boundary = rightmost_mean_window(left_cluster)` (was `left_cluster.windows[-1]`)
- `boundary_pair_reason` updated to `"existing_connected_segment_to_right_mean_boundary"` / `"mean_coordinate_cluster_boundary_fallback"`
- Metadata extended with `chosen_left_mean_x`, `chosen_right_mean_x`

#### 9. PMF variance anchors — `mean_x` replaces `x_most`

**`build_eq_cluster_patch`**: `bootstrap_direct_eq_mbar(..., float(window.mean_x), ...)` (was `window.x_most`)

**`build_neq_mts_patch`**:
```python
left_reference_x  = float(segment.left_boundary.mean_x)
right_reference_x = float(segment.right_boundary.mean_x)
```
Metadata extended:
```python
"anchor_coordinate": "mean_x",
"left_reference_x_most":  float(segment.left_boundary.x_most),
"right_reference_x_most": float(segment.right_boundary.x_most),
```

**`bootstrap_hs_patch`**: `reference_x = float(segment.left/right_boundary.mean_x)` (was `x_most`)

#### 10. Coverage and gap tests — `mean_x`

**`segment_covering_target`**: boundary interval now `[mean_x_left, mean_x_right]` (was `x_most`)

**`choose_failed_or_skipped_gap_target`**: `left_x / right_x` from `segment.left/right_boundary.mean_x`; added `"gap_coordinate": "mean_x"` to candidate dict.

#### 11. Growth stopping — `mean_x`

```python
frontier_row["frontiers_crossed_diagnostic"] = bool(
    left_frontier.center_x >= right_frontier.center_x
    or left_frontier.mean_x >= right_frontier.mean_x   # was x_most
)
frontier_row["left_frontier_mean_x"]  = float(left_frontier.mean_x)
frontier_row["right_frontier_mean_x"] = float(right_frontier.mean_x)
frontier_row["left_frontier_x_most"]  = float(left_frontier.x_most)
frontier_row["right_frontier_x_most"] = float(right_frontier.x_most)
frontier_row["crossing_coordinate_rule"] = "center_x_or_mean_x"
```

#### 12. `design_gt_rescue_window` — new function

Uses GT math to design rescue windows:

1. `gap_clusters_for_target(x_target, clusters)` — finds adjacent-cluster gap whose mean-based bounds bracket the target.
2. If target is inside a cluster, chooses nearest adjacent pair around it.
3. Fallback to `target_bin_fallback_no_bracketing_clusters` if no valid pair found.
4. Calls `choose_connected_boundary_pair(...)`, `eq_gt_tuple(...)`, `get_k0_x0_harmonic_fromEQ(...)`, `get_xs_ks_from_ms(...)`.
5. Returns `rescue_center_rule = "GT_ms_target_mean_coordinate"` (or midpoint/fallback variants).
6. Returns all GT diagnostics: `gt_left/right_cluster`, `gt_left/right_boundary`, `gt_m_L/R`, `gt_sigma_L/R`, `gt_x0_L/R`, `gt_k0_L/R`, `gt_s_eff`, `gt_used_midpoint_fallback`, `gt_fallback_reason`, `gt_boundary_pair_reason`, `gt_anchor_coordinate = "mean_x"`.
7. Retry scaling disabled: `rescue_k_retry_rule = "disabled_for_GT"`.

#### 13. Rescue loop call site

Changed `design_rescue_window(...)` → `design_gt_rescue_window(...)`.

New argument: `clusters=clusters, segment_store=segment_store` (replaces `generations_root`).

#### 14. `_RESCUE_SUMMARY_COLS` — extended with GT columns

Added 27 new column names between `rescue_scale` and `rescue_tail_q05`:
```
gt_left_cluster, gt_right_cluster, gt_left_boundary, gt_right_boundary,
gt_left_center_x, gt_right_center_x, gt_left_mean_x, gt_right_mean_x,
gt_left_std_x, gt_right_std_x, gt_left_x_most, gt_right_x_most,
gt_m_L, gt_m_R, gt_sigma_L, gt_sigma_R, gt_s_eff, gt_x_raw, gt_k_raw,
gt_x0_L, gt_k0_L, gt_x0_R, gt_k0_R,
gt_used_midpoint_fallback, gt_fallback_reason, gt_boundary_pair_reason, gt_anchor_coordinate
```

Also added `rescue_k_retry_rule` between `rescue_k_rule` and `matched_child_name`.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `eq-ensemble-table-code` (EQ ensemble table)

- Loads `mean_x`, `std_x` from `windows.csv` columns if present, else falls back to `window_summary.json` per window.
- Added derived columns: `mean_minus_x_most`, `mean_minus_center_x`.
- Displayed table now shows `x_m`, `mean_x`, `std_x`, `x_most`, `mean_minus_center_x`, `mean_minus_x_most`.
- **New plot**: `center_x` vs `mean_x` scatter with vertical segment lines showing displacement, colored by `side`.

#### Cell `50f5f325` (rescue summary table)

- Rescue table now includes all GT diagnostic columns: `gt_left/right_cluster`, `gt_left/right_boundary`, `gt_left/right_mean_x`, `gt_left/right_std_x`, `gt_x_raw`, `gt_k_raw`, `gt_used_midpoint_fallback`, `gt_fallback_reason`, `gt_boundary_pair_reason`, `rescue_center_rule`, `rescue_k_rule`, `rescue_k_retry_rule`.

### Verification

```
python -m py_compile scripts/mines_variance_fusion.py  # AST OK
```

`x_most` kept in all summaries as diagnostic. `mean_x` is now the canonical coordinate for:
- EQ cluster bounds
- Boundary window selection (rightmost/leftmost by `mean_x`)
- Source anchors for single-window patches
- EQ and NEQ bootstrap variance anchors
- Segment coverage tests
- Growth stopping condition
- GT rescue target bracketing and `eq_gt_tuple`

---

## [MINES_notebook_only_display_update]

**Instruction file:** `claude-plan/2026-05/2026-05-08-MINES_notebook_only_display_update_instruction.md`

No changes to `scripts/mines_variance_fusion.py`.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `d974aea2` (new — global style cell, inserted after `c4a86f93`)

```python
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
```

Helper functions added:
- `select_existing_columns(df, columns)` — returns subset of columns present in df, gracefully skips missing
- `set_x_coord_ticks(ax, x_min=None, x_max=None, step=1.0)` — sets integer-spaced ticks via `np.arange`

#### Cells with `plt.close()` removed → `plt.show()` added

| Cell ID | Description |
|---|---|
| `2240aee6` | Global PMF / variance / window centers |
| `d0f54c4b` | EQ distributions / NEQ endpoints |
| `eq-ensemble-table-code` | EQ ensemble table + center_x vs mean_x scatter |
| `717191c1` | PMF quality metrics |
| `514a054e` | Max global variance vs iteration |
| `8f30eeb0` | PMF snapshots after full coverage |
| `645c3d81` | Patch coverage + variance heatmaps (2 figures) |
| `66f5959f` | Retained NEQ patches diagnostic |
| `50f5f325` | Frontier JSD / rescue-round variance |

#### Grid lines

Applied globally via `"axes.grid": True, "grid.alpha": 0.3` in rcParams. No per-cell `ax.grid()` calls needed.

#### Legend fontsize overrides removed

All `legend(fontsize=8)` calls replaced with `legend()` or `legend(ncol=2)` so rcParams `legend.fontsize=14` takes effect. Exception: heatmap y-axis patch labels kept at `fontsize=8` (necessarily dense).

#### x-axis ticks at spacing 1.0

`set_x_coord_ticks(ax, x_min, x_max)` added to all reaction-coordinate axes:
- `2240aee6`: axes[0] PMF, axes[1] variance (using `analysis_xmin`/`analysis_xmax` from summary)
- `d0f54c4b`: axes[0] EQ distributions, axes[1] NEQ endpoints
- `140782e7`: axes[0] patch PMFs, axes[1] patch variances, axes[2] patch coverage
- `8f30eeb0`: snapshot PMF plot
- `645c3d81`: `ax2` dominant-source scatter
- `50f5f325`: axes[1] rescue-round variance
- `eq-ensemble-table-code`: center_x vs mean_x scatter (using `ax.get_xlim()` fallback)

#### EQ window table — extended columns

`_desired_eq_cols` now includes `x_target`, `x0`, `k0`, `rescue_center_x`, `rescue_k`, `gt_x0_L`, `gt_k0_L`, `gt_x0_R`, `gt_k0_R`, `gt_s_eff`, `gt_k_raw`, `gt_x_raw`. `select_existing_columns` used so missing columns are silently skipped.

#### Rescue summary table — extended with GT columns

`select_existing_columns` replaces manual list comprehension. Added `gt_x0_L`, `gt_k0_L`, `gt_x0_R`, `gt_k0_R`, `gt_s_eff` to the displayed column set (they were in `_RESCUE_SUMMARY_COLS` from the scheduler but missing from the notebook display).

#### Cell `140782e7` (patch PMFs)

Added `_ax_xmin`/`_ax_xmax` from summary, `set_x_coord_ticks` for all three axes, removed `fontsize=8` from all legend calls, added `plt.show()`.

### Verification

Confirmed zero `plt.close()` calls remain in the notebook (grep returned no output).

---

## [claudecode_notebook_rescue_gt_diagnostics]

**Instruction file:** `claude-plan/2026-05/2026-05-08-claudecode_notebook_rescue_gt_diagnostics.md`

No changes to `scripts/mines_variance_fusion.py`.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `d974aea2` (style + helpers) — extended

Added three new helpers:

- **`positive_for_log(values)`** — returns array copy with non-positive and non-finite values set to NaN, safe for log-scale axes.
- **`display_existing_columns(df, columns, max_rows=None)`** — selects existing columns from df and calls `display()`; silently skips missing columns.
- **`add_rescue_gt_derived_columns(rdf)`** — computes derived interpolation quantities from `rescue_summary.csv` columns: `gt_sigma_s`, `gt_x0_s`, `gt_k0_s`, `gt_m_s` using `gt_s_eff` as the interpolation parameter.

#### Log-scale applied to variance and RMSE axes

| Cell | Axis | Column(s) |
|---|---|---|
| `2240aee6` | axes[1] | `global_variance` |
| `140782e7` | axes[1] | `variance` (per-patch) |
| `717191c1` | axes[1] | `rmse_bestfit` |
| `514a054e` | axes[0] | `max_global_variance` |
| `50f5f325` | axes[1] | `global_variance` (rescue rounds) |

All use `positive_for_log(...)` before plotting and `ax.set_yscale("log")` + `ax.grid(True, which="both", alpha=0.3)`. Annotation loop in `514a054e` updated to check `sv > 0` before annotating on log axis.

#### Cell `eq-ensemble-table-code` — simplified window table

`_desired_eq_cols` replaced with compact `_window_cols`:
```python
["window", "side", "generation", "x_m", "mean_x", "std_x", "k", "q_next", "target_source"]
```
Removed: `x_most`, `mean_minus_center_x`, `mean_minus_x_most`, `barrier_crossing`, `k_rule`, `gt_*` columns.
Uses `display_existing_columns(eq_table, _window_cols)`.

#### New cell `d233b87d` (markdown) — `## Rescue GT diagnostics` heading

Describes the derived GT quantities with a symbol table.

#### New cell `3815abfc` (code) — GT diagnostics

- Calls `_rgt = add_rescue_gt_derived_columns(rescue_df)` (local variable, does not mutate global `rescue_df`)
- Prints warning if any `gt_k0_s ≤ 0` rows exist (masked on log plot)
- Displays detailed GT table (37 columns via `display_existing_columns`)
- Displays compact debug table (17 columns)
- **Plot A**: `x_rescue_target` vs `rescue_center_x` vs `gt_x_raw` per rescue round
- **Plot B**: `rescue_k`, `gt_k_raw`, `gt_k0_s` on log y-scale per rescue round
- Gracefully handles missing `rescue_df` (prints skip message)

### Verification

Confirmed zero `plt.close()` calls in notebook after all edits.
