# Execution Log — 2026-05-07

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| mines_protocol_update_2026_05_07 | PMF method selection, CFT/BAR growth stop, continuous centers, barrier_crossing removal | ~35 000 | ~9 000 | Read plan + py; 7 targeted edits; AST OK |
| claudecode_mines_sampling_visualization_changes | Rescue target selection, PMF coverage, patch diagnostics, notebook visualization | ~90 000 | ~18 000 | Parts 1–4 + 5 (AST OK); notebook 6 cells added/updated |
| include_all_neq_patches_fix | Persistent NEQ patch store, all-patch global fit, all_neq_patches.csv, patches_used.csv, notebook diagnostic | ~55 000 | ~12 000 | neq_patch_store + neq_patch_status; 8 new/updated functions; 2 new CSVs; 1 notebook cell; AST OK |

---

## [mines_protocol_update_2026_05_07]

**Instruction file:** `claude-plan/2026-05/2026-05-07-mines_protocol_update_2026_05_07.md`

### Changes — `scripts/mines_variance_fusion.py`

#### 1. New CLI arguments

```python
parser.add_argument("--pmf-method", choices=["neq", "eq", "hybrid"], default="neq")
parser.add_argument("--cft-ddf-threshold", default=1.0, type=float)
```

#### 2. Validate `t_neq > 0`

Added immediately after `apply_quick_test_overrides(parse_args())`:

```python
if int(args.t_neq) <= 0:
    raise ValueError("--t-neq must be > 0 because NEQ is required for offspring proposal, CFT/BAR stopping, and variance estimation.")
```

#### 3. Remove grid snapping from umbrella centers

**`finalize_child_proposal`**: removed `nearest_grid_value` calls; center is now a continuous float clipped only to the feasible progress interval:
```python
center_x = float(center_candidate)  # continuous, no grid snap
```

**`clamp_to_bounds_and_grid`** renamed to **`clamp_to_bounds`**; removed `nearest_grid_value` from body:
```python
def clamp_to_bounds(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))
```

**`design_rescue_window`**: replaced `nearest_grid_value(x_m, grid)` with `float(x_m)` and `nearest_grid_value(x_rescue_target, grid)` with `float(x_rescue_target)`. Updated `clamp_to_bounds_and_grid(...)` call to `clamp_to_bounds(...)`.

**`choose_uncovered_rescue_target`**: replaced two `nearest_grid_value` calls with direct `min(max(...))` clipping to analysis bounds.

**`choose_failed_or_skipped_gap_target`**: replaced `nearest_grid_value(midpoint, grid)` with `float(midpoint)`.

#### 4. Remove barrier-crossing k override

Old code used `if barrier_crossing: k = k_min; k_rule = "barrier_crossing_k_min"`.

New code always uses:
```python
k_value = float(min(max(raw_k, k_min), k_max))
k_rule = "force_matching_clipped"
barrier_crossing_action = "none_gt_slope_aware"
```

Return dict field renamed from `"barrier_crossing"` to `"barrier_crossing_diagnostic"` (value unchanged) and `"barrier_crossing_action"` added. Downstream references updated (`generation_row`, `load_child_design_records`).

#### 5. Add `compute_segment_cft_summary` helper

Inserted before `build_neq_mts_patch`:

```python
def compute_segment_cft_summary(segment: NEQSegment, ctx: dict[str, Any]) -> dict[str, Any]:
    forward_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    reverse_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    _x_forward, work_forward = trajectory_frames_to_arrays(forward_frames)
    _x_reverse, work_reverse = trajectory_frames_to_arrays(reverse_frames)
    cft = solve_segment_cft_delta_f_once(work_forward, work_reverse, kT=float(ctx["thermal_kT"]))
    return {
        "cft_solved_once": bool(cft.get("cft_solved", False)),
        "cft_delta_f": cft.get("delta_f", None),
        "cft_delta_f_unc": cft.get("delta_f_unc", None),
        "cft_method": cft.get("method", "BAR"),
        "cft_reason": cft.get("reason", ""),
    }
```

#### 6. Replace growth stop rule with CFT/BAR threshold

**Removed** stop-on-`frontiers_crossed` and stop-on-`frontiers_overlap` breaks. These are now stored as diagnostics in `frontier_row` only:
```python
frontier_row["frontiers_crossed_diagnostic"] = ...
frontier_row["frontiers_overlap_diagnostic"] = ...
```

**Added** CFT-based stop after NEQ runs (and after `ensure_segment_connectivity`):
```python
cft_now = compute_segment_cft_summary(active_segment, ctx)
cft_delta_f_now = cft_now.get("cft_delta_f")
stop_by_cft = bool(
    cft_now["cft_solved_once"]
    and cft_delta_f_now is not None
    and math.isfinite(float(cft_delta_f_now))
    and float(cft_delta_f_now) < float(args.cft_ddf_threshold)
)
growth_stop_rows.append({...})
if stop_by_cft:
    frontier_row["decision"] = "stop"
    frontier_row["reason"] = "cft_delta_f_below_threshold"
    stop_reason = "cft_delta_f_below_threshold"
    break
frontier_row["decision"] = "grow"
frontier_row["reason"] = "continue_growth" / "continue_growth_partial_neq"
```

Stop criterion uses `cft_delta_f < cft_ddf_threshold` (not `abs()`, not `cft_delta_f_unc`).

#### 7. Add `build_eq_pmf_with_neq_variance` helper

Inserted before `reconstruct_chain`. For each EQ_MBAR patch, takes NEQ_MTS variance from the global NEQ variance union; marks variance NaN where NEQ coverage is absent.

#### 8. PMF patch selection in `reconstruct_chain`

Added before `fit_global_pmf_from_patches`:

```python
pmf_method = str(getattr(args, "pmf_method", "hybrid"))
if pmf_method == "neq":
    patches_for_global = [p for p in patches if p.kind == "NEQ_MTS"]
    if not patches_for_global:
        raise RuntimeError("pmf_method=neq but no NEQ_MTS patches are available.")
elif pmf_method == "hybrid":
    patches_for_global = list(patches)
elif pmf_method == "eq":
    patches_for_global = build_eq_pmf_with_neq_variance(eq_patches, neq_patches, grid)
```

`fit_details` extended with:
```python
fit_details["pmf_method"] = pmf_method
fit_details["patch_selection_rule"] = "only_NEQ_MTS" | "EQ_MBAR_plus_NEQ_MTS" | "EQ_MBAR_pmf_with_EQ_NEQ_variance"
fit_details["variance_source"] = "NEQ_MTS_bootstrap" | "hybrid_patch_variance" | "EQ_NEQ_variance"
```

#### 9. New output columns and files

**`generation_summary.csv`** — new columns:
```
pmf_method, cft_solved_once, cft_delta_f, cft_delta_f_unc, cft_method, cft_reason,
cft_ddf_threshold, stop_by_cft, left_center_raw, left_center_x, right_center_raw, right_center_x
```

**`growth_stop_summary.csv`** — new file with columns:
```
generation, stage, active_segment, cft_solved_once, cft_delta_f, cft_delta_f_unc,
cft_method, cft_reason, cft_ddf_threshold, stop_by_cft, stop_reason, used_steps
```

**`run_request.json`** — new fields:
```json
{"pmf_method": "neq", "cft_ddf_threshold": 1.0, "t_neq_validation": "required_positive",
 "window_center_rule": "continuous_clipped_not_grid_snapped", "barrier_crossing_rule": "disabled_gt_slope_aware"}
```

**`mines_variance_fusion_summary.json`** — new fields:
```json
{"pmf_method": "neq", "cft_ddf_threshold": 1.0, "growth_stop_rule": "cft_delta_f_below_threshold",
 "window_center_rule": "continuous_clipped_not_grid_snapped", "barrier_crossing_rule": "disabled_gt_slope_aware"}
```
`"growth_stop_summary"` added to `summary_files`.

### Verification

```
python -c "import ast; ast.parse(open('scripts/mines_variance_fusion.py').read())"  # AST OK
python scripts/mines_variance_fusion.py --help  # shows --pmf-method, --cft-ddf-threshold
```

---

## [claudecode_mines_sampling_visualization_changes]

**Instruction file:** `claude-plan/2026-05/2026-05-07-claudecode_mines_sampling_visualization_changes.md`

### Changes — `scripts/mines_variance_fusion.py`

#### Part 1: Rescue target selection

**`choose_uncovered_rescue_target`** — min-size threshold already applied in previous session.

**`design_rescue_window`** — already rewritten in previous session:
- Center = `target_bin_x` (nearest grid point to `x_rescue_target`, clamped to analysis bounds)
- Sigma-based stiffness: `sigma_target = max(1.5*bin_width, 0.20)`, `k_from_sigma = kT/sigma_target²`, `rescue_k_base = max(args.k_rescue, k_from_sigma)`, `rescue_k = clamp(rescue_k_base * s_rescue^n_retry, k_min, k_max)`
- `matched_child_used_for_center = 0` (diagnostic only)
- New return fields: `target_bin_x`, `sigma_target`, `k_from_sigma`, `rescue_k_base`, `matched_child_used_for_center`

**Rescue loop in `main()`**:
- Added `ctx=ctx` to `design_rescue_window` call
- Added post-EQ tail diagnostics after `rescue_window = run_eq_window(...)`:
  - Computes `rescue_tail_q05/q50/q95/min/max/mean/std`, `rescue_tail_contains_target_bin`, `target_bin_tail_count`, `target_bin_tail_fraction`
  - A tail sample is in the target bin if `abs(x - target_bin_x) <= 0.5 * grid_dx`
- Updated `rescue_row` dict with new fields from `rescue_design` and `_tail_stats`
- Added module-level `_RESCUE_SUMMARY_COLS` and `_PMF_QUALITY_COLS` constants
- Both `rescue_summary.csv` writes (in-loop and empty-run) use `_RESCUE_SUMMARY_COLS`

**`count_previous_rescue_retries`** — updated for Part 1.6:
- Only counts previous attempts at the same target where `rescue_tail_contains_target_bin == False` or `target_bin_tail_fraction < 0.05`
- Successful rescues at the same target don't inflate the retry counter (k not over-scaled)

#### Part 2: PMF coverage metrics

**`compute_pmf_quality_metrics`** — updated:
- `analysis_mask` uses `half_dx = 0.5 * grid_dx` tolerance around bounds (fixes endpoint artifact)
- Added: `n_uncovered_bins`, `first_uncovered_x`, `last_uncovered_x`, `uncovered_x_values` (semicolon-separated, max 50 values)
- Added: `max_global_variance`, `x_at_max_global_variance`, `max_global_std`
- `variance_mask` now requires `analysis_mask & np.isfinite(global_pmf) & np.isfinite(global_variance)` (not just `analysis_mask`)
- All 3 `pmf_quality_vs_steps.csv` writes now use `ordered_fieldnames(quality_rows, extras=_PMF_QUALITY_COLS)`

#### Part 3: Patch/variance diagnostics

**New helper `_best_variance_patch_info(idx, patches, patch_offsets)`** — finds the patch with smallest variance covering bin `idx`.

**`write_global_outputs`** — updated:
- New optional `patches` parameter
- When patches are provided, adds per-bin columns: `best_variance_patch`, `best_variance_patch_kind`, `best_variance_value`, `n_eq_covering_patches`, `n_neq_covering_patches`

**`write_state_tables`** — now passes `patches=patches` to `write_global_outputs`

**New helper `write_patch_bin_contributions(out_root, grid, patches, fit_details)`**:
- Writes `patch_bin_contributions.csv` with columns: `x`, `patch_name`, `patch_kind`, `covered`, `local_pmf`, `variance`, `aligned_pmf`, `patch_offset`
- Called from `write_state_tables` after `write_global_outputs`

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `e7eff267` (data loading):
- Added loading of `generation_summary.csv`, `patch_bin_contributions.csv`
- Loads PMF snapshots from `snapshots/growth_reconstruct/` and `rescue/rescue_round_*/` into `_snapshot_pmfs` dict

#### Cell `2240aee6` (global PMF/variance plots) — Part 4.5:
- PMF plot: adds rescue target overlays (`target_bin_x` red dotted, `rescue_center_x` orange dash-dot)
- Variance plot: same overlays at higher alpha; adds uncovered interval spans

#### Cell `d0f54c4b` (EQ/NEQ distributions) — Part 4.6:
- For rescue windows: overlays `target_bin_x` (red dotted) and tail q05–q95 band (blue)
- Title updated to explain the overlays

#### Cell `717191c1` (quality metrics plots):
- Extended to 3 subplots; adds `n_uncovered_bins` vs ksteps as third panel

#### New cell `514a054e` (after `717191c1`) — Part 4.3: Max variance vs iteration:
- Two panels: `max_global_variance` and `x_at_max_global_variance` vs stage index

#### New cell `8f30eeb0` (after `514a054e`) — Part 4.2: PMF snapshots after full coverage:
- Identifies stages with `coverage_fraction >= 0.999`, loads matching snapshot PMFs
- Plots aligned PMFs (min over analysis interval shifted to 0) with colormap

#### New cell `645c3d81` (after `8f30eeb0`) — Part 4.7: Patch coverage heatmaps:
- If `patch_bin_contributions.csv` exists: coverage heatmap (Blues) and log10(variance) heatmap (YlOrRd)
- Dominant variance source scatter from `best_variance_patch_kind` in `global_pmf.csv`

#### Cell `50f5f325` (rescue summary + JSD tables) — Part 4.4:
- Updated rescue table columns to use new field names: `target_bin_x`, `sigma_target`, `k_from_sigma`, `rescue_k_base`, `rescue_retry_count`, `rescue_tail_*`, `matched_child_used_for_center`
- Added `tail_failed` derived column; prints count of failed rows; displays failed rows separately

### Verification

```
python -m py_compile scripts/mines_variance_fusion.py  # AST OK
```

Post-hoc data check: all existing files load correctly. New columns and `patch_bin_contributions.csv` will appear in future runs; notebook handles missing columns gracefully with warnings/fallbacks.

---

## [include_all_neq_patches_fix]

**Instruction file:** `claude-plan/2026-05/2026-05-07-include_all_neq_patches_fix.md`

### Changes — `scripts/mines_variance_fusion.py`

#### 1. Persistent NEQ patch dictionaries in `main()`

Added alongside `segment_store`:

```python
neq_patch_store: dict[str, PMFPatch] = {}
neq_patch_status: dict[str, dict[str, Any]] = {}
```

Both are passed into every `reconstruct_chain()` call and persist across all growth and rescue rounds.

#### 2. `reconstruct_chain()` signature and return value

New parameters: `neq_patch_store`, `neq_patch_status`.

Return tuple extended to 8 elements:

```python
return clusters, segments, patches, global_pmf, global_variance, fit_details, js_rows, patches_for_global
```

Both `reconstruct_chain` call sites in `main()` updated to unpack the 8th element.

#### 3. NEQ patch reuse logic

Inside `reconstruct_chain()`, when processing each neighbor segment:

```python
if segment.name in neq_patch_store:
    neq_patch = neq_patch_store[segment.name]   # reuse, no bootstrapping
else:
    neq_patch = build_neq_mts_patch(...)
    neq_patch_store[segment.name] = neq_patch
    neq_patch_status[segment.name] = { ... }
```

Existing valid patches survive EQ reclustering without reboostrapping.

#### 4. Patch selection by `pmf_method`

```python
neq_patches_from_store = [p for p in neq_patch_store.values() if np.count_nonzero(p.coverage_mask) > 0]
if pmf_method == "neq":
    patches_for_global = neq_patches_from_store
elif pmf_method == "hybrid":
    patches_for_global = eq_patches_current + neq_patches_from_store
elif pmf_method == "eq":
    patches_for_global = build_eq_pmf_with_neq_variance(eq_patches_current, neq_patches_from_store, grid)
```

No duplicate inclusion because store keys are unique.

#### 5. Per-segment classification against current cluster graph

After building `patches_for_global`, every stored segment is reclassified:

- `is_current_neighbor_edge`: segment connects the currently adjacent EQ cluster pair
- `is_internal_to_current_eq_cluster`: both boundary windows belong to the same current EQ cluster
- `is_long_range_or_obsolete_edge`: exists but not current neighbor and not internal
- `included_in_global_fit`: 1 if patch is in `patches_for_global`

#### 6. Helper functions added

```python
def build_window_to_cluster_map(clusters) -> dict[str, str]
def write_all_neq_patches_csv(out_root, neq_patch_store, neq_patch_status, out_root_base)
def write_patches_used_for_global_fit_csv(out_root, patches_for_global, fit_details, neq_patch_store, neq_patch_status, out_root_base)
```

#### 7. New output files

**`all_neq_patches.csv`** — written at run root and into each snapshot directory:

```
segment, left_boundary, right_boundary, patch_name, patch_kind, n_covered_bins,
is_current_neighbor_edge, is_internal_to_current_eq_cluster, is_long_range_or_obsolete_edge,
left_current_cluster, right_current_cluster, mts_patch_built, reused,
included_in_global_fit, reason, patch_root, pmf_file, variance_file, summary_file
```

**`patches_used_for_global_fit.csv`** — lists actual patches used in the global inverse-variance fusion:

```
name, kind, n_covered_bins, included_in_global_fit, inclusion_source,
patch_root, pmf_file, variance_file, aligned_pmf_file, summary_file
```

`inclusion_source` is one of: `current_eq_cluster_patch`, `current_neighbor_neq_patch`, `retained_old_neq_patch`, `other`.

**`mines_variance_fusion_summary.json`** — `summary_files` extended:

```json
{"all_neq_patches": "all_neq_patches.csv", "patches_used_for_global_fit": "patches_used_for_global_fit.csv"}
```

#### 8. `write_state_snapshot` updated

Passes `neq_patch_store`, `neq_patch_status`, `patches_for_global` to snapshot sub-call so snapshot directories contain `all_neq_patches.csv` and `patches_used_for_global_fit.csv` at each stage.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Cell `e7eff267` (data loading)

Added loading of `all_neq_patches_df` and `patches_used_df` from `all_neq_patches.csv` and `patches_used_for_global_fit.csv`. Status printed at end of cell.

#### New cell `66f5959f` (after `645c3d81`) — Retained NEQ patches diagnostic

- Warns (with table) if any `mts_patch_built=1` patch is `included_in_global_fit=0`.
- Classifies each row as `current_neighbor` / `retained_old` / `internal_to_eq_cluster`.
- Two-panel figure: scatter (segment vs n_covered_bins, colored/styled by category, faded if excluded) and horizontal bar chart.
- Full `all_neq_patches` DataFrame display.
- `patches_used_for_global_fit` summary table with `inclusion_source` column.

### Verification

```
python -m py_compile scripts/mines_variance_fusion.py  # AST OK
```

All new output files will appear in future runs. Notebook handles missing files gracefully.
