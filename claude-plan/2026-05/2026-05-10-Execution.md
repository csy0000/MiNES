# Execution Log — 2026-05-10

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| fix_gt_rescue_mean_bracketing | Replace cluster-based GT rescue pair selection with mean-bracketing | ~35 000 | ~8 000 | New helper; `design_rescue_window_mean_only_gt` updated; 6 new gt_ cols |
| mean_only_gt_rescue_re_verify | Re-verify mean-only GT rescue implementation (same content as 2026-05-09 instruction) | ~18 000 | ~1 000 | Implementation already complete; all acceptance criteria verified |
| notebook_networkx_and_mean_only_gt | Add NetworkX uncertainty network section; update GT markdown description | ~22 000 | ~4 000 | 2 new cells (neq-network-md, neq-network-code) at idx 15–16; d04aab74 updated |
| meanx_eq_clustering_bar_edges | Sort EQ clustering by mean_x; add BAR ddF for EQ edges; write eq_bar_edges.csv; update notebook | ~28 000 | ~6 000 | 7 changes in script + notebook neq-network-code updated |

---

## [fix_gt_rescue_mean_bracketing]

**Instruction file:** `claude-plan/2026-05/2026-05-10-fix_gt_rescue_mean_bracketing.md`

### Problem

GT rescue was producing repeated rescue windows with identical `x_m` and `k` because the pair selection used cluster-boundary logic (`gap_clusters_for_target` / `choose_connected_boundary_pair`) which does not update when rescue windows are added to the pool. The cluster boundaries are fixed; rescue windows are not added to clusters, so later rescue rounds kept using the same reference pair.

### Changes — `scripts/mines_variance_fusion.py`

#### New helper `choose_mean_bracketing_windows_for_gt_rescue`

Inserted before `design_rescue_window_mean_only_gt`. Takes `target_bin_x` and `windows: list[EnsembleWindow]`, returns the first adjacent pair `(left, right, meta)` satisfying `left.mean_x < target_bin_x < right.mean_x`.

- Filters to windows with finite, positive `std_x` and finite `mean_x`
- Sorts by `(mean_x, name)` to ensure stable ordering
- Iterates adjacent pairs and returns the first bracketing pair
- Returns `None` if no such pair exists

This helper is called with the full current `windows` list at each rescue round, so rescue windows added in previous rounds are included as potential GT reference windows.

#### Updated `design_rescue_window_mean_only_gt`

- Added `windows: list[EnsembleWindow]` as a required keyword argument
- Made `clusters` and `segment_store` optional (default `None`) — kept for backward compat but no longer used
- Replaced the cluster-based pair selection block with `choose_mean_bracketing_windows_for_gt_rescue(target_bin_x=target_bin_x, windows=windows)`
- Fallback case (no bracketing pair found) records `gt_fallback_reason = "no_adjacent_mean_bracketing_pair"` and `rescue_center_rule = "target_bin_fallback_no_mean_bracket"`
- Added new diagnostic keys to the return dict:
  - `gt_pair_rule`: set to `"adjacent_eq_windows_bracketing_target_by_mean_x"` when pair found
  - `gt_s_raw`: the raw computed `s = (target_bin_x - m_L) / (m_R - m_L)` (will be in (0,1) when pair was correctly bracketing)
  - `gt_s_used`: the actually used s (same as `gt_s_raw` when no fallback, else 0.5)
  - `gt_s_fallback_to_midpoint`: bool indicating whether s was overridden to 0.5
  - `gt_x_clipped`: bool — whether `x_res` was clipped from `x_res_raw`
  - `gt_k_clipped`: bool — whether `k_res` was clipped from `k_res_raw`
- `gt_left_cluster`, `gt_right_cluster` now set to `""` (no longer applicable)
- `gt_boundary_pair_reason` set to `""` (no longer applicable)

#### Updated call site (line ~5260)

Added `windows=windows`; removed `clusters=clusters` and `segment_store=segment_store`.

```python
rescue_design = design_rescue_window_mean_only_gt(
    target_info=target_info,
    windows=windows,          # full current window list, updated each round
    grid=grid,
    args=args,
    ...
)
```

#### Updated `_RESCUE_SUMMARY_COLS`

Added 6 new columns before the `mo_*` block:

```
gt_pair_rule
gt_s_raw
gt_s_used
gt_s_fallback_to_midpoint
gt_x_clipped
gt_k_clipped
```

(Existing `gt_s_eff`, `gt_used_midpoint_fallback`, `gt_boundary_pair_reason` retained for backward compat.)

### Expected behavior after fix

- For each rescue round, the full list of current EQ windows (including previously added rescue windows) is searched for the adjacent mean-bracketing pair
- If M4 is added with `mean_x` between M1 and M3, then the next rescue round looking in the same region will use M4 as a boundary instead of M1
- `gt_s_raw` should be in (0,1) whenever the pair was found, because the pair was selected by bracketing

---

## [notebook_networkx_and_mean_only_gt]

**Instruction file:** `claude-plan/2026-05/2026-05-02-claudecode_notebook_networkx_and_mean_only_gt.md`

### Task 1 — EQ/NEQ uncertainty network (new cells at idx 15–16)

Added two cells immediately before the rescue/frontier section (50f5f325):

#### Cell `neq-network-md` (markdown)

`## EQ/NEQ uncertainty network` — describes nodes, edge types, layout weighting.

#### Cell `neq-network-code` (code)

- Gracefully imports `networkx`; prints skip message if not installed.
- Loads `windows_df`, `clusters_df`, `segments_df` (reuses already-loaded variables; falls back to `read_csv_optional`).
- **Nodes**: all windows from `windows_df`, attributes `mean_x`, `center_x`, `generation`, `side`, `k`.
- **EQ/BAR edges**: from `clusters_df.window_names` (comma-separated), adjacent pairs sorted by `mean_x`; `ddF = NaN` / `uncertainty_source = "missing"` (not stored in clusters.csv).
- **NEQ/CFT edges**: from `segments_df` columns `boundary_left` / `boundary_right`; uncertainty from `cft_delta_f_unc` (first-existing fallback across `cft_delta_f_unc`, `cft_ddF`, `crooks_delta_f_unc`, `crooks_ddF`, `delta_f_unc`, `ddF`). Layout weight = `1 / max(ddF, 1e-6)` when finite, else `0.1`.
- Spring layout (`seed=7`, `weight="layout_weight"`).
- Node colors from `coolwarm` cmap (`analysis_xmin`→blue, `analysis_xmax`→red); colorbar labeled `"mean x"`.
- Solid black edges (EQ/BAR), dashed black (NEQ/CFT), edge labels `BAR ddF=?` / `CFT ddF=0.xxx`.
- Edge summary table displayed via `pd.DataFrame`.

### Task 2 — Simplify GT PMF visualization

The s-dependent harmonic curves in cell `f02a7b6b` were already removed in the previous session. Remaining action: updated the stale markdown in cell `d04aab74` from the old three-curve description (s=0 left, s=1 right, target s) to the correct single mean-only harmonic description.

New `d04aab74` markdown: explains that F₀(x) = ½k₀(x−x₀)² is estimated by force-balance from EQ means only, and that σ values are used only for target width and `kT_eff` (not for k₀ or x₀).

### Verification

```
total cells: 25 (was 23)
neq-network-md at idx 15: OK
neq-network-code at idx 16: OK
d04aab74 at idx 20: updated markdown OK
cell f02a7b6b: no s-dependent curves (unchanged from previous session)
```

---

## [meanx_eq_clustering_bar_edges]

**Instruction file:** `claude-plan/2026-05/2026-05-10-claudecode_meanx_eq_clustering_bar_edges.md`

### Changes — `scripts/mines_variance_fusion.py`

#### New helper `compute_eq_pair_bar_summary` (inserted after `eq_tail_samples`)

Computes BAR uncertainty between two adjacent EQ windows using their tail samples:
- Loads `x_L`, `x_R` from `eq_tail_samples`; filters non-finite.
- Computes reduced potential differences: `w_F[i] = beta*(U_R(x_L[i]) - U_L(x_L[i]))`, `w_R[i] = beta*(U_L(x_R[i]) - U_R(x_R[i]))`.
- Reshapes to `(-1, 1)` and calls `solve_segment_cft_delta_f_once(w_F, w_R, kT=kT)`.
- Returns `{"bar_solved", "bar_delta_f", "bar_delta_f_unc", "bar_method", "bar_reason"}`.
- Fallback reasons: `not_enough_samples`, `nonfinite_work_values`, `bar_solver_failed: <exc>`.

#### Updated `build_eq_clusters`

- Signature: added `ctx: dict[str, Any] | None = None`.
- Sort changed from `center_x` to `mean_x`: `sorted(windows, key=lambda row: (float(row.mean_x), str(row.name)))`.
- Each js_row now includes: `left_mean_x`, `right_mean_x`, `left_center_x`, `right_center_x`, `cluster_order_coordinate = "mean_x"`, plus `**bar_summary` from `compute_eq_pair_bar_summary`.
- Cluster windows explicitly sorted by `mean_x` before `EQCluster(...)` creation (both in the loop and for the final cluster).

#### Updated `build_cluster_rows`

Added columns: `order_coordinate = "mean_x"`, `left_x_coordinate = "mean_x"`, `right_x_coordinate = "mean_x"`, `window_mean_xs` (comma-separated floats), `window_center_xs`.

#### New helper `build_eq_bar_edge_rows`

Filters `js_rows` to `merged == True`, looks up cluster name via `w2c` map, returns rows for `eq_bar_edges.csv` with: `cluster`, `left_window`, `right_window`, `left_mean_x`, `right_mean_x`, `left_center_x`, `right_center_x`, `ddF = bar_delta_f_unc`, BAR fields, JSD fields, `kind = "EQ_BAR"`, `style = "solid"`, `color = "black"`.

#### Updated `write_state_tables`

- `neighbor_jsd.csv` extras expanded with: `left_mean_x`, `right_mean_x`, `left_center_x`, `right_center_x`, `cluster_order_coordinate`, `bar_solved`, `bar_delta_f`, `bar_delta_f_unc`, `bar_method`, `bar_reason`.
- Added write of `eq_bar_edges.csv` immediately after `neighbor_jsd.csv`.

#### Updated call site (`reconstruct_chain`)

`build_eq_clusters(windows, grid, float(args.js_threshold), ctx=ctx)`

#### Updated `find_first_rescue_pair`

- `left_boundary = rightmost_mean_window(left_cluster)` (was `cluster.windows[-1]`)
- `right_boundary = leftmost_mean_window(right_cluster)` (was `cluster.windows[0]`)
- Gap bounds: `lo/hi` now use `mean_x` (was `center_x`)
- Rescue check: `lo < float(w.mean_x) < hi` (was `w.center_x`)

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

#### Updated `neq-network-code` cell

EQ/BAR edges now read from `eq_bar_edges.csv` (new file). Reads `ddF` column (= `bar_delta_f_unc`) for edge label and layout weight. Fallback to `"[network] eq_bar_edges.csv not found"` if file missing. NEQ/CFT edges unchanged (still from `segments.csv / cft_delta_f_unc`).

### Verification

```
syntax: OK
sort by mean_x: OK
compute_eq_pair_bar_summary: OK
build_eq_clusters ctx param: OK
cluster_order_coordinate in js_rows: OK
bar_summary spread in js_rows: OK
build_eq_bar_edge_rows: OK
eq_bar_edges.csv write: OK
neighbor_jsd.csv extras: OK
build_cluster_rows metadata: OK
call site ctx: OK
find_first_rescue_pair mean_x: OK
notebook neq-network-code reads eq_bar_edges.csv: OK
```

---

### Regression check (manual, run after workflow)

```python
# For rows without gt_fallback_reason:
assert gt_left_mean_x < target_bin_x < gt_right_mean_x
assert 0.0 < gt_s_raw < 1.0
assert gt_s_used == gt_s_raw
```

### Verification

```
syntax: OK
helper defined: OK
windows param in signature: OK
clusters optional: OK
call site: OK (windows=windows, no clusters, no segment_store)
new gt_ cols in _RESCUE_SUMMARY_COLS: OK
helper logic: OK (sorts by mean_x, checks m_L < target < m_R)
new gt_ fields in return dicts: OK
```

---

## [mean_only_gt_rescue_re_verify]

**Instruction file:** `claude-plan/2026-05/2026-05-10-claudecode_mean_only_gt_rescue_instruction.md`

This instruction has identical content to `2026-05-09-claudecode_mean_only_gt_rescue_instruction.md`, which was fully implemented in the 2026-05-09 session. The implementation was verified to still be intact after the `fix_gt_rescue_mean_bracketing` changes made earlier today.

### Implementation status — all acceptance criteria confirmed

| Criterion | Status |
|---|---|
| k0 and x0 from force-balance only (`k_L*(m_L-x_L) - k_R*(m_R-x_R)`) | ✅ |
| sigmas used only for sigma_s and kT_eff | ✅ |
| s_raw fallback to 0.5 (not clipped) when `s_raw <= 0` or `s_raw >= 1` | ✅ |
| `k_res_raw = kT_eff / sigma_s**2 - k0` | ✅ |
| `x_res_raw = x_target + (k0/k_res)*(x_target - x0)` with signed k0 | ✅ |
| rescue summary records all 32 `mo_*` diagnostic columns | ✅ |
| notebook `mo-gt-rescue-code` cell plots F₀, U_res, F₀+U_res, vertical markers | ✅ |
| existing workflow outside rescue unchanged | ✅ |

### Column naming convention note

The instruction specifies unprefixed column names (e.g. `k0_mean_only`, `s_raw`). The implementation uses `mo_` prefixed names (e.g. `mo_k0_mean_only`, `mo_s_raw`) to avoid conflicts with existing `gt_` fields and to enable automatic spreading via `k.startswith("mo_")` in the rescue_row builder. The notebook reads `mo_*` columns consistently.

### Notebook state

```
23 total cells
0 plt.close() calls
mo-gt-rescue-md (cell 20): markdown description
mo-gt-rescue-code (cell 21): visualization code
mo_k0_mean_only referenced 3 times in notebook code
```
