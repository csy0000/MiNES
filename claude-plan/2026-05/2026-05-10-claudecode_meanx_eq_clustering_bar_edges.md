# Claude Code Task: Order EQ clustering by mean_x and write BAR ddF for merged EQ edges

You are modifying the current MiNES implementation, especially:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

The goal is to make EQ clustering spatially consistent with the sampled means and to provide BAR uncertainty (`ddF`) for EQ-cluster edges so the notebook can plot the uncertainty network.

---

## Background

Currently, `build_eq_clusters(...)` sorts EQ windows by `center_x`:

```python
ordered = sorted(windows, key=lambda row: (float(row.center_x), str(row.name)))
```

Then it compares adjacent windows in this center-ordered list using normalized JSD between EQ tail distributions. If the normalized JSD is below `js_threshold`, the new window is appended to the current cluster. Cluster bounds are already reported using `mean_x`:

```python
left_x=float(min(w.mean_x for w in current))
right_x=float(max(w.mean_x for w in current))
```

This can create inconsistent behavior when rescue or GT windows have centers that differ strongly from sampled means.

Change the clustering order to use `mean_x`.

---

# 1. Change EQ clustering order from center_x to mean_x

In `build_eq_clusters(...)`, replace the ordering rule:

```python
ordered = sorted(windows, key=lambda row: (float(row.center_x), str(row.name)))
```

with:

```python
ordered = sorted(windows, key=lambda row: (float(row.mean_x), str(row.name)))
```

Also update comments / metadata so the output clearly states:

```text
cluster_order_coordinate = mean_x
```

The merge criterion itself should remain the same:

```python
merged = finite(pair_jsd_norm) and pair_jsd_norm <= js_threshold
```

That means the algorithm should now be:

```text
all EQ windows
→ sort by mean_x
→ compare neighboring tail distributions by normalized JSD
→ merge if JSD <= js_threshold
→ each contiguous block becomes one EQCluster
→ cluster left_x/right_x are min/max mean_x
```

Do not change the JSD threshold logic.

---

# 2. Make cluster internal window order mean_x ordered

When an `EQCluster` is created, ensure `cluster.windows` is ordered by `mean_x`.

This should already happen naturally if `current` is built from the mean-sorted list, but make it explicit for safety:

```python
cluster_windows = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
```

Then create:

```python
EQCluster(
    name=cluster_name_from_windows(cluster_windows),
    windows=cluster_windows,
    left_x=float(min(w.mean_x for w in cluster_windows)),
    right_x=float(max(w.mean_x for w in cluster_windows)),
)
```

This matters because downstream code often uses:

```python
cluster.windows[0]
cluster.windows[-1]
```

or iterates over adjacent windows inside the cluster.

---

# 3. Update helper functions that implicitly assume center_x order

Search for logic that uses cluster edge windows or gap logic and make sure it is compatible with mean_x ordering.

At minimum check these functions:

```python
find_first_rescue_pair(...)
gap_clusters_for_target(...)
cluster_covering_target(...)
choose_connected_boundary_pair(...)
build_cluster_rows(...)
build_window_to_cluster_map(...)
```

Expected behavior:

- Cluster bounds `left_x` and `right_x` are in `mean_x` coordinate.
- A target is considered between clusters using `cluster.left_x` and `cluster.right_x`, i.e. mean-coordinate bounds.
- Boundary windows for inter-cluster NEQ should be chosen by mean coordinate:
  - rightmost mean window of the left cluster
  - leftmost mean window of the right cluster
  - unless an existing connected segment can be reused.

The current functions `rightmost_mean_window(...)` and `leftmost_mean_window(...)` are appropriate. Prefer them over raw `cluster.windows[-1]` or `cluster.windows[0]` where the intent is spatial boundary in sampled mean coordinate.

---

# 4. Add BAR ddF calculation for merged EQ-cluster edges

Add a function that computes BAR uncertainty between two neighboring EQ windows using their EQ tail samples.

Suggested function name:

```python
compute_eq_pair_bar_summary(left_window, right_window, ctx) -> dict[str, Any]
```

The function should return at least:

```python
{
    "bar_solved": bool,
    "bar_delta_f": float or "",
    "bar_delta_f_unc": float or "",
    "bar_method": "BAR" or "",
    "bar_reason": str,
}
```

## Reduced potential difference

For a 1D harmonic umbrella window with center `x_m` and spring constant `k_m`, the bias is:

```python
U_bias_m(x) = 0.5 * k_m * (x - x_m)**2
```

For samples from left window L, define:

```python
w_F = beta * (U_bias_R(x_L_samples) - U_bias_L(x_L_samples))
```

For samples from right window R, define reverse work-like values:

```python
w_R = beta * (U_bias_L(x_R_samples) - U_bias_R(x_R_samples))
```

where:

```python
beta = 1.0 / ctx["thermal_kT"]
```

Then estimate the free-energy difference and uncertainty between the two biased EQ ensembles using BAR.

Prefer existing utilities if available. Search the repository for an existing BAR estimator first, especially in:

```text
src/analysis/
```

Likely useful existing import/utility pattern:

```python
solve_segment_cft_delta_f_once(...)
```

is already used for NEQ/CFT uncertainty and returns:

```python
delta_f
delta_f_unc
method
reason
```

If this function can safely be reused for two arrays of bidirectional work values, reuse it. Otherwise, use `pymbar` BAR if available.

## Robust fallback

If BAR cannot be solved, do not crash the MiNES run.

Return:

```python
{
    "bar_solved": False,
    "bar_delta_f": "",
    "bar_delta_f_unc": "",
    "bar_method": "",
    "bar_reason": "..."
}
```

Reasons may include:

```text
not_enough_samples
bar_import_failed
bar_solver_failed: <exception>
nonfinite_work_values
```

---

# 5. Store BAR ddF in `neighbor_jsd.csv`

When two adjacent EQ windows are compared in `build_eq_clusters(...)`, add BAR information to the corresponding `js_rows` record.

Each row in `neighbor_jsd.csv` should include:

```text
left_window
right_window
left_mean_x
right_mean_x
left_center_x
right_center_x
pair_jsd_raw
pair_jsd_norm
pair_jsd
js_threshold
merged
cluster_order_coordinate
bar_solved
bar_delta_f
bar_delta_f_unc
bar_method
bar_reason
```

Important:

- Calculate BAR for every adjacent pair that is compared, not only for merged pairs.
- For the NetworkX plot, the notebook should use BAR edges only for rows where:
  ```python
  merged == True
  ```
- Still writing BAR for non-merged adjacent pairs is useful for diagnostics.

---

# 6. Add a dedicated `eq_bar_edges.csv`

In addition to `neighbor_jsd.csv`, write a simpler edge file for the notebook:

```text
eq_bar_edges.csv
```

Each row should be one internal edge between adjacent windows inside an EQ cluster.

Columns:

```text
cluster
left_window
right_window
left_mean_x
right_mean_x
left_center_x
right_center_x
ddF
bar_delta_f
bar_delta_f_unc
bar_solved
bar_method
bar_reason
pair_jsd
pair_jsd_norm
merged
kind
style
color
```

Set:

```text
ddF = bar_delta_f_unc
kind = EQ_BAR
style = solid
color = black
```

This file should only include edges where the two windows are in the same final EQ cluster, i.e. `merged == True`.

Recommended implementation:

- Use the `js_rows` generated by `build_eq_clusters`.
- Filter `merged == True`.
- Add cluster membership by using the final clusters.
- Write the file in `write_state_tables(...)` and in every snapshot via `write_state_snapshot(...)`.

This allows the notebook to directly read `eq_bar_edges.csv` without recomputing BAR.

---

# 7. Update `build_cluster_rows(...)`

Add explicit metadata to `clusters.csv`:

```text
order_coordinate = mean_x
left_x_coordinate = mean_x
right_x_coordinate = mean_x
```

Keep existing columns:

```text
name
left_x
right_x
window_names
n_windows
```

Optionally add:

```text
window_mean_xs
window_center_xs
```

This makes it easy to check whether clustering is consistent with sampled means.

---

# 8. Update the NetworkX notebook cell

Modify the notebook NetworkX uncertainty graph so that EQ/BAR edges are read from:

```text
eq_bar_edges.csv
```

Use the following conventions:

- Solid black edges for `kind == "EQ_BAR"`.
- Edge uncertainty:
  ```python
  ddF = row["ddF"]
  ```
  or fallback to:
  ```python
  bar_delta_f_unc
  ```
- Edge label:
  ```text
  BAR ddF=<value>
  ```

For NEQ edges, continue to use:

```text
segments.csv
```

or equivalent, with:

```text
cft_delta_f_unc
```

and draw them as dashed black edges.

The final graph should therefore combine:

```text
EQ edges from eq_bar_edges.csv      → solid black
NEQ edges from segments.csv         → dashed black
```

Nodes should still be all EQ windows from `windows.csv`, colored by `mean_x` from xmin blue to xmax red.

---

# 9. Acceptance criteria

The change is correct if:

1. `build_eq_clusters(...)` sorts windows by `mean_x`, not `center_x`.
2. The merge criterion still uses normalized JSD between adjacent EQ tail distributions.
3. Cluster `left_x` and `right_x` are still min/max `mean_x`.
4. `neighbor_jsd.csv` includes BAR information for all adjacent tested EQ pairs.
5. A new `eq_bar_edges.csv` is written.
6. `eq_bar_edges.csv` contains only merged/internal EQ-cluster edges.
7. `eq_bar_edges.csv` has `ddF = bar_delta_f_unc`.
8. The notebook NetworkX cell uses `eq_bar_edges.csv` for solid black EQ/BAR edges.
9. The notebook still uses CFT uncertainty from NEQ segments for dashed black NEQ/CFT edges.
10. The run does not fail if BAR cannot be computed; missing BAR uncertainty should appear as empty/NaN with a reason.

---

# 10. Do not change

Do not change the final global PMF fusion logic.

Do not change the NEQ/CFT estimator.

Do not change the rescue target priority logic, except where mean-coordinate cluster bounds need to be respected.

Do not remove existing diagnostic CSV files.
