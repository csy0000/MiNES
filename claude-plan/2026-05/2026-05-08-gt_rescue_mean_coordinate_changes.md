# ClaudeCode Task: Use GT Rescue and Replace Core `x_most` Logic with `mean_x`

## Context

The current MiNES implementation has two related issues:

1. **Rescue windows are not placed with the Gaussian Transport (GT) rule.**
   The current rescue logic places the rescue umbrella center directly at the target bin and chooses stiffness from a target-bin sigma/retry-scaling rule. This is the old rescue strategy and should be replaced.

2. **Some core chain/cluster logic uses `x_most` even though GT assumes local Gaussian ensembles.**
   Under the GT assumption, the sampled ensemble should be represented by its tail mean and tail standard deviation. Therefore, for algorithmic decisions such as cluster spatial bounds, target bracketing, GT interpolation, and PMF variance anchoring, we should use `mean_x` instead of `x_most`.

The desired coordinate conventions are:

```text
center_x / x_m  = umbrella control parameter
mean_x          = Gaussian/GT sampled-state descriptor
std_x           = Gaussian/GT sampled-state width
x_most          = diagnostic / optional robust mode descriptor
```

Do **not** delete `x_most`; keep it for diagnostics and plotting. But core GT-related decisions should use `mean_x`.

---

## High-level required changes

Implement the following:

1. Add `mean_x` and `std_x` fields to `EnsembleWindow`.
2. Compute `mean_x`, `std_x`, and `x_most` for every EQ window.
3. Use `mean_x` for EQ cluster spatial bounds.
4. Use `mean_x` for rescue target bracketing and segment-covering tests.
5. Use `mean_x` as the default PMF bootstrap variance anchor for EQ and NEQ patches.
6. Use GT mode to design rescue windows.
7. Keep `x_most` in all output tables as a diagnostic.
8. Add enough metadata to `rescue_summary.csv` to confirm whether GT was used or fallback occurred.

---

## 1. Update `EnsembleWindow`

Current dataclass includes:

```python
@dataclass
class EnsembleWindow:
    name: str
    center_x: float
    k: float
    root: Path
    eq_file: Path
    tail_file: Path
    eq_rows: list[dict[str, str]]
    tail_rows: list[dict[str, str]]
    x_most: float
    generation: int
    side: str
```

Change it to include `mean_x` and `std_x`:

```python
@dataclass
class EnsembleWindow:
    name: str
    center_x: float
    k: float
    root: Path
    eq_file: Path
    tail_file: Path
    eq_rows: list[dict[str, str]]
    tail_rows: list[dict[str, str]]
    mean_x: float
    std_x: float
    x_most: float
    generation: int
    side: str
```

---

## 2. Compute `mean_x`, `std_x`, and `x_most` in `run_eq_window(...)`

Where the code currently computes only:

```python
tail_x = np.asarray([float(row["x"]) for row in tail_rows], dtype=float)
x_most = float(mode_x_from_samples(tail_x, grid))
```

change to:

```python
tail_x = np.asarray([float(row["x"]) for row in tail_rows], dtype=float)
tail_x = tail_x[np.isfinite(tail_x)]
if tail_x.size < 2:
    raise RuntimeError(f"Need at least two finite tail samples for {name}.")

mean_x = float(np.mean(tail_x))
std_x = float(np.std(tail_x, ddof=1))
x_most = float(mode_x_from_samples(tail_x, grid))
```

Then construct the window with:

```python
window = EnsembleWindow(
    name=name,
    center_x=float(center_x),
    k=float(k),
    root=root,
    eq_file=eq_file,
    tail_file=tail_file,
    eq_rows=eq_rows,
    tail_rows=tail_rows,
    mean_x=mean_x,
    std_x=std_x,
    x_most=x_most,
    generation=int(generation),
    side=side,
)
```

---

## 3. Update window summaries

In `build_window_summary(...)`, include:

```python
"mean_x": float(window.mean_x),
"std_x": float(window.std_x),
"x_most": float(window.x_most),
"mean_minus_x_most": float(window.mean_x - window.x_most),
"mean_minus_center_x": float(window.mean_x - window.center_x),
"x_most_minus_center_x": float(window.x_most - window.center_x),
```

This keeps `x_most` visible as a diagnostic while making the new coordinate usage auditable.

---

## 4. Make `eq_gt_tuple(...)` use stored `mean_x` and `std_x`

Currently `eq_gt_tuple(...)` recomputes tail mean and sigma by calling `window_tail_mean_sigma(window)`.

Change it to:

```python
def eq_gt_tuple(window: "EnsembleWindow") -> tuple[float, float, float, float]:
    return (
        float(window.center_x),
        float(window.k),
        float(window.mean_x),
        float(window.std_x),
    )
```

This avoids duplicate definitions of the Gaussian state descriptor.

---

## 5. Use `mean_x` for EQ cluster spatial bounds

Current behavior in `build_eq_clusters(...)` sets:

```python
left_x = float(current[0].x_most)
right_x = float(current[-1].x_most)
```

Replace this with mean-based bounds:

```python
left_x = float(min(w.mean_x for w in current))
right_x = float(max(w.mean_x for w in current))
```

Do this for every cluster construction site in `build_eq_clusters(...)`.

Important: the windows may still be sorted by `center_x` for chain ordering. However, the spatial extent of a cluster should be represented by the sampled mean coordinate, not by `x_most`.

Also update cluster output rows to explicitly say what coordinate is used:

```python
"spatial_bound_coordinate": "mean_x"
```

---

## 6. Choose cluster boundary windows using `mean_x`

Where the code uses:

```python
left_boundary = left_cluster.windows[-1]
right_boundary = right_cluster.windows[0]
```

replace with helper functions:

```python
def rightmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return max(cluster.windows, key=lambda w: float(w.mean_x))


def leftmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return min(cluster.windows, key=lambda w: float(w.mean_x))
```

Then use:

```python
left_boundary = rightmost_mean_window(left_cluster)
right_boundary = leftmost_mean_window(right_cluster)
```

This should be the default for GT rescue, gap tests, and non-overlapping neighboring-pair logic.

If the existing `choose_connected_boundary_pair(...)` is used to preserve a previously connected NEQ segment, update it so that:

- candidate selection is still based on existing segment connectivity,
- but the fallback boundary windows are chosen by `mean_x`, not list position,
- metadata records whether the chosen pair came from an existing segment or mean-coordinate fallback.

Suggested update:

```python
def choose_connected_boundary_pair(
    left_cluster: EQCluster,
    right_cluster: EQCluster,
    segment_store: dict[tuple[str, str], "NEQSegment"],
) -> tuple["EnsembleWindow", "EnsembleWindow", dict[str, Any]]:
    right_boundary_default = leftmost_mean_window(right_cluster)
    right_name = right_boundary_default.name

    connected_left_candidates = [
        w for w in left_cluster.windows
        if (w.name, right_name) in segment_store
    ]

    if connected_left_candidates:
        left_boundary = min(
            connected_left_candidates,
            key=lambda w: abs(float(w.mean_x) - float(right_boundary_default.mean_x)),
        )
        right_boundary = right_boundary_default
        boundary_pair_reason = "existing_connected_segment_to_right_mean_boundary"
    else:
        left_boundary = rightmost_mean_window(left_cluster)
        right_boundary = right_boundary_default
        boundary_pair_reason = "mean_coordinate_cluster_boundary_fallback"

    metadata = {
        "chosen_left_window": left_boundary.name,
        "chosen_right_window": right_boundary.name,
        "chosen_left_mean_x": float(left_boundary.mean_x),
        "chosen_right_mean_x": float(right_boundary.mean_x),
        "boundary_pair_reason": boundary_pair_reason,
        "connected_left_candidate_names": [w.name for w in connected_left_candidates],
    }
    return left_boundary, right_boundary, metadata
```

If there are existing segments from a different right boundary, also consider searching all pairs `(left_window, right_window)` across the two clusters and selecting the connected pair with the smallest mean-coordinate gap.

---

## 7. Update source anchor helpers to use `mean_x`

Current single-window anchors return `x_most`.

Change:

```python
def source_left_anchor(source: EQCluster | EnsembleWindow) -> float:
    if isinstance(source, EQCluster):
        return float(source.left_x)
    return float(source.mean_x)


def source_right_anchor(source: EQCluster | EnsembleWindow) -> float:
    if isinstance(source, EQCluster):
        return float(source.right_x)
    return float(source.mean_x)
```

`EQCluster.left_x` and `EQCluster.right_x` should already be mean-based after the previous change.

---

## 8. Use `mean_x` as PMF variance anchor

### EQ MBAR patch

In `build_eq_cluster_patch(...)`, replace the anchor reference:

```python
reference_x=float(window.x_most)
```

with:

```python
reference_x=float(window.mean_x)
```

Also update metadata:

```python
"anchor_coordinate": "mean_x"
```

and optionally write per-anchor metadata:

```python
"anchor_mean_x_by_window": {w.name: float(w.mean_x) for w in cluster.windows},
"anchor_x_most_by_window": {w.name: float(w.x_most) for w in cluster.windows},
```

### NEQ MTS patch

In `build_neq_mts_patch(...)`, replace:

```python
left_reference_x = float(segment.left_boundary.x_most)
right_reference_x = float(segment.right_boundary.x_most)
```

with:

```python
left_reference_x = float(segment.left_boundary.mean_x)
right_reference_x = float(segment.right_boundary.mean_x)
```

Also add metadata:

```python
"anchor_coordinate": "mean_x",
"left_reference_mean_x": float(segment.left_boundary.mean_x),
"right_reference_mean_x": float(segment.right_boundary.mean_x),
"left_reference_x_most": float(segment.left_boundary.x_most),
"right_reference_x_most": float(segment.right_boundary.x_most),
```

### HS fallback patch

If HS fallback patches are still used, update their reference anchors to `mean_x` as well:

```python
reference_x = float(segment.left_boundary.mean_x)   # forward
reference_x = float(segment.right_boundary.mean_x)  # reverse
```

---

## 9. Update gap/coverage tests to use mean-based cluster bounds

The function `gap_clusters_for_target(...)` can keep its current logic if `cluster.left_x` and `cluster.right_x` are changed to mean-based bounds.

However, update the docstring and metadata to make this explicit:

```python
def gap_clusters_for_target(target_x: float, clusters: list[EQCluster]) -> tuple[EQCluster, EQCluster] | None:
    """Return adjacent clusters whose mean-coordinate spatial gap brackets target_x."""
```

Similarly, `cluster_covering_target(...)` can keep using `cluster.left_x/right_x`, but these are now mean-based.

Update `segment_covering_target(...)` from:

```python
left_x = float(segment.left_boundary.x_most)
right_x = float(segment.right_boundary.x_most)
```

to:

```python
left_x = float(segment.left_boundary.mean_x)
right_x = float(segment.right_boundary.mean_x)
```

---

## 10. Update failed/skipped segment rescue gap target

In `choose_failed_or_skipped_gap_target(...)`, replace:

```python
left_x = float(segment.left_boundary.x_most)
right_x = float(segment.right_boundary.x_most)
```

with:

```python
left_x = float(segment.left_boundary.mean_x)
right_x = float(segment.right_boundary.mean_x)
```

Add metadata:

```python
"gap_coordinate": "mean_x"
```

---

## 11. Update growth stopping condition

Where the growth stopping or frontier crossing condition uses:

```python
left_frontier.x_most >= right_frontier.x_most
```

replace with:

```python
left_frontier.mean_x >= right_frontier.mean_x
```

Keep the center-based safeguard:

```python
left_frontier.center_x >= right_frontier.center_x
or left_frontier.mean_x >= right_frontier.mean_x
```

Add both coordinates to the growth summary:

```python
"left_frontier_mean_x": float(left_frontier.mean_x),
"right_frontier_mean_x": float(right_frontier.mean_x),
"left_frontier_x_most": float(left_frontier.x_most),
"right_frontier_x_most": float(right_frontier.x_most),
"crossing_coordinate_rule": "center_x_or_mean_x",
```

---

## 12. Implement GT rescue window design

The current `design_rescue_window(...)` should no longer place the rescue center directly at `target_bin_x`.

Add a new function, or rewrite `design_rescue_window(...)`, to use GT:

```python
def design_gt_rescue_window(
    *,
    target_info: dict[str, Any],
    clusters: list[EQCluster],
    segment_store: dict[tuple[str, str], NEQSegment],
    grid: np.ndarray,
    args: argparse.Namespace,
    analysis_xmin: float,
    analysis_xmax: float,
    rescue_rows: list[dict[str, Any]] | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

### Algorithm

1. Read the target:

```python
x_target = float(target_info["target_x"])
```

2. Find the adjacent cluster gap whose mean-coordinate bounds bracket the target:

```python
gap = gap_clusters_for_target(x_target, clusters)
```

3. If the target is inside a cluster, choose the nearest cluster-neighbor pair around the target. For example:

```python
covering = cluster_covering_target(x_target, clusters)
```

If `covering` is not `None`, choose the adjacent pair with the smaller mean-distance to `x_target`:

- previous cluster + covering cluster, or
- covering cluster + next cluster.

4. If no valid pair exists, fallback to target-bin placement, but mark it clearly:

```python
rescue_center_rule = "target_bin_fallback_no_bracketing_clusters"
rescue_k_rule = "target_bin_sigma_fallback_no_bracketing_clusters"
```

5. Choose the boundary windows by mean-coordinate logic:

```python
left_boundary, right_boundary, boundary_meta = choose_connected_boundary_pair(
    left_cluster,
    right_cluster,
    segment_store,
)
```

or:

```python
left_boundary = rightmost_mean_window(left_cluster)
right_boundary = leftmost_mean_window(right_cluster)
```

6. Use the existing GT functions:

```python
EQ_L = eq_gt_tuple(left_boundary)
EQ_R = eq_gt_tuple(right_boundary)

x0_L, k0_L, x0_R, k0_R, harmonic_meta = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)

rescue_center_x, rescue_k, gt_meta = get_xs_ks_from_ms(
    x0_L,
    k0_L,
    x0_R,
    k0_R,
    ms=x_target,
    EQ_L=EQ_L,
    EQ_R=EQ_R,
    k_bound=(float(args.k_min), float(args.k_max)),
)
```

7. Clamp the resulting center to the analysis bounds only as a final safety check:

```python
rescue_center_x_raw = float(rescue_center_x)
rescue_center_x = clamp_to_bounds(rescue_center_x_raw, analysis_xmin, analysis_xmax)
```

8. If GT returns midpoint fallback, record it, but do not silently revert to old target-bin placement.

9. Preserve retry metadata for diagnostics, but do not multiply `k` by `s_rescue**n_retry` by default. That retry scaling was part of the old target-bin strategy. For GT rescue, the primary `k` should come from local harmonic/GT matching.

Optional: if repeated rescue fails to sample the target bin, allow a small retry adjustment, but it must be explicitly named:

```python
rescue_k_retry_rule = "disabled_for_GT"
```

or, if implemented:

```python
rescue_k_retry_rule = "GT_k_retry_clipped"
```

Default should be disabled.

---

## 13. Required `rescue_summary.csv` columns

Extend `_RESCUE_SUMMARY_COLS` with:

```text
gt_left_cluster
gt_right_cluster
gt_left_boundary
gt_right_boundary
gt_left_center_x
gt_right_center_x
gt_left_mean_x
gt_right_mean_x
gt_left_std_x
gt_right_std_x
gt_left_x_most
gt_right_x_most
gt_m_L
gt_m_R
gt_sigma_L
gt_sigma_R
gt_s_eff
gt_x_raw
gt_k_raw
gt_x0_L
gt_k0_L
gt_x0_R
gt_k0_R
gt_used_midpoint_fallback
gt_fallback_reason
gt_boundary_pair_reason
gt_anchor_coordinate
rescue_center_rule
rescue_k_rule
rescue_k_retry_rule
```

For successful GT rescue, set:

```text
rescue_center_rule = "GT_ms_target_mean_coordinate"
rescue_k_rule = "GT_ms_local_harmonic"
gt_anchor_coordinate = "mean_x"
rescue_k_retry_rule = "disabled_for_GT"
```

If `get_xs_ks_from_ms(...)` uses midpoint fallback, set:

```text
rescue_center_rule = "GT_ms_midpoint_fallback"
rescue_k_rule = "GT_ms_midpoint_fallback"
```

If no bracketing clusters exist and old placement is used as a last resort, set:

```text
rescue_center_rule = "target_bin_fallback_no_bracketing_clusters"
rescue_k_rule = "target_bin_sigma_fallback_no_bracketing_clusters"
```

---

## 14. Update rescue loop call site

Where the rescue loop currently calls:

```python
rescue_plan = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
    analysis_xmin=analysis_xmin,
    analysis_xmax=analysis_xmax,
    rescue_rows=rescue_rows,
    ctx=ctx,
)
```

change to:

```python
rescue_plan = design_gt_rescue_window(
    target_info=target_info,
    clusters=clusters,
    segment_store=segment_store,
    grid=grid,
    args=args,
    analysis_xmin=analysis_xmin,
    analysis_xmax=analysis_xmax,
    rescue_rows=rescue_rows,
    ctx=ctx,
)
```

The resulting EQ rescue window should still be run with:

```python
center_x = rescue_plan["rescue_center_x"]
k = rescue_plan["rescue_k"]
```

---

## 15. Update notebook/visualization outputs

Add the following to the EQ window information table:

```text
center_x
mean_x
std_x
x_most
mean_minus_x_most
mean_minus_center_x
x_most_minus_center_x
```

In plots, show both:

- umbrella center `center_x`, and
- sampled mean `mean_x`.

For rescue windows, mark:

- `x_rescue_target`,
- `target_bin_x`,
- `rescue_center_x`.

This is important because under GT rescue, these should generally be different:

```text
x_rescue_target is the desired sampled mean coordinate.
rescue_center_x is the umbrella center required to produce that sampled mean.
```

---

## 16. Acceptance criteria

The change is accepted if all of the following are true:

1. `EnsembleWindow` has `mean_x`, `std_x`, and `x_most`.
2. `x_most` remains in summaries but is no longer the core coordinate for GT logic.
3. `EQCluster.left_x/right_x` are based on `mean_x`.
4. `gap_clusters_for_target(...)` effectively brackets targets using mean-coordinate bounds.
5. `segment_covering_target(...)` uses boundary `mean_x` values.
6. `choose_failed_or_skipped_gap_target(...)` uses boundary `mean_x` values.
7. EQ bootstrap variance anchors use `mean_x` by default.
8. NEQ bootstrap variance anchors use boundary `mean_x` by default.
9. GT rescue uses `get_k0_x0_harmonic_fromEQ(...)` and `get_xs_ks_from_ms(...)`.
10. GT rescue no longer simply sets `rescue_center_x = target_bin_x`, except in an explicitly recorded fallback.
11. Repeated rescue attempts do not simply duplicate the same center with doubled `k` unless the GT computation itself gives that result or a clearly documented fallback occurs.
12. `rescue_summary.csv` contains enough GT metadata to audit the boundary pair, target mean, GT-derived center, raw/clipped stiffness, and fallback status.
13. Growth stopping uses `center_x` or `mean_x`, not `x_most`.
14. The notebook clearly shows `center_x`, `mean_x`, and `x_most` separately.

---

## Important conceptual rule

Use this convention everywhere:

```text
center_x = what we control in the simulation
mean_x   = where the biased ensemble actually samples on average
std_x    = Gaussian width of that sampled ensemble
x_most   = diagnostic mode of the sampled distribution
```

Therefore, when the algorithm asks:

> Which sampled region is covered?
> Which two ensembles bracket this target?
> What umbrella should produce a sampled mean at x_target?
> Where should bootstrap variance be anchored under the GT assumption?

use `mean_x`.

When the algorithm asks:

> Is the sampled distribution skewed or non-Gaussian?
> Is the histogram mode displaced from the mean?
> Is this EQ window suspicious near a barrier?

use `x_most` as a diagnostic.
