# Codex Task: Prioritize Uncovered Bins and Use Matched-Child Rescue Design in MiNES

You are working on the MiNES variance-fusion workflow, primarily:

```text
scripts/mines_variance_fusion.py
scripts/run_mines_variance_fusion.sh
notebooks/mines_variance_fusion_visualization.ipynb
```

Preserve the existing repository/output structure. Do not rewrite the workflow. Make targeted changes to the rescue logic only.

---

## Goals

Implement two rescue updates:

1. **Prioritize uncovered bins before high-variance bins.**
   - If the reconstructed PMF has continuously uncovered bins inside the analysis region, rescue those first.
   - Choose the midpoint of the widest uncovered interval as the rescue target.

2. **For uncovered-bin rescue, use the new MiNES rescue design rule.**
   - Find the previous child ensemble whose `target_x` is closest to the uncovered rescue target.
   - Repeat that child ensemble's center:
     ```text
     x_rescue_window = x_m
     ```
     where `x_m` is the matched child ensemble center.
   - Use a stronger force constant:
     ```text
     k_rescue_window = clip(s_rescue * k_m, k_min, k_max)
     ```
   - Default:
     ```text
     s_rescue = 2.0
     ```

---

# 1. Add rescue stiffness scaling argument

In `scripts/mines_variance_fusion.py`, add:

```python
parser.add_argument("--s-rescue", default=2.0, type=float)
```

In `scripts/run_mines_variance_fusion.sh`, add support for:

```bash
--s-rescue
```

Recommended bash default:

```bash
S_RESCUE=2.0
```

Pass it to Python:

```bash
--s-rescue "${S_RESCUE}"
```

---

# 2. Prioritize uncovered bins in rescue

The current rescue logic chooses the maximum finite global variance. This ignores uncovered bins, because uncovered bins usually have:

```text
global_pmf = NaN
global_variance = NaN
```

Change the rescue priority to:

```text
Priority 1: Widest continuously uncovered PMF interval inside the analysis region.
Priority 2: Skipped or MTS-failed NEQ gaps, if implemented/available.
Priority 3: Highest finite global variance among covered bins.
```

The required implementation is Priority 1 and Priority 3. Priority 2 can be implemented if straightforward, but must not break the workflow.

---

## 2.1 Define uncovered bins

Use the existing `analysis_xmin` and `analysis_xmax` values.

```python
analysis_mask = (
    np.isfinite(grid)
    & (grid >= float(analysis_xmin))
    & (grid <= float(analysis_xmax))
)
```

Default coverage rule:

```python
covered = analysis_mask & np.isfinite(global_pmf)
uncovered = analysis_mask & ~np.isfinite(global_pmf)
```

A bin is considered uncovered if `global_pmf` is not finite inside the analysis region.

Do not use `global_variance` for the default uncovered criterion, because some bins can have undefined variance even when the PMF is finite.

---

## 2.2 Add helper: find continuous uncovered intervals

Add:

```python
def find_uncovered_intervals(
    grid: np.ndarray,
    uncovered_mask: np.ndarray,
) -> list[dict[str, Any]]:
    ...
```

Return one dictionary per continuous uncovered interval:

```python
{
    "start_idx": int,
    "end_idx": int,
    "x_start": float,
    "x_end": float,
    "width": float,
    "n_bins": int,
    "midpoint": float,
}
```

Implementation details:

```python
indices = np.where(uncovered_mask)[0]
if len(indices) == 0:
    return []

intervals = []
start = indices[0]
prev = indices[0]

for idx in indices[1:]:
    if idx == prev + 1:
        prev = idx
    else:
        intervals.append(make_interval(start, prev))
        start = idx
        prev = idx

intervals.append(make_interval(start, prev))
```

Where:

```python
x_start = float(grid[start_idx])
x_end = float(grid[end_idx])
width = float(x_end - x_start)
midpoint = 0.5 * (x_start + x_end)
n_bins = end_idx - start_idx + 1
```

---

## 2.3 Add helper: choose uncovered rescue target

Add:

```python
def choose_uncovered_rescue_target(
    *,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    analysis_xmin: float,
    analysis_xmax: float,
) -> dict[str, Any] | None:
    ...
```

If no uncovered bins exist, return `None`.

If uncovered intervals exist:

1. Choose the widest interval.
2. If multiple intervals have the same width, choose the one with more bins.
3. If still tied, choose the one closest to the analysis-region center.

Return:

```python
{
    "target_x": float(nearest_grid_value(best["midpoint"], grid)),
    "target_variance": float("nan"),
    "target_priority": "uncovered_interval",
    "target_reason": "widest_uncovered_interval",
    "uncovered_start_x": best["x_start"],
    "uncovered_end_x": best["x_end"],
    "uncovered_width": best["width"],
    "uncovered_n_bins": best["n_bins"],
}
```

---

## 2.4 Add helper: priority rescue target

Add:

```python
def choose_rescue_target_priority(
    *,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    analysis_xmin: float,
    analysis_xmax: float,
    skipped_segment_rows: list[dict[str, Any]] | None = None,
    mts_failed_segments: list[NEQSegment] | None = None,
) -> dict[str, Any] | None:
    ...
```

Behavior:

```python
uncovered_target = choose_uncovered_rescue_target(
    grid=grid,
    global_pmf=global_pmf,
    analysis_xmin=analysis_xmin,
    analysis_xmax=analysis_xmax,
)
if uncovered_target is not None:
    return uncovered_target

# Optional Priority 2:
# choose skipped/MTS-failed gap midpoint if useful and safe.

target_x, target_variance = choose_rescue_target(grid, global_variance)
if math.isfinite(target_x) and math.isfinite(target_variance):
    return {
        "target_x": float(target_x),
        "target_variance": float(target_variance),
        "target_priority": "finite_variance",
        "target_reason": "max_finite_global_variance",
        "uncovered_start_x": "",
        "uncovered_end_x": "",
        "uncovered_width": "",
        "uncovered_n_bins": "",
    }

return None
```

Do not require finite `target_variance` for uncovered targets.

---

# 3. Matched-child rescue design for uncovered bins

When the target priority is:

```text
target_priority == "uncovered_interval"
```

do **not** simply place the rescue EQ window at the uncovered midpoint with fixed `k_rescue`.

Instead:

1. Use the uncovered midpoint only as:
   ```text
   x_rescue_target
   ```
   the target bin that was missed.

2. Find the previous child design whose `target_x` is closest to `x_rescue_target`.

3. Repeat that child's ensemble center and increase its stiffness:
   ```text
   rescue_center_x = matched_child.center_x
   rescue_k = clip(s_rescue * matched_child.k, k_min, k_max)
   ```

---

## 3.1 Load previous child designs

Add helper:

```python
def load_child_design_records(generations_root: Path) -> list[dict[str, Any]]:
    ...
```

Read files:

```text
generations/g*/left/child_design.json
generations/g*/right/child_design.json
```

Each record should include:

```text
design_file
name
side
target_x
center_x
k
raw_k
k_rule
target_source
barrier_crossing
```

Skip records where `target_x`, `center_x`, or `k` is missing or non-finite.

Recommended helper for finite conversion:

```python
def finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None
```

---

## 3.2 Match closest child target

Add helper:

```python
def match_child_design_for_rescue_target(
    child_designs: list[dict[str, Any]],
    x_rescue_target: float,
) -> dict[str, Any] | None:
    ...
```

Return the child design with smallest:

```python
abs(float(record["target_x"]) - float(x_rescue_target))
```

This corresponds to:

```text
argmin_m |x_m_target - x_rescue_target|
```

If no valid record exists, return `None`.

---

## 3.3 Add helper: design rescue window

Add:

```python
def design_rescue_window(
    *,
    target_info: dict[str, Any],
    generations_root: Path,
    grid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ...
```

For uncovered-interval targets:

```python
x_rescue_target = float(target_info["target_x"])
child_designs = load_child_design_records(generations_root)
matched = match_child_design_for_rescue_target(child_designs, x_rescue_target)
```

If a valid match exists:

```python
rescue_center_x = float(nearest_grid_value(float(matched["center_x"]), grid))
rescue_k = min(
    max(float(args.s_rescue) * float(matched["k"]), float(args.k_min)),
    float(args.k_max),
)
rescue_k_rule = "repeat_matched_target_child_with_scaled_k"
```

If no valid match exists:

```python
rescue_center_x = float(nearest_grid_value(x_rescue_target, grid))
rescue_k = float(args.k_rescue)
rescue_k_rule = "fallback_fixed_k_rescue_at_uncovered_target"
```

For non-uncovered targets:

```python
rescue_center_x = float(nearest_grid_value(float(target_info["target_x"]), grid))
rescue_k = float(args.k_rescue)
rescue_k_rule = "fixed_k_rescue"
matched = None
```

Return:

```python
{
    "x_rescue_target": float(x_rescue_target),
    "rescue_center_x": float(rescue_center_x),
    "rescue_k": float(rescue_k),
    "rescue_k_rule": str(rescue_k_rule),
    "s_rescue": float(args.s_rescue),
    "matched_child_design": str(matched["design_file"]) if matched else "",
    "matched_child_name": str(matched["name"]) if matched else "",
    "matched_child_side": str(matched["side"]) if matched else "",
    "matched_target_x": float(matched["target_x"]) if matched else "",
    "matched_target_distance": abs(float(matched["target_x"]) - x_rescue_target) if matched else "",
    "matched_child_center_x": float(matched["center_x"]) if matched else "",
    "matched_child_k": float(matched["k"]) if matched else "",
    "matched_child_raw_k": matched.get("raw_k", "") if matched else "",
    "matched_child_k_rule": matched.get("k_rule", "") if matched else "",
    "matched_child_target_source": matched.get("target_source", "") if matched else "",
}
```

---

# 4. Update rescue loop

In the rescue loop, replace the old target selection and rescue center/k calculation.

Currently it likely does:

```python
target_x, target_variance = choose_rescue_target(grid, global_variance)
rescue_center_x = nearest_grid_value(target_x, grid)
...
rescue_window = run_eq_window(..., center_x=rescue_center_x, k=float(args.k_rescue), ...)
```

Replace with:

```python
target_info = choose_rescue_target_priority(
    grid=grid,
    global_pmf=global_pmf,
    global_variance=global_variance,
    analysis_xmin=float(analysis_xmin),
    analysis_xmax=float(analysis_xmax),
    skipped_segment_rows=skipped_segment_rows,
    mts_failed_segments=mts_failed_segments(sorted(segment_store.values(), key=lambda item: item.name)),
)

if target_info is None:
    stop_reason = "no_rescue_target_available"
    break

rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
)

target_x = float(target_info["target_x"])
target_variance = target_info.get("target_variance", float("nan"))
rescue_center_x = float(rescue_design["rescue_center_x"])
rescue_k = float(rescue_design["rescue_k"])
```

Then create the rescue EQ window with:

```python
center_x=rescue_center_x
k=rescue_k
```

not with `args.k_rescue`.

---

# 5. Rescue action classification

If:

```python
target_info["target_priority"] == "uncovered_interval"
```

force:

```python
action = "add_bridge_eq_window"
reason = "widest_uncovered_interval"
```

For non-uncovered targets, keep the existing classification:

```python
gap_pair = gap_clusters_for_target(target_x, clusters)
containing_cluster = cluster_covering_target(target_x, clusters)
containing_segment = segment_covering_target(target_x, segments)
...
```

---

# 6. Output diagnostics

Add these fields to `rescue_summary.csv` and each `rescue_decision.json`:

```text
target_priority
target_reason
uncovered_start_x
uncovered_end_x
uncovered_width
uncovered_n_bins
x_rescue_target
rescue_center_x
rescue_k
rescue_k_rule
s_rescue
matched_child_design
matched_child_name
matched_child_side
matched_target_x
matched_target_distance
matched_child_center_x
matched_child_k
matched_child_raw_k
matched_child_k_rule
matched_child_target_source
```

For non-uncovered targets, leave matched-child fields blank.

Also add to `mines_variance_fusion_summary.json`:

```json
{
  "s_rescue": 2.0,
  "rescue_priority": "uncovered_interval_before_finite_variance",
  "rescue_k_design": "repeat_matched_target_child_with_scaled_k_for_uncovered_intervals"
}
```

---

# 7. Notebook update

Update the rescue section of:

```text
notebooks/mines_variance_fusion_visualization.ipynb
```

to display the new rescue fields if present:

```text
round
target_priority
target_reason
x_rescue_target
rescue_center_x
rescue_k
rescue_k_rule
matched_child_name
matched_target_x
matched_target_distance
matched_child_center_x
matched_child_k
s_rescue
action
added_window
```

Optional plot:
- Shade uncovered intervals.
- Mark `x_rescue_target`.
- Mark actual `rescue_center_x`.

This is optional; the table is required.

---

# 8. Acceptance criteria

The task is complete if:

1. Rescue checks uncovered PMF bins before finite-variance bins.
2. If a continuous uncovered interval exists inside the analysis region, the chosen `target_priority` is `uncovered_interval`.
3. The uncovered rescue target is the midpoint of the widest uncovered interval.
4. For uncovered rescue, the actual rescue EQ window center is the matched child `center_x`, not necessarily the uncovered midpoint.
5. The matched child is selected by minimum distance between `child_design["target_x"]` and `x_rescue_target`.
6. The rescue force constant is:
   ```text
   clip(s_rescue * matched_child_k, k_min, k_max)
   ```
   with default `s_rescue = 2.0`.
7. If no valid matched child exists, the code falls back to:
   ```text
   center_x = x_rescue_target
   k = k_rescue
   ```
8. High-variance covered rescue keeps the previous fixed `k_rescue` behavior.
9. `rescue_summary.csv` records both the target information and the actual rescue-window design.
10. The notebook displays the new rescue diagnostics.

---

# Conceptual summary

The final rescue behavior should be:

```text
If PMF has uncovered interval:
    x_rescue_target = midpoint of widest uncovered interval
    matched_child = child design with closest target_x to x_rescue_target
    if matched_child exists:
        rescue_center_x = matched_child.center_x
        rescue_k = clip(s_rescue * matched_child.k, k_min, k_max)
    else:
        rescue_center_x = x_rescue_target
        rescue_k = k_rescue
else:
    target highest finite variance
    rescue_center_x = target_x
    rescue_k = k_rescue
```

This makes rescue first repair missing PMF support and, for uncovered bins, repeat the child ensemble that was originally trying to target that region with stronger control.
