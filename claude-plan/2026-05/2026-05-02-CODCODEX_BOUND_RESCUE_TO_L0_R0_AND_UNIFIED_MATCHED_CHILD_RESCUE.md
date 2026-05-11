# Codex Task: Restrict MiNES Analysis/Rescue to [x_L0, x_R0] and Apply Unified Matched-Child Rescue

You are working on the MiNES variance-fusion workflow, primarily:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

Preserve the existing output structure. Make targeted changes to analysis bounds, rescue target selection, and rescue-window design.

---

## Motivation

The current workflow can select rescue targets or compute analysis diagnostics using bins outside the original endpoint range:

```text
[x_L0, x_R0]
```

This is not desired. The method should focus only on the original space of interest bounded by the initial left and right endpoint ensembles.

Additionally, rescue should use the unified matched-child rule for both uncovered bins and high-variance bins:

```text
For target bin x*:
    find previous child m with closest target_x to x*
    if found:
        x_rescue = x_m
        k_rescue = clip(s_rescue^(retry + 1) * k_m, k_min, k_max)
    else:
        x_rescue = x*
        k_rescue = k_rescue_default
```

Default:

```text
s_rescue = 2.0
```

---

# 1. Define hard analysis bounds from L0 and R0

After creating `left0` and `right0`, define the hard endpoint bounds:

```python
endpoint_xmin = float(min(left0.center_x, right0.center_x))
endpoint_xmax = float(max(left0.center_x, right0.center_x))
```

These bounds represent the physical/algorithmic region of interest.

## 1.1 Default analysis bounds

Currently, if `--analysis-xmin` and `--analysis-xmax` are not given, the code defaults to the full grid. Change the default so that:

```python
analysis_xmin = endpoint_xmin if args.analysis_xmin is None else float(args.analysis_xmin)
analysis_xmax = endpoint_xmax if args.analysis_xmax is None else float(args.analysis_xmax)
```

Then clamp user-provided values to the endpoint bounds:

```python
analysis_xmin = max(float(analysis_xmin), endpoint_xmin)
analysis_xmax = min(float(analysis_xmax), endpoint_xmax)
```

If the user supplies invalid bounds after clamping:

```python
if analysis_xmin >= analysis_xmax:
    raise ValueError("Invalid analysis bounds after clamping to [x_L0, x_R0].")
```

## 1.2 Write bounds to run metadata

Add to `run_request.json` and `mines_variance_fusion_summary.json`:

```json
{
  "endpoint_xmin": ...,
  "endpoint_xmax": ...,
  "analysis_xmin": ...,
  "analysis_xmax": ...,
  "analysis_bounds_rule": "clamped_to_initial_endpoint_centers"
}
```

---

# 2. Restrict rescue target selection to analysis bounds

The current rescue functions already accept `analysis_xmin` and `analysis_xmax`. Ensure all rescue target selection obeys these bounds.

## 2.1 Uncovered rescue

`choose_uncovered_rescue_target(...)` should only consider bins satisfying:

```python
analysis_mask = (
    np.isfinite(grid)
    & (grid >= analysis_xmin)
    & (grid <= analysis_xmax)
)
```

This is already conceptually correct. Confirm that the function never returns a target outside `[analysis_xmin, analysis_xmax]`.

After computing:

```python
target_x = nearest_grid_value(best["midpoint"], grid)
```

add a final clamp/snap:

```python
target_x = min(max(float(target_x), float(analysis_xmin)), float(analysis_xmax))
target_x = float(nearest_grid_value(target_x, grid))
```

## 2.2 High-variance rescue

Modify `choose_rescue_target(...)` or add a bounded version:

```python
def choose_rescue_target(
    grid: np.ndarray,
    global_variance: np.ndarray,
    analysis_xmin: float | None = None,
    analysis_xmax: float | None = None,
) -> tuple[float, float]:
    ...
```

Only consider finite variance bins inside `[analysis_xmin, analysis_xmax]`.

Required logic:

```python
finite = np.isfinite(global_variance)
if analysis_xmin is not None:
    finite &= grid >= float(analysis_xmin)
if analysis_xmax is not None:
    finite &= grid <= float(analysis_xmax)

if not np.any(finite):
    return float("nan"), float("nan")
```

Then choose the max finite variance only from this bounded mask.

In `choose_rescue_target_priority(...)`, call:

```python
target_x, target_variance = choose_rescue_target(
    grid,
    global_variance,
    analysis_xmin=analysis_xmin,
    analysis_xmax=analysis_xmax,
)
```

Do not allow max-variance rescue outside `[x_L0, x_R0]`.

## 2.3 Failed/skipped gap rescue

In `choose_failed_or_skipped_gap_target(...)`, only create candidates whose midpoint lies inside the analysis bounds. This is already partly done. Also clamp/snap any returned target to `[analysis_xmin, analysis_xmax]`.

---

# 3. Restrict PMF quality metrics to endpoint-bounded analysis region

`compute_pmf_quality_metrics(...)` should continue to use:

```python
analysis_mask = (grid >= analysis_xmin) & (grid <= analysis_xmax)
```

Because `analysis_xmin/xmax` are now clamped to `[x_L0, x_R0]`, coverage and RMSE will no longer include bins outside the initial endpoint range.

---

# 4. Unified matched-child rescue design for both uncovered and high-variance bins

The current uploaded file still treats high-variance rescue as:

```python
x_rescue = x*
k_rescue = k_rescue
```

and only applies matched-child scaling for uncovered intervals. Change this.

## 4.1 Retry counting should apply to all target priorities

Update:

```python
def count_previous_rescue_retries(...):
```

Currently it returns `0` unless `target_priority == "uncovered_interval"`. Remove that restriction.

New behavior:

Count previous rescue rows if either:

```text
same matched_child_name
```

or:

```text
abs(previous.x_rescue_target - current.x_rescue_target) <= grid_dx
```

Do not require the same `target_priority`. If a region was uncovered in one round and becomes high variance in the next, this should count as a retry.

Pseudo-code:

```python
def count_previous_rescue_retries(...):
    count = 0
    for row in rescue_rows:
        prev_name = str(row.get("matched_child_name", ""))
        same_child = (
            matched_child_name is not None
            and matched_child_name != ""
            and prev_name == str(matched_child_name)
        )

        prev_x = finite_float_or_none(row.get("x_rescue_target"))
        same_target = (
            prev_x is not None
            and abs(prev_x - float(x_rescue_target)) <= float(grid_dx)
        )

        if same_child or same_target:
            count += 1
    return count
```

## 4.2 Unified `design_rescue_window(...)`

For any target priority:

```python
x_rescue_target = float(target_info["target_x"])
child_designs = load_child_design_records(generations_root)
matched = match_child_design_for_rescue_target(child_designs, x_rescue_target)
```

If a valid matched child exists:

```python
n_retry = count_previous_rescue_retries(...)
rescue_scale = float(args.s_rescue) ** float(n_retry + 1)

rescue_center_x = float(nearest_grid_value(float(matched["center_x"]), grid))
rescue_k = clip(
    rescue_scale * float(matched["k"]),
    float(args.k_min),
    float(args.k_max),
)
rescue_k_rule = "repeat_matched_target_child_with_retry_scaled_k"
```

If no matched child exists:

```python
n_retry = 0
rescue_scale = 1.0
rescue_center_x = float(nearest_grid_value(x_rescue_target, grid))
rescue_k = float(args.k_rescue)
rescue_k_rule = "fallback_fixed_k_rescue_at_target"
```

This applies to both:

```text
target_priority == "uncovered_interval"
target_priority == "finite_variance"
```

---

# 5. Keep rescue window centers inside [x_L0, x_R0]

Even when using matched child `center_x`, the final rescue center must be inside the bounded analysis region.

Add a helper:

```python
def clamp_to_bounds_and_grid(
    value: float,
    grid: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    clamped = min(max(float(value), float(lower)), float(upper))
    return float(nearest_grid_value(clamped, grid))
```

Modify `design_rescue_window(...)` to accept bounds:

```python
def design_rescue_window(
    *,
    target_info: dict[str, Any],
    generations_root: Path,
    grid: np.ndarray,
    args: argparse.Namespace,
    analysis_xmin: float,
    analysis_xmax: float,
    rescue_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
```

Then:

```python
rescue_center_x = clamp_to_bounds_and_grid(
    matched["center_x"] if matched else x_rescue_target,
    grid,
    analysis_xmin,
    analysis_xmax,
)
```

Record both unclamped and final centers:

```text
rescue_center_x_raw
rescue_center_x
rescue_center_clamped_to_bounds
```

## 5.1 Update rescue-loop call

Change:

```python
rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
    rescue_rows=rescue_rows,
)
```

to:

```python
rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
    analysis_xmin=float(analysis_xmin),
    analysis_xmax=float(analysis_xmax),
    rescue_rows=rescue_rows,
)
```

---

# 6. Optional: constrain child proposal centers to [x_L0, x_R0]

The main request is to not **consider** bins outside `[x_L0, x_R0]`. Rescue and diagnostics must obey this.

It is also safer to ensure new child centers never exceed the endpoint bounds. If straightforward, pass endpoint bounds to `finalize_child_proposal(...)` and clamp final `center_x` to:

```python
center_x = min(max(center_x, endpoint_xmin), endpoint_xmax)
center_x = nearest_grid_value(center_x, grid)
```

Do not force this if it risks breaking the current growth logic. Rescue/analysis bound enforcement is required; child-proposal clipping is optional.

---

# 7. Notebook update

Update the notebook so plots and tables emphasize the bounded analysis region.

## 7.1 EQ distribution plot

In the EQ distribution plot:

- Add vertical lines for `x_L0` and `x_R0`, or for `analysis_xmin` and `analysis_xmax`.
- Optionally shade the outside regions to indicate they are not considered.

Use `mines_variance_fusion_summary.json` or `run_request.json` to read:

```text
endpoint_xmin
endpoint_xmax
analysis_xmin
analysis_xmax
```

## 7.2 Rescue table

Display:

```text
round
target_priority
x_rescue_target
rescue_center_x_raw
rescue_center_x
rescue_center_clamped_to_bounds
rescue_k
rescue_k_rule
rescue_retry_count
rescue_scale
matched_child_name
matched_target_x
matched_child_center_x
matched_child_k
```

---

# 8. Acceptance criteria

The implementation is correct if:

1. Default `analysis_xmin/xmax` are `[x_L0, x_R0]`, not full grid bounds.
2. User-provided `analysis_xmin/xmax` are clamped to `[x_L0, x_R0]`.
3. Uncovered-bin rescue only considers uncovered bins inside `[x_L0, x_R0]`.
4. High-variance rescue only considers finite-variance bins inside `[x_L0, x_R0]`.
5. Rescue centers are clamped to `[x_L0, x_R0]` even if the matched child center lies outside.
6. Both uncovered and high-variance targets use the matched-child rescue rule when a valid child design exists.
7. Repeated rescue attempts for the same target or same matched child increase the retry count and therefore increase `k_rescue`.
8. `rescue_summary.csv` records whether the rescue center was clamped.
9. Coverage/RMSE diagnostics in `pmf_quality_vs_steps.csv` use only the bounded analysis region.
10. The notebook shows the endpoint/analysis bounds on the EQ distribution plot.

---

# Conceptual summary

The desired behavior is:

```text
The MiNES run only evaluates and rescues the region between the initial endpoints L0 and R0.

Outside [x_L0, x_R0]:
    do not count coverage
    do not compute RMSE
    do not select uncovered rescue targets
    do not select high-variance rescue targets

For any rescue target x* inside the bounds:
    choose child m with closest target_x to x*
    if m exists:
        x_rescue = clamp(x_m, x_L0, x_R0)
        k_rescue = clip(s_rescue^(retry+1) * k_m, k_min, k_max)
    else:
        x_rescue = x*
        k_rescue = k_rescue_default
```
