# Codex Task: Fix MiNES Rescue Placement and Escalating Rescue Stiffness

You are working on the MiNES variance-fusion workflow, primarily:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

Preserve the existing output structure. Make targeted changes to the rescue design logic only.

---

## Problem

The current rescue behavior can repeatedly create rescue windows with the same center and stiffness, for example:

```text
x_rescue = matched_child.center_x
k_rescue = s_rescue * matched_child.k
```

This can fail repeatedly if the matched child ensemble already missed the target region. The rescue should instead place the new window at the actual problematic bin while using the matched child only to infer a stronger force constant.

There are two required fixes:

1. **High-variance rescue should use the former/default rule.**
   - Do not use matched-child `center_x` for high-variance covered bins.
   - Use:
     ```text
     x_rescue = x*
     k_rescue = k_rescue_default
     ```
     where `x*` is the high-variance target bin.

2. **Uncovered-bin rescue should use the uncovered target location, not the matched child center.**
   - Use the previous child design only to infer `k`.
   - Use:
     ```text
     x_rescue = x*
     k_rescue = clip(s_rescue^(n_retry + 1) * k_m, k_min, k_max)
     ```
   - Here `x*` is the uncovered target bin, `k_m` is the force constant of the matched child ensemble, and `n_retry` is the number of previous rescue attempts for the same missed target or matched child.

---

# 1. Keep target priority unchanged

Keep the current rescue target priority:

```text
Priority 1: widest uncovered interval inside analysis region
Priority 2: skipped / MTS-failed gap, if implemented
Priority 3: highest finite global variance
```

For uncovered intervals, keep:

```python
x_rescue_target = midpoint_of_widest_uncovered_interval_snapped_to_grid
```

For finite-variance rescue, keep:

```python
x_rescue_target = argmax(global_variance)
```

---

# 2. Add retry-aware rescue stiffness escalation

Add helper:

```python
def count_previous_rescue_retries(
    rescue_rows: list[dict[str, Any]],
    *,
    target_priority: str,
    x_rescue_target: float,
    matched_child_name: str | None,
    grid_dx: float,
) -> int:
    ...
```

Recommended behavior:

- For uncovered-interval rescue:
  - Count previous rescue rows where:
    ```text
    target_priority == "uncovered_interval"
    ```
    and either:
    ```text
    matched_child_name is the same
    ```
    or:
    ```text
    abs(previous.x_rescue_target - current.x_rescue_target) <= grid_dx
    ```
- For non-uncovered rescue:
  - Return `0`.

This retry count is used so the first rescue attempt is:

```text
s_rescue^1 * k_m
```

the second failed/repeated attempt is:

```text
s_rescue^2 * k_m
```

and so on.

---

# 3. Update `design_rescue_window(...)`

Modify `design_rescue_window` to accept `rescue_rows`:

```python
def design_rescue_window(
    *,
    target_info: dict[str, Any],
    generations_root: Path,
    grid: np.ndarray,
    args: argparse.Namespace,
    rescue_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ...
```

Compute:

```python
grid_dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else float(args.bin_width)
x_rescue_target = float(target_info["target_x"])
target_priority = str(target_info.get("target_priority", "finite_variance"))
```

---

## 3.1 Uncovered-interval rescue

For:

```python
target_priority == "uncovered_interval"
```

still find the matched child by closest previous `target_x`:

```python
matched = match_child_design_for_rescue_target(child_designs, x_rescue_target)
```

But if a valid match exists, set:

```python
rescue_center_x = nearest_grid_value(x_rescue_target, grid)
```

not:

```python
rescue_center_x = matched["center_x"]
```

Use the matched child only for the force constant:

```python
n_retry = count_previous_rescue_retries(
    rescue_rows or [],
    target_priority=target_priority,
    x_rescue_target=x_rescue_target,
    matched_child_name=str(matched["name"]),
    grid_dx=grid_dx,
)

rescue_scale = float(args.s_rescue) ** float(n_retry + 1)
rescue_k = clip(
    rescue_scale * float(matched["k"]),
    float(args.k_min),
    float(args.k_max),
)
rescue_k_rule = "target_bin_with_retry_scaled_matched_child_k"
```

If no valid matched child exists:

```python
rescue_center_x = nearest_grid_value(x_rescue_target, grid)
rescue_k = float(args.k_rescue)
rescue_k_rule = "fallback_fixed_k_rescue_at_uncovered_target"
n_retry = 0
rescue_scale = 1.0
```

Rationale:

```text
The matched child identifies which previous ensemble tried to sample this region.
The actual rescue window should be placed at the missing bin x*, because repeating x_m can repeat the same failure.
```

---

## 3.2 High-variance finite-bin rescue

For:

```python
target_priority == "finite_variance"
```

or any non-uncovered target, use the former/default behavior:

```python
rescue_center_x = nearest_grid_value(x_rescue_target, grid)
rescue_k = float(args.k_rescue)
rescue_k_rule = "fixed_k_rescue_at_target"
matched = None
n_retry = 0
rescue_scale = 1.0
```

Do not use the matched child center or matched child stiffness for high-variance covered bins.

---

# 4. Update rescue loop call

In the rescue loop, change:

```python
rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
)
```

to:

```python
rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
    rescue_rows=rescue_rows,
)
```

Then continue to create the rescue EQ window with:

```python
center_x=float(rescue_design["rescue_center_x"])
k=float(rescue_design["rescue_k"])
```

---

# 5. Output diagnostics

Add these fields to `rescue_summary.csv` and `rescue_decision.json`:

```text
x_rescue_target
rescue_center_x
rescue_k
rescue_k_rule
s_rescue
rescue_retry_count
rescue_scale
matched_child_name
matched_target_x
matched_target_distance
matched_child_center_x
matched_child_k
```

For uncovered rescue with a valid match, the expected behavior is:

```text
x_rescue_target = x*
rescue_center_x = x*
matched_child_center_x = x_m
rescue_k = clip(s_rescue^(retry+1) * k_m, k_min, k_max)
```

For high-variance rescue, the expected behavior is:

```text
x_rescue_target = x*
rescue_center_x = x*
rescue_k = k_rescue
rescue_k_rule = fixed_k_rescue_at_target
matched_child_name = blank
```

---

# 6. Summary JSON

Update `mines_variance_fusion_summary.json`:

```json
{
  "rescue_k_design": "uncovered_targets_use_x_target_with_retry_scaled_matched_child_k",
  "high_variance_rescue_design": "fixed_k_rescue_at_target",
  "s_rescue": 2.0
}
```

---

# 7. Notebook update

In the rescue table, display:

```text
round
target_priority
x_rescue_target
rescue_center_x
rescue_k
rescue_k_rule
rescue_retry_count
rescue_scale
matched_child_name
matched_target_x
matched_child_center_x
matched_child_k
added_window
```

This makes repeated rescue attempts diagnosable.

---

# 8. Acceptance criteria

The task is complete if:

1. For `target_priority == "uncovered_interval"`, the rescue window is centered at the actual uncovered target:
   ```text
   rescue_center_x = x_rescue_target
   ```
   not at the matched child `center_x`.

2. For uncovered rescue with matched child `m`, the force constant is:
   ```text
   rescue_k = clip(s_rescue^(n_retry + 1) * k_m, k_min, k_max)
   ```

3. Repeated attempts for the same uncovered target or same matched child increase the stiffness.

4. For `target_priority == "finite_variance"`, the rescue uses:
   ```text
   rescue_center_x = x_rescue_target
   rescue_k = k_rescue
   ```
   and does not use matched-child placement.

5. `rescue_summary.csv` records retry count, rescue scale, matched child information, and the final actual `rescue_center_x` and `rescue_k`.

---

# Conceptual summary

The corrected rescue rule should be:

```text
If uncovered interval:
    x* = midpoint of widest uncovered interval
    m = child design with closest target_x to x*
    if m exists:
        x_rescue = x*
        k_rescue = clip(s_rescue^(retry+1) * k_m, k_min, k_max)
    else:
        x_rescue = x*
        k_rescue = k_rescue_default

If high finite variance:
    x* = argmax finite global variance
    x_rescue = x*
    k_rescue = k_rescue_default
```

This avoids repeatedly placing the same rescue window at the old child center while still using the previous failed targeting information to choose an increasingly strong restoring force.
