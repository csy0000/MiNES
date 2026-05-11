# Codex Task: Shift Rescue Center Immediately When `s_rescue * k_m` Exceeds `k_max`

You are working on:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

Make a targeted change to `design_rescue_window(...)`.

---

## Problem

The current logic still repeats the matched child center on the first saturated rescue attempt.

Current behavior is effectively:

```python
rescue_scale = s_rescue ** (n_retry + 1)
rescue_k_unclipped = rescue_scale * matched_child_k
rescue_k = clip(rescue_k_unclipped, k_min, k_max)

if rescue_k is saturated:
    f = 0.5 * (2 - n_retry)
    x_rescue = x_target + f * (x_m - x_target)
```

For `n_retry = 0`:

```text
f = 1
x_rescue = x_m
```

Therefore, if the matched child already has `k_m = k_max`, the first rescue still repeats the same ensemble:

```text
x_rescue = x_m
k_rescue = k_max
```

This is the behavior currently seen with repeated rescue from `R2`.

---

# Required behavior

If the first scaled stiffness is already above `k_max`:

```text
s_rescue * k_m >= k_max
```

then rescue should **immediately shift the center**, even for `n_retry = 0`.

The stiffness should remain:

```text
k_rescue = k_max
```

but the center should not remain at `x_m`.

---

# 1. Add a saturated retry index

Inside `design_rescue_window(...)`, after matching the child:

```python
x_target = float(target_info["target_x"])
x_m = float(matched["center_x"])
k_m = float(matched["k"])
n_retry = count_previous_rescue_retries(...)
```

Compute:

```python
first_scaled_k = float(args.s_rescue) * k_m
first_scaled_saturates = first_scaled_k >= float(args.k_max) - 1.0e-12
```

Then compute stiffness as before:

```python
rescue_scale = float(args.s_rescue) ** float(n_retry + 1)
rescue_k_unclipped = rescue_scale * k_m
rescue_k = clip(rescue_k_unclipped, args.k_min, args.k_max)
rescue_k_saturated = rescue_k >= float(args.k_max) - 1.0e-12
```

Now define the center-shift retry index:

```python
if rescue_k_saturated and first_scaled_saturates:
    center_retry_index = n_retry + 1
elif rescue_k_saturated:
    center_retry_index = n_retry
else:
    center_retry_index = 0
```

Rationale:

- If `s_rescue * k_m` already exceeds `k_max`, then the first rescue has no useful stiffness escalation left.
- Therefore it should behave like a shifted rescue attempt, not a repeated `x_m` attempt.

---

# 2. Use shifted center when first scaled stiffness saturates

Replace the current saturated-center rule with:

```python
if not rescue_k_saturated:
    f_raw = 1.0
    f = 1.0
    rescue_center_x_raw = x_m
    rescue_center_rule = "matched_child_center_before_kmax"

else:
    if first_scaled_saturates:
        center_retry_index = n_retry + 1
        rescue_center_rule = "signed_f_shift_immediate_because_first_scaled_k_exceeds_kmax"
    else:
        center_retry_index = n_retry
        rescue_center_rule = "signed_f_shift_after_kmax"

    f_raw = float(args.rescue_center_f_slope) * (
        float(args.rescue_center_f_start) - float(center_retry_index)
    )
    f = min(
        max(f_raw, float(args.rescue_center_f_min)),
        float(args.rescue_center_f_max),
    )
    rescue_center_x_raw = x_target + f * (x_m - x_target)
```

With defaults:

```text
f = 0.5 * (2 - center_retry_index)
```

If `s_rescue * k_m >= k_max`:

```text
n_retry = 0 -> center_retry_index = 1 -> f = 0.5
n_retry = 1 -> center_retry_index = 2 -> f = 0.0
n_retry = 2 -> center_retry_index = 3 -> f = -0.5
```

So the first saturated rescue is placed halfway between `x_m` and `x_target`, not exactly at `x_m`.

---

# 3. Keep clamping

After computing `rescue_center_x_raw`, continue to clamp and snap:

```python
rescue_center_x = clamp_to_bounds_and_grid(
    rescue_center_x_raw,
    grid,
    analysis_xmin,
    analysis_xmax,
)
```

This is especially important because negative `f` can overshoot beyond `x_target`.

---

# 4. Add diagnostics

Add these fields to `rescue_summary.csv` and `rescue_decision.json`:

```text
matched_child_k
first_scaled_k
first_scaled_saturates
center_retry_index
rescue_center_f_raw
rescue_center_f
rescue_center_x_raw
rescue_center_x
rescue_center_rule
rescue_k_unclipped
rescue_k_saturated
```

Definitions:

```text
first_scaled_k
    s_rescue * matched_child_k

first_scaled_saturates
    True if first_scaled_k >= k_max

center_retry_index
    The retry index used in f = 0.5 * (2 - center_retry_index)
```

---

# 5. Notebook update

In the rescue table, display:

```text
round
target_priority
x_rescue_target
matched_child_name
matched_child_center_x
matched_child_k
first_scaled_k
first_scaled_saturates
center_retry_index
rescue_center_f
rescue_center_x
rescue_k
rescue_k_saturated
rescue_center_rule
added_window
```

This makes it clear why a rescue window moved immediately instead of repeating the matched child.

---

# 6. Acceptance criteria

The implementation is correct if:

1. If `s_rescue * k_m < k_max`, the first rescue still uses:

   ```text
   x_rescue = x_m
   k_rescue = s_rescue * k_m
   ```

2. If `s_rescue * k_m >= k_max`, the first rescue uses:

   ```text
   k_rescue = k_max
   x_rescue = x_target + 0.5 * (x_m - x_target)
   ```

   with default `f = 0.5 * (2 - center_retry_index)` and `center_retry_index = 1`.

3. The second saturated retry uses:

   ```text
   x_rescue = x_target
   ```

4. The third saturated retry can overshoot beyond `x_target` because `f` becomes negative.

5. Rescue centers remain clamped to `[x_L0, x_R0]`.

6. `rescue_summary.csv` records `first_scaled_k`, `first_scaled_saturates`, and `center_retry_index`.

---

# Conceptual summary

The corrected saturated-rescue rule is:

```text
If matched child exists:
    first_scaled_k = s_rescue * k_m

    if first_scaled_k < k_max:
        first rescue can repeat x_m with stronger k

    if first_scaled_k >= k_max:
        stronger k is impossible
        first rescue must already move x away from x_m toward x_target
```

This prevents repeated rescue windows like:

```text
R2: x=-0.2, k=50
M1: x=-0.2, k=50
```

and instead produces:

```text
M1: x halfway between R2 center and x_target, k=50
```
