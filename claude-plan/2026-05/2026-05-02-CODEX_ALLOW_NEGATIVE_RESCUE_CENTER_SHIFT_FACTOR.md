# Codex Task: Allow Negative Rescue Center Shift Factor After kmax Saturation

You are working on the MiNES variance-fusion workflow, primarily:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

Preserve the existing output structure. Make a targeted change to the rescue-center shift rule.

---

## Goal

When rescue stiffness has saturated at:

```text
k_rescue = k_max
```

and repeated rescue attempts keep failing, the center-shift factor should be allowed to become negative.

Use the proposed rule:

```text
f = 0.5 * (2 - n_retry)
```

where `n_retry` is the number of previous rescue attempts for the same target region or same matched child.

Then:

```text
x_rescue = x_target + f * (x_m - x_target)
```

Equivalently:

```text
x_rescue = x_target - f * (x_target - x_m)
```

This formulation makes the behavior intuitive:

```text
n_retry = 0: f =  1.0  -> x_rescue = x_m
n_retry = 1: f =  0.5  -> halfway between x_target and x_m
n_retry = 2: f =  0.0  -> x_rescue = x_target
n_retry = 3: f = -0.5  -> move beyond x_target, away from x_m
n_retry = 4: f = -1.0  -> move further beyond x_target
```

So after force-constant escalation reaches `k_max`, the algorithm first moves from the matched child center toward the target, then can overshoot beyond the target if the target remains unresolved.

---

# 1. Replace nonnegative shift-fraction rule

If the current code uses a rule like:

```python
center_shift_fraction = min(
    1.0,
    float(args.rescue_center_shift_fraction) * float(saturated_retry_count),
)
rescue_center_x_raw = matched_center_x + center_shift_fraction * (
    x_rescue_target - matched_center_x
)
```

replace it with the signed `f` rule.

---

# 2. Add or update CLI options

Keep or add:

```python
parser.add_argument("--rescue-center-f-slope", default=0.5, type=float)
parser.add_argument("--rescue-center-f-start", default=2.0, type=float)
parser.add_argument("--rescue-center-f-min", default=-2.0, type=float)
parser.add_argument("--rescue-center-f-max", default=1.0, type=float)
```

These implement:

```python
f_raw = rescue_center_f_slope * (rescue_center_f_start - n_retry)
f = clip(f_raw, rescue_center_f_min, rescue_center_f_max)
```

Default values reproduce:

```text
f = 0.5 * (2 - n_retry)
```

with clipping to avoid unbounded overshoot.

If you prefer fewer CLI options, at minimum hard-code:

```python
f_raw = 0.5 * (2.0 - float(n_retry))
f = min(max(f_raw, -2.0), 1.0)
```

but CLI options are preferred for diagnostics and tuning.

---

# 3. New rescue-center formula

When a matched child exists:

```python
x_target = float(target_info["target_x"])
x_m = float(matched["center_x"])
```

If `k_rescue` has not saturated:

```python
rescue_center_x_raw = x_m
rescue_center_rule = "matched_child_center_before_kmax"
f_raw = 1.0
f = 1.0
```

If `k_rescue` has saturated at `k_max`, use:

```python
f_raw = float(args.rescue_center_f_slope) * (
    float(args.rescue_center_f_start) - float(n_retry)
)
f = min(
    max(f_raw, float(args.rescue_center_f_min)),
    float(args.rescue_center_f_max),
)

rescue_center_x_raw = x_target + f * (x_m - x_target)
rescue_center_rule = "signed_f_shift_after_kmax"
```

Then clamp/snap:

```python
rescue_center_x = clamp_to_bounds_and_grid(
    rescue_center_x_raw,
    grid,
    analysis_xmin,
    analysis_xmax,
)
```

Important: because `f` may be negative, `rescue_center_x_raw` can lie beyond `x_target`. The final clamped center must still remain inside `[x_L0, x_R0]`.

---

# 4. Stiffness rule remains unchanged

Keep:

```python
rescue_scale = float(args.s_rescue) ** float(n_retry + 1)
rescue_k_unclipped = rescue_scale * float(matched["k"])
rescue_k = clip(rescue_k_unclipped, args.k_min, args.k_max)
rescue_k_saturated = rescue_k >= args.k_max - 1.0e-12
```

Only the center rule changes after saturation.

---

# 5. Retry counting

Keep the existing retry logic:

```text
same matched child OR same target region
```

Do not require the same `target_priority`.

This is important because a region can start as uncovered and later become high variance. It should still be treated as a retry sequence.

---

# 6. Diagnostics

Add these fields to `rescue_summary.csv` and `rescue_decision.json`:

```text
rescue_center_f_raw
rescue_center_f
rescue_center_f_slope
rescue_center_f_start
rescue_center_f_min
rescue_center_f_max
rescue_center_x_raw
rescue_center_x
rescue_center_clamped_to_bounds
rescue_center_rule
rescue_k_unclipped
rescue_k_saturated
rescue_retry_count
```

For example:

```text
n_retry = 0, f =  1.0, x_rescue = x_m
n_retry = 1, f =  0.5, x_rescue halfway between x_m and x_target
n_retry = 2, f =  0.0, x_rescue = x_target
n_retry = 3, f = -0.5, x_rescue beyond x_target
```

---

# 7. Summary JSON

Add:

```json
{
  "rescue_center_rule": "signed_f_shift_after_kmax",
  "rescue_center_f_formula": "f = rescue_center_f_slope * (rescue_center_f_start - n_retry)",
  "rescue_center_f_default": "0.5 * (2 - n_retry)",
  "rescue_center_f_min": -2.0,
  "rescue_center_f_max": 1.0
}
```

---

# 8. Notebook update

Update the rescue table to include:

```text
round
target_priority
x_rescue_target
matched_child_center_x
rescue_center_f
rescue_center_x_raw
rescue_center_x
rescue_center_rule
rescue_k
rescue_k_saturated
rescue_retry_count
added_window
```

This should make the center movement sequence visible.

---

# 9. Acceptance criteria

The task is complete if:

1. After `k_rescue` saturates at `k_max`, the center shift uses:
   ```text
   f = 0.5 * (2 - n_retry)
   ```
   by default.

2. `f` can become negative.

3. The rescue center is computed as:
   ```text
   x_rescue = x_target + f * (x_m - x_target)
   ```

4. For:
   ```text
   n_retry = 0, 1, 2, 3
   ```
   the center sequence is:
   ```text
   x_m, halfway, x_target, beyond x_target
   ```

5. The final rescue center is clamped to `[x_L0, x_R0]`.

6. `rescue_summary.csv` records `f_raw`, `f`, `rescue_center_x_raw`, `rescue_center_x`, and whether clamping happened.

---

# Conceptual summary

Once increasing `k` is no longer possible, rescue should not keep repeating the same umbrella. It should move the umbrella center according to:

```text
x_rescue = x_target + f * (x_m - x_target)
f = 0.5 * (2 - n_retry)
```

This starts from the matched child center, moves toward the target, and then overshoots beyond the target if repeated rescue attempts still fail.
