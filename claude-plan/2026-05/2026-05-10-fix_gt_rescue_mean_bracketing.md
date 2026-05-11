# Fix MiNES GT rescue: choose adjacent EQ windows by observed mean_x

## Context

The current MiNES rescue implementation is producing repeated rescue windows with the same center and spring constant, for example:

```text
M4: x_m = 0.1187, mean_x = -1.3157, k = 12.3371
M5: x_m = 0.1187, mean_x = -1.3682, k = 12.3371
```

This indicates that the GT rescue code is likely reusing the same reference pair and/or falling back to the same midpoint proposal. The intended behavior is different: every rescue round should recompute the GT reference pair from the currently available EQ windows.

The GT rescue interpolation must be based on **observed EQ means** (`mean_x`), not `center_x`, `x_most`, generation order, cluster order, or existing NEQ-connected boundaries.

---

## Intended algorithm

For a rescue target bin `target_bin_x`, select two adjacent EQ windows whose observed EQ means bracket the target:

```text
m_L < target_bin_x < m_R
```

where:

```text
m_L = left_window.mean_x
m_R = right_window.mean_x
```

Then compute the GT progress parameter:

```python
s_raw = (target_bin_x - m_L) / (m_R - m_L)
```

If the bracketing pair was selected correctly, then normally:

```python
0.0 < s_raw < 1.0
```

Only if `s_raw <= 0.0` or `s_raw >= 1.0`, use the explicit fallback:

```python
s_used = 0.5
```

and record that fallback in the rescue diagnostics.

---

## Concrete example from current run

The current windows have this `mean_x` ordering:

```text
L0: -10.1974
L1:  -7.6472
M1:  -1.9263
M5:  -1.3682
M4:  -1.3157
M3:   1.9994
M2:   3.1230
R1:   7.3908
R0:   9.6739
```

Therefore:

```text
If target_bin_x is between M1.mean_x and M3.mean_x before M4 exists:
    use M1 and M3

After M4 exists:
    if target_bin_x is between M1.mean_x and M4.mean_x:
        use M1 and M4

    if target_bin_x is between M4.mean_x and M3.mean_x:
        use M4 and M3
```

In particular, if M4 has already been sampled and its mean lies between M1 and M3, then a later rescue target between M4 and M3 must use the pair `M4/M3`, not the old pair `M1/M3`.

---

## Required code changes

### 1. Add a mean-bracketing helper

Add a helper similar to this:

```python
def choose_mean_bracketing_windows_for_gt_rescue(
    *,
    target_bin_x: float,
    windows: list[EnsembleWindow],
) -> tuple[EnsembleWindow, EnsembleWindow, dict[str, Any]] | None:
    candidates = [
        w for w in windows
        if math.isfinite(float(w.mean_x))
        and math.isfinite(float(w.std_x))
        and float(w.std_x) > 0.0
    ]
    candidates = sorted(candidates, key=lambda w: (float(w.mean_x), str(w.name)))

    for left, right in zip(candidates[:-1], candidates[1:]):
        m_L = float(left.mean_x)
        m_R = float(right.mean_x)
        if m_L < float(target_bin_x) < m_R:
            return left, right, {
                "gt_pair_rule": "adjacent_eq_windows_bracketing_target_by_mean_x",
                "gt_left_boundary": left.name,
                "gt_right_boundary": right.name,
                "gt_left_mean_x": m_L,
                "gt_right_mean_x": m_R,
                "gt_left_center_x": float(left.center_x),
                "gt_right_center_x": float(right.center_x),
                "gt_left_std_x": float(left.std_x),
                "gt_right_std_x": float(right.std_x),
                "gt_left_x_most": float(left.x_most),
                "gt_right_x_most": float(right.x_most),
                "gt_n_candidate_windows": len(candidates),
            }

    return None
```

Important: this helper must be called using the full current list of EQ windows, including all rescue windows that have already been sampled.

---

### 2. Stop using cluster boundary selection for GT rescue

In the GT rescue path, do **not** select the GT reference windows via:

```python
gap_clusters_for_target(...)
choose_connected_boundary_pair(...)
cluster order
existing connected NEQ segment availability
```

These can select a pair that does not satisfy:

```text
left.mean_x < target_bin_x < right.mean_x
```

Instead, call the new helper directly using all current EQ windows.

---

### 3. Use `target_bin_x` as the GT target mean

The rescue target should first be snapped to the nearest grid point:

```python
target_bin_index = int(np.argmin(np.abs(grid_arr - x_target)))
target_bin_x = float(grid_arr[target_bin_index])
```

Then the GT target mean should be:

```python
ms = target_bin_x
```

Do not use the unsnapped continuous `target_info["target_x"]` for `ms`.

---

### 4. Compute and record `s_raw` and `s_used`

After selecting the mean-bracketing windows:

```python
m_L = float(left_boundary.mean_x)
m_R = float(right_boundary.mean_x)

if abs(m_R - m_L) < 1.0e-12:
    s_raw = float("nan")
    s_used = 0.5
    s_fallback = True
    fallback_reason = "degenerate_mean_bracket"
else:
    s_raw = (target_bin_x - m_L) / (m_R - m_L)
    if s_raw <= 0.0 or s_raw >= 1.0:
        s_used = 0.5
        s_fallback = True
        fallback_reason = "s_raw_outside_unit_interval_after_mean_bracketing"
    else:
        s_used = s_raw
        s_fallback = False
        fallback_reason = ""
```

Then compute the GT proposal using `s_used`.

If the current code uses `get_xs_ks_from_ms(...)`, either:

1. update that function so it uses this external `s_used` logic and does not silently choose a midpoint fallback; or
2. add a new rescue-specific helper, for example `get_xs_ks_from_gt_rescue_s(...)`, that directly takes `s_used` and `ms=target_bin_x`.

The key requirement is that `s_raw`, `s_used`, and whether the fallback was triggered must be explicit in the rescue output.

---

### 5. GT formula to use after pair selection

For the selected windows:

```python
EQ_L = eq_gt_tuple(left_boundary)   # center_x, k, mean_x, std_x
EQ_R = eq_gt_tuple(right_boundary)

x0_L, k0_L, x0_R, k0_R, harmonic_meta = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)
```

Then, for the rescue target:

```python
ms = target_bin_x
sigma_s = (1.0 - s_used) * sigma_L + s_used * sigma_R
x0_s = (1.0 - s_used) * x0_L + s_used * x0_R
k0_s = (1.0 - s_used) * k0_L + s_used * k0_R
K_s = 1.0 / (sigma_s * sigma_s)
k_raw = K_s - k0_s
k_res = clip(k_raw, k_min, k_max)
x_raw = ((k0_s + k_res) * ms - k0_s * x0_s) / k_res
x_res = clip(x_raw, min(left_boundary.center_x, right_boundary.center_x), max(left_boundary.center_x, right_boundary.center_x))
```

Use the existing clipping utility if available.

---

### 6. Fallback when no bracketing pair exists

If no adjacent mean-bracketing pair exists, do **not** reuse an arbitrary cluster boundary pair. Use an explicit fallback and record it.

Acceptable fallback:

```python
# nearest two windows by mean_x around target, or target-bin centered rescue
rescue_center_x = clamp_to_bounds(target_bin_x, analysis_xmin, analysis_xmax)
sigma_target = max(1.5 * grid_dx, 0.20)
k_from_sigma = kT / sigma_target**2
rescue_k = clip(max(args.k_rescue, k_from_sigma), args.k_min, args.k_max)
```

Record:

```text
gt_fallback_reason = "no_adjacent_mean_bracketing_pair"
rescue_center_rule = "target_bin_fallback_no_mean_bracket"
rescue_k_rule = "target_bin_sigma_fallback_no_mean_bracket"
```

---

## Required diagnostics in `rescue_summary.csv`

Ensure every rescue round writes these columns:

```text
x_rescue_target
target_bin_x
target_bin_index
rescue_center_x_raw
rescue_center_x
rescue_k
rescue_center_rule
rescue_k_rule

gt_pair_rule
gt_left_boundary
gt_right_boundary
gt_left_mean_x
gt_right_mean_x
gt_left_center_x
gt_right_center_x
gt_left_std_x
gt_right_std_x
gt_left_x_most
gt_right_x_most
gt_s_raw
gt_s_used
gt_s_fallback_to_midpoint
gt_fallback_reason
gt_x0_L
gt_k0_L
gt_x0_R
gt_k0_R
gt_x_raw
gt_k_raw
gt_x_clipped
gt_k_clipped
```

These diagnostics are necessary to verify that M5 uses M4/M3 after M4 has been added, instead of reusing M1/M3.

---

## Acceptance criteria

1. GT rescue chooses the two adjacent windows bracketing `target_bin_x` by `mean_x`.
2. The bracketing list is recomputed after every rescue EQ window is sampled.
3. `center_x`, `x_most`, cluster order, generation order, and existing NEQ connectivity are not used to choose the GT rescue reference pair.
4. `target_bin_x`, not the unsnapped target, is used as `ms` in GT rescue.
5. `s_raw`, `s_used`, and fallback status are written to `rescue_summary.csv`.
6. If a previous rescue window creates a new mean bracket, the next rescue uses it. Example: if M4.mean_x lies between M1.mean_x and M3.mean_x, then a later target between M4.mean_x and M3.mean_x must use M4/M3.
7. Repeated rescue windows with identical `x_m` and `k` should only occur if the target and selected mean-bracketing pair are genuinely identical, and the diagnostics must make that clear.

---

## Minimal regression check

After running the workflow, inspect `windows.csv` and `rescue_summary.csv`.

For each rescue row:

```python
assert gt_left_mean_x < target_bin_x < gt_right_mean_x
assert 0.0 < gt_s_raw < 1.0
assert gt_s_used == gt_s_raw
```

except for rows where `gt_fallback_reason` is non-empty.

Also check that after a new rescue window appears between two old reference means, later rescue rounds can use the new rescue window as one of the GT references.
