# ClaudeCode Instruction: Update MiNES to Mean-Only GT and Quantile Crossing Stop

## Goal

Update the current MiNES implementation so that the NEQ scheduling still uses GT by default, but the GT harmonic approximation is changed from the current s-dependent harmonic interpolation to a **single mean-only harmonic background per neighboring EQ-window pair**.

Do **not** add or modify rescue strategy logic in this task. The rescue strategy is not part of the current implementation target.

---

## High-level changes

1. Keep NEQ scheduling as GT by default.
2. Remove the current s-dependent harmonic interpolation from GT scheduling.
3. Add a window-level EQ map sorted by sampled `mean_x`.
4. For each neighboring EQ-window pair, estimate exactly one mean-only `k0_segment` and `x0_segment`.
5. Use this single segment-level `k0_segment, x0_segment` for GT protocol scheduling.
6. Classify each neighboring EQ-map segment as `transition` or `regular`.
7. Replace the growth stop criterion with quantile-based EoP/EQ crossing.
8. Keep CFT/BAR quantities only as diagnostics, not as growth-stop control.
9. Update diagnostics and notebook outputs to remove s-dependent `k0/x0` reporting.

---

## 1. Keep GT scheduling default

The current code already defaults to GT through `--neq-protocol-mode GT`. Keep this behavior.

Do not remove the CLI option. The user should still be able to run:

```bash
--neq-protocol-mode GT
```

or use GT implicitly by default.

The important change is the internal GT formula, not the CLI mode.

---

## 2. Remove s-dependent harmonic interpolation

Remove the old GT logic where endpoint harmonic estimates are interpolated along protocol time:

```python
x0_s = (1.0 - s) * x0_L + s * x0_R
k0_s = (1.0 - s) * k0_L + s * k0_R
```

The updated protocol must not use any s-dependent harmonic background.

Remove or stop using these quantities in the GT scheduling rule:

```text
x0_L, k0_L
x0_R, k0_R
x0_s, k0_s
```

It is acceptable to keep old fields only if needed for backward-compatible reading of old outputs, but new protocol scheduling and new diagnostics should not rely on them.

---

## 3. Add mean-only harmonic estimator

Implement a new helper that estimates one shared harmonic background from two EQ windows using only:

```text
left center:   x_L
left spring:   k_L
left mean:     m_L
right center:  x_R
right spring:  k_R
right mean:    m_R
```

Do not use standard deviations in this estimator.

The estimator is based on the biased equilibrium mean relation:

```text
m_i = (k0 * x0 + k_i * x_i) / (k0 + k_i)
```

which gives:

```text
k0 * (x0 - m_i) = k_i * (m_i - x_i)
```

For the two windows, estimate:

```python
numerator = k_R * (m_R - x_R) - k_L * (m_L - x_L)
denominator = m_L - m_R
k0_segment = numerator / denominator
x0_segment = m_L + k_L * (m_L - x_L) / k0_segment
```

Add robust fallback behavior:

- If `abs(m_L - m_R) < eps`, mark the segment as invalid or use a safe midpoint fallback.
- If `abs(k0_segment) < eps`, mark `x0_segment` as invalid or use a safe fallback.
- Always write the fallback reason to diagnostics.

Suggested function:

```python
def estimate_mean_only_k0_x0_from_eq_pair(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    *,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    ...
```

Suggested return keys:

```text
k0_segment
x0_segment
valid
fallback_used
fallback_reason
left_center_x
right_center_x
left_k
right_k
left_mean_x
right_mean_x
```

---

## 4. Build window-level EQ map

After every growth generation, rebuild an EQ map at the **individual window level**.

Do not build this map at the cluster level.

Algorithm:

1. Collect all EQ windows.
2. Sort them by `(mean_x, name)`.
3. For each neighboring pair `(left_window, right_window)`, estimate `k0_segment, x0_segment` using the new mean-only estimator.
4. Classify the segment as `transition` or `regular`.
5. Write all rows to `eq_map_segments.csv`.

Suggested function:

```python
def build_eq_map_segments(windows: list[EnsembleWindow]) -> list[dict[str, Any]]:
    ordered = sorted(windows, key=lambda w: (float(w.mean_x), str(w.name)))
    rows = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        harmonic = estimate_mean_only_k0_x0_from_eq_pair(left, right)
        ...
    return rows
```

Suggested CSV columns:

```text
segment_name
left_window
right_window
left_mean_x
right_mean_x
left_center_x
right_center_x
left_k
right_k
k0_segment
x0_segment
harmonic_valid
fallback_used
fallback_reason
segment_type
classification_reason
```

---

## 5. Segment classification rule

For each neighboring pair sorted by sampled `mean_x`, classify the segment as follows:

```python
lo = min(left.mean_x, right.mean_x)
hi = max(left.mean_x, right.mean_x)

is_transition = (
    harmonic_valid
    and lo <= x0_segment <= hi
    and k0_segment < 0.0
)
```

If `is_transition` is true:

```text
segment_type = "transition"
```

Otherwise:

```text
segment_type = "regular"
```

For now, this classification is diagnostic only. Both `transition` and `regular` segments should use the same GT protocol rule.

---

## 6. Update GT protocol generation

The GT protocol may still use an interpolation parameter `s` to define target means along the path:

```python
m_s = (1.0 - s) * m_L + s * m_R
```

This is allowed.

But the harmonic background must be fixed for the whole segment:

```python
k0 = k0_segment
x0 = x0_segment
```

Do not compute or use:

```python
k0_s
x0_s
```

For each protocol timepoint, compute the bias spring and center using the fixed mean-only background.

A reasonable relation is:

```text
m_s = (k0 * x0 + k_s * x_s) / (k0 + k_s)
```

Use the existing GT logic as much as possible, but replace all s-dependent background terms with the fixed segment-level `k0_segment, x0_segment`.

The protocol diagnostics should report:

```text
step_index
s
mode = "GT_mean_only"
m_target
x_raw
x
k_raw
k
x_clipped
k_clipped
k0_segment
x0_segment
segment_type
```

Do not report `x0_s` or `k0_s` for the new protocol.

---

## 7. Replace growth stopping rule with quantile crossing

The growth phase should stop if either the NEQ end-of-protocol distributions cross or the EQ tail distributions cross.

Use quantiles, not strict minima/maxima.

Add constants or CLI arguments:

```python
crossing_quantile_low = 0.05
crossing_quantile_high = 0.95
```

### EoP crossing condition

Use:

```python
q95(forward_EoP) >= q05(reverse_EoP)
```

where:

```text
forward_EoP = final x values from forward NEQ trajectories
reverse_EoP = final x values from reverse NEQ trajectories
```

Suggested helper:

```python
def neq_eop_quantile_crossing(segment: NEQSegment, q_low=0.05, q_high=0.95) -> dict[str, Any]:
    fwd = endpoint_x_from_trajectories(segment.forward_trajectories)
    rev = endpoint_x_from_trajectories(segment.reverse_trajectories)
    fwd_hi = np.quantile(fwd, q_high)
    rev_lo = np.quantile(rev, q_low)
    crossed = bool(fwd_hi >= rev_lo)
    return {...}
```

### EQ crossing condition

Use:

```python
q95(left_EQ_tail) >= q05(right_EQ_tail)
```

where `left_EQ_tail` and `right_EQ_tail` are the frontier/boundary EQ windows.

Suggested helper:

```python
def eq_tail_quantile_crossing(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    q_low=0.05,
    q_high=0.95,
) -> dict[str, Any]:
    left_x = eq_tail_samples(left_window)
    right_x = eq_tail_samples(right_window)
    left_hi = np.quantile(left_x, q_high)
    right_lo = np.quantile(right_x, q_low)
    crossed = bool(left_hi >= right_lo)
    return {...}
```

---

## 8. Demote CFT/BAR from stop rule to diagnostics

Do not use CFT/BAR uncertainty or `cft_delta_f` threshold to decide when growth stops.

If the current code has logic like:

```text
growth_stop_rule = "cft_delta_f_below_threshold"
```

replace it with:

```text
growth_stop_rule = "quantile_eop_or_eq_crossing"
```

Keep BAR/CFT calculations if they are useful for diagnostics, tables, or plots.

---

## 9. Update growth summary output

Write quantile crossing diagnostics to the growth summary.

Suggested columns:

```text
generation
stop_by_eop_crossing
stop_by_eq_crossing
forward_eop_q95
reverse_eop_q05
left_eq_q95
right_eq_q05
crossing_quantile_low
crossing_quantile_high
growth_stop_reason
```

`growth_stop_reason` should be one of:

```text
eop_quantile_crossing
eq_quantile_crossing
eop_and_eq_quantile_crossing
max_generations
budget_exhausted
other
```

Do not report CFT/BAR as the controlling stop reason.

---

## 10. Update state snapshots and final outputs

Every generation/state snapshot should include the new `eq_map_segments.csv`.

If there is a final raw output directory, also write the final EQ map there.

The outputs should make it easy to inspect:

```text
which neighboring windows define each segment
what mean-only k0/x0 was inferred
whether the segment is transition or regular
which segment was used for each GT protocol
```

---

## 11. Update notebook diagnostics

Remove plots/tables that show or depend on s-dependent harmonic backgrounds:

```text
k0(s)
x0(s)
s-dependent harmonic PMFs
```

Replace them with diagnostics for the mean-only EQ map:

```text
segment_type
k0_segment
x0_segment
left_mean_x
right_mean_x
left_window
right_window
```

If plotting derived harmonic PMFs, only plot the segment-level mean-only harmonic background. Do not plot a family of s-dependent harmonic curves.

The GT visualization should show at most:

```text
Analytical PMF
Global MiNES PMF before rescue, if available
Global MiNES PMF after rescue, if available
Mean-only derived PMF from segment-level k0/x0
```

---

## 12. Explicit non-goals for this task

Do not implement rescue retry logic.

Do not add:

```text
alpha_rescale
JSD_rescue_fail
same-center rescue retry
max rescue retries per target
```

Do not change the rescue strategy in this task.

Do not add new rescue behavior unless required to keep existing code running.

---

## Acceptance criteria

The task is complete when:

1. NEQ scheduling still defaults to GT.
2. The GT scheduler no longer uses s-dependent `k0_s` or `x0_s`.
3. Each neighboring EQ-window pair has exactly one mean-only `k0_segment, x0_segment`.
4. The EQ map is built at the individual window level, sorted by sampled `mean_x`.
5. `eq_map_segments.csv` is written after each generation and in the final output.
6. Each EQ-map segment is classified as `transition` or `regular`.
7. Growth stops based on:
   - `q95(forward_EoP) >= q05(reverse_EoP)`, or
   - `q95(left_EQ_tail) >= q05(right_EQ_tail)`.
8. CFT/BAR quantities are kept only as diagnostics, not as growth-stop criteria.
9. Protocol diagnostics no longer report or rely on `k0_s/x0_s`.
10. Notebook diagnostics no longer show s-dependent harmonic interpolation.
11. No rescue retry strategy is added.

