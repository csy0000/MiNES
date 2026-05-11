# ClaudeCode Instruction: Rescue-only NEQ-fitted GT window placement for MiNES

## Context

We are working on the current MiNES implementation, especially `scripts/mines_variance_fusion.py` or the current equivalent MiNES workflow file. The goal is to keep the **chain-growing process unchanged**, but improve the **rescue phase**.

The current chain-growing process should continue to use the existing **mean-only GT rule** for NEQ schedules and child-window seeding. Do not introduce NEQ-fitted background estimation into the chain-growing stage.

This task only changes the **rescue-window design**.

---

## Main goal

During rescue, instead of using only the mean-only background estimate, add an option to fit a local quadratic background from the **segment-local NEQ-derived PMF** and use that fitted background to construct the rescue window by **Gaussian transport (GT)**.

The rescue window should be designed by targeting:

1. the sampled mean at the target bin, and
2. a target standard deviation obtained by linear interpolation between the two neighboring boundary EQ windows.

Do **not** use force matching in the rescue phase. Force matching is only for child-window seeding in the chain-growing process.

---

## High-level behavior

For each rescue target:

1. Identify the local segment/gap containing the target bin.
2. Identify its two neighboring EQ boundary windows:
   - `left_boundary`: lower-mean boundary window
   - `right_boundary`: higher-mean boundary window
3. Build or reuse the segment-local NEQ PMF:
   - if the segment is EOP-connected, use the bidirectional MTS PMF patch;
   - if the segment is not EOP-connected, use Hummer–Szabo/FEP fallback patch or patches.
4. Fit a local quadratic background from that segment-local NEQ PMF.
5. Use the fitted quadratic background in a GT formula to compute the rescue bias center and spring constant.
6. If the NEQ fit is invalid, fall back to the existing mean-only rescue rule.

---

## Command-line changes

Add a new command-line option:

```bash
--rescue-background-fit-method {neq,mean-only,auto}
```

Recommended behavior:

```text
neq       Try NEQ-fitted quadratic background first; fallback to mean-only if invalid.
mean-only Use the existing mean-only rescue design only.
auto      Same as neq for now, but write explicit fallback diagnostics.
```

Default should be:

```text
neq
```

Also add optional fit-control arguments if convenient:

```bash
--rescue-neq-fit-min-bins 5
--rescue-neq-fit-k0-min-abs 1e-8
--rescue-neq-fit-x0-margin-factor 0.25
```

These can also be hard-coded initially if you want a smaller change.

---

## Weighted quadratic fit

Implement a helper function, for example:

```python
def fit_quadratic_background_from_segment_patch(
    *,
    segment: NEQSegment,
    patch: PMFPatch,
    grid: np.ndarray,
    variance_floor: float,
    min_fit_bins: int = 5,
    k0_min_abs: float = 1.0e-8,
) -> dict[str, Any]:
    ...
```

The fit is segment-local. Do not fit to the global PMF. Do not use bins from unrelated segments.

The target objective is:

```text
L = Σ_i [F_model(x_i) - F_NEQ(x_i)]² / σ_i²
```

where the model is:

```text
F_model(x) = 0.5 * k0 * (x - x0)^2 + F0
```

Use the patch variance as:

```text
w_i = 1 / (variance_i + variance_floor)
```

Include only bins where all of the following are true:

```python
patch.coverage_mask[i] is True
np.isfinite(patch.pmf[i])
np.isfinite(patch.variance[i])
```

Use a numerically stable linear weighted least-squares form:

```text
F_model(x) = a*x^2 + b*x + c
```

Then convert:

```text
k0 = 2*a
x0 = -b / (2*a)
F0 = c - 0.5*k0*x0^2
```

The fit result dictionary should include at least:

```text
fit_accepted
fit_source
segment
patch_name
n_fit_bins
x_fit_min
x_fit_max
k0
x0
F0
a
b
c
weighted_rmse
reduced_chi2
fallback_reason
```

Acceptance criteria:

```text
n_fit_bins >= min_fit_bins
k0, x0, F0 are finite
abs(k0) >= k0_min_abs
weighted_rmse is finite
x0 is not absurdly far outside the local segment interval
```

For the x0 sanity check, use the local boundary means:

```text
m_L = left_boundary.mean_x
m_R = right_boundary.mean_x
segment_width = abs(m_R - m_L)
margin = max(2*bin_width, x0_margin_factor * segment_width)
accept x0 only if min(m_L,m_R)-margin <= x0 <= max(m_L,m_R)+margin
```

If any criterion fails, set `fit_accepted=False` and provide `fallback_reason`.

---

## Which NEQ patch to fit

For rescue, use only NEQ-derived patches from the same segment being rescued.

Recommended first implementation:

```text
1. Prefer the segment's NEQ_MTS patch if it exists and has enough finite variance-covered bins.
2. If no valid MTS patch exists, use HS/FEP fallback patch from the same segment.
3. If both forward and reverse HS fallback patches exist, fit them separately and choose the accepted fit with the lower weighted RMSE.
4. If no accepted NEQ fit exists, fallback to mean-only rescue design.
```

Do not combine forward and reverse HS PMFs into one fit unless their relative offset is explicitly fitted. For now, separate fits and choosing the better accepted fit is safer.

Important: if the segment is EOP-disconnected, still try to build or reuse HS/FEP fallback patches for the fit.

---

## Rescue GT construction

Once a valid NEQ fit is available, use it to construct the rescue window with GT.

For the local segment boundaries define:

```text
m_L     = left_boundary.mean_x
m_R     = right_boundary.mean_x
sigma_L = left_boundary.std_x
sigma_R = right_boundary.std_x
x_L     = left_boundary.center_x
x_R     = right_boundary.center_x
```

The rescue target is the selected target bin:

```text
m_target = x_target_bin
```

Compute the interpolation coordinate:

```text
s = (m_target - m_L) / (m_R - m_L)
```

If `m_R` and `m_L` are nearly equal, fallback to mean-only rescue design.

If `s <= 0` or `s >= 1`, set:

```text
s = 0.5
```

Then interpolate the target standard deviation linearly:

```text
sigma_target = (1 - s) * sigma_L + s * sigma_R
K_target = 1 / sigma_target^2
```

Using the fitted quadratic background parameters `k0` and `x0`, compute:

```text
k_raw = K_target - k0
k_rescue = clip(k_raw, k_min, k_max)
```

Then compute the umbrella center that should produce the target mean under the fitted quadratic background:

```text
x_raw = ((k0 + k_rescue) * m_target - k0 * x0) / k_rescue
```

Clamp the rescue center to the local segment interval and the analysis range:

```text
x_rescue = clamp(x_raw, min(x_L, x_R), max(x_L, x_R))
x_rescue = clamp(x_rescue, analysis_xmin, analysis_xmax)
```

Use:

```text
center_x = x_rescue
k = k_rescue
```

for the rescue EQ window.

Do not use force matching here.

---

## No transition-crossing rule in rescue

Do not apply the child-window transition-crossing logic to rescue.

For rescue, the fitted `x0` is only used in the GT equations. We are directly targeting the mean at the selected target bin, so there is no left-growing or right-growing child window to accept or reject.

---

## Fallback behavior

If any of the following happens, use the current mean-only rescue design:

```text
no segment-local NEQ patch exists
NEQ patch has too few valid bins
quadratic fit is invalid
sigma_target is invalid
k_raw is non-finite
k_rescue is non-positive or non-finite after clipping
x_raw is non-finite
```

The fallback must be recorded in the diagnostics.

---

## Keep chain-growing unchanged

Do not alter:

```text
chain-growing child proposal
chain-growing mean-only GT schedule
chain-growing NEQ protocol generation
force matching used for child seeding
```

The current mean-only GT mechanism should remain the default for chain-growing NEQ protocols.

---

## Diagnostics and output files

Extend `rescue_summary.csv` with columns such as:

```text
rescue_background_fit_method
rescue_fit_source
rescue_fit_segment
rescue_fit_patch
rescue_fit_accepted
rescue_fit_fallback_reason
rescue_fit_n_bins
rescue_fit_x_min
rescue_fit_x_max
rescue_fit_k0
rescue_fit_x0
rescue_fit_F0
rescue_fit_weighted_rmse
rescue_fit_reduced_chi2
rescue_gt_m_L
rescue_gt_m_R
rescue_gt_sigma_L
rescue_gt_sigma_R
rescue_gt_m_target
rescue_gt_s_raw
rescue_gt_s_used
rescue_gt_sigma_target
rescue_gt_K_target
rescue_gt_k_raw
rescue_gt_k_final
rescue_gt_x_raw
rescue_gt_x_final
rescue_gt_x_clipped_to_segment
rescue_gt_x_clipped_to_analysis_range
rescue_design_rule
```

Also write a separate table:

```text
rescue_neq_quadratic_fits.csv
```

Each attempted NEQ fit should write one row with:

```text
round
segment
patch_name
patch_kind
fit_source
fit_accepted
fallback_reason
n_fit_bins
x_fit_min
x_fit_max
k0
x0
F0
a
b
c
weighted_rmse
reduced_chi2
variance_floor
```

If possible, also write a per-fit JSON file inside the rescue round directory:

```text
rescue_round_<n>/neq_quadratic_fit_summary.json
```

---

## Notebook updates

Update the MiNES visualization notebook to include rescue-fit diagnostics.

Add a table display for:

```text
rescue_neq_quadratic_fits.csv
```

Add a plot for each rescue round if data are available:

```text
x vs segment-local NEQ PMF used for fitting
x vs fitted quadratic background
mark x0
mark m_target
mark x_rescue
```

The notebook should be robust if the new file does not exist.

---

## Acceptance criteria

The change is correct if:

1. Chain-growing behavior remains unchanged and still uses mean-only GT.
2. A new CLI option `--rescue-background-fit-method` exists.
3. Rescue can use segment-local NEQ PMF patches to fit a weighted quadratic background.
4. The weighted objective is squared:
   `Σ_i [F_model(x_i) - F_NEQ(x_i)]² / σ_i²`.
5. Rescue window construction uses GT:
   - target mean equals the rescue target bin;
   - target standard deviation is linearly interpolated between neighboring boundary EQ windows;
   - `k_rescue = K_target - k0`;
   - `x_rescue = ((k0 + k_rescue) * m_target - k0*x0) / k_rescue`.
6. Rescue does not use force matching.
7. Rescue does not apply left/right child transition-crossing logic.
8. Invalid NEQ fits cleanly fallback to the current mean-only rescue rule.
9. `rescue_summary.csv` records all fit, GT, and fallback diagnostics.
10. `rescue_neq_quadratic_fits.csv` is written.
11. Global PMF fusion remains unchanged.
