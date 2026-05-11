# ClaudeCode Task: Replace GT rescue design with mean-only background + self-calibrated sigma targeting

## Goal

Update the MiNES rescue-window design so that the local underlying harmonic background is estimated **only from the displacement between each window mean and its umbrella center**, not from the sampled standard deviations.

The sampled standard deviations should still be used, but only to choose the **target rescue width** through interpolation. This is a hybrid rescue rule:

1. Estimate local background harmonic parameters `k0` and `x0` from means only.
2. Compute the target interpolation coordinate `s` from the target bin and neighboring means.
3. If `s <= 0` or `s >= 1`, directly use `s = 0.5` as a fallback.
4. Interpolate the target width `sigma_s` from neighboring sigmas.
5. Self-calibrate an effective `kT_eff` from neighboring windows instead of directly reading `thermal_kT`.
6. Use `kT_eff`, `sigma_s`, and mean-derived `k0` to compute `k_res`.
7. Use signed `k0` and `x0` to shift the rescue center so that the expected biased mean lands at `x_target`.
8. Add diagnostics to the rescue summary and notebook so we can verify why each rescue window is placed where it is.

---

## Files to modify

Primary file:

```text
scripts/mines_variance_fusion.py
```

Notebook:

```text
notebooks/mines_variance_fusion_visualization.ipynb
```

Only modify other files if they are necessary for tests or imports.

---

## Current problem

The current GT rescue design estimates local harmonic background parameters using sampled standard deviations. This has proven unreliable for rescue windows near difficult/high-variance regions because `sigma_L` and `sigma_R` can distort the inferred `k0` and produce weak or misplaced rescue umbrellas.

We now want standard deviations to influence only the desired rescue width, **not the underlying PMF force/curvature estimate**.

---

## New rescue model

Assume the local unbiased/background PMF is harmonic:

```math
U_0(x) = \frac{1}{2} k_0 (x-x_0)^2
```

Each biased window has umbrella:

```math
U_i^{bias}(x) = \frac{1}{2} k_i (x-x_i)^2
```

At the biased equilibrium mean `m_i`, force balance gives:

```math
k_0(m_i-x_0) + k_i(m_i-x_i) = 0
```

For the left and right neighboring anchor windows:

```math
k_0(m_L-x_0) + k_L(m_L-x_L) = 0
```

```math
k_0(m_R-x_0) + k_R(m_R-x_R) = 0
```

Subtract to derive the mean-only background curvature:

```math
k_0 = \frac{k_L(m_L-x_L)-k_R(m_R-x_R)}{m_R-m_L}
```

Then derive the background harmonic center:

```math
x_0 = m_L + \frac{k_L}{k_0}(m_L-x_L)
```

The equivalent right-window expression may be logged as a consistency check:

```math
x_{0,Rcheck} = m_R + \frac{k_R}{k_0}(m_R-x_R)
```

Do not use `sigma_L` or `sigma_R` to estimate `k0` or `x0`.

---

## Targeting high-variance bin

Let the high-variance target bin be:

```math
x^{target}
```

Set the desired biased mean directly to this bin:

```math
m_s = x^{target}
```

Compute the raw interpolation coordinate:

```math
s_{raw} = \frac{x^{target}-m_L}{m_R-m_L}
```

Use the requested fallback:

```math
s =
\begin{cases}
s_{raw}, & 0 < s_{raw} < 1 \\
0.5, & s_{raw} \le 0 \text{ or } s_{raw} \ge 1
\end{cases}
```

Do **not** clip `s_raw` to `[0,1]`. Use exactly the fallback-to-midpoint behavior above.

Then define the target width by linear interpolation of observed neighboring sigmas:

```math
\sigma_s = (1-s)\sigma_L + s\sigma_R
```

Here `sigma_L` and `sigma_R` should be computed from the EQ tail samples of the left and right anchor windows.

---

## Self-calibrated effective kT

Because the thermal scale is implicitly present in the relation between observed width and total biased curvature, estimate an effective `kT` from the two neighboring windows:

```math
kT_{eff,L} = (k_0+k_L)\sigma_L^2
```

```math
kT_{eff,R} = (k_0+k_R)\sigma_R^2
```

Use the average:

```math
kT_{eff} = \frac{1}{2}\left(kT_{eff,L}+kT_{eff,R}\right)
```

Add a diagnostic ratio:

```math
kT_{eff,ratio} = \frac{\max(kT_{eff,L}, kT_{eff,R})}{\min(kT_{eff,L}, kT_{eff,R})}
```

Handle invalid or nonpositive values robustly:

- If either `kT_eff_L` or `kT_eff_R` is not finite, record a diagnostic flag.
- If both are finite but one is `<= 0`, record a diagnostic flag.
- If `kT_eff <= 0`, fall back to `ctx["thermal_kT"]` if available, otherwise `1.0`, and mark `kT_eff_fallback_used = True`.
- Otherwise use the self-calibrated `kT_eff` and mark `kT_eff_fallback_used = False`.

---

## Rescue stiffness

Compute the raw rescue stiffness using the target width and mean-only background curvature:

```math
k_{res}^{raw} = \frac{kT_{eff}}{\sigma_s^2} - k_0
```

Then clip to the configured bounds:

```math
k_{res} = \mathrm{clip}(k_{res}^{raw}, k_{min}, k_{max})
```

Use the existing `k_min` and `k_max` arguments.

Do not use `abs(k0)` in the center formula. The sign of `k0` is physically important.

---

## Rescue center

Choose the rescue umbrella center so that the expected biased mean lands at the target bin:

```math
x_{res}^{raw} = x^{target} + \frac{k_0}{k_{res}}(x^{target}-x_0)
```

Then clip spatially to the relevant analysis/gap bounds:

```math
x_{res} = \mathrm{clip}(x_{res}^{raw}, x_{lower}, x_{upper})
```

Use the same spatial bounds that the current rescue design uses for preventing out-of-range rescue windows, preferably the analysis bounds or the selected neighboring gap bounds depending on the current code structure.

Record whether clipping occurred.

---

## Required implementation structure

Add a new helper function, or replace the existing GT rescue helper, with a clear name such as:

```python
def design_rescue_window_mean_only_gt(...):
    ...
```

Suggested inputs:

```python
left_window: EnsembleWindow
right_window: EnsembleWindow
x_target: float
analysis_xmin: float
analysis_xmax: float
args: argparse.Namespace
ctx: dict[str, Any]
```

Suggested output: a dictionary containing all design parameters, including `rescue_center_x` and `rescue_k`.

The function should use:

```python
x_L = left_window.center_x
k_L = left_window.k
m_L = left_window.mean_x
sigma_L = left_window.std_x

x_R = right_window.center_x
k_R = right_window.k
m_R = right_window.mean_x
sigma_R = right_window.std_x
```

If `m_R == m_L` or nearly equal, fall back to a safe midpoint method and record:

```text
fallback_reason = "degenerate_neighbor_means"
```

Safe midpoint fallback:

```python
s = 0.5
sigma_s = 0.5 * (sigma_L + sigma_R)
k_res_raw = max(args.k_rescue, args.k_min)
k_res = clip(k_res_raw, args.k_min, args.k_max)
x_res_raw = x_target
x_res = clip(x_res_raw, analysis_xmin, analysis_xmax)
```

If `k0` is nearly zero, use the same formula where possible, but avoid division by zero in `x0`:

- If `abs(k0) < 1e-12`, set `x0 = nan`, set `x_res_raw = x_target`, and record `fallback_reason = "near_zero_k0_center_shift_disabled"`.
- Still compute `k_res_raw = kT_eff / sigma_s**2 - k0` if `sigma_s` and `kT_eff` are valid.

If `sigma_s <= 0` or not finite, use a fallback sigma:

```python
sigma_s = max(0.5 * (abs(sigma_L) + abs(sigma_R)), args.bin_width, 1e-6)
```

and record `sigma_s_fallback_used = True`.

---

## Rescue summary diagnostics

Add the following columns to `rescue_summary.csv` and any per-round rescue JSON summary:

```text
rescue_design_method
left_window
right_window
x_L
k_L
m_L
sigma_L
x_R
k_R
m_R
sigma_R
x_target
s_raw
s_used
s_fallback_to_midpoint
sigma_s
k0_mean_only
x0_mean_only
x0_right_check
x0_left_right_abs_diff
kT_eff_L
kT_eff_R
kT_eff
kT_eff_ratio
kT_eff_fallback_used
k_res_raw
k_res
k_res_clipped_to
x_res_raw
x_res
x_res_clipped
fallback_reason
```

Use empty strings for unavailable values only if the existing CSV writer requires that; otherwise use NaN/null consistently.

Set:

```text
rescue_design_method = "mean_only_background_sigma_gt_width"
```

---

## Notebook visualization changes

Update `notebooks/mines_variance_fusion_visualization.ipynb` to show the new diagnostics.

For each rescue window, plot:

1. Read/global PMF.
2. Mean-only harmonic background estimate:

```math
F_0(x) = \frac{1}{2} k_0 (x-x_0)^2
```

using `k0_mean_only` and `x0_mean_only`.

3. Rescue umbrella bias:

```math
U_{res}(x) = \frac{1}{2} k_{res}(x-x_{res})^2
```

4. Optional biased local potential:

```math
F_0(x) + U_{res}(x)
```

5. Vertical markers:

```text
x_target
x_res
m_L
m_R
x_L
x_R
```

Also show a compact table with:

```text
window, x_target, x_res, k_res, s_raw, s_used, sigma_s, k0_mean_only, x0_mean_only, kT_eff_ratio, fallback_reason
```

The notebook should still be robust if old runs do not contain the new columns: if the new diagnostics are missing, print a clear message instead of failing.

---

## Acceptance criteria

1. Rescue background `k0` and `x0` are estimated only from `x_L, k_L, m_L, x_R, k_R, m_R` using force-balance formulas.
2. `sigma_L` and `sigma_R` are used only to compute `sigma_s` and `kT_eff`, not to estimate `k0` or `x0`.
3. If `s_raw <= 0` or `s_raw >= 1`, the code sets `s = 0.5` exactly.
4. `k_res_raw = kT_eff / sigma_s**2 - k0`.
5. `x_res_raw = x_target + (k0 / k_res) * (x_target - x0)` using signed `k0`.
6. The rescue summary records all listed diagnostics.
7. The notebook can visualize the new mean-only background and rescue design for each rescue window.
8. Existing workflow behavior outside rescue design should remain unchanged.
