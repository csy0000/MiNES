# ClaudeCode Task: Improve MiNES rescue sampling and visualization after completed simulations

## Context

The latest MiNES runs have already been completed using the current `scripts/mines_variance_fusion.py` and `notebooks/mines_variance_fusion_visualization.ipynb`. Do **not** assume the user wants to rerun all simulations immediately. Your task is to improve the code so that:

1. Future rescue windows are placed in physically meaningful bins.
2. Coverage diagnostics reflect the true PMF support over the analysis interval.
3. Existing completed simulations can be inspected more clearly from disk outputs.
4. The notebook makes it obvious why each rescue window was chosen and whether it improved the target bin.

The current run was done with:

```bash
--pmf-method hybrid
```

The attached results show that rescue windows were sometimes placed far away from the high-variance bins. The convergence plot also reports coverage below 100%, even though the PMF appears finite everywhere between `x=-10` and `x=10`.

---

## Files to modify

Modify these files:

```text
scripts/mines_variance_fusion.py
notebooks/mines_variance_fusion_visualization.ipynb
```

Do not modify unrelated AUS or legacy workflow code.

---

# Part 1: Fix and harden rescue target selection

## 1.1 Separate true uncovered bins from numerical edge artifacts

Current behavior can prioritize an `uncovered_interval` even when the PMF is effectively finite everywhere in the analysis interval. This causes rescue to chase small numerical/endpoint artifacts instead of the high-variance barrier region.

Update `choose_uncovered_rescue_target(...)` and/or `choose_rescue_target_priority(...)` as follows:

### Required behavior

Define the analysis mask with a tolerance:

```python
half_dx = 0.5 * abs(grid[1] - grid[0])
analysis_mask = (
    np.isfinite(grid)
    & (grid >= analysis_xmin - half_dx)
    & (grid <= analysis_xmax + half_dx)
)
```

For coverage, treat a bin as covered if either:

```python
np.isfinite(global_pmf[idx])
```

or, if `n_covering_patches` is available from the fit details/global PMF CSV:

```python
n_covering_patches[idx] > 0
```

If the only uncovered bins are isolated endpoint/tolerance artifacts, do **not** let them dominate rescue selection.

Add arguments or internal constants:

```python
min_uncovered_rescue_bins = 2
min_uncovered_rescue_width = 0.75 * bin_width
```

Only prioritize uncovered intervals if:

```python
uncovered_n_bins >= min_uncovered_rescue_bins
```

or

```python
uncovered_width >= min_uncovered_rescue_width
```

Otherwise, fall through to finite-variance targeting.

### Required diagnostics

When an uncovered interval is ignored as a numerical artifact, write this information to the rescue decision metadata:

```text
ignored_uncovered_start_x
ignored_uncovered_end_x
ignored_uncovered_n_bins
ignored_uncovered_width
ignored_uncovered_reason = "below_min_uncovered_rescue_size"
```

---

## 1.2 Make max-variance rescue the default after full PMF support is reached

If the PMF is finite over all analysis bins, rescue must target:

```python
x* = argmax_x global_variance[x]
```

under the same analysis mask.

Required selection rule:

1. True uncovered interval with meaningful size.
2. Failed/skipped gap with meaningful missing support.
3. Finite global PMF bin with maximum global variance.

Do **not** target old child-design centers unless they coincide with the selected bin.

---

## 1.3 Rescue center must be the selected target bin

The rescue EQ window should sample the selected bin directly. Replace any logic that moves the rescue center toward a previous child-design center.

Use:

```python
target_bin_index = int(np.argmin(np.abs(grid - x_rescue_target)))
target_bin_x = float(grid[target_bin_index])
rescue_center_x_raw = target_bin_x
rescue_center_x = clamp_to_bounds(target_bin_x, analysis_xmin, analysis_xmax)
```

The following should be true except for boundary clipping:

```python
rescue_center_x == target_bin_x
```

Keep the previous child-design matching only as **diagnostic metadata**, not as a placement rule.

Required metadata:

```text
matched_child_name
matched_child_center_x
matched_child_target_x
matched_target_distance
matched_child_used_for_center = 0
```

---

## 1.4 Improve rescue force constant choice

Current rescue placement can repeatedly use bad windows. Implement a target-bin-centric stiffness policy.

Required behavior:

```python
sigma_target = max(1.5 * bin_width, 0.20)
k_from_sigma = kT / sigma_target**2
rescue_k_base = max(k_rescue, k_from_sigma)
rescue_k = clamp(rescue_k_base * s_rescue**retry_count, k_min, k_max)
```

Use `ctx["thermal_kT"]` as `kT`.

Write these fields into `rescue_summary.csv`:

```text
sigma_target
k_from_sigma
rescue_k_base
rescue_k
rescue_k_rule = "target_bin_sigma_rule_with_retry_scaling"
```

If `rescue_k == k_max`, keep the center fixed at the selected target bin. Do **not** move the center away from the target because `k` saturated.

---

## 1.5 Add post-EQ target-bin sampling diagnostics

After each rescue EQ simulation, check whether the target bin was actually sampled by the rescue window tail.

Use the rescue tail samples:

```python
x_tail = eq_tail_samples(rescue_window)
```

Compute:

```text
rescue_tail_q05
rescue_tail_q50
rescue_tail_q95
rescue_tail_min
rescue_tail_max
rescue_tail_mean
rescue_tail_std
rescue_tail_contains_target_bin
target_bin_tail_count
target_bin_tail_fraction
```

A tail sample belongs to the target bin if:

```python
abs(x_tail - target_bin_x) <= 0.5 * bin_width
```

Add these columns to `rescue_summary.csv` and `rescue_round_XX/rescue_decision.json`.

This is essential because a rescue window centered at the correct coordinate can still fail if the local force/background potential pulls the sampled distribution away from the target.

---

## 1.6 Add retry logic based on failed target-bin sampling

For future runs, if a previous rescue round targeted the same bin but:

```text
rescue_tail_contains_target_bin == False
```

or

```text
target_bin_tail_fraction < 0.05
```

then the next retry should keep the center at `target_bin_x` and increase `k` by `s_rescue`, up to `k_max`.

Do not shift the center toward old child centers.

---

# Part 2: Improve PMF coverage metrics

## 2.1 Coverage should be based on analysis interval and finite PMF support

Update `compute_pmf_quality_metrics(...)`.

Use tolerance around the bounds:

```python
half_dx = 0.5 * abs(grid[1] - grid[0])
analysis_mask = (
    np.isfinite(grid)
    & (grid >= analysis_xmin - half_dx)
    & (grid <= analysis_xmax + half_dx)
)
```

Define:

```python
covered_mask = analysis_mask & np.isfinite(global_pmf)
```

If `global_pmf.csv` includes `n_covering_patches`, keep `finite global_pmf` as the primary coverage definition for `pmf_quality_vs_steps.csv`.

Add diagnostics:

```text
n_uncovered_bins
first_uncovered_x
last_uncovered_x
uncovered_x_values
```

`uncovered_x_values` can be a semicolon-separated string. Limit it to something reasonable, for example the first 50 values plus an indicator if truncated.

## 2.2 Add max-variance diagnostics

Add these columns to `pmf_quality_vs_steps.csv`:

```text
max_global_variance
x_at_max_global_variance
max_global_std
```

The max should be computed only over:

```python
analysis_mask & np.isfinite(global_pmf) & np.isfinite(global_variance)
```

## 2.3 Write all expected quality columns every time

The current script writes `pmf_quality_vs_steps.csv` more than once. Make sure every write uses the same full `extras` field list, including:

```text
stage
used_steps
used_ksteps
analysis_xmin
analysis_xmax
n_interest_bins
n_covered_bins
n_uncovered_bins
first_uncovered_x
last_uncovered_x
uncovered_x_values
coverage_fraction
coverage_percent
rmse_bestfit
bestfit_offset
n_error_bins
mean_global_variance
median_global_variance
max_global_variance
x_at_max_global_variance
max_global_std
```

---

# Part 3: Improve patch/variance diagnostics for hybrid PMF

The user is running with `--pmf-method hybrid`. For debugging, add per-bin diagnostics showing what patch sources contribute to the global PMF and variance.

## 3.1 Add dominant variance source per bin

When writing `global_pmf.csv`, add columns:

```text
best_variance_patch
best_variance_patch_kind
best_variance_value
n_eq_covering_patches
n_neq_covering_patches
```

For each bin, among all patches covering the bin, find the patch with the smallest variance.

This will make it clear whether the high-variance barrier region is controlled by EQ patches, NEQ MTS patches, or a disconnected/fallback source.

## 3.2 Add patch coverage summary

Write a new file:

```text
patch_bin_contributions.csv
```

Columns:

```text
x
patch_name
patch_kind
covered
local_pmf
variance
aligned_pmf
patch_offset
```

This can be generated after `fit_global_pmf_from_patches(...)` using `fit_details["patch_offsets"]`.

---

# Part 4: Improve notebook visualization using completed simulations

The notebook must work on already completed simulations. It should not require rerunning MiNES.

## 4.1 Robust file discovery

The notebook should load from:

```python
run_root = Path(system_root) / "MINES" / label / "raw" / f"seed_{seed}"
```

Load if present:

```text
global_pmf.csv
pmf_quality_vs_steps.csv
windows.csv
clusters.csv
segments.csv
patches.csv
rescue_summary.csv
generation_summary.csv
global_fit_summary.json
snapshots/growth_reconstruct/global_pmf.csv
snapshots/rescue_round_*/global_pmf.csv
rescue/rescue_round_*/global_pmf.csv
patch_bin_contributions.csv
```

The notebook should print a clear warning for missing optional files, not fail.

---

## 4.2 Plot PMF snapshots after coverage reaches 1.0

Add a plot titled:

```text
PMF snapshots after full coverage
```

Use `pmf_quality_vs_steps.csv` to identify rows where:

```python
coverage_fraction >= 0.999
```

For each such stage, locate the corresponding snapshot:

```text
stage == "growth_reconstruct" -> snapshots/growth_reconstruct/global_pmf.csv
stage == "rescue_round_01" -> snapshots/rescue_round_01/global_pmf.csv
stage == "rescue_round_02" -> snapshots/rescue_round_02/global_pmf.csv
...
```

Plot:

```text
x vs global_pmf
```

Overlay analytic PMF if available.

Use a consistent alignment before comparing snapshots:

```python
shift each PMF so its finite minimum over [analysis_xmin, analysis_xmax] is zero
```

---

## 4.3 Plot max variance as a function of iteration

Add a plot titled:

```text
Max global variance vs iteration
```

Use `pmf_quality_vs_steps.csv`.

X-axis options:

1. stage index, categorical or integer order; and/or
2. `used_ksteps`

Y-axis:

```text
max_global_variance
```

Also plot:

```text
x_at_max_global_variance
```

as a second figure or table, so the user can see whether the max-variance location moves after rescue.

---

## 4.4 Improve rescue summary table

Add a displayed DataFrame for rescue rows with these columns if present:

```text
round
target_priority
target_reason
x_rescue_target
target_bin_x
rescue_center_x
added_center_x
rescue_k
variance_before_eq
variance_after_eq
variance_delta_after_minus_before
rescue_tail_q05
rescue_tail_q50
rescue_tail_q95
rescue_tail_contains_target_bin
target_bin_tail_fraction
rescue_design_rationale
```

Add a derived column:

```python
variance_improved = variance_after_eq < variance_before_eq
```

Highlight or sort failed rescue rows where:

```python
variance_improved == False
```

or

```python
rescue_tail_contains_target_bin == False
```

---

## 4.5 Add rescue target overlay to PMF and variance plots

On the global PMF and global variance plots, draw vertical lines for:

```text
target_bin_x
rescue_center_x
```

Use different linestyles and include labels.

This makes wrong placement immediately visible.

---

## 4.6 Add EQ distribution plot with rescue target bins

Extend the EQ window distribution plot.

For every rescue window, mark:

```text
target_bin_x
rescue_center_x
rescue_tail_q05/q95 if available
```

This should show whether each rescue window’s EQ distribution actually overlaps the targeted bin.

---

## 4.7 Add patch coverage and patch source diagnostics

If `patch_bin_contributions.csv` exists, add:

1. Heatmap of patch coverage:
   - x-axis: grid x
   - y-axis: patch name
   - value: covered

2. Heatmap or line plot of `log10(variance)` by patch:
   - x-axis: grid x
   - y-axis: patch name
   - value: `log10(variance)`

3. Plot the dominant variance source from `global_pmf.csv`:
   - x-axis: x
   - y-axis/category/color: `best_variance_patch_kind`

This will show whether the hybrid estimator’s uncertainty is dominated by EQ or NEQ components.

---

# Part 5: Acceptance tests

After modifications, run static checks and a post-hoc notebook/data loading check.

## 5.1 Python syntax check

Run:

```bash
python -m py_compile scripts/mines_variance_fusion.py
```

## 5.2 Post-hoc notebook data check

Without rerunning simulations, run a short Python script that loads the latest run root and verifies that the notebook-required files can be read.

Example:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

system_root = Path("results/doublewell_1d")
label = "mines_variance_fusion"
seed = 10101
run_root = system_root / "MINES" / label / "raw" / f"seed_{seed}"
print("run_root", run_root)
for name in ["global_pmf.csv", "pmf_quality_vs_steps.csv", "windows.csv", "rescue_summary.csv"]:
    path = run_root / name
    print(name, path.exists())
    if path.exists():
        df = pd.read_csv(path)
        print("  shape", df.shape)
        print("  columns", list(df.columns)[:20])
PY
```

If the actual run root differs, update the `system_root`, `label`, and `seed` variables in the notebook only; do not hard-code the user’s local path into library code.

## 5.3 Expected behavior for future runs

For future simulations, `rescue_summary.csv` should satisfy:

```text
abs(rescue_center_x - target_bin_x) <= 0.5 * bin_width
```

and should report whether the target bin was sampled:

```text
rescue_tail_contains_target_bin
```

For a run whose PMF is finite everywhere in `[analysis_xmin, analysis_xmax]`, `coverage_percent` should be exactly or numerically close to 100%.

---

# Important notes

1. Do not replace global inverse-variance fusion with hard stitching.
2. Do not use old child-design centers as rescue centers.
3. Do not let one-bin endpoint artifacts override max-variance rescue.
4. Keep all rescue choices auditable in `rescue_summary.csv` and `rescue_round_XX/rescue_decision.json`.
5. The notebook must remain useful for simulations that have already been completed.
