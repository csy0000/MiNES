# MiNES Protocol Update — PMF Method Selection and CFT/BAR Growth Stop

Date: 2026-05-07

This document specifies changes to the current `scripts/mines_variance_fusion.py` workflow. The goal is to simplify chain growth stopping, add selectable PMF evaluation modes, and remove unnecessary grid snapping/barrier-crossing logic for offspring umbrella centers.

## Scope

Modify the MiNES-only workflow, primarily:

```text
scripts/mines_variance_fusion.py
```

Update helper scripts or notebooks only if needed to expose or visualize the new options.

Do not reintroduce AUS logic. Keep this as a clean MiNES implementation.

---

## 1. Add PMF evaluation mode flag

Add a command-line argument:

```bash
--pmf-method {neq,eq,hybrid}
```

Default:

```bash
--pmf-method neq
```

### Required behavior

The flag controls which PMF patches are used to construct the final global PMF. It must also control the variance-estimation workflow as described below.

| mode | PMF construction | Variance estimation | Notes |
|---|---|---|---|
| `neq` | Use only NEQ/MTS patches | Use NEQ/MTS bootstrap variance | New default |
| `hybrid` | Use both EQ/MBAR and NEQ/MTS patches | Use current hybrid variance behavior | This is the current behavior |
| `eq` | Use EQ/MBAR PMF patches for PMF values | Still use the EQ-NEQ method to evaluate variance | NEQ must still run |

### Important rule for `--pmf-method eq`

Even when `--pmf-method eq` is selected, NEQ simulations must still run because they are used for:

1. Offspring proposal.
2. CFT/BAR growth stopping.
3. EQ-NEQ variance evaluation.

For now, enforce:

```python
if args.t_neq <= 0:
    raise ValueError("--t-neq must be > 0 for this protocol.")
```

This validation should apply to all PMF modes.

### Suggested implementation structure

After all candidate patches are built, choose which patches enter the global fit according to `args.pmf_method`.

For example:

```python
if args.pmf_method == "neq":
    patches_for_global = [p for p in patches if p.kind == "NEQ_MTS"]
elif args.pmf_method == "hybrid":
    patches_for_global = list(patches)
elif args.pmf_method == "eq":
    patches_for_global = build_eq_pmf_with_eq_neq_variance(...)
else:
    raise ValueError(f"Unknown pmf method: {args.pmf_method}")
```

The exact implementation for `eq` may differ depending on existing data structures. The key requirement is:

- PMF values come from EQ/MBAR patches.
- Variance comes from the EQ-NEQ variance evaluation path, not pure EQ-only bootstrap variance.

If the current code does not already have a clean EQ-NEQ variance object, implement this explicitly and document the generated variance source in patch metadata.

Add metadata fields to outputs:

```text
pmf_method
pmf_patch_selection_rule
variance_source
```

The global fit summary should record the chosen PMF method.

---

## 2. Enforce positive NEQ duration

Add argument validation after parsing and quick-test overrides:

```python
if int(args.t_neq) <= 0:
    raise ValueError("--t-neq must be > 0 because NEQ is required for offspring proposal, CFT/BAR stopping, and variance estimation.")
```

This is required even for `--pmf-method eq`.

---

## 3. Remove grid snapping for umbrella centers

Currently, child and rescue centers are snapped to the nearest analysis grid point with calls like:

```python
nearest_grid_value(..., grid)
```

This should be removed for window centers.

### New rule

Window centers do not need to lie on PMF grid points.

For offspring windows, use:

```text
center_raw = target_x + x_leap    # left child
center_raw = target_x - x_leap    # right child
center_x = center_raw clipped to the progress-feasible interval
```

Do not snap `center_x` to a grid point.

The grid is still used for:

- PMF binning.
- Coverage masks.
- Analysis bounds.
- PMF output tables.
- Variance output tables.
- Plotting.

But umbrella centers themselves can be continuous floating-point values.

### Functions to update

Update at least these functions if present:

```python
finalize_child_proposal(...)
clamp_to_bounds_and_grid(...)
design_rescue_window(...)
```

Recommended rename:

```python
clamp_to_bounds_and_grid -> clamp_to_bounds
```

or keep the old function name but remove grid snapping internally and update comments/metadata to avoid confusion.

### Output metadata

Where possible, replace metadata such as:

```text
center_snapped_to_grid
rescue_center_clamped_to_bounds
```

with clear continuous-center diagnostics:

```text
center_raw
center_x
center_clipped_to_progress
rescue_center_x_raw
rescue_center_x
rescue_center_clamped_to_bounds
```

---

## 4. Remove explicit barrier-crossing k override

The current child proposal contains a special branch similar to:

```python
if barrier_crossing:
    k_value = k_min
    k_rule = "barrier_crossing_k_min"
else:
    k_value = clip(raw_k, k_min, k_max)
    k_rule = "force_matching_clipped"
```

Remove or disable this special override.

### Reason

The new GT protocol is slope-aware, so the barrier-crossing diagnosis is assumed to be handled implicitly by GT. The explicit `barrier_crossing_k_min` rule is no longer needed and may interfere with adaptive placement.

### New k behavior

Always compute the child spring constant through the regular rule:

```python
raw_k = matched_force / gap
k = min(max(raw_k, k_min), k_max)
k_rule = "force_matching_clipped"
```

Keep diagnostics if useful, but do not use them to override `k`:

```text
barrier_crossing_diagnostic
barrier_crossing_displacement
barrier_crossing_tol
```

If retaining the diagnostic, make clear in metadata that it is informational only:

```text
barrier_crossing_action = "none_gt_slope_aware"
```

---

## 5. Replace chain-growth stop rule with CFT/BAR threshold

The current stop rule uses EoP crossing and/or window-center crossing. Replace this with the first generation where the CFT/BAR free-energy result satisfies a threshold condition.

### Add argument

```bash
--cft-ddf-threshold 1.0
```

Default:

```python
1.0
```

### Meaning

During each growth generation, after bidirectional NEQ is run for the active segment, solve Crooks Fluctuation Theorem / BAR using the NEQ work distributions.

Use the `delta_f` value returned by the BAR/CFT solver, not its uncertainty.

The growth stop condition is:

```python
if cft_solved and cft_delta_f < args.cft_ddf_threshold:
    stop_reason = "cft_delta_f_below_threshold"
    stop_growth = True
```

Do not use absolute value. `delta_f` is assumed to be positive.

### Important distinction

Use:

```text
cft_delta_f
```

not:

```text
cft_delta_f_unc
```

`cft_delta_f_unc` is the uncertainty estimate returned by the BAR/CFT solve. It is not the stopping metric requested here.

### Suggested helper

If the current code computes CFT only inside `build_neq_mts_patch(...)`, refactor or add a helper so the growth loop can evaluate CFT immediately after running the active NEQ segment:

```python
def compute_segment_cft_summary(segment: NEQSegment, ctx: dict[str, Any]) -> dict[str, Any]:
    forward_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    reverse_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    _x_forward, work_forward = trajectory_frames_to_arrays(forward_frames)
    _x_reverse, work_reverse = trajectory_frames_to_arrays(reverse_frames)
    cft = solve_segment_cft_delta_f_once(
        work_forward,
        work_reverse,
        kT=float(ctx["thermal_kT"]),
    )
    return {
        "cft_solved_once": bool(cft.get("cft_solved", False)),
        "cft_delta_f": cft.get("delta_f", None),
        "cft_delta_f_unc": cft.get("delta_f_unc", None),
        "cft_method": cft.get("method", "BAR"),
        "cft_reason": cft.get("reason", ""),
    }
```

Reuse this summary inside `build_neq_mts_patch(...)` rather than solving CFT twice.

### Growth loop behavior

New sequence per generation should be:

1. Check budget for growth.
2. Run bidirectional NEQ for the active frontier segment.
3. Compute CFT/BAR summary from NEQ work distributions.
4. Log the CFT/BAR result.
5. If `cft_delta_f < cft_ddf_threshold`, stop growth immediately.
6. Otherwise, propose left/right offspring windows and continue.

This means the CFT stop condition is evaluated before adding the next child windows for that generation.

### Remove/suppress old stop criteria

Remove or disable growth stopping based on:

```text
frontiers_crossed
frontiers_overlap
window center crossing
EoP crossing
```

EoP diagnostics may still be computed for reporting/connectivity, but they should not be the primary chain-growth stop rule.

---

## 6. Output and audit requirements

Update outputs to make the new behavior auditable.

### `run_request.json`

Add:

```json
{
  "pmf_method": "neq",
  "cft_ddf_threshold": 1.0,
  "t_neq_validation": "required_positive",
  "window_center_rule": "continuous_clipped_not_grid_snapped",
  "barrier_crossing_rule": "disabled_gt_slope_aware"
}
```

### `generation_summary.csv`

Add columns:

```text
pmf_method
cft_solved_once
cft_delta_f
cft_delta_f_unc
cft_method
cft_reason
cft_ddf_threshold
stop_by_cft
stop_reason
left_center_raw
left_center_x
right_center_raw
right_center_x
left_k_rule
right_k_rule
```

### `frontier_jsd.csv` or replacement growth status file

If `frontier_jsd.csv` is retained, do not imply JS is the growth stop criterion. Prefer adding a new file:

```text
growth_stop_summary.csv
```

Suggested columns:

```text
generation
stage
active_segment
cft_solved_once
cft_delta_f
cft_delta_f_unc
cft_method
cft_reason
cft_ddf_threshold
stop_by_cft
stop_reason
used_steps
```

### `global_fit_summary.json`

Add:

```json
{
  "pmf_method": "neq",
  "patch_selection_rule": "only_NEQ_MTS",
  "variance_source": "NEQ_MTS_bootstrap"
}
```

For `hybrid`:

```json
{
  "patch_selection_rule": "EQ_MBAR_plus_NEQ_MTS",
  "variance_source": "hybrid_patch_variance"
}
```

For `eq`:

```json
{
  "patch_selection_rule": "EQ_MBAR_pmf_with_EQ_NEQ_variance",
  "variance_source": "EQ_NEQ_variance"
}
```

---

## 7. Acceptance criteria

The implementation is correct if all of the following are true:

1. `scripts/mines_variance_fusion.py` accepts `--pmf-method {neq,eq,hybrid}`.
2. The default PMF method is `neq`.
3. `--pmf-method hybrid` reproduces the current EQ+NEQ patch fusion behavior.
4. `--pmf-method neq` uses only NEQ/MTS patches in the final global PMF fit.
5. `--pmf-method eq` uses EQ/MBAR PMF values but still uses the EQ-NEQ method for variance estimation.
6. NEQ simulations still run for `--pmf-method eq`.
7. The script raises an error if `--t-neq <= 0`.
8. Window centers are no longer snapped to PMF grid points.
9. Offspring centers use continuous `center_raw` after clipping to the feasible progress interval.
10. Rescue centers are no longer snapped to grid points.
11. The explicit `barrier_crossing_k_min` k override is removed or disabled.
12. `force_matching_clipped` remains the normal k rule for offspring windows unless replaced by a GT-specific slope-aware k rule.
13. The growth stop criterion is based on `cft_delta_f < cft_ddf_threshold`.
14. The default `--cft-ddf-threshold` is `1.0`.
15. The stop criterion uses `cft_delta_f`, not `cft_delta_f_unc`.
16. The stop criterion does not use `abs(cft_delta_f)` because `delta_f` is assumed positive.
17. EoP crossing and center crossing no longer stop growth, though they may remain as diagnostics.
18. Output files record `pmf_method`, CFT/BAR values, stop reason, and continuous center rules.
19. The code remains MiNES-only and does not add AUS logic.

---

## 8. Notes for implementation

- Keep the code modular. Prefer adding small helpers for PMF patch selection and CFT/BAR summary calculation.
- Avoid duplicating BAR/CFT calculations. If a segment already has a valid `cft_summary`, reuse it.
- Do not hard-stitch PMFs. Global PMF construction should still use inverse-variance weighted patch fusion with fitted additive offsets.
- In `neq` mode, if no NEQ patches are available, fail clearly with a useful error message rather than silently falling back to EQ.
- In `eq` mode, if EQ-NEQ variance cannot be computed for a region, mark the variance as missing or use a documented fallback. Do not silently pretend the variance is pure EQ bootstrap variance unless explicitly requested later.
