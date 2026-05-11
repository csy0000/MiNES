# Codex task: Refactor GT protocol and rescue strategy in `scripts/mines_variance_fusion.py`

You are working on the current MiNES variance-fusion workflow, especially the file:

```text
scripts/mines_variance_fusion.py
```

The current implementation already contains GT bridge protocol logic and rescue logic, but the GT logic should be refactored into reusable functions, and the rescue strategy should be replaced with a simpler GT-based rescue rule.

Do **not** change the broad MiNES workflow, PMF fusion, EQ/NEQ simulation interfaces, or output directory structure unless required for this task.

---

## Goal

Use the same GT strategy for:

1. NEQ bridge protocol generation.
2. EQ rescue-window placement.

The code should infer local endpoint-specific harmonic approximations from two neighboring EQ ensembles, then use those approximations to generate:

- a GT NEQ protocol path from progress `s in [0, 1]`;
- a GT-derived rescue EQ window targeting a desired sampled mean `ms`.

---

# 1. Add EQ summary helper

Each EQ ensemble should be represented as:

```python
EQ = [x_umb, k_umb, m_sample, sigma_sample]
```

where:

- `x_umb`: umbrella center of the EQ window;
- `k_umb`: umbrella stiffness of the EQ window;
- `m_sample`: mean of the EQ tail samples;
- `sigma_sample`: standard deviation of the EQ tail samples.

Add a helper:

```python
def eq_gt_tuple(window: EnsembleWindow) -> tuple[float, float, float, float]:
    """
    Return [x_umb, k_umb, m_sample, sigma_sample] for a window.
    Use center_x and k for the umbrella settings, and tail samples for mean/sigma.
    """
```

This should use the existing tail-sample logic, e.g. `window_tail_mean_sigma(window)`.

---

# 2. Add `get_k0_x0_harmonic_fromEQ`

Add:

```python
def get_k0_x0_harmonic_fromEQ(
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
) -> tuple[float, float, float, float, dict[str, Any]]:
    """
    Infer endpoint-specific local latent harmonic potentials from two EQ ensembles.

    EQ_L and EQ_R are:
        [x_umb, k_umb, m_sample, sigma_sample]

    Return:
        x0_L, k0_L, x0_R, k0_R, metadata

    These describe endpoint-local latent harmonic potentials:
        U0_L(x) = 0.5 * k0_L * (x - x0_L)^2
        U0_R(x) = 0.5 * k0_R * (x - x0_R)^2
    """
```

For each endpoint:

```python
K_L = 1.0 / sigma_L**2
k0_L = K_L - k_L
x0_L = (K_L * m_L - k_L * x_L) / k0_L

K_R = 1.0 / sigma_R**2
k0_R = K_R - k_R
x0_R = (K_R * m_R - k_R * x_R) / k0_R
```

Important:

- Do **not** clamp `k0_L` or `k0_R`.
- Negative `k0` is meaningful and should be preserved.
- `k_bound` is **not** an argument to this function.
- The umbrella stiffnesses `k_L` and `k_R` are already bounded by the iterative protocol.

Numerical guard:

```python
eps = 1.0e-12
if abs(k0_L) < eps:
    x0_L = m_L
    mark metadata fallback for left
if abs(k0_R) < eps:
    x0_R = m_R
    mark metadata fallback for right
```

The metadata should include at least:

```text
x_L, k_L, m_L, sigma_L, K_L, k0_L, x0_L
x_R, k_R, m_R, sigma_R, K_R, k0_R, x0_R
left_x0_fallback
right_x0_fallback
```

---

# 3. Add midpoint fallback helper

Add a helper used by rescue-window design:

```python
def midpoint_xs_ks_from_EQ(
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
    k_bound: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """
    Midpoint fallback method for one discrete EQ window.
    """
```

Use:

```python
x_mid = 0.5 * (x_L + x_R)
k_mid_raw = (0.5 * (math.sqrt(k_L) + math.sqrt(k_R))) ** 2
k_mid = clip(k_mid_raw, k_min, k_max)
```

Return metadata including:

```text
x_mid
k_mid_raw
k_mid
method = "midpoint_fallback"
```

---

# 4. Add `get_xs_ks_from_s`

Add:

```python
def get_xs_ks_from_s(
    x0_L: float,
    k0_L: float,
    x0_R: float,
    k0_R: float,
    s: float,
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
    k_bound: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """
    Generate GT umbrella parameters for NEQ protocol progress s in [0, 1].

    This function is for continuous NEQ protocol generation.
    It should clip xs into [x_L, x_R] and clip ks into k_bound.
    It should not use midpoint fallback for out-of-range xs.
    """
```

Use the usual convention:

```python
s = 0.0 -> EQ_L
s = 1.0 -> EQ_R
```

For each `s`:

```python
x0_s = (1.0 - s) * x0_L + s * x0_R
k0_s = (1.0 - s) * k0_L + s * k0_R

m_s = (1.0 - s) * m_L + s * m_R
sigma_s = (1.0 - s) * sigma_L + s * sigma_R

K_s = 1.0 / sigma_s**2
k_s_raw = K_s - k0_s
k_s = clip(k_s_raw, k_min, k_max)

x_s_raw = ((k0_s + k_s) * m_s - k0_s * x0_s) / k_s
x_s = clip(x_s_raw, min(x_L, x_R), max(x_L, x_R))
```

This formulation should automatically reproduce the boundary condition at `s=0` and `s=1`, assuming endpoint stiffnesses are within `k_bound`.

For numerical robustness, it is acceptable to explicitly preserve endpoints when `s <= 0.0` or `s >= 1.0`, but the diagnostics should still show that the GT formula is endpoint-consistent.

Metadata should include:

```text
method = "GT"
s
x0_s
k0_s
m_target
sigma_target
K_target
k_raw
k
k_clipped
x_raw
x
x_clipped
```

---

# 5. Add `get_xs_ks_from_ms`

Add:

```python
def get_xs_ks_from_ms(
    x0_L: float,
    k0_L: float,
    x0_R: float,
    k0_R: float,
    ms: float,
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
    k_bound: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """
    Generate one rescue EQ umbrella whose sampled mean should target ms.

    This function is for discrete EQ rescue-window placement.

    If s_eff hits 0 or 1, fall back to midpoint method with s=0.5.
    If the derived x_s_raw is outside [x_L, x_R], fall back to midpoint method.
    """
```

Compute:

```python
s_eff = (ms - m_L) / (m_R - m_L)
```

The current algorithm should not permit `m_L == m_R`. If it happens anyway, raise a clear `RuntimeError`.

Rules:

```python
if s_eff <= 0.0 or s_eff >= 1.0:
    use midpoint fallback
```

Otherwise:

```python
x0_s = (1.0 - s_eff) * x0_L + s_eff * x0_R
k0_s = (1.0 - s_eff) * k0_L + s_eff * k0_R

sigma_s = (1.0 - s_eff) * sigma_L + s_eff * sigma_R
K_s = 1.0 / sigma_s**2

k_s_raw = K_s - k0_s
k_s = clip(k_s_raw, k_min, k_max)

x_s_raw = ((k0_s + k_s) * ms - k0_s * x0_s) / k_s
```

If:

```python
x_s_raw < min(x_L, x_R) or x_s_raw > max(x_L, x_R)
```

then use midpoint fallback.

Do **not** clip `x_s_raw` for rescue. Either accept it if it is inside the interval, or use midpoint fallback.

Metadata should include:

```text
method = "GT_ms" or "midpoint_fallback"
ms
s_eff
x0_s
k0_s
sigma_target
K_target
k_raw
k
k_clipped
x_raw
x
used_midpoint_fallback
fallback_reason
```

Fallback reasons should distinguish:

```text
"s_eff_endpoint"
"x_raw_out_of_bounds"
```

---

# 6. Refactor `build_gt_bridge_protocol`

Replace the current inline GT calculations inside `build_gt_bridge_protocol` with the new helper functions.

The current implementation computes local GT quantities directly inside `build_gt_bridge_protocol`. It should instead do:

```python
EQ_L = eq_gt_tuple(left_window)
EQ_R = eq_gt_tuple(right_window)

x0_L, k0_L, x0_R, k0_R, harmonic_meta = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)

for s in s_values:
    x_s, k_s, gt_meta = get_xs_ks_from_s(
        x0_L, k0_L, x0_R, k0_R,
        s,
        EQ_L,
        EQ_R,
        (k_min, k_max),
    )
```

Preserve all existing protocol output files:

```text
forward_path.csv
reverse_path.csv
forward_protocol_diagnostics.csv
reverse_protocol_diagnostics.csv
protocol_summary.json
```

Diagnostics rows should include the new metadata fields:

```text
x0_L, k0_L, x0_R, k0_R
x0_s, k0_s
m_target, sigma_target, K_target
x_raw, x, x_clipped
k_raw, k, k_clipped
```

For reverse protocols, call the same machinery with `left_window=boundary_right` and `right_window=boundary_left`, as the current code already does.

---

# 7. Replace rescue strategy with GT rescue

Discard the current rescue-window design strategy based on:

- uncovered intervals;
- high-variance bins;
- matched child designs;
- retry-scaled rescue stiffness;
- signed center shifts;
- `s_rescue`;
- `rescue_center_f_*`.

The new rescue stage should use this rule:

```text
Choose the first non-overlapping neighboring pair in the chain that still needs an EQ rescue window.
```

For that neighboring pair:

1. Build `EQ_L` and `EQ_R` from the two concrete boundary windows.
2. Infer endpoint latent harmonics:

```python
x0_L, k0_L, x0_R, k0_R, harmonic_meta = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)
```

3. Evaluate the latent GT estimate at `s=0.5`:

```python
x0_half = 0.5 * (x0_L + x0_R)
k0_half = 0.5 * (k0_L + k0_R)
```

4. Choose target sampled mean:

```python
if k0_half < 0:
    ms = x0_half
    rescue_target_rule = "negative_k0_half_target_x0_half"
else:
    ms = 0.5 * (m_L + m_R)
    rescue_target_rule = "positive_k0_half_target_mean_midpoint"
```

5. Generate rescue window:

```python
rescue_center_x, rescue_k, rescue_gt_meta = get_xs_ks_from_ms(
    x0_L=x0_L,
    k0_L=k0_L,
    x0_R=x0_R,
    k0_R=k0_R,
    ms=ms,
    EQ_L=EQ_L,
    EQ_R=EQ_R,
    k_bound=(args.k_min, args.k_max),
)
```

6. Run one EQ rescue window with:

```python
center_x = rescue_center_x
k = rescue_k
side = "rescue"
```

7. Reconstruct chain, clusters, segments, patches, and global PMF as before.

---

# 8. Define “still needs an EQ rescue window”

Implement a simple, auditable rule.

A neighboring pair still needs an EQ rescue window if:

1. It is a non-overlapping neighboring pair after clustering.
2. There is no existing rescue window whose center lies strictly between the two boundary umbrella centers.

Use umbrella centers for this decision:

```python
lo = min(left_boundary.center_x, right_boundary.center_x)
hi = max(left_boundary.center_x, right_boundary.center_x)

has_existing_rescue = any(
    w.side == "rescue" and lo < w.center_x < hi
    for w in windows
)
```

Pick the first such pair in left-to-right chain order.

If no such pair exists, stop rescue with:

```text
stop_reason = "no_nonoverlapping_pair_needs_rescue"
```

---

# 9. Refactor inter-cluster NEQ boundary selection

The current logic tends to use geometric cluster boundaries. Replace or augment this with a “best connected boundary window” rule.

For two neighboring clusters:

```python
left_cluster
right_cluster
```

Let:

```python
right_boundary = right_cluster.windows[0]
```

Search for existing NEQ segments connecting any window in `left_cluster` to `right_boundary`:

```python
connected_left_candidates = [
    w for w in left_cluster.windows
    if (w.name, right_boundary.name) in segment_store
]
```

If candidates exist:

```python
left_boundary = min(connected_left_candidates, key=lambda w: w.center_x)
boundary_pair_reason = "existing_connected_segment_to_right_left_boundary_most_left"
```

Here “most left” means smallest `center_x`.

If no candidates exist:

```python
left_boundary = left_cluster.windows[-1]
boundary_pair_reason = "nearest_cluster_boundary_fallback"
```

Use this concrete pair for:

- finding/reusing the NEQ segment;
- running new NEQ if needed;
- estimating the NEQ MTS PMF and variance;
- diagnostics.

Add a helper such as:

```python
def choose_connected_boundary_pair(
    left_cluster: EQCluster,
    right_cluster: EQCluster,
    segment_store: dict[tuple[str, str], NEQSegment],
) -> tuple[EnsembleWindow, EnsembleWindow, dict[str, Any]]:
    ...
```

Metadata should include:

```text
chosen_left_window
chosen_right_window
boundary_pair_reason
connected_left_candidate_names
```

---

# 10. Coverage definition

Coverage should be derived from the actual data used for the patch.

For EQ cluster patches:

```python
coverage_EQ = finite_pmf & finite_variance
coverage_EQ &= coverage_mask_from_samples(
    all_tail_samples_from_cluster_windows,
    grid,
)
```

For NEQ MTS patches:

```python
coverage_NEQ = finite_pmf & finite_variance
```

Preferably also restrict NEQ coverage to bins visited by the NEQ trajectories:

```python
all_neq_x = all x values from forward and reverse trajectories
coverage_NEQ &= coverage_mask_from_samples(all_neq_x, grid)
```

This ensures that PMF/variance support reflects sampled data, not abstract cluster bounds.

---

# 11. Rescue outputs and diagnostics

Keep `rescue_summary.csv`, but update the columns to reflect the new GT rescue logic.

At minimum, include:

```text
round
left_boundary
right_boundary
x_L
k_L
m_L
sigma_L
x_R
k_R
m_R
sigma_R
x0_L
k0_L
x0_R
k0_R
x0_half
k0_half
ms
rescue_target_rule
rescue_center_x
rescue_k
method
used_midpoint_fallback
fallback_reason
s_eff
x_raw
k_raw
x_clipped
k_clipped
added_window
used_steps
```

Also write one JSON file per rescue round:

```text
rescue/rescue_round_<NN>/rescue_decision.json
```

Include the full harmonic metadata and rescue GT metadata.

---

# 12. Remove or deprecate obsolete rescue arguments

The following command-line arguments are no longer used by the new rescue design:

```text
--s-rescue
--rescue-center-f-slope
--rescue-center-f-start
--rescue-center-f-min
--rescue-center-f-max
--k-rescue
```

Either:

1. remove them from `parse_args`, or
2. leave them accepted for backward CLI compatibility but mark them unused in metadata.

If keeping them, do not let them affect the new rescue window.

---

# 13. Summary metadata

Update `mines_variance_fusion_summary.json`.

Replace old rescue metadata such as:

```text
rescue_priority = uncovered_interval_before_finite_variance
rescue_k_design = matched_child_center_with_retry_scaled_k_clamped_to_analysis_bounds
rescue_center_rule = signed_f_shift_after_kmax
```

with:

```text
rescue_strategy = first_nonoverlapping_pair_gt_rescue
rescue_target_rule = negative_k0_half_targets_x0_half_else_mean_midpoint
rescue_gt_out_of_bounds_rule = midpoint_fallback
neq_gt_strategy = endpoint_local_harmonic_interpolation
intercluster_boundary_rule = existing_connected_segment_to_right_left_boundary_most_left_else_nearest_boundary
coverage_rule = actual_patch_data_support
```

---

# 14. Acceptance criteria

The implementation is correct if:

1. `get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)` exists and returns `x0_L, k0_L, x0_R, k0_R, metadata`.
2. `get_xs_ks_from_s(...)` exists and is used by GT NEQ protocol generation.
3. `get_xs_ks_from_s(...)`:
   - interpolates endpoint latent harmonics;
   - clips `ks` into `[k_min, k_max]`;
   - clips `xs` into `[x_L, x_R]`;
   - does not use midpoint fallback.
4. `get_xs_ks_from_s(...)` gives endpoint-consistent protocol values at `s=0` and `s=1`.
5. `get_xs_ks_from_ms(...)` exists and is used for EQ rescue-window design.
6. `get_xs_ks_from_ms(...)`:
   - computes `s_eff` from `ms`;
   - falls back to midpoint if `s_eff <= 0` or `s_eff >= 1`;
   - falls back to midpoint if `x_raw` is outside `[x_L, x_R]`;
   - does not clip `x_raw` for rescue.
7. Rescue no longer uses uncovered-bin or high-variance-bin target selection.
8. Rescue chooses the first non-overlapping neighboring pair that still needs an EQ rescue window.
9. Rescue target mean is:
   - `ms = x0_half` if `k0_half < 0`;
   - otherwise `ms = 0.5 * (m_L + m_R)`.
10. Inter-cluster NEQ patch selection uses the most-left existing connected window when available.
11. “Most-left” means smallest `center_x`.
12. EQ coverage uses actual EQ tail samples.
13. NEQ coverage uses finite PMF/variance and preferably bins visited by NEQ trajectories.
14. Protocol diagnostics, rescue summaries, segment summaries, and global summaries include the new GT metadata.
15. Existing PMF fusion behavior remains inverse-variance weighted global fusion with fitted patch offsets.
16. Existing EQ MBAR and NEQ MTS reconstruction behavior is preserved except for the boundary-pair selection and coverage clarification.

---

# Implementation note

Prioritize clarity and diagnostics over cleverness. The goal is to make the GT path and rescue-window construction easy to inspect from CSV/JSON outputs.
