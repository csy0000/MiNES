# Task: Fix MiNES rescue targeting and add operation-level timing logs

You are working in the current repository for adaptive nonequilibrium sampling.

Relevant file:

```text
scripts/mines_variance_fusion.py
```

The current protocol has two problems:

1. The rescue window does not prioritize uncovered bins.
2. Rescue rounds are much slower than normal chain-growing rounds, but the code does not yet provide enough timing diagnostics to identify the bottleneck.

Do not rewrite the scientific estimators unless necessary. This task should only change rescue targeting and diagnostics/timing.

---

## Problem 1: Rescue does not prioritize uncovered bins

The current rescue loop appears to choose a rescue position from the first unresolved or non-overlapping cluster pair, then builds a Gaussian-transport midpoint-style rescue window. This can ignore large uncovered regions in the global PMF.

There is already target-selection logic in the file:

```python
choose_uncovered_rescue_target(...)
choose_failed_or_skipped_gap_target(...)
choose_rescue_target_priority(...)
design_rescue_window(...)
```

But the actual rescue loop is not using this logic.

### Required rescue priority

In every rescue round, select the target in this order:

1. **Uncovered bins inside `[analysis_xmin, analysis_xmax]`**
   - Define uncovered bins as:
     ```python
     analysis_mask & ~np.isfinite(global_pmf)
     ```
   - Find contiguous uncovered intervals.
   - Choose the widest uncovered interval.
   - Break ties by larger `n_bins`.
   - Break remaining ties by closeness to the analysis-domain center.
   - Target the midpoint of the chosen interval, snapped to the grid.

2. **Failed or skipped NEQ/MTS gaps**
   - If there are no uncovered bins, prioritize gaps associated with MTS-failed or skipped segments.
   - Only target a failed/skipped gap if the interval still contains uncovered bins.

3. **Finite high-variance bins**
   - If there are no uncovered or failed/skipped uncovered gaps, choose the finite bin with maximum `global_variance`.

The rescue loop must call:

```python
target_info = choose_rescue_target_priority(
    grid=grid,
    global_pmf=global_pmf,
    global_variance=global_variance,
    analysis_xmin=float(analysis_xmin),
    analysis_xmax=float(analysis_xmax),
    skipped_segment_rows=skipped_segment_rows,
    mts_failed_segments=mts_failed_segments(all_segments),
)
```

If `target_info is None`, stop rescue with:

```python
stop_reason = "no_rescue_target_available"
```

---

## Replace rescue placement logic

In the loop:

```python
while rescue_counter < int(args.max_rescue_rounds):
```

remove or demote the current dependency on:

```python
find_first_rescue_pair(...)
```

as the primary rescue selector.

Instead, after choosing `target_info`, call:

```python
rescue_design = design_rescue_window(
    target_info=target_info,
    generations_root=generations_root,
    grid=grid,
    args=args,
    analysis_xmin=float(analysis_xmin),
    analysis_xmax=float(analysis_xmax),
    rescue_rows=rescue_rows,
)
```

Then run the rescue EQ window at:

```python
center_x = rescue_design["rescue_center_x"]
k = rescue_design["rescue_k"]
```

The rescue window should be created like the existing rescue window, but using `rescue_design` instead of GT midpoint placement.

Keep the old GT midpoint logic only as optional fallback/debug information if needed. It should not be the default rescue-placement rule.

---

## Required rescue diagnostics

`rescue_summary.csv` must include these columns:

```text
round
target_priority
target_reason
x_rescue_target
target_variance
uncovered_start_x
uncovered_end_x
uncovered_width
uncovered_n_bins
rescue_center_x_raw
rescue_center_x
rescue_center_clamped_to_bounds
rescue_k
rescue_k_unclipped
rescue_k_saturated
rescue_k_rule
rescue_center_rule
rescue_center_f_raw
rescue_center_f
matched_child_name
matched_child_side
matched_target_x
matched_target_distance
matched_child_center_x
matched_child_k
matched_child_raw_k
matched_child_k_rule
matched_child_target_source
rescue_retry_count
rescue_scale
added_window
added_center_x
added_k
used_steps
```

The corresponding `rescue_round_<n>/rescue_decision.json` must also include:

```python
{
    **target_info,
    **rescue_design,
    "added_window": rescue_name,
    "added_center_x": rescue_window.center_x,
    "added_k": rescue_window.k,
    "used_steps": budget.used_steps,
}
```

---

## Acceptance tests for rescue targeting

Add or update checks so that:

1. If `global_pmf` has any `NaN` bins inside the analysis interval, rescue target priority is:
   ```text
   uncovered_interval
   ```

2. If there are multiple uncovered intervals, the chosen target lies inside the widest interval.

3. If there are no uncovered bins but finite variance exists, rescue target priority is:
   ```text
   finite_variance
   ```

4. `rescue_summary.csv` contains the target-priority and rescue-design columns listed above.

5. The first rescue point in a run with a visibly uncovered central/right gap lands inside that uncovered interval, not simply at the first cluster-pair midpoint.

---

# Problem 2: Add operation-level timing logs

Rescue rounds are much slower than normal growth rounds. Add timing instrumentation so we can tell whether the bottleneck is:

- EQ dynamics
- NEQ dynamics
- patch reconstruction
- bootstrap
- CFT / MTS
- global PMF fusion
- state-table writing
- repeated disk I/O

---

## Add timing context manager

Near the imports, add:

```python
from contextlib import contextmanager
import time
```

Then add this helper:

```python
@contextmanager
def timed_operation(
    timing_rows: list[dict[str, Any]],
    *,
    stage: str,
    operation: str,
    item: str = "",
    metadata: dict[str, Any] | None = None,
):
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    status = "ok"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "error"
        error = repr(exc)
        raise
    finally:
        t1_wall = time.perf_counter()
        t1_cpu = time.process_time()
        row: dict[str, Any] = {
            "stage": str(stage),
            "operation": str(operation),
            "item": str(item),
            "wall_seconds": float(t1_wall - t0_wall),
            "cpu_seconds": float(t1_cpu - t0_cpu),
            "status": status,
            "error": error,
        }
        if metadata:
            row.update(metadata)
        timing_rows.append(row)
```

Initialize near the other top-level row collections in `main()`:

```python
timing_rows: list[dict[str, Any]] = []
```

---

## Add timing summary helper

Add:

```python
def summarize_timing_rows(timing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in timing_rows:
        groups.setdefault(str(row.get("operation", "")), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for operation, rows in sorted(groups.items()):
        wall = [float(row.get("wall_seconds", 0.0) or 0.0) for row in rows]
        cpu = [float(row.get("cpu_seconds", 0.0) or 0.0) for row in rows]
        summary_rows.append(
            {
                "operation": operation,
                "count": int(len(rows)),
                "total_wall_seconds": float(sum(wall)),
                "mean_wall_seconds": float(sum(wall) / len(wall)) if wall else 0.0,
                "max_wall_seconds": float(max(wall)) if wall else 0.0,
                "total_cpu_seconds": float(sum(cpu)),
                "mean_cpu_seconds": float(sum(cpu) / len(cpu)) if cpu else 0.0,
                "max_cpu_seconds": float(max(cpu)) if cpu else 0.0,
                "n_error": int(sum(1 for row in rows if row.get("status") == "error")),
            }
        )
    return summary_rows
```

Add:

```python
def write_timing_outputs(out_root: Path, timing_rows: list[dict[str, Any]]) -> None:
    write_csv(
        out_root / "operation_timing.csv",
        ordered_fieldnames(
            timing_rows,
            extras=[
                "stage",
                "operation",
                "item",
                "wall_seconds",
                "cpu_seconds",
                "status",
                "error",
            ],
        ),
        timing_rows,
    )
    summary_rows = summarize_timing_rows(timing_rows)
    write_csv(
        out_root / "operation_timing_summary.csv",
        ordered_fieldnames(
            summary_rows,
            extras=[
                "operation",
                "count",
                "total_wall_seconds",
                "mean_wall_seconds",
                "max_wall_seconds",
                "total_cpu_seconds",
                "mean_cpu_seconds",
                "max_cpu_seconds",
                "n_error",
            ],
        ),
        summary_rows,
    )
```

Call `write_timing_outputs(out_root, timing_rows)` after every major stage and at the end of `main()`.

Also wrap the outer body of `main()` with a `try/finally` or add a local guarded writer so that `operation_timing.csv` is written even if a later operation fails.

---

## Instrument these operations

Wrap at least the following operations with `timed_operation(...)`:

- `run_eq_window`
- `run_neq_protocol`
- `build_eq_cluster_patch`
- `build_neq_mts_patch`
- `fit_global_pmf_from_patches`
- `reconstruct_chain`
- `choose_rescue_target_priority`
- `design_rescue_window`
- `write_state_tables`
- `build_hs_fallback_patches`

Example for rescue EQ:

```python
with timed_operation(
    timing_rows,
    stage=rescue_stage,
    operation="run_eq_window",
    item=rescue_name,
    metadata={
        "side": "rescue",
        "center_x": rescue_design["rescue_center_x"],
        "k": rescue_design["rescue_k"],
        "n_eq_steps": int(args.n_eq_steps),
    },
):
    rescue_window = run_eq_window(...)
```

Example for target selection:

```python
with timed_operation(
    timing_rows,
    stage=rescue_stage,
    operation="choose_rescue_target_priority",
    item="rescue_target",
    metadata={
        "n_windows": len(windows),
        "n_segments": len(all_segments),
    },
):
    target_info = choose_rescue_target_priority(...)
```

Example for reconstruct-chain:

```python
with timed_operation(
    timing_rows,
    stage=rescue_stage,
    operation="reconstruct_chain",
    item="all_windows",
    metadata={
        "n_windows": len(windows),
        "n_segments_before": len(segment_store),
    },
):
    clusters, segments, patches, global_pmf, global_variance, fit_details, js_rows = reconstruct_chain(...)
```

---

## Required timing metadata

Timing rows should include enough metadata to compare rescue EQ with normal growth EQ:

```text
side
center_x
k
n_eq_steps
n_neq_traj
t_neq
n_windows
n_clusters
n_segments
n_patches
stage_index
rescue_round
```

Include the metadata when it is naturally available. Do not overcomplicate function signatures if the values are not available at a given call site.

---

## Timing output files

The workflow must write:

```text
operation_timing.csv
operation_timing_summary.csv
```

`operation_timing.csv` should contain one row per timed operation.

Required base columns:

```text
stage
operation
item
wall_seconds
cpu_seconds
status
error
```

`operation_timing_summary.csv` should aggregate by operation and contain:

```text
operation
count
total_wall_seconds
mean_wall_seconds
max_wall_seconds
total_cpu_seconds
mean_cpu_seconds
max_cpu_seconds
n_error
```

---

## Acceptance tests for timing

1. `operation_timing.csv` is always written, even if a later operation fails.
2. Rescue rounds include timing rows with `stage=rescue_round_XX`.
3. At minimum, each rescue round records:
   - target selection time
   - rescue design time
   - rescue EQ runtime
   - reconstruct-chain runtime
   - state-table writing time
4. Timing rows include enough metadata to compare rescue EQ with normal growth EQ:
   - `side`
   - `center_x`
   - `k`
   - `n_eq_steps`
5. `operation_timing_summary.csv` aggregates wall and CPU time by operation.

---

## Important constraints

Do not change these scientific components unless absolutely necessary:

- MBAR estimator
- MTS estimator
- CFT/BAR logic
- bootstrap algorithms
- global inverse-variance PMF fusion

This task should only modify:

1. Rescue target selection and rescue window design integration.
2. Rescue diagnostics.
3. Operation-level timing logs.

---

## Final expected behavior

After the fix:

1. If there are uncovered bins in the analysis domain, rescue targets those bins before targeting high finite variance.
2. `rescue_summary.csv` clearly states why each rescue target was chosen.
3. The rescue window center and force constant come from `design_rescue_window(...)`.
4. The first rescue round no longer defaults to the first unresolved cluster midpoint when a larger uncovered interval exists elsewhere.
5. `operation_timing.csv` and `operation_timing_summary.csv` allow direct comparison between normal chain growth and rescue rounds.
