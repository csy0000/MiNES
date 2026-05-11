# Execution Log — 2026-05-06

## Token Usage Summary

| Time code | Task | Approx. input tokens | Approx. output tokens | Notes |
|---|---|---|---|---|
| CODEX_SHIFT_IMMEDIATE | Rescue center shift when first scaled k saturates | ~18 000 | ~3 500 | Read plan, read py, wrote targeted changes to `design_rescue_window` |
| notebook-stall-fix | Diagnose and fix `%matplotlib inline` + `plt.close` stalls | ~10 000 | ~1 200 | Read ipynb, used NotebookEdit for 5 cells |
| gt_linear_protocol | Add GT and linear NEQ protocol modes | ~32 000 | ~8 000 | Read plan + py; large rewrite of `run_neq_protocol`, new helpers, shell script update |
| codex_gt_rescue_refactor | Add GT helpers, choose_connected_boundary_pair, find_first_rescue_pair, NEQ trajectory coverage | ~45 000 | ~9 000 | 14 sections implemented; GT bridge refactored; AST OK |
| codex_mines_rescue_timing_task | Restore priority rescue loop + add operation-level timing | ~55 000 | ~11 000 | timed_operation + write_timing_outputs; rescue loop uses choose_rescue_target_priority; atexit guard; AST OK |

---

## [CODEX_SHIFT_IMMEDIATELY_WHEN_SCALED_K_EXCEEDS_KMAX]

**Instruction file:** `claude-plan/2026-05/2026-05-02-CODEX_SHIFT_IMMEDIATELY_WHEN_SCALED_K_EXCEEDS_KMAX.md`

### Problem

When `matched_child_k` is already at `k_max`, the first rescue attempt (`n_retry = 0`) used:
```
f = 0.5 * (2 - 0) = 1.0  →  x_rescue = x_m
```
This repeated the same window (`x_m, k_max`) instead of moving toward the target.

### Changes — `scripts/mines_variance_fusion.py`, `design_rescue_window`

**New variables computed after matching the child:**
```python
k_m = float(matched["k"])
first_scaled_k = float(args.s_rescue) * k_m
first_scaled_saturates = first_scaled_k >= float(args.k_max) - 1.0e-12
```

**Stiffness block (unchanged):**
```python
rescue_scale = float(args.s_rescue) ** float(n_retry + 1)
rescue_k_unclipped = rescue_scale * k_m
rescue_k = clip(rescue_k_unclipped, args.k_min, args.k_max)
rescue_k_saturated = rescue_k >= float(args.k_max) - 1.0e-12
```

**New center block:**
```python
if not rescue_k_saturated:
    center_retry_index = 0
    rescue_center_rule = "matched_child_center_before_kmax"
    f_raw = f = 1.0
    rescue_center_x_raw = x_m
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
    f = min(max(f_raw, float(args.rescue_center_f_min)), float(args.rescue_center_f_max))
    rescue_center_x_raw = x_target + f * (x_m - x_target)
```

**Result with defaults (`slope=0.5, start=2`):**

| Condition | n_retry | center_retry_index | f | x_rescue |
|---|---|---|---|---|
| `first_scaled_k < k_max` | 0 | 0 | 1.0 | x_m |
| `first_scaled_k >= k_max` | 0 | 1 | 0.5 | midpoint(x_m, x_target) |
| `first_scaled_k >= k_max` | 1 | 2 | 0.0 | x_target |
| `first_scaled_k >= k_max` | 2 | 3 | −0.5 | overshoot beyond x_target |

**New fields added to `rescue_summary.csv` and `rescue_decision.json`:**
- `matched_child_k`
- `first_scaled_k`
- `first_scaled_saturates`
- `center_retry_index`
- `rescue_center_f_raw`
- `rescue_center_f`
- `rescue_center_x_raw`
- `rescue_center_x`
- `rescue_center_rule`
- `rescue_k_unclipped`
- `rescue_k_saturated`

**Fallback (no matched child):**
```python
first_scaled_k = float("nan")
first_scaled_saturates = False
center_retry_index = 0
```

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

Rescue table column list in cell `50f5f325` extended to include:
```
matched_child_name, matched_child_center_x, matched_child_k,
first_scaled_k, first_scaled_saturates, center_retry_index,
rescue_center_f, rescue_center_x, rescue_k, rescue_k_saturated, rescue_center_rule
```

---

## [notebook-stall-fix]

### Problem

`mines_variance_fusion_visualization.ipynb` stalled mid-run without error. Two root causes:

1. **Missing `%matplotlib inline`** — without this magic, matplotlib may open a blocking GUI window in some notebook environments, halting execution indefinitely.
2. **Missing `plt.close(fig)`** — each plot cell allocated a new figure but never released it, causing unbounded memory growth and eventual kernel hang.

### Changes — `analysis/notebook/mines_variance_fusion_visualization.ipynb`

| Cell ID | Fix |
|---|---|
| `c4a86f93` | Added `%matplotlib inline` as first line of the imports cell |
| `2240aee6` | Added `plt.close(fig)` at end |
| `d0f54c4b` | Added `plt.close(fig)` at end |
| `140782e7` | Added `plt.close(fig)` at end |
| `717191c1` | Added `plt.close(fig)` at end |
| `50f5f325` | Added `plt.close(fig)` before the `display()` calls |

---

## [gt_linear_protocol_modes]

**Instruction file:** `claude-plan/2026-05/2026-05-06-gt_linear_protocol_modes.md`

### Design

Protocol files (`forward_path.csv`, `reverse_path.csv`) must be written **before** the binary runs, so the binary and MTS reconstruction use the same path.  The binary already supports `-fpath` (reads `lambda,x0,y0,k` CSV).  `run_neq_backward` in C++ reverses the path automatically via `idx = nsteps - 1 - step`, so only the forward file is needed for the `-fpath` flag.

### Changes — `scripts/mines_variance_fusion.py`

**New CLI argument** (after `--rescue-center-f-max`):
```python
parser.add_argument(
    "--neq-protocol-mode",
    choices=["GT", "linear", "gt"],
    default="GT",
)
```

**New `NEQSegment` fields** (7 added after `remaining_budget_before_segment`):
```python
protocol_mode: str = "GT"
protocol_metadata: dict[str, Any] = field(default_factory=dict)
protocol_k_min: float | None = None
protocol_k_max: float | None = None
protocol_x_min: float | None = None
protocol_x_max: float | None = None
protocol_clip_fraction_k: float = 0.0
protocol_clip_fraction_x: float = 0.0
```

**New helper functions** (inserted after `linear_path_centers`):

- `window_tail_mean_sigma(window)` — returns `(mean, sigma)` from the EQ tail samples; raises if `size < 2` or `sigma ≤ 0`
- `clip_with_flag(value, lower, upper)` — returns `(clipped_value, was_clipped)` 
- `build_linear_bridge_protocol(*, left_window, right_window, n_time, k_min, k_max)` — exact endpoint enforcement at idx=0 and idx=n−1; interior: `x_s` linear, `k_s = (sqrt interp)²`, clipped
- `build_gt_bridge_protocol(*, left_window, right_window, n_time, k_min, k_max)` — Gaussian-transport bridge:
  - `K_L = 1/σ_L²`, `K_R = 1/σ_R²`
  - `k0_local = 0.5·((K_L−k_L) + (K_R−k_R))`
  - `q_local = 0.5·((K_L·m_L − k_L·x_L) + (K_R·m_R − k_R·x_R))`
  - Interior step s: interpolate `m_s, σ_s, K_s`; then `k_s = clip(K_s − k0_local, k_min, k_max)`, `x_s = clip(((k0_local+k_s)·m_s − q_local)/k_s, x_low, x_high)`
- `build_bridge_protocol(*, mode, ...)` — dispatches to GT or linear; raises on unknown mode
- `run_neq_edge_with_protocol_path(*)` — wraps `run_checked`; passes `-fpath <protocol_path>` to the binary

**`run_neq_protocol` rewrite** — new `neq_protocol_mode: str = "GT"` parameter:
1. Build forward + reverse protocols via `build_bridge_protocol`
2. Write `forward_path.csv`, `reverse_path.csv`, `forward_path_diagnostics.csv`, `reverse_path_diagnostics.csv`, `protocol_summary.json` **before** simulation
3. Call `run_neq_edge_with_protocol_path` with `protocol_path=forward_path_file`
4. After sim, trim arrays if actual `n_time < n_time_requested`
5. `protocol_k` is now `float(np.nanmean(forward_ks))` (mean, not a constant)
6. `NEQSegment` populated with all 7 new protocol fields
7. `segment_summary.json` extended with protocol fields and 3 new file pointers

**`build_segment_rows`** — 11 new protocol columns added:
`protocol_mode, protocol_k_min, protocol_k_max, protocol_x_min, protocol_x_max, protocol_clip_fraction_k, protocol_clip_fraction_x` plus 4 metadata fields.

**Call sites updated** (both `reconstruct_chain` ~line 2698 and main growth ~line 3418):
```python
neq_protocol_mode=str(args.neq_protocol_mode)
```

**`mines_variance_fusion_summary.json`** — added:
```python
"neq_protocol_mode": str(args.neq_protocol_mode).upper()
```

### Changes — `scripts/run_mines_variance_fusion.sh`

Added default:
```bash
NEQ_PROTOCOL_MODE="GT"
```

Added parser case:
```bash
--neq-protocol-mode)
  NEQ_PROTOCOL_MODE="$2"
  shift 2
  ;;
```

Added to CMD array:
```bash
--neq-protocol-mode "${NEQ_PROTOCOL_MODE}"
```

---

## [codex_mines_rescue_timing_task]

**Instruction file:** `claude-plan/2026-05/2026-05-06-codex_mines_rescue_timing_task.md`

### Problem 1: Rescue targeting restored to priority-based logic

Reverted the rescue loop (added in `codex_gt_rescue_refactor`) from `find_first_rescue_pair` back to `choose_rescue_target_priority` + `design_rescue_window`. The GT helpers (`find_first_rescue_pair`, `get_xs_ks_from_ms`, etc.) remain in the file as utility functions.

**Rescue loop now:**
1. Calls `choose_rescue_target_priority(...)` with timing wrapper → priority order: uncovered interval → failed/skipped gap → max finite variance
2. If `target_info is None`, stops with `stop_reason = "no_rescue_target_available"`
3. Calls `design_rescue_window(target_info=target_info, ...)` to get `rescue_center_x` and `rescue_k`
4. Runs `run_eq_window(...)` at that center
5. Runs `reconstruct_chain(...)` with `timing_rows=timing_rows`
6. Writes `rescue_decision.json` as `{**target_info, **rescue_design, ...}`

**`rescue_summary.csv` columns** (from spec):
```
round, target_priority, target_reason, x_rescue_target, target_variance,
uncovered_start_x, uncovered_end_x, uncovered_width, uncovered_n_bins,
rescue_center_x_raw, rescue_center_x, rescue_center_clamped_to_bounds,
rescue_k, rescue_k_unclipped, rescue_k_saturated, rescue_k_rule, rescue_center_rule,
rescue_center_f_raw, rescue_center_f, matched_child_name, matched_child_side,
matched_target_x, matched_target_distance, matched_child_center_x, matched_child_k,
matched_child_raw_k, matched_child_k_rule, matched_child_target_source,
rescue_retry_count, rescue_scale, added_window, added_center_x, added_k, used_steps
```

### Problem 2: Operation-level timing logs

**New imports** (top of file):
```python
import atexit
import time
from contextlib import contextmanager
from typing import Any, Generator
```

**`timed_operation` context manager** — records wall + CPU time, status, error, and arbitrary metadata per invocation.

**`summarize_timing_rows`** — groups by operation, aggregates count/total/mean/max wall+cpu, n_error.

**`write_timing_outputs`** — writes `operation_timing.csv` and `operation_timing_summary.csv`.

**`timing_rows: list[dict[str, Any]] = []`** initialized in `main()`. A `_flush_timing` closure is registered with `atexit.register` so that `operation_timing.csv` is written even if an unhandled exception terminates `main()` before the end. An explicit `write_timing_outputs(out_root, timing_rows)` call also runs at the end of `main()`.

**Instrumented operations:**

| Operation | Stage | Where |
|---|---|---|
| `run_eq_window` (L0) | `g00` | Before `windows = [left0, right0]` |
| `run_eq_window` (R0) | `g00` | Before `windows = [left0, right0]` |
| `run_eq_window` (left child) | growth stage | Growth loop |
| `run_eq_window` (right child) | growth stage | Growth loop |
| `run_neq_protocol` | growth stage | Growth loop |
| `reconstruct_chain` | growth stage | Post-growth |
| `choose_rescue_target_priority` | rescue stage | Rescue loop |
| `design_rescue_window` | rescue stage | Rescue loop |
| `run_eq_window` (rescue) | rescue stage | Rescue loop |
| `reconstruct_chain` | rescue stage | Rescue loop |
| `write_state_tables` | rescue stage | Rescue loop |
| `build_eq_cluster_patch` | (stage from reconstruct_chain) | Inside `reconstruct_chain` |
| `build_neq_mts_patch` | (stage from reconstruct_chain) | Inside `reconstruct_chain` |
| `fit_global_pmf_from_patches` | (stage from reconstruct_chain) | Inside `reconstruct_chain` |
| `build_hs_fallback_patches` | `post_growth` | Post-growth block |

**`reconstruct_chain` signature** extended with `timing_rows: list[dict[str, Any]] | None = None`; internally uses `_timing = timing_rows if timing_rows is not None else []`.

**`mines_variance_fusion_summary.json`** metadata updated:
- `rescue_strategy`: `"choose_rescue_target_priority"` (was `"first_nonoverlapping_pair_gt_rescue"`)
- `rescue_target_rule`: `"uncovered_interval_first_then_failed_gap_then_max_finite_variance"` (was GT rule)
- Removed `rescue_gt_out_of_bounds_rule` key
- Added `"operation_timing"` and `"operation_timing_summary"` to `summary_files`

### Verification

```
python -c "import ast; ast.parse(open('scripts/mines_variance_fusion.py').read())"  # AST OK
python scripts/mines_variance_fusion.py --help  # exits 0
```
